"""
Script to make the topic clustering of free text fields.
"""

import logging
import configparser
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer  # for compute_embeddings
import umap  # for reduce_dim_umap
from sklearn.feature_extraction.text import CountVectorizer  # for c_tf_idf
import hdbscan  # for compute_clusters
import openai  # for get_cluster_name_from_gpt
import time  # for get_cluster_name_from_gpt

# For cleaning text
from src.utils.helper_funcs import (
    find_project_root,
    ensure_dir_exists,
    get_new_filename,
)
from src.data_analysis.free_text import remove_stopwords_and_questionwords

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


tqdm.pandas()

# KEYS
PROJECT_ROOT = find_project_root()
config = configparser.ConfigParser()
CONFIG_PATH = PROJECT_ROOT / "config" / "config.ini"
config.read(CONFIG_PATH)
API_TOKEN = config.get("openai", "api_key")

openai.api_key = API_TOKEN


############################################################################################################
# COMPUTING EMBEDDINGS WITH SENTENCE TRANSFORMERS
############################################################################################################


def compute_embeddings(
    input_texts: pd.Series, embedding_model: str, batch_size: int, cache_dir: str
):
    # Load texts and write to df
    df = input_texts.to_frame()
    logging.info(f"Loaded {df.shape[0]} text rows")

    # Load model for computing embeddings
    model = SentenceTransformer(embedding_model, cache_folder=cache_dir)
    logging.info(f"Loaded embedding model: {embedding_model}")

    # Compute embeddings for each text
    embeddings = model.encode(
        list(df[df.columns[0]]),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    logging.info(f"Computed embeddings with shape {embeddings.shape}")

    return embeddings


############################################################################################################
# CREATING CLUSTERS WITH UMAP AND HDBSCAN
############################################################################################################


def c_tf_idf(texts, m, ngram_range=(1, 2)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(texts)
    t = count.transform(texts).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_words_per_cluster(tf_idf, count, texts_by_cluster, top_n):
    words = count.get_feature_names_out()
    labels = list(texts_by_cluster["cluster_id"])
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -top_n:]
    top_words = {
        label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1]
        for i, label in enumerate(labels)
    }

    return top_words


def reduce_dim_umap(embeddings, umap_dim, umap_min_dist, umap_n_neighbors):
    umap_embeddings = umap.UMAP(
        random_state=42,
        n_neighbors=umap_n_neighbors,
        n_components=umap_dim,
        min_dist=umap_min_dist,
        metric="cosine",
    ).fit_transform(embeddings)

    return umap_embeddings


def compute_clusters(
    input_texts: pd.Series,
    orig_texts: pd.Series,
    input_ids: pd.Series,
    input_embeddings: torch.float32,
    umap_dim: int,
    umap_min_dist: float,
    umap_n_neighbors: int,
    hdb_min_cluster_size: int,
    top_n_texts: int,
    top_n_words: int,
):
    logging.info("Loading data...")

    # Load texts and embeddings
    input_texts = list(input_texts)
    input_ids = list(input_ids)
    orig_texts = list(orig_texts)
    logging.info(f"  Loaded {len(input_texts)} texts.")
    logging.info(f"  Loaded embeddings with shape {input_embeddings.shape}.\n")

    logging.info("Computing clusters...")

    # Reduce dimensionality of embeddings to umap_dim with UMAP
    umap_embeddings = reduce_dim_umap(
        input_embeddings, umap_dim, umap_min_dist, umap_n_neighbors
    )
    logging.info(f"  Reduced embedding dimensionality to {umap_dim} with UMAP.")

    # Create clusters with HDBScan
    cluster = hdbscan.HDBSCAN(
        min_cluster_size=hdb_min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit(umap_embeddings)
    logging.info(
        f"  Created {len(set(cluster.labels_))-1} clusters with a minimum size of {hdb_min_cluster_size} texts with HDBScan."
    )

    # Create dataframe where texts are labelled with cluster_ids and umap embeddings
    umap_list = []
    for row in umap_embeddings:
        umap_list.append(row.tolist())

    text_df = pd.DataFrame(
        {
            "id": input_ids,
            "text": input_texts,
            "orig_text": orig_texts,
            "cluster_id": cluster.labels_,
            "umap_embedding": umap_list,
        }
    )

    # Compute centroids of each cluster as average of umap embeddings
    cluster_centroids = []
    for i in range(len(set(cluster.labels_))):
        cluster_centroids.append(
            np.mean(umap_embeddings[cluster.labels_ == i], axis=0).tolist()
        )

    # compute distance of each text to its cluster centroid
    cluster_distances = []
    for i in range(len(umap_embeddings)):
        cluster_distances.append(
            np.linalg.norm(umap_embeddings[i] - cluster_centroids[cluster.labels_[i]])
        )

    # add distance to centroid to text dataframe
    text_df["distance_to_centroid"] = cluster_distances

    # Create 2d embeddings for visualisation
    umap_embeddings_2d = reduce_dim_umap(
        input_embeddings, 2, umap_min_dist, umap_n_neighbors
    )
    result_2d = pd.DataFrame(umap_embeddings_2d, columns=["x2", "y2"])
    text_df = pd.concat([text_df, result_2d], axis=1)

    # Create df where every cluster has all texts within that cluster concatenated into a single string
    texts_by_cluster = text_df.groupby(["cluster_id"], as_index=False).agg(
        {"text": " ".join}
    )

    # Run tf-idf, then use that to identify top 20 uni/bigrams for each cluster
    tf_idf, count = c_tf_idf(texts_by_cluster.text.values, m=len(text_df))
    top_words = extract_top_words_per_cluster(
        tf_idf, count, texts_by_cluster, top_n=top_n_words
    )

    # Create cluster_df of cluster IDs with size of cluster
    cluster_df = pd.DataFrame(text_df.cluster_id.value_counts())
    cluster_df.reset_index(inplace=True)
    cluster_df.columns = ["cluster_id", "cluster_size"]

    # Write top texts closest to centroid to column in cluster_df
    centroid_texts = (
        text_df.groupby("cluster_id")
        .apply(lambda x: x.nsmallest(top_n_texts, "distance_to_centroid"))
        .reset_index(drop=True)
    )
    centroid_texts = (
        centroid_texts.groupby("cluster_id")
        .agg({"orig_text": lambda x: list(x)})
        .reset_index()
    )
    centroid_texts.rename(columns={"orig_text": "top_texts"}, inplace=True)
    cluster_df = cluster_df.merge(centroid_texts, on="cluster_id", how="left")

    # Write top words without scores to columns in cluster_df
    cluster_df["top_words"] = cluster_df.cluster_id.apply(
        lambda x: [pair[0] for pair in top_words[x]]
    )

    logging.info("\nResults:")
    logging.info(f"  {text_df.shape[0]} texts were clustered.")
    logging.info(
        f"  {text_df[text_df.cluster_id != -1].shape[0]} texts ({round(text_df[text_df.cluster_id != -1].shape[0]/text_df.shape[0]*100, 2)}%) were assigned to one of {len(set(cluster.labels_))-1} clusters."
    )
    logging.info(
        f"  {text_df[text_df.cluster_id == -1].shape[0]} texts ({round(text_df[text_df.cluster_id == -1].shape[0]/text_df.shape[0]*100, 2)}%) were not assigned to any cluster.\n"
    )

    return text_df, cluster_df


############################################################################################################
# NAMING CLUSTERS WITH GPT
############################################################################################################


def get_cluster_name_from_gpt(gpt_model, top_texts, top_words):
    prompt = f"Your task is to create a short clear title for a cluster of texts (around 2-4 words) based on the following information.\n\n\
    Typical texts in the cluster are: {top_texts}\n\n\
    Typical words used in the cluster are: {top_words}\n\n\
    Give the cluster title:"

    prompt_input = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=prompt_input,
                temperature=0,
                max_tokens=48,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                request_timeout=10,  # seconds to wait for response
            )
            break
        except openai.error.OpenAIError as e:
            logging.info(f"OpenAIError: {e}. Retrying after 1s backoff.")
            time.sleep(2)

    return response.choices[0].message["content"]


def name_clusters(cluster_df: pd.DataFrame, gpt_model: str):
    logging.info(f"Loaded cluster_df with {cluster_df.shape[0]} clusters")

    logging.info("Naming clusters with GPT...")
    # write gpt completion to new column
    cluster_df["gpt_description"] = cluster_df.progress_apply(
        lambda row: get_cluster_name_from_gpt(gpt_model, row.top_texts, row.top_words),
        axis=1,
    )

    # reorder columns and drop unnecessary ones
    cluster_df = cluster_df[
        ["cluster_id", "cluster_size", "gpt_description", "top_words", "top_texts"]
    ]

    return cluster_df


############################################################################################################
# FULL CLUSTER PIPELINE
############################################################################################################


# Select and load free text field
def load_text(df_dict, text_field, question_mapping):
    if text_field == "self_description":
        id_field = "user_id"
        text_ids = df_dict["survey"][[id_field, text_field]]
        umap_n_neighbors = 15
        hdb_min_cluster_size = 20
        question = question_mapping.loc[text_field, "full_question"]

    elif text_field == "system_string":
        id_field = "user_id"
        text_ids = df_dict["survey"][[id_field, text_field]]
        umap_n_neighbors = 15
        hdb_min_cluster_size = 20
        question = question_mapping.loc[text_field, "full_question"]

    elif text_field == "open_feedback":
        id_field = "conversation_id"
        text_ids = df_dict["conversations"][[id_field, text_field]]
        umap_n_neighbors = 15
        hdb_min_cluster_size = 80
        question = "Give the model some feedback on how it can improve or act differently. Hypothetically, what would you have wanted to see in an ideal answer here? What (if anything) was missing? What was good and what was bad? What would you change to make it better? Please write 2-5 sentences in your own words."

    elif text_field == "opening_prompt":
        id_field = "conversation_id"
        text_ids = df_dict["conversations"][[id_field, text_field]]
        umap_n_neighbors = 15
        hdb_min_cluster_size = 80
        question = ""

    elif text_field == "all_texts":
        utterances = df_dict["utterances"]
        survey = df_dict["survey"]
        # For each user get all their prompts
        user_texts = (
            utterances.groupby("user_id")["user_prompt"]
            .unique()
            .apply(list)
            .reset_index()
        )
        # Get their written feedback too
        conversations = df_dict["conversations"]
        user_feedback = (
            conversations.groupby("user_id")["open_feedback"]
            .unique()
            .apply(list)
            .reset_index()
        )
        # Combine prompts and feedback
        user_texts = user_texts.merge(
            user_feedback,
            left_on="user_id",
            right_on="user_id",
            how="left",
        )

        # Now merge their survey texts
        user_texts = user_texts.merge(
            survey[["user_id", "self_description", "system_string"]],
            left_on="user_id",
            right_on="user_id",
            how="left",
        )
        # Add all text columns to one list
        user_texts["all_texts"] = user_texts.apply(
            lambda x: x["user_prompt"]
            + x["open_feedback"]
            + [x["self_description"]]
            + [x["system_string"]],
            axis=1,
        )

        # Combine into one string
        user_texts["all_texts"] = user_texts["all_texts"].apply(lambda x: " ".join(x))
        id_field = "user_id"
        text_field = "all_texts"
        text_ids = user_texts[[id_field, text_field]]
        umap_n_neighbors = 10
        hdb_min_cluster_size = 5
        question = ""

    else:
        raise ValueError(
            "text_field must be one of: 'self_description', 'system_string', 'open_feedback', 'opeining_prompt', 'concat_text"
        )

    logging.info(f"Selected {len(text_ids)} {text_field} texts for analysis")

    return (
        text_field,
        id_field,
        text_ids,
        question,
        umap_n_neighbors,
        hdb_min_cluster_size,
    )


def describe_texts(texts):
    logging.info(f"mean length: {texts.str.len().mean()}")
    logging.info(f"median length: {texts.str.len().median()}")
    logging.info(f"max length: {texts.str.len().max()}")
    logging.info(f"min length: {texts.str.len().min()}")


def full_cluster(df_dict, text_field, question_mapping, clean_text=False):
    (
        text_field,
        id_field,
        text_ids,
        question,
        umap_n_neighbors,
        hdb_min_cluster_size,
    ) = load_text(df_dict, text_field, question_mapping)

    # drop any NA values from texts
    logging.info(f"Dropped {text_ids[text_field].isna().sum()} NA values")
    text_ids = text_ids.dropna()
    logging.info(f"{len(text_ids)} texts remain")

    # Initialise lists
    texts = text_ids[text_field]
    ids = text_ids[id_field]

    describe_texts(texts)

    # Mostly we don't to clean texts as it affects the sentence embeddings
    if clean_text:
        orig_texts = texts.copy()
        texts = texts.apply(
            lambda text: remove_stopwords_and_questionwords(text, question)
        )
    else:
        orig_texts = texts.copy()

    # Calculate embeddings for texts
    embeddings = compute_embeddings(
        input_texts=texts,
        embedding_model="all-mpnet-base-v2",  # best-performing model
        cache_dir="../.cache",  # for storing the model
        batch_size=16,
    )

    # Cluster texts
    text_df, cluster_df = compute_clusters(
        input_texts=texts,
        orig_texts=orig_texts,
        input_ids=ids,
        input_embeddings=embeddings,
        umap_dim=20,
        umap_min_dist=0.0,
        umap_n_neighbors=umap_n_neighbors,
        hdb_min_cluster_size=hdb_min_cluster_size,
        top_n_texts=5,
        top_n_words=10,
    )

    # Name clusters
    if text_field != "all_texts":
        cluster_df = name_clusters(cluster_df=cluster_df, gpt_model="gpt-4")
    else:
        # Reorder columns and drop unnecessary ones
        cluster_df["gpt_description"] = None
        cluster_df = cluster_df[
            ["cluster_id", "cluster_size", "gpt_description", "top_words", "top_texts"]
        ]

    # If cluster_id is -1, set gpt_description as outliers
    cluster_df.loc[cluster_df["cluster_id"] == -1, "gpt_description"] = "Outliers"

    # Save
    OUTPUT_PATH = PROJECT_ROOT / "results" / "clusters"
    ensure_dir_exists(OUTPUT_PATH)

    # Check and update filename for text_df if necessary
    text_df_filename = get_new_filename(OUTPUT_PATH / f"{text_field}_text_df.csv")
    text_df.to_csv(text_df_filename, index=False)

    # Check and update filename for cluster_df if necessary
    cluster_df_filename = get_new_filename(OUTPUT_PATH / f"{text_field}_cluster_df.csv")
    cluster_df.to_csv(cluster_df_filename, index=False)

    logging.info("Finished.")


def main(free_text_col):
    "Run cluster pipeline"
    # Load data
    df_dict = dict()

    logging.info("Loading data...")
    INPUT_PATH = PROJECT_ROOT / "data"
    STORE_PATH = PROJECT_ROOT / "data" / "storage" / "mappings"
    for filename in [
        "survey.jsonl",
        "conversations.jsonl",
        "utterances.jsonl",
    ]:
        loaded_csv = pd.read_json(f"{INPUT_PATH}/{filename}", lines=True)
        df_dict[filename[:-6]] = loaded_csv
        logging.info(f"Loaded {filename}: {df_dict[filename[:-6]].shape}")

    question_mapping = pd.read_json(
        STORE_PATH / "survey_question_mapping.json", orient="index"
    )

    logging.info(f"Running clustering for {free_text_col}...")
    full_cluster(df_dict, free_text_col, question_mapping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process free text columns for clustering."
    )
    parser.add_argument(
        "free_text_col", type=str, help="Name of the free text column to process"
    )
    args = parser.parse_args()

    main(args.free_text_col)
