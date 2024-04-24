"""
Script to make the local neighbourhood clustering of user prompts and/or model responses.
Uses temporal cluster from https://github.com/meedan/temporal_clustering/
"""

import argparse
from sentence_transformers import SentenceTransformer
from temporal_clustering import temporal_cluster
from src.utils.helper_funcs import find_project_root
from src.utils.data_loader import load_data


def main(text_col):
    PROJECT_ROOT = find_project_root()
    # Load conversations
    data_dict = load_data(survey=False, models=False, metadata=False)
    if text_col == "opening_prompt":
        df = data_dict["conversations"]
    elif text_col == "model_response":
        df = data_dict["utterances"]
    df = df.sort_values(text_col)

    # Load data
    texts = df[text_col].to_list()

    # Set up model
    embedding_model = "all-mpnet-base-v2"
    cache_dir = "../.cache"
    model = SentenceTransformer(embedding_model, cache_folder=cache_dir)
    embeddings = model.encode(texts, show_progress_bar=True)

    df = df[["conversation_id", "user_id"]].copy()

    # Loop through thresholds
    for threshold in [0.05, 0.125, 0.2]:
        cluster_membership = temporal_cluster(embeddings, threshold)
        df["cluster_membership"] = cluster_membership
        save_name = str(threshold).replace(".", "-")

        # Construct file path
        file_path = (
            PROJECT_ROOT / f"results/clusters/local_clusters_{text_col}_{save_name}.csv"
        )

        # Check if file exists to avoid overwriting
        if not file_path.exists():
            df.to_csv(file_path, index=False)
        else:
            print(f"File {file_path} already exists. Skipping to avoid overwriting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process text columns for local clustering."
    )
    parser.add_argument("text_col", type=str, help="Name of the text column to process")

    args = parser.parse_args()
    main(args.text_col)
