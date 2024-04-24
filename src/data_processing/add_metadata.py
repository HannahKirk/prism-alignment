"""
Loads text instances and adds metadata on language, PII, and moderation.
"""

import json
import logging
import pandas as pd
import scrubadub
import langid
import openai
from tqdm import tqdm
from retrying import retry
from src.utils.helper_funcs import find_project_root, ensure_dir_exists, save_as_jsonl

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


def clean_text(text):
    """
    Uses scrubadub https://scrubadub.readthedocs.io/en/stable/ to find PII.
    Args:
        text: input string

    Returns: 1 if PII found, 0 if not

    """
    flag = 0
    if pd.notna(text):
        cleaned_text = scrubadub.clean(text)
        if cleaned_text != text:
            flag = 1
    return flag


def identify_language(text):
    """
    Use langid https://github.com/saffsd/langid.py to identify language of text.
    The function can return the norm probability as well.
    Args:
        text: input text

    Returns: the identified language (e.g., 'en') and the norm probablity
    """
    if pd.notna(text):
        lang, _ = langid.classify(text)
    else:
        lang = None
    return lang


# Moderation
@retry(
    wait_exponential_multiplier=1000, wait_exponential_max=10000
)  # 2^x * 1000 milliseconds between each retry, up to 10 seconds, then 10 seconds afterwards
def moderate_text(text, OPENAI_API_KEY):
    """Uses OpenAI moderation API to moderate text attributes."""
    openai.api_key = OPENAI_API_KEY
    # Call the moderation API checkpoint
    response = openai.Moderation.create(
        input=text,
    )
    # Return full results from moderation endpoint
    return response["results"][0]


def load_api_key(path):
    with open(f"{path}/config.json", "r") as config_file:
        config = json.load(config_file)
        return config.get("openai_api_key", "")


def process_dataset(df, OPENAI_API_KEY):
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        if pd.isna(row["language_flag"]):
            df.at[index, "language_flag"] = identify_language(row["text"])

        if pd.isna(row["pii_flag"]):
            df.at[index, "pii_flag"] = clean_text(row["text"])

        if pd.isna(row["moderation_flag"]):
            df.at[index, "moderation_flag"] = moderate_text(row["text"], OPENAI_API_KEY)

    return df


def round_moderation_scores(df):
    for index, row in tqdm(
        df.iterrows(), total=df.shape[0], desc="Rounding moderation scores"
    ):
        if pd.notna(row["moderation_flag"]):
            moderation_data = row["moderation_flag"]
            if "category_scores" in moderation_data:
                for category, score in moderation_data["category_scores"].items():
                    moderation_data["category_scores"][category] = round(score, 2)
                df.at[index, "moderation_flag"] = moderation_data

    return df


def main():
    """Get metadata"""
    # Set data path
    PROJECT_ROOT = find_project_root()
    INPUT_PATH = PROJECT_ROOT / "data" / "metadata"
    OUTPUT_PATH = INPUT_PATH

    # Load the OpenAI API key from the configuration file
    OPENAI_API_KEY = load_api_key(PROJECT_ROOT)

    df = pd.read_json(f"{INPUT_PATH}/texts_for_metadata_processing.jsonl", lines=True)
    processed_df = process_dataset(df, OPENAI_API_KEY)
    processed_df = round_moderation_scores(processed_df)
    processed_df["pii_flag"] = processed_df["pii_flag"].astype(bool)

    # Drop the 'text' column
    processed_df = processed_df.drop(columns=["text"])

    # Save processed DataFrame
    ensure_dir_exists(OUTPUT_PATH)
    path = f"{OUTPUT_PATH}/metadata.jsonl"
    save_as_jsonl(processed_df, path, is_already_records=False)
    logging.info("Dataset processed and saved successfully.")


if __name__ == "__main__":
    main()
