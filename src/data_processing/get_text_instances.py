"""
Extract all text instances to prepare them for metadata processing.
"""

import logging
import pandas as pd
import numpy as np
from src.utils.helper_funcs import find_project_root, ensure_dir_exists, save_as_jsonl
from src.utils.data_loader import load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


def main():
    """Run main"""
    # Set a path to store any results and make sure it exists
    PROJECT_ROOT = find_project_root()
    OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata"

    # Load our data
    data_dict = load_data()

    # Initialise frames
    frames = []

    # Set up instance dictionary
    instance_dict = {
        "survey": {
            "text_cols": ["system_string", "self_description"],
            "ids": ["user_id"],
            "stage": "survey",
        },
        "conversations": {
            "text_cols": ["open_feedback"],
            "ids": ["user_id", "conversation_id"],
            "stage": "conversations",
        },
        "utterances": {
            "text_cols": ["user_prompt", "model_response"],
            "ids": [
                "user_id",
                "conversation_id",
                "utterance_id",
                "interaction_id",
            ],
            "stage": "conversations",
        },
    }

    # Loop through datasets and columns to extract text instances
    for dataset, details in instance_dict.items():
        for text_col in details["text_cols"]:
            subset = data_dict[dataset][details["ids"] + [text_col]].copy()
            if text_col == "user_prompt":
                subset = subset.drop_duplicates(subset=["interaction_id"])
                # Set utterance ID as nan for user prompt as it applies to multiple rows
                subset["utterance_id"] = np.nan
            subset["column_id"] = text_col
            subset["stage"] = details["stage"]
            subset = subset.rename(columns={text_col: "text"})
            frames.append(subset)

    # Concat
    metadata = pd.concat(frames, ignore_index=True, axis=0)
    order = [
        "column_id",
        "user_id",
        "conversation_id",
        "interaction_id",
        "utterance_id",
        "stage",
        "text",
    ]
    metadata = metadata[order]

    # Save to jsonl
    ensure_dir_exists(OUTPUT_PATH)
    logging.info(
        "Texts ready for metaprocessing. Total %d instances", metadata.shape[0]
    )
    save_as_jsonl(
        metadata,
        OUTPUT_PATH / "texts_for_metadata_processing.jsonl",
        is_already_records=False,
    )


if __name__ == "__main__":
    main()
