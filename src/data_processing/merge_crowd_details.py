"""
Loads crowdworker checks, subsets to approved workers and merges with survey responses.
"""

import logging
import pandas as pd
from src.utils.helper_funcs import find_project_root


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


def main():
    PROJECT_ROOT = find_project_root()
    REVIEW_PATH = PROJECT_ROOT / "data" / "review"
    DATA_PATH = PROJECT_ROOT / "data" / "interim"

    # Load the data
    survey = pd.read_csv(DATA_PATH / "survey.csv")
    workers = pd.read_csv(REVIEW_PATH / "workers_with_checks.csv")
    studies = pd.read_csv(REVIEW_PATH / "studies_with_checks.csv")
    conversations = pd.read_json(DATA_PATH / "conversations.jsonl", lines=True)

    # Calculate sample sizes
    logging.info("Number of unique IDs in survey: %s", survey["user_id"].nunique())
    logging.info(
        "Number of unique countries in survey: %s", survey["birth_country"].nunique()
    )
    logging.info(
        "Number of unique IDs in conversations: %s",
        conversations["user_id"].nunique(),
    )
    logging.info(
        "Number of unique IDs in workers: %s", workers["participant_id"].nunique()
    )

    # Only keep the approved workers
    reduced_workers = workers[workers["include_flag"] == 1]
    logging.info("Number of total workers: %s", len(workers))
    logging.info("Number of accepted workers: %s", len(reduced_workers))
    reduced_survey = survey[survey["user_id"].isin(reduced_workers["participant_id"])]
    logging.info(
        "Number of unique IDs in survey: %s",
        reduced_survey["user_id"].nunique(),
    )
    logging.info(
        "Number of unique countries in survey: %s",
        reduced_survey["birth_country"].nunique(),
    )

    # Reduce cols pre-merge
    keep_cols = [
        "participant_id",  # This is the Prolific ID (not publicly released)
        "conversations_quota",
        "study_id",
        "ethnicity_simplified",
    ]

    reduced_workers = reduced_workers[keep_cols]

    reduced_workers = reduced_workers.merge(
        studies[["study_id", "study_locale"]], on="study_id", how="left"
    )

    # Merge with survey
    merged_survey = reduced_survey.merge(
        reduced_workers,
        left_on="user_id",
        right_on="participant_id",
        how="left",
        indicator=True,
    )
    merge_counts = merged_survey["_merge"].value_counts().to_string()
    logging.info("Merge counts: %s", merge_counts)
    merged_survey = merged_survey.drop(columns=["_merge"])

    # Create new columns for tracking conversations
    merged_survey["survey_only"] = merged_survey["conversations_quota"] == 0
    survey_only_counts = merged_survey["survey_only"].value_counts().to_string()
    logging.info("Crowdworker survey only counts: %s", survey_only_counts)

    # Save interim data
    merged_survey.to_csv(DATA_PATH / "merged_survey.csv", index=False)


if __name__ == "__main__":
    main()
