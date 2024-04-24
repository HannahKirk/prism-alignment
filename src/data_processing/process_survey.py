"""
Loads raw survey data, then cleans and simplifies the data.
"""

import argparse
import re
import json
import logging
import numpy as np
import pandas as pd
from src.utils.helper_funcs import (
    find_project_root,
    strip_html_tags,
    ensure_dir_exists,
    to_snake_case,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


def convert_headers_to_dict(header_df):
    """
    Convert headers to a dictionary.

    Args:
        header_df (pd.DataFrame): The DataFrame containing header data.

    Returns:
        dict: The dictionary mapping survey_numeric to header and text.
    """
    mapping_dict = {}
    # Iterate through each row in the CSV data
    for _, row in header_df.iterrows():
        if pd.isna(row["survey_numeric"]) is False:
            # Add to the dictionary
            mapping_dict[row["survey_numeric"]] = {
                "header": row["header"],
                "text": row["survey_text"],
            }
    return mapping_dict


# Define the function to process the columns
def process_columns(df, col_stub, binary_conversion=False):
    """
    Process multiple columns of a DataFrame into a dictionary column.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        col_stub (str): The column stub to filter columns.
        binary_conversion (bool, optional): Flag to indicate binary conversion. Defaults to False.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    # Filter columns based on the stub
    columns = [col for col in df.columns if col.startswith(col_stub)]

    # Process each column
    for col in columns:
        # For text cols, replace -1 with nan
        if "other_text" in col:
            df[col] = df[col].apply(
                lambda x: np.nan if pd.isna(x) else (np.nan if x == "-1" else x)
            )
        else:
            if binary_conversion:

                # Replace -1 with 0, keep NaN as NaN, else replace with 1
                # -1 means "seen but unanswered"
                # Note NaN means they did not see the question due to survey flows
                df[col] = df[col].apply(
                    lambda x: (
                        False if x == "-1" else (True if not pd.isna(x) else np.nan)
                    )
                )
            else:
                continue

    # Create a dictionary column with conditional conversion
    df[col_stub + "_all"] = df.apply(
        lambda row: {
            col.split(":")[1]: (
                (int(row[col]) if "other_text" not in col else row[col])
                if not pd.isna(row[col])
                else None
            )
            for col in columns
        },
        axis=1,
    )

    return df


def extract_usecase_choice(s):
    """
    Extracts the use case choice from a string.

    Args:
        s (str): The string to extract the use case choice from.

    Returns:
        str or None: The use case choice if found, otherwise None.
    """
    match = re.search(r"\*\*(.+?)\*\*", s)
    if match:
        return match.group(1)
    return np.nan


def extract_statedpref_choice(s):
    """
    Extracts the stated preference choice from a string.

    Args:
        s (str): The string to extract the stated preference choice from.

    Returns:
        str or None: The stated preference choice if found, otherwise None.
    """
    match = re.search(r"- \.\.\.(.+)", s)
    if match:
        return match.group(1).strip()
    return np.nan


def replace_string_for_order_cols(s):
    """
    Replaces a string with a new string for order columns.

    Args:
        s (str): The string to replace.

    Returns:
        str: The new string with the replacements made.
    """
    return re.sub(r"(\w+)_do_(\d+)", r"order_\1_\2", s)


def clean_survey_columns(df, header_mappings):
    """
    Cleans the columns of a survey DataFrame.

    Args:
        df (pandas.DataFrame): The survey DataFrame to clean.

    Returns:
        pandas.DataFrame: The cleaned survey DataFrame.
    """
    df = df.copy()
    logging.info("Total number of examples created to date: %s", len(df))
    # Remove any unneeded cols that are NA
    drop_cols = [
        "RecipientLastName",
        "RecipientFirstName",
        "RecipientEmail",
        "ExternalReference",
        "DistributionChannel",
        "preamble",
    ]
    df.drop(drop_cols, inplace=True, axis=1)

    # Specific remaps
    remaps = {
        "Duration (in seconds)": "timing_duration_s",
        "PROLIFIC_PID": "user_id",
    }

    qualtrics_meta_cols = [
        "StartDate",
        "EndDate",
        "Status",
        "IPAddress",
        "Progress",
        "Finished",
        "RecordedDate",
        "ResponseId",
        "RecipientLastName",
        "RecipientFirstName",
        "RecipientEmail",
        "ExternalReference",
        "LocationLatitude",
        "LocationLongitude",
        "DistributionChannel",
        "UserLanguage",
    ]

    # Rename columns
    for c in qualtrics_meta_cols:
        remaps[c] = f"meta_{to_snake_case(c)}"
    df.rename(columns=remaps, inplace=True)

    # Replace white space and lower_case
    df.columns = [c.replace(" ", "_").lower() for c in df.columns]

    # Reformat order cols
    df.columns = [
        replace_string_for_order_cols(col) if "_do_" in col else col
        for col in df.columns
    ]

    # Change stated prefs columns
    # Iterate over the dictionary items
    mappers = ["prefs", "usecases"]
    col_stubs = ["stated_prefs", "lm_usecases"]
    for val_mapper, col_stub in zip(mappers, col_stubs):
        for key, value in header_mappings[val_mapper].items():
            for order in ["", "order_"]:
                # Construct the old and new column names
                old_column_name = f"{order}{col_stub}_{key}"
                new_column_name = f"{order}{col_stub}:{value['header']}"

                # Check if the old column name exists in the DataFrame
                if old_column_name in df.columns:
                    # Rename the column
                    df.rename(columns={old_column_name: new_column_name}, inplace=True)

    # Use infer_objects to infer better data types
    df = df.infer_objects()

    return df


def create_qualtrics_mapping(df):
    """
    Create a mapping of Qualtrics survey questions to their corresponding column names
        in the cleaned data.

    Args:
        df (pandas.DataFrame): The cleaned survey data.

    Returns:
        dict: A dictionary mapping Qualtrics survey questions to their corresponding column names
            in the cleaned data.
    """
    # Extract the three rows
    headers = df.columns
    questions = df.iloc[0]
    import_ids = df.iloc[1]

    # Construct the dictionary
    storage_dict = {}
    for header, question, import_id in zip(headers, questions, import_ids):
        short_question = strip_html_tags(question).lower()
        if "other [please type]" in short_question:
            if "_TEXT" in import_id:
                short_question = "other: written text"
            else:
                short_question = "other"

        elif "usecases" in header:
            short_question = extract_usecase_choice(short_question)
        elif "stated_prefs" in header:
            short_question = extract_statedpref_choice(short_question)
        else:
            short_question = "NA"

        storage_dict[header] = {
            "full_question": question,
            "short_question": short_question,
        }

    # Remove extra header rows
    df = df.iloc[2:]

    return df, storage_dict


def clean_and_simplify_survey_data(df, cutoff_date):
    """
    Cleans and simplifies survey data by replacing white space and lower casing column names,
    reformatting order columns, converting columns to datetime, filtering based on a cutoff date,
    removing preview rows, and checking that user IDs are unique.

    Args:
        df (pandas.DataFrame): The survey data to be cleaned and simplified.
        cutoff_date (str): The date to filter rows after in format YYYY-MM-DD.

    Returns:
        pandas.DataFrame: The cleaned and simplified survey data.
    """
    # Replace white space and lower_case
    df.columns = [c.replace(" ", "_").lower() for c in df.columns]
    # Reformat order cols
    df.columns = [
        replace_string_for_order_cols(col) if "_do_" in col else col
        for col in df.columns
    ]

    # Convert cols to datetime and filter
    datetime_cols = ["meta_start_date", "meta_end_date", "meta_recorded_date"]

    for col in datetime_cols:
        df.loc[:, col] = pd.to_datetime(df[col])

    # Filter rows where StartDate is after the cut-off date
    df = df[df["meta_recorded_date"] >= pd.to_datetime(cutoff_date, format="%Y-%m-%d")]
    logging.info(
        "Filtering to on or after %s, number of examples remaining: %s",
        cutoff_date,
        len(df),
    )

    # Remove any rows without a user id
    df = df[df["user_id"].notna()]
    logging.info(
        "Removing rows w/out participant ID, number of examples remaining: %s",
        len(df),
    )

    # Remove any rows with testing keyword ids (should be removed with date cutoff)
    # Note these keywords are from old project names or testing purposes
    testing_keywords = ["test", "perdi", "griffin", "arg", "gamma", "beta"]
    # Creating a regex pattern from the keywords
    pattern = "(?i)" + "|".join(testing_keywords)
    df = df[~df["user_id"].str.contains(pattern)]
    logging.info(
        "Removing rows with testing ID, number of examples remaining: %s",
        len(df),
    )

    # Remove any preview rows (shouldn't be any left)
    df = df[df["meta_status"] != "Survey Preview"]
    logging.info(
        "Filtering rows to exclude previews, number of examples remaining: %s", len(df)
    )

    # Process 'lm_usecases' with binary conversion
    df = process_columns(df, "lm_usecases", binary_conversion=True)

    # Convert 'llm_frequency' missing values
    df["lm_frequency_use"] = df["lm_frequency_use"].map(
        lambda x: np.nan if pd.isna(x) else x
    )

    # Process 'stated_prefs'
    df = process_columns(df, "stated_prefs", binary_conversion=False)

    # Process order cols
    df = process_columns(df, "order_stated_prefs", binary_conversion=False)
    df = process_columns(df, "order_lm_usecases", binary_conversion=False)

    # Demographics cleaning
    # Self-describe columns
    str_fmt_0 = "Self-describe [please type]"
    str_fmt_1 = "Prefer to self-describe"
    for col, answer_id, str_fmt in zip(
        ["religion", "ethnicity", "gender"],
        [1, 1, 4],
        [str_fmt_0, str_fmt_0, str_fmt_1],
    ):
        if pd.isna(df[col]).all() or pd.isna(df[f"{col}_{answer_id}_text"]).all():
            continue
        selected_answers = df[col].to_list()
        typed_answers = df[f"{col}_{answer_id}_text"].to_list()
        combined = []
        for select, typed_ans in zip(selected_answers, typed_answers):
            if isinstance(select, str) and select != str_fmt:
                combined.append(select)
            elif isinstance(typed_ans, str):
                combined.append(typed_ans.lower())
            elif pd.isna(typed_ans):
                combined.append("Prefer not to say")
            else:
                combined.append(typed_ans)
        df[col] = combined
        df.drop(f"{col}_{answer_id}_text", inplace=True, axis=1)

    # Replace error of one individual whose recorded Prolific age is 23 but put Under 18
    # Note they also recorded over 18 in the informed consent question
    df["age"] = df["age"].apply(lambda x: "18-24 years old" if x == "Under 18" else x)

    # Country cols
    df["same_birth_reside_country"] = df.apply(
        lambda x: "Yes" if x["birth_country"] == x["reside_country"] else "No", axis=1
    )

    # Timing clean up
    df["timing_duration_s"] = df["timing_duration_s"].astype(float)
    df["timing_duration_mins"] = np.round(df["timing_duration_s"] / 60, 2)

    # Separate based on column name prefixes
    meta_df = df[[col for col in df.columns if col.startswith("meta_")]]
    timing_df = df[[col for col in df.columns if col.startswith("timing_")]]
    order_df = df[[col for col in df.columns if col.startswith("order_")]]
    main_df = df[
        [
            col
            for col in df.columns
            if not col.startswith(("meta_", "timing_", "order_"))
        ]
    ]
    recombined_df = pd.concat([main_df, order_df, timing_df, meta_df], axis=1)

    # Record a copy of the survey before dropping rows for worker review explanations
    df = recombined_df.copy()
    df = df.reset_index().rename(columns={"index": "tmp_survey_id"})

    full_df = df.copy()

    # Drop rows where under 18
    df = df[df["consent_age"] == "I certify that I am 18 years of age or over"]
    logging.info(
        "Filtering rows of under 18 participants, number of examples remaining: %s",
        len(df),
    )
    df = df[df["consent"] == "Yes, I consent to take part"]
    logging.info(
        "Filtering rows without consent, number of examples remaining: %s", len(df)
    )

    # Reset qualtrics bug where progress = 96% are finished and have clicked onto conversations stage
    df.loc[df["meta_progress"] == "96", "meta_finished"] = "True"

    # Now remove any rows where survey wasn't finished
    df = df[df["meta_finished"] == "True"]
    logging.info(
        "Filtering rows that weren't finished, number of examples remaining: %s",
        len(df),
    )

    # Check unique username IDs are indeed unique
    assert not df["user_id"].duplicated().any()
    df = df.reset_index()

    return df, full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cutoff_date",
        type=str,
        required=True,
        help="cut off to keep on or after this date, format YYYY-MM-DD",
    )
    args = parser.parse_args()

    PROJECT_ROOT = find_project_root()
    INPUT_PATH = PROJECT_ROOT / "data" / "raw"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "interim"
    STORE_PATH = PROJECT_ROOT / "data" / "storage" / "mappings"
    REVIEW_PATH = PROJECT_ROOT / "data" / "review"

    # Load raw survey data
    raw_df = pd.read_csv(f"{INPUT_PATH}/HumanFeedbackSurvey.csv")

    # Load question mapping
    header_mapping_dict = {}
    for mapper in ["prefs", "usecases"]:
        headers_for_mapper = pd.read_csv(f"{STORE_PATH}/header_{mapper}_mapping.csv")
        header_dict = convert_headers_to_dict(headers_for_mapper)
        header_mapping_dict[mapper] = header_dict
    survey_df = clean_survey_columns(raw_df, header_mapping_dict)
    (
        survey_df,
        question_mapping_dict,
    ) = create_qualtrics_mapping(survey_df)

    survey_df, survey_for_review = clean_and_simplify_survey_data(
        survey_df, args.cutoff_date
    )

    # Save storage_dict as a JSON file for future analysis purposes
    ensure_dir_exists(STORE_PATH)
    with open(
        f"{STORE_PATH}/survey_question_mapping.json", "w", encoding="utf-8"
    ) as file:
        json.dump(question_mapping_dict, file)

    # Save cleaned survey data
    ensure_dir_exists(OUTPUT_PATH)
    survey_df.to_csv(f"{OUTPUT_PATH}/survey.csv", index=False)

    # Save full version for crowdworker review
    ensure_dir_exists(REVIEW_PATH)
    survey_for_review.to_csv(f"{REVIEW_PATH}/survey_for_review.csv", index=False)
