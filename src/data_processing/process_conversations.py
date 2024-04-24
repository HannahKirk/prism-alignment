"""
Loads raw conversations, then cleans and simplifies the data.
"""

import argparse
import logging
import json
import re
from datetime import datetime
import pandas as pd
import numpy as np
from src.utils.helper_funcs import (
    find_project_root,
    strip_html_tags,
    ensure_dir_exists,
    save_as_jsonl,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


def filter_conversations(input_data, cutoff_date, keywords_to_test):
    """
    Filter conversations based on the cutoff date and testing keywords.

    Args:
        input_data (list): A list of conversation data in JSON format.
        cutoff_date (str): The cutoff date in ISO format.
        keywords_to_test (list): A list of testing keywords.

    Returns:
        list: A filtered list of conversation data.
    """
    # Compile a regex pattern
    pattern = re.compile("(?i)" + "|".join(keywords_to_test))

    # Check if user_name contains any testing keywords
    def contains_keywords(user_name):
        return pattern.search(user_name) is not None

    # Filter the data
    output_data = []
    for entry in input_data:
        generated_datetime = datetime.fromisoformat(
            entry["example_info"]["generated_datetime"]
        )
        user_name = entry["example_info"]["user_name"]

        # Check if the entry meets the conditions for exclusion
        if generated_datetime >= datetime.fromisoformat(
            cutoff_date
        ) and not contains_keywords(user_name):
            output_data.append(entry)

    return output_data


def convert_headers_to_dict(header_df_to_convert, factor):
    """Convert the header header DataFrame to a dictionary"""
    mapping_dict = {}
    # Iterate through each row in the CSV data
    for _, row in header_df_to_convert.iterrows():
        mapping_dict[row[f"conversations_{factor}"]] = {"header": row["header"]}
    return mapping_dict


def process_timestamps(conversation_content, string_format):
    """
    Process timestamps in conversation content.

    Args:
        conversation_content (dict): A dictionary containing conversation content.
        string_format (str): The desired format for the timestamps.

    Returns:
        tuple: A tuple containing formatted initial timestamp, formatted end timestamp,
               duration in seconds, and duration in minutes.
    """
    initial_timestamp = conversation_content.get("initial_timestamp")
    end_timestamp = conversation_content.get("final_timestamp")
    if initial_timestamp is not None and end_timestamp is not None:
        # Convert initial timestamps to datetime objects
        initial_timestamp = pd.to_datetime(initial_timestamp, unit="ms")
        end_timestamp = pd.to_datetime(end_timestamp, unit="ms")
        # Calculate duration
        duration_s = np.round((end_timestamp - initial_timestamp).total_seconds(), 2)
        duration_mins = np.round(duration_s / 60, 2)
        # Convert timestamps to formatted strings
        formatted_initial = initial_timestamp.strftime(string_format)
        formatted_end = end_timestamp.strftime(string_format)
        return formatted_initial, formatted_end, duration_s, duration_mins
    return np.nan, np.nan, np.nan, np.nan


def process_attributes(attributes, header_dict):
    """
    Generic function to process attributes.

    Args:
        attributes (list): A list of attribute dictionaries.
        header_dict (dict): A dictionary mapping labels to headers.

    Returns:
        dict: A dictionary of processed attributes.
    """
    attributes_dict = {}
    for attr in attributes:
        label = attr["label"].split("</b>")[
            -1
        ]  # Handles labels with or without <b> tags
        new_label = header_dict.get(label, {}).get(
            "header", label
        )  # Fallback to original label if originally label was NaN (an allowed option)
        value = attr["value"]
        attributes_dict[new_label] = np.nan if value == "N/A" else int(value)

    return attributes_dict


# Process empty text with clear missing value
def process_text(text):
    """Simple function to check different empty modes"""
    if pd.isna(text) is True:
        return "EMPTY STRING"
    text = text.strip()
    if text == "":
        return "EMPTY STRING"
    if text == "None":
        return "EMPTY STRING"
    return text


def reformat_conversation(item, header_dict_for_performance, header_dict_for_choice):
    """
    Reformat the conversation data.

    Args:
        item (dict): The conversation item.
        header_dict_performance (dict): A dictionary mapping labels to performance headers.
        header_dict_choice (dict): A dictionary mapping labels to choice headers.

    Returns:
        dict: The reformatted conversation data.
    """
    # Load basic example info
    reformatted = {
        "convo_id": item["example_info"]["id"],
        "user_id": item["example_info"]["user_name"],
    }

    # Load content json
    conversation_content = json.loads(item["example_info"]["input_json"])

    # Process conversation attributes to outer level
    # Time
    string_format = "%Y-%m-%d %H:%M:%S"
    generated_datetime = datetime.fromisoformat(
        item["example_info"]["generated_datetime"]
    ).strftime(string_format)
    _, _, duration_s, duration_mins = process_timestamps(
        conversation_content, string_format
    )
    reformatted["generated_datetime"] = generated_datetime
    reformatted["timing_duration_s"] = duration_s
    reformatted["timing_duration_mins"] = duration_mins

    # Conversation type
    convo_type = (
        strip_html_tags(conversation_content["category"]["category"])
        .split(".")[0]
        .lower()
        .strip()
    )
    reformatted["conversation_type"] = convo_type

    # Opening prompt
    reformatted["opening_prompt"] = process_text(
        conversation_content["original_prompt"]
    )

    # Process conversation history to nested level
    conversation_history = []

    # First turn (note we index by 0 in the code base!)
    conversation_history.append(
        {
            "turn": 0,
            "role": "user",
            "content": process_text(conversation_content["original_prompt"]),
        }
    )
    answers = conversation_content["generated_answers"]
    best_answer_id = conversation_content["best_answer"]["id"]
    for ans in answers:
        provider = list(ans["model_name"].keys())[0]
        model_name = ans["model_name"][provider]["model_name"]
        conversation_history.append(
            {
                "turn": 0,
                "role": "model",
                "content": process_text(ans["text"]),
                "model_provider": provider,
                "model_name": model_name,
                "score": ans["score"],
                "if_chosen": ans["id"] == best_answer_id,
                "within_turn_id": ans["id"],
            }
        )

    #  Subsequent turns
    for i, user_msg in enumerate(
        conversation_content["chat_history"]["user"][1:], start=1
    ):
        # Append user message
        conversation_history.append(
            {
                "turn": i,
                "role": "user",
                "content": process_text(user_msg["text"]),
            }
        )

        # Iterate through model responses
        model_responses = conversation_content["historical_responses_model"][i - 1][
            "responses_model"
        ]
        chosen_response = conversation_content["chat_history"]["bot"][i]["text"]
        best_model_provider = list(
            conversation_content["best_answer"]["model_name"].keys()
        )[0]
        best_model_name = conversation_content["best_answer"]["model_name"][
            best_model_provider
        ]["model_name"]
        for j, resp in enumerate(model_responses):
            conversation_history.append(
                {
                    "turn": i,
                    "role": "model",
                    "content": process_text(resp["text"]),
                    "model_provider": best_model_provider,
                    "model_name": best_model_name,
                    "score": resp["score"],
                    "if_chosen": resp["text"] == chosen_response,
                    "within_turn_id": j,
                }
            )

    # Num turns are 1-indexed because its a count of turns
    num_turns = max([turn["turn"] for turn in conversation_history]) + 1
    reformatted["conversation_turns"] = num_turns

    # Append conversation history to outer level
    reformatted["conversation_history"] = conversation_history

    # Process finegrained json
    finegrained_json = json.loads(item["example_info"]["metadata_json"])
    performance_attrs = finegrained_json.get("response_attributes", [])
    performance_attributes_dict = process_attributes(
        performance_attrs, header_dict_for_performance
    )
    choice_attrs = finegrained_json.get("choice_attributes", [])
    choice_attributes_dict = process_attributes(choice_attrs, header_dict_for_choice)

    open_feedback = finegrained_json.get("indicate_explain", "")

    reformatted["performance_attributes"] = performance_attributes_dict
    reformatted["choice_attributes"] = choice_attributes_dict
    reformatted["open_feedback"] = open_feedback

    return reformatted


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

    # Open the file for reading with explicit encoding parameter
    with open(f"{INPUT_PATH}/conversations.json", "r", encoding="utf-8") as file:
        json_data = json.load(file)
    logging.info("Initial number of examples: %s", len(json_data))

    # Load mappings
    header_df = pd.read_csv(f"{STORE_PATH}/header_prefs_mapping.csv")
    header_dict_performance = convert_headers_to_dict(header_df, "performance_factors")
    header_dict_choice = convert_headers_to_dict(header_df, "choice_factors")

    # First filter (note these keywords are from old project names or testing purposes)
    testing_keywords = ["test", "perdi", "griffin", "arg", "gamma", "beta"]
    filtered_data = filter_conversations(json_data, args.cutoff_date, testing_keywords)
    logging.info("Filtered number of examples: %s", len(filtered_data))

    # Now reformat
    reformatted_data = [
        reformat_conversation(item, header_dict_performance, header_dict_choice)
        for item in filtered_data
    ]

    # Now save
    ensure_dir_exists(OUTPUT_PATH)
    save_as_jsonl(reformatted_data, f"{OUTPUT_PATH}/conversations.jsonl")
