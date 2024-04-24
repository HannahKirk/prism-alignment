"""
Uses GPT API to assist in labelling of self-describe text for survey data.
Note we found some issues with these labels so verified them manually with human annotators.
"""

import os
import logging
import openai
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


def label_descriptors(descriptors_list, category_string, groups, extra_instruct):
    """Use the GPT API to label descriptors"""
    prompt_template = (
        f"Here is a list of self-described {category_string}:\n{descriptors_list}.\n"
    )
    system_msg = (
        "You are a helpful assistant who assists in social science data analysis."
    )
    user_msg = f"""{prompt_template}\nI want to label each item in the list as some new categories.
The categories are: {groups}.
Use Other as sparingly as possible.
{extra_instruct}
\nGo through each item in my list and label it. Please return a text format to convert as a csv. The first column is the original item and second column is the new category label.
Use ';' as the seperator, and a newline for each row.
Just response with the csv and nothing else."""
    # Call the API
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    # Extract the generated name from the API response
    raw_text = response["choices"][0]["message"]["content"].strip()
    return user_msg, raw_text


# Simple clean
def simple_text_clean(text):
    """Remove punctuation, lowercase, strip leading and trailing whitespace"""
    text = text.strip().lower().replace(";", ",")
    # Remove puncutation too
    text = "".join([c for c in text if c not in ("?", ".", "'", '"', "!", ",")])
    return text


def convert_string_response_to_df(string_response):
    """Convert the string response from the GPT API to a DataFrame"""
    # Splitting the text into lines and then columns
    data = [line.split(";") for line in string_response.strip().split("\n")]
    # Creating a DataFrame
    labels_df = pd.DataFrame(data, columns=["self_described", "categorised"])
    return labels_df


def get_labels(STORE_PATH, survey, category_string, groups, extra_instruct):
    """Add new labels to the existing labels file, and return a dictionary of labels"""
    labels_file = f"{STORE_PATH}/{category_string}_gpt4_labels.csv"
    prompt_file = f"{STORE_PATH}/{category_string}_gpt4_prompt.txt"

    # Load existing labels if the file exists
    if os.path.exists(labels_file):
        labels_df = pd.read_csv(labels_file)
    else:
        labels_df = pd.DataFrame(columns=["self_described", "categorised"])

    # Convert DataFrame to a dictionary for easy lookup
    existing_labels_dict = dict(
        zip(labels_df["self_described"], labels_df["categorised"])
    )

    # Extract unique descriptors from the survey
    raw_descriptors = [
        d
        for d in survey[category_string].unique()
        if d != "Prefer not to say" and pd.isna(d) is False
    ]

    # Check each descriptor against existing labels
    new_labels = []
    for descriptor in raw_descriptors:
        clean_descriptor = simple_text_clean(descriptor)
        if clean_descriptor not in existing_labels_dict:
            new_labels.append(clean_descriptor)

    # If there are new labels to be classified
    if new_labels:
        # Log the number of new labels to be processed
        logging.info("Labels to process for %s: %s", category_string, len(new_labels))
        prompt, generated_labels_text = label_descriptors(
            new_labels, category_string, groups, extra_instruct
        )
        new_labels_df = convert_string_response_to_df(generated_labels_text)
        labels_df = pd.concat([labels_df, new_labels_df]).drop_duplicates()
        # Drop rows where column is "original_item":"new_item" (i.e, header rows)
        labels_df = labels_df[labels_df["self_described"] != "original_item"]

        # Save the updated labels DataFrame
        labels_df.to_csv(labels_file, index=False)

        # Save the prompt only if the prompt file doesn't exist
        if not os.path.exists(prompt_file):
            logging.info("Saving prompt format at %s", prompt_file)
            with open(prompt_file, "w", encoding="utf-8") as file:
                file.write(prompt)
        else:
            logging.info("Prompt format already saved at %s", prompt_file)
    else:
        logging.info("No new labels to process for %s", category_string)

    # Convert to remapping dict for use in the survey processing
    labels_df.set_index("self_described", inplace=True)
    labels_dict = labels_df["categorised"].to_dict()

    return labels_dict
