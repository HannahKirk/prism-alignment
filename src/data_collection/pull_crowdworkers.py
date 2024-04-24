"""
This script interacts with the Prolific API to process studies and download user data.
"""

import os
import configparser
import io
import logging
import pandas as pd
import requests
from src.utils.helper_funcs import find_project_root, ensure_dir_exists


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


# Function to get project details
def get_project(project_id, base_url, headers):
    """
    Get project details from the Prolific API.

    Args:
        project_id (str): The ID of the project.
        base_url (str): The base URL of the Prolific API.
        headers (dict): The headers for the API request.

    Returns:
        dict: The project details as a dictionary.
    """
    url = f"{base_url}/projects/{project_id}/"
    response = requests.get(url, headers=headers, timeout=20)
    return response.json()


# Function to list all studies in a project
def list_studies_in_project(project_id, base_url, headers):
    """
    List all studies in a project from the Prolific API.

    Args:
        project_id (str): The ID of the project.
        base_url (str): The base URL of the Prolific API.
        headers (dict): The headers for the API request.

    Returns:
        list: A list of study details.
    """
    url = f"{base_url}/projects/{project_id}/studies/"
    response = requests.get(url, headers=headers, timeout=20)
    json_response = response.json()
    return json_response["results"]


# Function to list study submissions
def list_study_submissions(study_id, base_url, headers):
    """
    List study submissions from the Prolific API.

    Args:
        study_id (str): The ID of the study.
        base_url (str): The base URL of the Prolific API.
        headers (dict): The headers for the API request.

    Returns:
        tuple: A tuple containing the submissions and the count of submissions.
    """
    url = f"{base_url}/studies/{study_id}/submissions/"
    response = requests.get(url, headers=headers, timeout=20)
    json_response = response.json()
    count = json_response["meta"]["count"]
    submissions = json_response["results"]
    return submissions, count


# Function to download user data and return as DataFrame
def download_user_data(study_id, base_url, headers):
    """
    Download user data from the Prolific API and return as a DataFrame.

    Args:
        study_id (str): The ID of the study.
        base_url (str): The base URL of the Prolific API.
        headers (dict): The headers for the API request.

    Returns:
        pd.DataFrame: The downloaded user data as a DataFrame.
    """
    url = f"{base_url}/studies/{study_id}/export/"
    response = requests.get(url, headers=headers, timeout=60)  # Added timeout argument
    if response.status_code == 200:
        # Convert CSV response to DataFrame
        return pd.read_csv(io.StringIO(response.text))
    logging.error("Error: %s", response.status_code)
    return None


def clean_internal_name(name):
    """
    Clean internal name by removing unwanted characters.

    Args:
        name (str): The name to be cleaned.

    Returns:
        str: The cleaned internal name.
    """
    return (
        name.replace("griffin - ", "")
        .replace(" ", "_")
        .replace("-", "")
        .replace("__", "_")
        .lower()
    )


def append_unique_data(user_data, data_path, id_column="submission_id"):
    file_path = f"{data_path}/workers_for_review.csv"
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        new_data = user_data[~user_data[id_column].isin(existing_data[id_column])]
        if not new_data.empty:
            new_data.to_csv(file_path, mode="a", header=False, index=False)
            logging.info("Appended %s new records to %s", len(new_data), data_path)
        else:
            logging.info("No new records to append.")
    else:
        user_data.to_csv(file_path, mode="w", header=True, index=False)
        logging.info("Created file and saved data to %s", data_path)


def main():
    # Defining project root as a static variable
    PROJECT_ROOT = find_project_root()

    # Config params
    config = configparser.ConfigParser()
    CONFIG_PATH = PROJECT_ROOT / "config" / "config.ini"
    config.read(CONFIG_PATH)
    API_TOKEN = config.get("prolific", "api_key")
    PROJECT_ID = config.get("prolific", "project_id")
    DATA_PATH = PROJECT_ROOT / "data" / "review"
    ensure_dir_exists(DATA_PATH)

    # Base URL and headers for Prolific API
    BASE_URL = "https://api.prolific.com/api/v1"
    HEADERS = {"Authorization": f"Token {API_TOKEN}"}

    # project_details = get_project(PROJECT_ID, base_url, headers)
    studies = list_studies_in_project(PROJECT_ID, BASE_URL, HEADERS)
    studies_df = pd.DataFrame(studies)
    studies_df = studies_df.rename(columns={"id": "study_id"})
    studies_df["internal_name_code"] = studies_df["internal_name"].map(
        clean_internal_name
    )

    status_counts = studies_df["status"].value_counts().to_string()
    logging.info("Study status counts: %s", status_counts)

    live_status = ["ACTIVE", "AWAITING REVIEW", "PAUSED", "COMPLETED"]
    review_ids = studies_df["study_id"][
        studies_df["status"].isin(live_status)
    ].to_list()
    review_names = studies_df["internal_name_code"][
        studies_df["status"].isin(live_status)
    ].to_list()

    logging.info("There are %s live studies", len(review_ids))

    all_user_data = []
    # Loop through studies or submissions as needed
    for study_id, study_name in zip(review_ids, review_names):
        logging.info("Loading study %s", study_name)
        # List study submissions
        _, count = list_study_submissions(study_id, BASE_URL, HEADERS)
        logging.info("This study has %s submissions", count)
        # Download data
        user_data = download_user_data(study_id, BASE_URL, HEADERS)
        user_data.columns = [c.lower().replace(" ", "_") for c in user_data.columns]
        user_data["study_id"] = study_id
        user_data["study_name"] = study_name
        all_user_data.append(user_data)

    # Concat frames and export
    ensure_dir_exists(DATA_PATH)
    all_user_data = pd.concat(all_user_data, axis=0, ignore_index=True)
    all_user_data.to_csv(f"{DATA_PATH}/workers_for_review.csv", index=False)
    studies_df.to_csv(f"{DATA_PATH}/studies.csv", index=False)


if __name__ == "__main__":
    main()
