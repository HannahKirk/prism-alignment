"""
This script pulls data from a Dynabench task and saves them to a JSON file.
"""

import json
import configparser
import logging
from pathlib import Path
import requests
from src.utils.helper_funcs import find_project_root, ensure_dir_exists

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


def download_data(download_url, task_id, output_path):
    """
    Downloads conversations from a Dynabench task and saves them to a JSON file.

    Args:
        download_url (str): The URL to download the conversations from.
        task_id (int): The ID of the Dynabench task to download conversations for.
        output_path (str): The path to save the downloaded conversations to.

    Returns:
        None
    """
    # Endpoint and headers
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    # Data payload
    data = {"task_id": task_id}

    # Make the POST request with a timeout of 10 seconds
    response = requests.post(download_url, headers=headers, json=data, timeout=50)

    # Check if the request was successful
    if response.status_code == 200:
        logging.info("Response code ok: %s", response.status_code)
        # Write the returned JSON data to a file with UTF-8 encoding
        ensure_dir_exists(output_path)
        with open(f"{output_path}/conversations.json", "w", encoding="utf-8") as f:
            json.dump(response.json(), f, indent=4)
        logging.info("Complete. Conversations download finished.")
    else:
        logging.error(
            "Request failed with status code %s: %s",
            response.status_code,
            response.text,
        )


def main():
    """Pull conversations"""
    # Defining project root as a static variable
    PROJECT_ROOT = find_project_root()
    # Config
    config = configparser.ConfigParser()
    CONFIG_PATH = PROJECT_ROOT / "config" / "config.ini"
    config.read(CONFIG_PATH)
    DOWNLOAD_URL = config.get("dynabench", "download_url")
    PROJECT_ROOT = find_project_root()
    OUTPUT_PATH = Path(PROJECT_ROOT, "data", "raw")
    TASK_ID = config.get("dynabench", "task_id")

    download_data(DOWNLOAD_URL, TASK_ID, OUTPUT_PATH)


if __name__ == "__main__":
    main()
