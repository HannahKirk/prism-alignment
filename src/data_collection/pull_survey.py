"""
This script exports survey responses from Qualtrics and saves them as a CSV file.
See: https://api.qualtrics.com/u9e5lh4172v0v-survey-response-export-guide
"""

import configparser
import io
import sys
import time
import logging
import zipfile
import requests
from src.utils.helper_funcs import find_project_root, ensure_dir_exists

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


def download_survey_data(data_center, survey_id, api_token, output_path):
    """
    Downloads survey data from Qualtrics and saves it as a CSV file.

    Args:
        data_center (str): The data center to use for the API request.
        survey_id (str): The ID of the survey to download data from.
        api_token (str): The API token to use for the API request.
        output_path (str): The path to save the downloaded data to.
    """
    # Setting static parameters
    request_check_progress = 0.0
    progress_status = "inProgress"
    url = f"https://{data_center}.qualtrics.com/API/v3/surveys/{survey_id}/export-responses/"
    headers = {
        "content-type": "application/json",
        "x-api-token": api_token,
    }

    # Step 1: Creating Data Export
    data = {
        "format": "csv",
        "useLabels": True,
        "seenUnansweredRecode": -1,
        "includeDisplayOrder": True,
        # "includeLabelColumns":True
    }

    download_request_response = requests.request(
        "POST", url, json=data, headers=headers, timeout=10
    )
    logging.info(download_request_response.json())

    try:
        progress_id = download_request_response.json()["result"]["progressId"]
    except KeyError:
        logging.error(download_request_response.json())
        sys.exit(2)

    is_file = None

    max_retries = 5
    retry_count = 0

    # Step 2: Checking on Data Export Progress and waiting until export is ready
    while (
        progress_status != "complete"
        and progress_status != "failed"
        and is_file is None
    ):
        if is_file is None:
            logging.info("File not ready")
        else:
            logging.info("ProgressStatus= %s", progress_status)

        request_check_url = url + progress_id
        request_check_response = requests.request(
            "GET", request_check_url, headers=headers, timeout=10
        )

        try:
            is_file = request_check_response.json()["result"]["fileId"]
            file_id = request_check_response.json()["result"]["fileId"]
        except KeyError:
            pass

        logging.info(request_check_response.json())
        request_check_progress = request_check_response.json()["result"][
            "percentComplete"
        ]
        logging.info("Download is %s complete", str(request_check_progress))
        progress_status = request_check_response.json()["result"]["status"]

        if progress_status not in ["complete", "failed"]:
            # Implement exponential backoff with a maximum number of retries
            retry_count += 1
            if retry_count > max_retries:
                print("Exceeded maximum retries. Exiting.")
                sys.exit(1)

            # Calculate the next sleep interval using exponential backoff
            sleep_interval = 10**retry_count
            logging.info("Retrying in %s seconds...", sleep_interval)
            time.sleep(sleep_interval)

    # Step 3: Downloading file
    request_download_url = url + file_id + "/file"
    request_download = requests.request(
        "GET", request_download_url, headers=headers, stream=True, timeout=10
    )

    # Step 4: Unzipping the file
    with zipfile.ZipFile(io.BytesIO(request_download.content)) as zip_file:
        ensure_dir_exists(output_path)
        zip_file.extractall(output_path)
    logging.info("Complete. Survey download finished.")


def main():
    "Pull survey"
    # Defining project root as a static variable
    PROJECT_ROOT = find_project_root()

    # Config params
    config = configparser.ConfigParser()
    CONFIG_PATH = PROJECT_ROOT / "config" / "config.ini"
    config.read(CONFIG_PATH)
    MY_API_TOKEN = config.get("qualtrics", "api_key")
    THIS_SURVEY_ID = config.get("qualtrics", "survey_id")
    DATA_CENTER = config.get("qualtrics", "data_center")
    OUTPUT_PATH = PROJECT_ROOT / "data" / "raw"

    download_survey_data(DATA_CENTER, THIS_SURVEY_ID, MY_API_TOKEN, OUTPUT_PATH)


if __name__ == "__main__":
    main()
