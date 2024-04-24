"""
This module contains helper functions used across the repo.
"""

import pathlib
import re
import os
import json
from io import StringIO
from datetime import datetime
import requests
import pandas as pd


def find_project_root():
    """
    Find the project root directory.

    Returns:
        The path to the project root directory.

    Raises:
        FileNotFoundError: If the '.project_root' file indicating the project
            root directory is not found.
    """
    current_path = pathlib.Path(__file__).parent.resolve()
    for parent in current_path.parents:
        if (parent / ".project_root").exists():
            return parent
    raise FileNotFoundError(
        "Could not find the '.project_root' file indicating the project root directory."
    )


def strip_html_tags(s):
    """
    Removes HTML tags from a string.

    Args:
        s (str): The string to remove HTML tags from.

    Returns:
        str: The input string with HTML tags removed.
    """
    if isinstance(s, str):
        return re.sub(r"<[^>]+>", "", s)
    return s


def to_snake_case(s):
    """
    Convert CamelCase to snake_case.

    Args:
        s (str): The string to convert.

    Returns:
        str: The input string converted to snake_case.
    """
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def ensure_dir_exists(directory):
    """
    Ensure that the specified directory exists. If it doesn't, create it.

    Parameters:
    directory (str): The directory to check or create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_from_git_url(url):
    """Sends a GET request to a URL"""
    response = requests.get(url, timeout=30)

    # Check if the request was successful
    if response.status_code == 200:
        # Read the content of the file into a pandas DataFrame
        data = StringIO(response.text)
        df = pd.read_csv(data)
        return df
    else:
        print("Failed to retrieve the file")


def save_as_jsonl(data, file_path, is_already_records=True):
    """
    Save the data as a JSONL file.

    Args:
        data (list): The data to be saved.
        file_path (str): The path of the file to save.
        is_already_records (bool, optional): Whether the data is already in records format. Defaults to True.

    Returns:
        None
    """
    if is_already_records:
        save_data = data
    else:
        save_data = data.to_dict(orient="records")
    with open(file_path, "w", encoding="utf-8") as f:
        for item in save_data:
            json_string = json.dumps(item)
            f.write(json_string + "\n")


def get_new_filename(file_path):
    """Function to append datetime to filename if it exists"""
    if file_path.exists():
        # Get current datetime
        now = datetime.now()
        # Format datetime as a string
        datetime_str = now.strftime("%Y%m%d_%H%M%S")
        # Append datetime before file extension
        new_filename = file_path.stem + "_" + datetime_str + file_path.suffix
        return file_path.parent / new_filename
    else:
        return file_path
