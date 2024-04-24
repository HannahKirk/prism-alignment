"""
This module contains data loader. For now it loads from file.
You can load direct from HF or Github URL [TODO].
"""

import pandas as pd
from pandas import json_normalize
from src.utils.helper_funcs import find_project_root


def load_data(
    survey=True, conversations=True, utterances=True, models=True, metadata=True
):
    """Function to load our various data splits from file.

    Args:
        survey (bool, optional): whether to load survey data. Defaults to True.
        conversations (bool, optional): whether to load conversations data. Defaults to True.
        utterances (bool, optional): whether to load utterances data (long format) of conversations. Defaults to True.
        models (bool, optional): where to load models data. Defaults to True.

    Returns:
        dictionary: Dictionary of dataframes for any specified sub-datasets.
    """
    PROJECT_ROOT = find_project_root()
    INPUT_PATH = PROJECT_ROOT / "data"
    data_dictionary = {}
    if survey:
        data_dictionary["survey"] = pd.read_json(
            INPUT_PATH / "survey.jsonl", lines=True
        )
    if conversations:
        data_dictionary["conversations"] = pd.read_json(
            INPUT_PATH / "conversations.jsonl", lines=True
        )
    if utterances:
        data_dictionary["utterances"] = pd.read_json(
            INPUT_PATH / "utterances.jsonl", lines=True
        )
    if models:
        data_dictionary["models"] = pd.read_json(
            INPUT_PATH / "models.jsonl", lines=True
        )
    if metadata:
        data_dictionary["metadata"] = pd.read_json(
            INPUT_PATH / "metadata" / "metadata.jsonl", lines=True
        )
    return data_dictionary


def unnest_columns(df, nested_columns=None):
    """
    Unnest the columns from the nested json in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame.

    Returns:
        pandas.DataFrame: The df DataFrame with unnested columns.
    """
    # Set defaults
    if nested_columns is None:
        nested_columns = [
            column
            for column in df.columns
            if any(
                isinstance(item, (list, dict))
                for item in df[column]
                if pd.notnull(item)
            )
        ]
    # Normalize each column and join back to the original DataFrame
    for col in nested_columns:
        df_expanded = json_normalize(df[col])
        df_expanded.columns = [
            f"{col}_{subcol}" for subcol in df_expanded.columns
        ]  # Prefixing column names
        df = df.join(df_expanded)
    return df


def pivot_to_wide_format(input_df):
    """Function to pivot long format data (utterances) to wide format (interactions).

    Args:
        df (pd.DataFrame): Long format dataframe of utterances.

    Returns:
        pd.DataFrame: Wide format dataframe of interactions.
    """
    df = input_df.copy()
    # Set the fixed columns
    df["suffix"] = df.groupby(["interaction_id"]).cumcount()
    df["suffix"] = df["suffix"].map({0: "a", 1: "b", 2: "c", 3: "d"})

    # Set the index to the grouping columns and the new suffix
    df.set_index(
        [
            "interaction_id",
            "conversation_id",
            "turn",
            "user_prompt",
            "user_id",
            "suffix",
        ],
        inplace=True,
    )

    # Unstack the suffix level
    wide_df = df.unstack(level="suffix")

    # Flatten the MultiIndex in columns
    wide_df.columns = ["_".join(col).strip() for col in wide_df.columns.values]

    # Reset the index to make 'convo_id', 'turn', 'user_prompt', 'user_id' regular columns again
    wide_df.reset_index(inplace=True)

    return wide_df
