"""
Loads survey data after initial clean and prepares it for public release.
"""

import configparser
import logging
import ast
import argparse
import openai
import pandas as pd
import numpy as np
import pycountry
from src.utils.helper_funcs import (
    find_project_root,
    ensure_dir_exists,
    download_from_git_url,
    save_as_jsonl,
)
from src.data_processing.lm_label_cleaner import simple_text_clean, get_labels


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


def country_to_iso(input_name, manual_fixes):
    """Convert country name to ISO code."""
    if input_name == "Prefer not to say":
        return "Prefer not to say"
    # First, check for manual fixes
    elif input_name in manual_fixes:
        return manual_fixes[input_name]
    try:
        # Attempt using the 'name' attribute
        country = pycountry.countries.get(name=input_name)
        if country:
            return country.alpha_3

        # Attempt using the 'official_name' attribute
        country = pycountry.countries.get(official_name=input_name)
        if country:
            return country.alpha_3

        # Attempt using the 'search_fuzzy' method
        country = pycountry.countries.search_fuzzy(input_name)[0]
        return country.alpha_3

    except (AttributeError, LookupError):
        # Message if no match found
        logging.info("Could not convert %s to ISO code", input_name)
        return np.nan


def iso_to_country(input_iso):
    """Convert ISO code to back to standardised country name."""
    if input_iso == "Prefer not to say":
        return "Prefer not to say"
    try:
        return pycountry.countries.get(alpha_3=input_iso).name
    except (AttributeError, LookupError):
        # Message if no match found
        logging.info("Could not convert %s to country name", input_iso)
        return np.nan


def map_lm_labels(x, labels_dict):
    """Maps cleaned text to labels"""
    if pd.isna(x) is False:
        return labels_dict[simple_text_clean(x)]
    else:
        return x


def clean_lm_labels(x):
    """# If there are "/" split on the "/", capitalize first letter of each word,
    rejoin with a space and the /"""
    if "atheist/agnostic/secular" in x:
        return "Non-religious"
    if "/" in x:
        return " / ".join([word.title() for word in x.split("/")])
    return x.capitalize()


def reformat_response(item, outer_cols, categorised_cols, location_cols, nested_cols):
    """Reformat response to JSONL format"""
    reformatted = {}
    for col in outer_cols:
        reformatted[col] = item[col]

    for col in categorised_cols:
        outer_stub = col.split("_")[-1]
        if outer_stub not in reformatted.keys():
            reformatted[outer_stub] = {}
        if "self_describe" in col:
            inner_stub = "self_described"
        elif "lm_categorised" in col:
            inner_stub = "gpt4_categorised"
        reformatted[outer_stub][inner_stub] = item[col]

    for col in location_cols:
        if "location" not in reformatted.keys():
            reformatted["location"] = {}
        reformatted["location"][col] = item[col]

    for col in nested_cols:
        reformatted[col.replace("_all", "")] = ast.literal_eval(item[col])

    return reformatted


def main():
    """Run main"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged_version",
        action="store_true",
        help="Specify this flag to use the crowdworker merged version of the survey",
    )
    args = parser.parse_args()

    PROJECT_ROOT = find_project_root()
    # Set data path
    INPUT_PATH = PROJECT_ROOT / "data" / "interim"
    OUTPUT_PATH = PROJECT_ROOT / "data"

    # Config params
    config = configparser.ConfigParser()
    CONFIG_PATH = PROJECT_ROOT / "config" / "config.ini"
    config.read(CONFIG_PATH)
    API_TOKEN = config.get("openai", "api_key")

    # Set up your OpenAI API credentials
    openai.api_key = API_TOKEN

    # Load survey data
    survey = pd.read_csv(
        f'{INPUT_PATH}/{"merged_" if args.merged_version else ""}survey.csv'
    )

    # String cleaning for long answers
    education_mapping = {
        "Graduate or professional degree (MA, MS, MBA, PhD, JD, MD, DDS)": "Graduate / Professional degree",
        "Vocational or Similar": "Vocational",
    }
    work_mapping = {
        "Unemployed and looking for work": "Unemployed, seeking work",
        "Unemployed and not looking for work": "Unemployed, not seeking work",
        "A homemaker or stay-at-home parent": "Homemaker / Stay-at-home parent",
    }

    martial_mapping = {
        "Divorced/Separated": "Divorced / Separated",
    }

    # Replace these values with a simple func
    def replace_manual_values(x, remap_dict):
        if x in remap_dict.keys():
            return remap_dict[x]
        return x

    survey["education"] = survey["education"].map(
        lambda x: replace_manual_values(x, education_mapping)
    )
    survey["employment_status"] = survey["employment_status"].map(
        lambda x: replace_manual_values(x, work_mapping)
    )

    survey["marital_status"] = survey["marital_status"].map(
        lambda x: replace_manual_values(x, martial_mapping)
    )

    for category_string in ["ethnicity", "religion"]:
        if category_string == "ethnicity":
            groups = """White, Black/African, Hispanic/Latino, Asian, Indigenous/First Peoples, Middle Eastern/Arab, Mixed, Other"""
            extra_instruct = (
                """If someone mentions a mix of ethnicities, include as "Mixed"."""
            )
        elif category_string == "religion":
            groups = """Christian, Jewish, Atheist/Agnostic/Secular, Muslim, Hindu, Sikh, Buddhist, Folk Religion, Spiritual, Other"""
            extra_instruct = ""

        logging.info("Using gpt-4 to clean labels for %s", category_string)
        cat_labels_dict = get_labels(
            INPUT_PATH, survey, category_string, groups, extra_instruct
        )

        survey[category_string] = survey[category_string].map(
            lambda x: simple_text_clean(x)
        )

        survey[f"lm_categorised_{category_string}"] = (
            survey[category_string].map(cat_labels_dict).str.lower()
        )

        survey.loc[
            survey[category_string] == "prefer not to say",
            f"lm_categorised_{category_string}",
        ] = "prefer not to say"

        survey = survey.rename(
            columns={category_string: f"self_described_{category_string}"}
        )

        # Do a bit of additional cleaning on the categorised strings
        survey[f"lm_categorised_{category_string}"] = survey[
            f"lm_categorised_{category_string}"
        ].apply(clean_lm_labels)

    # Add country using pycountry library
    # Dictionary of manual fixes
    manual_fixes = {
        "Hong Kong (S.A.R.)": "HKG",
        "Venezuela, Bolivarian Republic of...": "VEN",
        "Turkey": "TUR",
    }

    # Add ISO codes and remap to standardised country name
    for col in ["birth_country", "reside_country"]:
        survey[f"{col}ISO"] = survey[col].apply(
            lambda x: country_to_iso(x, manual_fixes)
        )
        survey[col] = survey[f"{col}ISO"].map(iso_to_country)

    # Download region
    iso_data = download_from_git_url(
        "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
    )
    # Merging for birth_country
    survey = survey.merge(
        iso_data[["alpha-3", "region", "sub-region"]],
        how="left",
        left_on="birth_countryISO",
        right_on="alpha-3",
    )

    # If birth_country is Prefer not to say, then set region and sub-region to Prefer not to say
    survey.loc[
        survey["birth_country"] == "Prefer not to say",
        ["region", "sub-region", "same_birth_reside_country"],
    ] = "Prefer not to say"
    survey.rename(
        columns={
            "region": "birth_region",
            "sub-region": "birth_subregion",
        },
        inplace=True,
    )

    # Merging for reside_country
    survey = survey.merge(
        iso_data[["alpha-3", "region", "sub-region"]],
        how="left",
        left_on="reside_countryISO",
        right_on="alpha-3",
    )

    # If reside_country is Prefer not to say, then set region and sub-region to Prefer not to say
    survey.loc[
        survey["reside_country"] == "Prefer not to say",
        ["region", "sub-region", "same_birth_reside_country"],
    ] = "Prefer not to say"
    survey.rename(
        columns={
            "region": "reside_region",
            "sub-region": "reside_subregion",
        },
        inplace=True,
    )

    # Remap regions
    def create_new_region(row):
        if row["birth_countryISO"] == "USA":
            return "US"
        elif row["birth_countryISO"] == "GBR":
            return "UK"
        elif row["birth_countryISO"] == "MEX":
            return "Latin America and the Caribbean"
        elif row["birth_subregion"] == "Latin America and the Caribbean":
            return "Latin America and the Caribbean"
        elif row["birth_subregion"] == "Northern America":
            return "Northern America"
        elif row["birth_subregion"] == "Western Asia":
            return "Middle East"
        elif row["birth_subregion"] == "Australia and New Zealand":
            return "Australia and New Zealand"
        else:
            return row["birth_region"]

    survey["special_region"] = survey.apply(create_new_region, axis=1)

    # Reset index for a user ID intended for public release
    survey.index = pd.RangeIndex(start=0, stop=len(survey), step=1)
    survey["prolific_id"] = survey["user_id"].copy()
    survey["user_id"] = survey.index.map(lambda x: f"user{x}")

    # Save mapping locally (prolific ID is not be publicly released)
    survey[["prolific_id", "user_id"]].to_csv(
        f"{INPUT_PATH}/prolific_id_mapping.csv", index=False
    )

    # Rename
    survey = survey.rename(
        columns={
            "meta_recorded_date": "generated_datetime",
            "conversations_quota": "num_completed_conversations",
        }
    )

    # Typing
    survey["num_completed_conversations"] = survey[
        "num_completed_conversations"
    ].astype(int)

    # ID cols
    id_cols = [
        "user_id",
        "survey_only",
        "num_completed_conversations",
    ]

    # Preamble cols
    preamble_cols = ["consent", "consent_age"]

    # LM use cols
    lm_use_cols = [
        "lm_familiarity",
        "lm_indirect_use",
        "lm_direct_use",
        "lm_frequency_use",
    ]

    # Free-text cols
    free_text_cols = ["self_description", "system_string"]

    # Demographics
    age = ["age"]
    gender = ["gender"]
    employment = ["employment_status"]
    marital_status = ["marital_status"]
    education = ["education"]

    location = [
        "birth_country",
        "birth_countryISO",
        "birth_region",
        "birth_subregion",
        "reside_country",
        "reside_region",
        "reside_subregion",
        "reside_countryISO",
        "same_birth_reside_country",
        "special_region",
    ]

    ethnicity = [
        "self_described_ethnicity",
        "lm_categorised_ethnicity",
    ]

    religion = [
        "self_described_religion",
        "lm_categorised_religion",
    ]

    languages = ["english_proficiency"]

    demographics = (
        age
        + gender
        + employment
        + education
        + marital_status
        + location
        + ethnicity
        + religion
        + languages
    )

    # Timings cols
    datetime_cols = ["timing_duration_s", "timing_duration_mins", "generated_datetime"]

    # Study metadata cols
    study_columns = ["study_id", "study_locale"]

    # Convert to JSONL
    nested_cols = [
        "lm_usecases_all",
        "stated_prefs_all",
        "order_lm_usecases_all",
        "order_stated_prefs_all",
    ]
    categorised_cols = religion + ethnicity
    location_cols = location
    remaining_demo_cols = [
        col
        for col in demographics
        if col not in categorised_cols and col not in location_cols
    ]
    outer_cols = (
        id_cols
        + preamble_cols
        + lm_use_cols
        + free_text_cols
        + remaining_demo_cols
        + study_columns
        + datetime_cols
    )

    # Reformat as JSONL
    reformatted_data = []
    for _, row in survey.iterrows():
        reformatted_data.append(
            reformat_response(
                row, outer_cols, categorised_cols, location_cols, nested_cols
            )
        )

    # Now save
    ensure_dir_exists(OUTPUT_PATH)
    save_as_jsonl(reformatted_data, f"{OUTPUT_PATH}/survey.jsonl")


if __name__ == "__main__":
    main()
