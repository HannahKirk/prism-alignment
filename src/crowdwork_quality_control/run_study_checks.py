"""
This script runs the checks on study data to keep track of costs and quotas during data collection."
"""

import logging
import pandas as pd
import numpy as np
from src.utils.helper_funcs import find_project_root

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


def prolific_formula(num_participants, reward):
    participant_fees = num_participants * reward
    service_fees = participant_fees * 0.33
    VAT = np.round(service_fees * 0.2, 2)
    total = participant_fees + service_fees + VAT
    return total


def calculate_open_spots(row):
    if row["status"] == "UNPUBLISHED":
        return 0
    elif row["status"] == "COMPLETED":
        return 0
    else:
        open_spots = row["total_available_places"] - row["places_taken"]
        return open_spots


def calculate_overquota(row):
    if row["status"] == "UNPUBLISHED":
        return 0
    else:
        overquota = row["number_of_submissions"] - row["total_available_places"]
        if overquota < 0:
            return 0
        else:
            return overquota


def calculate_remaining_to_open(row, quota_col):
    remainder = row[quota_col] - row["number_of_approved_submissions"]
    return remainder


def calculate_completion_rate(row, quota_col):
    remainder = (row["places_taken"]) / row[quota_col] * 100
    return np.round(remainder, 1)


def calculate_my_cost(row, which_quota="orig_quota"):
    num_participants = row[which_quota]
    reward = row["reward"]
    cost = prolific_formula(num_participants, reward)
    if "REP_SAMPLE" in row["study_type"]:
        cost = cost + 466.21
    return cost


def check_payments(cost_estimate):
    if cost_estimate == np.nan:
        return np.nan
    elif cost_estimate < 9:
        return "UNDER"
    elif cost_estimate > 9:
        return "OVER"
    else:
        return "OK"


def calculate_adjusted_cost(row, which_group, which_quota):
    num_participants = row[which_quota]
    if row[f"payment_check_{which_group}"] == "UNDER":
        new_reward = row[f"median_time_hr_{which_group}"] * 9
        new_cost = prolific_formula(num_participants, new_reward)
        if "REP_SAMPLE" in row["study_type"]:
            new_cost = new_cost + 466.21
        return new_cost
    else:
        old_cost = row["total_cost"]
        return old_cost


def main():
    PROJECT_ROOT = find_project_root()
    REVIEW_PATH = PROJECT_ROOT / "data" / "review"
    INTERIM_PATH = PROJECT_ROOT / "data" / "interim"

    # Load the data
    studies = pd.read_csv(REVIEW_PATH / "studies.csv")
    regional_counts = pd.read_csv(REVIEW_PATH / "prolific_counts_per_country.csv")
    study_quota = pd.read_csv(REVIEW_PATH / "study_quota_per_country.csv")
    workers = pd.read_csv(REVIEW_PATH / "workers_with_checks.csv")

    # Set up naming
    studies["study_group"] = studies["internal_name_code"].map(
        lambda x: x.split(" - ")[-1]
    )

    studies["study_locale"] = studies["study_group"].map(
        lambda x: x.replace("pilot_study", "")
        .strip()
        .replace("_", " ")
        .replace("launchb", "")
        .lower()
        .replace("rep sample", "")
        .strip()
    )
    studies["if_pilot"] = studies["internal_name_code"].str.contains("pilot")
    studies["launch_round"] = studies["internal_name_code"].map(
        lambda x: "b" if "launchb" in x else "a"
    )

    # Match regions
    un_regions = []
    coarse_regions = []
    for _, row in studies.iterrows():
        group = row["study_locale"]
        if group == "asia":
            un_region = "asia"
            coarse_region = "asia"
        else:
            un_region = regional_counts["UN_regional_group"][
                regional_counts["reside_country"].str.lower() == group
            ].values[0]
            coarse_region = regional_counts["regional_group"][
                regional_counts["reside_country"].str.lower() == group
            ].values[0]

        un_regions.append(un_region.lower())
        coarse_regions.append(coarse_region.lower())

    studies["un_region"] = un_regions
    studies["coarse_region"] = coarse_regions

    studies[["study_locale", "un_region", "coarse_region"]].to_csv(
        INTERIM_PATH / "region_mapping.csv", index=False
    )

    # Merge quotas
    study_quota["group"] = study_quota["group"].str.lower()
    studies = studies.merge(
        study_quota, left_on="study_locale", right_on="group", how="left"
    )
    # Set quotas to five when if_pilot is true
    for col in ["orig_quota", "reduced_quota"]:
        studies.loc[studies["if_pilot"] is True, col] = 5
    # Tracking of count of approved people per study
    approved_counts = []
    for _, study in studies.iterrows():
        study_id = study["study_id"]
        study_workers = workers[workers["study_id"] == study_id]
        approved_workers = study_workers[
            study_workers["action"].isin(["APPROVE", "APPROVED"])
        ]
        approved_counts.append(len(approved_workers))

    studies["number_of_approved_submissions"] = approved_counts

    # Tracking of study places
    studies["number_of_open_spots"] = studies.apply(calculate_open_spots, axis=1)
    studies["overquota_count"] = studies.apply(calculate_overquota, axis=1)
    for quota_col in ["orig_quota", "reduced_quota"]:
        studies[f"remainder_to_open_{quota_col}"] = studies.apply(
            calculate_remaining_to_open, axis=1, quota_col=quota_col
        )
        studies[f"completion_rate_{quota_col}"] = studies.apply(
            calculate_completion_rate, axis=1, quota_col=quota_col
        )

    # Tracking of timings
    median_time_all = []
    median_time_approved = []
    for study_id in studies["study_id"].unique():
        study_workers = workers[workers["study_id"] == study_id]
        approved_workers = study_workers[study_workers["action"] == "APPROVE"]
        median_time_approved.append(
            np.round(approved_workers["time_taken"].median() / 60 / 60, 1)
        )
        median_time_all.append(
            np.round(study_workers["time_taken"].median() / 60 / 60, 1)
        )

    # Costing rebasing (to keep track of costs)
    cost_cols = ["reward", "total_cost"]
    for c in cost_cols:
        studies[c] = studies[c] / 100
    studies["my_total_cost_orig"] = studies.apply(
        calculate_my_cost, axis=1, which_quota="orig_quota"
    )
    studies["my_total_cost_reduced"] = studies.apply(
        calculate_my_cost, axis=1, which_quota="reduced_quota"
    )

    studies["my_total_cost_approved"] = studies.apply(
        calculate_my_cost, axis=1, which_quota="number_of_approved_submissions"
    )
    cost_cols.extend(["my_total_cost_orig", "my_total_cost_reduced"])

    studies["median_time_hr_all"] = median_time_all
    studies["median_time_hr_approved"] = median_time_approved

    logging.info("Mean time all: %s", studies["median_time_hr_all"].median())
    logging.info(
        "Mean time approved only: %s", studies["median_time_hr_approved"].median()
    )

    # Tracking of adjusted costs
    for c in ["median_time_hr_all", "median_time_hr_approved"]:
        c_stub = c.split("_")[-1]
        studies[f"estimated_cost_{c_stub}"] = np.round(9 / studies[c], 2)
        studies[f"payment_check_{c_stub}"] = studies[f"estimated_cost_{c_stub}"].map(
            check_payments
        )
        cost_cols.extend([f"estimated_cost_{c_stub}", f"payment_check_{c_stub}"])
        for quota_col in [
            "orig_quota",
            "reduced_quota",
            "total_available_places",
            "number_of_submissions",
            "number_of_approved_submissions",
        ]:
            q_stub = quota_col
            studies[f"adjusted_cost_{c_stub}_{q_stub}"] = studies.apply(
                calculate_adjusted_cost,
                axis=1,
                which_group=c_stub,
                which_quota=quota_col,
            )
            cost_cols.append(f"adjusted_cost_{c_stub}_{q_stub}")

    # Sort cols
    base_cols = [
        "internal_name_code",
        "study_id",
        "study_type",
        "study_locale",
        "un_region",
        "coarse_region",
        "if_pilot",
        "orig_quota",
        "reduced_quota",
        "status",
    ]

    place_tracker_cols = [
        "total_available_places",
        "places_taken",
        "number_of_submissions",
        "number_of_open_spots",
        "overquota_count",
        "remainder_to_open_orig_quota",
        "completion_rate_orig_quota",
        "remainder_to_open_reduced_quota",
        "completion_rate_reduced_quota",
        "number_of_approved_submissions",
    ]

    cost_tracker_cols = ["median_time_hr_all", "median_time_hr_approved"] + cost_cols

    other_cols = [
        c
        for c in studies.columns
        if c not in [base_cols] + [place_tracker_cols] + [cost_tracker_cols]
    ]

    # Save studies
    studies = studies[base_cols + place_tracker_cols + cost_tracker_cols]
    studies.to_csv(REVIEW_PATH / "studies_with_checks.csv", index=False)

    # Save aggregated studies
    country_grpby_cols = ["study_locale", "un_region", "coarse_region"]
    un_region_grpby_cols = ["un_region"]
    coarse_region_grpby_cols = ["coarse_region"]
    runs = ["country", "un_region", "coarse_region"]
    for run, group_by_cols in zip(
        runs, [country_grpby_cols, un_region_grpby_cols, coarse_region_grpby_cols]
    ):
        # Group by the specified columns
        studies_grp = studies.groupby(group_by_cols)

        # Aggregate
        aggregated = studies_grp.agg(
            {
                "orig_quota": "sum",
                "reduced_quota": "sum",
                "total_available_places": "sum",
                "places_taken": "sum",
                "number_of_submissions": "sum",
                "number_of_approved_submissions": "sum",
                "number_of_open_spots": "sum",
                "remainder_to_open_orig_quota": "sum",
                "completion_rate_orig_quota": "mean",
                "remainder_to_open_reduced_quota": "sum",
                "completion_rate_reduced_quota": "mean",
            }
        ).reset_index()
        # Reorder the columns to bring group by columns to the front
        other_cols = [col for col in aggregated.columns if col not in group_by_cols]
        ordered_columns = group_by_cols + other_cols

        # Reassign the DataFrame with ordered columns
        aggregated = aggregated[ordered_columns]

        aggregated.to_csv(REVIEW_PATH / f"studies_{run}_with_checks.csv", index=False)

    # Totals
    column_names = []
    values = []

    # Total cost
    for c in [c for c in studies.columns if "cost" in c]:
        column_names.append(c)
        values.append(studies[c].sum())

    for c in ["orig_quota", "reduced_quota", "places_taken", "number_of_submissions"]:
        column_names.append(c)
        values.append(studies[c].sum())

    for c in ["median_time_hr_all", "median_time_hr_approved"]:
        column_names.append(c)
        values.append(studies[c].mean())

    summary_table = pd.DataFrame(
        {"metric": column_names, "value": [np.round(v, 2) for v in values]}
    )

    summary_table.to_csv(REVIEW_PATH / "studies_summary.csv", index=False)


if __name__ == "__main__":
    main()
