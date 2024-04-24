"""
This script runs the checks on workers data to process approvals and bonuses."
"""

from datetime import datetime
import pandas as pd
import numpy as np
from src.utils.helper_funcs import find_project_root, ensure_dir_exists


def count_prefer_not_say(row):
    count = 0
    for value in row:
        if isinstance(value, str) and "prefer not to say" in value.lower():
            count += 1
    return count


def count_default_sliders(row, col_keywords):
    count = 0
    # Iterate over each item in the row along with its column name
    for col_name, value in row.items():
        for col_keyword in col_keywords:
            # Check if the column name contains the slider keyword and the value equals 50
            if col_name.startswith(col_keyword) and value == 50:
                count += 1
    return count


def calculate_average_free_text_length(row, cols, unit="char"):
    values = []
    # Iterate over each item in the row along with its column name
    for col_name in cols:
        # Compute length
        if pd.isna(row[col_name]):
            values.append(0)
        else:
            if unit == "char":
                values.append(len(row[col_name]))
            elif unit == "word":
                values.append(len(row[col_name].split(" ")))
            elif unit == "sent":
                values.append(len(row[col_name].split(".")))
    mean = np.mean(values)
    return mean


def evaluate_checks(checks_list):
    results = {}
    for check in checks_list:
        if check["type"] == "binary":
            results[check["name"]] = {
                "result": 1 if check["condition"] else 0,
                "signal": check["signal"],
                "stage": check["stage"],
                "type": "binary",
            }
        else:
            results[check["name"]] = {
                "result": check["condition"],
                "signal": check["signal"],
                "stage": check["stage"],
                "type": "values",
            }

    return results


# "https://researcher-help.prolific.com/hc/en-gb/articles/360009092394-Approvals-rejections-returns"
def run_internal_checks(row, mean_time_s, std_time_s):
    checks_list = [
        {
            "name": "no completion_code",
            "condition": row["completion_code"] != "CVQAGUHV",
            "signal": "investigate",
            "stage": "internal",
            "type": "binary",
        },
        {
            "name": "very_fast",
            "condition": row["time_taken"] < mean_time_s - 3 * std_time_s,
            "signal": "reject",
            "stage": "internal",
            "type": "binary",
        },
    ]
    results = evaluate_checks(checks_list)
    return results


def run_survey_checks(uid, survey_df, mean_time_s, std_time_s):
    slider_keywords = ["stated_prefs"]
    freetext_cols = ["system_string", "self_description"]

    base_checks_list = []

    # Check if uid exists in survey_df
    uid_exists = uid in survey_df["user_id"].values
    base_checks_list.append(
        {
            "name": "uid_not_found_survey",
            "condition": not uid_exists,
            "signal": "return",
            "stage": "survey",
            "type": "binary",
        }
    )

    # Check if uid is unique in survey_df
    uid_is_unique = survey_df[survey_df["user_id"] == uid].shape[0] == 1
    base_checks_list.append(
        {
            "name": "uid_not_unique",
            "condition": not uid_is_unique,
            "signal": "investigate",
            "stage": "survey",
            "type": "binary",
        }
    )

    # Evaluate the base checks
    base_checks_results = evaluate_checks(base_checks_list)

    # Proceed with other checks only if uid exists and is unique
    if uid_exists and uid_is_unique:
        row = survey_df[survey_df["user_id"] == uid].iloc[0]

        checks_list = [
            {
                "name": "no_consent",
                "condition": row["consent"] != "Yes, I consent to take part",
                "signal": "reject",
                "stage": "survey",
                "type": "binary",
            },
            {
                "name": "under_18",
                "condition": row["consent_age"]
                != "I certify that I am 18 years of age or over",
                "signal": "reject",
                "stage": "survey",
                "type": "binary",
            },
            {
                "name": "very_fast",
                "condition": row["timing_duration_s"] < mean_time_s - 3 * std_time_s,
                "signal": "reject",
                "stage": "survey",
                "type": "binary",
            },
            {
                "name": "mobile_operating_system",
                "condition": (
                    np.nan
                    if pd.isna(row["meta_operating_system"])
                    else any(
                        os in row["meta_operating_system"]
                        for os in ["iPhone", "Android"]
                    )
                ),
                "signal": "return",
                "stage": "survey",
                "type": "binary",
            },
            {
                "name": "prefer_not_say_count",
                "condition": count_prefer_not_say(row),
                "signal": "investigate",
                "stage": "survey",
                "type": "value",
            },
            {
                "name": "default_slider_count",
                "condition": count_default_sliders(row, slider_keywords),
                "signal": "investigate",
                "stage": "survey",
                "type": "value",
            },
            {
                "name": "mean_free_text_chars",
                "condition": calculate_average_free_text_length(
                    row, freetext_cols, unit="char"
                ),
                "signal": "investigate",
                "stage": "survey",
                "type": "value",
            },
            {
                "name": "mean_free_text_words",
                "condition": calculate_average_free_text_length(
                    row, freetext_cols, unit="word"
                ),
                "signal": "investigate",
                "stage": "survey",
                "type": "value",
            },
            {
                "name": "mean_free_text_sentences",
                "condition": calculate_average_free_text_length(
                    row, freetext_cols, unit="sent"
                ),
                "signal": "investigate",
                "stage": "survey",
                "type": "value",
            },
        ]

        other_checks_results = evaluate_checks(checks_list)
    else:
        row = None
        other_checks_results = {}

    # Combine the results from both checks
    combined_results = {**base_checks_results, **other_checks_results}

    return combined_results


def run_conversations_checks(uid, conversations_df, mean_time_s=None, std_time_s=None):
    slider_keywords = ["performance_attributes", "choice_attributes"]
    freetext_cols = ["open_feedback"]
    quota_threshold = 6

    base_checks_list = []

    # Check if uid exists in survey_df
    uid_exists = uid in conversations_df["user_id"].values
    base_checks_list.append(
        {
            "name": "uid_not_found_conversations",
            "condition": not uid_exists,
            "signal": "return",
            "stage": "conversations",
            "type": "binary",
        }
    )

    # Check if uid right quota
    uid_quota_filled = (
        conversations_df[conversations_df["user_id"] == uid].shape[0] >= quota_threshold
    )
    base_checks_list.append(
        {
            "name": "uid_quota_not_filled",
            "condition": not uid_quota_filled,
            "signal": "investigate",
            "stage": "conversations",
            "type": "binary",
        }
    )

    # Evaluate the base checks
    base_checks_results = evaluate_checks(base_checks_list)

    # Proceed with other checks only if uid exists
    if uid_exists:
        rows = []
        for _, row in conversations_df[conversations_df["user_id"] == uid].iterrows():
            rows.append(row)

        checks_list = [
            {
                "name": "num_convos_for_uid",
                "condition": len(rows),
                "signal": "investigate",
                "stage": "conversations",
                "type": "value",
            },
            {
                "name": "mean_conversation_turns",
                "condition": np.mean(
                    [row["conversation_turns"] for row in rows if row is not None]
                ),
                "signal": "investigate",
                "stage": "conversations",
                "type": "value",
            },
            {
                "name": "std_conversation_turns",
                "condition": np.std(
                    [row["conversation_turns"] for row in rows if row is not None]
                ),
                "signal": "investigate",
                "stage": "conversations",
                "type": "value",
            },
            {
                "name": "default_slider_count",
                "condition": np.mean(
                    [
                        count_default_sliders(row, slider_keywords)
                        for row in rows
                        if row is not None
                    ]
                ),
                "signal": "investigate",
                "stage": "conversations",
                "type": "value",
            },
            {
                "name": "mean_free_text_chars",
                "condition": np.mean(
                    [
                        calculate_average_free_text_length(
                            row, freetext_cols, unit="char"
                        )
                        for row in rows
                        if row is not None
                    ]
                ),
                "signal": "investigate",
                "stage": "conversations",
                "type": "value",
            },
            {
                "name": "mean_free_text_words",
                "condition": np.mean(
                    [
                        calculate_average_free_text_length(
                            row, freetext_cols, unit="word"
                        )
                        for row in rows
                        if row is not None
                    ]
                ),
                "signal": "investigate",
                "stage": "conversations",
                "type": "value",
            },
            {
                "name": "mean_free_text_sentences",
                "condition": np.mean(
                    [
                        calculate_average_free_text_length(
                            row, freetext_cols, unit="sent"
                        )
                        for row in rows
                        if row is not None
                    ]
                ),
                "signal": "investigate",
                "stage": "conversations",
                "type": "value",
            },
        ]

        other_checks_results = evaluate_checks(checks_list)
    else:
        row = None
        other_checks_results = {}

    # Combine the results from both checks
    combined_results = {**base_checks_results, **other_checks_results}

    return combined_results


def get_timing_expectations(df, col, unit_col="seconds"):
    timing_dict = {}
    if unit_col == "seconds":
        timing_dict["mean_s"] = df[col].mean()
        timing_dict["std_s"] = df[col].std()
        timing_dict["mean_m"] = timing_dict["mean_s"] / 60
        timing_dict["std_m"] = timing_dict["std_s"] / 60
    elif unit_col == "minutes":
        timing_dict["mean_m"] = df[col].mean()
        timing_dict["std_m"] = df[col].std()
        timing_dict["mean_s"] = timing_dict["mean_m"] * 60
        timing_dict["std_s"] = timing_dict["std_m"] * 60
    print(
        f"Mean time: {np.round(timing_dict['mean_m'],2)} mins, std: {np.round(timing_dict['std_m'],2)}"
    )
    return timing_dict


def process_worker_submissions(workers_df, survey_df, conversations_df):
    quota_threshold = 6

    check_functions = {
        "internal": run_internal_checks,
        "survey": run_survey_checks,
        "conversations": run_conversations_checks,
    }

    # Extend DataFrame to include new columns
    for check in check_functions.keys():
        workers_df[f"{check}_checks_dict"] = None
        workers_df[f"{check}_evidence_dict"] = None

    # Initialise
    workers_df["total_reject_signals"] = 0
    workers_df["total_return_signals"] = 0
    workers_df["rejects_list"] = None
    workers_df["returns_list"] = None
    workers_df["action"] = None

    # Get timing dictionaries
    timing_dict_of_dicts = {}
    for check, df, time_col in zip(
        check_functions.keys(),
        [workers_df, survey_df, conversations_df],
        ["time_taken", "timing_duration_s", "timing_duration_s"],
    ):
        print(f"Getting timing expectations for {check}")
        timing_dict_of_dicts[check] = get_timing_expectations(
            df, time_col, unit_col="seconds"
        )

    for index, row in workers_df.iterrows():
        rejects_list = []
        returns_list = []
        total_return_signals = total_reject_signals = 0
        for check, func in check_functions.items():
            uid = row["participant_id"]
            if check == "internal":
                timing_dict = timing_dict_of_dicts[check]
                check_dict = func(row, timing_dict["mean_s"], timing_dict["std_s"])
            elif check == "survey":
                timing_dict = timing_dict_of_dicts[check]
                check_dict = func(
                    uid, survey_df, timing_dict["mean_s"], timing_dict["std_s"]
                )
            elif check == "conversations":
                timing_dict = timing_dict_of_dicts[check]
                check_dict = func(
                    uid,
                    conversations_df,
                    timing_dict["mean_s"],
                    timing_dict["std_s"],
                )
            workers_df.at[index, f"{check}_checks_dict"] = check_dict
            # Count violations and warnings
            reject_signal_count = return_signal_count = 0
            for check_name, value in check_dict.items():
                if value["signal"] == "investigate":
                    col_name = f"inv_{check}_{check_name}"
                    workers_df.at[index, col_name] = value["result"]
                else:
                    col_name = f"{check}_{check_name}"
                    workers_df.at[index, col_name] = value["result"]
                    if value["result"] == 1:
                        if value["signal"] == "reject":
                            reject_signal_count += 1
                            rejects_list.append(check_name)
                        elif value["signal"] == "return":
                            return_signal_count += 1
                            returns_list.append(check_name)

            workers_df.at[index, f"{check}_evidence_dict"] = {
                "return_signals": return_signal_count,
                "reject_signals": reject_signal_count,
            }
            total_return_signals += return_signal_count
            total_reject_signals += reject_signal_count

        # Update total violations and warnings
        workers_df.at[index, "total_return_signals"] = total_return_signals
        workers_df.at[index, "total_reject_signals"] = total_reject_signals
        workers_df.at[index, "rejects_list"] = rejects_list
        workers_df.at[index, "returns_list"] = returns_list
        workers_df.at[index, "action"] = "FILL_IN"

    workers_df = workers_df.rename(
        columns={"inv_conversations_num_convos_for_uid": "conversations_quota"}
    )

    # Approvals and returns
    for index, row in workers_df.iterrows():
        if row["status"] == "ACTIVE":
            workers_df.at[index, "action"] = "ACTIVE"
        # Approve people who returned but completed
        elif row["status"] == "APPROVED":
            workers_df.at[index, "action"] = "APPROVED"
        elif row["status"] == "REJECTED":
            workers_df.at[index, "action"] = "REJECTED"
        elif row["status"] == "RETURNED":
            if (
                (row["conversations_quota"] >= quota_threshold)
                and (row["total_reject_signals"] == 0)
                and (row["total_return_signals"] <= 1)
            ):
                workers_df.at[index, "action"] = "APPROVE_POST_RETURN"
            else:
                workers_df.at[index, "action"] = "RETURN_LEAVE"
        # Approve people who timed out but completed
        elif row["status"] == "TIMED-OUT":
            if (
                (row["conversations_quota"] >= quota_threshold)
                and (row["total_reject_signals"] == 0)
                and (row["total_return_signals"] == 0)
            ):
                workers_df.at[index, "action"] = "APPROVE_POST_TIMEOUT"
            else:
                workers_df.at[index, "action"] = "REJECT"
        elif row["status"] == "AWAITING REVIEW":
            if (
                (row["conversations_quota"] >= quota_threshold)
                and (row["total_reject_signals"] == 0)
                and (row["total_return_signals"] == 0)
            ):
                workers_df.at[index, "action"] = "APPROVE"
            elif (row["total_reject_signals"] == 0) and (
                row["total_return_signals"] >= 1
            ):
                workers_df.at[index, "action"] = "RETURN_REQUEST"
            elif row["conversations_quota"] < quota_threshold:
                workers_df.at[index, "action"] = "REJECT"
            elif row["total_reject_signals"] > 0:
                workers_df.at[index, "action"] = "REJECT"
            else:
                workers_df.at[index, "action"] = "CHECK"

    # Replace NaN in conversation quota with 0 (they did not go to phase 2)
    workers_df["conversations_quota"] = workers_df["conversations_quota"].fillna(0)

    # For duplicates, approve first submission , reject second submission
    for participant_id in workers_df["participant_id"].unique():
        uid_occurrences = workers_df[
            workers_df["participant_id"] == participant_id
        ].index
        uid_not_unique = len(uid_occurrences) > 1
        if uid_not_unique:
            for idx in uid_occurrences[1:]:
                workers_df.at[idx, "action"] = "APPROVE_DUPLICATE"

    # Create column for people who have nearly completed
    workers_df["simple_action"] = workers_df["action"].map(lambda x: x.split("_")[0])
    workers_df = workers_df.rename(
        columns={"action": "action_full", "simple_action": "action"}
    )

    # Work out additional bonus for workers affected by DoS attacks
    count = 0
    for index, row in workers_df.iterrows():
        study_name = row["study_name"]
        time_started = row["started_at"]
        # Convert time_started to datetime if it's a string
        if isinstance(time_started, str):
            time_started = datetime.fromisoformat(time_started.rstrip("Z"))
        # Define the start and end times
        crash1_start_time = datetime.fromisoformat("2023-11-28T17:50:00.556000")
        crash1_end_time = datetime.fromisoformat("2023-11-28T23:30:00.556000")
        crash2_start_time = datetime.fromisoformat("2023-12-01T17:15:00.000000")
        crash2_end_time = datetime.fromisoformat("2023-12-01T23:00:00.000000")
        action = row["action"]
        if action == "APPROVE":
            if study_name in ["us_rep_sample", "uk_rep_sample"]:
                if crash1_start_time < time_started < crash1_end_time:
                    count += 1
                    workers_df.at[index, "action_full"] = "APPROVE_BONUS_CRASH_ONE"
                    workers_df.at[index, "action"] = "APPROVE"
                elif crash2_start_time < time_started < crash2_end_time:
                    count += 1
                    workers_df.at[index, "action_full"] = "APPROVE_BONUS_CRASH_TWO"
                    workers_df.at[index, "action"] = "APPROVE"

    print(f"Number of people who got bonus: {count}")

    # Work out which workers need partial payment for partial completion of task
    partial_reason = []
    partial_flag = []
    include_flag = []
    include_reason = []
    for index, row in workers_df.iterrows():
        if "DUPLICATE" in row["action_full"]:
            include_flag.append(0)
            include_reason.append("No, Duplicate submission")
            partial_reason.append("No, Already paid (duplicate)")
            partial_flag.append(0)
        elif (row["action"] == "APPROVED") or (row["action"] == "APPROVE"):
            partial_reason.append("No, Paid in full")
            partial_flag.append(0)
            if row["conversations_quota"] >= 6:
                include_flag.append(1)
                include_reason.append("Yes, Completed >= 6 convos")
            else:
                include_flag.append(1)
                include_reason.append(
                    f"Yes, Completed {row['conversations_quota']} convos"
                )
        else:
            if row["conversations_quota"] >= 6:
                partial_reason.append("Yes, convos completed")
                partial_flag.append(1)
                include_flag.append(1)
                include_reason.append("Yes, Completed >= 6 convos")
            elif 1 <= row["conversations_quota"] < 6:
                partial_reason.append("Yes, some convos completed")
                partial_flag.append(1)
                include_flag.append(1)
                include_reason.append(
                    f"Yes, Completed {row['conversations_quota']} convos"
                )
            elif (
                (row["survey_uid_not_found_survey"] == 0)
                and (row["conversations_uid_not_found_conversations"] == 1)
                and (row["total_reject_signals"] == 0)
            ):
                partial_reason.append("Yes, survey completed")
                partial_flag.append(1)
                include_flag.append(1)
                include_reason.append("Yes, Completed survey")
            elif (
                (row["survey_uid_not_found_survey"] == 0)
                and (row["conversations_uid_not_found_conversations"] == 0)
                and (row["total_reject_signals"] == 0)
                and (row["survey_mobile_operating_system"] == 1)
            ):
                partial_reason.append("Yes (mobile operating system)")
                partial_flag.append(1)
                include_flag.append(1)
                include_reason.append("Yes, Completed convos on mobile")
            elif (
                (row["survey_uid_not_found_survey"] == 0)
                and (row["conversations_uid_not_found_conversations"] == 0)
                and (row["total_reject_signals"] == 0)
            ):
                partial_reason.append("Yes")
                partial_flag.append(1)
                include_flag.append(1)
                include_reason.append("Yes, ")
            else:
                partial_reason.append("No, Not paid")
                partial_flag.append(0)
                include_flag.append(0)
                include_reason.append(
                    "No, not completed any stage or other reasons for reject"
                )
    workers_df["partial_reason"] = partial_reason
    workers_df["partial_flag"] = partial_flag
    workers_df["include_flag"] = include_flag
    workers_df["include_reason"] = include_reason

    # Reorder columns
    decision_cols = [
        "conversations_quota",
        "total_return_signals",
        "total_reject_signals",
        "rejects_list",
        "returns_list",
        "action_full",
        "action",
        "partial_flag",
        "partial_reason",
        "include_flag",
        "include_reason",
    ]

    dynamic_cols = [col for col in workers_df.columns if "inv" in col]
    fixed_cols = [
        col for col in workers_df.columns if col not in dynamic_cols + decision_cols
    ]
    new_col_order = fixed_cols + dynamic_cols + decision_cols
    workers_df = workers_df[new_col_order]

    return workers_df


def main():
    PROJECT_ROOT = find_project_root()
    REVIEW_PATH = PROJECT_ROOT / "data" / "review"
    INTERIM_PATH = PROJECT_ROOT / "data" / "interim"
    worker_df = pd.read_csv(f"{REVIEW_PATH}/workers_for_review.csv")
    survey_df = pd.read_csv(f"{INTERIM_PATH}/survey.csv")
    # This is same as the clean file because we didn't drop any needed rows
    # Load conversations data
    conversations_df = pd.read_json(f"{INTERIM_PATH}/conversations.jsonl", lines=True)
    result_df = process_worker_submissions(worker_df, survey_df, conversations_df)
    ensure_dir_exists(REVIEW_PATH)
    result_df.to_csv(f"{REVIEW_PATH}/workers_with_checks.csv", index=False)


if __name__ == "__main__":
    main()
