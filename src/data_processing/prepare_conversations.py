"""
Loads conversations data after initial clean and prepares it for public release.
Also creates the long-format utterances.jsonl data.
"""

import logging
import pandas as pd
from src.utils.helper_funcs import find_project_root, ensure_dir_exists, save_as_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


def get_balance_conversations(convos, SEED=0, return_per_user_counts=False):
    """Balance conversations to have equal number of each conversation type per user i.e,
    retain the maximum possible balanced subset."""
    # Identify unique conversation types
    unique_convo_types = convos["conversation_type"].unique()

    balanced_convos_list = []
    for user in convos["user_id"].unique():
        group = convos[convos["user_id"] == user]
        convo_counts = group["conversation_type"].value_counts()

        # Check if all conversation types are present
        if not all(
            convo_type in convo_counts.index for convo_type in unique_convo_types
        ):
            continue

        # Find the minimum count among the conversation types
        min_count = convo_counts.loc[unique_convo_types].min()

        # Sample min_count conversations of each type
        for c in unique_convo_types:
            balanced_convos_list.append(
                group[group["conversation_type"] == c].sample(
                    min_count, random_state=SEED
                )
            )

    # Concatenate the balanced conversations
    balanced_convos = pd.concat(balanced_convos_list)
    if return_per_user_counts:
        # Count of each conversation type per user before and after balancing
        user_convo_counts_before = (
            convos.groupby(["user_id", "conversation_type"])
            .size()
            .unstack(fill_value=0)
        )
        user_convo_counts_after = (
            balanced_convos.groupby(["user_id", "conversation_type"])
            .size()
            .unstack(fill_value=0)
        )
        return balanced_convos, user_convo_counts_before, user_convo_counts_after
    else:
        return balanced_convos


def main():
    """Prepare Conversations"""
    PROJECT_ROOT = find_project_root()
    INPUT_PATH = PROJECT_ROOT / "data" / "interim"
    OUTPUT_PATH = PROJECT_ROOT / "data"
    # Check the outdir exists
    ensure_dir_exists(OUTPUT_PATH)

    # Load interim conversations data
    conversations = pd.read_json(f"{INPUT_PATH}/conversations.jsonl", lines=True)
    logging.info("Original number of conversations: %s", len(conversations))

    # Load clean survey
    survey = pd.read_json(f"{OUTPUT_PATH}/survey.jsonl", lines=True)

    # Map prolific ids to user ids (note this is not publicly avaliable)
    id_mapping = pd.read_csv(f"{INPUT_PATH}/prolific_id_mapping.csv")
    # Convert into a dictionary
    id_mapping = id_mapping.set_index("prolific_id")["user_id"].to_dict()
    # Now map to new id
    conversations["user_id"] = conversations["user_id"].map(id_mapping)

    # Remove any conversations with an id not in surveys
    conversations = conversations[conversations["user_id"].isin(survey["user_id"])]

    logging.info(
        "Number of conversations from users also in survey: %s", len(conversations)
    )

    # Reset conversation id for public release
    conversations.index = pd.RangeIndex(start=0, stop=len(conversations), step=1)
    conversations["conversation_id"] = conversations.index.map(lambda i: f"c{i}")
    conversations.drop(columns=["convo_id"], inplace=True)

    # Create balanced subset
    balanced_conversations = get_balance_conversations(
        conversations, SEED=0, return_per_user_counts=False
    )

    # Add indicator column if conversation is included in balanced subset
    conversations["included_in_balanced_subset"] = conversations[
        "conversation_id"
    ].isin(balanced_conversations["conversation_id"])
    conversations["included_in_balanced_subset"].value_counts()
    counts_included = (
        conversations["included_in_balanced_subset"].value_counts().to_string()
    )

    logging.info(
        "Number of balanced conversations across convo types: %s", counts_included
    )

    # Sort order for release
    id_cols = [
        "conversation_id",
        "user_id",
        "included_in_balanced_subset",
    ]
    other_cols = [col for col in conversations.columns if col not in id_cols]
    new_column_order = id_cols + other_cols
    conversations = conversations[new_column_order]

    # Create utterances
    utterances = []
    for _, row in conversations.iterrows():
        # Store convo-level attributes
        convo_id = row["conversation_id"]
        user_id = row["user_id"]
        included_in_balanced_subset = row["included_in_balanced_subset"]
        conversation_type = row["conversation_type"]
        conversation_history = row["conversation_history"]

        # Keep track of the current turn and user input
        current_turn = None
        current_user_input = None
        for item in conversation_history:
            if item["role"] == "user":
                # Update the current turn and user input
                current_turn = item["turn"]
                current_user_input = item["content"]
            elif item["role"] == "model" and item["turn"] == current_turn:
                # For each model response, create a new row with the current turn, user input, and other convo-level attributes
                utterances.append(
                    {
                        "conversation_id": convo_id,
                        "user_id": user_id,
                        "turn": current_turn,
                        "within_turn_id": item["within_turn_id"],
                        "included_in_balanced_subset": included_in_balanced_subset,
                        "conversation_type": conversation_type,
                        "user_prompt": current_user_input,
                        "model_response": item["content"],
                        "model_name": item["model_name"],
                        "model_provider": item["model_provider"],
                        "score": item["score"],
                        "if_chosen": item["if_chosen"],
                    }
                )

    # Create dataframe from list of utterances
    utterances = pd.DataFrame(utterances)
    # Create interactions_id based on grouped convo_id and turn
    interactions_id = [
        f"int{x}" for x in utterances.groupby(["conversation_id", "turn"]).ngroup()
    ]
    utterances.insert(0, "interaction_id", interactions_id)
    # Create utterance_id based on index
    utterances.index = pd.RangeIndex(start=0, stop=len(utterances), step=1)
    utterances_id = utterances.index.map(lambda i: f"ut{i}")
    utterances.insert(0, "utterance_id", utterances_id)

    # Save both to jsonl
    save_as_jsonl(
        conversations, f"{OUTPUT_PATH}/conversations.jsonl", is_already_records=False
    )
    save_as_jsonl(
        utterances, f"{OUTPUT_PATH}/utterances.jsonl", is_already_records=False
    )


if __name__ == "__main__":
    main()
