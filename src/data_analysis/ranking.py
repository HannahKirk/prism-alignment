"""
Ranking and score normalisation functions.
"""

import itertools
import math
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from src.viz.plot_config import *

################################################################################################
# Normalisation
################################################################################################


def normalize_score(row, user_stats):
    """Normalize the score for each user to account for within user variation"""
    user_mean = user_stats.loc[row["user_id"], "mean"]
    user_std = user_stats.loc[row["user_id"], "std"]
    # Check if standard deviation is zero
    if user_std != 0:
        normalized_score = (row["score"] - user_mean) / user_std
    else:
        # Set normalized_score to 0
        normalized_score = 0
    return normalized_score


def calculate_within_turn_rank(group):
    """Calculate the within-turn rank for each utterance an interaction turn,
    adding some random noise to break ties"""
    # Add a small random noise to the scores
    noise = np.random.uniform(-0.0001, 0.0001, len(group))
    group["score_with_noise"] = group["score"] + noise

    # Rank the utterances with the noisy score
    group["within_turn_rank"] = group["score_with_noise"].rank(
        ascending=False, method="first"
    )

    # Drop the temporary 'score_with_noise' column
    group = group.drop(columns=["score_with_noise"])

    return group


def get_additional_utterance_metrics(utterances):
    # Calculate normalised score
    for mode in ["openers", "full"]:
        if mode == "openers":
            df_for_normalisation = utterances[utterances["turn"] == 0]
        elif mode == "full":
            df_for_normalisation = utterances.copy()
        user_stats = df_for_normalisation.groupby("user_id")["score"].agg(
            ["mean", "std"]
        )

        utterances[f"normalized_score_{mode}"] = utterances.apply(
            normalize_score, axis=1, args=(user_stats,)
        )

    # Apply within_turn rank
    utterances = utterances.groupby(["interaction_id"]).apply(
        calculate_within_turn_rank, include_groups=False
    )

    utterances = utterances.reset_index().drop(columns=["level_1"])

    return utterances


################################################################################################
# Get Battles
################################################################################################


# Function to extract the battles from wide format (interactions) data
def extract_battles(openers, tie_thresh=5, keep_user_id=False):
    """Extract the battles from the conversations."""
    battles = []
    for _, row in openers.iterrows():
        model_store = {}
        user_id = row["user_id"]
        for suffix in ["a", "b", "c", "d"]:
            model_name = row[f"model_name_{suffix}"]
            model_score = row[f"score_{suffix}"]
            if pd.notna(model_name) and pd.notna(model_score):
                model_store[model_name] = model_score

        # Iterate over all pairs of models
        for (model_a, score_a), (model_b, score_b) in itertools.combinations(
            model_store.items(), 2
        ):
            # Logic to determine the winner
            if abs(score_a - score_b) <= tie_thresh:
                winner = "tie"
            elif score_a > score_b:
                winner = "model_a"
            else:
                winner = "model_b"

            # Creating the battle dictionary
            battle = {
                "model_a": model_a,
                "score_a": score_a,
                "model_b": model_b,
                "score_b": score_b,
                "winner": winner,
            }
            if keep_user_id:
                battle["user_id"] = user_id
            battles.append(battle)

    return battles


################################################################################################
# Our implementation of rank centrality
################################################################################################


def compute_rank_centrality(battles, iterations=1000, alpha=1):
    tie_mask = battles["winner"] == "tie"
    winner_a_mask = battles["winner"] == "model_a"
    winner_b_mask = battles["winner"] == "model_b"

    # Handle ties
    ties = battles.loc[tie_mask, ["model_a", "model_b"]].values
    tie_comparisons = np.concatenate([ties, ties[:, [1, 0]]])

    # Handle non-ties
    winner_a_comparisons = battles.loc[winner_a_mask, ["model_a", "model_b"]].values
    winner_b_comparisons = battles.loc[winner_b_mask, ["model_b", "model_a"]].values

    # Concatenate all comparisons
    comparisons = np.concatenate(
        [tie_comparisons, winner_a_comparisons, winner_b_comparisons]
    )
    c_df = pd.DataFrame(comparisons, columns=["winners", "losers"])

    # Default to order of winners
    unique_items = list(c_df["winners"].value_counts().index)
    unique_losers = sorted(list(c_df["losers"].unique()))

    # Add unique losers that are not already in the unique_items_ordered list
    for loser in unique_losers:
        if loser not in unique_items:
            unique_items.append(loser)

    # Get n items
    n_items = len(unique_items)

    # Get item2index
    item2index = {item: i for i, item in enumerate(unique_items)}

    # Set up the transition matrix A of 1s (with reg param)
    A = np.ones((len(unique_items), len(unique_items))) * alpha

    # Fill the diagonals with 0 (self-loops)
    np.fill_diagonal(A, 0)

    # Populate transition matrix
    for w, l in comparisons:
        # Rows are the winning model, column is losing model
        A[item2index[w], item2index[l]] += 1

        A_sum = (
            A[np.triu_indices_from(A, 1)] + A[np.tril_indices_from(A, -1)]
        ) + 1e-6  # to prevent division by zero

    # Normalise by ratio
    A[np.triu_indices_from(A, 1)] /= A_sum
    A[np.tril_indices_from(A, -1)] /= A_sum

    # Normalise by max degree
    d_max = n_items - 1
    A /= d_max

    # Add self-loop probabilities
    # Find residual probability per column
    residuals = 1 - np.sum(A, axis=0)

    # Fill the diagonal with the residuals
    np.fill_diagonal(A, residuals)

    # Get steady state by iters
    vec = np.array([1 / n_items] * n_items)
    scores = np.matmul(np.linalg.matrix_power(A, iterations), vec)

    score_dict = {item: scores[index] for item, index in item2index.items()}

    return score_dict


################################################################################################
# FASTCHAT FUNCTIONS: see https://github.com/lm-sys/FastChat
# We use the same functions and args for greater comparability.
################################################################################################


def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for rd, model_a, model_b, winner in battles[
        ["model_a", "model_b", "winner"]
    ].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == "model_a":
            sa = 1
        elif winner == "model_b":
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    return dict(rating)


def compute_mle_elo(
    df, SCALE=400, BASE=10, INIT_RATING=1000, worst_model="flan-t5-xxl"
):  # worst_model = "oasst-pythia-12b"):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # calibrate worst model to 800
    elo_scores += 800 - elo_scores[models[worst_model]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round=1000, frac=1.0):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        tmp_battles = battles.sample(frac=frac, replace=True)
        rows.append(func_compute_elo(tmp_battles))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def get_median_elo_from_bootstrap(bootstrap_df):
    median = dict(bootstrap_df.quantile(0.5))
    median = {k: int(v + 0.5) for k, v in median.items()}
    return median


def compute_pairwise_win_fraction(battles, model_order, limit_show_number=None):
    # Times each model wins as Model A
    a_win_ptbl = pd.pivot_table(
        battles[battles["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting times each model wins as Model B
    b_win_ptbl = pd.pivot_table(
        battles[battles["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting number of A-B pairs
    num_battles_ptbl = pd.pivot_table(
        battles, index="model_a", columns="model_b", aggfunc="size", fill_value=0
    )

    # Computing the proportion of wins for each model as A and as B
    # against all other models
    row_beats_col_freq = (a_win_ptbl + b_win_ptbl.T) / (
        num_battles_ptbl + num_battles_ptbl.T
    )

    if model_order is None:
        prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
        model_order = list(prop_wins.keys())

    if limit_show_number is not None:
        model_order = model_order[:limit_show_number]

    # Arrange ordering according to proprition of wins
    row_beats_col = row_beats_col_freq.loc[model_order, model_order]
    return row_beats_col
