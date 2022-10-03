"""Run experiment."""

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from decimal import Decimal

import lightgbm as lgbm
import numpy as np
import optuna
import pandas as pd
import requests
from sklearn.metrics import ndcg_score

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname).1s %(asctime)s] %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
warnings.simplefilter("ignore")

# pylint: disable=invalid-name

# Paths
THIS_DIR = os.path.dirname(__file__)
QUERY = os.path.join(THIS_DIR, "query.sql")

# Default Args
MAX_PLYRS_P_CLUB = 5
DROPOUT = 0.0
N_TIMES = 1
N_TRIALS = 100
TIMEOUT = None

# Columns
TARGET_COL = "total_points"
ID_COL = "player_id"
SEASON_COL = "season"
ROUND_COL = "round"
TIME_COL = "timestamp"
CLUB_COL = "club"
PRICE_COL = "price"
POSITION_COL = "position"
POSITION_ID_COL = "position_id"

# Base Model
MODEL = lgbm.LGBMRanker(n_estimators=100, n_jobs=-1)


def fit(model, X, y, q):
    """Fit model."""
    model.fit(
        X,
        y.clip(0, 30).round(0).astype("int32"),
        group=q,
        categorical_feature="auto",
    )


def score(model, X, y, q):
    """Make predictions."""
    start_idx = q.cumsum().shift().fillna(0).astype("int32")
    end_idx = q.cumsum()

    y_true_rnd = [y.iloc[s:e].values for s, e in zip(start_idx, end_idx)]
    y_test_rnd = [model.predict(X.iloc[s:e]) for s, e in zip(start_idx, end_idx)]

    scores = [
        ndcg_score(y_true.reshape(1, -1), y_test.reshape(1, -1))
        for y_true, y_test in zip(y_true_rnd, y_test_rnd)
    ]
    return np.mean(scores)


@dataclass
class Objective:
    """Optuna objective."""

    X_train: pd.DataFrame
    y_train: pd.Series
    q_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    q_test: pd.Series

    def __call__(self, trial: optuna.Trial):
        params = dict(
            boosting_type=trial.suggest_categorical(
                "boosting_type", ["gbdt", "dart", "goss"]
            ),
            num_leaves=trial.suggest_int("num_leaves", 2, 1024),
            max_depth=trial.suggest_int("max_depth", 2, 128),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e0, log=True),
            # subsample_for_bin=trial.suggest_int("subsample_for_bin", 1000, 200000),
            # min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 5),
            min_child_samples=trial.suggest_int("min_child_samples", 1, 16),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            # subsample_freq=trial.suggest_float("subsample_freq", 0.0, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1e0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 1e0, log=True),
        )
        MODEL.set_params(**params)

        cols = [
            col
            for col in self.X_train.columns
            if trial.suggest_categorical(f"column__{col}", [True, False])
        ]

        fit(MODEL, self.X_train[cols], self.y_train, self.q_train)
        return score(MODEL, self.X_test[cols], self.y_test, self.q_test)


class DecimalEncoder(json.JSONEncoder):
    """Decimal encoder for JSON."""

    def default(self, o):
        """Encode Decimal."""
        if isinstance(o, Decimal):
            return float(o)
        return json.JSONEncoder.default(self, o)


def draft(data, max_players_per_club, dropout):
    """Simulate a Cartola FC season."""
    scheme = {
        "goalkeeper": 1,
        "fullback": 2,
        "defender": 2,
        "midfielder": 3,
        "forward": 3,
        "coach": 0,
    }
    body = {
        "game": "custom",
        "scheme": scheme,
        "price": 140,
        "max_players_per_club": max_players_per_club,
        "bench": False,
        "dropout": dropout,
        "players": data.to_dict(orient="records"),
    }
    res = requests.post(
        os.getenv("DRAFT_URL"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.getenv("DRAFT_KEY"),
        },
        data=json.dumps(body, cls=DecimalEncoder),
        timeout=30,
    )
    if res.status_code >= 300:
        raise ValueError(res.text)

    content = json.loads(res.content.decode())
    if content["status"] == "FAILED":
        raise ValueError(content["cause"])

    return sum(p["actual_points"] for p in json.loads(content["output"])["players"])


def main(n_trials, timeout, max_plyrs_per_club, dropout, n_times):
    """Main exec."""
    # pylint: disable=too-many-locals

    # Read data
    with open(QUERY, encoding="utf-8") as file:
        data = pd.read_gbq(file.read())

    # These are the columns that will only be used for the drafting simulation.
    # They must be removed from training data.
    draft_cols = [TARGET_COL, SEASON_COL, ROUND_COL, CLUB_COL, POSITION_COL, PRICE_COL]

    # This is a ranking model. It needs to know the groups in order to learn.
    # Groups are expected to be a list of group sizes.
    # This implies that the dataset must be ordered as the groups are together.
    data = data.sort_values(["season", "round"])
    groups = data["round"] + 38 * (data["season"] - data["season"].min())  # Group ID
    groups = groups.astype("int32")

    # Data points where data will be splitted.
    split = [
        groups.max() - (4 * 38),  # Traning start.
        groups.max() - (2 * 38),  # Training end and validation start.
        groups.max() - (1 * 38),  # Validation end and testing start.
        groups.max() - (0 * 38),  # Testing end.
    ]

    # Split data
    train_index = data[(groups >= split[0]) & (groups < split[1])].index
    valid_index = data[(groups >= split[1]) & (groups < split[2])].index
    test_index = data[(groups >= split[2]) & (groups < split[3])].index

    # Split features and target.
    X = (
        data.drop(columns=draft_cols)
        .astype("float32")
        .astype({POSITION_ID_COL: "int32"})
    )
    y = data[TARGET_COL]

    # Split each dataset into features and targets.
    # Notice that sorting is False on the groups, because it must keep original order.

    X_train = X.loc[train_index]
    y_train = y.loc[train_index]
    q_train = groups.loc[train_index].value_counts(sort=False)

    X_valid = X.loc[valid_index]
    y_valid = y.loc[valid_index]
    q_valid = groups.loc[valid_index].value_counts(sort=False)

    X_test = X.loc[test_index]
    y_test = y.loc[test_index]
    q_test = groups.loc[test_index].value_counts(sort=False)

    # Hyperparams tuning with optuna.
    obj = Objective(X_train, y_train, q_train, X_valid, y_valid, q_valid)
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, timeout=timeout)
    valid_score = study.best_value

    logging.info("validation scoring: %.3f", valid_score)

    # Apply best params and columns and retrain to score agains test set.
    best_params = {
        key: val
        for key, val in study.best_params.items()
        if not key.startswith("column__")
    }
    best_cols = {
        key.removesuffix("column__"): val
        for key, val in study.best_params.items()
        if key.startswith("column__")
    }
    MODEL.set_params(**best_params)
    fit(
        MODEL,
        pd.concat((X_train, X_valid))[best_cols],
        pd.concat((y_train, y_valid)),
        pd.concat((q_train, q_valid)),
    )
    test_score = score(MODEL, X_test[best_cols], y_test, q_test)
    logging.info("testing scoring: %.3f", test_score)

    # Drafting Simulation.
    # Test the model in a real-life drafting scenario on the test set.
    # It will iterate round by round from the test set
    # simulating how would be the scoring of a team drafted using this model.

    history = []
    for idx, rnd in data.loc[test_index].groupby([SEASON_COL, ROUND_COL]):

        # Data fot this specifc round.
        rnd[f"{TARGET_COL}_pred"] = MODEL.predict(X.loc[rnd.index][best_cols])
        mapping = {
            ID_COL: "id",
            CLUB_COL: "club",
            POSITION_COL: "position",
            PRICE_COL: "price",
            TARGET_COL: "actual_points",
            "total_points_pred": "points",
        }
        rnd = rnd.reset_index().rename(mapping, axis=1)[list(mapping.values())]

        # Take the exponential from the prediction to make it easier to draft
        rnd["points"] = np.exp(rnd["points"])

        # Make several concurrent calls agains the Draft API
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(n_times):
                futures.append(executor.submit(draft, rnd, max_plyrs_per_club, dropout))
            draft_scores = pd.Series(
                [fut.result() for fut in concurrent.futures.as_completed(futures)],
                index=[f"run{i}" for i in range(len(futures))],
                name=idx,
            )

        logging.info("%04d-%02d: %.1f", idx[0], idx[1], draft_scores.mean())

        # Test again, but for a perfect scenario. Instead of using predictions
        # use the actual points to see how a perfect model would be.
        rnd["points"] = rnd["actual_points"]
        draft_scores["max"] = draft(rnd, 5, 0.0)

        # Evaluate how much points the most frequent team would score.
        # draft_scores["mode"] = ... TODO

        history.append(draft_scores)

    history = pd.concat(history, axis=1).transpose()

    overall_mean_score = history.drop(columns=["max"]).mean().mean()
    logging.info("Overall Mean Draft: %.2f", overall_mean_score)

    overall_mean_norm_score = (
        history.drop(columns=["max"]).divide(history["max"], axis=0).mean().mean()
    )
    logging.info("Overall Mean Normalized Draft: %.2f", overall_mean_norm_score)

    # Retrain model on all datasets and export it.
    fit(
        MODEL,
        pd.concat((X_train, X_valid, X_test))[best_cols],
        pd.concat((y_train, y_valid, y_test)),
        pd.concat((q_train, q_valid, q_test)),
    )
    MODEL.booster_.save_model(os.path.join(THIS_DIR, "model.txt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-players-per-club", default=MAX_PLYRS_P_CLUB, type=int)
    parser.add_argument("--dropout", default=DROPOUT, type=float)
    parser.add_argument("--n-times", default=N_TIMES, type=int)
    parser.add_argument("--n-trials", default=N_TRIALS, type=int)
    parser.add_argument("--timeout", default=TIMEOUT, type=int)
    parser.add_argument("-m", "--message", default="")
    args = parser.parse_args()

    main(
        n_trials=args.n_trials,
        timeout=args.timeout,
        max_plyrs_per_club=args.max_players_per_club,
        dropout=args.dropout,
        n_times=args.n_times,
    )
