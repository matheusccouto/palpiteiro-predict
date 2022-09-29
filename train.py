"""Run experiment."""

import argparse
import concurrent.futures
import json
import logging
import os
from dataclasses import dataclass
from decimal import Decimal

import lightgbm as lgbm
import numpy as np
import optuna
import pandas as pd
import requests
from sklearn.metrics import ndcg_score

# Paths
THIS_DIR = os.path.dirname(__file__)
QUERY = os.path.join(THIS_DIR, "query.sql")

# Default Args
MAX_PLYRS_P_CLUB = 5
DROPOUT = 0.333
N_TIMES = 5
N_TRIALS = 1
TIMEOUT = None

# Columns
TARGET_COL = "total_points"
ID_COL = "player_id"
SEASON_COL = "season"
ROUND_COL = "round"
TIME_COL = "timestamp"
CLUB_COL = "club"
PRICE_COL = "price"
POSITION_COL = "position_id"

# Base Model
MODEL = lgbm.LGBMRanker()


def fit(model, X, y, q):
    """Fit model."""
    model.fit(
        X,
        y.clip(0, 30).round(0).astype("int32"),
        group=q,
        categorical_feature=[POSITION_COL],
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
        fit(MODEL, self.X_train, self.y_train, self.q_train)
        return score(MODEL, self.X_test, self.y_test, self.q_test)


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


def main(n_trials, timeout, max_players_per_club, dropout, n_times):
    """Main exec."""
    with open(QUERY, encoding="utf-8") as file:
        data = pd.read_gbq(file.read())

    to_drop = [TARGET_COL, SEASON_COL, ROUND_COL, CLUB_COL, PRICE_COL]
    groups = data["round"] + 38 * (data["season"] - data["season"].min())
    groups = groups.astype("int32")

    split = [
        groups.max() - (3 * 38),
        groups.max() - (2 * 38),
        groups.max() - (1 * 38),
        groups.max() - (0 * 38),
    ]

    train_index = data[(groups >= split[0]) & (groups < split[1])].index
    valid_index = data[(groups >= split[1]) & (groups < split[2])].index
    test_index = data[(groups >= split[2]) & (groups < split[3])].index

    X = data.drop(columns=to_drop).astype("float32").astype({POSITION_COL: "int32"})
    y = data[TARGET_COL]

    X_train = X.loc[train_index]
    y_train = y.loc[train_index]
    q_train = groups.loc[train_index].value_counts(sort=False)

    X_valid = X.loc[valid_index]
    y_valid = y.loc[valid_index]
    q_valid = groups.loc[valid_index].value_counts(sort=False)

    X_test = X.loc[test_index]
    y_test = y.loc[test_index]
    q_test = groups.loc[test_index].value_counts(sort=False)

    obj = Objective(X_train, y_train, q_train, X_valid, y_valid, q_valid)
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, timeout=timeout)
    valid_score = study.best_value
    print(f"{valid_score=}")

    MODEL.set_params(**study.best_params)
    fit(
        MODEL,
        pd.concat((X_train, X_valid)),
        pd.concat((y_train, y_valid)),
        pd.concat((q_train, q_valid)),
    )
    test_score = score(MODEL, X_test, y_test, q_test)
    print(f"{test_score=}")

    mean_points = []
    max_points = []
    draft_history = []
    for i, rnd in data.loc[test_index].groupby([SEASON_COL, ROUND_COL]):

        rnd[f"{TARGET_COL}_pred"] = MODEL.predict(X.loc[rnd.index])

        mapping = {
            ID_COL: "id",
            CLUB_COL: "club",
            POSITION_COL: "position",
            PRICE_COL: "price",
            TARGET_COL: "actual_points",
            "total_points_pred": "points",
        }
        rnd = rnd.reset_index().rename(mapping, axis=1)[list(mapping.values())]

        def run_draft():
            """Wrapper to handle error on draft()."""
            try:
                return draft(data, max_players_per_club, dropout)
            except ValueError as err:
                if "There are not enough players to form a line-up" in str(err):
                    return draft(data, max_players_per_club, dropout)
                raise err

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(n_times):
                futures.append(executor.submit(run_draft))
            draft_scores = pd.Series(
                [fut.result() for fut in concurrent.futures.as_completed(futures)],
                index=[f"run{i}" for i in range(len(futures))],
                name=i,
            )

        data["points"] = data["actual_points"]
        the_best = draft(data, 5, 0.0)
        # the_popular = ...

        mean_draft = np.mean(draft_scores)
        mean_draft_norm = mean_draft / the_best

        logging.info("Mean Draft: %.1f (%.2f)", mean_draft, mean_draft_norm)
        logging.info("%s", 40 * "-")

        draft_scores["max"] = the_best
        draft_history.append(draft_scores)
        mean_points.append(mean_draft_norm)

    overall_mean_points = np.mean(mean_points)

    logging.info("Overall Mean Draft: %.2f", overall_mean_points)

    fit(
        MODEL,
        pd.concat((X_train, X_valid, X_test)),
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
        max_players_per_club=args.max_players_per_club,
        dropout=args.dropout,
        n_times=args.n_times,
    )
