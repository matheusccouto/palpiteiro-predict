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
from typing import List

import lightgbm as lgbm
import numpy as np
import optuna
import optuna.logging
import optuna.visualization
import pandas as pd
import requests
import wandb


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname).1s %(asctime)s] %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
warnings.simplefilter("ignore")
# optuna.logging.set_verbosity(optuna.logging.WARN)

# pylint: disable=invalid-name

# Paths
THIS_DIR = os.path.dirname(__file__)
QUERY = os.path.join(THIS_DIR, "query.sql")

# Default Args
MAX_PLYRS_P_CLUB = 5
DROPOUT = 0.0
DROPOUT_TYPE = "all"
N_TIMES = 1
N_TRIALS = 10
TIMEOUT = None

# Columns naming.
TARGET_COL = "total_points"
ID_COL = "player_id"
SEASON_COL = "season"
ROUND_COL = "round"
TIME_COL = "timestamp"
CLUB_COL = "club"
PRICE_COL = "price"
POSITION_COL = "position"

# Columns that should not be used on model training.
COLS_TO_NOT_TRAIN_ON = [
    ID_COL,
    TARGET_COL,
    SEASON_COL,
    ROUND_COL,
    CLUB_COL,
    PRICE_COL,
    POSITION_COL,
]

# Base Model
MODEL = lgbm.LGBMRegressor(
    n_estimators=100,
    n_jobs=-1,
)


# TODO do a greedy draft for tuning only
def draft(data, max_players_per_club, dropout, dropout_type=None):
    """Simulate a Cartola FC season."""
    scheme = {
        "goalkeeper": 1,
        "fullback": 2,
        "defender": 2,
        "midfielder": 3,
        "forward": 3,
        "coach": 0,
    }
    mapping = {
        ID_COL: "id",
        CLUB_COL: "club",
        POSITION_COL: "position",
        PRICE_COL: "price",
        TARGET_COL: "actual_points",
        "points": "points",
    }
    data = data.rename(mapping, axis=1)[list(mapping.values())].to_dict(
        orient="records"
    )
    body = {
        "game": "custom",
        "scheme": scheme,
        "price": 140,
        "max_players_per_club": max_players_per_club,
        "bench": False,
        "dropout": dropout,
        "dropout_type": dropout_type,
        "players": data,
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


def fit(model, data, features):
    """Fit model."""
    model.fit(data[features].astype("float32"), data[TARGET_COL])


def score(model, data, features, max_players_per_club, dropout, dropout_type, n_times):
    """Make predictions and evaluate how many points would have been scored."""
    with concurrent.futures.ThreadPoolExecutor() as executor:

        futures = []
        for _, rnd in data.groupby([SEASON_COL, ROUND_COL]):
            rnd["actual_points"] = rnd[TARGET_COL]
            rnd["points"] = model.predict(rnd[features].astype("float32"))
            for _ in range(n_times):
                future = executor.submit(
                    draft,
                    data=rnd,
                    max_players_per_club=max_players_per_club,
                    dropout=dropout,
                    dropout_type=dropout_type,
                )
                futures.append(future)

        scores = [fut.result() for fut in concurrent.futures.as_completed(futures)]

    return np.mean(scores)


@dataclass
class Objective:
    """Optuna objective."""

    train: pd.DataFrame
    test: pd.DataFrame
    features: List[str]

    def __call__(self, trial: optuna.Trial):
        params = dict(
            boosting_type=trial.suggest_categorical(
                "boosting_type", ["gbdt", "dart", "goss"]
            ),
            num_leaves=trial.suggest_int("num_leaves", 2, 2048),
            max_depth=trial.suggest_int("max_depth", 16, 256),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            # subsample_for_bin=trial.suggest_int("subsample_for_bin", 1000, 200000),
            # min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 5),
            min_child_samples=trial.suggest_int("min_child_samples", 1, 64),
            subsample=trial.suggest_float("subsample", 0.333, 1.0),
            # subsample_freq=trial.suggest_float("subsample_freq", 0.0, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.333, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e0, log=True),
        )
        MODEL.set_params(**params)

        selected_features = [
            col
            for col in self.features
            if trial.suggest_categorical(f"col__{col}", [True, False])
        ]

        fit(MODEL, self.train, selected_features)
        return score(
            model=MODEL,
            data=self.test,
            features=selected_features,
            max_players_per_club=MAX_PLYRS_P_CLUB,
            dropout=DROPOUT,
            dropout_type=DROPOUT_TYPE,
            n_times=N_TIMES,
        )


class DecimalEncoder(json.JSONEncoder):
    """Decimal encoder for JSON."""

    def default(self, o):
        """Encode Decimal."""
        if isinstance(o, Decimal):
            return float(o)
        return json.JSONEncoder.default(self, o)


def main(
    n_trials,
    timeout,
    max_plyrs_per_club,
    dropout,
    dropout_type,
    n_times,
    tags,
    notes,
):
    """Main exec."""
    # pylint: disable=too-many-locals,too-many-statements,too-many-arguments
    logging.info("Number of Trials: %s", n_trials)
    logging.info("Timeout: %s", timeout)
    logging.info("Max Players per Club: %s", max_plyrs_per_club)
    logging.info("Dropout: %s", dropout)
    logging.info("Dropout type: %s", dropout)
    logging.info("Number of Times: %s", n_times)
    logging.info("Notes: %s", notes)
    logging.info("Tags: %s", tags)

    wandb.init(project="palpiteiro-predict", tags=tags, notes=notes)
    wandb.log(
        {
            "n_trials": n_trials,
            "timeout": timeout,
            "max_players_per_club": max_plyrs_per_club,
            "dropout": dropout,
            "n_times": n_times,
        }
    )

    # Read data
    with open(QUERY, encoding="utf-8") as file:
        file_content = file.read()

    data = pd.read_gbq(file_content).dropna(subset=[TARGET_COL])
    artifact = wandb.Artifact("query", type="query")
    artifact.add_file("query.sql", name="query.sql.txt")
    wandb.log_artifact(artifact)

    features = [col for col in data.columns if col not in COLS_TO_NOT_TRAIN_ON]

    rounds = [group for i, group in data.groupby([SEASON_COL, ROUND_COL], sort=True)]
    train = pd.concat(rounds[:-38])
    valid = pd.concat(rounds[-38:-19])
    test = pd.concat(rounds[-19:])

    # Hyperparams tuning with optuna.
    obj = Objective(train, valid, features, max_plyrs_per_club, dropout, dropout_type, n_times)
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, timeout=timeout)
    valid_score = study.best_value

    wandb.log({"validation_scoring": valid_score})
    logging.info("validation scoring: %.3f", valid_score)

    # Apply best params and columns and retrain to score agains test set.
    best_params = {
        key: val
        for key, val in study.best_params.items()
        if not key.startswith("col__")
    }
    wandb.log({"params": pd.DataFrame(best_params, index=[0])})
    best_features = {
        key: val for key, val in study.best_params.items() if key.startswith("col__")
    }
    wandb.log(
        {
            "features": pd.DataFrame(
                {col.replace("col__", ""): val for col, val in best_features.items()},
                index=[0],
            )
        }
    )
    selected_features = [
        col.replace("col__", "") for col, val in best_features.items() if val
    ]

    # Optuna plots
    wandb.log(
        {
            "optimization_history": optuna.visualization.plot_optimization_history(
                study=study
            ),
            "params_parallel_coordinate": optuna.visualization.plot_parallel_coordinate(
                study=study, params=list(best_params.keys())
            ),
            "columns_parallel_coordinate": optuna.visualization.plot_parallel_coordinate(
                study=study, params=list(best_features.keys())
            ),
            "params_importance": optuna.visualization.plot_param_importances(
                study=study, params=list(best_params.keys())
            ),
            "columns_importance": optuna.visualization.plot_param_importances(
                study=study, params=list(best_features.keys())
            ),
        }
    )
    MODEL.set_params(**best_params)
    train_valid = pd.concat((train, valid))
    fit(MODEL, train_valid, selected_features)
    test_score = score(
        model=MODEL,
        data=test,
        features=selected_features,
        max_players_per_club=max_plyrs_per_club,
        dropout=dropout,
        dropout_type=dropout_type,
        n_times=n_times,
    )
    wandb.log({"testing_scoring": test_score})
    logging.info("testing scoring: %.3f", test_score)

    # Retrain model on all datasets and export it.
    fit(MODEL, data, selected_features)
    MODEL.booster_.save_model(os.path.join(THIS_DIR, "model.txt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-players-per-club", default=MAX_PLYRS_P_CLUB, type=int)
    parser.add_argument("--dropout", default=DROPOUT, type=float)
    parser.add_argument("--n-times", default=N_TIMES, type=int)
    parser.add_argument("--n-trials", default=N_TRIALS, type=int)
    parser.add_argument("--timeout", default=TIMEOUT, type=int)
    parser.add_argument("-m", "--message", default="")
    parser.add_argument("-t", "--tags", action="append", nargs="+", default=[])
    args = parser.parse_args()

    main(
        n_trials=args.n_trials,
        timeout=args.timeout,
        max_plyrs_per_club=args.max_players_per_club,
        dropout=args.dropout,
        n_times=args.n_times,
        tags=[t for group in args.tags for t in group],
        notes=args.message,
    )
