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
import optuna.logging
import optuna.visualization
import pandas as pd
import wandb


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname).1s %(asctime)s] %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
warnings.simplefilter("ignore")
optuna.logging.set_verbosity(optuna.logging.WARN)

# pylint: disable=invalid-name

# Paths
THIS_DIR = os.path.dirname(__file__)
QUERY = os.path.join(THIS_DIR, "query.sql")

# Default Args
MAX_PLYRS_P_CLUB = 5
DROPOUT = 0.4
DROPOUT_TYPE = "position_club"
N_TIMES = 50
N_TRIALS = 10
TIMEOUT = None

# Columns
TARGET_COL = "total_points"
POINTS_COL = "total_points"
ID_COL = "player_id"
SEASON_COL = "season"
ROUND_COL = "round"
TIME_COL = "timestamp"
CLUB_COL = "club"
PRICE_COL = "price"
POSITION_COL = "position"
POSITION_ID_COL = "position_id"

# Scheme to use on drafting
SCHEME = {
    "goalkeeper": 1,
    "fullback": 2,
    "defender": 2,
    "midfielder": 3,
    "forward": 3,
    "coach": 0,
}

# These are the columns that will only be used for the drafting simulation.
# They must be removed from training data.
DRAFT_COLS = [
    ID_COL,
    # POINTS_COL,
    # TARGET_COL,
    SEASON_COL,
    ROUND_COL,
    CLUB_COL,
    POSITION_COL,
    PRICE_COL,
]

# Base Model
K = 20
MODEL = lgbm.LGBMRegressor(
    n_estimators=100,
    n_jobs=-1,
)


def logging_callback(study, frozen_trial):  # pylint: disable=unused-argument
    """Optuna logging callback."""
    logging.info(
        "Trial %04d finished with value: %.3f ",
        frozen_trial.number,
        frozen_trial.value,
    )


def draft(data, max_players_per_club, dropout, dropout_type):
    """Simulate a Cartola express  scoring."""
    line_up = {pos: [] for pos in SCHEME}
    clubs = {club: 0 for club in data[CLUB_COL].unique()}
    count = 0

    if dropout:

        if "all" in dropout_type:
            data = data.sample(frac=1 - dropout)

        if "club" in dropout_type:
            data = data[
                data["club"].isin(
                    data[CLUB_COL].drop_duplicates().sample(frac=1 - dropout)
                )
            ]

        if "position" in dropout_type:
            data = data.groupby(POSITION_COL, group_keys=False).apply(
                lambda x: x.sample(frac=1 - dropout)
            )

    for _, player in data.sort_values("points", ascending=False).iterrows():

        if len(line_up[player[POSITION_COL]]) >= SCHEME[player[POSITION_COL]]:
            continue

        if clubs[player[CLUB_COL]] >= max_players_per_club:
            continue

        line_up[player[POSITION_COL]].append(player["actual_points"])
        count += 1
        clubs[player[CLUB_COL]] += 1

        if count > 11:
            break

    return sum(player for position in line_up.values() for player in position)


def fit(model, X, y):
    """Fit model."""
    if POSITION_ID_COL in X.columns:
        cats = [POSITION_ID_COL]
    else:
        cats = "auto"

    model.fit(
        X.astype("float32"),
        y.astype("float32"),
        # y.clip(0, 30).round(0).astype("int32"),
        # group=q,
        categorical_feature=cats,
    )


# def score(model, X, y, q, k, cols):
#     """Make predictions."""
#     start_idx = q.cumsum().shift().fillna(0).astype("int32")
#     end_idx = q.cumsum()

#     y_true_rnd = [y.iloc[s:e].values.tolist() for s, e in zip(start_idx, end_idx)]
#     y_test_rnd = [
#         model.predict(X[cols].iloc[s:e].astype("float32")).tolist()
#         for s, e in zip(start_idx, end_idx)
#     ]

#     scores = [
#         ndcg_score([y_true], [y_test], k=k)
#         for y_true, y_test in zip(y_true_rnd, y_test_rnd)
#     ]
#     return np.mean(scores)


def score(
    model,
    X,
    y,
    cols,
    max_players_per_club,
    dropout,
    dropout_type,
    populars,
    bests,
    n_times,
):
    """Make predictions."""
    scores = []
    for idx, rnd in X.groupby([SEASON_COL, ROUND_COL]):

        pop = populars.query(f"season=={idx[0]} and round=={idx[1]}")["points"].iloc[0]
        best = bests.query(f"season=={idx[0]} and round=={idx[1]}")["points"].iloc[0]

        rnd["points"] = model.predict(rnd[cols].astype("float32"))
        rnd["actual_points"] = y
        for _ in range(n_times):
            points = draft(rnd, max_players_per_club, dropout, dropout_type)
            scores.append((points - pop) / (best - pop))

    return np.mean(scores)


@dataclass
class Objective:
    """Optuna objective."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    max_players_per_club: int
    dropout: float
    dropout_type: str
    populars: pd.DataFrame
    bests: pd.DataFrame
    n_times: int

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

        cols = [
            col
            for col in self.X_train.drop(columns=DRAFT_COLS).columns
            if trial.suggest_categorical(f"col__{col}", [True, False])
        ]

        fit(MODEL, self.X_train[cols], self.y_train)
        return score(
            model=MODEL,
            X=self.X_test,
            y=self.y_test,
            cols=cols,
            max_players_per_club=self.max_players_per_club,
            dropout=self.dropout,
            dropout_type=self.dropout_type,
            populars=self.populars,
            bests=self.bests,
            n_times=self.n_times,
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
    k,
    tags,
    notes,
):
    """Main exec."""
    # pylint: disable=too-many-locals,too-many-statements,too-many-arguments
    logging.info("Number of Trials: %s", n_trials)
    logging.info("Timeout: %s", timeout)
    logging.info("Max Players per Club: %s", max_plyrs_per_club)
    logging.info("Dropout: %s", dropout)
    logging.info("Dropout Type: %s", dropout_type)
    logging.info("Number of Times: %s", n_times)
    logging.info("NDCG K: %s", k)
    logging.info("Notes: %s", notes)
    logging.info("Tags: %s", tags)

    wandb.init(project="palpiteiro-predict", tags=tags, notes=notes)
    wandb.log(
        {
            "n_trials": n_trials,
            "timeout": timeout,
            "max_players_per_club": max_plyrs_per_club,
            "dropout": dropout,
            "dropout_type": dropout_type,
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

    # This is a ranking model. It needs to know the groups in order to learn.
    # Groups are expected to be a list of group sizes.
    # This implies that the dataset must be ordered as the groups are together.
    data = data.sort_values(["season", "round"])
    groups = data["round"] + 38 * (data["season"] - data["season"].min())  # Group ID

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
    X = data.drop(columns=TARGET_COL)
    y = data[TARGET_COL]

    # Split each dataset into features and targets.
    # Notice that sorting is False on the groups, because it must keep original order.

    X_train = X.loc[train_index]
    y_train = y.loc[train_index]
    q_train = groups.loc[train_index].value_counts(sort=False)

    X_valid = X.loc[valid_index]
    y_valid = y.loc[valid_index]

    X_test = X.loc[test_index]
    y_test = y.loc[test_index]

    # Read reference data
    populars = pd.read_gbq("SELECT * FROM express.fct_popular_points")
    bests = pd.read_gbq("SELECT * FROM express.fct_best_expected_points")

    # Hyperparams tuning with optuna.
    obj = Objective(
        X_train,
        y_train,
        X_valid,
        y_valid,
        max_players_per_club=max_plyrs_per_club,
        dropout=dropout,
        dropout_type=dropout_type,
        populars=populars,
        bests=bests,
        n_times=n_times,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(
        obj,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[logging_callback],
    )
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
    best_cols = {
        key: val for key, val in study.best_params.items() if key.startswith("col__")
    }
    wandb.log(
        {
            "columns": pd.DataFrame(
                {col.replace("col__", ""): val for col, val in best_cols.items()},
                index=[0],
            )
        }
    )
    selected_cols = [col.replace("col__", "") for col, val in best_cols.items() if val]

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
                study=study, params=list(best_cols.keys())
            ),
            "params_importance": optuna.visualization.plot_param_importances(
                study=study, params=list(best_params.keys())
            ),
            "columns_importance": optuna.visualization.plot_param_importances(
                study=study, params=list(best_cols.keys())
            ),
        }
    )
    MODEL.set_params(**best_params)
    fit(
        MODEL,
        pd.concat((X_train, X_valid))[selected_cols],
        pd.concat((y_train, y_valid)),
    )
    test_score = score(
        MODEL,
        X_test,
        y_test,
        cols=selected_cols,
        max_players_per_club=max_plyrs_per_club,
        dropout=dropout,
        dropout_type=dropout_type,
        populars=populars,
        bests=bests,
        n_times=n_times,
    )
    wandb.log({"testing_scoring": test_score})
    logging.info("testing scoring: %.3f", test_score)

    # Drafting Simulation.
    # Test the model in a real-life drafting scenario on the test set.
    # It will iterate round by round from the test set
    # simulating how would be the scoring of a team drafted using this model.

    prizes = pd.read_gbq("SELECT * FROM express.fct_prize")

    history = []
    for idx, rnd in data.loc[test_index].groupby([SEASON_COL, ROUND_COL]):

        if not (
            idx[0] in prizes["season"].unique() and idx[1] in prizes["round"].unique()
        ):
            continue

        # Data for this specifc round.
        rnd[f"{POINTS_COL}_pred"] = MODEL.predict(
            X.loc[rnd.index][selected_cols].astype("float32")
        )
        mapping = {
            ID_COL: "id",
            CLUB_COL: "club",
            POSITION_COL: "position",
            PRICE_COL: "price",
            POINTS_COL: "actual_points",
            "total_points_pred": "points",
        }
        rnd = rnd.reset_index().rename(mapping, axis=1)[list(mapping.values())]

        # Take the exponential from the prediction to make it easier to draft
        rnd["points"] = np.exp(rnd["points"])

        # Make several concurrent calls agains the Draft API
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for _ in range(n_times):
                futures.append(
                    executor.submit(
                        draft, rnd, max_plyrs_per_club, dropout, dropout_type
                    )
                )
            draft_scores = pd.Series(
                [fut.result() for fut in concurrent.futures.as_completed(futures)],
                index=[f"run{i}" for i in range(len(futures))],
                name=f"{idx[0]}-{idx[1]:02d}",
            )

        mean_points = draft_scores.mean()

        max_points = bests.query("season==@idx[0] and round==@idx[1]")["points"].iloc[0]
        draft_scores["max"] = max_points

        # Get how many points a team with the most popular players would have scored.
        mode_points = populars.query("season==@idx[0] and round==@idx[1]")[
            "points"
        ].iloc[0]
        draft_scores["mode"] = mode_points

        prizes_rnd = prizes.query("season==@idx[0] and round==@idx[1]")
        prizes_rnd = prizes_rnd.sort_values("rank")
        if prizes_rnd.shape[0] > 0:

            def get_prize(pnt):
                try:
                    # pylint: disable=cell-var-from-loop
                    return prizes_rnd.query(f"points<={pnt}")["prizes"].iloc[0]
                except IndexError:
                    return 0

            real_prizes = [
                get_prize(pnt)
                # pylint: disable=not-an-iterable
                for pnt in draft_scores.drop(["max", "mode"])
            ]

        else:
            real_prizes = [np.nan]
        draft_scores["prize"] = pd.Series(real_prizes)

        log_str = (
            "%04d-%02d: %03.1f  " "Mode: %03.1f  " "Max %03.1f  " "Prizes %03.1f  "
        )
        logging.info(
            log_str,
            idx[0],
            idx[1],
            mean_points,
            mode_points,
            max_points,
            sum(real_prizes),
        )

        history.append(draft_scores)

    history = pd.concat(history, axis=1).transpose()

    overall_norm_score = (
        history.drop(columns=["max", "mode", "prize"])
        .subtract(history["mode"], axis=0)
        .divide(history["max"].subtract(history["mode"]), axis=0)
        .mean()
        .mean()
    )
    wandb.log({"Overall Normalized Score": overall_norm_score})
    logging.info("Overall Normalized Score: %.2f", overall_norm_score)

    investment = 10 * sum(prz.dropna().shape[0] for prz in history["prize"])
    overall_roi = (
        history["prize"].apply(lambda x: x.sum(skipna=False)).sum() - investment
    ) / investment
    wandb.log({"Overall ROI": overall_roi})
    logging.info("Overall ROI: %.2f", overall_roi)

    # Retrain model on all datasets and export it.
    fit(
        MODEL,
        pd.concat((X_train, X_valid, X_test))[selected_cols],
        pd.concat((y_train, y_valid, y_test)),
    )
    MODEL.booster_.save_model(os.path.join(THIS_DIR, "model.txt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-players-per-club", default=MAX_PLYRS_P_CLUB, type=int)
    parser.add_argument("--dropout", default=DROPOUT, type=float)
    parser.add_argument("--dropout-type", default=DROPOUT_TYPE, type=str)
    parser.add_argument("--n-times", default=N_TIMES, type=int)
    parser.add_argument("--n-trials", default=N_TRIALS, type=int)
    parser.add_argument("--timeout", default=TIMEOUT, type=int)
    parser.add_argument("--k", default=K, type=int)
    parser.add_argument("-m", "--message", default="")
    parser.add_argument("-t", "--tags", action="append", nargs="+", default=[])
    args = parser.parse_args()

    main(
        n_trials=args.n_trials,
        timeout=args.timeout,
        max_plyrs_per_club=args.max_players_per_club,
        dropout=args.dropout,
        dropout_type=args.dropout_type,
        n_times=args.n_times,
        k=args.k,
        tags=[t for group in args.tags for t in group],
        notes=args.message,
    )
