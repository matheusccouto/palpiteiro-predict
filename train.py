"""Train model."""
# %%

import argparse
import os

import lightgbm as lgbm
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Paths
THIS_DIR = os.path.dirname(__file__)
QUERY = os.path.join(THIS_DIR, "query.sql")

# Default Args
MAX_PLAYERS_PER_CLUB = 5
DROPOUT = 0.333
TIMES = 5
N_TRIALS = 100
TIMEOUT = None

# Columns
TARGET_COL = "total_points"
SEASON_COL = "season"
ROUND_COL = "round"
TIME_COL = "timestamp"
CLUB_COL = "club"
PRICE_COL = "price"


class Objective:
    """Optuna objective."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, trial: optuna.Trial):
        params = dict()
        results = lgbm.cv(
            params=params,
            train_set=self.dataset,
            folds=TimeSeriesSplit(n_splits=5, test_size=0.2),
        )
        return results["metric1-mean"]


def main(n_trials, timeout):
    """Main exec."""
    with open(QUERY, encoding="utf-8") as file:
        data = pd.read_gbq(file.read())

    to_drop = [TARGET_COL, SEASON_COL, ROUND_COL, CLUB_COL, PRICE_COL]
    groups = data["round"] + 38 * (data["season"] - data["season"].min())

    split = [
        groups.max() - (3 * 38),
        groups.max() - (2 * 38),
        groups.max() - (1 * 38),
        groups.max() - (0 * 38),
    ]

    train_index = data[(groups >= split[0]) & (groups < split[1])].index
    valid_index = data[(groups >= split[1]) & (groups < split[2])].index
    test_index = data[(groups >= split[2]) & (groups < split[3])].index

    X = data.drop(columns=to_drop)
    y = data[TARGET_COL]

    train = lgbm.Dataset(
        data=X.loc[train_index],
        label=y.loc[train_index],
        group=groups.loc[train_index].value_counts(sort=False),
    )
    valid = lgbm.Dataset(
        data=X.loc[valid_index],
        label=y.loc[valid_index],
        group=groups.loc[valid_index].value_counts(sort=False),
        reference=train,
    )
    test = lgbm.Dataset(
        data=X.loc[test_index],
        label=y.loc[test_index],
        group=groups.loc[test_index].value_counts(sort=False),
    )

    obj = Objective(train)
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, timeout=timeout)

    obj.booster.set_params(**study.best_params)

    with open(os.path.join(THIS_DIR, f"{pos}.joblib"), mode="wb") as file:
        joblib.dump(est, file)


def score(y_true, y_pred, groups, p=1.0):
    """Mean groups NDCG"""
    return r2_score(y_true, y_pred)
    # i = 0
    # scores = []
    # for g in groups.cumsum():
    #     scores.append(
    #         ndcg_score(
    #             y_true[i:g].values.reshape(1, -1),
    #             y_pred[i:g].values.reshape(1, -1),
    #             k=round(p * len(y_pred[i:g])),
    #         )
    #     )
    #     i = g
    # return np.mean(scores)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-players-per-club", default=MAX_PLAYERS_PER_CLUB, type=int
    )
    parser.add_argument("--dropout", default=DROPOUT, type=float)
    parser.add_argument("--times", default=TIMES, type=int)
    parser.add_argument("--n-trials", default=N_TRIALS, type=int)
    parser.add_argument("--timeout", default=TIMEOUT, type=int)
    parser.add_argument("-m", "--message", default="")
    args = parser.parse_args()

    main(n_trials=args.n_trials, timeout=args.timeout)
