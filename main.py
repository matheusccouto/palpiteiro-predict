"""Google Cloud Functions to predict player points."""

import json
import os

import joblib
from google.cloud import storage

BUCKET_NAME = os.environ["BUCKET_NAME"]
POSITIONS = ["goalkeeper", "defender", "fullback", "midfielder", "forward"]
MODEL_PATH = "points/{position}.joblib"
FEATURES_PATH = "points/{position}_features.json"

bucket = storage.Client().get_bucket(BUCKET_NAME)


def load_joblib(
    path,
):
    """Load model with joblib."""
    blob = bucket.blob(path)
    with blob.open(mode="rb") as file:
        return joblib.load(file)


def load_json(path):
    """Load json."""
    blob = bucket.blob(path)
    with blob.open(encoding="utf-8") as file:
        return json.load(file)


# Load models outside the handler for caching in between calls.
models = {p: load_joblib(MODEL_PATH.format(position=p)) for p in POSITIONS}
features = {p: load_json(FEATURES_PATH.format(position=p)) for p in POSITIONS}


def handler(request):
    """HTTP Cloud Function handler."""
    body = request.get_json()
    data = body["calls"]

    data = [dict(zip(NAMES, row)) for row in data]
    data = [
        (
            row["position"],
            [val for col, val in row.items() if col in features[row["position"]]],
        )
        for row in data
    ]

    pred = [models[pos].predict([row])[0] for pos, row in data]
    return {"replies": pred}