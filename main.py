"""Google Cloud Functions to predict player points."""

import json
import logging
import os

import lightgbm as lgbm
import pandas as pd
from google.cloud import storage

BUCKET_NAME = os.getenv("BUCKET_NAME")
MODEL_PATH = "points/model.txt"

# Load model outside the handler for caching in between calls.
client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.blob(MODEL_PATH)
with blob.open(encoding="utf-8") as file:
    model = lgbm.Booster(model_str=file.read())

# Retrieve feature names.
features = model.feature_name()


def handler(request):
    """HTTP Cloud Function handler."""
    body = request.get_json()
    print(
        json.dumps(
            dict(
                severity="INFO",
                message=body
            )
        )
    )
    data = pd.DataFrame.from_records(body["calls"])[features].astype("float32")
    pred = model.predict(data, validate_features=True)
    return {"replies": pred}
