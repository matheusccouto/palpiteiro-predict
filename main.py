"""Google Cloud Functions to predict player points."""

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
    model = lgbm.Booster(model_file=file.read())


def handler(request):
    """HTTP Cloud Function handler."""
    body = request.get_json()
    data = pd.DataFrame.from_records(body["calls"])
    pred = model.predict(data, validate_features=True)
    return {"replies": pred}
