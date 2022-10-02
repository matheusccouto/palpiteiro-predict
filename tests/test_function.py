"""Unit tests for google cloud function."""

import json
import os
from unittest.mock import Mock

import pandas as pd
import pytest

import main

THIS_DIR = os.path.dirname(__file__)
SAMPLE_PATH = os.path.join(THIS_DIR, "sample.csv")


@pytest.fixture(name="req")
def request_fixture():
    """Sample data for testing."""
    body = {"calls": pd.read_csv(SAMPLE_PATH)["column_name"].apply(json.loads).tolist()}
    return Mock(get_json=Mock(return_value=body))


def test_count(req):
    """Test function handler."""
    assert len(main.handler(req)["replies"]) == 10
