"""Unit tests for google cloud function."""

import json
import os
from unittest.mock import Mock

import pandas as pd
import pytest

import main

THIS_DIR = os.path.dirname(__file__)
SAMPLE_PATH = os.path.join(THIS_DIR, "sample.json")


@pytest.fixture(name="req")
def request_fixture():
    """Sample data for testing."""
    with open(SAMPLE_PATH, encoding="utf-8") as file:
        return Mock(get_json=Mock(return_value=json.load(file)))


def test_count(req):
    """Test function handler."""
    assert len(main.handler(req)["replies"]) == 7
