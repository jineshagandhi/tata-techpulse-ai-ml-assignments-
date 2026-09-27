import json
import os

import joblib
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, build_features
from src.train import MIN_TEST_F1, MIN_TEST_RECALL, MODEL_DIR


@pytest.fixture(scope="module")
def exported():
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    assert os.path.exists(model_path), "Run `python -m src.train` before the tests"

    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    return joblib.load(model_path), metadata


def reading(**values):
    row = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 20,
    }
    row.update(values)
    return build_features(pd.DataFrame([row]))


def test_metadata_matches_training_features(exported):
    _, metadata = exported
    assert metadata["features"] == FEATURE_COLUMNS


def test_exported_model_meets_quality_gate(exported):
    _, metadata = exported
    assert metadata["metrics"]["test_f1"] >= MIN_TEST_F1
    assert metadata["metrics"]["test_recall"] >= MIN_TEST_RECALL


def test_normal_operation_is_not_a_failure(exported):
    model, _ = exported
    assert model.predict(reading())[0] == 0


def test_power_overload_is_a_failure(exported):
    # About 11 kW, well above the 9 kW power failure limit in the dataset
    model, _ = exported
    assert model.predict(reading(**{"Rotational speed [rpm]": 1300, "Torque [Nm]": 80.0}))[0] == 1


def test_worn_tool_under_high_torque_is_riskier(exported):
    model, _ = exported
    new_tool = model.predict_proba(reading(**{"Torque [Nm]": 60.0, "Tool wear [min]": 10}))[0, 1]
    worn_tool = model.predict_proba(reading(**{"Torque [Nm]": 60.0, "Tool wear [min]": 230}))[0, 1]
    assert worn_tool > new_tool
