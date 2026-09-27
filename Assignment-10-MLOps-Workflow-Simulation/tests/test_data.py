import pandas as pd

from src.features import FEATURE_COLUMNS, build_features
from src.validate_data import DATA_PATH, validate


def load_raw():
    return pd.read_csv(DATA_PATH)


def test_real_dataset_passes_validation():
    assert validate(load_raw()) == []


def test_validation_catches_bad_sensor_values():
    df = load_raw()
    df.loc[0, "Torque [Nm]"] = -5
    df.loc[1, "Air temperature [K]"] = None

    errors = validate(df)

    assert any("Torque" in error for error in errors)
    assert any("missing values" in error for error in errors)


def test_validation_catches_missing_column():
    df = load_raw().drop(columns=["Tool wear [min]"])

    assert validate(df) == ["Missing columns: ['Tool wear [min]']"]


def test_single_row_gives_same_features_as_full_dataset():
    df = load_raw()

    full = build_features(df)
    single = build_features(df.iloc[[42]])

    assert list(single.columns) == FEATURE_COLUMNS
    pd.testing.assert_frame_equal(single, full.iloc[[42]])


def test_engineered_features():
    row = pd.DataFrame([{
        "Type": "H",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }])

    features = build_features(row).iloc[0]

    assert features["Type_L"] == 0 and features["Type_M"] == 0
    assert round(features["Power [W]"]) == 6283
    assert features["Temperature difference [K]"] == 10.0
    assert features["Strain [min Nm]"] == 4000.0
