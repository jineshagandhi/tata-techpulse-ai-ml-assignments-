"""Feature engineering shared by training and the prediction API."""

import numpy as np
import pandas as pd

TARGET = "Machine failure"

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

PRODUCT_TYPES = ["L", "M", "H"]

FEATURE_COLUMNS = SENSOR_COLUMNS + [
    "Type_L",
    "Type_M",
    "Power [W]",
    "Temperature difference [K]",
    "Strain [min Nm]",
]


def build_features(df):
    """Turn raw sensor readings into the columns the model is trained on.

    Type is encoded by hand instead of with get_dummies, so a single API
    request produces exactly the same columns as the full training set.
    """
    features = df[SENSOR_COLUMNS].astype(float).copy()

    features["Type_L"] = (df["Type"] == "L").astype(int)
    features["Type_M"] = (df["Type"] == "M").astype(int)

    # Power and temperature difference match the power failure and heat
    # dissipation failure modes, strain matches the overstrain failure mode
    features["Power [W]"] = (
        df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
    )
    features["Temperature difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    features["Strain [min Nm]"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    return features[FEATURE_COLUMNS]


def load_dataset(path):
    df = pd.read_csv(path)
    return build_features(df), df[TARGET]
