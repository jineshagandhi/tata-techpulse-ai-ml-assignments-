"""Pipeline stage 1: check the raw data before any training happens.

Run with:  python -m src.validate_data
The script exits with code 1 if a check fails, which stops the CI pipeline.
"""

import sys

import pandas as pd

from src.features import PRODUCT_TYPES, SENSOR_COLUMNS, TARGET

DATA_PATH = "data/ai4i2020.csv"

# Physically sensible ranges for each sensor
SENSOR_RANGES = {
    "Air temperature [K]": (290, 310),
    "Process temperature [K]": (300, 320),
    "Rotational speed [rpm]": (1000, 3000),
    "Torque [Nm]": (0, 100),
    "Tool wear [min]": (0, 300),
}

MIN_ROWS = 5000


def validate(df):
    errors = []

    missing_columns = [
        column for column in SENSOR_COLUMNS + ["Type", TARGET]
        if column not in df.columns
    ]
    if missing_columns:
        return ["Missing columns: %s" % missing_columns]

    if len(df) < MIN_ROWS:
        errors.append("Only %d rows, expected at least %d" % (len(df), MIN_ROWS))

    null_count = int(df[SENSOR_COLUMNS + ["Type", TARGET]].isnull().sum().sum())
    if null_count:
        errors.append("%d missing values found" % null_count)

    for column, (low, high) in SENSOR_RANGES.items():
        outside = int(((df[column] < low) | (df[column] > high)).sum())
        if outside:
            errors.append("%s: %d values outside %s-%s" % (column, outside, low, high))

    unknown_types = set(df["Type"].unique()) - set(PRODUCT_TYPES)
    if unknown_types:
        errors.append("Unknown product types: %s" % sorted(unknown_types))

    if not set(df[TARGET].unique()) <= {0, 1}:
        errors.append("Target column must only contain 0 and 1")

    failure_rate = df[TARGET].mean()
    if not 0.005 <= failure_rate <= 0.20:
        errors.append("Failure rate %.3f is outside the expected range" % failure_rate)

    return errors


def main():
    df = pd.read_csv(DATA_PATH)
    errors = validate(df)

    print("DATA VALIDATION")
    print("Rows        :", len(df))
    print("Failure rate: %.2f percent" % (df[TARGET].mean() * 100))

    if errors:
        print("FAILED")
        for error in errors:
            print(" -", error)
        sys.exit(1)

    print("PASSED")


if __name__ == "__main__":
    main()
