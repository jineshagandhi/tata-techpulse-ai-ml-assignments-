"""Pipeline stage 3: serve the exported model as a REST API.

Run locally with:  uvicorn app.main:app --reload
Interactive docs:  http://127.0.0.1:8000/docs
"""

import json
import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import build_features

MODEL_DIR = os.environ.get("MODEL_DIR", "model")

model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts machine failure from sensor readings (AI4I 2020 dataset).",
    version=metadata["model_version"],
)


class SensorReading(BaseModel):
    product_type: Literal["L", "M", "H"] = Field(description="Product quality: L, M or H")
    air_temperature: float = Field(ge=250, le=350, description="Air temperature in K")
    process_temperature: float = Field(ge=250, le=350, description="Process temperature in K")
    rotational_speed: float = Field(gt=0, le=5000, description="Rotational speed in rpm")
    torque: float = Field(ge=0, le=150, description="Torque in Nm")
    tool_wear: float = Field(ge=0, le=500, description="Tool wear in minutes")

    model_config = {
        "json_schema_extra": {
            "example": {
                "product_type": "L",
                "air_temperature": 298.9,
                "process_temperature": 309.1,
                "rotational_speed": 2861,
                "torque": 4.6,
                "tool_wear": 143,
            }
        }
    }


class Prediction(BaseModel):
    failure: bool
    failure_probability: float
    model_version: str


def to_dataframe(reading):
    return pd.DataFrame([{
        "Type": reading.product_type,
        "Air temperature [K]": reading.air_temperature,
        "Process temperature [K]": reading.process_temperature,
        "Rotational speed [rpm]": reading.rotational_speed,
        "Torque [Nm]": reading.torque,
        "Tool wear [min]": reading.tool_wear,
    }])


@app.get("/health")
def health():
    return {"status": "ok", "model_version": metadata["model_version"]}


@app.get("/model-info")
def model_info():
    return metadata


@app.post("/predict", response_model=Prediction)
def predict(reading: SensorReading):
    features = build_features(to_dataframe(reading))
    probability = float(model.predict_proba(features)[0, 1])

    return Prediction(
        failure=probability >= 0.5,
        failure_probability=round(probability, 4),
        model_version=metadata["model_version"],
    )
