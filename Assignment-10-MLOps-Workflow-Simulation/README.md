# Assignment 10: MLOps Workflow Simulation

This project builds a CI/CD pipeline for a machine failure prediction model using MLflow and Docker.  
Every change to this folder is validated, trained, tested, packaged into a Docker image, and smoke tested automatically by GitHub Actions.

## Pipeline

```
 data/ai4i2020.csv
        |
 1. Validate data       src/validate_data.py   schema, missing values, sensor ranges, failure rate
        |
 2. Train and track     src/train.py           3 candidate models, every run logged to MLflow
        |
 3. Quality gate        src/train.py           test F1 >= 0.70 and recall >= 0.70, otherwise the pipeline stops
        |
 4. Register            MLflow Model Registry  new version with alias "champion", exported to model/
        |
 5. Test                tests/                 data checks, model behaviour, API responses (15 tests)
        |
 6. Package             Dockerfile             FastAPI service with the exported model
        |
 7. Smoke test          GitHub Actions         start the container, call /health and /predict
```

The workflow file is at [`.github/workflows/assignment-10-mlops.yml`](../.github/workflows/assignment-10-mlops.yml) in the repository root.

## Model

The model predicts machine failure from the AI4I 2020 sensor readings, using the engineered features found in Assignment 9 (power, temperature difference, tool strain).

| Candidate | CV F1 | Test F1 | Test recall |
|-----------|-------|---------|-------------|
| Logistic Regression | 0.271 | 0.299 | 0.882 |
| **Random Forest** (selected) | **0.870** | **0.882** | **0.824** |
| Hist Gradient Boosting | 0.854 | 0.875 | 0.824 |

The model is selected on 5-fold cross validation F1 using the training data only. The test set is kept for the quality gate.

## Run the pipeline locally

Run all commands from this folder.

```
pip install -r requirements.txt

python -m src.validate_data
python -m src.train
python -m pytest -v
```

View the MLflow runs and the model registry:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.

## Run the API

Without Docker:

```
uvicorn app.main:app --reload
```

With Docker (Docker Desktop must be running):

```
docker build -t predictive-maintenance-api .
docker run -d -p 8000:8000 --name api predictive-maintenance-api
```

Interactive API docs are at http://127.0.0.1:8000/docs.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status and model version |
| `/model-info` | GET | Model metadata, MLflow run ID, and metrics |
| `/predict` | POST | Failure prediction for one sensor reading |

Example request:

```json
{
  "product_type": "L",
  "air_temperature": 300.0,
  "process_temperature": 310.0,
  "rotational_speed": 1300,
  "torque": 80.0,
  "tool_wear": 20
}
```

Example response:

```json
{
  "failure": true,
  "failure_probability": 0.9585,
  "model_version": "1"
}
```

## Generated files

`mlflow.db`, `mlruns/` and `model/` are created by `python -m src.train` and are not committed to git. In CI they are uploaded as workflow artifacts instead.

## Dataset

AI4I 2020 Predictive Maintenance dataset, 10,000 sensor readings, included in `data/`.

- https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
