"""Pipeline stage 2: train candidate models, track them with MLflow,
apply a quality gate, and register and export the best model.

Run with:  python -m src.train
The script exits with code 1 if the best model fails the quality gate,
so a weaker model can never reach the Docker image.
"""

import json
import os
import sys
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import skops.io
from mlflow import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, load_dataset

DATA_PATH = "data/ai4i2020.csv"
MODEL_DIR = "model"
EXPERIMENT_NAME = "predictive-maintenance"
REGISTERED_MODEL_NAME = "predictive-maintenance-classifier"

# The model is only promoted if it clears both thresholds on the test set
MIN_TEST_F1 = 0.70
MIN_TEST_RECALL = 0.70

RANDOM_STATE = 42


def candidate_models():
    return {
        "logistic_regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]),
            {"model": "LogisticRegression", "class_weight": "balanced", "max_iter": 1000},
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {"model": "RandomForestClassifier", "n_estimators": 300,
             "min_samples_leaf": 2, "class_weight": "balanced_subsample"},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_iter=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {"model": "HistGradientBoostingClassifier", "learning_rate": 0.1,
             "max_iter": 300, "class_weight": "balanced"},
        ),
    }


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "test_precision": precision_score(y_test, predictions),
        "test_recall": recall_score(y_test, predictions),
        "test_f1": f1_score(y_test, predictions),
        "test_roc_auc": roc_auc_score(y_test, probabilities),
    }, confusion_matrix(y_test, predictions).tolist()


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # GITHUB_SHA is set by GitHub Actions, so every run links back to a commit
    commit = os.environ.get("GITHUB_SHA", "local")

    results = []

    for name, (model, params) in candidate_models().items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"stage": "candidate", "git_commit": commit})
            mlflow.log_params(params)
            mlflow.log_params({
                "n_features": len(FEATURE_COLUMNS),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            })

            cv_scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())
            mlflow.log_metric("cv_f1_std", cv_scores.std())

            model.fit(X_train, y_train)
            metrics, matrix = evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.log_dict({"confusion_matrix": matrix}, "confusion_matrix.json")

            results.append({
                "name": name,
                "run_id": run.info.run_id,
                "model": model,
                "cv_f1_mean": cv_scores.mean(),
                **metrics,
            })

        print("%-24s cv F1 %.3f   test F1 %.3f   test recall %.3f" % (
            name, cv_scores.mean(), metrics["test_f1"], metrics["test_recall"]
        ))

    # Selection uses cross validation on the training data only,
    # the test set is kept for the quality gate
    best = max(results, key=lambda result: result["cv_f1_mean"])
    print("\nSelected model:", best["name"])

    gate_passed = (
        best["test_f1"] >= MIN_TEST_F1 and best["test_recall"] >= MIN_TEST_RECALL
    )

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("quality_gate", "passed" if gate_passed else "failed")

        if not gate_passed:
            print("QUALITY GATE FAILED: test F1 %.3f (min %.2f), test recall %.3f (min %.2f)" % (
                best["test_f1"], MIN_TEST_F1, best["test_recall"], MIN_TEST_RECALL
            ))
            sys.exit(1)

        mlflow.set_tag("stage", "champion")

        # MLflow saves the model with skops, which only loads types it is told
        # to trust. The model was trained in this process, so its types are safe.
        trusted_types = skops.io.get_untrusted_types(data=skops.io.dumps(best["model"]))

        model_info = mlflow.sklearn.log_model(
            best["model"],
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_test.head(3),
            skops_trusted_types=trusted_types,
        )

    version = model_info.registered_model_version
    MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version)

    print("QUALITY GATE PASSED")
    print("Registered %s version %s with alias 'champion'" % (REGISTERED_MODEL_NAME, version))

    # The API container only needs scikit-learn, so the model is also
    # exported as a plain joblib file together with its metadata
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best["model"], os.path.join(MODEL_DIR, "model.joblib"))

    metadata = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": str(version),
        "algorithm": best["name"],
        "mlflow_run_id": best["run_id"],
        "git_commit": commit,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": FEATURE_COLUMNS,
        "metrics": {
            key: round(float(best[key]), 4)
            for key in ["cv_f1_mean", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
        },
    }

    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Exported model to %s/" % MODEL_DIR)


if __name__ == "__main__":
    main()
