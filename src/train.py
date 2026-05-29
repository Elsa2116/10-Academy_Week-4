"""Train, tune, evaluate, and track credit risk models with MLflow."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.data_processing import RANDOM_STATE, build_feature_pipeline, build_modeling_table

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "CreditRiskProbabilityModel"
TARGET = "is_high_risk"
DROP_COLS = ["CustomerId", "risk_cluster", TARGET]


def prepare_xy(raw_path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create modeling table and split features/target."""
    raw = pd.read_csv(raw_path)
    modeling = build_modeling_table(raw)
    y = modeling[TARGET].astype(int)
    X = modeling.drop(columns=[c for c in DROP_COLS if c in modeling.columns])
    return X, y, modeling


def candidate_models() -> Dict[str, Tuple[object, dict]]:
    """Return estimators and small tuning grids for reproducible training."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {"model__n_estimators": [100, 200], "model__max_depth": [3, 6, None]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1]},
        ),
    }


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model using required classification metrics."""
    preds = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = preds
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else 0.0,
    }
    return metrics


def train(raw_path: Path, experiment_name: str, output_dir: Path) -> dict:
    """Train all candidates, log MLflow runs, and save the best model."""
    mlflow.set_experiment(experiment_name)
    X, y, modeling = prepare_xy(raw_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 and y.value_counts().min() >= 2 else None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_path = output_dir / "processed_credit_risk.csv"
    modeling.to_csv(processed_path, index=False)

    best = {"name": None, "score": -1, "model": None, "metrics": None, "params": None}
    for name, (estimator, param_grid) in candidate_models().items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                steps=[("features", build_feature_pipeline()), ("model", estimator)]
            )
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="roc_auc" if y_train.nunique() > 1 else "accuracy",
                cv=3,
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)
            metrics = evaluate(grid.best_estimator_, X_test, y_test)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(processed_path), artifact_path="data")
            mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")
            LOGGER.info("%s metrics: %s", name, metrics)

            selection_score = metrics["roc_auc"] if y_test.nunique() > 1 else metrics["accuracy"]
            if selection_score > best["score"]:
                best.update(
                    {
                        "name": name,
                        "score": selection_score,
                        "model": grid.best_estimator_,
                        "metrics": metrics,
                        "params": grid.best_params_,
                    }
                )

    model_path = output_dir / "best_model.joblib"
    joblib.dump(best["model"], model_path)
    summary_path = output_dir / "model_metrics.json"
    summary_path.write_text(json.dumps({k: v for k, v in best.items() if k != "model"}, indent=2))

    with mlflow.start_run(run_name="best_model_registration"):
        mlflow.log_params(best["params"] or {})
        mlflow.log_metrics(best["metrics"] or {})
        mlflow.log_artifact(str(summary_path), artifact_path="reports")
        mlflow.sklearn.log_model(
            best["model"],
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
    return {
        "model_path": str(model_path),
        "summary_path": str(summary_path),
        "best_model": best["name"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit risk models.")
    parser.add_argument("--input", required=True, type=Path, help="Raw transaction CSV path")
    parser.add_argument("--experiment-name", default="credit-risk-model")
    parser.add_argument("--output-dir", default=Path("data/processed"), type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = train(args.input, args.experiment_name, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
