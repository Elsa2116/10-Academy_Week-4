"""FastAPI app for serving the credit risk model."""
from __future__ import annotations

import os
from pathlib import Path

import mlflow.sklearn
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictionRequest, PredictionResponse
from src.predict import load_model, predict_one

MODEL_URI = os.getenv("MODEL_URI", "models:/CreditRiskProbabilityModel/latest")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "data/processed/best_model.joblib")

app = FastAPI(
    title="Credit Risk Probability API",
    description="Scores customers using transaction-derived risk features.",
    version="1.0.0",
)
_model = None
_model_type = None


def get_model():
    """Load the model once, preferring a local file and falling back to MLflow registry."""
    global _model, _model_type
    if _model is not None:
        return _model, _model_type
    local_path = Path(LOCAL_MODEL_PATH)
    try:
        if local_path.exists():
            _model = load_model(local_path)
        else:
            _model = mlflow.sklearn.load_model(MODEL_URI)
        _model_type = "sklearn"
        return _model, _model_type
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model is not available: {exc}") from exc


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Return risk probability and credit score for one customer."""
    model, _ = get_model()
    try:
        result = predict_one(model, request.features)
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
