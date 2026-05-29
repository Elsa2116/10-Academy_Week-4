"""Prediction utilities for the credit risk model."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import joblib
import pandas as pd


def load_model(model_path: str | Path):
    """Load a serialized sklearn pipeline."""
    return joblib.load(model_path)


def probability_to_score(probability: float, min_score: int = 300, max_score: int = 850) -> int:
    """Convert risk probability to a credit score where lower risk gets a higher score."""
    probability = min(max(float(probability), 0.0), 1.0)
    return int(round(max_score - probability * (max_score - min_score)))


def predict_one(model, features: Mapping[str, object]) -> Dict[str, object]:
    """Predict risk probability and score for one customer feature row."""
    X = pd.DataFrame([dict(features)])
    probability = float(model.predict_proba(X)[:, 1][0])
    label = int(probability >= 0.5)
    return {
        "risk_probability": probability,
        "is_high_risk": label,
        "credit_score": probability_to_score(probability),
    }
