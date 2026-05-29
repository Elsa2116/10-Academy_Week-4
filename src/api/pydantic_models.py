"""Pydantic schemas for credit risk API."""
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Flexible feature payload for scoring one customer."""

    features: Dict[str, Any] = Field(
        ...,
        description="Model feature dictionary matching the trained feature table columns.",
        examples=[
            {
                "total_amount": 5000,
                "avg_amount": 250,
                "transaction_count": 20,
                "std_amount": 100,
                "total_value": 5000,
                "avg_value": 250,
                "recency": 7,
                "frequency": 20,
                "monetary": 5000,
            }
        ],
    )


class PredictionResponse(BaseModel):
    """Prediction response returned by the API."""

    risk_probability: float
    is_high_risk: int
    credit_score: int
