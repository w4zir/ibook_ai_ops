from __future__ import annotations

"""
Pydantic models and thin wrappers for the fraud detection BentoML service.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class FraudRequest(BaseModel):
    """
    Incoming request schema for fraud prediction.

    In the simplest form, callers provide identifiers that can be resolved
    to features via Feast. Advanced callers may also pass pre-computed
    feature values in `feature_overrides`.
    """

    # NOTE: Upstream systems may represent identifiers either as opaque strings
    # (for example ``\"user_221f1d1b\"``) or as integers. To keep the BentoML
    # schema compatible with both the simulator and potential numeric callers,
    # we accept both types here.
    user_id: Union[str, int] = Field(
        ...,
        description="User identifier (string or int).",
    )
    event_id: Union[str, int] = Field(
        ...,
        description="Event identifier (string or int).",
    )
    amount: float = Field(..., ge=0, description="Transaction amount.")
    feature_overrides: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional direct feature values overriding Feast lookups.",
    )


class FraudResponse(BaseModel):
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of fraud.")
    is_fraud: bool = Field(..., description="Boolean decision based on configured threshold.")


class FraudBatchRequest(BaseModel):
    requests: List[FraudRequest]


class FraudBatchResponse(BaseModel):
    predictions: List[FraudResponse]


class FraudFeedbackItem(BaseModel):
    """Ground-truth feedback for a single prediction."""

    user_id: Union[str, int] = Field(..., description="User identifier matching the original prediction.")
    event_id: Union[str, int] = Field(..., description="Event identifier matching the original prediction.")
    predicted_fraud: bool = Field(..., description="The model's prediction (is_fraud from the response).")
    actual_fraud: bool = Field(..., description="Ground-truth: was the transaction actually fraudulent?")


class FraudFeedbackRequest(BaseModel):
    """Batch of ground-truth feedback items."""

    feedbacks: List[FraudFeedbackItem]


class FraudFeedbackResponse(BaseModel):
    """Acknowledgement of feedback submission."""

    accepted: int = Field(..., description="Number of feedback items accepted.")
    failure_rate: float = Field(..., ge=0.0, le=1.0, description="Current failure rate in the monitoring window.")
    window_samples: int = Field(..., ge=0, description="Number of samples in the current monitoring window.")
    training_triggered: bool = Field(default=False, description="Whether auto-retraining was triggered.")


@dataclass
class FraudModelRuntime:
    """
    Lightweight runtime wrapper around the underlying ML model.

    The underlying model is expected to expose either `predict_proba(X)[:, 1]`
    or `predict(X)` returning probabilities.
    """

    model: Any
    threshold: float = 0.5

    def predict_scores(self, features: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(features.to_numpy())[:, 1]
        else:
            # Fallback to `predict` if available; assume it returns probabilities.
            scores = self.model.predict(features.to_numpy())
        return np.asarray(scores, dtype=float)

    def predict_batch(self, features: pd.DataFrame) -> List[FraudResponse]:
        scores = self.predict_scores(features)
        return [
            FraudResponse(fraud_score=float(score), is_fraud=bool(score >= self.threshold))
            for score in scores
        ]


__all__ = [
    "FraudRequest",
    "FraudResponse",
    "FraudBatchRequest",
    "FraudBatchResponse",
    "FraudFeedbackItem",
    "FraudFeedbackRequest",
    "FraudFeedbackResponse",
    "FraudModelRuntime",
]

