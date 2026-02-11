from __future__ import annotations

import types
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("mlflow")

from services.bentoml.services.fraud_detection import service as fraud_service
from services.bentoml.services.dynamic_pricing import service as pricing_service


@pytest.fixture(autouse=True)
def _patch_feast_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch Feast clients to avoid external dependencies.
    def fake_fraud_features(entity_rows, feature_refs=None):  # type: ignore[override]
        return pd.DataFrame(
            {
                "user_purchase_behavior__lifetime_purchases": [10] * len(entity_rows),
                "user_purchase_behavior__fraud_risk_score": [0.3] * len(entity_rows),
            }
        )

    def fake_pricing_features(entity_rows, feature_refs=None):  # type: ignore[override]
        return pd.DataFrame(
            {
                "event_realtime_metrics__current_inventory": [100] * len(entity_rows),
                "event_realtime_metrics__sell_through_rate_5min": [1.0] * len(entity_rows),
            }
        )

    monkeypatch.setattr(
        "services.bentoml.common.feast_client.get_fraud_features_for_entities",
        fake_fraud_features,
    )
    # Also patch the reference imported into the fraud_detection service module.
    monkeypatch.setattr(
        "services.bentoml.services.fraud_detection.service.get_fraud_features_for_entities",
        fake_fraud_features,
    )
    monkeypatch.setattr(
        "services.bentoml.common.feast_client.get_pricing_features_for_entities",
        fake_pricing_features,
    )
    # Also patch the reference imported into the dynamic_pricing service module.
    monkeypatch.setattr(
        "services.bentoml.services.dynamic_pricing.service.get_pricing_features_for_entities",
        fake_pricing_features,
    )

    # Patch MLflow model loading for fraud service.
    class DummyPyfuncModel:
        def predict(self, X):
            import numpy as np

            return np.full(shape=(len(X),), fill_value=0.7)

        def predict_proba(self, X):
            import numpy as np

            scores = np.full(shape=(len(X),), fill_value=0.7)
            return np.column_stack([1 - scores, scores])

    def fake_resolve_latest_model(*_: Any, **__: Any) -> Any:
        return types.SimpleNamespace(model_uri="models:/fraud_detection/1")

    def fake_load_model(*_: Any, **__: Any) -> Any:
        return DummyPyfuncModel()

    monkeypatch.setattr(
        "services.bentoml.services.fraud_detection.service.resolve_latest_model",
        fake_resolve_latest_model,
    )
    monkeypatch.setattr(
        "mlflow.pyfunc.load_model",
        fake_load_model,
    )


def test_fraud_prediction_api_logic() -> None:
    batch = fraud_service.FraudBatchRequest(
        requests=[
            fraud_service.FraudRequest(user_id=1, event_id=2, amount=100.0),
        ]
    )

    response = fraud_service.handle_predict(batch)
    assert len(response.predictions) == 1
    pred = response.predictions[0]
    assert 0.0 <= pred.fraud_score <= 1.0
    assert pred.is_fraud is True


def test_fraud_feedback_api_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the /feedback endpoint records outcomes and returns metrics."""
    from services.bentoml.services.fraud_detection.model import (
        FraudFeedbackItem,
        FraudFeedbackRequest,
    )

    # Reset the global tracker so we start fresh.
    monkeypatch.setattr(fraud_service, "_FAILURE_TRACKER", None)
    monkeypatch.setattr(fraud_service, "_ORCHESTRATOR", None)

    feedback_req = FraudFeedbackRequest(
        feedbacks=[
            FraudFeedbackItem(user_id=1, event_id=2, predicted_fraud=True, actual_fraud=True),
            FraudFeedbackItem(user_id=3, event_id=4, predicted_fraud=True, actual_fraud=False),
            FraudFeedbackItem(user_id=5, event_id=6, predicted_fraud=False, actual_fraud=False),
        ]
    )
    response = fraud_service.handle_feedback(feedback_req)
    assert response.accepted == 3
    assert response.failure_rate >= 0.0
    assert response.window_samples >= 0


def test_pricing_recommendation_api_logic() -> None:
    batch = pricing_service.PricingBatchRequest(
        requests=[
            pricing_service.PricingRequest(event_id=1, current_price=100.0),
        ]
    )

    response = pricing_service.handle_pricing(batch)
    assert len(response.predictions) == 1
    pred = response.predictions[0]
    assert pred.recommended_price > 0
    assert pred.lower_confidence < pred.recommended_price < pred.upper_confidence

