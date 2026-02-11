from __future__ import annotations

import pandas as pd

from services.bentoml.services.fraud_detection import model as fraud_model
from services.bentoml.services.fraud_detection import service as fraud_service


def test_build_feature_frame_resets_index_and_applies_overrides() -> None:
    batch = fraud_model.FraudBatchRequest(
        requests=[
            fraud_model.FraudRequest(
                user_id="user_1",
                event_id="event_1",
                amount=100.0,
                feature_overrides={"user_purchase_behavior__lifetime_purchases": 42.0},
            ),
            fraud_model.FraudRequest(
                user_id="user_2",
                event_id="event_2",
                amount=50.0,
                feature_overrides={},
            ),
        ]
    )

    # Simulate a Feast DataFrame with a non-default index and feature columns.
    base_df = pd.DataFrame(
        {
            "user_purchase_behavior__lifetime_purchases": [10.0, 20.0],
            "user_purchase_behavior__fraud_risk_score": [0.1, 0.2],
        },
        index=[5, 7],
    )

    def fake_get_fraud_features_for_entities(entity_rows, feature_refs=None):  # type: ignore[override]
        # Validate that entity_rows were built from the FraudRequest IDs.
        assert len(entity_rows) == 2
        # String IDs like "user_1" / "event_2" are normalized to integer keys.
        assert entity_rows[0]["user_id"] == 1
        assert entity_rows[1]["event_id"] == 2
        return base_df

    # Patch the local reference used inside the service module.
    fraud_service.get_fraud_features_for_entities = fake_get_fraud_features_for_entities  # type: ignore[assignment]

    features_df = fraud_service._build_feature_frame(batch)

    # Index should be reset to a simple RangeIndex for positional mapping.
    assert list(features_df.index) == [0, 1]

    # Overrides should be applied to the correct row/column.
    assert features_df.loc[0, "user_purchase_behavior__lifetime_purchases"] == 42.0
    assert features_df.loc[1, "user_purchase_behavior__lifetime_purchases"] == 20.0

    # Columns should be ordered and limited to the model's feature set.
    assert list(features_df.columns) == [
        "user_purchase_behavior__lifetime_purchases",
        "user_purchase_behavior__fraud_risk_score",
    ]

