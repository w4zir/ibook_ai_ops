from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from common import feature_utils


class _DummyOnlineResponse:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_df(self) -> pd.DataFrame:
        return self._df


class DummyFeatureStore:
    def __init__(self) -> None:
        self.online_calls: List[Dict[str, Any]] = []
        self.historical_calls: List[Dict[str, Any]] = []
        self._online_df = pd.DataFrame({"x": [1, 2, 3]})
        self._historical_df = pd.DataFrame({"y": [4, 5, 6]})

    def get_online_features(self, features, entity_rows):
        self.online_calls.append({"features": list(features), "entity_rows": list(entity_rows)})
        return _DummyOnlineResponse(self._online_df)

    def get_historical_features(self, entity_df, features):
        self.historical_calls.append(
            {"entity_df": entity_df.copy(), "features": list(features)}
        )

        class _DummyHistorical:
            def __init__(self, df: pd.DataFrame) -> None:
                self._df = df

            def to_df(self) -> pd.DataFrame:
                return self._df

        return _DummyHistorical(self._historical_df)

    def list_feature_views(self):
        class _V:
            def __init__(self, name: str) -> None:
                self.name = name

        return [_V("event_realtime_metrics"), _V("user_purchase_behavior")]


def test_fetch_online_features_calls_store_and_uses_cache(monkeypatch):
    store = DummyFeatureStore()
    features = ["event_realtime_metrics:current_inventory"]
    entity_rows = [{"event_id": 1}]

    # First call should hit the store.
    df1 = feature_utils.fetch_online_features(
        features=features,
        entity_rows=entity_rows,
        store=store,
        use_cache=True,
    )
    assert not df1.empty
    assert len(store.online_calls) == 1

    # Second call with the same arguments should be served from cache.
    df2 = feature_utils.fetch_online_features(
        features=features,
        entity_rows=entity_rows,
        store=store,
        use_cache=True,
    )
    assert len(store.online_calls) == 1
    # DataFrames should be equal but not the same object.
    pd.testing.assert_frame_equal(df1, df2)
    assert df1 is not df2


def test_build_training_dataset_uses_historical_features(monkeypatch):
    store = DummyFeatureStore()
    entity_df = pd.DataFrame(
        {
            "event_id": [1, 2],
            "event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    features = ["event_historical_metrics:total_tickets_sold"]

    df = feature_utils.build_training_dataset(
        entity_df=entity_df,
        feature_refs=features,
        store=store,
    )

    # Verify the store was called and a non-empty DataFrame was returned.
    assert len(store.historical_calls) == 1
    assert not df.empty


def test_feature_store_healthcheck_reports_feature_views(monkeypatch):
    # Patch get_feature_store to return our dummy implementation.
    monkeypatch.setattr(feature_utils, "get_feature_store", lambda repo_path=None: DummyFeatureStore())

    result = feature_utils.feature_store_healthcheck(repo_path="some/path")
    assert result["feature_view_count"] == 2
    assert "event_realtime_metrics" in result["feature_view_names"]
    assert "user_purchase_behavior" in result["feature_view_names"]

