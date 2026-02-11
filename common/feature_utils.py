from __future__ import annotations

"""
Helpers for interacting with the Feast feature store.

These utilities are intentionally small and thin wrappers around Feast so that
they are easy to unit test and mock in higher-level components.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:  # pragma: no cover - import guard for environments without Feast installed
    from feast import FeatureStore
except Exception:  # pragma: no cover
    FeatureStore = object  # type: ignore[misc,assignment]


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FEAST_REPO = _REPO_ROOT / "services" / "feast" / "feature_repo"


def _normalize_repo_path(repo_path: Optional[str]) -> str:
    if repo_path is None:
        return str(_DEFAULT_FEAST_REPO)
    return str(Path(repo_path).resolve())


def get_feature_store(repo_path: Optional[str] = None) -> FeatureStore:
    """
    Construct a Feast FeatureStore using the configured repository path.

    The default repo path points at `services/feast/feature_repo`.

    When ``REDIS_HOST`` or ``REDIS_PORT`` environment variables are set (e.g.
    inside a Docker container), the Redis online-store connection string is
    overridden so that the static ``feature_store.yaml`` does not need to
    differ between local-dev and containerised environments.
    """
    path = _normalize_repo_path(repo_path)

    fs = FeatureStore(repo_path=path)

    # When REDIS_HOST / REDIS_PORT env vars are present, patch the online-store
    # connection string on the already-loaded config.  Feast lazily initialises
    # the online store, so mutating the config before the first query is safe.
    redis_host = os.getenv("REDIS_HOST")
    redis_port = os.getenv("REDIS_PORT")
    if redis_host or redis_port:
        host = redis_host or "localhost"
        port = redis_port or "6379"
        if hasattr(fs.config, "online_store") and hasattr(fs.config.online_store, "connection_string"):
            fs.config.online_store.connection_string = f"{host}:{port}"

    return fs


def _entity_rows_key(entity_rows: Sequence[Dict[str, Any]]) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
    """
    Build a hashable cache key from entity rows.
    """
    return tuple(
        tuple(sorted(row.items()))
        for row in entity_rows
    )


@dataclass
class _OnlineCacheEntry:
    features: List[str]
    entity_key: Tuple[Tuple[Tuple[str, Any], ...], ...]
    df: pd.DataFrame


class _OnlineFeatureCache:
    """
    Very small in-process cache for online feature fetches.

    This is not intended as a replacement for Redis; it simply avoids duplicate
    round trips for identical requests within a single process.
    """

    def __init__(self) -> None:
        self._entries: Dict[Tuple[int, Tuple[str, ...], Tuple[Tuple[Tuple[str, Any], ...], ...]], _OnlineCacheEntry] = {}

    def get(
        self,
        store: FeatureStore,
        features: Sequence[str],
        entity_rows: Sequence[Dict[str, Any]],
    ) -> Optional[pd.DataFrame]:
        key = (id(store), tuple(sorted(features)), _entity_rows_key(entity_rows))
        entry = self._entries.get(key)
        if entry is None:
            return None
        # Return a copy to avoid callers mutating cached data in-place.
        return entry.df.copy()

    def put(
        self,
        store: FeatureStore,
        features: Sequence[str],
        entity_rows: Sequence[Dict[str, Any]],
        df: pd.DataFrame,
    ) -> None:
        key = (id(store), tuple(sorted(features)), _entity_rows_key(entity_rows))
        self._entries[key] = _OnlineCacheEntry(
            features=list(features),
            entity_key=_entity_rows_key(entity_rows),
            df=df.copy(),
        )


_ONLINE_CACHE = _OnlineFeatureCache()


def fetch_online_features(
    features: Sequence[str],
    entity_rows: Sequence[Dict[str, Any]],
    store: Optional[FeatureStore] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch online features from Feast.

    Args:
        features: List of Feast feature references, e.g. ``["event_realtime_metrics:current_inventory"]``.
        entity_rows: Sequence of entity dictionaries matching the join keys of the feature views.
        store: Optional FeatureStore instance. If not provided, a new one is created with the default repo.
        use_cache: If True, use a small in-process cache keyed by (store, features, entity_rows).
    """
    fs = store or get_feature_store()

    if use_cache:
        cached = _ONLINE_CACHE.get(fs, features, entity_rows)
        if cached is not None:
            return cached

    response = fs.get_online_features(features=features, entity_rows=entity_rows)
    df = response.to_df()

    if use_cache:
        _ONLINE_CACHE.put(fs, features, entity_rows, df)

    return df


def build_training_dataset(
    entity_df: pd.DataFrame,
    feature_refs: Sequence[str],
    store: Optional[FeatureStore] = None,
) -> pd.DataFrame:
    """
    Create a point-in-time correct training dataset using Feast historical retrieval.

    Args:
        entity_df: DataFrame with entity keys and an ``event_timestamp`` column.
        feature_refs: Sequence of feature references (``<feature_view>:<feature_name>``).
        store: Optional FeatureStore instance.
    """
    fs = store or get_feature_store()
    retrieval = fs.get_historical_features(
        entity_df=entity_df,
        features=list(feature_refs),
    )
    return retrieval.to_df()


@lru_cache(maxsize=1)
def feature_store_healthcheck(repo_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Lightweight health check for the feature store configuration.

    This does **not** guarantee Redis or BigQuery connectivity, but it verifies:
    - The repo can be loaded.
    - The registry can be read.
    - At least one FeatureView is defined.
    """
    fs = get_feature_store(repo_path=repo_path)
    views = fs.list_feature_views()
    return {
        "repo_path": _normalize_repo_path(repo_path),
        "feature_view_count": len(views),
        "feature_view_names": sorted(v.name for v in views),
    }


__all__ = [
    "get_feature_store",
    "fetch_online_features",
    "build_training_dataset",
    "feature_store_healthcheck",
]

