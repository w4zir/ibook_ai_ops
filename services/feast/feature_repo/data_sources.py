from __future__ import annotations

from pathlib import Path
from typing import Optional

from feast import FileSource
from feast.data_source import DataSource

from common.config import AppConfig, get_config

try:
    # BigQuerySource is only used when running in production with BigQuery configured.
    from feast import BigQuerySource
except Exception:  # pragma: no cover - import is optional in local/dev
    BigQuerySource = None  # type: ignore


_REPO_ROOT = Path(__file__).resolve().parents[3]
_FEAST_DATA_DIR = _REPO_ROOT / "data" / "processed" / "feast"


def _ensure_local_paths() -> None:
    """
    Ensure the local Feast data directory exists.

    This is safe to call from both application code and tests.
    """
    _FEAST_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _use_bigquery(cfg: AppConfig) -> bool:
    """
    Determine whether to use BigQuery-backed sources.

    We only switch to BigQuery when:
    - ENVIRONMENT=production
    - FEAST_OFFLINE_STORE=bigquery
    - FEAST_BIGQUERY_DATASET is set
    """
    return (
        cfg.environment == "production"
        and cfg.feast.offline_store == "bigquery"
        and bool(cfg.feast.bigquery_dataset)
    )


def get_event_metrics_source(cfg: Optional[AppConfig] = None) -> DataSource:
    """
    Source for event-level metrics (both realtime and historical views).
    """
    cfg = cfg or get_config()

    if _use_bigquery(cfg) and BigQuerySource is not None:
        dataset = cfg.feast.bigquery_dataset
        assert dataset is not None  # for type checkers
        return BigQuerySource(
            name="event_metrics_bq",
            table=f"{dataset}.event_metrics",
            timestamp_field="event_timestamp",
            created_timestamp_column="ingested_at",
        )

    _ensure_local_paths()
    return FileSource(
        name="event_metrics_file",
        path=str(_FEAST_DATA_DIR / "event_metrics.parquet"),
        timestamp_field="event_timestamp",
        created_timestamp_column="ingested_at",
    )


def get_user_metrics_source(cfg: Optional[AppConfig] = None) -> DataSource:
    """
    Source for user-level purchase behavior metrics.
    """
    cfg = cfg or get_config()

    if _use_bigquery(cfg) and BigQuerySource is not None:
        dataset = cfg.feast.bigquery_dataset
        assert dataset is not None
        return BigQuerySource(
            name="user_metrics_bq",
            table=f"{dataset}.user_metrics",
            timestamp_field="event_timestamp",
            created_timestamp_column="ingested_at",
        )

    _ensure_local_paths()
    return FileSource(
        name="user_metrics_file",
        path=str(_FEAST_DATA_DIR / "user_metrics.parquet"),
        timestamp_field="event_timestamp",
        created_timestamp_column="ingested_at",
    )

