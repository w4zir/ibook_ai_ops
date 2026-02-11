import pytest
from pydantic import ValidationError

from common.config import load_config


def test_load_local_config_defaults(minimal_local_env):
    cfg = load_config(env_file=None)
    assert cfg.environment == "local"
    assert cfg.postgres.host == "localhost"
    assert cfg.redis.port == 6379
    assert cfg.feast.offline_store == "duckdb"
    assert cfg.feast.duckdb_path.endswith("feast_offline.duckdb")


def test_airflow_and_mlflow_uris_derived(minimal_local_env):
    cfg = load_config(env_file=None)
    assert cfg.postgres.airflow_sqlalchemy_conn.endswith("/airflow")
    assert cfg.postgres.mlflow_backend_store_uri.endswith("/mlflow")
    assert "postgresql+psycopg2://" in cfg.postgres.airflow_sqlalchemy_conn


def test_production_requires_bigquery_and_gcs(monkeypatch: pytest.MonkeyPatch, minimal_local_env):
    # Switch to production and intentionally omit required fields.
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("FEAST_OFFLINE_STORE", "bigquery")
    monkeypatch.delenv("FEAST_BIGQUERY_DATASET", raising=False)
    monkeypatch.delenv("STORAGE_GCS_BUCKET", raising=False)

    with pytest.raises(ValidationError):
        load_config(env_file=None)


def test_production_valid_when_required_fields_present(monkeypatch: pytest.MonkeyPatch, minimal_local_env):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("FEAST_OFFLINE_STORE", "bigquery")
    monkeypatch.setenv("FEAST_BIGQUERY_DATASET", "ibook_mlops_feast")
    monkeypatch.setenv("STORAGE_GCS_BUCKET", "ibook-mlops-artifacts")

    cfg = load_config(env_file=None)
    assert cfg.environment == "production"
    assert cfg.feast.offline_store == "bigquery"
    assert cfg.feast.bigquery_dataset == "ibook_mlops_feast"
    assert cfg.storage.gcs_bucket == "ibook-mlops-artifacts"


# ---------------------------------------------------------------------------
# AutoTrainingConfig
# ---------------------------------------------------------------------------


def test_auto_training_defaults(minimal_local_env):
    cfg = load_config(env_file=None)
    at = cfg.auto_training
    assert at.enabled is True
    assert at.failure_rate_threshold == 0.4
    assert at.monitoring_window_seconds == 300
    assert at.cooldown_seconds == 120
    assert at.min_samples == 20
    assert at.training_dataset_size == 512


def test_auto_training_from_env(monkeypatch: pytest.MonkeyPatch, minimal_local_env):
    monkeypatch.setenv("AUTO_TRAINING_ENABLED", "false")
    monkeypatch.setenv("AUTO_TRAINING_FAILURE_RATE_THRESHOLD", "0.25")
    monkeypatch.setenv("AUTO_TRAINING_MONITORING_WINDOW_SECONDS", "120")
    monkeypatch.setenv("AUTO_TRAINING_COOLDOWN_SECONDS", "60")
    monkeypatch.setenv("AUTO_TRAINING_MIN_SAMPLES", "50")
    monkeypatch.setenv("AUTO_TRAINING_TRAINING_DATASET_SIZE", "256")

    cfg = load_config(env_file=None)
    at = cfg.auto_training
    assert at.enabled is False
    assert at.failure_rate_threshold == 0.25
    assert at.monitoring_window_seconds == 120
    assert at.cooldown_seconds == 60
    assert at.min_samples == 50
    assert at.training_dataset_size == 256

