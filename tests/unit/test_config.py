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

