import os

import pytest

from common.config import get_config, load_config


@pytest.fixture(autouse=True)
def _clear_config_cache():
    # Ensure one test's env doesn't leak into another.
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture
def minimal_local_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_USER", "webook")
    monkeypatch.setenv("POSTGRES_PASSWORD", "webook")
    monkeypatch.setenv("POSTGRES_AIRFLOW_DB", "airflow")
    monkeypatch.setenv("POSTGRES_MLFLOW_DB", "mlflow")

    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")

    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minioadmin")
    monkeypatch.setenv("MINIO_BUCKET", "mlflow")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", "s3://mlflow/")

    monkeypatch.setenv("AIRFLOW_WEBSERVER_URL", "http://localhost:8080")
    monkeypatch.setenv("FEAST_OFFLINE_STORE", "duckdb")
    monkeypatch.setenv("FEAST_DUCKDB_PATH", "data/processed/feast_offline.duckdb")


@pytest.fixture
def config(minimal_local_env):
    # Load from environment (not from file) for deterministic tests.
    return load_config(env_file=None)

