from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class PostgresConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = Field(default="webook")
    password: str = Field(default="webook")

    airflow_db: str = Field(default="airflow")
    mlflow_db: str = Field(default="mlflow")

    def sqlalchemy_uri(self, db_name: str) -> str:
        # Airflow uses SQLAlchemy connection strings.
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{db_name}"

    @property
    def airflow_sqlalchemy_conn(self) -> str:
        return self.sqlalchemy_uri(self.airflow_db)

    @property
    def mlflow_backend_store_uri(self) -> str:
        return self.sqlalchemy_uri(self.mlflow_db)


class RedisConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    password: Optional[str] = Field(default=None)


class MinioConfig(BaseModel):
    endpoint: str = Field(default="http://localhost:9000")
    access_key: str = Field(default="minioadmin")
    secret_key: str = Field(default="minioadmin")
    bucket: str = Field(default="mlflow")


class FeastConfig(BaseModel):
    offline_store: Literal["duckdb", "bigquery"] = Field(default="duckdb")
    duckdb_path: str = Field(default="data/processed/feast_offline.duckdb")
    bigquery_dataset: Optional[str] = Field(default=None)


class MlflowConfig(BaseModel):
    tracking_uri: str = Field(default="http://localhost:5000")
    # For local, MLflow will use MinIO via S3 API by default.
    artifact_root: str = Field(default="s3://mlflow/")


class AirflowConfig(BaseModel):
    webserver_url: str = Field(default="http://localhost:8080")


class StorageConfig(BaseModel):
    # Local uses MinIO; production can use GCS.
    gcs_bucket: Optional[str] = Field(default=None)


class AppConfig(BaseModel):
    environment: Literal["local", "production"] = Field(default="local")

    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)

    feast: FeastConfig = Field(default_factory=FeastConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    airflow: AirflowConfig = Field(default_factory=AirflowConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @field_validator("environment", mode="before")
    @classmethod
    def _normalize_environment(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @model_validator(mode="after")
    def _validate_environment_requirements(self) -> "AppConfig":
        if self.environment == "production":
            if self.feast.offline_store != "bigquery":
                raise ValueError("In production, feast.offline_store must be 'bigquery'.")
            if not self.feast.bigquery_dataset:
                raise ValueError("In production, FEAST_BIGQUERY_DATASET is required.")
            if not self.storage.gcs_bucket:
                raise ValueError("In production, STORAGE_GCS_BUCKET is required.")
        return self


def _load_env(env_file: Optional[str]) -> None:
    # Load env file if present; do not override explicit process environment.
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        # Fall back to .env.local if it exists.
        load_dotenv(".env.local", override=False)


def _from_env(prefix: str, key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(f"{prefix}{key}", default)


def load_config(env_file: Optional[str] = None) -> AppConfig:
    """
    Load configuration from environment variables (and optional env file).

    Env var names are grouped with prefixes:
    - ENVIRONMENT
    - POSTGRES_*
    - REDIS_*
    - MINIO_*
    - FEAST_*
    - MLFLOW_*
    - AIRFLOW_*
    - STORAGE_*
    """
    _load_env(env_file)

    environment = os.getenv("ENVIRONMENT", "local")

    postgres = PostgresConfig(
        host=_from_env("POSTGRES_", "HOST", "localhost") or "localhost",
        port=int(_from_env("POSTGRES_", "PORT", "5432") or "5432"),
        user=_from_env("POSTGRES_", "USER", "webook") or "webook",
        password=_from_env("POSTGRES_", "PASSWORD", "webook") or "webook",
        airflow_db=_from_env("POSTGRES_", "AIRFLOW_DB", "airflow") or "airflow",
        mlflow_db=_from_env("POSTGRES_", "MLFLOW_DB", "mlflow") or "mlflow",
    )

    redis = RedisConfig(
        host=_from_env("REDIS_", "HOST", "localhost") or "localhost",
        port=int(_from_env("REDIS_", "PORT", "6379") or "6379"),
        password=_from_env("REDIS_", "PASSWORD", None),
    )

    minio = MinioConfig(
        endpoint=_from_env("MINIO_", "ENDPOINT", "http://localhost:9000") or "http://localhost:9000",
        access_key=_from_env("MINIO_", "ACCESS_KEY", "minioadmin") or "minioadmin",
        secret_key=_from_env("MINIO_", "SECRET_KEY", "minioadmin") or "minioadmin",
        bucket=_from_env("MINIO_", "BUCKET", "mlflow") or "mlflow",
    )

    feast = FeastConfig(
        offline_store=(os.getenv("FEAST_OFFLINE_STORE", "duckdb") or "duckdb").strip().lower(),  # type: ignore[arg-type]
        duckdb_path=_from_env("FEAST_", "DUCKDB_PATH", "data/processed/feast_offline.duckdb")
        or "data/processed/feast_offline.duckdb",
        bigquery_dataset=_from_env("FEAST_", "BIGQUERY_DATASET", None),
    )

    storage = StorageConfig(gcs_bucket=_from_env("STORAGE_", "GCS_BUCKET", None))

    mlflow = MlflowConfig(
        tracking_uri=_from_env("MLFLOW_", "TRACKING_URI", "http://localhost:5000") or "http://localhost:5000",
        artifact_root=_from_env("MLFLOW_", "ARTIFACT_ROOT", "s3://mlflow/") or "s3://mlflow/",
    )

    airflow = AirflowConfig(webserver_url=_from_env("AIRFLOW_", "WEBSERVER_URL", "http://localhost:8080") or "http://localhost:8080")

    try:
        return AppConfig(
            environment=environment,
            postgres=postgres,
            redis=redis,
            minio=minio,
            feast=feast,
            mlflow=mlflow,
            airflow=airflow,
            storage=storage,
        )
    except ValidationError as e:
        # Re-raise with a message that’s easier to spot in logs/tests.
        raise ValidationError.from_exception_data(e.title, e.errors()) from e


@lru_cache(maxsize=1)
def get_config(env_file: Optional[str] = None) -> AppConfig:
    """
    Cached config loader. In tests, prefer calling `load_config()` directly
    or clear this cache via `get_config.cache_clear()`.
    """
    return load_config(env_file=env_file)

