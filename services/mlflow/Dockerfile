FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir mlflow==2.10.2 psycopg2-binary==2.9.9 boto3==1.34.34

EXPOSE 5000

# Expected env vars:
# - MLFLOW_BACKEND_STORE_URI (e.g. postgresql+psycopg2://user:pass@host:5432/mlflow)
# - MLFLOW_DEFAULT_ARTIFACT_ROOT (e.g. s3://mlflow/)
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" --default-artifact-root \"$MLFLOW_DEFAULT_ARTIFACT_ROOT\""]

