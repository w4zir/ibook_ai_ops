from __future__ import annotations

"""
Airflow DAG for Phase 5: model training pipeline.

This DAG is a weekly orchestration wrapper around the fraud model training
utilities in `common.model_utils`. It is designed to be:

- **Config-aware**: training utilities pick up MLflow configuration from
  `common.config.get_config`.
- **Test-friendly**: tasks operate on small, synthetic in-memory data so that
  unit tests and local runs stay fast.
- **Production-shaped**: the task graph mirrors a real promotion workflow with
  explicit training, evaluation, registration (to Staging for canary), and
  canary steps; deploy_canary, monitor_canary, and finalize_promotion remain
  stubs for now.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from mlflow.tracking import MlflowClient

from common.config import get_config
from common.model_utils import TrainingResult, build_fraud_training_dataframe, train_fraud_model

try:
    from .utils import get_retries_for_dag
except ImportError:
    from utils import get_retries_for_dag


DAG_ID = "model_training_pipeline"


def _bentoml_base_url() -> str:
    return (
        os.environ.get("BENTOML_BASE_URL")
        or os.environ.get("FRAUD_API_BASE_URL")
        or "http://localhost:7001"
    ).rstrip("/")


def _registered_model_name() -> str:
    """Model name in MLflow Model Registry; align with BentoML's BENTOML_MODEL_NAME."""
    return os.environ.get("BENTOML_MODEL_NAME", "fraud_detection")


def _build_training_dataset(**context: Any) -> pd.DataFrame:
    """
    Build a tiny synthetic user-metrics dataset suitable for fraud training.

    In a later phase this will be wired to Feast via `common.feature_utils`.
    For now we generate a deterministic DataFrame that mimics the schema of
    the Feast-backed user metrics. When triggered by ml_monitoring_pipeline
    with drift context, logs conf for observability.
    """
    dag_run = context.get("dag_run")
    if dag_run and getattr(dag_run, "conf", None):
        logging.info(
            "Model training run triggered with conf: drift_score=%s triggered_by=%s",
            dag_run.conf.get("drift_score"),
            dag_run.conf.get("triggered_by"),
        )
    rng = np.random.default_rng(seed=42)
    n_rows = 512

    user_metrics = pd.DataFrame(
        {
            "user_id": np.arange(1, n_rows + 1, dtype=int),
            "lifetime_purchases": rng.integers(0, 100, size=n_rows),
            "fraud_risk_score": rng.uniform(0.0, 1.0, size=n_rows),
        }
    )

    df = build_fraud_training_dataframe(user_metrics)
    logging.info("Built synthetic training DataFrame with shape %s", df.shape)
    return df


def _train_model(**context: Any) -> Dict[str, Any]:
    """
    Train the fraud model on the synthetic dataset and log to MLflow.

    We keep the number of Optuna trials intentionally low via the default
    parameters in `train_fraud_model` so that a single DAG run is fast.
    """
    ti = context["ti"]
    df: pd.DataFrame = ti.xcom_pull(task_ids="build_training_dataset")
    if df is None or df.empty:
        raise ValueError("Training dataset is missing or empty.")

    result: TrainingResult = train_fraud_model(df, target_column="is_fraud_label")
    logging.info(
        "Trained fraud model: run_id=%s roc_auc=%.4f accuracy=%.4f",
        result.run_id,
        result.roc_auc,
        result.accuracy,
    )
    # Return a small payload for downstream tasks.
    return {
        "run_id": result.run_id,
        "roc_auc": result.roc_auc,
        "accuracy": result.accuracy,
    }


def _evaluate_against_baseline(**context: Any) -> Dict[str, Any]:
    """
    Compare training metrics against a simple static baseline.

    In a full implementation this would look up historical performance from
    MLflow or a metrics store. For Phase 5 we use static thresholds.
    """
    ti = context["ti"]
    metrics: Dict[str, Any] = ti.xcom_pull(task_ids="train_model") or {}

    roc_auc = float(metrics.get("roc_auc", 0.0))
    accuracy = float(metrics.get("accuracy", 0.0))

    baseline_roc_auc = 0.75
    baseline_accuracy = 0.80

    is_better = roc_auc >= baseline_roc_auc and accuracy >= baseline_accuracy
    logging.info(
        "Evaluation against baseline: roc_auc=%.4f (>= %.2f?) accuracy=%.4f (>= %.2f?) -> %s",
        roc_auc,
        baseline_roc_auc,
        accuracy,
        baseline_accuracy,
        "ACCEPT" if is_better else "REJECT",
    )

    return {
        "run_id": metrics.get("run_id"),
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "accepted": is_better,
    }


def _register_and_mark_candidate(**context: Any) -> None:
    """
    Register the trained model run in MLflow Model Registry and transition it
    to Staging for canary. Only runs when the candidate passed baseline evaluation.
    """
    ti = context["ti"]
    evaluation: Dict[str, Any] = ti.xcom_pull(task_ids="evaluate_against_baseline") or {}
    if not evaluation.get("accepted"):
        logging.info(
            "Model run_id=%s did not pass baseline; skipping registration.",
            evaluation.get("run_id"),
        )
        return

    run_id = evaluation.get("run_id")
    if not run_id:
        raise ValueError("evaluate_against_baseline did not return run_id.")

    cfg = get_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    model_name = _registered_model_name()
    model_uri = f"runs:/{run_id}/model"

    client = MlflowClient()
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = int(mv.version)
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage="Staging",
    )
    logging.info(
        "Registered model run_id=%s as %s version %s and transitioned to Staging.",
        run_id,
        model_name,
        version,
    )


def _deploy_canary(**context: Any) -> None:
    """
    Stub for canary deployment of the trained model.
    """
    ti = context["ti"]
    evaluation: Dict[str, Any] = ti.xcom_pull(task_ids="evaluate_against_baseline") or {}
    if not evaluation.get("accepted"):
        logging.info("Skipping canary deployment because candidate model was rejected.")
        return

    logging.info(
        "Would deploy canary for model run_id=%s to BentoML staging environment.",
        evaluation.get("run_id"),
    )


def _monitor_canary(**_: Any) -> None:
    """
    Stub for canary monitoring over a fixed window.
    """
    logging.info(
        "Canary monitoring stub executed; in a real setup this would track metrics "
        "for ~24h and decide whether to promote or rollback."
    )


def _finalize_promotion(**context: Any) -> None:
    """
    Stub for finalizing promotion decision.
    """
    ti = context["ti"]
    evaluation: Dict[str, Any] = ti.xcom_pull(task_ids="evaluate_against_baseline") or {}
    if evaluation.get("accepted"):
        logging.info(
            "Model run_id=%s would be promoted to 'Production' after successful canary.",
            evaluation.get("run_id"),
        )
    else:
        logging.info("Model candidate was not accepted; keeping existing production model.")


def _notify_bentoml_reload(**context: Any) -> None:
    """
    Call BentoML fraud service POST /admin_reload to hot-reload the new model.
    Only runs when the candidate model was accepted.
    """
    import urllib.request

    ti = context["ti"]
    evaluation: Dict[str, Any] = ti.xcom_pull(task_ids="evaluate_against_baseline") or {}
    if not evaluation.get("accepted"):
        logging.info("Model not accepted; skipping BentoML reload.")
        return

    url = f"{_bentoml_base_url()}/admin_reload"
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            data = json.loads(body) if body else {}
        status = data.get("status", "")
        logging.info("BentoML reload response: %s", data)
        if status != "reloaded":
            raise RuntimeError(f"BentoML reload returned status={status}: {data.get('detail', '')}")
    except Exception as e:
        logging.exception("Failed to call admin_reload at %s: %s", url, e)
        raise


default_args = {
    "owner": "ml-platform",
    "retries": get_retries_for_dag(DAG_ID, 1),
    "retry_delay": timedelta(minutes=10),
}


with DAG(
    dag_id=DAG_ID,
    description="Weekly fraud model training and evaluation pipeline (Phase 5).",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "training", "phase5"],
) as dag:
    start = EmptyOperator(task_id="start")

    build_training_dataset = PythonOperator(
        task_id="build_training_dataset",
        python_callable=_build_training_dataset,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        provide_context=True,
    )

    evaluate_against_baseline = PythonOperator(
        task_id="evaluate_against_baseline",
        python_callable=_evaluate_against_baseline,
        provide_context=True,
    )

    register_and_mark_candidate = PythonOperator(
        task_id="register_and_mark_candidate",
        python_callable=_register_and_mark_candidate,
        provide_context=True,
    )

    deploy_canary = PythonOperator(
        task_id="deploy_canary",
        python_callable=_deploy_canary,
        provide_context=True,
    )

    monitor_canary = PythonOperator(
        task_id="monitor_canary",
        python_callable=_monitor_canary,
    )

    finalize_promotion = PythonOperator(
        task_id="finalize_promotion",
        python_callable=_finalize_promotion,
        provide_context=True,
    )

    notify_bentoml_reload = PythonOperator(
        task_id="notify_bentoml_reload",
        python_callable=_notify_bentoml_reload,
        provide_context=True,
    )

    end = EmptyOperator(task_id="end")

    (
        start
        >> build_training_dataset
        >> train_model
        >> evaluate_against_baseline
        >> register_and_mark_candidate
        >> deploy_canary
        >> monitor_canary
        >> finalize_promotion
        >> notify_bentoml_reload
        >> end
    )


__all__ = ["dag"]

