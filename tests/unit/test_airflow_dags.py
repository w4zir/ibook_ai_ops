from __future__ import annotations

import pytest


airflow = pytest.importorskip("airflow")
from services.airflow.dags.feature_engineering_pipeline import dag as feature_dag  # noqa: E402
from services.airflow.dags.model_training_pipeline import dag as training_dag  # noqa: E402
from services.airflow.dags.ml_monitoring_pipeline import dag as monitoring_dag  # noqa: E402


def test_feature_engineering_dag_shape() -> None:
    dag = feature_dag
    assert dag.schedule_interval == "@hourly"
    assert dag.catchup is False

    task_ids = {t.task_id for t in dag.tasks}
    for expected in [
        "start",
        "extract_realtime_events",
        "compute_batch_features",
        "validate_features",
        "materialize_to_feast",
        "check_for_drift",
        "maybe_trigger_training",
        "end",
    ]:
        assert expected in task_ids, f"Missing task_id={expected!r} in feature DAG"


def test_model_training_dag_shape() -> None:
    dag = training_dag
    assert dag.schedule_interval == "@weekly"
    assert dag.catchup is False

    task_ids = {t.task_id for t in dag.tasks}
    for expected in [
        "start",
        "build_training_dataset",
        "train_model",
        "evaluate_against_baseline",
        "register_and_mark_candidate",
        "deploy_canary",
        "monitor_canary",
        "finalize_promotion",
        "end",
    ]:
        assert expected in task_ids, f"Missing task_id={expected!r} in training DAG"


def test_ml_monitoring_dag_shape() -> None:
    dag = monitoring_dag
    assert dag.schedule_interval == "@daily"
    assert dag.catchup is False

    task_ids = {t.task_id for t in dag.tasks}
    for expected in [
        "start",
        "collect_production_metrics",
        "compute_drift_reports",
        "check_thresholds",
        "send_alerts",
        "branch_trigger_retraining",
        "trigger_retraining",
        "skip_retraining",
        "end",
    ]:
        assert expected in task_ids, f"Missing task_id={expected!r} in monitoring DAG"


def test_ml_monitoring_branch_returns_trigger_or_skip() -> None:
    """Branch chooses trigger_retraining when needs_retrain else skip_retraining."""
    from services.airflow.dags.ml_monitoring_pipeline import _branch_trigger_retraining

    class FakeTI:
        def __init__(self, decision: dict) -> None:
            self._decision = decision
        def xcom_pull(self, task_ids: str, key: str | None = None) -> dict:
            return self._decision

    out = _branch_trigger_retraining(ti=FakeTI({"needs_retrain": True}))
    assert out == "trigger_retraining"
    out = _branch_trigger_retraining(ti=FakeTI({"needs_retrain": False}))
    assert out == "skip_retraining"


def test_register_and_mark_candidate_skips_when_not_accepted() -> None:
    """When evaluation is not accepted, no MLflow registration or transition is performed."""
    import unittest.mock as mock
    from services.airflow.dags.model_training_pipeline import _register_and_mark_candidate

    class FakeTI:
        def xcom_pull(self, task_ids: str, **kwargs: object) -> dict:
            return {"accepted": False, "run_id": "run-123"}

    with mock.patch("services.airflow.dags.model_training_pipeline.mlflow") as mlflow_mock:
        with mock.patch("services.airflow.dags.model_training_pipeline.get_config") as get_cfg:
            get_cfg.return_value = mock.MagicMock(mlflow=mock.MagicMock(tracking_uri="http://mlflow:5000"))
            _register_and_mark_candidate(ti=FakeTI())
        mlflow_mock.register_model.assert_not_called()


def test_register_and_mark_candidate_registers_and_transitions_to_staging_when_accepted() -> None:
    """When accepted, run is registered in MLflow and version is transitioned to Staging."""
    import unittest.mock as mock
    from services.airflow.dags.model_training_pipeline import _register_and_mark_candidate

    class FakeTI:
        def xcom_pull(self, task_ids: str, **kwargs: object) -> dict:
            return {"accepted": True, "run_id": "run-456"}

    fake_mv = mock.MagicMock()
    fake_mv.version = "2"

    with mock.patch("services.airflow.dags.model_training_pipeline.get_config") as get_cfg:
        get_cfg.return_value = mock.MagicMock(mlflow=mock.MagicMock(tracking_uri="http://mlflow:5000"))
        with mock.patch("services.airflow.dags.model_training_pipeline.mlflow") as mlflow_mock:
            mlflow_mock.register_model.return_value = fake_mv
            with mock.patch("services.airflow.dags.model_training_pipeline.MlflowClient") as client_cls:
                mock_client = mock.MagicMock()
                client_cls.return_value = mock_client
                _register_and_mark_candidate(ti=FakeTI())

    mlflow_mock.set_tracking_uri.assert_called_once_with("http://mlflow:5000")
    mlflow_mock.register_model.assert_called_once_with(
        model_uri="runs:/run-456/model",
        name="fraud_detection",
    )
    mock_client.transition_model_version_stage.assert_called_once_with(
        name="fraud_detection",
        version="2",
        stage="Staging",
    )


def test_feature_pipeline_drift_uses_seed_reference() -> None:
    """Drift check uses seed-derived reference (no reference path; _build_seed_reference_features)."""
    from services.airflow.dags.feature_engineering_pipeline import (
        _get_feature_paths,
        _build_seed_reference_features,
    )

    paths = _get_feature_paths()
    assert len(paths) == 2, "Drift uses current + summary paths only; reference is regenerated from seed"
    current_path, summary_path = paths
    assert "user_realtime_features" in str(current_path)
    assert "drift_summary" in str(summary_path)

    ref_df = _build_seed_reference_features()
    assert ref_df is not None and not ref_df.empty
    for col in ("user_txn_count_1h", "user_txn_amount_1h", "user_distinct_events_1h", "user_avg_amount_24h"):
        assert col in ref_df.columns, f"Seed reference must include {col}"

