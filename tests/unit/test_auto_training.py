"""Tests for the auto-training orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from common.auto_training import (
    AutoTrainingOrchestrator,
    RetrainingResult,
    _build_synthetic_dataset,
)
from common.config import AutoTrainingConfig


# ---------------------------------------------------------------------------
# Synthetic dataset builder
# ---------------------------------------------------------------------------


class TestBuildSyntheticDataset:
    def test_default_shape(self) -> None:
        df = _build_synthetic_dataset(n_rows=64, seed=42)
        assert len(df) == 64
        assert "is_fraud_label" in df.columns
        assert "lifetime_purchases" in df.columns
        assert "fraud_risk_score" in df.columns

    def test_deterministic_with_seed(self) -> None:
        df1 = _build_synthetic_dataset(n_rows=32, seed=123)
        df2 = _build_synthetic_dataset(n_rows=32, seed=123)
        assert df1.equals(df2)

    def test_different_seeds_produce_different_data(self) -> None:
        df1 = _build_synthetic_dataset(n_rows=32, seed=1)
        df2 = _build_synthetic_dataset(n_rows=32, seed=2)
        assert not df1.equals(df2)


# ---------------------------------------------------------------------------
# RetrainingResult
# ---------------------------------------------------------------------------


class TestRetrainingResult:
    def test_success_result(self) -> None:
        r = RetrainingResult(success=True, run_id="abc", roc_auc=0.95, accuracy=0.90, reason="success")
        assert r.success is True
        assert r.run_id == "abc"

    def test_failure_result(self) -> None:
        r = RetrainingResult(success=False, reason="training_failed")
        assert r.success is False
        assert r.run_id is None


# ---------------------------------------------------------------------------
# AutoTrainingOrchestrator
# ---------------------------------------------------------------------------


class TestAutoTrainingOrchestrator:
    @pytest.fixture
    def config(self) -> AutoTrainingConfig:
        return AutoTrainingConfig(
            enabled=True,
            failure_rate_threshold=0.4,
            monitoring_window_seconds=60,
            cooldown_seconds=0,
            min_samples=5,
            training_dataset_size=64,
        )

    @patch("common.auto_training._register_model_in_mlflow")
    @patch("common.auto_training.train_fraud_model")
    def test_successful_retrain(
        self,
        mock_train: MagicMock,
        mock_register: MagicMock,
        config: AutoTrainingConfig,
    ) -> None:
        from common.model_utils import TrainingResult

        import xgboost as xgb

        dummy_model = xgb.XGBClassifier(n_estimators=2, max_depth=2)
        mock_train.return_value = TrainingResult(
            model=dummy_model,
            feature_names=["lifetime_purchases", "fraud_risk_score"],
            best_params={"max_depth": 2},
            roc_auc=0.92,
            accuracy=0.88,
            run_id="run_123",
        )
        mock_register.return_value = "5"

        on_model_ready = MagicMock()
        orch = AutoTrainingOrchestrator(
            config=config,
            on_model_ready=on_model_ready,
        )
        result = orch.run(failure_rate=0.5, n_samples=100)

        assert result.success is True
        assert result.run_id == "run_123"
        assert result.roc_auc == 0.92
        assert result.model_version == "5"
        on_model_ready.assert_called_once()

    @patch("common.auto_training._register_model_in_mlflow")
    @patch("common.auto_training.train_fraud_model")
    def test_low_quality_model_rejected(
        self,
        mock_train: MagicMock,
        mock_register: MagicMock,
        config: AutoTrainingConfig,
    ) -> None:
        from common.model_utils import TrainingResult

        import xgboost as xgb

        dummy_model = xgb.XGBClassifier(n_estimators=2, max_depth=2)
        mock_train.return_value = TrainingResult(
            model=dummy_model,
            feature_names=["lifetime_purchases", "fraud_risk_score"],
            best_params={"max_depth": 2},
            roc_auc=0.45,  # Below minimum.
            accuracy=0.50,
            run_id="run_bad",
        )
        on_model_ready = MagicMock()
        orch = AutoTrainingOrchestrator(
            config=config,
            on_model_ready=on_model_ready,
        )
        result = orch.run(failure_rate=0.5, n_samples=100)

        assert result.success is False
        assert result.reason == "quality_below_threshold"
        mock_register.assert_not_called()
        on_model_ready.assert_not_called()

    @patch("common.auto_training._build_synthetic_dataset", side_effect=RuntimeError("data error"))
    def test_dataset_failure_handled(
        self,
        mock_build: MagicMock,
        config: AutoTrainingConfig,
    ) -> None:
        orch = AutoTrainingOrchestrator(config=config)
        result = orch.run(failure_rate=0.5, n_samples=100)

        assert result.success is False
        assert result.reason == "dataset_build_failed"

    @patch("common.auto_training.train_fraud_model", side_effect=RuntimeError("train error"))
    def test_training_failure_handled(
        self,
        mock_train: MagicMock,
        config: AutoTrainingConfig,
    ) -> None:
        orch = AutoTrainingOrchestrator(config=config)
        result = orch.run(failure_rate=0.5, n_samples=100)

        assert result.success is False
        assert result.reason == "training_failed"
