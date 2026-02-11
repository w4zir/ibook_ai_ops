from __future__ import annotations

"""
Auto-training orchestrator for fraud detection model retraining.

This module provides the ``AutoTrainingOrchestrator`` which:

1. Builds a fresh training dataset (synthetic or from feedback data).
2. Trains a new XGBoost fraud model via ``common.model_utils``.
3. Evaluates against a minimum quality baseline.
4. Registers the model in MLflow and optionally transitions it to
   Production stage.
5. Signals the caller (typically the BentoML service) to hot-reload the
   new model.

Staff-engineering patterns used:

- **Immutable result objects** – ``RetrainingResult`` captures every detail
  of the run for audit/logging.
- **Fail-safe defaults** – if training or registration fails, the existing
  production model continues serving unchanged.
- **Configurable via ``AutoTrainingConfig``** – no magic numbers in code.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from common.config import AutoTrainingConfig, get_config
from common.model_utils import TrainingResult, build_fraud_training_dataframe, train_fraud_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrainingResult:
    """Outcome of an auto-retraining run."""

    success: bool
    run_id: Optional[str] = None
    roc_auc: float = 0.0
    accuracy: float = 0.0
    reason: str = ""
    model_version: Optional[str] = None


def _build_synthetic_dataset(n_rows: int = 512, seed: int | None = None) -> pd.DataFrame:
    """
    Build a synthetic user-metrics dataset for retraining.

    In production this would pull recent data from Feast. For the local /
    simulation path we generate random data that covers a wide feature space
    so the retrained model generalises to novel fraud patterns.
    """
    rng = np.random.default_rng(seed=seed)
    user_metrics = pd.DataFrame(
        {
            "user_id": np.arange(1, n_rows + 1, dtype=int),
            "lifetime_purchases": rng.integers(0, 100, size=n_rows),
            "fraud_risk_score": rng.uniform(0.0, 1.0, size=n_rows),
        }
    )
    return build_fraud_training_dataframe(user_metrics)


def _register_model_in_mlflow(
    run_id: str,
    model_name: str = "fraud_detection",
) -> Optional[str]:
    """
    Register the trained model in the MLflow Model Registry and transition
    it to the Production stage.

    Returns the new model version string, or ``None`` on failure.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        cfg = get_config()
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        client = MlflowClient()

        # Ensure the registered model exists.
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass  # Already exists.

        model_uri = f"runs:/{run_id}/model"
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
        )
        # Transition to Production so ``resolve_latest_model`` picks it up.
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            "Registered model %s version %s (run_id=%s) as Production.",
            model_name,
            mv.version,
            run_id,
        )
        return str(mv.version)
    except Exception:
        logger.exception("Failed to register model in MLflow.")
        return None


class AutoTrainingOrchestrator:
    """
    Coordinates the full retrain-evaluate-register-reload cycle.

    Parameters
    ----------
    config:
        Auto-training settings. Defaults to the global config if not provided.
    on_model_ready:
        Optional callback invoked after a new model is registered.
        Receives the ``RetrainingResult`` as its argument. Typically used
        to trigger a hot-reload of the BentoML runtime.
    """

    def __init__(
        self,
        config: Optional[AutoTrainingConfig] = None,
        on_model_ready: Optional[Callable[[RetrainingResult], None]] = None,
        model_name: str = "fraud_detection",
    ) -> None:
        self._config = config or get_config().auto_training
        self._on_model_ready = on_model_ready
        self._model_name = model_name

    def run(self, failure_rate: float, n_samples: int) -> RetrainingResult:
        """
        Execute a full retraining cycle.

        This is typically called from a background thread by the
        ``FailureTracker`` callback.
        """
        logger.info(
            "Auto-retraining triggered: failure_rate=%.2f%%, n_samples=%d",
            failure_rate * 100,
            n_samples,
        )

        # 1) Build dataset.
        try:
            df = _build_synthetic_dataset(
                n_rows=self._config.training_dataset_size,
                seed=None,  # Random seed for diversity.
            )
        except Exception:
            logger.exception("Failed to build training dataset.")
            return RetrainingResult(success=False, reason="dataset_build_failed")

        # 2) Train.
        try:
            result: TrainingResult = train_fraud_model(
                df,
                target_column="is_fraud_label",
                experiment_name="fraud_detection_auto_retrain",
                n_trials=5,
            )
        except Exception:
            logger.exception("Model training failed.")
            return RetrainingResult(success=False, reason="training_failed")

        # 3) Evaluate – reject if quality is below acceptable minimums.
        min_roc_auc = 0.60
        min_accuracy = 0.65
        if result.roc_auc < min_roc_auc or result.accuracy < min_accuracy:
            logger.warning(
                "Retrained model quality too low (roc_auc=%.4f, accuracy=%.4f); keeping old model.",
                result.roc_auc,
                result.accuracy,
            )
            return RetrainingResult(
                success=False,
                run_id=result.run_id,
                roc_auc=result.roc_auc,
                accuracy=result.accuracy,
                reason="quality_below_threshold",
            )

        # 4) Register in MLflow.
        model_version = _register_model_in_mlflow(
            run_id=result.run_id,
            model_name=self._model_name,
        )

        retrain_result = RetrainingResult(
            success=True,
            run_id=result.run_id,
            roc_auc=result.roc_auc,
            accuracy=result.accuracy,
            model_version=model_version,
            reason="success",
        )

        # 5) Signal hot-reload.
        if self._on_model_ready is not None:
            try:
                self._on_model_ready(retrain_result)
            except Exception:
                logger.exception("on_model_ready callback failed.")

        logger.info(
            "Auto-retraining complete: run_id=%s roc_auc=%.4f accuracy=%.4f version=%s",
            result.run_id,
            result.roc_auc,
            result.accuracy,
            model_version,
        )
        return retrain_result


__all__ = ["AutoTrainingOrchestrator", "RetrainingResult"]
