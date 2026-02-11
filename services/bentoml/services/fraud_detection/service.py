from __future__ import annotations

"""
BentoML service entrypoint for fraud detection.

The BentoML-specific decorators are intentionally kept thin; the core request
handling logic is implemented in pure Python functions so it can be exercised
directly from unit tests without running a HTTP server.
"""

from typing import Any, List

try:  # pragma: no cover - import guard for environments without BentoML
    import bentoml
    from bentoml.io import JSON
except Exception:  # pragma: no cover
    bentoml = None  # type: ignore[assignment]
    JSON = None  # type: ignore[assignment]

import mlflow.pyfunc
import pandas as pd

from services.bentoml.common.config import get_bentoml_settings
from services.bentoml.common.feast_client import get_fraud_features_for_entities
from services.bentoml.common.metrics import record_error, record_request, track_latency
from services.bentoml.common.mlflow_client import ResolvedModel, resolve_latest_model
from services.bentoml.services.fraud_detection.model import (
    FraudRequest,
    FraudResponse,
    FraudBatchRequest,
    FraudBatchResponse,
    FraudModelRuntime,
)


SERVICE_NAME = "fraud_detection"


def _load_model() -> FraudModelRuntime:
    settings = get_bentoml_settings()
    resolved: ResolvedModel = resolve_latest_model(settings=settings)
    model = mlflow.pyfunc.load_model(resolved.model_uri)
    # Threshold can later be made configurable if needed.
    return FraudModelRuntime(model=model, threshold=0.5)


_RUNTIME: FraudModelRuntime | None = None


def _get_runtime() -> FraudModelRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _load_model()
    return _RUNTIME


def _normalize_entity_id(raw_id: Any) -> int:
    """
    Normalize user/event identifiers into an integer key for Feast.

    Feast infers the entity key dtype from the offline store, which in this
    project uses integer IDs (see `scripts/seed-data.py`). Upstream callers,
    including the simulator, may send opaque string IDs like ``"user_ab12cd34"``.

    To keep the API flexible while satisfying Feast's INT32 expectation, we:
    - pass through integers unchanged
    - parse purely numeric strings directly (``"42"`` -> ``42``)
    - parse numeric suffixes like ``"user_123"`` -> ``123``
    - fall back to a stable hash (modulo 2**31-1) for other strings

    This guarantees that Feast always receives an integer while keeping the
    mapping stable within a process.
    """
    if isinstance(raw_id, int):
        return raw_id

    if isinstance(raw_id, str):
        s = raw_id.strip()
        if s.isdigit():
            return int(s)

        # Common pattern: "user_123" / "event_456" – try to parse the suffix.
        if "_" in s:
            suffix = s.split("_")[-1]
            if suffix.isdigit():
                return int(suffix)
            # If the suffix looks like hex, use that as a stable numeric key.
            try:
                return int(suffix, 16) % (2**31 - 1)
            except ValueError:
                pass

        # Fallback: stable hash into INT32 range.
        return abs(hash(s)) % (2**31 - 1)

    # Last-resort fallback for unexpected types.
    return abs(hash(str(raw_id))) % (2**31 - 1)


def _build_feature_frame(batch: FraudBatchRequest) -> pd.DataFrame:
    """
    Build the feature DataFrame for a batch of fraud requests.

    For now this uses Feast to look up features by (user_id, event_id) and
    applies any per-request `feature_overrides` if present.

    The Feast repo is configured with integer entity keys, but callers may send
    string identifiers. We normalize identifiers into stable integers before
    querying Feast so that both styles are accepted.
    """
    entity_rows = [
        {
            "user_id": _normalize_entity_id(r.user_id),
            "event_id": _normalize_entity_id(r.event_id),
        }
        for r in batch.requests
    ]
    features_df = get_fraud_features_for_entities(entity_rows)

    # Ensure that the index is positional (0..N-1) so we can safely map
    # per-request overrides onto the correct rows even if Feast returns a
    # DataFrame with a non-default index.
    features_df = features_df.reset_index(drop=True)

    # Apply overrides column-wise, guarding against missing rows/columns.
    for idx, req in enumerate(batch.requests):
        if not req.feature_overrides:
            continue
        if idx >= len(features_df.index):
            # Mismatch between requested entities and returned features; skip
            # this override but keep processing others.
            continue
        for name, value in req.feature_overrides.items():
            if name not in features_df.columns:
                # Unknown feature name – ignore the override rather than erroring.
                continue
            features_df.loc[idx, name] = value

    # Normalize to the stable training feature set and column ordering.
    # Online Feast lookups produce feature columns with a double-underscore
    # separator (e.g. ``\"user_purchase_behavior__lifetime_purchases\"``),
    # whereas the training pipeline used a compact DataFrame with
    # ``\"lifetime_purchases\"`` and ``\"fraud_risk_score\"``. The underlying
    # model only depends on column order, so we select and order the Feast
    # columns explicitly here.
    feast_feature_cols = [
        "user_purchase_behavior__lifetime_purchases",
        "user_purchase_behavior__fraud_risk_score",
    ]
    for col in feast_feature_cols:
        if col not in features_df.columns:
            # Backfill missing features with zeros to keep the input shape
            # stable. This is a conservative default that avoids runtime
            # failures when a feature view is temporarily missing.
            features_df[col] = 0.0

    features_df = features_df[feast_feature_cols]
    return features_df


def handle_predict(batch: FraudBatchRequest) -> FraudBatchResponse:
    """
    Core prediction logic, independent of BentoML's IO system.
    """
    runtime = _get_runtime()
    features_df = _build_feature_frame(batch)
    predictions = runtime.predict_batch(features_df)
    return FraudBatchResponse(predictions=predictions)


def _health_payload(ok: bool, detail: str) -> dict:
    return {"ok": ok, "detail": detail}


def handle_healthcheck() -> dict:
    """
    Simple healthcheck: checks that the model can be loaded.
    """
    try:
        _ = _get_runtime()
        return _health_payload(True, "model_loaded")
    except Exception as exc:  # pragma: no cover - defensive
        return _health_payload(False, f"error: {exc}")


svc = None
if bentoml is not None and JSON is not None:  # pragma: no cover - HTTP layer
    # --- BentoML service definition -------------------------------------------
    svc = bentoml.Service(SERVICE_NAME)

    @svc.api(input=JSON(pydantic_model=FraudBatchRequest), output=JSON(pydantic_model=FraudBatchResponse))
    def predict(batch: FraudBatchRequest) -> FraudBatchResponse:
        endpoint = "/predict"
        with track_latency(SERVICE_NAME, endpoint):
            try:
                response = handle_predict(batch)
            except Exception as exc:
                # Record metrics and re-raise so BentoML returns a 5xx with a
                # detailed stack trace in the server logs.
                record_error(SERVICE_NAME, endpoint)
                record_request(SERVICE_NAME, endpoint, http_status=500)
                # Attach a short context note to aid debugging without leaking
                # internal implementation details to clients.
                raise RuntimeError(f"fraud_detection.predict failed: {exc}") from exc
            else:
                record_request(SERVICE_NAME, endpoint, http_status=200)
                return response

    @svc.api(input=JSON(), output=JSON())
    def health(_: dict) -> dict:
        endpoint = "/healthz"
        with track_latency(SERVICE_NAME, endpoint):
            payload = handle_healthcheck()
            status = 200 if payload.get("ok") else 500
            record_request(SERVICE_NAME, endpoint, http_status=status)
            return payload

