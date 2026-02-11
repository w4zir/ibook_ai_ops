"""Tests for RealtimeRunner - short duration and output structure."""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from simulator.runners.realtime_runner import RealtimeRunner
from simulator.scenarios.normal_traffic import NormalTrafficScenario


def test_realtime_runner_short_duration() -> None:
    """Run for a few seconds and verify transactions are generated and metrics returned."""
    runner = RealtimeRunner(
        scenario_class=NormalTrafficScenario,
        duration_seconds=2,
        rps=5,
    )
    result = runner.run()
    assert "responses" in result
    assert "duration_seconds" in result
    assert len(result["responses"]) >= 1
    assert result["duration_seconds"] >= 1.0
    assert "peak_rps" in result
    assert "error_rate" in result


def test_realtime_runner_returns_metrics() -> None:
    """Runner returns dict with latency and throughput metrics after run."""
    runner = RealtimeRunner(
        scenario_class=NormalTrafficScenario,
        duration_seconds=1,
        rps=10,
    )
    result = runner.run()
    assert "p99_latency_ms" in result or "responses" in result
    assert "peak_rps" in result
    assert "error_rate" in result
    assert isinstance(result["responses"], list)


class _FakeHTTPResponse:
    """Minimal fake HTTPResponse for urllib.request.urlopen context manager."""

    def __init__(self, body: Dict[str, Any], status: int = 200) -> None:
        self._body = json.dumps(body).encode("utf-8")
        self.status = status

    def read(self) -> bytes:  # pragma: no cover - trivial
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None


def test_send_request_builds_bentoml_payload_and_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """_send_request should call BentoML /predict with a FraudBatchRequest-like payload."""
    captured = {}

    def fake_urlopen(req, timeout: float = 5) -> _FakeHTTPResponse:  # type: ignore[override]
        nonlocal captured
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        body = req.data.decode("utf-8")
        captured["payload"] = json.loads(body)
        # Simulate a successful FraudBatchResponse.
        return _FakeHTTPResponse({"predictions": [{"fraud_score": 0.8, "is_fraud": True}]}, status=200)

    import urllib.request as urllib_request  # Local import so monkeypatch works reliably.

    monkeypatch.setattr(urllib_request, "urlopen", fake_urlopen)

    runner = RealtimeRunner(
        scenario_class=NormalTrafficScenario,
        duration_seconds=1,
        rps=1,
        fraud_api_base_url="http://localhost:7001",
    )

    txn = {
        "user_id": 123,
        "event_id": 456,
        "total_amount": 99.5,
        "is_fraud": True,
    }
    resp = runner._send_request(txn)

    assert captured["url"].endswith("/predict")
    assert captured["payload"]["requests"][0]["user_id"] == 123
    assert captured["payload"]["requests"][0]["event_id"] == 456
    assert captured["payload"]["requests"][0]["amount"] == pytest.approx(99.5)

    assert resp["status"] == 200
    assert resp["fraud_score"] == pytest.approx(0.8)
    assert resp["blocked"] is True
    assert resp["predicted_is_fraud"] is True


def test_error_counting_uses_non_2xx_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors should increment only for non-2xx HTTP status or transport failures."""

    def fake_send_request_ok(_txn: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": 200,
            "latency_ms": 10.0,
            "fraud_score": 0.1,
            "blocked": False,
        }

    def fake_send_request_error(_txn: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": 500,
            "latency_ms": 20.0,
            "fraud_score": 0.9,
            "blocked": True,
        }

    # First run: all OK responses -> error_rate should be 0.
    runner_ok = RealtimeRunner(
        scenario_class=NormalTrafficScenario,
        duration_seconds=1,
        rps=5,
    )
    monkeypatch.setattr(runner_ok, "_send_request", fake_send_request_ok)  # type: ignore[assignment]
    result_ok = runner_ok.run()
    assert result_ok["error_rate"] == pytest.approx(0.0)

    # Second run: all error responses -> error_rate should be 1.
    runner_err = RealtimeRunner(
        scenario_class=NormalTrafficScenario,
        duration_seconds=1,
        rps=5,
    )
    monkeypatch.setattr(runner_err, "_send_request", fake_send_request_error)  # type: ignore[assignment]
    result_err = runner_err.run()
    assert result_err["error_rate"] == pytest.approx(1.0)
