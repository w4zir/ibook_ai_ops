"""System degradation scenario - partial failures, circuit breakers, fallback."""

import logging
import random
from typing import Any, Dict, List

import numpy as np

from simulator.config import config
from simulator.core.event_generator import EventGenerator
from simulator.core.transaction_generator import TransactionGenerator
from simulator.core.user_generator import UserGenerator
from simulator.scenarios.base_scenario import BaseScenario

logger = logging.getLogger(__name__)


def _degraded_response(transaction: Dict[str, Any], fail_rate: float) -> Dict[str, Any]:
    if random.random() < fail_rate:
        return {
            "status": 503,
            "latency_ms": random.uniform(500, 3000),
            "error": "timeout",
            "timed_out": True,
            "blocked": False,
            "is_fraud": transaction.get("is_fraud", False),
        }
    return {
        "status": 200,
        "latency_ms": random.uniform(100, 400),
        "fraud_score": 0.5,
        "blocked": False,
        "is_fraud": transaction.get("is_fraud", False),
    }


class SystemDegradationScenario(BaseScenario):
    """Partial service failures - Redis slow, model timeout; validates fallbacks."""

    DEFAULT_DURATION_MINUTES = 5

    def __init__(self, duration_minutes: int | None = None) -> None:
        mins = duration_minutes if duration_minutes is not None else self.DEFAULT_DURATION_MINUTES
        super().__init__(
            name="System Degradation",
            description="Partial failures - circuit breakers and graceful degradation",
            duration_minutes=mins,
            expected_metrics={
                "error_rate": 0.05,
                "fallback_used_pct": 50,
            },
        )
        self.events: List[Dict[str, Any]] = []
        self.users: List[Dict[str, Any]] = []
        self.event_gen = EventGenerator()
        self.user_gen = UserGenerator()
        self.txn_gen = TransactionGenerator()

    def setup(self) -> None:
        logger.info("Setting up system degradation scenario...")
        self.events = self.event_gen.generate_batch(count=5)
        self.users = self.user_gen.generate_batch(count=300)

    def run(self) -> None:
        logger.info("Generating traffic with simulated degradation...")
        effective = self.get_effective_duration_minutes()
        scale = effective / self.DEFAULT_DURATION_MINUTES
        count = max(50, int(200 * scale))
        transactions = self.txn_gen.generate_batch(
            self.events, self.users, count=count, time_range_hours=max(1, int(scale))
        )
        fail_rate = 0.05
        responses = [_degraded_response(t, fail_rate) for t in transactions]
        for i, t in enumerate(transactions):
            responses[i]["is_fraud"] = t.get("is_fraud", False)
        self.results["responses"] = responses
        self.results["transactions"] = transactions

    def teardown(self) -> None:
        responses = self.results.get("responses", [])
        if not responses:
            return
        errors = [r for r in responses if r.get("status", 200) >= 400]
        self.results["error_rate"] = len(errors) / len(responses)
        fallbacks = [r for r in responses if r.get("latency_ms", 0) > 200]
        self.results["fallback_used_pct"] = (len(fallbacks) / len(responses)) * 100
        timeouts = [
            r
            for r in responses
            if r.get("timed_out") or r.get("error") == "timeout"
        ]
        self.results["timeout_count"] = len(timeouts)
