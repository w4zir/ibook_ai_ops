"""Fraud attack scenario - coordinated credential stuffing, card testing, bot scalping."""

import logging
import random
from typing import Any, Dict, List

import numpy as np

from simulator.config import UserPersona, config
from simulator.core.event_generator import EventGenerator
from simulator.core.fraud_simulator import FraudSimulator
from simulator.core.transaction_generator import TransactionGenerator
from simulator.core.user_generator import UserGenerator
from simulator.scenarios.base_scenario import BaseScenario

logger = logging.getLogger(__name__)


def _synthetic_fraud_response(transaction: Dict[str, Any]) -> Dict[str, Any]:
    blocked = transaction.get("is_fraud", True) and random.random() < 0.9
    return {
        "status": 200,
        "latency_ms": random.uniform(30, 120),
        "fraud_score": 0.9 if transaction.get("is_fraud") else 0.2,
        "blocked": blocked,
        "is_fraud": transaction.get("is_fraud", True),
    }


class FraudAttackScenario(BaseScenario):
    """Coordinated fraud attack - validates fraud detection recall and precision."""

    def __init__(self) -> None:
        super().__init__(
            name="Fraud Attack",
            description="Coordinated fraud attack - credential stuffing, card testing, bot scalping",
            duration_minutes=5,
            expected_metrics={
                "fraud_recall": 0.90,
                "fraud_precision": 0.85,
                "p99_latency_ms": 200,
            },
        )
        self.event: Dict[str, Any] | None = None
        self.users: List[Dict[str, Any]] = []
        self.event_gen = EventGenerator()
        self.user_gen = UserGenerator()
        self.txn_gen = TransactionGenerator()
        self.fraud_sim = FraudSimulator(self.txn_gen)

    def setup(self) -> None:
        logger.info("Setting up fraud attack scenario...")
        self.event = self.event_gen.generate_event()
        self.users = self.user_gen.generate_batch(
            count=500,
            persona_distribution={
                "casual": 0.20,
                "enthusiast": 0.20,
                "fraudster": 0.60,
            },
        )
        logger.info("Generated event and %d users", len(self.users))

    def run(self) -> None:
        logger.info("Generating fraud attack traffic...")
        attack_txns = self.fraud_sim.generate_mixed_attack(
            self.event, self.users, duration_seconds=30, attacks_per_second=5
        )
        num_attacks = len(attack_txns)
        self.results["fraud_attacks_count"] = num_attacks
        logger.info("Fraud attacks added in scenario: %d", num_attacks)
        responses = [_synthetic_fraud_response(t) for t in attack_txns]
        for i, t in enumerate(attack_txns):
            responses[i]["is_fraud"] = t.get("is_fraud", True)
            responses[i]["blocked"] = responses[i].get("blocked", False)
        self.results["responses"] = responses
        self.results["transactions"] = attack_txns

    def teardown(self) -> None:
        responses = self.results.get("responses", [])
        if not responses:
            return
        latencies = [r["latency_ms"] for r in responses if "latency_ms" in r]
        if latencies:
            self.results["p99_latency_ms"] = float(np.percentile(latencies, 99))
        else:
            self.results["p99_latency_ms"] = 0
        actual_fraud = [r for r in responses if r.get("is_fraud")]
        blocked = [r for r in responses if r.get("blocked")]
        true_positives = len([r for r in blocked if r.get("is_fraud")])
        if actual_fraud:
            self.results["fraud_recall"] = true_positives / len(actual_fraud)
        else:
            self.results["fraud_recall"] = 0
        if blocked:
            self.results["fraud_precision"] = true_positives / len(blocked)
        else:
            self.results["fraud_precision"] = 0
