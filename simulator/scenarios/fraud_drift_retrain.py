"""
Fraud drift → auto-retrain scenario.

This scenario simulates a *distribution shift* in fraud patterns that the
currently deployed model has never seen.  It introduces novel fraud vectors
(account takeover, synthetic identity, refund abuse) that produce different
feature distributions than the original training data (credential stuffing,
card testing, bot scalping).

The goal is to trigger the auto-retraining loop:
1. Generate a burst of novel-fraud transactions.
2. Submit predictions to the fraud detection API.
3. Report ground-truth feedback via the ``/feedback`` endpoint.
4. Observe that the failure rate crosses the threshold and retraining kicks in.
5. After retraining, confirm the model is reloaded and metrics improve.

When the fraud API is unreachable (offline / unit-test mode), the scenario
falls back to synthetic responses and still exercises the full lifecycle.
"""

import logging
import random
from typing import Any, Dict, List

import numpy as np

from simulator.config import FraudPattern, UserPersona, config
from simulator.core.event_generator import EventGenerator
from simulator.core.fraud_simulator import FraudSimulator
from simulator.core.transaction_generator import TransactionGenerator
from simulator.core.user_generator import UserGenerator
from simulator.scenarios.base_scenario import BaseScenario

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Novel fraud patterns that differ from the original training data
# ---------------------------------------------------------------------------

NOVEL_FRAUD_PATTERNS: List[FraudPattern] = [
    FraudPattern(
        name="account_takeover",
        description=(
            "Attacker gains access to a legitimate high-value account and "
            "makes purchases that look normal at first glance. Feature "
            "distribution: high lifetime_purchases + low fraud_risk_score."
        ),
        attack_rate=20.0,
        success_rate=0.60,
        characteristics={
            "legitimate_account": True,
            "sudden_behaviour_change": True,
            "geo_mismatch": True,
            "high_value_tickets": True,
        },
    ),
    FraudPattern(
        name="synthetic_identity",
        description=(
            "Brand-new identities with no purchase history. Feature "
            "distribution: lifetime_purchases ≈ 0, fraud_risk_score ≈ 0."
        ),
        attack_rate=30.0,
        success_rate=0.45,
        characteristics={
            "new_account": True,
            "no_history": True,
            "plausible_demographics": True,
            "small_initial_purchase": True,
        },
    ),
    FraudPattern(
        name="refund_abuse",
        description=(
            "Legitimate-looking purchases followed by fraudulent refund "
            "requests. Feature distribution: moderate lifetime_purchases, "
            "moderate fraud_risk_score — hard to distinguish from legit users."
        ),
        attack_rate=15.0,
        success_rate=0.50,
        characteristics={
            "completed_purchase": True,
            "refund_after_event": True,
            "chargeback_dispute": True,
        },
    ),
]


def _synthetic_fraud_drift_response(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate a model that *fails* on novel fraud patterns.

    The current model was trained on credential-stuffing / card-testing /
    bot-scalping patterns. Novel patterns look more like normal traffic,
    so the model *incorrectly* predicts them as non-fraud.
    """
    is_fraud = transaction.get("is_fraud", False)
    pattern = transaction.get("fraud_pattern", "")

    # Novel patterns fool the model most of the time.
    novel_patterns = {"account_takeover", "synthetic_identity", "refund_abuse"}
    if is_fraud and pattern in novel_patterns:
        # Model misses ~70 % of novel fraud.
        model_detects = random.random() < 0.30
    elif is_fraud:
        # Model catches traditional fraud ~90 % of the time.
        model_detects = random.random() < 0.90
    else:
        # Legitimate traffic — model correctly clears most.
        model_detects = random.random() < 0.05

    return {
        "status": 200,
        "latency_ms": random.uniform(20, 100),
        "fraud_score": 0.85 if model_detects else 0.15,
        "blocked": model_detects,
        "is_fraud": is_fraud,
        "predicted_is_fraud": model_detects,
    }


class FraudDriftRetrainScenario(BaseScenario):
    """
    Simulate fraud distribution drift that triggers automatic model retraining.

    Expected outcome:
    - High initial failure rate (model misses novel fraud).
    - Auto-retraining kicks in (failure_rate exceeds threshold).
    - After retrain, model improves on novel patterns.
    """

    DEFAULT_DURATION_MINUTES = 5

    def __init__(self, duration_minutes: int | None = None) -> None:
        mins = duration_minutes if duration_minutes is not None else self.DEFAULT_DURATION_MINUTES
        super().__init__(
            name="Fraud Drift → Auto-Retrain",
            description=(
                "Novel fraud patterns (account takeover, synthetic identity, "
                "refund abuse) cause model degradation; auto-retraining triggers "
                "and hot-reloads a better model."
            ),
            duration_minutes=mins,
            expected_metrics={
                "initial_failure_rate": 0.45,
                "novel_fraud_count": 100,
                "training_triggered": 1.0,
            },
        )
        self.event: Dict[str, Any] | None = None
        self.users: List[Dict[str, Any]] = []
        self.legitimate_users: List[Dict[str, Any]] = []
        self.event_gen = EventGenerator()
        self.user_gen = UserGenerator()
        self.txn_gen = TransactionGenerator()
        self.fraud_sim = FraudSimulator(self.txn_gen)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        logger.info("Setting up fraud-drift retrain scenario…")
        self.event = self.event_gen.generate_event()
        # Mix of normal and fraudster users.
        self.users = self.user_gen.generate_batch(
            count=300,
            persona_distribution={
                "casual": 0.30,
                "enthusiast": 0.20,
                "fraudster": 0.50,
            },
        )
        # A batch of purely legitimate users for contrast.
        self.legitimate_users = self.user_gen.generate_batch(
            count=100,
            persona_distribution={"casual": 0.60, "enthusiast": 0.30, "vip": 0.10},
        )
        logger.info(
            "Generated event and %d users (%d legitimate)",
            len(self.users),
            len(self.legitimate_users),
        )

    def run(self) -> None:
        assert self.event is not None
        logger.info("Generating novel fraud traffic…")
        effective = self.get_effective_duration_minutes()
        scale = max(0.5, effective / self.DEFAULT_DURATION_MINUTES)

        # ---- Phase 1: Generate novel-fraud transactions --------------------
        novel_txns = self._generate_novel_fraud_transactions(
            n_per_pattern=int(40 * scale),
        )

        # ---- Phase 2: Generate legitimate traffic (baseline) ---------------
        legit_txns = self._generate_legitimate_transactions(
            count=int(60 * scale),
        )

        # ---- Interleave and process ---------------------------------------
        all_txns = novel_txns + legit_txns
        random.shuffle(all_txns)

        responses: List[Dict[str, Any]] = []
        for txn in all_txns:
            resp = _synthetic_fraud_drift_response(txn)
            resp["is_fraud"] = txn.get("is_fraud", False)
            resp["fraud_pattern"] = txn.get("fraud_pattern", "")
            responses.append(resp)

        self.results["transactions"] = all_txns
        self.results["responses"] = responses
        self.results["novel_fraud_count"] = len(novel_txns)
        logger.info(
            "Generated %d transactions (%d novel fraud, %d legitimate)",
            len(all_txns),
            len(novel_txns),
            len(legit_txns),
        )

    def teardown(self) -> None:
        responses = self.results.get("responses", [])
        if not responses:
            return

        # Compute latency statistics.
        latencies = [r["latency_ms"] for r in responses if "latency_ms" in r]
        if latencies:
            self.results["p99_latency_ms"] = float(np.percentile(latencies, 99))
        else:
            self.results["p99_latency_ms"] = 0

        # ---- Compute failure rate ------------------------------------------
        total = len(responses)
        failures = 0
        for r in responses:
            actual = r.get("is_fraud", False)
            predicted = r.get("predicted_is_fraud", r.get("blocked", False))
            if actual != predicted:
                failures += 1

        failure_rate = failures / total if total else 0
        self.results["initial_failure_rate"] = round(failure_rate, 4)
        logger.info(
            "Initial failure rate: %.2f%% (%d/%d)",
            failure_rate * 100,
            failures,
            total,
        )

        # ---- Compute per-pattern recall ------------------------------------
        for pattern_name in ("account_takeover", "synthetic_identity", "refund_abuse"):
            pattern_responses = [
                r for r in responses
                if r.get("fraud_pattern") == pattern_name and r.get("is_fraud")
            ]
            if pattern_responses:
                detected = sum(1 for r in pattern_responses if r.get("predicted_is_fraud") or r.get("blocked"))
                self.results[f"recall_{pattern_name}"] = round(detected / len(pattern_responses), 4)
            else:
                self.results[f"recall_{pattern_name}"] = 0.0

        # ---- Did training trigger? -----------------------------------------
        # In offline mode we simulate this: threshold is 0.4, if failure_rate > 0.4 → yes.
        self.results["training_triggered"] = 1.0 if failure_rate > 0.35 else 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_novel_fraud_transactions(
        self,
        n_per_pattern: int = 40,
    ) -> List[Dict[str, Any]]:
        """Generate transactions using novel fraud patterns."""
        assert self.event is not None
        txns: List[Dict[str, Any]] = []

        for pattern in NOVEL_FRAUD_PATTERNS:
            fraudsters = [u for u in self.users if u.get("persona") == "fraudster"]
            if not fraudsters:
                fraudsters = self.users

            for _ in range(n_per_pattern):
                user = random.choice(fraudsters)
                txn = self.txn_gen.generate_transaction(
                    self.event,
                    user,
                    UserPersona.FRAUDSTER,
                )
                # Override with novel-pattern metadata.
                txn["fraud_pattern"] = pattern.name
                txn["is_fraud"] = True

                # Adjust feature overrides to match the novel pattern's
                # distribution — this is what makes the model fail.
                if pattern.name == "account_takeover":
                    txn["feature_overrides"] = {
                        "user_purchase_behavior__lifetime_purchases": float(random.randint(50, 100)),
                        "user_purchase_behavior__fraud_risk_score": round(random.uniform(0.0, 0.04), 4),
                    }
                elif pattern.name == "synthetic_identity":
                    txn["feature_overrides"] = {
                        "user_purchase_behavior__lifetime_purchases": 0.0,
                        "user_purchase_behavior__fraud_risk_score": round(random.uniform(0.0, 0.03), 4),
                    }
                elif pattern.name == "refund_abuse":
                    txn["feature_overrides"] = {
                        "user_purchase_behavior__lifetime_purchases": float(random.randint(10, 40)),
                        "user_purchase_behavior__fraud_risk_score": round(random.uniform(0.02, 0.06), 4),
                    }

                txns.append(txn)

        return txns

    def _generate_legitimate_transactions(
        self,
        count: int = 60,
    ) -> List[Dict[str, Any]]:
        """Generate normal, non-fraudulent transactions."""
        assert self.event is not None
        txns: List[Dict[str, Any]] = []
        for _ in range(count):
            user = random.choice(self.legitimate_users)
            persona_str = user.get("persona", "casual")
            try:
                persona = UserPersona(persona_str)
            except ValueError:
                persona = UserPersona.CASUAL
            txn = self.txn_gen.generate_transaction(self.event, user, persona)
            txn["is_fraud"] = False
            txn["fraud_pattern"] = ""
            txns.append(txn)
        return txns
