"""Tests for simulator scenarios - setup, lifecycle, validators."""

from __future__ import annotations

import pytest

from simulator.config import UserPersona
from simulator.core.event_generator import EventGenerator
from simulator.core.transaction_generator import TransactionGenerator
from simulator.core.user_generator import UserGenerator
from simulator.scenarios.base_scenario import BaseScenario
from simulator.scenarios.black_friday import BlackFridayScenario
from simulator.scenarios.flash_sale import FlashSaleScenario
from simulator.scenarios.fraud_attack import FraudAttackScenario
from simulator.scenarios.fraud_drift_retrain import FraudDriftRetrainScenario
from simulator.scenarios.gradual_drift import GradualDriftScenario
from simulator.scenarios.mixed import MixedScenario
from simulator.scenarios.normal_traffic import NormalTrafficScenario
from simulator.scenarios.system_degradation import SystemDegradationScenario


def test_event_generator_produces_valid_events() -> None:
    gen = EventGenerator()
    event = gen.generate_event()
    assert "event_id" in event
    assert "name" in event
    assert "category" in event
    assert "capacity" in event
    assert "pricing_tiers" in event
    assert len(event["pricing_tiers"]) == 3
    assert event["pricing_tiers"][0]["tier"] == "VIP"


def test_user_generator_produces_valid_users() -> None:
    gen = UserGenerator()
    user = gen.generate_user()
    assert "user_id" in user
    assert "persona" in user
    assert user["persona"] in ("casual", "enthusiast", "vip", "scalper", "fraudster")


def test_transaction_generator_produces_valid_transactions() -> None:
    event_gen = EventGenerator()
    user_gen = UserGenerator()
    txn_gen = TransactionGenerator()
    event = event_gen.generate_event()
    user = user_gen.generate_user()
    txn = txn_gen.generate_transaction(event, user, UserPersona.CASUAL)
    assert "transaction_id" in txn
    assert "event_id" in txn
    assert "user_id" in txn
    assert "total_amount" in txn
    assert "fraud_indicators" in txn


@pytest.mark.parametrize(
    "scenario_class",
    [
        NormalTrafficScenario,
        FlashSaleScenario,
        FraudAttackScenario,
        FraudDriftRetrainScenario,
        GradualDriftScenario,
        SystemDegradationScenario,
        BlackFridayScenario,
    ],
)
def test_scenario_setup_runs_without_error(scenario_class: type) -> None:
    scenario = scenario_class()
    scenario.setup()
    assert scenario.results is not None or hasattr(scenario, "users") or hasattr(scenario, "events")


def test_base_scenario_validate_returns_structure() -> None:
    """BaseScenario.validate expects results to contain expected_metrics keys."""
    class ConcreteScenario(BaseScenario):
        def setup(self) -> None:
            self.results["peak_rps"] = 100
            self.results["p99_latency_ms"] = 50
            self.results["error_rate"] = 0.01

        def run(self) -> None:
            pass

        def teardown(self) -> None:
            pass

    scenario = ConcreteScenario(
        name="Concrete",
        description="Test",
        duration_minutes=1,
        expected_metrics={"peak_rps": 100, "p99_latency_ms": 50, "error_rate": 0.01},
    )
    scenario.results = {"peak_rps": 100, "p99_latency_ms": 50, "error_rate": 0.01}
    validation = scenario.validate()
    assert "passed" in validation
    assert "failures" in validation
    assert "metrics" in validation
    assert validation["passed"] is True


def test_normal_traffic_execute_completes() -> None:
    scenario = NormalTrafficScenario()
    result = scenario.execute()
    assert "passed" in result
    assert "duration_seconds" in scenario.results or "peak_rps" in scenario.results


def test_scenario_duration_override() -> None:
    """Execute with duration_override_minutes scales workload."""
    scenario = NormalTrafficScenario()
    result = scenario.execute(duration_override_minutes=2)
    assert "passed" in result
    assert scenario.get_effective_duration_minutes() == 2
    assert "duration_seconds" in scenario.results
    assert "responses" in scenario.results
    assert len(scenario.results["responses"]) >= 50


def test_mixed_scenario_setup_runs_without_error() -> None:
    scenario = MixedScenario(
        scenario_weights={"normal-traffic": 0.5, "flash-sale": 0.5},
        duration_minutes=2,
    )
    scenario.setup()
    assert len(scenario._scenario_instances) >= 1
    assert scenario.results is not None or hasattr(scenario, "_scenario_instances")


def test_mixed_scenario_execute_completes() -> None:
    scenario = MixedScenario(
        scenario_weights={"normal-traffic": 1.0},
        duration_minutes=1,
    )
    result = scenario.execute(duration_override_minutes=1)
    assert "passed" in result
    assert "duration_seconds" in scenario.results or "peak_rps" in scenario.results
    assert "responses" in scenario.results
    assert len(scenario.results["responses"]) >= 1


def test_mixed_scenario_custom_weights() -> None:
    scenario = MixedScenario(
        scenario_weights={"normal-traffic": 70, "fraud-attack": 30},
        duration_minutes=1,
    )
    scenario.setup()
    assert "normal-traffic" in scenario._scenario_instances
    assert "fraud-attack" in scenario._scenario_instances
    scenario.run()
    assert len(scenario.results.get("responses", [])) >= 1


# ---------------------------------------------------------------------------
# FraudDriftRetrainScenario
# ---------------------------------------------------------------------------


def test_fraud_drift_retrain_setup() -> None:
    scenario = FraudDriftRetrainScenario()
    scenario.setup()
    assert scenario.event is not None
    assert len(scenario.users) > 0
    assert len(scenario.legitimate_users) > 0


def test_fraud_drift_retrain_execute_completes() -> None:
    scenario = FraudDriftRetrainScenario()
    result = scenario.execute(duration_override_minutes=1)
    assert "passed" in result
    assert "metrics" in result
    assert "duration_seconds" in scenario.results


def test_fraud_drift_retrain_produces_novel_fraud() -> None:
    scenario = FraudDriftRetrainScenario()
    scenario.setup()
    scenario.run()
    scenario.teardown()

    assert scenario.results["novel_fraud_count"] > 0
    responses = scenario.results["responses"]
    assert len(responses) > 0

    # Should contain novel patterns.
    patterns = {r.get("fraud_pattern") for r in responses}
    assert "account_takeover" in patterns or "synthetic_identity" in patterns or "refund_abuse" in patterns


def test_fraud_drift_retrain_high_initial_failure_rate() -> None:
    """Novel fraud patterns should cause a high failure rate."""
    scenario = FraudDriftRetrainScenario()
    scenario.setup()
    scenario.run()
    scenario.teardown()

    # The model is expected to miss most novel fraud → high failure rate.
    initial_failure_rate = scenario.results.get("initial_failure_rate", 0)
    assert initial_failure_rate > 0.2, f"Expected high failure rate, got {initial_failure_rate}"


def test_fraud_drift_retrain_training_triggered() -> None:
    """With enough novel fraud, auto-retraining should be triggered."""
    scenario = FraudDriftRetrainScenario()
    scenario.setup()
    scenario.run()
    scenario.teardown()

    assert scenario.results.get("training_triggered") == 1.0
