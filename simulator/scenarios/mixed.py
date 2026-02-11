"""Mixed scenario - interleaved weighted mix of multiple scenarios over a configurable duration."""

import logging
import random
from typing import Any, Dict, List, Type

import numpy as np

from simulator.scenarios.base_scenario import BaseScenario
from simulator.scenarios.black_friday import BlackFridayScenario
from simulator.scenarios.flash_sale import FlashSaleScenario
from simulator.scenarios.fraud_attack import FraudAttackScenario
from simulator.scenarios.gradual_drift import GradualDriftScenario
from simulator.scenarios.normal_traffic import NormalTrafficScenario
from simulator.scenarios.system_degradation import SystemDegradationScenario

logger = logging.getLogger(__name__)

# Default mix: scenario name -> (class, weight)
DEFAULT_MIX: Dict[str, tuple[Type[BaseScenario], float]] = {
    "normal-traffic": (NormalTrafficScenario, 0.40),
    "flash-sale": (FlashSaleScenario, 0.20),
    "fraud-attack": (FraudAttackScenario, 0.20),
    "system-degradation": (SystemDegradationScenario, 0.10),
    "black-friday": (BlackFridayScenario, 0.10),
}


class MixedScenario(BaseScenario):
    """Run a weighted mix of scenarios in time-sliced rounds."""

    DEFAULT_DURATION_MINUTES = 30

    def __init__(
        self,
        scenario_weights: Dict[str, float] | None = None,
        duration_minutes: int | None = None,
    ) -> None:
        if scenario_weights is None:
            scenario_weights = {k: v[1] for k, v in DEFAULT_MIX.items()}
        mins = duration_minutes if duration_minutes is not None else self.DEFAULT_DURATION_MINUTES
        super().__init__(
            name="Mixed Scenarios",
            description="Interleaved mix of normal, flash sale, fraud, degradation, black Friday",
            duration_minutes=mins,
            expected_metrics={
                "peak_rps": 500,
                "p99_latency_ms": 200,
                "error_rate": 0.05,
            },
        )
        self._scenario_weights = scenario_weights
        self._scenario_classes: Dict[str, Type[BaseScenario]] = {}
        self._scenario_instances: Dict[str, BaseScenario] = {}
        self._names: List[str] = []
        self._weights: List[float] = []
        self._build_mix()

    def _build_mix(self) -> None:
        """Build name/weight lists from scenario_weights; resolve class from DEFAULT_MIX or by name."""
        name_to_class = {k: v[0] for k, v in DEFAULT_MIX.items()}
        for name, weight in self._scenario_weights.items():
            if weight <= 0:
                continue
            cls = name_to_class.get(name)
            if cls is None:
                logger.warning("Unknown scenario name %s, skipping", name)
                continue
            self._names.append(name)
            self._weights.append(weight)
            self._scenario_classes[name] = cls
        if not self._weights:
            self._names = list(DEFAULT_MIX.keys())
            self._weights = [v[1] for v in DEFAULT_MIX.values()]
            self._scenario_classes = {k: v[0] for k, v in DEFAULT_MIX.items()}
        total = sum(self._weights)
        self._weights = [w / total for w in self._weights]

    def setup(self) -> None:
        """Create one instance per scenario type and run setup."""
        logger.info("Setting up mixed scenario (%d types)...", len(self._names))
        self._scenario_instances.clear()
        effective = self.get_effective_duration_minutes()
        num_slices = max(5, effective)
        slice_min = effective / num_slices
        for name in self._names:
            cls = self._scenario_classes[name]
            instance = cls(duration_minutes=max(1, int(slice_min)))
            instance.setup()
            self._scenario_instances[name] = instance
        self.results["responses"] = []
        self.results["transactions"] = []

    def run(self) -> None:
        """Execute time-sliced rounds: each slice picks a scenario by weight and runs it."""
        effective = self.get_effective_duration_minutes()
        num_slices = max(5, effective)
        slice_min = max(1, effective / num_slices)
        logger.info("Running mixed scenario: %d slices of ~%s min", num_slices, slice_min)
        for i in range(num_slices):
            name = random.choices(self._names, weights=self._weights, k=1)[0]
            scenario = self._scenario_instances[name]
            scenario._effective_duration_minutes = slice_min
            scenario.run()
            self.results.setdefault("responses", []).extend(
                scenario.results.get("responses", [])
            )
            self.results.setdefault("transactions", []).extend(
                scenario.results.get("transactions", [])
            )
            if (i + 1) % 5 == 0:
                logger.info("Mixed slice %d/%d done", i + 1, num_slices)

    def teardown(self) -> None:
        """Aggregate metrics from collected responses."""
        responses = self.results.get("responses", [])
        if not responses:
            logger.warning("No responses in mixed scenario")
            return
        latencies = [r["latency_ms"] for r in responses if "latency_ms" in r]
        if latencies:
            self.results["p50_latency_ms"] = float(np.percentile(latencies, 50))
            self.results["p95_latency_ms"] = float(np.percentile(latencies, 95))
            self.results["p99_latency_ms"] = float(np.percentile(latencies, 99))
        else:
            self.results["p99_latency_ms"] = 0
        errors = [r for r in responses if r.get("status", 200) >= 400]
        self.results["error_rate"] = len(errors) / len(responses)
        duration_sec = self.results.get("duration_seconds", 1)
        self.results["peak_rps"] = len(responses) / duration_sec
        timeouts = [
            r
            for r in responses
            if r.get("timed_out") or r.get("error") == "timeout"
        ]
        self.results["timeout_count"] = len(timeouts)
        logger.info(
            "Mixed teardown: %d responses, error_rate=%.3f, timeout_count=%d",
            len(responses),
            self.results["error_rate"],
            self.results["timeout_count"],
        )
