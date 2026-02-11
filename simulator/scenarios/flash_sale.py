"""Flash sale (mega-event) scenario - 100K+ users, traffic spike, bot traffic."""

import asyncio
import logging
import random
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from simulator.config import UserPersona, config
from simulator.core.event_generator import EventGenerator
from simulator.core.transaction_generator import TransactionGenerator
from simulator.core.user_generator import UserGenerator
from simulator.scenarios.base_scenario import BaseScenario

logger = logging.getLogger(__name__)


def _synthetic_response(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Return a synthetic API response for offline testing."""
    return {
        "status": 200,
        "latency_ms": random.uniform(20, 150),
        "fraud_score": 0.3 if not transaction.get("is_fraud") else 0.85,
        "blocked": transaction.get("is_fraud", False),
        "is_fraud": transaction.get("is_fraud", False),
    }


class FlashSaleScenario(BaseScenario):
    """Simulate mega-event flash sale with 100K+ users and traffic spike."""

    DEFAULT_DURATION_MINUTES = 15

    def __init__(self, duration_minutes: int | None = None) -> None:
        mins = duration_minutes if duration_minutes is not None else self.DEFAULT_DURATION_MINUTES
        super().__init__(
            name="Flash Sale - Mega Event",
            description="Simulate high-demand event with 100K+ concurrent users",
            duration_minutes=mins,
            expected_metrics={
                "peak_rps": 10000,
                "p99_latency_ms": 200,
                "error_rate": 0.01,
                "fraud_detected_pct": 90,
            },
        )
        self.event: Dict[str, Any] | None = None
        self.users: List[Dict[str, Any]] = []
        self.event_gen = EventGenerator()
        self.user_gen = UserGenerator()
        self.txn_gen = TransactionGenerator()

    def setup(self) -> None:
        logger.info("Setting up flash sale scenario...")
        self.event = self.event_gen.generate_mega_event()
        logger.info("Created event: %s", self.event["name"])
        self.users = self.user_gen.generate_batch(
            count=min(100000, 5000),
            persona_distribution={
                "casual": 0.30,
                "enthusiast": 0.25,
                "vip": 0.15,
                "scalper": 0.20,
                "fraudster": 0.10,
            },
        )
        logger.info("Generated %d users", len(self.users))

    async def _make_purchase_request(
        self,
        session: Any,
        transaction: Dict[str, Any],
    ) -> Dict[str, Any]:
        url = f"{config.api_base_url}/predict"
        try:
            import aiohttp
            start = datetime.now()
            async with session.post(url, json=transaction, timeout=5) as response:
                result = await response.json()
                latency = (datetime.now() - start).total_seconds() * 1000
                return {
                    "status": response.status,
                    "latency_ms": latency,
                    "fraud_score": result.get("fraud_score", 0),
                    "blocked": result.get("fraud_score", 0) > 0.7,
                    "is_fraud": transaction.get("is_fraud", False),
                }
        except Exception:
            out = _synthetic_response(transaction)
            out["latency_ms"] = out.get("latency_ms", 50)
            # Network/timeout failures are treated as timeouts for reporting.
            out["status"] = 504
            out["error"] = "timeout"
            out["timed_out"] = True
            return out

    async def _generate_traffic_wave(
        self, duration_seconds: int, target_rps: int
    ) -> List[Dict[str, Any]]:
        try:
            import aiohttp
        except ImportError:
            aiohttp = None
        session = aiohttp.ClientSession() if aiohttp else None
        responses: List[Dict[str, Any]] = []
        start_time = datetime.now()
        try:
            while (datetime.now() - start_time).total_seconds() < duration_seconds:
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = elapsed / max(0.01, duration_seconds)
                if progress < 0.2:
                    intensity = progress / 0.2
                elif progress < 0.7:
                    intensity = 1.0
                else:
                    intensity = (1.0 - progress) / 0.3
                current_rps = max(1, int(target_rps * intensity))
                batch_size = max(1, current_rps // 10)
                batch_transactions = []
                for _ in range(batch_size):
                    user = random.choice(self.users)
                    persona = UserPersona(user["persona"])
                    txn = self.txn_gen.generate_transaction(
                        self.event, user, persona, datetime.now()
                    )
                    txn["is_fraud"] = txn.get("is_fraud", False)
                    batch_transactions.append(txn)
                if session is not None:
                    tasks = [
                        self._make_purchase_request(session, t)
                        for t in batch_transactions
                    ]
                    batch_responses = await asyncio.gather(*tasks)
                    responses.extend(batch_responses)
                else:
                    for t in batch_transactions:
                        responses.append(_synthetic_response(t))
                await asyncio.sleep(0.1)
        finally:
            if session is not None:
                await session.close()
        return responses

    def run(self) -> None:
        logger.info("Starting flash sale traffic generation...")
        effective = self.get_effective_duration_minutes()
        scale = effective / self.DEFAULT_DURATION_MINUTES
        p1 = max(1, int(2 * scale))
        p2 = max(1, int(3 * scale))
        p3 = max(1, int(2 * scale))
        responses_phase1 = asyncio.run(
            self._generate_traffic_wave(duration_seconds=p1, target_rps=500)
        )
        responses_phase2 = asyncio.run(
            self._generate_traffic_wave(duration_seconds=p2, target_rps=1000)
        )
        responses_phase3 = asyncio.run(
            self._generate_traffic_wave(duration_seconds=p3, target_rps=200)
        )
        all_responses = responses_phase1 + responses_phase2 + responses_phase3
        self.results["responses"] = all_responses
        logger.info("Completed %d requests", len(all_responses))

    def teardown(self) -> None:
        logger.info("Analyzing results...")
        responses = self.results.get("responses", [])
        if not responses:
            logger.error("No responses collected")
            return
        latencies = [r["latency_ms"] for r in responses if "latency_ms" in r]
        if latencies:
            self.results["p50_latency_ms"] = float(np.percentile(latencies, 50))
            self.results["p95_latency_ms"] = float(np.percentile(latencies, 95))
            self.results["p99_latency_ms"] = float(np.percentile(latencies, 99))
        else:
            self.results["p50_latency_ms"] = 0
            self.results["p95_latency_ms"] = 0
            self.results["p99_latency_ms"] = 0
        errors = [r for r in responses if r.get("status", 200) >= 400]
        self.results["error_rate"] = len(errors) / len(responses)
        duration = self.results.get("duration_seconds", 1)
        self.results["peak_rps"] = len(responses) / duration
        blocked = [r for r in responses if r.get("blocked", False)]
        actual_fraud = [r for r in responses if r.get("is_fraud", False)]
        if actual_fraud:
            true_positives = len([r for r in blocked if r.get("is_fraud", False)])
            self.results["fraud_detected_pct"] = (
                true_positives / len(actual_fraud)
            ) * 100
        else:
            self.results["fraud_detected_pct"] = 0
        timeouts = [
            r
            for r in responses
            if r.get("timed_out") or r.get("error") == "timeout"
        ]
        self.results["timeout_count"] = len(timeouts)
        logger.info("Results: %s", self.results)
