"""Locust integration - run scenarios as load tests."""

from typing import Any, Dict, List

from simulator.config import config


class LoadTestRunner:
    """Adapter to run simulator scenarios via Locust."""

    def __init__(
        self,
        api_base_url: str | None = None,
        fraud_api_base_url: str | None = None,
    ) -> None:
        # Generic API base URL for non-fraud endpoints.
        self.api_base_url = api_base_url or config.api_base_url
        # Dedicated BentoML fraud detection API base URL.
        self.fraud_api_base_url = fraud_api_base_url or config.fraud_api_base_url

    def get_locust_tasks(self) -> List[tuple[Any, int]]:
        """Return Locust task list (callable, weight) for use in a Locust User class."""
        return []

    def build_fraud_payload(self, user_id: str, event_id: str, amount: float) -> Dict[str, Any]:
        """Build payload for fraud detection API (BentoML FraudBatchRequest-compatible)."""
        return {
            "requests": [
                {
                    "user_id": user_id,
                    "event_id": event_id,
                    "amount": amount,
                }
            ]
        }

    def build_pricing_payload(self, event_id: str, current_price: float) -> Dict[str, Any]:
        """Build payload for dynamic pricing API."""
        return {
            "event_id": event_id,
            "current_price": current_price,
        }
