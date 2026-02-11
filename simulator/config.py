"""Simulator configuration - event types, user personas, traffic patterns, SLAs."""

import os
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class EventCategory(str, Enum):
    SPORTS = "sports"
    CONCERT = "concert"
    FAMILY = "family"
    CULTURAL = "cultural"
    COMEDY = "comedy"
    FESTIVAL = "festival"


class UserPersona(str, Enum):
    CASUAL = "casual"
    ENTHUSIAST = "enthusiast"
    VIP = "vip"
    SCALPER = "scalper"
    FRAUDSTER = "fraudster"


class EventConfig(BaseModel):
    category: EventCategory
    capacity: int = Field(ge=100, le=100000)
    price_range: tuple[float, float] = Field(default=(50, 500))
    popularity_score: float = Field(ge=0, le=1)
    release_strategy: str = Field(default="instant")


class UserBehavior(BaseModel):
    persona: UserPersona
    purchase_frequency: float = 0.0
    price_sensitivity: float = 0.0
    fraud_probability: float = 0.0
    bot_likelihood: float = 0.0
    cart_abandonment_rate: float = 0.0
    avg_tickets_per_purchase: float = 1.0


class TrafficPattern(BaseModel):
    base_rps: int = 100
    peak_multiplier: float = 10.0
    pattern_type: str = "daily"
    peak_hours: List[int] = Field(default_factory=lambda: [12, 13, 18, 19, 20])


class FraudPattern(BaseModel):
    name: str
    description: str
    attack_rate: float = 0.0
    success_rate: float = 0.0
    characteristics: Dict[str, Any] = Field(default_factory=dict)


def _default_user_behaviors() -> Dict[UserPersona, UserBehavior]:
    return {
        UserPersona.CASUAL: UserBehavior(
            persona=UserPersona.CASUAL,
            purchase_frequency=0.5,
            price_sensitivity=0.7,
            fraud_probability=0.0,
            bot_likelihood=0.0,
            cart_abandonment_rate=0.4,
            avg_tickets_per_purchase=2.0,
        ),
        UserPersona.ENTHUSIAST: UserBehavior(
            persona=UserPersona.ENTHUSIAST,
            purchase_frequency=3.0,
            price_sensitivity=0.3,
            fraud_probability=0.0,
            bot_likelihood=0.1,
            cart_abandonment_rate=0.2,
            avg_tickets_per_purchase=3.5,
        ),
        UserPersona.VIP: UserBehavior(
            persona=UserPersona.VIP,
            purchase_frequency=5.0,
            price_sensitivity=0.1,
            fraud_probability=0.0,
            bot_likelihood=0.0,
            cart_abandonment_rate=0.1,
            avg_tickets_per_purchase=5.0,
        ),
        UserPersona.SCALPER: UserBehavior(
            persona=UserPersona.SCALPER,
            purchase_frequency=10.0,
            price_sensitivity=0.5,
            fraud_probability=0.2,
            bot_likelihood=0.8,
            cart_abandonment_rate=0.05,
            avg_tickets_per_purchase=8.0,
        ),
        UserPersona.FRAUDSTER: UserBehavior(
            persona=UserPersona.FRAUDSTER,
            purchase_frequency=20.0,
            price_sensitivity=0.0,
            fraud_probability=1.0,
            bot_likelihood=0.95,
            cart_abandonment_rate=0.8,
            avg_tickets_per_purchase=10.0,
        ),
    }


def _default_fraud_patterns() -> List[FraudPattern]:
    return [
        FraudPattern(
            name="credential_stuffing",
            description="Automated login attempts with stolen credentials",
            attack_rate=50.0,
            success_rate=0.05,
            characteristics={
                "user_agent_rotation": True,
                "ip_rotation": True,
                "velocity_anomaly": True,
            },
        ),
        FraudPattern(
            name="card_testing",
            description="Testing stolen credit cards with small purchases",
            attack_rate=10.0,
            success_rate=0.15,
            characteristics={
                "low_value_transactions": True,
                "multiple_cards_same_user": True,
                "rapid_succession": True,
            },
        ),
        FraudPattern(
            name="bot_scalping",
            description="Automated bulk ticket purchases",
            attack_rate=100.0,
            success_rate=0.30,
            characteristics={
                "high_velocity": True,
                "predictable_patterns": True,
                "multiple_accounts": True,
            },
        ),
    ]


class SimulatorConfig(BaseModel):
    """Master configuration for simulator."""

    environment: str = "local"
    # Legacy API base URL; kept for backwards compatibility.
    api_base_url: str = Field(default_factory=lambda: os.getenv("SIMULATOR_API_BASE_URL", "http://localhost:3001"))
    # Dedicated BentoML fraud detection API base URL.
    fraud_api_base_url: str = Field(default_factory=lambda: os.getenv("FRAUD_API_BASE_URL", "http://localhost:7001"))

    event_categories_distribution: Dict[EventCategory, float] = Field(
        default_factory=lambda: {
            EventCategory.SPORTS: 0.30,
            EventCategory.CONCERT: 0.35,
            EventCategory.FAMILY: 0.15,
            EventCategory.CULTURAL: 0.10,
            EventCategory.COMEDY: 0.05,
            EventCategory.FESTIVAL: 0.05,
        }
    )

    user_personas_distribution: Dict[UserPersona, float] = Field(
        default_factory=lambda: {
            UserPersona.CASUAL: 0.50,
            UserPersona.ENTHUSIAST: 0.30,
            UserPersona.VIP: 0.10,
            UserPersona.SCALPER: 0.07,
            UserPersona.FRAUDSTER: 0.03,
        }
    )

    user_behaviors: Dict[UserPersona, UserBehavior] = Field(default_factory=_default_user_behaviors)

    fraud_patterns: List[FraudPattern] = Field(default_factory=_default_fraud_patterns)

    sla_targets: Dict[str, float] = Field(
        default_factory=lambda: {
            "latency_p99_ms": 200.0,
            "latency_p95_ms": 150.0,
            "latency_p50_ms": 50.0,
            "error_rate": 0.01,
            "availability": 0.999,
        }
    )

    business_targets: Dict[str, float] = Field(
        default_factory=lambda: {
            "fraud_detection_recall": 0.95,
            "fraud_detection_precision": 0.90,
            "pricing_revenue_uplift": 0.08,
            "recommendation_ctr": 0.15,
        }
    )

config = SimulatorConfig()
