# Event Ticketing Simulator - Test Scenarios for MLOps Platform
## Comprehensive Testing & Stress Testing System

---

## 🎯 Overview

The **Ibook Event Simulator** generates realistic ticketing scenarios to test the MLOps platform under various conditions including:
- Normal operations
- Flash sales (mega-events)
- Fraud attacks
- System degradation
- Data drift
- Black Friday-style traffic spikes

---

## 📁 Project Structure Updates

Add to existing `ibook-mlops/` structure:

```
ibook-mlops/
├── simulator/
│   ├── __init__.py
│   ├── config.py                    # Simulator configuration
│   ├── core/
│   │   ├── __init__.py
│   │   ├── event_generator.py      # Generate events
│   │   ├── user_generator.py       # Generate user profiles
│   │   ├── transaction_generator.py # Generate transactions
│   │   └── fraud_simulator.py      # Simulate fraud patterns
│   ├── scenarios/
│   │   ├── __init__.py
│   │   ├── base_scenario.py        # Base class for scenarios
│   │   ├── normal_traffic.py       # Normal day operations
│   │   ├── flash_sale.py           # Mega-event launch
│   │   ├── fraud_attack.py         # Coordinated fraud
│   │   ├── fraud_drift_retrain.py  # Novel fraud → auto-retrain
│   │   ├── gradual_drift.py        # Seasonal changes
│   │   ├── system_degradation.py   # Partial failures
│   │   ├── ab_test.py              # A/B testing scenarios
│   │   └── black_friday.py         # Extreme load
│   ├── runners/
│   │   ├── __init__.py
│   │   ├── local_runner.py         # Run locally via API
│   │   ├── load_test_runner.py     # Locust integration
│   │   └── chaos_runner.py         # Chaos engineering
│   ├── validators/
│   │   ├── __init__.py
│   │   ├── latency_validator.py    # Check SLAs
│   │   ├── accuracy_validator.py   # Model performance
│   │   ├── drift_validator.py      # Detect drift
│   │   └── business_validator.py   # Revenue metrics
│   ├── visualizers/
│   │   ├── __init__.py
│   │   ├── dashboard.py            # Streamlit dashboard
│   │   └── report_generator.py     # HTML/PDF reports
│   └── cli.py                       # Command-line interface
│
├── docker-compose.simulator.yml     # Simulator services
├── scripts/
│   └── run-simulation.sh            # Run simulations
└── tests/
    └── simulation/
        ├── __init__.py
        ├── test_scenarios.py
        └── test_validators.py
```

---

## 🚀 Implementation Guide

### Phase 1: Core Simulator Components

#### 1.1 Simulator Configuration

**AI Prompt for Cursor/Claude Code:**
```
Create simulator/config.py with configuration for:
1. Event types (sports, concerts, family, cultural) with distributions
2. User personas (casual, enthusiast, scalper, fraud) with behaviors
3. Traffic patterns (hourly, daily, seasonal)
4. Fraud patterns (bot attacks, stolen cards, account takeover)
5. System parameters (latency targets, error rates, throughput)

Use Pydantic for validation. Support both local and production modes.
```

**File: `simulator/config.py`**

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime, timedelta

class EventCategory(str, Enum):
    SPORTS = "sports"
    CONCERT = "concert"
    FAMILY = "family"
    CULTURAL = "cultural"
    COMEDY = "comedy"
    FESTIVAL = "festival"

class UserPersona(str, Enum):
    CASUAL = "casual"              # Occasional buyer
    ENTHUSIAST = "enthusiast"      # Regular buyer
    VIP = "vip"                    # High-value customer
    SCALPER = "scalper"            # Reseller
    FRAUDSTER = "fraudster"        # Malicious actor

class EventConfig(BaseModel):
    category: EventCategory
    capacity: int = Field(ge=100, le=100000)
    price_range: tuple[float, float] = Field(default=(50, 500))
    popularity_score: float = Field(ge=0, le=1)  # 0=low, 1=viral
    release_strategy: str = Field(default="instant")  # instant, phased, waitlist

class UserBehavior(BaseModel):
    persona: UserPersona
    purchase_frequency: float  # purchases per month
    price_sensitivity: float  # 0=price insensitive, 1=very sensitive
    fraud_probability: float  # 0=legitimate, 1=fraudulent
    bot_likelihood: float  # 0=human, 1=automated
    cart_abandonment_rate: float
    avg_tickets_per_purchase: float

class TrafficPattern(BaseModel):
    base_rps: int = 100  # Requests per second
    peak_multiplier: float = 10.0
    pattern_type: str = "daily"  # daily, weekly, event-driven
    peak_hours: List[int] = [12, 13, 18, 19, 20]  # Hour of day

class FraudPattern(BaseModel):
    name: str
    description: str
    attack_rate: float  # Transactions per second
    success_rate: float  # How many bypass detection
    characteristics: Dict[str, any]

class SimulatorConfig(BaseModel):
    """Master configuration for simulator."""
    
    # Environment
    environment: str = "local"  # local, staging, production
    api_base_url: str = "http://localhost:3001"
    
    # Event distributions
    event_categories_distribution: Dict[EventCategory, float] = {
        EventCategory.SPORTS: 0.30,
        EventCategory.CONCERT: 0.35,
        EventCategory.FAMILY: 0.15,
        EventCategory.CULTURAL: 0.10,
        EventCategory.COMEDY: 0.05,
        EventCategory.FESTIVAL: 0.05,
    }
    
    # User distributions
    user_personas_distribution: Dict[UserPersona, float] = {
        UserPersona.CASUAL: 0.50,
        UserPersona.ENTHUSIAST: 0.30,
        UserPersona.VIP: 0.10,
        UserPersona.SCALPER: 0.07,
        UserPersona.FRAUDSTER: 0.03,
    }
    
    # Default behaviors by persona
    user_behaviors: Dict[UserPersona, UserBehavior] = {
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
            cart_abandonment_rate=0.8,  # High due to fraud detection
            avg_tickets_per_purchase=10.0,
        ),
    }
    
    # Fraud patterns
    fraud_patterns: List[FraudPattern] = [
        FraudPattern(
            name="credential_stuffing",
            description="Automated login attempts with stolen credentials",
            attack_rate=50.0,
            success_rate=0.05,
            characteristics={
                "user_agent_rotation": True,
                "ip_rotation": True,
                "velocity_anomaly": True,
            }
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
            }
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
            }
        ),
    ]
    
    # System SLAs
    sla_targets: Dict[str, float] = {
        "latency_p99_ms": 200.0,
        "latency_p95_ms": 150.0,
        "latency_p50_ms": 50.0,
        "error_rate": 0.01,  # 1%
        "availability": 0.999,  # 99.9%
    }
    
    # Business metrics
    business_targets: Dict[str, float] = {
        "fraud_detection_recall": 0.95,  # Catch 95% of fraud
        "fraud_detection_precision": 0.90,  # 90% of blocks are actual fraud
        "pricing_revenue_uplift": 0.08,  # 8% vs baseline
        "recommendation_ctr": 0.15,  # 15% click-through
    }

# Singleton instance
config = SimulatorConfig()
```

---

#### 1.2 Event Generator

**AI Prompt:**
```
Create simulator/core/event_generator.py that:
1. Generates realistic event data (name, category, venue, date, capacity)
2. Assigns pricing tiers (VIP, Premium, Standard)
3. Simulates event popularity using power-law distribution
4. Creates event timelines (announcement, presale, general sale, event date)
5. Supports Saudi Arabia context (Riyadh Season, national holidays)

Include seasonality patterns (summer festivals, winter sports).
```

**File: `simulator/core/event_generator.py`**

```python
import random
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np
from faker import Faker
from simulator.config import EventCategory, EventConfig, config

fake = Faker(['ar_SA', 'en_US'])  # Arabic and English

class EventGenerator:
    """Generate realistic ticketing events."""
    
    def __init__(self):
        self.event_templates = {
            EventCategory.SPORTS: [
                "Saudi Pro League: {team1} vs {team2}",
                "Formula 1 Saudi Arabian Grand Prix",
                "WWE Live in Riyadh",
                "Basketball Championship Finals",
                "Esports Tournament: {game}",
            ],
            EventCategory.CONCERT: [
                "{artist} Live in Concert",
                "Riyadh Season Festival - {artist}",
                "Jazz Night at {venue}",
                "Arabic Music Festival",
                "International Pop Concert",
            ],
            EventCategory.FAMILY: [
                "Disney on Ice: {show}",
                "Cirque du Soleil: {show}",
                "Kids Fun Fest",
                "Educational Workshop: {topic}",
                "Family Movie Night",
            ],
            EventCategory.CULTURAL: [
                "Saudi Heritage Festival",
                "Art Exhibition: {artist}",
                "Theater Play: {title}",
                "Poetry Night",
                "Cultural Dance Performance",
            ],
        }
        
        self.venues = [
            "King Fahd International Stadium",
            "Kingdom Arena",
            "Boulevard Riyadh City",
            "Diriyah Season",
            "Princess Nourah University Stadium",
            "Al Majdoue Square",
        ]
        
        self.teams = ["Al Hilal", "Al Nassr", "Al Ittihad", "Al Ahli"]
        self.artists = [
            "Amr Diab", "Nancy Ajram", "Tamer Hosny",
            "Ed Sheeran", "Bruno Mars", "Coldplay"
        ]
    
    def generate_event(
        self, 
        category: EventCategory = None,
        popularity_override: float = None
    ) -> Dict:
        """Generate a single event."""
        
        if category is None:
            category = np.random.choice(
                list(config.event_categories_distribution.keys()),
                p=list(config.event_categories_distribution.values())
            )
        
        template = random.choice(self.event_templates[category])
        
        # Fill template placeholders
        event_name = template.format(
            team1=random.choice(self.teams),
            team2=random.choice(self.teams),
            artist=random.choice(self.artists),
            game="League of Legends",
            venue=random.choice(self.venues),
            show=random.choice(["Frozen", "Aladdin", "Lion King"]),
            topic=random.choice(["Science", "Art", "Coding"]),
            title=fake.catch_phrase()
        )
        
        # Popularity follows power-law (few mega-events, many small)
        if popularity_override:
            popularity = popularity_override
        else:
            popularity = np.random.power(2)  # Skewed towards 0
        
        # Capacity based on popularity
        if popularity > 0.9:  # Mega event
            capacity = random.randint(20000, 100000)
            price_range = (200, 2000)
        elif popularity > 0.7:  # Large event
            capacity = random.randint(5000, 20000)
            price_range = (100, 800)
        else:  # Regular event
            capacity = random.randint(500, 5000)
            price_range = (50, 300)
        
        # Event timeline
        announcement_date = datetime.now() - timedelta(days=random.randint(7, 60))
        presale_date = announcement_date + timedelta(days=random.randint(3, 14))
        general_sale_date = presale_date + timedelta(days=random.randint(1, 7))
        event_date = general_sale_date + timedelta(days=random.randint(14, 90))
        
        return {
            "event_id": f"event_{fake.uuid4()[:8]}",
            "name": event_name,
            "category": category.value,
            "venue": random.choice(self.venues),
            "capacity": capacity,
            "popularity_score": popularity,
            "price_range": price_range,
            "pricing_tiers": self._generate_pricing_tiers(price_range, capacity),
            "announcement_date": announcement_date.isoformat(),
            "presale_date": presale_date.isoformat(),
            "general_sale_date": general_sale_date.isoformat(),
            "event_date": event_date.isoformat(),
            "metadata": {
                "is_mega_event": popularity > 0.9,
                "expected_sellout_hours": self._estimate_sellout_time(popularity, capacity),
            }
        }
    
    def _generate_pricing_tiers(self, price_range: tuple, capacity: int) -> List[Dict]:
        """Generate VIP, Premium, Standard tiers."""
        min_price, max_price = price_range
        
        return [
            {
                "tier": "VIP",
                "price": max_price,
                "capacity": int(capacity * 0.10),
                "benefits": ["Meet & Greet", "Premium Seating", "VIP Lounge"]
            },
            {
                "tier": "Premium",
                "price": (min_price + max_price) / 2,
                "capacity": int(capacity * 0.30),
                "benefits": ["Priority Entry", "Better View"]
            },
            {
                "tier": "Standard",
                "price": min_price,
                "capacity": int(capacity * 0.60),
                "benefits": []
            },
        ]
    
    def _estimate_sellout_time(self, popularity: float, capacity: int) -> float:
        """Estimate hours to sell out."""
        base_rate = 100  # Tickets per hour for average event
        
        if popularity > 0.9:
            rate = base_rate * 1000  # Sells out in minutes
        elif popularity > 0.7:
            rate = base_rate * 100
        else:
            rate = base_rate
        
        return capacity / rate
    
    def generate_batch(self, count: int = 100) -> List[Dict]:
        """Generate multiple events."""
        return [self.generate_event() for _ in range(count)]
    
    def generate_mega_event(self) -> Dict:
        """Generate a mega-event (Riyadh Season style)."""
        return self.generate_event(
            category=random.choice([EventCategory.CONCERT, EventCategory.SPORTS]),
            popularity_override=0.95
        )
```

---

#### 1.3 Transaction Generator

**AI Prompt:**
```
Create simulator/core/transaction_generator.py that:
1. Generates realistic purchase transactions
2. Simulates user journey: browse → cart → checkout → payment
3. Includes cart abandonment logic
4. Adds realistic payment methods (card, wallet, bank transfer)
5. Simulates fraud indicators (velocity, device fingerprint, IP geolocation)
6. Creates time-series data with realistic distributions

Support both normal and fraudulent transactions.
```

**File: `simulator/core/transaction_generator.py`**

```python
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from faker import Faker
from simulator.config import UserPersona, config

fake = Faker()

class TransactionGenerator:
    """Generate realistic ticket purchase transactions."""
    
    def __init__(self):
        self.payment_methods = {
            "card": 0.60,
            "wallet": 0.25,
            "bank_transfer": 0.15,
        }
        
        self.device_types = ["mobile", "desktop", "tablet"]
        self.browsers = ["Chrome", "Safari", "Firefox", "Edge"]
        
    def generate_transaction(
        self,
        event: Dict,
        user: Dict,
        persona: UserPersona,
        timestamp: datetime = None
    ) -> Dict:
        """Generate a single transaction."""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        behavior = config.user_behaviors[persona]
        
        # Ticket selection
        num_tickets = max(1, int(np.random.normal(
            behavior.avg_tickets_per_purchase, 
            behavior.avg_tickets_per_purchase * 0.3
        )))
        
        # Pricing tier selection (VIPs prefer VIP tier)
        tier_probs = self._get_tier_probabilities(persona)
        tier = np.random.choice(
            [t["tier"] for t in event["pricing_tiers"]],
            p=tier_probs
        )
        
        tier_info = next(t for t in event["pricing_tiers"] if t["tier"] == tier)
        price_per_ticket = tier_info["price"]
        
        # Apply dynamic pricing (if model is active)
        if random.random() < 0.5:  # 50% get dynamic pricing
            price_adjustment = np.random.uniform(-0.15, 0.25)  # -15% to +25%
            price_per_ticket *= (1 + price_adjustment)
        
        total_amount = price_per_ticket * num_tickets
        
        # Cart abandonment
        is_abandoned = random.random() < behavior.cart_abandonment_rate
        
        # Payment method
        payment_method = np.random.choice(
            list(self.payment_methods.keys()),
            p=list(self.payment_methods.values())
        )
        
        # Device/browser fingerprinting
        device_type = random.choice(self.device_types)
        browser = random.choice(self.browsers)
        user_agent = f"{browser}/{random.randint(90, 120)}.0 ({device_type})"
        
        # IP geolocation (Saudi Arabia or suspicious locations)
        if behavior.fraud_probability > 0.5:
            # Fraudsters often use VPNs
            ip_country = random.choice(["SA", "US", "RU", "CN", "NG"])
        else:
            ip_country = "SA"  # Legitimate Saudi users
        
        # Fraud indicators
        fraud_indicators = self._calculate_fraud_indicators(
            user, behavior, timestamp, payment_method, ip_country
        )
        
        transaction = {
            "transaction_id": f"txn_{fake.uuid4()[:12]}",
            "event_id": event["event_id"],
            "user_id": user["user_id"],
            "timestamp": timestamp.isoformat(),
            
            # Purchase details
            "num_tickets": num_tickets,
            "tier": tier,
            "price_per_ticket": round(price_per_ticket, 2),
            "total_amount": round(total_amount, 2),
            
            # Payment
            "payment_method": payment_method,
            "payment_status": "abandoned" if is_abandoned else "pending",
            
            # Device/session
            "device_type": device_type,
            "browser": browser,
            "user_agent": user_agent,
            "ip_address": fake.ipv4(),
            "ip_country": ip_country,
            "session_id": f"sess_{fake.uuid4()[:8]}",
            
            # Fraud signals
            "fraud_indicators": fraud_indicators,
            "is_fraud": behavior.fraud_probability > random.random(),
            
            # Metadata
            "persona": persona.value,
            "is_bot": behavior.bot_likelihood > random.random(),
        }
        
        # Complete transaction if not abandoned
        if not is_abandoned:
            if transaction["is_fraud"]:
                # Fraud attempts may fail
                transaction["payment_status"] = random.choice(
                    ["failed", "blocked", "completed"],
                    p=[0.5, 0.3, 0.2]
                )
            else:
                transaction["payment_status"] = "completed"
        
        return transaction
    
    def _get_tier_probabilities(self, persona: UserPersona) -> List[float]:
        """Get probability distribution over pricing tiers."""
        if persona == UserPersona.VIP:
            return [0.7, 0.2, 0.1]  # Prefer VIP
        elif persona == UserPersona.ENTHUSIAST:
            return [0.2, 0.5, 0.3]  # Prefer Premium
        else:
            return [0.05, 0.25, 0.7]  # Prefer Standard
    
    def _calculate_fraud_indicators(
        self,
        user: Dict,
        behavior,
        timestamp: datetime,
        payment_method: str,
        ip_country: str
    ) -> Dict:
        """Calculate fraud risk signals."""
        
        # Velocity: transactions in last hour
        velocity_score = random.random() * behavior.fraud_probability * 10
        
        # Device mismatch (if user has history)
        device_mismatch = random.random() < behavior.fraud_probability
        
        # IP geolocation mismatch
        geo_mismatch = ip_country != "SA" and ip_country != user.get("country", "SA")
        
        # Time-of-day anomaly (late night suspicious)
        hour = timestamp.hour
        time_anomaly = (hour < 4 or hour > 23) and behavior.fraud_probability > 0.5
        
        return {
            "velocity_score": round(velocity_score, 2),
            "device_mismatch": device_mismatch,
            "geo_mismatch": geo_mismatch,
            "time_anomaly": time_anomaly,
            "risk_score": round(
                (velocity_score / 10 * 0.3 + 
                 device_mismatch * 0.2 + 
                 geo_mismatch * 0.3 + 
                 time_anomaly * 0.2), 
                2
            )
        }
    
    def generate_batch(
        self,
        events: List[Dict],
        users: List[Dict],
        count: int = 1000,
        time_range_hours: int = 24
    ) -> List[Dict]:
        """Generate batch of transactions over time period."""
        
        transactions = []
        start_time = datetime.now() - timedelta(hours=time_range_hours)
        
        for _ in range(count):
            # Random event and user
            event = random.choice(events)
            user = random.choice(users)
            persona = UserPersona(user["persona"])
            
            # Random timestamp in range
            random_seconds = random.randint(0, time_range_hours * 3600)
            timestamp = start_time + timedelta(seconds=random_seconds)
            
            transaction = self.generate_transaction(event, user, persona, timestamp)
            transactions.append(transaction)
        
        return sorted(transactions, key=lambda x: x["timestamp"])
```

---

### Phase 2: Scenario Definitions

#### 2.1 Base Scenario Class

**AI Prompt:**
```
Create simulator/scenarios/base_scenario.py with abstract base class for scenarios:
1. Define common interface (setup, run, teardown, validate)
2. Include scenario metadata (name, description, duration, expected_load)
3. Support configurable parameters
4. Provide hooks for before/after actions
5. Include automatic validation and reporting
```

**File: `simulator/scenarios/base_scenario.py`**

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class BaseScenario(ABC):
    """Abstract base class for test scenarios."""
    
    def __init__(
        self,
        name: str,
        description: str,
        duration_minutes: int,
        expected_metrics: Dict[str, float]
    ):
        self.name = name
        self.description = description
        self.duration_minutes = duration_minutes
        self.expected_metrics = expected_metrics
        
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.results: Dict = {}
        
    @abstractmethod
    def setup(self) -> None:
        """Setup phase - prepare data, initialize state."""
        pass
    
    @abstractmethod
    def run(self) -> None:
        """Main execution phase - generate load, simulate scenario."""
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """Cleanup phase - collect metrics, generate reports."""
        pass
    
    def validate(self) -> Dict:
        """Validate results against expected metrics."""
        validation_results = {
            "scenario": self.name,
            "passed": True,
            "failures": [],
            "metrics": {}
        }
        
        for metric_name, expected_value in self.expected_metrics.items():
            actual_value = self.results.get(metric_name)
            
            if actual_value is None:
                validation_results["failures"].append(
                    f"Metric '{metric_name}' not found in results"
                )
                validation_results["passed"] = False
                continue
            
            # Check if within 10% tolerance
            tolerance = 0.10
            if abs(actual_value - expected_value) / expected_value > tolerance:
                validation_results["failures"].append(
                    f"{metric_name}: expected {expected_value}, got {actual_value}"
                )
                validation_results["passed"] = False
            
            validation_results["metrics"][metric_name] = {
                "expected": expected_value,
                "actual": actual_value,
                "passed": abs(actual_value - expected_value) / expected_value <= tolerance
            }
        
        return validation_results
    
    def execute(self) -> Dict:
        """Execute full scenario lifecycle."""
        logger.info(f"Starting scenario: {self.name}")
        self.start_time = datetime.now()
        
        try:
            self.setup()
            logger.info("Setup complete")
            
            self.run()
            logger.info("Run complete")
            
            self.teardown()
            logger.info("Teardown complete")
            
        except Exception as e:
            logger.error(f"Scenario failed: {e}")
            self.results["error"] = str(e)
        
        finally:
            self.end_time = datetime.now()
            self.results["duration_seconds"] = (
                self.end_time - self.start_time
            ).total_seconds()
        
        return self.validate()
```

---

#### 2.2 Flash Sale Scenario (Mega-Event)

**AI Prompt:**
```
Create simulator/scenarios/flash_sale.py that simulates a mega-event launch:
1. 100K+ concurrent users
2. Traffic spike within 5 minutes of sale start
3. Bot attacks (30% of traffic)
4. Inventory depletion tracking
5. System stress at peak load
6. Validate: p99 latency, error rate, fraud detection accuracy

Simulate Riyadh Season concert announcement.
```

**File: `simulator/scenarios/flash_sale.py`**

```python
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np
from simulator.scenarios.base_scenario import BaseScenario
from simulator.core.event_generator import EventGenerator
from simulator.core.user_generator import UserGenerator
from simulator.core.transaction_generator import TransactionGenerator
from simulator.config import config
import logging

logger = logging.getLogger(__name__)

class FlashSaleScenario(BaseScenario):
    """
    Simulates a mega-event flash sale (e.g., Bruno Mars in Riyadh).
    
    Characteristics:
    - 100K concurrent users
    - 10-minute peak window
    - 30% bot traffic
    - Tickets sell out in <1 hour
    """
    
    def __init__(self):
        super().__init__(
            name="Flash Sale - Mega Event",
            description="Simulate high-demand event with 100K+ concurrent users",
            duration_minutes=15,
            expected_metrics={
                "peak_rps": 10000,
                "p99_latency_ms": 200,
                "error_rate": 0.01,
                "fraud_detected_pct": 90,
            }
        )
        
        self.event = None
        self.users = []
        self.transactions = []
        
        self.event_gen = EventGenerator()
        self.user_gen = UserGenerator()
        self.txn_gen = TransactionGenerator()
        
    def setup(self) -> None:
        """Create mega-event and user population."""
        logger.info("Setting up flash sale scenario...")
        
        # Create mega-event
        self.event = self.event_gen.generate_mega_event()
        logger.info(f"Created event: {self.event['name']}")
        logger.info(f"Capacity: {self.event['capacity']}, Popularity: {self.event['popularity_score']}")
        
        # Generate user population (70% legitimate, 30% bots/scalpers)
        self.users = self.user_gen.generate_batch(
            count=100000,
            persona_distribution={
                "casual": 0.30,
                "enthusiast": 0.25,
                "vip": 0.15,
                "scalper": 0.20,
                "fraudster": 0.10,
            }
        )
        logger.info(f"Generated {len(self.users)} users")
        
    async def _make_purchase_request(
        self,
        session: aiohttp.ClientSession,
        transaction: Dict
    ) -> Dict:
        """Make async HTTP request to fraud detection API."""
        url = f"{config.api_base_url}/predict"
        
        try:
            start = datetime.now()
            async with session.post(url, json=transaction, timeout=5) as response:
                result = await response.json()
                latency = (datetime.now() - start).total_seconds() * 1000
                
                return {
                    "status": response.status,
                    "latency_ms": latency,
                    "fraud_score": result.get("fraud_score", 0),
                    "blocked": result.get("fraud_score", 0) > 0.7,
                }
        except asyncio.TimeoutError:
            return {"status": 504, "latency_ms": 5000, "error": "timeout"}
        except Exception as e:
            return {"status": 500, "latency_ms": 0, "error": str(e)}
    
    async def _generate_traffic_wave(self, duration_seconds: int, target_rps: int):
        """Generate traffic with time-based intensity curve."""
        async with aiohttp.ClientSession() as session:
            start_time = datetime.now()
            requests_sent = 0
            responses = []
            
            while (datetime.now() - start_time).total_seconds() < duration_seconds:
                # Calculate current RPS based on time (ramp up, peak, ramp down)
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = elapsed / duration_seconds
                
                # Traffic curve: slow start, peak at 50%, gradual decline
                if progress < 0.2:
                    intensity = progress / 0.2  # Ramp up
                elif progress < 0.7:
                    intensity = 1.0  # Peak
                else:
                    intensity = (1.0 - progress) / 0.3  # Ramp down
                
                current_rps = int(target_rps * intensity)
                
                # Generate batch of transactions
                batch_size = max(1, current_rps // 10)  # 100ms batches
                batch_transactions = []
                
                for _ in range(batch_size):
                    user = np.random.choice(self.users)
                    persona = user["persona"]
                    
                    txn = self.txn_gen.generate_transaction(
                        self.event, user, persona, datetime.now()
                    )
                    batch_transactions.append(txn)
                
                # Send batch concurrently
                tasks = [
                    self._make_purchase_request(session, txn)
                    for txn in batch_transactions
                ]
                
                batch_responses = await asyncio.gather(*tasks)
                responses.extend(batch_responses)
                requests_sent += len(batch_responses)
                
                # Log progress
                if requests_sent % 1000 == 0:
                    logger.info(f"Sent {requests_sent} requests, current RPS: {current_rps}")
                
                # Rate limiting (sleep to achieve target RPS)
                await asyncio.sleep(0.1)
            
            return responses
    
    def run(self) -> None:
        """Execute flash sale simulation."""
        logger.info("Starting flash sale traffic generation...")
        
        # Phase 1: Pre-sale buzz (low traffic) - 2 minutes
        logger.info("Phase 1: Pre-sale buzz")
        responses_phase1 = asyncio.run(
            self._generate_traffic_wave(duration_seconds=120, target_rps=500)
        )
        
        # Phase 2: Sale opens (traffic spike) - 5 minutes
        logger.info("Phase 2: Flash sale opens - PEAK TRAFFIC")
        responses_phase2 = asyncio.run(
            self._generate_traffic_wave(duration_seconds=300, target_rps=10000)
        )
        
        # Phase 3: Post-peak (declining traffic) - 8 minutes
        logger.info("Phase 3: Post-peak slowdown")
        responses_phase3 = asyncio.run(
            self._generate_traffic_wave(duration_seconds=480, target_rps=2000)
        )
        
        all_responses = responses_phase1 + responses_phase2 + responses_phase3
        self.results["responses"] = all_responses
        
        logger.info(f"Completed {len(all_responses)} requests")
    
    def teardown(self) -> None:
        """Collect and analyze metrics."""
        logger.info("Analyzing results...")
        
        responses = self.results.get("responses", [])
        
        if not responses:
            logger.error("No responses collected")
            return
        
        # Latency metrics
        latencies = [r["latency_ms"] for r in responses if "latency_ms" in r]
        self.results["p50_latency_ms"] = np.percentile(latencies, 50)
        self.results["p95_latency_ms"] = np.percentile(latencies, 95)
        self.results["p99_latency_ms"] = np.percentile(latencies, 99)
        
        # Error rate
        errors = [r for r in responses if r.get("status", 200) >= 400]
        self.results["error_rate"] = len(errors) / len(responses)
        
        # Throughput
        duration = self.results.get("duration_seconds", 1)
        self.results["peak_rps"] = len(responses) / duration
        
        # Fraud detection performance
        blocked = [r for r in responses if r.get("blocked", False)]
        actual_fraud = [r for r in responses if r.get("is_fraud", False)]
        
        if actual_fraud:
            true_positives = len([r for r in blocked if r.get("is_fraud", False)])
            self.results["fraud_detected_pct"] = (true_positives / len(actual_fraud)) * 100
        else:
            self.results["fraud_detected_pct"] = 0
        
        logger.info(f"Results: {self.results}")
```

---

#### 2.3 Additional Scenarios (Summary)

Create these scenario files following the same pattern:

**`simulator/scenarios/normal_traffic.py`**
- Simulates typical daily operations
- 100-500 RPS sustained
- Low fraud rate (3%)
- Validates: baseline performance, model accuracy

**`simulator/scenarios/fraud_attack.py`**
- Coordinated fraud attack
- 1000+ fraudulent attempts
- Tests: fraud detection recall, precision, response time

**`simulator/scenarios/fraud_drift_retrain.py`**
- Novel fraud patterns: **account takeover**, **synthetic identity**, **refund abuse**
- These patterns produce feature distributions the original model has never seen
- Account takeover: high `lifetime_purchases` + low `fraud_risk_score` (looks legitimate)
- Synthetic identity: zero purchase history, near-zero risk score (brand-new accounts)
- Refund abuse: moderate features indistinguishable from legitimate users
- The model misses ~70% of novel fraud, causing failure rate to exceed the auto-training threshold
- Tests: initial failure rate, training trigger, per-pattern recall
- Run: `python -m simulator.cli run fraud-drift-retrain -o reports/fraud-drift-retrain-report.html`

**`simulator/scenarios/gradual_drift.py`**
- Simulates seasonal changes over weeks
- User behavior shifts
- Price sensitivity changes
- Tests: drift detection, auto-retraining triggers

**`simulator/scenarios/system_degradation.py`**
- Partial service failures (Redis slow, ML model timeout)
- Tests: circuit breakers, fallback mechanisms, graceful degradation

---

### Phase 3: Scenario Runners

#### 3.1 CLI Interface

**AI Prompt:**
```
Create simulator/cli.py with command-line interface:
1. List available scenarios
2. Run specific scenario with parameters
3. Run all scenarios (test suite)
4. Generate comparison reports
5. Support dry-run mode

Use Click library for CLI.
```

**File: `simulator/cli.py`**

```python
import click
import logging
from simulator.scenarios.flash_sale import FlashSaleScenario
from simulator.scenarios.normal_traffic import NormalTrafficScenario
from simulator.scenarios.fraud_attack import FraudAttackScenario
from simulator.visualizers.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCENARIOS = {
    "flash-sale": FlashSaleScenario,
    "normal-traffic": NormalTrafficScenario,
    "fraud-attack": FraudAttackScenario,
}

@click.group()
def cli():
    """Ibook MLOps Platform Simulator"""
    pass

@cli.command()
def list_scenarios():
    """List all available scenarios."""
    click.echo("\nAvailable Scenarios:")
    click.echo("=" * 50)
    for name, scenario_class in SCENARIOS.items():
        instance = scenario_class()
        click.echo(f"\n{name}:")
        click.echo(f"  Description: {instance.description}")
        click.echo(f"  Duration: {instance.duration_minutes} minutes")

@cli.command()
@click.argument('scenario_name', type=click.Choice(list(SCENARIOS.keys())))
@click.option('--output', '-o', default='report.html', help='Output report file')
@click.option('--dry-run', is_flag=True, help='Validate setup without running')
def run(scenario_name, output, dry_run):
    """Run a specific scenario."""
    click.echo(f"\nRunning scenario: {scenario_name}")
    
    scenario_class = SCENARIOS[scenario_name]
    scenario = scenario_class()
    
    if dry_run:
        click.echo("DRY RUN: Setup only")
        scenario.setup()
        click.echo("Setup successful ✓")
        return
    
    # Execute scenario
    with click.progressbar(length=100, label='Running') as bar:
        results = scenario.execute()
        bar.update(100)
    
    # Display results
    click.echo("\nResults:")
    click.echo("=" * 50)
    if results["passed"]:
        click.echo(click.style("✓ PASSED", fg='green'))
    else:
        click.echo(click.style("✗ FAILED", fg='red'))
        for failure in results["failures"]:
            click.echo(f"  - {failure}")
    
    # Generate report
    click.echo(f"\nGenerating report: {output}")
    report_gen = ReportGenerator()
    report_gen.generate_html(scenario, results, output)
    
    click.echo(f"\nReport saved: {output}")

@cli.command()
@click.option('--output-dir', '-o', default='reports/', help='Output directory')
def run_all(output_dir):
    """Run all scenarios (test suite)."""
    click.echo("Running full test suite...")
    
    results = {}
    for name, scenario_class in SCENARIOS.items():
        click.echo(f"\n{'='*60}")
        click.echo(f"Scenario: {name}")
        click.echo('='*60)
        
        scenario = scenario_class()
        result = scenario.execute()
        results[name] = result
        
        status = "✓ PASSED" if result["passed"] else "✗ FAILED"
        click.echo(f"Result: {status}")
    
    # Summary
    click.echo("\n" + "="*60)
    click.echo("SUMMARY")
    click.echo("="*60)
    
    passed = sum(1 for r in results.values() if r["passed"])
    total = len(results)
    
    click.echo(f"Passed: {passed}/{total}")
    
    if passed == total:
        click.echo(click.style("\n✓ ALL SCENARIOS PASSED", fg='green', bold=True))
    else:
        click.echo(click.style(f"\n✗ {total - passed} SCENARIOS FAILED", fg='red', bold=True))

if __name__ == '__main__':
    cli()
```

---

### Phase 4: Integration with Existing System

#### 4.1 Docker Compose for Simulator

**File: `docker-compose.simulator.yml`**

```yaml
version: '3.9'

services:
  simulator:
    build:
      context: .
      dockerfile: simulator/Dockerfile
    container_name: ibook-simulator
    environment:
      ENVIRONMENT: local
      API_BASE_URL: http://bentoml-fraud-detection:3000
      MLOPS_STACK_URL: http://mlflow:5000
    volumes:
      - ./simulator:/app/simulator
      - ./data:/app/data
      - ./reports:/app/reports
    depends_on:
      - bentoml-fraud-detection
      - mlflow
    command: tail -f /dev/null  # Keep running for interactive use
    networks:
      - mlops-network

  # Simulator Dashboard (Streamlit)
  simulator-dashboard:
    build:
      context: ./simulator
      dockerfile: Dockerfile.dashboard
    container_name: ibook-simulator-dashboard
    ports:
      - "8501:8501"
    environment:
      STREAMLIT_SERVER_PORT: 8501
    volumes:
      - ./reports:/app/reports
    networks:
      - mlops-network

networks:
  mlops-network:
    external: true
```

---

#### 4.2 Makefile Updates

Add to existing `Makefile`:

```makefile
# Simulator commands
.PHONY: sim-start sim-list sim-run sim-run-all sim-dashboard

sim-start:
	@echo "Starting simulator..."
	@docker-compose -f docker-compose.simulator.yml up -d
	@echo "✓ Simulator started"
	@echo "Dashboard: http://localhost:8501"

sim-list:
	@docker exec ibook-simulator python -m simulator.cli list-scenarios

sim-run:
	@echo "Running scenario: $(scenario)"
	@docker exec ibook-simulator python -m simulator.cli run $(scenario)

sim-run-all:
	@echo "Running all scenarios..."
	@docker exec ibook-simulator python -m simulator.cli run-all

sim-dashboard:
	@open http://localhost:8501

sim-stop:
	@docker-compose -f docker-compose.simulator.yml down

# Example usage:
# make sim-run scenario=flash-sale
# make sim-run-all
```

---

### Phase 5: Validation & Reporting

#### 5.1 Report Generator

**AI Prompt:**
```
Create simulator/visualizers/report_generator.py that:
1. Generates HTML reports with charts (Plotly)
2. Includes: latency histograms, error rate over time, fraud detection metrics
3. Compares actual vs expected metrics
4. Shows pass/fail status clearly
5. Exports to PDF (optional)

Use Jinja2 templates for HTML generation.
```

**File: `simulator/visualizers/report_generator.py`**

```python
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from jinja2 import Template
from datetime import datetime
from typing import Dict
import json

class ReportGenerator:
    """Generate HTML/PDF reports for simulation results."""
    
    def __init__(self):
        self.template = self._load_template()
    
    def generate_html(self, scenario, results: Dict, output_path: str):
        """Generate HTML report."""
        
        # Create charts
        charts = {
            "latency": self._create_latency_chart(results),
            "throughput": self._create_throughput_chart(results),
            "errors": self._create_error_chart(results),
            "fraud": self._create_fraud_chart(results),
        }
        
        # Render template
        html = self.template.render(
            scenario_name=scenario.name,
            description=scenario.description,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            results=results,
            charts=charts,
        )
        
        with open(output_path, 'w') as f:
            f.write(html)
    
    def _create_latency_chart(self, results: Dict) -> str:
        """Create latency histogram."""
        responses = results.get("responses", [])
        latencies = [r.get("latency_ms", 0) for r in responses]
        
        fig = go.Figure(data=[go.Histogram(x=latencies, nbinsx=50)])
        fig.update_layout(
            title="Latency Distribution",
            xaxis_title="Latency (ms)",
            yaxis_title="Count",
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="latency-chart")
    
    def _create_throughput_chart(self, results: Dict) -> str:
        """Create throughput over time chart."""
        # Simplified - in real implementation, bucket by time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=[results.get("peak_rps", 0)],
            mode='lines+markers',
            name='RPS'
        ))
        fig.update_layout(title="Throughput Over Time")
        
        return fig.to_html(include_plotlyjs='cdn', div_id="throughput-chart")
    
    def _create_error_chart(self, results: Dict) -> str:
        """Create error rate chart."""
        error_rate = results.get("error_rate", 0) * 100
        
        fig = go.Figure(data=[
            go.Bar(x=["Error Rate"], y=[error_rate])
        ])
        fig.update_layout(
            title="Error Rate",
            yaxis_title="Percentage (%)"
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="error-chart")
    
    def _create_fraud_chart(self, results: Dict) -> str:
        """Create fraud detection metrics."""
        fraud_detected = results.get("fraud_detected_pct", 0)
        
        fig = go.Figure(data=[
            go.Bar(
                x=["Detected"],
                y=[fraud_detected],
                marker_color='green'
            )
        ])
        fig.update_layout(
            title="Fraud Detection Rate",
            yaxis_title="Percentage (%)"
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="fraud-chart")
    
    def _load_template(self) -> Template:
        """Load HTML template."""
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>Simulation Report - {{ scenario_name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
        .metric-card { background: #ecf0f1; padding: 20px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
        .status-pass { color: #27ae60; }
        .status-fail { color: #e74c3c; }
        .chart-container { margin: 30px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ scenario_name }}</h1>
        <p>{{ description }}</p>
        <p><small>Generated: {{ timestamp }}</small></p>
    </div>
    
    <h2>Overall Status: 
        <span class="{% if results.passed %}status-pass{% else %}status-fail{% endif %}">
            {% if results.passed %}✓ PASSED{% else %}✗ FAILED{% endif %}
        </span>
    </h2>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{{ "%.0f"|format(results.get('p99_latency_ms', 0)) }}</div>
            <div>p99 Latency (ms)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.2f"|format(results.get('error_rate', 0) * 100) }}%</div>
            <div>Error Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.0f"|format(results.get('peak_rps', 0)) }}</div>
            <div>Peak RPS</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.0f"|format(results.get('fraud_detected_pct', 0)) }}%</div>
            <div>Fraud Detected</div>
        </div>
    </div>
    
    <div class="chart-container">{{ charts.latency|safe }}</div>
    <div class="chart-container">{{ charts.throughput|safe }}</div>
    <div class="chart-container">{{ charts.errors|safe }}</div>
    <div class="chart-container">{{ charts.fraud|safe }}</div>
    
    {% if not results.passed %}
    <h2>Failures:</h2>
    <ul>
    {% for failure in results.failures %}
        <li class="status-fail">{{ failure }}</li>
    {% endfor %}
    </ul>
    {% endif %}
</body>
</html>
        """
        return Template(template_str)
```

---

## 📋 Updated PLAN.md Integration

Add to **Phase 10** in existing PLAN.md:

### Phase 10.3: Simulator Setup (Days 36-40)

**Week 9: Simulator Development**

**Day 36-37: Core Simulator**
- [ ] Create simulator configuration
- [ ] Implement event generator
- [ ] Implement user generator
- [ ] Implement transaction generator
- [ ] Test data generation locally

**Day 38-39: Scenarios**
- [ ] Create base scenario class
- [ ] Implement flash sale scenario
- [ ] Implement normal traffic scenario
- [ ] Implement fraud attack scenario
- [ ] Test scenarios against local stack

**Day 40: Integration & Reporting**
- [ ] Create CLI interface
- [ ] Integrate with Docker Compose
- [ ] Build report generator
- [ ] Run full test suite
- [ ] Document usage

---

## Duration configuration and mix/realtime modes

- **Duration override:** Use `--duration N` (minutes) with `run` or `run-all` to scale scenario workload (e.g. `python -m simulator.cli run normal-traffic --duration 10`).
- **Mix mode:** Run a weighted combination of scenarios for a set duration: `python -m simulator.cli mix --duration 30 --scenarios "normal-traffic:40,flash-sale:20,fraud-attack:20,system-degradation:10,black-friday:10"`. See `simulator/scenarios/mixed.py` (MixedScenario).
- **Realtime runner:** Stream traffic at a fixed RPS for wall-clock seconds: `python -m simulator.cli realtime normal-traffic --duration 60 --rps 100`. See `simulator/runners/realtime_runner.py`; supports graceful Ctrl+C.

For detailed testing steps, expected results, and tool usage, see **[docs/simulator_testing.md](simulator_testing.md)**.

---

## 🎯 Usage Examples

### Run Single Scenario

```bash
# List available scenarios
make sim-list

# Run flash sale scenario
make sim-run scenario=flash-sale

# Run with custom duration (minutes)
python -m simulator.cli run flash-sale --duration 10 -o reports/flash-sale-report.html

# View report
open reports/flash-sale-report.html
```

### Run Full Test Suite

```bash
# Run all scenarios
make sim-run-all

# View dashboard
make sim-dashboard
```

### Fraud Drift Auto-Retrain Scenario

```bash
# Run the fraud drift scenario to test auto-retraining
python -m simulator.cli run fraud-drift-retrain -o reports/fraud-drift-retrain-report.html

# Run in realtime mode against the live BentoML service
python -m simulator.cli realtime fraud-drift-retrain --duration 120 --rps 50

# Combine with other scenarios
python -m simulator.cli mix --duration 30 --scenarios "normal-traffic:40,fraud-drift-retrain:30,fraud-attack:30"
```

### Mix and realtime

```bash
# Mix mode (weighted scenarios, 30 min default)
make sim-mix
make sim-mix duration=15
python -m simulator.cli mix --duration 30 -o reports/mix-report.html

# Realtime (wall-clock seconds, configurable RPS)
make sim-realtime
make sim-realtime scenario=normal-traffic duration=120 rps=200
python -m simulator.cli realtime normal-traffic --duration 60 --rps 100 -o reports/realtime-report.html
```

### Custom Scenario Execution

```bash
# Run with custom duration (minutes)
python -m simulator.cli run flash-sale --duration 30 -o reports/flash-sale-report.html

# Dry run (setup only)
python -m simulator.cli run flash-sale --dry-run
```

### Integration with CI/CD

Add to `.github/workflows/mlops-cicd.yml`:

```yaml
  simulation-tests:
    name: Run Simulation Tests
    needs: deploy-staging
    runs-on: ubuntu-latest
    
    steps:
      - name: Run simulator against staging
        run: |
          docker-compose -f docker-compose.simulator.yml up -d
          docker exec ibook-simulator python -m simulator.cli run-all --env staging
      
      - name: Check results
        run: |
          if grep -q "FAILED" reports/summary.txt; then
            echo "Simulation tests failed"
            exit 1
          fi
      
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: simulation-reports
          path: reports/
```

---

## 🔥 Chaos Engineering Extension

### Optional: Add Chaos Scenarios

**`simulator/scenarios/chaos/redis_failure.py`**
- Kill Redis during peak load
- Test fallback to batch features

**`simulator/scenarios/chaos/model_timeout.py`**
- Inject latency into ML model
- Test circuit breakers

**`simulator/scenarios/chaos/network_partition.py`**
- Simulate network split
- Test service mesh resilience

---

## 📊 Success Criteria

**Simulator should validate:**
- [ ] System handles 10K+ RPS sustained
- [ ] p99 latency < 200ms under load
- [ ] Error rate < 1% during flash sales
- [ ] Fraud detection recall > 95%
- [ ] Graceful degradation during failures
- [ ] Auto-retraining triggers on drift
- [ ] Auto-retraining triggers on failure rate breach (novel fraud patterns)
- [ ] Model hot-reload completes without downtime
- [ ] Zero data loss during chaos events

---

## 🎓 Benefits

1. **Reproducible Testing:** Same scenarios, different environments
2. **Confidence in Production:** Test before mega-events
3. **Performance Baseline:** Know your limits
4. **Fraud Validation:** Verify detection accuracy
5. **Chaos Readiness:** Find failures before users do
6. **Documentation:** Reports serve as system proof

This simulator transforms your MLOps platform from "hopefully works" to **"proven at scale"**! 🚀
