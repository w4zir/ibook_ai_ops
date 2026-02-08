"""Generate realistic ticketing events (Saudi context, pricing tiers, popularity)."""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np

from simulator.config import EventCategory, config

try:
    from faker import Faker
    fake = Faker(["ar_SA", "en_US"])
except Exception:
    fake = None


def _fake_uuid4() -> str:
    if fake:
        return fake.uuid4()[:8]
    return f"{random.getrandbits(32):08x}"


def _fake_catch_phrase() -> str:
    if fake:
        return fake.catch_phrase()
    return "Event Title"


class EventGenerator:
    """Generate realistic ticketing events."""

    def __init__(self) -> None:
        self.event_templates: Dict[EventCategory, List[str]] = {
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
            EventCategory.COMEDY: [
                "Comedy Night: {venue}",
                "Stand-up Show - Riyadh",
                "Arabic Comedy Festival",
            ],
            EventCategory.FESTIVAL: [
                "Riyadh Season Opening",
                "Diriyah Festival",
                "Winter Festival {venue}",
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
            "Ed Sheeran", "Bruno Mars", "Coldplay",
        ]

    def generate_event(
        self,
        category: EventCategory | None = None,
        popularity_override: float | None = None,
    ) -> Dict[str, Any]:
        """Generate a single event."""
        if category is None:
            categories = list(config.event_categories_distribution.keys())
            probs = list(config.event_categories_distribution.values())
            idx = np.random.choice(len(categories), p=probs)
            category = categories[idx]

        templates = self.event_templates.get(category, self.event_templates[EventCategory.CONCERT])
        template = random.choice(templates)

        event_name = template.format(
            team1=random.choice(self.teams),
            team2=random.choice(self.teams),
            artist=random.choice(self.artists),
            game="League of Legends",
            venue=random.choice(self.venues),
            show=random.choice(["Frozen", "Aladdin", "Lion King"]),
            topic=random.choice(["Science", "Art", "Coding"]),
            title=_fake_catch_phrase(),
        )

        if popularity_override is not None:
            popularity = popularity_override
        else:
            popularity = float(np.random.power(2))

        if popularity > 0.9:
            capacity = random.randint(20000, 100000)
            price_range = (200.0, 2000.0)
        elif popularity > 0.7:
            capacity = random.randint(5000, 20000)
            price_range = (100.0, 800.0)
        else:
            capacity = random.randint(500, 5000)
            price_range = (50.0, 300.0)

        announcement_date = datetime.now() - timedelta(days=random.randint(7, 60))
        presale_date = announcement_date + timedelta(days=random.randint(3, 14))
        general_sale_date = presale_date + timedelta(days=random.randint(1, 7))
        event_date = general_sale_date + timedelta(days=random.randint(14, 90))

        return {
            "event_id": f"event_{_fake_uuid4()}",
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
            },
        }

    def _generate_pricing_tiers(self, price_range: tuple[float, float], capacity: int) -> List[Dict[str, Any]]:
        min_price, max_price = price_range
        return [
            {
                "tier": "VIP",
                "price": max_price,
                "capacity": int(capacity * 0.10),
                "benefits": ["Meet & Greet", "Premium Seating", "VIP Lounge"],
            },
            {
                "tier": "Premium",
                "price": (min_price + max_price) / 2,
                "capacity": int(capacity * 0.30),
                "benefits": ["Priority Entry", "Better View"],
            },
            {
                "tier": "Standard",
                "price": min_price,
                "capacity": int(capacity * 0.60),
                "benefits": [],
            },
        ]

    def _estimate_sellout_time(self, popularity: float, capacity: int) -> float:
        base_rate = 100.0
        if popularity > 0.9:
            rate = base_rate * 1000
        elif popularity > 0.7:
            rate = base_rate * 100
        else:
            rate = base_rate
        return capacity / rate

    def generate_batch(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate multiple events."""
        return [self.generate_event() for _ in range(count)]

    def generate_mega_event(self) -> Dict[str, Any]:
        """Generate a mega-event (Riyadh Season style)."""
        return self.generate_event(
            category=random.choice([EventCategory.CONCERT, EventCategory.SPORTS]),
            popularity_override=0.95,
        )
