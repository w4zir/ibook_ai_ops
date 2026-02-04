from __future__ import annotations

from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64, String

from common.config import get_config
from services.feast.feature_repo.data_sources import (
    get_event_metrics_source,
    get_user_metrics_source,
)


_cfg = get_config()

# Entities
event = Entity(name="event", join_keys=["event_id"])
user = Entity(name="user", join_keys=["user_id"])
promoter = Entity(name="promoter", join_keys=["promoter_id"])


_event_source = get_event_metrics_source(_cfg)
_user_source = get_user_metrics_source(_cfg)


event_realtime_metrics = FeatureView(
    name="event_realtime_metrics",
    entities=[event],
    ttl=timedelta(minutes=60),
    schema=[
        Field(name="current_inventory", dtype=Int64),
        Field(name="sell_through_rate_5min", dtype=Float32),
        Field(name="concurrent_viewers", dtype=Int64),
    ],
    online=True,
    batch_source=_event_source,
)


event_historical_metrics = FeatureView(
    name="event_historical_metrics",
    entities=[event, promoter],
    ttl=timedelta(days=365),
    schema=[
        Field(name="total_tickets_sold", dtype=Int64),
        Field(name="avg_ticket_price", dtype=Float32),
        Field(name="promoter_success_rate", dtype=Float32),
    ],
    online=True,
    batch_source=_event_source,
)


user_purchase_behavior = FeatureView(
    name="user_purchase_behavior",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="lifetime_purchases", dtype=Int64),
        Field(name="fraud_risk_score", dtype=Float32),
        Field(name="preferred_category", dtype=String),
    ],
    online=True,
    batch_source=_user_source,
)


__all__ = [
    "event",
    "user",
    "promoter",
    "event_realtime_metrics",
    "event_historical_metrics",
    "user_purchase_behavior",
]

