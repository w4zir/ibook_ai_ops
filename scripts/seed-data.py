from __future__ import annotations

"""
Synthetic data generator for the Ibook MLOps feature store.

This script creates Parquet files under `data/processed/feast/` that are used
by the Feast feature repository in `services/feast/feature_repo`.

Generated datasets:
- event_metrics.parquet
- user_metrics.parquet
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEAST_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "feast"


def _init_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _generate_events(rng: np.random.Generator, n_events: int) -> pd.DataFrame:
    categories = ["sports", "concerts", "family", "cultural"]
    base_start = datetime.now(timezone.utc) - timedelta(days=30)

    event_ids = np.arange(1, n_events + 1)
    promoters = rng.integers(1, 21, size=n_events)
    event_categories = rng.choice(categories, size=n_events, replace=True)
    start_times = [
        base_start + timedelta(days=int(offset))
        for offset in rng.integers(-10, 20, size=n_events)
    ]
    capacities = rng.integers(500, 5000, size=n_events)
    base_prices = rng.uniform(50, 500, size=n_events)

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "promoter_id": promoters,
            "category": event_categories,
            "start_time": start_times,
            "capacity": capacities,
            "base_price": base_prices,
        }
    )


def _generate_users(rng: np.random.Generator, n_users: int) -> pd.DataFrame:
    user_ids = np.arange(1, n_users + 1)
    loyalty_tier = rng.choice(["bronze", "silver", "gold", "platinum"], size=n_users)
    signup_offset_days = rng.integers(30, 365 * 3, size=n_users)
    signup_dates = [
        datetime.now(timezone.utc) - timedelta(days=int(d)) for d in signup_offset_days
    ]

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "loyalty_tier": loyalty_tier,
            "signup_date": signup_dates,
        }
    )


def _generate_transactions(
    rng: np.random.Generator,
    events: pd.DataFrame,
    users: pd.DataFrame,
    n_transactions: int,
) -> pd.DataFrame:
    """
    Generate synthetic ticket transactions with simple seasonal patterns and fraud.
    """
    event_ids = events["event_id"].to_numpy()
    user_ids = users["user_id"].to_numpy()

    chosen_events = rng.choice(event_ids, size=n_transactions, replace=True)
    chosen_users = rng.choice(user_ids, size=n_transactions, replace=True)

    now = datetime.now(timezone.utc)
    timestamps = [
        now - timedelta(days=int(d), minutes=int(m))
        for d, m in zip(
            rng.integers(0, 60, size=n_transactions),
            rng.integers(0, 24 * 60, size=n_transactions),
        )
    ]

    base_price_lookup = events.set_index("event_id")["base_price"]
    base_prices = base_price_lookup.reindex(chosen_events).to_numpy()
    surge = rng.uniform(0.9, 1.5, size=n_transactions)
    ticket_prices = base_prices * surge

    quantities = rng.integers(1, 5, size=n_transactions)

    # Simple fraud model: higher fraud risk for high ticket price and low loyalty tier.
    user_tier_lookup = users.set_index("user_id")["loyalty_tier"]
    user_tiers = user_tier_lookup.reindex(chosen_users).to_numpy()
    fraud_base_prob = 0.03  # ~3% base rate
    fraud_boost = np.where(ticket_prices > 300, 0.05, 0.0)
    fraud_penalty = np.where(user_tiers == "platinum", -0.02, 0.0)
    fraud_prob = np.clip(fraud_base_prob + fraud_boost + fraud_penalty, 0.001, 0.3)
    is_fraud = rng.binomial(1, fraud_prob).astype(bool)

    return pd.DataFrame(
        {
            "event_timestamp": timestamps,
            "event_id": chosen_events,
            "user_id": chosen_users,
            "quantity": quantities,
            "ticket_price": ticket_prices,
            "is_fraud": is_fraud,
        }
    )


def _build_event_metrics(
    events: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    # Aggregate per-event metrics from transactions.
    tx_grouped = transactions.groupby("event_id").agg(
        total_tickets_sold=("quantity", "sum"),
        avg_ticket_price=("ticket_price", "mean"),
        concurrent_viewers=("quantity", "max"),
    )

    df = events.merge(tx_grouped, left_on="event_id", right_index=True, how="left")
    df["total_tickets_sold"] = df["total_tickets_sold"].fillna(0).astype(int)
    df["avg_ticket_price"] = df["avg_ticket_price"].fillna(df["base_price"])
    df["concurrent_viewers"] = df["concurrent_viewers"].fillna(0).astype(int)

    # Simple sell-through approximation and current inventory.
    df["sell_through_rate_5min"] = np.clip(
        df["total_tickets_sold"] / df["capacity"], 0.0, 1.0
    )
    df["current_inventory"] = (df["capacity"] - df["total_tickets_sold"]).clip(lower=0)

    # Promoter success as normalized tickets sold per promoter.
    promoter_sales = df.groupby("promoter_id")["total_tickets_sold"].transform("mean")
    max_sales = promoter_sales.max() or 1
    df["promoter_success_rate"] = (promoter_sales / max_sales).fillna(0.0)

    # Feast requires an event_timestamp column on the source.
    df["event_timestamp"] = pd.to_datetime(df["start_time"], utc=True)
    df["ingested_at"] = datetime.now(timezone.utc)

    return df[
        [
            "event_id",
            "promoter_id",
            "event_timestamp",
            "current_inventory",
            "sell_through_rate_5min",
            "concurrent_viewers",
            "total_tickets_sold",
            "avg_ticket_price",
            "promoter_success_rate",
        ]
    ].sort_values(["event_id"])


def _build_user_metrics(
    users: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    tx_grouped = transactions.groupby("user_id").agg(
        lifetime_purchases=("quantity", "sum"),
        fraud_rate=("is_fraud", "mean"),
    )
    df = users.merge(tx_grouped, left_on="user_id", right_index=True, how="left")
    df["lifetime_purchases"] = df["lifetime_purchases"].fillna(0).astype(int)

    # Fraud risk score scaled 0-1.
    df["fraud_risk_score"] = df["fraud_rate"].fillna(0.0).clip(0.0, 1.0)

    # Preferred category is random for now; in a real system this would be derived.
    categories = ["sports", "concerts", "family", "cultural"]
    rng = _init_rng(999)
    df["preferred_category"] = rng.choice(categories, size=len(df))

    # Use signup_date as a stable timestamp for point-in-time joins.
    df["event_timestamp"] = pd.to_datetime(df["signup_date"], utc=True)
    df["ingested_at"] = datetime.now(timezone.utc)

    return df[
        [
            "user_id",
            "event_timestamp",
            "lifetime_purchases",
            "fraud_risk_score",
            "preferred_category",
        ]
    ].sort_values(["user_id"])


def generate_synthetic_data(
    n_events: int = 100, n_users: int = 1000, n_transactions: int = 10_000, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic event and user metrics for Feast.

    Returns:
        (event_metrics_df, user_metrics_df)
    """
    rng = _init_rng(seed)
    events = _generate_events(rng, n_events=n_events)
    users = _generate_users(rng, n_users=n_users)
    transactions = _generate_transactions(
        rng, events=events, users=users, n_transactions=n_transactions
    )

    event_metrics = _build_event_metrics(events, transactions)
    user_metrics = _build_user_metrics(users, transactions)
    return event_metrics, user_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Feast data.")
    parser.add_argument("--events", type=int, default=100, help="Number of events.")
    parser.add_argument("--users", type=int, default=1000, help="Number of users.")
    parser.add_argument(
        "--transactions", type=int, default=10_000, help="Number of transactions."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    FEAST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    event_metrics, user_metrics = generate_synthetic_data(
        n_events=args.events,
        n_users=args.users,
        n_transactions=args.transactions,
        seed=args.seed,
    )

    event_path = FEAST_DATA_DIR / "event_metrics.parquet"
    user_path = FEAST_DATA_DIR / "user_metrics.parquet"

    event_metrics.to_parquet(event_path, index=False)
    user_metrics.to_parquet(user_path, index=False)

    print(f"Wrote {len(event_metrics)} event rows to {event_path}")
    print(f"Wrote {len(user_metrics)} user rows to {user_path}")


if __name__ == "__main__":
    main()

