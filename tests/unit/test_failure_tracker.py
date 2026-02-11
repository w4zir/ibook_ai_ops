"""Tests for the sliding-window failure tracker."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from services.bentoml.common.failure_tracker import FailureTracker, PredictionOutcome


# ---------------------------------------------------------------------------
# PredictionOutcome
# ---------------------------------------------------------------------------


class TestPredictionOutcome:
    def test_correct_when_match(self) -> None:
        o = PredictionOutcome(timestamp=0, predicted_fraud=True, actual_fraud=True)
        assert o.is_correct is True

    def test_incorrect_false_negative(self) -> None:
        o = PredictionOutcome(timestamp=0, predicted_fraud=False, actual_fraud=True)
        assert o.is_correct is False

    def test_incorrect_false_positive(self) -> None:
        o = PredictionOutcome(timestamp=0, predicted_fraud=True, actual_fraud=False)
        assert o.is_correct is False

    def test_correct_both_false(self) -> None:
        o = PredictionOutcome(timestamp=0, predicted_fraud=False, actual_fraud=False)
        assert o.is_correct is True


# ---------------------------------------------------------------------------
# FailureTracker basics
# ---------------------------------------------------------------------------


class TestFailureTrackerBasics:
    def test_empty_tracker_returns_zero(self) -> None:
        tracker = FailureTracker(window_seconds=60, min_samples=5)
        rate, n = tracker.get_failure_rate()
        assert rate == 0.0
        assert n == 0

    def test_below_min_samples_returns_zero(self) -> None:
        tracker = FailureTracker(window_seconds=60, min_samples=10)
        for _ in range(5):
            tracker.record(predicted_fraud=True, actual_fraud=True)
        rate, n = tracker.get_failure_rate()
        assert rate == 0.0
        assert n == 5

    def test_all_correct_returns_zero_rate(self) -> None:
        tracker = FailureTracker(window_seconds=60, min_samples=3)
        for _ in range(10):
            tracker.record(predicted_fraud=False, actual_fraud=False)
        rate, n = tracker.get_failure_rate()
        assert rate == 0.0
        assert n == 10

    def test_all_incorrect_returns_one(self) -> None:
        tracker = FailureTracker(window_seconds=60, min_samples=3)
        for _ in range(10):
            tracker.record(predicted_fraud=True, actual_fraud=False)
        rate, n = tracker.get_failure_rate()
        assert rate == 1.0
        assert n == 10

    def test_mixed_outcomes(self) -> None:
        tracker = FailureTracker(window_seconds=60, min_samples=2)
        # 6 correct + 4 incorrect = 0.4 failure rate
        for _ in range(6):
            tracker.record(predicted_fraud=True, actual_fraud=True)
        for _ in range(4):
            tracker.record(predicted_fraud=True, actual_fraud=False)
        rate, n = tracker.get_failure_rate()
        assert abs(rate - 0.4) < 1e-9
        assert n == 10

    def test_reset_clears_data(self) -> None:
        tracker = FailureTracker(window_seconds=60, min_samples=1)
        for _ in range(5):
            tracker.record(predicted_fraud=True, actual_fraud=False)
        tracker.reset()
        rate, n = tracker.get_failure_rate()
        assert rate == 0.0
        assert n == 0


# ---------------------------------------------------------------------------
# Sliding window pruning
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_old_entries_are_pruned(self) -> None:
        tracker = FailureTracker(window_seconds=5, min_samples=1)
        now = time.time()
        # Record entries 10 seconds in the past (outside the 5s window).
        for i in range(5):
            tracker.record(
                predicted_fraud=True,
                actual_fraud=False,
                timestamp=now - 10 + i * 0.1,
            )
        # Record entries within the window.
        for i in range(3):
            tracker.record(
                predicted_fraud=True,
                actual_fraud=True,
                timestamp=now - 1 + i * 0.1,
            )
        rate, n = tracker.get_failure_rate()
        # Only the 3 recent (correct) entries should remain.
        assert n == 3
        assert rate == 0.0


# ---------------------------------------------------------------------------
# Threshold callback
# ---------------------------------------------------------------------------


class TestThresholdCallback:
    def test_callback_fires_when_threshold_breached(self) -> None:
        callback = MagicMock()
        tracker = FailureTracker(
            window_seconds=60,
            failure_rate_threshold=0.3,
            cooldown_seconds=0,
            min_samples=5,
            on_threshold_breached=callback,
        )
        # Record 5 failures out of 5 → 100% failure rate.
        for _ in range(5):
            tracker.record(predicted_fraud=True, actual_fraud=False)

        # The callback is fired in a daemon thread; wait for it.
        time.sleep(0.3)
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == 1.0  # failure_rate
        assert args[1] == 5  # n_samples

    def test_callback_not_fired_below_threshold(self) -> None:
        callback = MagicMock()
        tracker = FailureTracker(
            window_seconds=60,
            failure_rate_threshold=0.5,
            cooldown_seconds=0,
            min_samples=5,
            on_threshold_breached=callback,
        )
        # 2 failures out of 10 = 20% → below 50% threshold.
        for _ in range(8):
            tracker.record(predicted_fraud=True, actual_fraud=True)
        for _ in range(2):
            tracker.record(predicted_fraud=True, actual_fraud=False)

        time.sleep(0.3)
        callback.assert_not_called()

    def test_cooldown_prevents_rapid_callbacks(self) -> None:
        callback = MagicMock()
        tracker = FailureTracker(
            window_seconds=60,
            failure_rate_threshold=0.3,
            cooldown_seconds=10,  # 10 second cooldown
            min_samples=3,
            on_threshold_breached=callback,
        )
        # Trigger first breach.
        for _ in range(5):
            tracker.record(predicted_fraud=True, actual_fraud=False)
        time.sleep(0.3)

        # Immediately try again – should be suppressed by cooldown.
        callback.reset_mock()
        # Reset training_in_progress so it doesn't block due to that flag.
        tracker.training_in_progress = False
        for _ in range(5):
            tracker.record(predicted_fraud=True, actual_fraud=False)
        time.sleep(0.3)
        callback.assert_not_called()

    def test_training_in_progress_blocks_trigger(self) -> None:
        callback = MagicMock()
        tracker = FailureTracker(
            window_seconds=60,
            failure_rate_threshold=0.3,
            cooldown_seconds=0,
            min_samples=3,
            on_threshold_breached=callback,
        )
        tracker.training_in_progress = True
        for _ in range(5):
            tracker.record(predicted_fraud=True, actual_fraud=False)
        time.sleep(0.3)
        callback.assert_not_called()

    def test_callback_exception_resets_training_flag(self) -> None:
        def bad_callback(rate: float, n: int) -> None:
            raise RuntimeError("boom")

        tracker = FailureTracker(
            window_seconds=60,
            failure_rate_threshold=0.3,
            cooldown_seconds=0,
            min_samples=3,
            on_threshold_breached=bad_callback,
        )
        for _ in range(5):
            tracker.record(predicted_fraud=True, actual_fraud=False)
        time.sleep(0.5)
        # Even after callback fails, training_in_progress should be reset.
        assert tracker.training_in_progress is False
