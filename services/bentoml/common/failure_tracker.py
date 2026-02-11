from __future__ import annotations

"""
Thread-safe sliding-window tracker for fraud prediction outcomes.

The ``FailureTracker`` records each prediction outcome (correct / incorrect)
with a timestamp.  At any point the caller can query the *failure rate* over
the most recent ``window_seconds`` and decide whether to trigger retraining.

Design goals:
- **Lock-free reads are not required** – correctness matters more than
  throughput here, and a simple ``threading.Lock`` is fine for the expected
  call rate (hundreds / sec, not millions).
- **Memory-bounded** – stale entries outside the window are pruned lazily on
  every query so the deque never grows unboundedly.
- **Callback-driven** – callers can register an ``on_threshold_breached``
  callback that fires *at most once per cooldown period* when the failure
  rate crosses the configured threshold.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PredictionOutcome:
    """A single prediction outcome recorded by the tracker."""

    timestamp: float
    predicted_fraud: bool
    actual_fraud: bool

    @property
    def is_correct(self) -> bool:
        return self.predicted_fraud == self.actual_fraud


class FailureTracker:
    """
    Sliding-window failure-rate tracker with optional threshold callback.

    Parameters
    ----------
    window_seconds:
        Size of the sliding window in seconds.
    failure_rate_threshold:
        Failure rate (0–1) that triggers the ``on_threshold_breached`` callback.
    cooldown_seconds:
        Minimum elapsed seconds between consecutive callback invocations.
    min_samples:
        Minimum number of samples in the window before the failure rate is
        considered meaningful.
    on_threshold_breached:
        Optional callback invoked (in a daemon thread) when the failure rate
        crosses the threshold.  Receives ``(failure_rate, n_samples)`` as args.
    """

    def __init__(
        self,
        window_seconds: int = 300,
        failure_rate_threshold: float = 0.4,
        cooldown_seconds: int = 120,
        min_samples: int = 20,
        on_threshold_breached: Optional[Callable[[float, int], None]] = None,
    ) -> None:
        self._window_seconds = window_seconds
        self._threshold = failure_rate_threshold
        self._cooldown = cooldown_seconds
        self._min_samples = min_samples
        self._on_threshold_breached = on_threshold_breached

        self._outcomes: Deque[PredictionOutcome] = deque()
        self._lock = threading.Lock()
        self._last_trigger_time: float = 0.0
        self._training_in_progress = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        predicted_fraud: bool,
        actual_fraud: bool,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a single prediction outcome and check the threshold."""
        ts = timestamp if timestamp is not None else time.time()
        outcome = PredictionOutcome(
            timestamp=ts,
            predicted_fraud=predicted_fraud,
            actual_fraud=actual_fraud,
        )
        with self._lock:
            self._outcomes.append(outcome)
            self._prune_locked()

        # Check threshold *after* releasing the lock so the callback can
        # safely call ``get_failure_rate`` without deadlocking.
        self._maybe_trigger()

    def get_failure_rate(self) -> Tuple[float, int]:
        """
        Return ``(failure_rate, n_samples)`` over the current window.

        If there are fewer than ``min_samples`` outcomes, the failure rate is
        reported as ``0.0`` (i.e. "not enough data to decide").
        """
        with self._lock:
            self._prune_locked()
            n = len(self._outcomes)
            if n < self._min_samples:
                return 0.0, n
            failures = sum(1 for o in self._outcomes if not o.is_correct)
            return failures / n, n

    @property
    def training_in_progress(self) -> bool:
        return self._training_in_progress

    @training_in_progress.setter
    def training_in_progress(self, value: bool) -> None:
        self._training_in_progress = value

    def reset(self) -> None:
        """Clear all recorded outcomes (useful after a model reload)."""
        with self._lock:
            self._outcomes.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prune_locked(self) -> None:
        """Remove entries older than the window. Caller must hold ``_lock``."""
        cutoff = time.time() - self._window_seconds
        while self._outcomes and self._outcomes[0].timestamp < cutoff:
            self._outcomes.popleft()

    def _maybe_trigger(self) -> None:
        """Check the failure rate and fire the callback if warranted."""
        if self._on_threshold_breached is None:
            return
        if self._training_in_progress:
            return

        failure_rate, n_samples = self.get_failure_rate()
        if n_samples < self._min_samples:
            return
        if failure_rate < self._threshold:
            return

        now = time.time()
        if (now - self._last_trigger_time) < self._cooldown:
            return

        # Mark as triggered *before* launching the callback thread so that
        # concurrent calls to ``record`` don't fire duplicates.
        self._last_trigger_time = now
        self._training_in_progress = True

        logger.warning(
            "Failure rate %.2f%% (%d samples) exceeds threshold %.2f%% – triggering auto-retrain.",
            failure_rate * 100,
            n_samples,
            self._threshold * 100,
        )

        # Fire the callback in a daemon thread so it never blocks the
        # serving hot path.
        t = threading.Thread(
            target=self._run_callback,
            args=(failure_rate, n_samples),
            daemon=True,
            name="auto-retrain-trigger",
        )
        t.start()

    def _run_callback(self, failure_rate: float, n_samples: int) -> None:
        try:
            assert self._on_threshold_breached is not None
            self._on_threshold_breached(failure_rate, n_samples)
        except Exception:
            logger.exception("Auto-retrain callback failed.")
        finally:
            self._training_in_progress = False


__all__ = ["FailureTracker", "PredictionOutcome"]
