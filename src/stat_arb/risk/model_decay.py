"""Model decay monitoring for Kalman filter and half-life trends.

Tracks Kalman filter fallback rates and half-life drift to detect
when the model parameters are degrading over time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class DecayMetrics:
    """Immutable snapshot of model decay indicators.

    Attributes:
        kalman_fallback_rate: Fraction of recent Kalman updates that fell
            back to OLS (0 = all Kalman, 1 = all OLS).
        median_half_life: Median half-life over the lookback window.
        half_life_trend: Second-half median minus first-half median.
            Positive = half-lives increasing (slower reversion = decay).
        totals: Total number of recorded outcomes.
    """

    kalman_fallback_rate: float
    median_half_life: float
    half_life_trend: float
    totals: int


class ModelDecayMonitor:
    """Track model parameter degradation over time.

    Args:
        lookback: Number of recent observations to retain.
    """

    def __init__(self, lookback: int = 20) -> None:
        self._kalman_outcomes: deque[bool] = deque(maxlen=lookback)
        self._half_lives: deque[float] = deque(maxlen=lookback)

    def record_kalman_outcome(self, success: bool) -> None:
        """Record whether a Kalman filter update succeeded.

        Args:
            success: ``True`` if Kalman converged, ``False`` if OLS fallback.
        """
        self._kalman_outcomes.append(success)

    def record_half_life(self, half_life: float) -> None:
        """Record a newly estimated half-life.

        Args:
            half_life: OU half-life in days.
        """
        self._half_lives.append(half_life)

    def get_metrics(self) -> DecayMetrics:
        """Compute current decay metrics.

        Returns:
            ``DecayMetrics`` snapshot.  Defaults to zero values when
            insufficient data is available.
        """
        # Kalman fallback rate
        total = len(self._kalman_outcomes)
        if total > 0:
            fallbacks = sum(1 for ok in self._kalman_outcomes if not ok)
            fallback_rate = fallbacks / total
        else:
            fallback_rate = 0.0

        # Half-life statistics
        hl_list = list(self._half_lives)
        n = len(hl_list)
        if n > 0:
            sorted_hl = sorted(hl_list)
            median_hl = sorted_hl[n // 2]

            # Trend: second-half median minus first-half median
            if n >= 4:
                mid = n // 2
                first_half = sorted(hl_list[:mid])
                second_half = sorted(hl_list[mid:])
                trend = second_half[len(second_half) // 2] - first_half[len(first_half) // 2]
            else:
                trend = 0.0
        else:
            median_hl = 0.0
            trend = 0.0

        return DecayMetrics(
            kalman_fallback_rate=fallback_rate,
            median_half_life=median_hl,
            half_life_trend=trend,
            totals=total,
        )
