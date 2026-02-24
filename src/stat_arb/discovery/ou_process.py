"""Ornstein-Uhlenbeck half-life estimation for mean-reverting spreads.

Estimates the speed of mean reversion by regressing spread changes on
lagged spread levels: dS_t = theta * S_{t-1} + epsilon.
"""

from __future__ import annotations

import numpy as np


def estimate_ou_half_life(spread: np.ndarray) -> float:
    """Estimate the half-life of mean reversion for an OU process.

    Fits the discrete AR(1) model ``dS_t = theta * S_{t-1} + epsilon``
    and computes ``half_life = -ln(2) / theta``.

    Args:
        spread: 1-D array of spread values (y - beta*x - intercept).

    Returns:
        Half-life in days.  Returns ``inf`` if theta >= 0 (no mean reversion).
    """
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)

    # OLS: dS = theta * S_{t-1}
    # theta = sum(S_{t-1} * dS) / sum(S_{t-1}^2)
    denom = np.dot(spread_lag, spread_lag)
    if denom == 0:
        return float("inf")

    theta = np.dot(spread_lag, spread_diff) / denom

    # Non-negative theta means no mean-reversion — spread is unit root or explosive
    if theta >= 0:
        return float("inf")

    half_life = -np.log(2) / theta
    return float(half_life)
