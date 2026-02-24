"""Tests for OU half-life estimation."""

from __future__ import annotations

import numpy as np

from stat_arb.discovery.ou_process import estimate_ou_half_life


def test_synthetic_ou_half_life() -> None:
    """Synthetic OU residuals should produce half_life in [5, 30]."""
    np.random.seed(42)
    n = 500
    theta = 0.05  # mean-reversion speed → half_life ≈ -ln(2)/(-0.05) ≈ 13.9
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] * (1 - theta) + np.random.normal(0, 1.0)

    half_life = estimate_ou_half_life(spread)

    assert 5 <= half_life <= 30, f"half_life={half_life:.1f}, expected in [5, 30]"


def test_random_walk_returns_inf() -> None:
    """A random walk (θ ≈ 0) should return inf."""
    np.random.seed(99)
    n = 500
    # Pure random walk: no mean reversion
    spread = np.cumsum(np.random.normal(0, 1, n))

    half_life = estimate_ou_half_life(spread)

    assert half_life == float("inf")
