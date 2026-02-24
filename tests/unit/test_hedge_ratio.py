"""Tests for HedgeRatioEstimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.discovery.hedge_ratio import HedgeRatioEstimator


def test_kalman_recovers_beta() -> None:
    """Kalman filter should recover beta close to 1.2 for synthetic data."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range(start="2023-01-01", periods=n)

    x_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))

    true_beta = 1.2
    theta = 0.1
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i - 1] * (1 - theta) + np.random.normal(0, 1.0)
    y_prices = true_beta * x_prices + 5.0 + noise

    y = pd.Series(y_prices, index=dates, name="SYM_Y")
    x = pd.Series(x_prices, index=dates, name="SYM_X")

    estimator = HedgeRatioEstimator()
    result = estimator.estimate(y, x)

    assert abs(result.beta - true_beta) < 0.15, f"beta={result.beta}, expected ~{true_beta}"


def test_beta_series_length() -> None:
    """Beta series should have the same length as input."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range(start="2023-01-01", periods=n)

    x_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n)))
    y_prices = 1.5 * x_prices + np.random.normal(0, 2, n)

    y = pd.Series(y_prices, index=dates)
    x = pd.Series(x_prices, index=dates)

    estimator = HedgeRatioEstimator()
    result = estimator.estimate(y, x)

    assert len(result.beta_series) == n
