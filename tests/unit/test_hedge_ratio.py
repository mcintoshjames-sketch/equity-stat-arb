"""Tests for HedgeRatioEstimator."""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd

from stat_arb.config.settings import DiscoveryConfig
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

    estimator = HedgeRatioEstimator(DiscoveryConfig())
    result = estimator.estimate(y, x)

    assert result is not None
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

    estimator = HedgeRatioEstimator(DiscoveryConfig())
    result = estimator.estimate(y, x)

    assert result is not None
    assert len(result.beta_series) == n


def test_ols_fallback_disabled_returns_none() -> None:
    """When Kalman fails and fallback is disabled, estimate returns None."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range(start="2023-01-01", periods=n)
    y = pd.Series(np.random.normal(100, 1, n), index=dates)
    x = pd.Series(np.random.normal(50, 1, n), index=dates)

    config = DiscoveryConfig(use_ols_fallback=False)
    estimator = HedgeRatioEstimator(config)

    with patch.object(estimator, "_kalman_estimate", side_effect=RuntimeError("EM fail")):
        result = estimator.estimate(y, x, "AAA", "BBB")

    assert result is None


def test_ols_fallback_logs_warning_with_symbols(caplog: object) -> None:
    """When Kalman fails, WARNING log should include pair symbols."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range(start="2023-01-01", periods=n)
    y = pd.Series(np.random.normal(100, 1, n), index=dates)
    x = pd.Series(np.random.normal(50, 1, n), index=dates)

    config = DiscoveryConfig(use_ols_fallback=True)
    estimator = HedgeRatioEstimator(config)

    with (
        patch.object(estimator, "_kalman_estimate", side_effect=RuntimeError("EM fail")),
        caplog.at_level(logging.WARNING),  # type: ignore[union-attr]
    ):
        result = estimator.estimate(y, x, "XOM", "CVX")

    assert result is not None
    assert "XOM/CVX" in caplog.text
    assert "falling back to rolling OLS" in caplog.text
