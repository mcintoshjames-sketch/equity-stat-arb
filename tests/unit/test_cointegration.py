"""Tests for CointegrationTester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.config.settings import DiscoveryConfig
from stat_arb.discovery.cointegration import CointegrationTester


@pytest.fixture
def tester() -> CointegrationTester:
    return CointegrationTester(DiscoveryConfig())


def test_cointegrated_pair_detected(tester: CointegrationTester) -> None:
    """Synthetic cointegrated series should produce a CointegrationResult."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range(start="2023-01-01", periods=n)

    # Random walk for X
    x_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))

    # Y cointegrated with X: Y = 1.2 * X + OU noise
    theta = 0.1
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i - 1] * (1 - theta) + np.random.normal(0, 1.0)
    y_prices = 1.2 * x_prices + 5.0 + noise

    y = pd.Series(y_prices, index=dates, name="SYM_Y")
    x = pd.Series(x_prices, index=dates, name="SYM_X")

    result = tester.test_pair(y, x)

    assert result is not None
    assert result.coint_pvalue < 0.05
    assert result.adf_pvalue < 0.05
    assert len(result.residuals) == n


def test_random_walks_not_cointegrated(tester: CointegrationTester) -> None:
    """Two independent random walks should return None."""
    np.random.seed(99)
    n = 300
    dates = pd.bdate_range(start="2023-01-01", periods=n)

    x_prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n)))
    y_prices = 50 * np.exp(np.cumsum(np.random.normal(-0.0005, 0.03, n)))

    y = pd.Series(y_prices, index=dates, name="A")
    x = pd.Series(x_prices, index=dates, name="B")

    result = tester.test_pair(y, x)
    assert result is None
