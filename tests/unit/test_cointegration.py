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
    assert result.adf_stat < 0  # ADF stat should be negative
    assert isinstance(result.swapped, bool)
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


def test_dual_direction_picks_stronger_adf(tester: CointegrationTester) -> None:
    """Dual EG should pick the direction with more negative ADF stat."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range(start="2023-01-01", periods=n)

    x_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))

    theta = 0.1
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i - 1] * (1 - theta) + np.random.normal(0, 1.0)
    y_prices = 1.2 * x_prices + 5.0 + noise

    y = pd.Series(y_prices, index=dates)
    x = pd.Series(x_prices, index=dates)

    # Run both directions manually
    result_fwd = tester._test_direction(y, x, swapped=False)
    result_rev = tester._test_direction(x, y, swapped=True)

    # At least one should pass
    assert result_fwd is not None or result_rev is not None

    # The combined test should pick the one with more negative ADF
    result = tester.test_pair(y, x)
    assert result is not None

    if result_fwd is not None and result_rev is not None:
        expected_swapped = result_rev.adf_stat < result_fwd.adf_stat
        assert result.swapped == expected_swapped
