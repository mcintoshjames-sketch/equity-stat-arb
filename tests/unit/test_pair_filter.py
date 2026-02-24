"""Tests for PairFilter and QualifiedPair."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stat_arb.config.settings import DiscoveryConfig
from stat_arb.discovery.pair_filter import PairFilter, QualifiedPair


@pytest.fixture
def pair_filter() -> PairFilter:
    return PairFilter(DiscoveryConfig())


def test_qualified_pair_returned_for_cointegrated_data(pair_filter: PairFilter) -> None:
    """Synthetic cointegrated pair should pass all gates."""
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range(start="2022-01-01", periods=n)

    x_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))

    # Strong mean-reversion (theta=0.3) for clear Hurst < 0.5
    theta = 0.3
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i - 1] * (1 - theta) + np.random.normal(0, 1.0)
    y_prices = 1.2 * x_prices + 5.0 + noise

    y = pd.Series(y_prices, index=dates, name="SYM_Y")
    x = pd.Series(x_prices, index=dates, name="SYM_X")

    result = pair_filter.evaluate(
        symbol_y="SYM_Y",
        symbol_x="SYM_X",
        sector="technology",
        y_prices=y,
        x_prices=x,
        formation_start=date(2023, 1, 2),
        formation_end=date(2024, 3, 8),
    )

    assert result is not None
    assert isinstance(result, QualifiedPair)
    assert result.symbol_y == "SYM_Y"
    assert result.symbol_x == "SYM_X"
    assert result.sector == "technology"
    assert result.hedge_ratio != 0
    assert result.spread_std > 0
    assert result.half_life > 0
    assert result.coint_pvalue < 0.05
    assert result.hurst < 0.5


def test_non_cointegrated_pair_rejected(pair_filter: PairFilter) -> None:
    """Two independent random walks should return None."""
    np.random.seed(99)
    n = 300
    dates = pd.bdate_range(start="2023-01-01", periods=n)

    x_prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n)))
    y_prices = 50 * np.exp(np.cumsum(np.random.normal(-0.0005, 0.03, n)))

    y = pd.Series(y_prices, index=dates)
    x = pd.Series(x_prices, index=dates)

    result = pair_filter.evaluate(
        symbol_y="A",
        symbol_x="B",
        sector="financials",
        y_prices=y,
        x_prices=x,
        formation_start=date(2023, 1, 2),
        formation_end=date(2024, 3, 8),
    )

    assert result is None
