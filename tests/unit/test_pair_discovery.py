"""Tests for PairDiscovery orchestrator."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from stat_arb.config.settings import DiscoveryConfig
from stat_arb.data.universe import Universe
from stat_arb.discovery.pair_discovery import PairDiscovery


@pytest.fixture
def synthetic_universe() -> Universe:
    """Small universe with two correlated + cointegrated symbols."""
    return Universe(
        symbols=["SYM_Y", "SYM_X", "SYM_Z"],
        sector_map={"SYM_Y": "tech", "SYM_X": "tech", "SYM_Z": "tech"},
        sector_symbols={"tech": ["SYM_Y", "SYM_X", "SYM_Z"]},
    )


@pytest.fixture
def mock_price_repo() -> MagicMock:
    """Mock PriceRepository returning synthetic cointegrated prices."""
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

    # SYM_Z is independent (not cointegrated with Y or X)
    z_prices = 80 * np.exp(np.cumsum(np.random.normal(-0.0003, 0.025, n)))

    df = pd.DataFrame(
        {"SYM_Y": y_prices, "SYM_X": x_prices, "SYM_Z": z_prices},
        index=dates,
    )
    df.index.name = "date"

    repo = MagicMock()
    repo.get_close_prices.return_value = df
    return repo


def test_discovery_end_to_end(synthetic_universe: Universe, mock_price_repo: MagicMock) -> None:
    """Orchestrator should find at least one qualified pair from synthetic data."""
    config = DiscoveryConfig(parallel_n_jobs=1)
    discovery = PairDiscovery(config, mock_price_repo)

    results = discovery.discover(
        universe=synthetic_universe,
        formation_start=date(2023, 1, 2),
        formation_end=date(2024, 3, 8),
    )

    assert isinstance(results, list)
    # Should find SYM_Y/SYM_X as cointegrated
    assert len(results) >= 1

    pair = results[0]
    assert pair.symbol_y in ("SYM_Y", "SYM_X")
    assert pair.symbol_x in ("SYM_Y", "SYM_X")
    assert pair.coint_pvalue < 0.05


def test_min_common_obs_filters_short_series(synthetic_universe: Universe) -> None:
    """Setting min_common_obs higher than available data should reject all pairs."""
    np.random.seed(42)
    # Only 50 data points — below min_common_obs=100
    dates = pd.bdate_range(start="2023-01-01", periods=50)
    df = pd.DataFrame(
        {
            "SYM_Y": np.random.randn(50) + 100,
            "SYM_X": np.random.randn(50) + 100,
            "SYM_Z": np.random.randn(50) + 80,
        },
        index=dates,
    )
    df.index.name = "date"

    repo = MagicMock()
    repo.get_close_prices.return_value = df

    config = DiscoveryConfig(parallel_n_jobs=1, min_common_obs=100)
    discovery = PairDiscovery(config, repo)

    results = discovery.discover(
        universe=synthetic_universe,
        formation_start=date(2023, 1, 2),
        formation_end=date(2023, 6, 1),
    )
    assert results == []


def test_discovery_empty_prices(synthetic_universe: Universe) -> None:
    """Orchestrator should return empty list when no price data."""
    repo = MagicMock()
    repo.get_close_prices.return_value = pd.DataFrame()

    config = DiscoveryConfig(parallel_n_jobs=1)
    discovery = PairDiscovery(config, repo)

    results = discovery.discover(
        universe=synthetic_universe,
        formation_start=date(2023, 1, 2),
        formation_end=date(2024, 3, 8),
    )

    assert results == []
