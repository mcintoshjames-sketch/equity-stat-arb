"""Shared test fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from stat_arb.config.settings import (
    AppConfig,
    DatabaseConfig,
    DiscoveryConfig,
    RiskConfig,
    SchwabConfig,
    SignalConfig,
    SizingConfig,
    UniverseConfig,
)
from stat_arb.data.db import create_tables, get_session, init_db


@pytest.fixture
def db_config():
    return DatabaseConfig(url="sqlite:///:memory:", echo=False)


@pytest.fixture
def db_engine(db_config):
    engine = init_db(db_config)
    create_tables()
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    session = get_session()
    yield session
    session.close()


@pytest.fixture
def universe_config():
    return UniverseConfig(
        sectors={
            "technology": ["AAPL", "MSFT", "GOOG"],
            "financials": ["JPM", "BAC", "GS"],
        },
        min_price=10.0,
        min_avg_volume=500_000,
    )


@pytest.fixture
def discovery_config():
    return DiscoveryConfig()


@pytest.fixture
def signal_config():
    return SignalConfig()


@pytest.fixture
def sizing_config():
    return SizingConfig()


@pytest.fixture
def risk_config():
    return RiskConfig()


@pytest.fixture
def schwab_config():
    return SchwabConfig(
        app_key="test_key",
        app_secret="test_secret",
    )


@pytest.fixture
def app_config(schwab_config, universe_config):
    return AppConfig(
        schwab=schwab_config,
        universe=universe_config,
        database=DatabaseConfig(url="sqlite:///:memory:"),
    )


@pytest.fixture
def synthetic_cointegrated_prices() -> pd.DataFrame:
    """Generate synthetic cointegrated price series for testing.

    Creates two series: Y = beta * X + noise, where noise is mean-reverting.
    """
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range(start="2023-01-01", periods=n)

    # Random walk for X
    x_returns = np.random.normal(0.0005, 0.02, n)
    x_prices = 100 * np.exp(np.cumsum(x_returns))

    # Y cointegrated with X: Y = 1.2 * X + OU noise
    beta = 1.2
    intercept = 5.0
    theta = 0.1  # mean reversion speed
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i - 1] * (1 - theta) + np.random.normal(0, 1.0)
    y_prices = beta * x_prices + intercept + noise

    df = pd.DataFrame(
        {"SYM_Y": y_prices, "SYM_X": x_prices},
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def mock_schwab_client():
    """Mock SchwabDataClient for unit tests."""
    client = MagicMock()
    client.fetch_price_history.return_value = pd.DataFrame()
    client.fetch_batch_quotes.return_value = {}
    client.get_account_value.return_value = 50_000.0
    client.get_positions.return_value = []
    return client
