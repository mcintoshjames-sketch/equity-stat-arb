"""Tests for config loading and validation."""

from pathlib import Path

import pytest

from stat_arb.config.constants import BrokerMode, Signal, OrderSide, WindowPhase, PairStatus
from stat_arb.config.settings import (
    AppConfig,
    DatabaseConfig,
    DiscoveryConfig,
    LoggingConfig,
    SchwabConfig,
    SignalConfig,
    UniverseConfig,
    load_config,
)


class TestConstants:
    def test_broker_modes(self):
        assert BrokerMode.PAPER.value == "paper"
        assert BrokerMode.LIVE.value == "live"
        assert BrokerMode.SIM.value == "sim"

    def test_signals(self):
        assert Signal.LONG_SPREAD.value == "long_spread"
        assert Signal.SHORT_SPREAD.value == "short_spread"
        assert Signal.FLAT.value == "flat"

    def test_order_side(self):
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"


class TestDiscoveryConfig:
    def test_defaults(self):
        cfg = DiscoveryConfig()
        assert cfg.min_half_life_days == 5
        assert cfg.max_half_life_days == 30
        assert cfg.min_correlation == 0.78
        assert cfg.parallel_n_jobs == -1

    def test_custom_half_life(self):
        cfg = DiscoveryConfig(min_half_life_days=3, max_half_life_days=45)
        assert cfg.min_half_life_days == 3
        assert cfg.max_half_life_days == 45


class TestSchwabConfig:
    def test_secret_masking(self):
        cfg = SchwabConfig(app_key="mykey", app_secret="mysecret")
        # SecretStr should mask the value in repr
        assert "mykey" not in repr(cfg.app_key)
        # But get_secret_value works
        assert cfg.app_key.get_secret_value() == "mykey"

    def test_tokens_db_expansion(self):
        cfg = SchwabConfig(app_key="k", app_secret="s", tokens_db="~/test.db")
        assert "~" not in cfg.tokens_db


class TestUniverseConfig:
    def test_sectors(self, universe_config):
        assert "technology" in universe_config.sectors
        assert "AAPL" in universe_config.sectors["technology"]


class TestLoadConfig:
    def test_load_default_yaml(self):
        yaml_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        if yaml_path.exists():
            cfg = load_config(yaml_path)
            assert isinstance(cfg, AppConfig)
            assert cfg.broker_mode == BrokerMode.PAPER
            assert cfg.discovery.max_half_life_days == 30
            assert cfg.discovery.min_correlation == 0.78

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")
