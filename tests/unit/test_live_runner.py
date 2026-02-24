"""Tests for the LiveRunner."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from stat_arb.config.constants import BrokerMode
from stat_arb.config.settings import RiskConfig, SignalConfig, SizingConfig
from stat_arb.execution.sizing import PositionSizer
from stat_arb.live.runner import LiveRunner
from stat_arb.risk.risk_manager import RiskManager


def _mock_engine(events: list | None = None):
    """Create a mock StatArbEngine."""
    engine = MagicMock()
    engine.step.return_value = events or []
    return engine


def _mock_universe():
    """Create a mock Universe."""
    u = MagicMock()
    u.symbols = ["AAA", "BBB"]
    return u


def _mock_price_repo():
    """Create a mock PriceRepository."""
    repo = MagicMock()
    repo.get_close_prices.return_value = MagicMock(empty=True)
    return repo


def _make_runner(
    engine=None, schwab_client=None, broker_mode=BrokerMode.PAPER, events=None,
) -> LiveRunner:
    """Build a LiveRunner with mocked dependencies."""
    engine = engine or _mock_engine(events)
    return LiveRunner(
        engine=engine,
        sizer=PositionSizer(SizingConfig()),
        risk_manager=RiskManager(RiskConfig()),
        price_repo=_mock_price_repo(),
        universe=_mock_universe(),
        schwab_client=schwab_client,
        broker_mode=broker_mode,
        signal_config=SignalConfig(),
    )


class TestLiveRunner:
    def test_run_once_no_quotes(self) -> None:
        """run_once returns 0 when no quotes available."""
        runner = _make_runner()
        count = runner.run_once()
        assert count == 0

    def test_run_once_with_schwab_quotes(self) -> None:
        """run_once fetches quotes and calls engine.step."""
        schwab = MagicMock()
        schwab.fetch_batch_quotes.return_value = {
            "AAA": {"bidPrice": 100.0, "askPrice": 100.5, "lastPrice": 100.25},
            "BBB": {"bidPrice": 50.0, "askPrice": 50.2, "lastPrice": 50.1},
        }
        engine = _mock_engine([])
        runner = _make_runner(engine=engine, schwab_client=schwab)
        count = runner.run_once()

        assert count == 0
        engine.step.assert_called_once()

    def test_paper_broker_created_by_default(self) -> None:
        """Default broker_mode=paper creates a PaperBroker."""
        from stat_arb.execution.paper_broker import PaperBroker

        runner = _make_runner(broker_mode=BrokerMode.PAPER)
        assert isinstance(runner._broker, PaperBroker)

    def test_live_broker_requires_schwab_client(self) -> None:
        """broker_mode=live without schwab_client raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Cannot use broker_mode=live"):
            _make_runner(broker_mode=BrokerMode.LIVE, schwab_client=None)

    def test_shutdown_flag_stops_loop(self) -> None:
        """Setting _shutdown = True breaks the loop."""
        runner = _make_runner()
        runner._shutdown = True
        # run_loop should return immediately
        runner.run_loop()

    def test_next_run_time_future(self) -> None:
        """_next_run_time returns a future datetime."""
        # Monday morning UTC
        now = datetime(2025, 6, 2, 10, 0, 0, tzinfo=UTC)
        nrt = LiveRunner._next_run_time(now)
        assert nrt > now
        # Should be same day (Monday) after market close
        assert nrt.weekday() < 5

    def test_next_run_time_skips_weekend(self) -> None:
        """_next_run_time skips Saturday/Sunday."""
        # Saturday evening UTC
        now = datetime(2025, 6, 7, 22, 0, 0, tzinfo=UTC)
        nrt = LiveRunner._next_run_time(now)
        assert nrt.weekday() == 0  # Monday

    def test_mid_price_extraction(self) -> None:
        """_mid_price computes (bid + ask) / 2."""
        quotes = {"AAA": {"bid": 100.0, "ask": 102.0}}
        assert LiveRunner._mid_price(quotes, "AAA") == pytest.approx(101.0)
        assert LiveRunner._mid_price(quotes, "ZZZ") is None
