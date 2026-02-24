"""Tests for the backtest layer: sim broker, backtest results."""

from __future__ import annotations

from datetime import date

import pytest

from stat_arb.backtest.results import BacktestResult, TradeRecord
from stat_arb.backtest.sim_broker import SimBroker
from stat_arb.config.constants import ExitReason, OrderSide, Signal
from stat_arb.execution.broker_base import Order

# ---------------------------------------------------------------------------
# SimBroker
# ---------------------------------------------------------------------------


class TestSimBroker:
    def test_buy_fill_price_adverse(self) -> None:
        """BUY fills above close (adverse slippage)."""
        broker = SimBroker(slippage_bps=10.0, initial_cash=100_000.0)
        broker.set_date(date(2024, 1, 2), {"AAA": 100.0})
        order = Order(
            symbol="AAA", side=OrderSide.BUY, quantity=10,
            pair_id=1, is_entry=True,
        )
        fills = broker.submit_orders([order])
        assert len(fills) == 1
        # 100 * (1 + 10/10000) = 100.10
        assert fills[0].price == pytest.approx(100.10)

    def test_sell_fill_price_adverse(self) -> None:
        """SELL fills below close (adverse slippage)."""
        broker = SimBroker(slippage_bps=10.0, initial_cash=100_000.0)
        broker.set_date(date(2024, 1, 2), {"AAA": 100.0})
        order = Order(
            symbol="AAA", side=OrderSide.SELL, quantity=10,
            pair_id=1, is_entry=True,
        )
        fills = broker.submit_orders([order])
        assert len(fills) == 1
        # 100 * (1 - 10/10000) = 99.90
        assert fills[0].price == pytest.approx(99.90)

    def test_portfolio_value(self) -> None:
        """Portfolio = cash + mark-to-market."""
        broker = SimBroker(slippage_bps=0.0, initial_cash=50_000.0)
        broker.set_date(date(2024, 1, 2), {"AAA": 100.0})
        broker.submit_orders([
            Order(symbol="AAA", side=OrderSide.BUY, quantity=10, pair_id=1, is_entry=True),
        ])
        # cash = 50000 - 10*100 = 49000, positions = 10*100 = 1000
        assert broker.get_portfolio_value() == pytest.approx(50_000.0)

    def test_reset(self) -> None:
        """Reset restores initial state."""
        broker = SimBroker(slippage_bps=0.0, initial_cash=50_000.0)
        broker.set_date(date(2024, 1, 2), {"AAA": 100.0})
        broker.submit_orders([
            Order(symbol="AAA", side=OrderSide.BUY, quantity=10, pair_id=1, is_entry=True),
        ])
        broker.reset()
        assert broker.get_portfolio_value() == pytest.approx(50_000.0)
        assert broker.get_gross_exposure() == 0.0

    def test_missing_price_skips(self) -> None:
        """Orders for symbols without prices are skipped."""
        broker = SimBroker(slippage_bps=10.0)
        broker.set_date(date(2024, 1, 2), {})
        order = Order(
            symbol="ZZZ", side=OrderSide.BUY, quantity=5,
            pair_id=1, is_entry=True,
        )
        fills = broker.submit_orders([order])
        assert fills == []

    def test_gross_exposure(self) -> None:
        """Gross exposure = sum |pos × price|."""
        broker = SimBroker(slippage_bps=0.0, initial_cash=100_000.0)
        broker.set_date(date(2024, 1, 2), {"AAA": 100.0, "BBB": 50.0})
        broker.submit_orders([
            Order(symbol="AAA", side=OrderSide.BUY, quantity=10, pair_id=1, is_entry=True),
            Order(symbol="BBB", side=OrderSide.SELL, quantity=20, pair_id=1, is_entry=True),
        ])
        # |10*100| + |20*50| = 1000 + 1000 = 2000
        assert broker.get_gross_exposure() == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


class TestBacktestResult:
    def test_total_return(self) -> None:
        """Total return from equity curve."""
        result = BacktestResult(equity_curve=[100.0, 110.0, 120.0])
        assert result.total_return == pytest.approx(0.20)

    def test_total_return_empty(self) -> None:
        """Empty equity curve returns 0."""
        result = BacktestResult(equity_curve=[])
        assert result.total_return == 0.0

    def test_sharpe_positive(self) -> None:
        """Sharpe should be positive for steadily increasing equity."""
        curve = [100.0 + i * 0.5 for i in range(100)]
        result = BacktestResult(equity_curve=curve)
        assert result.sharpe > 0

    def test_max_drawdown_known_curve(self) -> None:
        """Known drawdown from a peak-valley-recovery curve."""
        result = BacktestResult(equity_curve=[100.0, 120.0, 90.0, 110.0])
        # Peak=120, trough=90, dd=30/120=0.25
        assert result.max_drawdown == pytest.approx(0.25)

    def test_win_rate(self) -> None:
        """Win rate from trade records."""
        trades = [
            TradeRecord(
                pair_key="A/B", signal=Signal.LONG_SPREAD,
                entry_date=date(2024, 1, 1), exit_date=date(2024, 1, 10),
                entry_z=-2.5, exit_z=0.5, pnl=pnl,
                exit_reason=ExitReason.MEAN_REVERSION,
            )
            for pnl in [100.0, -50.0, 200.0, -30.0]
        ]
        result = BacktestResult(trades=trades)
        assert result.win_rate == pytest.approx(0.5)

    def test_trade_record_immutability(self) -> None:
        """TradeRecord is frozen."""
        trade = TradeRecord(
            pair_key="A/B", signal=Signal.LONG_SPREAD,
            entry_date=date(2024, 1, 1), exit_date=date(2024, 1, 10),
            entry_z=-2.5, exit_z=0.5, pnl=100.0,
            exit_reason=ExitReason.MEAN_REVERSION,
        )
        with pytest.raises(AttributeError):
            trade.pnl = 999.0  # type: ignore[misc]
