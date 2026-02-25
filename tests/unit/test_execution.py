"""Tests for the execution layer: sizing, order builder, paper broker, schwab broker."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from stat_arb.config.constants import OrderSide, PositionDirection, Signal
from stat_arb.config.settings import SchwabBrokerConfig, SignalConfig, SizingConfig
from stat_arb.discovery.pair_filter import QualifiedPair
from stat_arb.engine.signals import SignalEvent
from stat_arb.execution.broker_base import Order
from stat_arb.execution.order_builder import build_orders
from stat_arb.execution.paper_broker import PaperBroker
from stat_arb.execution.sizing import PositionSizer, SizeResult


def _make_pair(**overrides: object) -> QualifiedPair:
    defaults = dict(
        symbol_y="AAA",
        symbol_x="BBB",
        sector="tech",
        formation_start=date(2023, 1, 2),
        formation_end=date(2023, 12, 29),
        hedge_ratio=1.2,
        intercept=5.0,
        spread_mean=0.0,
        spread_std=2.0,
        half_life=10.0,
        coint_pvalue=0.01,
        adf_pvalue=0.005,
        hurst=0.35,
    )
    defaults.update(overrides)
    return QualifiedPair(**defaults)  # type: ignore[arg-type]


def _make_event(signal: Signal, **pair_overrides: object) -> SignalEvent:
    pair = _make_pair(**pair_overrides)
    return SignalEvent(
        signal=signal,
        pair=pair,
        z_score=-2.5 if signal == Signal.LONG_SPREAD else 2.5,
        estimated_round_trip_cost=1.0,
    )


# ---------------------------------------------------------------------------
# PositionSizer
# ---------------------------------------------------------------------------


class TestPositionSizer:
    def test_floor_shares(self) -> None:
        """Shares should be floored (not rounded)."""
        cfg = SizingConfig(dollars_per_leg=1000.0, max_gross_per_pair=5000.0)
        sizer = PositionSizer(cfg)
        result = sizer.size(mid_y=33.0, mid_x=47.0)
        # floor(1000/33) = 30, floor(1000/47) = 21
        assert result.qty_y == 30
        assert result.qty_x == 21

    def test_max_gross_cap(self) -> None:
        """Gross notional should be capped at max_gross_per_pair."""
        cfg = SizingConfig(dollars_per_leg=2000.0, max_gross_per_pair=2500.0)
        sizer = PositionSizer(cfg)
        result = sizer.size(mid_y=10.0, mid_x=10.0)
        assert result.gross_notional <= 2500.0

    def test_zero_price_raises(self) -> None:
        """Zero or negative mid price should raise ValueError."""
        cfg = SizingConfig()
        sizer = PositionSizer(cfg)
        with pytest.raises(ValueError, match="positive"):
            sizer.size(mid_y=0.0, mid_x=50.0)
        with pytest.raises(ValueError, match="positive"):
            sizer.size(mid_y=50.0, mid_x=-1.0)


# ---------------------------------------------------------------------------
# build_orders
# ---------------------------------------------------------------------------


class TestBuildOrders:
    def test_long_spread(self) -> None:
        """LONG_SPREAD → BUY Y + SELL X."""
        event = _make_event(Signal.LONG_SPREAD)
        size = SizeResult(qty_y=10, qty_x=15, notional_y=500.0, notional_x=750.0)
        orders = build_orders(event, size, pair_id=1)
        assert len(orders) == 2
        assert orders[0].symbol == "AAA"
        assert orders[0].side == OrderSide.BUY
        assert orders[0].is_entry is True
        assert orders[1].symbol == "BBB"
        assert orders[1].side == OrderSide.SELL

    def test_short_spread(self) -> None:
        """SHORT_SPREAD → SELL Y + BUY X."""
        event = _make_event(Signal.SHORT_SPREAD)
        size = SizeResult(qty_y=10, qty_x=15, notional_y=500.0, notional_x=750.0)
        orders = build_orders(event, size, pair_id=2)
        assert len(orders) == 2
        assert orders[0].side == OrderSide.SELL
        assert orders[1].side == OrderSide.BUY

    def test_exit_long_spread(self) -> None:
        """EXIT with LONG direction → SELL Y + BUY X."""
        event = _make_event(Signal.EXIT)
        size = SizeResult(qty_y=10, qty_x=15, notional_y=500.0, notional_x=750.0)
        orders = build_orders(event, size, pair_id=3, direction=PositionDirection.LONG)
        assert len(orders) == 2
        assert all(not o.is_entry for o in orders)
        assert orders[0].symbol == "AAA"
        assert orders[0].side == OrderSide.SELL
        assert orders[1].symbol == "BBB"
        assert orders[1].side == OrderSide.BUY

    def test_exit_short_spread(self) -> None:
        """EXIT with SHORT direction → BUY Y + SELL X (reverses short entry)."""
        event = _make_event(Signal.EXIT)
        size = SizeResult(qty_y=10, qty_x=15, notional_y=500.0, notional_x=750.0)
        orders = build_orders(event, size, pair_id=3, direction=PositionDirection.SHORT)
        assert len(orders) == 2
        assert all(not o.is_entry for o in orders)
        # SHORT entry was SELL Y + BUY X → exit is BUY Y + SELL X
        assert orders[0].symbol == "AAA"
        assert orders[0].side == OrderSide.BUY
        assert orders[1].symbol == "BBB"
        assert orders[1].side == OrderSide.SELL

    def test_exit_default_direction(self) -> None:
        """EXIT without direction defaults to LONG (backward compat)."""
        event = _make_event(Signal.EXIT)
        size = SizeResult(qty_y=10, qty_x=15, notional_y=500.0, notional_x=750.0)
        orders = build_orders(event, size, pair_id=3)
        assert len(orders) == 2
        assert all(not o.is_entry for o in orders)
        assert orders[0].side == OrderSide.SELL
        assert orders[1].side == OrderSide.BUY

    def test_stop_short_spread(self) -> None:
        """STOP with SHORT direction → BUY Y + SELL X."""
        event = _make_event(Signal.STOP)
        size = SizeResult(qty_y=10, qty_x=15, notional_y=500.0, notional_x=750.0)
        orders = build_orders(event, size, pair_id=3, direction=PositionDirection.SHORT)
        assert len(orders) == 2
        assert orders[0].side == OrderSide.BUY
        assert orders[1].side == OrderSide.SELL

    def test_flat_returns_empty(self) -> None:
        """FLAT signal → no orders."""
        event = _make_event(Signal.FLAT)
        size = SizeResult(qty_y=10, qty_x=15, notional_y=500.0, notional_x=750.0)
        orders = build_orders(event, size, pair_id=4)
        assert orders == []


# ---------------------------------------------------------------------------
# PaperBroker
# ---------------------------------------------------------------------------


class TestPaperBroker:
    def _make_broker(self, slippage: float = 0.5) -> PaperBroker:
        cfg = SignalConfig(slippage_multiplier=slippage)
        return PaperBroker(cfg, initial_cash=10_000.0)

    def test_buy_fill_price_adverse(self) -> None:
        """BUY fills above mid (adverse)."""
        broker = self._make_broker(slippage=1.0)
        broker.update_quotes({"AAA": {"bid": 99.0, "ask": 101.0}})
        order = Order(
            symbol="AAA", side=OrderSide.BUY, quantity=10,
            pair_id=1, is_entry=True,
        )
        fills = broker.submit_orders([order])
        assert len(fills) == 1
        # mid=100, half_spread=1, slippage=1.0*1=1 → fill at 101
        assert fills[0].price == pytest.approx(101.0)

    def test_sell_fill_price_adverse(self) -> None:
        """SELL fills below mid (adverse)."""
        broker = self._make_broker(slippage=1.0)
        broker.update_quotes({"AAA": {"bid": 99.0, "ask": 101.0}})
        order = Order(
            symbol="AAA", side=OrderSide.SELL, quantity=10,
            pair_id=1, is_entry=True,
        )
        fills = broker.submit_orders([order])
        assert len(fills) == 1
        # mid=100, half_spread=1, slippage=1.0*1=1 → fill at 99
        assert fills[0].price == pytest.approx(99.0)

    def test_portfolio_value(self) -> None:
        """Portfolio value = cash + mark-to-market."""
        broker = self._make_broker(slippage=0.01)
        broker.update_quotes({"AAA": {"bid": 100.0, "ask": 100.0}})
        order = Order(
            symbol="AAA", side=OrderSide.BUY, quantity=10,
            pair_id=1, is_entry=True,
        )
        broker.submit_orders([order])
        # With zero spread, slippage has no effect: fill at mid=100
        # cash = 10000 - 10*100 = 9000, positions = 10*100 = 1000
        assert broker.get_portfolio_value() == pytest.approx(10_000.0)

    def test_gross_exposure(self) -> None:
        """Gross exposure = sum |pos × mid|."""
        broker = self._make_broker(slippage=0.01)
        broker.update_quotes({
            "AAA": {"bid": 100.0, "ask": 100.0},
            "BBB": {"bid": 50.0, "ask": 50.0},
        })
        broker.submit_orders([
            Order(symbol="AAA", side=OrderSide.BUY, quantity=10, pair_id=1, is_entry=True),
            Order(symbol="BBB", side=OrderSide.SELL, quantity=20, pair_id=1, is_entry=True),
        ])
        # |10*100| + |-20*50| = 1000 + 1000 = 2000
        assert broker.get_gross_exposure() == pytest.approx(2000.0)

    def test_missing_quote_skips_order(self) -> None:
        """Orders for symbols without quotes are skipped."""
        broker = self._make_broker()
        broker.update_quotes({})
        order = Order(
            symbol="ZZZ", side=OrderSide.BUY, quantity=5,
            pair_id=1, is_entry=True,
        )
        fills = broker.submit_orders([order])
        assert fills == []


# ---------------------------------------------------------------------------
# LiveSchwabBroker
# ---------------------------------------------------------------------------


class TestLiveSchwabBroker:
    def test_market_order_format(self) -> None:
        """Default order format is MARKET."""
        from stat_arb.execution.schwab_broker import LiveSchwabBroker

        client = MagicMock()
        client.place_order.return_value = "ORD-123"
        config = SchwabBrokerConfig(use_limit_orders=False)
        broker = LiveSchwabBroker(client, config)

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=10,
            pair_id=1, is_entry=True,
        )
        fills = broker.submit_orders([order])
        assert len(fills) == 1

        # Verify the order dict passed to place_order
        call_args = client.place_order.call_args[0][0]
        assert call_args["orderType"] == "MARKET"
        assert call_args["orderLegCollection"][0]["instruction"] == "BUY"
        assert call_args["orderLegCollection"][0]["quantity"] == 10

    def test_limit_order_format(self) -> None:
        """With use_limit_orders=True, order type is LIMIT."""
        from stat_arb.execution.schwab_broker import LiveSchwabBroker

        client = MagicMock()
        client.place_order.return_value = "ORD-456"
        config = SchwabBrokerConfig(use_limit_orders=True)
        broker = LiveSchwabBroker(client, config)

        order = Order(
            symbol="AAPL", side=OrderSide.SELL, quantity=5,
            pair_id=2, is_entry=False,
        )
        fills = broker.submit_orders([order])
        assert len(fills) == 1

        call_args = client.place_order.call_args[0][0]
        assert call_args["orderType"] == "LIMIT"
        assert call_args["orderLegCollection"][0]["instruction"] == "SELL"
