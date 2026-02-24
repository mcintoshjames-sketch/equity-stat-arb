"""Walk-forward backtesting engine.

Replays historical data day-by-day through the stat-arb engine,
sizing, risk, and sim broker to produce a ``BacktestResult``.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from stat_arb.backtest.results import BacktestResult, TradeRecord
from stat_arb.config.constants import ExitReason, PositionDirection, Signal
from stat_arb.execution.order_builder import build_orders

if TYPE_CHECKING:
    from stat_arb.backtest.sim_broker import SimBroker
    from stat_arb.data.price_repo import PriceRepository
    from stat_arb.data.universe import Universe
    from stat_arb.engine.stat_arb_engine import StatArbEngine
    from stat_arb.execution.sizing import PositionSizer
    from stat_arb.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """Walk-forward backtesting orchestrator.

    Day loop: get prices -> set sim_broker date -> build synthetic quotes
    -> engine.step() -> size -> risk check -> execute -> record equity.

    Args:
        engine: StatArbEngine with walk-forward scheduler and signal gen.
        price_repo: Historical price repository.
        risk_manager: Portfolio risk gate.
        sizer: Position sizer.
        sim_broker: Simulation broker for fills.
        universe: Tradable symbol universe.
    """

    def __init__(
        self,
        engine: StatArbEngine,
        price_repo: PriceRepository,
        risk_manager: RiskManager,
        sizer: PositionSizer,
        sim_broker: SimBroker,
        universe: Universe,
    ) -> None:
        self._engine = engine
        self._price_repo = price_repo
        self._risk = risk_manager
        self._sizer = sizer
        self._broker = sim_broker
        self._universe = universe

    def run(self, start_date: date, end_date: date) -> BacktestResult:
        """Execute the walk-forward backtest.

        Args:
            start_date: First date in the simulation.
            end_date: Last date in the simulation.

        Returns:
            ``BacktestResult`` with equity curve, trades, and metrics.
        """
        self._broker.reset()

        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
        )

        # Active positions: pair_key -> (direction, entry_date, entry_z, pair_id)
        active_positions: dict[str, _OpenPosition] = {}
        next_pair_id = 1

        trading_days = pd.bdate_range(start=start_date, end=end_date)

        for ts in trading_days:
            current_date = ts.date()

            # Get prices for all universe symbols
            prices = self._get_prices(current_date)
            if not prices:
                continue

            # Set broker state
            self._broker.set_date(current_date, prices)

            # Build synthetic quotes (bid=ask=close for backtest)
            quotes = {
                sym: {"bid": p, "ask": p}
                for sym, p in prices.items()
            }

            # Run engine step
            events = self._engine.step(current_date, quotes)

            for event in events:
                if event.signal == Signal.FLAT:
                    continue

                pair_key = f"{event.pair.symbol_y}/{event.pair.symbol_x}"
                mid_y = prices.get(event.pair.symbol_y)
                mid_x = prices.get(event.pair.symbol_x)
                if mid_y is None or mid_x is None:
                    continue

                if event.signal in (Signal.LONG_SPREAD, Signal.SHORT_SPREAD):
                    # Entry signal
                    if pair_key in active_positions:
                        continue  # already in position

                    size = self._sizer.size(mid_y, mid_x)
                    pair_id = next_pair_id
                    next_pair_id += 1

                    decision = self._risk.check(
                        event, size, self._broker,
                        len(active_positions),
                    )
                    if decision.decision.value != "approved":
                        continue

                    orders = build_orders(event, size, pair_id)
                    self._broker.submit_orders(orders)

                    direction = (
                        PositionDirection.LONG
                        if event.signal == Signal.LONG_SPREAD
                        else PositionDirection.SHORT
                    )
                    active_positions[pair_key] = _OpenPosition(
                        direction=direction,
                        entry_date=current_date,
                        entry_z=event.z_score,
                        pair_id=pair_id,
                        size=size,
                    )
                    self._risk.register_pair(pair_id, event.pair.sector)

                elif event.signal in (Signal.EXIT, Signal.STOP):
                    # Exit signal
                    pos = active_positions.pop(pair_key, None)
                    if pos is None:
                        continue

                    size = pos.size
                    orders = build_orders(event, size, pos.pair_id)
                    fills = self._broker.submit_orders(orders)

                    # Compute P&L from fills
                    pnl = sum(
                        (-f.price * f.quantity if f.side.value == "BUY"
                         else f.price * f.quantity)
                        for f in fills
                    )

                    exit_reason = (
                        ExitReason.MEAN_REVERSION
                        if event.signal == Signal.EXIT
                        else ExitReason.STOP_LOSS
                    )

                    trade = TradeRecord(
                        pair_key=pair_key,
                        signal=(
                            Signal.LONG_SPREAD
                            if pos.direction == PositionDirection.LONG
                            else Signal.SHORT_SPREAD
                        ),
                        entry_date=pos.entry_date,
                        exit_date=current_date,
                        entry_z=pos.entry_z,
                        exit_z=event.z_score,
                        pnl=pnl,
                        exit_reason=exit_reason,
                    )
                    result.trades.append(trade)
                    self._risk.unregister_pair(pos.pair_id)

            # Record equity point
            portfolio_value = self._broker.get_portfolio_value()
            result.equity_curve.append(portfolio_value)
            self._risk.update_peak(portfolio_value)

        return result

    def _get_prices(self, current_date: date) -> dict[str, float]:
        """Fetch close prices for the universe on a given date."""
        df = self._price_repo.get_close_prices(
            self._universe.symbols, current_date, current_date,
        )
        if df.empty:
            return {}
        prices: dict[str, float] = {}
        for sym in df.columns:
            val = df[sym].iloc[0] if len(df) > 0 else None
            if val is not None and not pd.isna(val):
                prices[sym] = float(val)
        return prices


class _OpenPosition:
    """Mutable tracking for an active backtest position."""

    __slots__ = ("direction", "entry_date", "entry_z", "pair_id", "size")

    def __init__(
        self,
        direction: PositionDirection,
        entry_date: date,
        entry_z: float,
        pair_id: int,
        size: object,
    ) -> None:
        self.direction = direction
        self.entry_date = entry_date
        self.entry_z = entry_z
        self.pair_id = pair_id
        self.size = size
