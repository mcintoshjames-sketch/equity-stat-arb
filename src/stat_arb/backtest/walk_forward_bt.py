"""Walk-forward backtesting engine.

Replays historical data day-by-day through the stat-arb engine,
sizing, risk, and sim broker to produce a ``BacktestResult``.

At window transitions, the ``InventoryRebalancer`` reconciles physical
inventory against the new formation parameters.  Pairs that re-qualify
receive marginal delta orders (top-up / trim); pairs that drop are
force-exited.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from stat_arb.backtest.results import BacktestResult, TradeRecord
from stat_arb.config.constants import (
    ExitReason,
    PositionDirection,
    RebalanceAction,
    Signal,
)
from stat_arb.execution.order_builder import build_orders
from stat_arb.execution.rebalancer import (
    InventoryRebalancer,
    OpenPositionView,
    RebalanceResult,
)

if TYPE_CHECKING:
    from stat_arb.backtest.sim_broker import SimBroker
    from stat_arb.data.price_repo import PriceRepository
    from stat_arb.data.universe import Universe
    from stat_arb.discovery.pair_filter import QualifiedPair
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
        self._rebalancer = InventoryRebalancer(sizer._config)

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

        # Active positions: pair_key -> _OpenPosition with signed quantities
        active_positions: dict[str, _OpenPosition] = {}
        next_pair_id = 1

        # Track previous window's pairs for rebalancing
        prev_pairs: dict[str, QualifiedPair] = {}
        last_window_key: tuple[date, date] | None = None

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

            # --- Window transition: rebalance before signal processing ---
            window = self._engine._wf.current_window(current_date)
            if window is not None:
                window_key = (window.trading_start, window.trading_end)
                if (
                    window_key != last_window_key
                    and current_date == window.trading_start
                    and active_positions
                ):
                    self._handle_window_transition(
                        active_positions, prev_pairs, prices,
                        current_date, result,
                    )
                if window_key != last_window_key:
                    # Snapshot current pairs for next transition
                    prev_pairs = {
                        f"{p.symbol_y}/{p.symbol_x}": p
                        for p in self._engine._wf.active_pairs
                    }
                    last_window_key = window_key

            # Refresh earnings blackout cache
            if self._risk._earnings is not None:
                self._risk._earnings.refresh(self._universe.symbols, current_date)

            # Run engine step
            self._risk.reset_step_counters()
            events = self._engine.step(current_date, quotes)

            # Process rebalance orders from rolling scheduler
            for rb in self._engine.pending_rebalance:
                self._handle_rolling_rebalance(
                    rb, active_positions, prices, current_date, result,
                )

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
                        current_date=current_date,
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
                    sign = 1 if direction == PositionDirection.LONG else -1
                    active_positions[pair_key] = _OpenPosition(
                        direction=direction,
                        entry_date=current_date,
                        entry_z=event.z_score,
                        pair_id=pair_id,
                        signed_qty_y=sign * size.qty_y,
                        signed_qty_x=-sign * size.qty_x,
                    )
                    cohort_id = getattr(event.pair, "cohort_id", None)
                    self._risk.register_pair(
                        pair_id, event.pair.sector, cohort_id=cohort_id,
                    )
                    self._risk.record_entry()

                elif event.signal in (Signal.EXIT, Signal.STOP):
                    # Exit signal
                    pos = active_positions.pop(pair_key, None)
                    if pos is None:
                        continue

                    self._close_position(
                        pos, pair_key, event, current_date, result,
                    )

            # Record equity point
            portfolio_value = self._broker.get_portfolio_value()
            result.equity_curve.append(portfolio_value)
            self._risk.update_peak(portfolio_value)

        return result

    def _handle_window_transition(
        self,
        active_positions: dict[str, _OpenPosition],
        prev_pairs: dict[str, QualifiedPair],
        prices: dict[str, float],
        current_date: date,
        result: BacktestResult,
    ) -> None:
        """Reconcile inventory at a window boundary.

        Force-exits pairs that didn't re-qualify.  Applies marginal
        delta rebalancing for pairs that rolled over with new β.
        """
        new_pairs = {
            f"{p.symbol_y}/{p.symbol_x}": p
            for p in self._engine._wf.active_pairs
        }

        # Build position views for the rebalancer
        pos_views = {
            key: OpenPositionView(
                pair_key=key,
                direction=pos.direction,
                signed_qty_y=pos.signed_qty_y,
                signed_qty_x=pos.signed_qty_x,
                pair_id=pos.pair_id,
            )
            for key, pos in active_positions.items()
        }

        rebalance_results = self._rebalancer.reconcile(
            pos_views, prev_pairs, new_pairs, prices,
        )

        for rb in rebalance_results:
            if rb.action == RebalanceAction.FORCE_EXIT:
                pos = active_positions.pop(rb.pair_key, None)
                if pos is None:
                    continue
                fills = self._broker.submit_orders(rb.orders)
                pnl = sum(
                    (-f.price * f.quantity if f.side.value == "BUY"
                     else f.price * f.quantity)
                    for f in fills
                )
                trade = TradeRecord(
                    pair_key=rb.pair_key,
                    signal=(
                        Signal.LONG_SPREAD
                        if pos.direction == PositionDirection.LONG
                        else Signal.SHORT_SPREAD
                    ),
                    entry_date=pos.entry_date,
                    exit_date=current_date,
                    entry_z=pos.entry_z,
                    exit_z=0.0,
                    pnl=pnl,
                    exit_reason=ExitReason.STRUCTURAL_BREAK,
                )
                result.trades.append(trade)
                self._risk.unregister_pair(pos.pair_id)

            elif rb.action == RebalanceAction.ROLLOVER:
                self._broker.submit_orders(rb.orders)
                pos = active_positions[rb.pair_key]
                pos.signed_qty_y += rb.delta_qty_y
                pos.signed_qty_x += rb.delta_qty_x

    def _handle_rolling_rebalance(
        self,
        rb: RebalanceResult,
        active_positions: dict[str, _OpenPosition],
        prices: dict[str, float],
        current_date: date,
        result: BacktestResult,
    ) -> None:
        """Process a rebalance result from the rolling scheduler."""
        if rb.action == RebalanceAction.FORCE_EXIT:
            pos = active_positions.pop(rb.pair_key, None)
            if pos is None:
                return
            if rb.orders:
                fills = self._broker.submit_orders(rb.orders)
                pnl = sum(
                    (-f.price * f.quantity if f.side.value == "BUY"
                     else f.price * f.quantity)
                    for f in fills
                )
            else:
                pnl = 0.0
            trade = TradeRecord(
                pair_key=rb.pair_key,
                signal=(
                    Signal.LONG_SPREAD
                    if pos.direction == PositionDirection.LONG
                    else Signal.SHORT_SPREAD
                ),
                entry_date=pos.entry_date,
                exit_date=current_date,
                entry_z=pos.entry_z,
                exit_z=0.0,
                pnl=pnl,
                exit_reason=ExitReason.STRUCTURAL_BREAK,
            )
            result.trades.append(trade)
            self._risk.unregister_pair(pos.pair_id)

        elif rb.action == RebalanceAction.ROLLOVER:
            if rb.orders:
                self._broker.submit_orders(rb.orders)
            pos = active_positions.get(rb.pair_key)
            if pos is not None:
                pos.signed_qty_y += rb.delta_qty_y
                pos.signed_qty_x += rb.delta_qty_x

    def _close_position(
        self,
        pos: _OpenPosition,
        pair_key: str,
        event: object,
        current_date: date,
        result: BacktestResult,
    ) -> None:
        """Close a position and record the trade."""
        from stat_arb.execution.sizing import SizeResult

        size = SizeResult(
            qty_y=abs(pos.signed_qty_y),
            qty_x=abs(pos.signed_qty_x),
            notional_y=0.0,
            notional_x=0.0,
        )
        orders = build_orders(event, size, pos.pair_id)
        fills = self._broker.submit_orders(orders)

        pnl = sum(
            (-f.price * f.quantity if f.side.value == "BUY"
             else f.price * f.quantity)
            for f in fills
        )

        # Prefer explicit exit_reason from engine-level force-exits
        event_exit_reason = getattr(event, "exit_reason", None)
        if event_exit_reason is not None:
            exit_reason = event_exit_reason
        elif event.signal == Signal.EXIT:
            exit_reason = ExitReason.MEAN_REVERSION
        else:
            exit_reason = ExitReason.STOP_LOSS

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
    """Mutable tracking for an active backtest position.

    Tracks signed per-leg quantities so the rebalancer can compute
    deltas at window transitions.  Convention:

    - ``signed_qty_y > 0`` → long Y shares
    - ``signed_qty_x < 0`` → short X shares (LONG_SPREAD)
    """

    __slots__ = (
        "direction", "entry_date", "entry_z", "pair_id",
        "signed_qty_y", "signed_qty_x",
    )

    def __init__(
        self,
        direction: PositionDirection,
        entry_date: date,
        entry_z: float,
        pair_id: int,
        signed_qty_y: int,
        signed_qty_x: int,
    ) -> None:
        self.direction = direction
        self.entry_date = entry_date
        self.entry_z = entry_z
        self.pair_id = pair_id
        self.signed_qty_y = signed_qty_y
        self.signed_qty_x = signed_qty_x
