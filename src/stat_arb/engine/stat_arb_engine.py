"""Main stat-arb trading engine orchestrator.

Ties together the walk-forward scheduler, spread computer, signal
generator, and execution layer into a single step-based loop.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

from stat_arb.config.constants import (
    ExitReason,
    PositionDirection,
    RebalanceAction,
    Signal,
    WindowPhase,
)
from stat_arb.engine.signals import SignalEvent
from stat_arb.execution.rebalancer import RebalanceResult

if TYPE_CHECKING:
    from stat_arb.config.settings import SignalConfig, SizingConfig
    from stat_arb.data.universe import Universe
    from stat_arb.discovery.pair_discovery import PairDiscovery
    from stat_arb.discovery.pair_filter import QualifiedPair
    from stat_arb.engine.rolling_scheduler import RollingScheduler
    from stat_arb.engine.signals import SignalGenerator
    from stat_arb.engine.spread import SpreadComputer
    from stat_arb.engine.walk_forward import WalkForwardScheduler
    from stat_arb.execution.rebalancer import InventoryRebalancer
    from stat_arb.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class StatArbEngine:
    """Top-level orchestrator for the pairs trading system.

    Coordinates the walk-forward schedule, spread computation, signal
    generation, and (future) broker execution in a synchronous
    step-per-day loop.

    Args:
        signal_config: Z-score thresholds and slippage multiplier.
        sizing_config: Dollar sizing per leg.
        spread_computer: Computes live Z-scores and slippage.
        signal_generator: State-machine signal emitter.
        walk_forward: Window scheduler with discovery integration.
        pair_discovery: Pair discovery pipeline.
        universe: Tradable symbol universe.
        risk_manager: Optional risk manager for external pipeline use.
            Stored but not called by the engine — the caller pipeline
            invokes it between ``step()`` and execution.
    """

    def __init__(
        self,
        signal_config: SignalConfig,
        sizing_config: SizingConfig,
        spread_computer: SpreadComputer,
        signal_generator: SignalGenerator,
        walk_forward: WalkForwardScheduler,
        pair_discovery: PairDiscovery,
        universe: Universe,
        risk_manager: RiskManager | None = None,
        rolling_scheduler: RollingScheduler | None = None,
        rebalancer: InventoryRebalancer | None = None,
    ) -> None:
        self._signal_config = signal_config
        self._sizing_config = sizing_config
        self._spread = spread_computer
        self._signals = signal_generator
        self._wf = walk_forward
        self._discovery = pair_discovery
        self._universe = universe
        self._risk_manager = risk_manager
        self._rolling = rolling_scheduler
        self._rebalancer = rebalancer
        self._formation_done: set[tuple[date, date]] = set()
        self._pending_rebalance: list[RebalanceResult] = []

    @property
    def pending_rebalance(self) -> list[RebalanceResult]:
        """Rebalance results from the most recent rolling step."""
        return self._pending_rebalance

    def step(
        self,
        current_date: date,
        quotes: dict[str, dict[str, float]],
    ) -> list[SignalEvent]:
        """Process one trading day.

        Dispatches to ``_step_rolling`` or ``_step_walk_forward``
        depending on whether a rolling scheduler is configured.

        Args:
            current_date: Today's date.
            quotes: Real-time quotes keyed by symbol.

        Returns:
            List of ``SignalEvent`` objects generated on this step.
        """
        if self._rolling is not None:
            return self._step_rolling(current_date, quotes)
        return self._step_walk_forward(current_date, quotes)

    def _step_walk_forward(
        self,
        current_date: date,
        quotes: dict[str, dict[str, float]],
    ) -> list[SignalEvent]:
        """Original walk-forward step logic.

        1. Determine the current walk-forward phase.
        2. If formation day and formation not yet run -> run discovery.
        3. If trading day -> compute Z-scores -> generate signals.
        """
        self._pending_rebalance = []

        window = self._wf.current_window(current_date)
        if window is None:
            logger.debug("Date %s outside all walk-forward windows", current_date)
            return []

        phase = self._wf.current_phase(current_date)

        # --- Formation phase: run discovery once per window ---
        if phase == WindowPhase.FORMATION:
            key = (window.formation_start, window.formation_end)
            if key not in self._formation_done:
                # Only trigger on formation_end to use full window data
                if current_date == window.formation_end:
                    self._wf.run_formation(
                        window, self._discovery, self._universe,
                    )
                    self._formation_done.add(key)
            return []

        # --- Trading phase: generate signals ---
        if phase != WindowPhase.TRADING:
            return []

        active_pairs = self._wf.active_pairs
        if not active_pairs:
            return []

        # Earnings blackout force-exit
        earnings_exits: list[SignalEvent] = []
        if self._risk_manager is not None:
            mid_prices = self._mid_prices(quotes)
            for pair in active_pairs:
                if self._risk_manager.check_earnings_blackout(
                    pair.symbol_y, pair.symbol_x, current_date,
                ):
                    if pair.symbol_y in mid_prices and pair.symbol_x in mid_prices:
                        z_score = self._spread.compute_z_score(pair, mid_prices)
                        earnings_exits.append(SignalEvent(
                            signal=Signal.STOP,
                            pair=pair,
                            z_score=z_score,
                            estimated_round_trip_cost=0.0,
                            exit_reason=ExitReason.EARNINGS_BLACKOUT,
                        ))

        signal_events = self._generate_signals(active_pairs, current_date, quotes)
        return earnings_exits + signal_events

    def _step_rolling(
        self,
        current_date: date,
        quotes: dict[str, dict[str, float]],
    ) -> list[SignalEvent]:
        """Rolling scheduler step: discovery + expiry + rebalance + signals.

        1. Call rolling scheduler step (handles discovery + expiry).
        2. Rebalance if expired/refreshed pairs exist.
        3. Check per-pair PnL stops.
        4. Compute z-scores and generate signals for active pairs.
        """
        assert self._rolling is not None  # noqa: S101
        self._pending_rebalance = []

        self._rolling.step(current_date)

        # Rebalance on expiry/refresh
        if (self._rolling.expired_keys or self._rolling.refreshed_keys) and self._rebalancer:
            self._pending_rebalance = self._compute_rolling_rebalance(quotes)

        active_pairs = self._rolling.active_pairs
        if not active_pairs:
            return []

        mid_prices = self._mid_prices(quotes)

        events: list[SignalEvent] = []

        # PnL stop checks
        if self._risk_manager is not None:
            for pair in active_pairs:
                key = (pair.symbol_y, pair.symbol_x)
                pair_id = hash(key) & 0x7FFFFFFF
                if self._risk_manager.check_pair_pnl_stop(pair_id):
                    if pair.symbol_y in mid_prices and pair.symbol_x in mid_prices:
                        z_score = self._spread.compute_z_score(pair, mid_prices)
                        events.append(SignalEvent(
                            signal=Signal.STOP,
                            pair=pair,
                            z_score=z_score,
                            estimated_round_trip_cost=0.0,
                        ))

        # Earnings blackout force-exit
        if self._risk_manager is not None:
            for pair in active_pairs:
                if self._risk_manager.check_earnings_blackout(
                    pair.symbol_y, pair.symbol_x, current_date,
                ):
                    if pair.symbol_y in mid_prices and pair.symbol_x in mid_prices:
                        z_score = self._spread.compute_z_score(pair, mid_prices)
                        events.append(SignalEvent(
                            signal=Signal.STOP,
                            pair=pair,
                            z_score=z_score,
                            estimated_round_trip_cost=0.0,
                            exit_reason=ExitReason.EARNINGS_BLACKOUT,
                        ))

        # Generate normal signals
        signal_events = self._generate_signals(active_pairs, current_date, quotes)
        events.extend(signal_events)

        return events

    def _generate_signals(
        self,
        active_pairs: list,
        current_date: date,
        quotes: dict[str, dict[str, float]],
    ) -> list[SignalEvent]:
        """Compute z-scores and generate signals for active pairs."""
        mid_prices = self._mid_prices(quotes)

        events: list[SignalEvent] = []
        for pair in active_pairs:
            if pair.symbol_y not in mid_prices or pair.symbol_x not in mid_prices:
                logger.debug(
                    "Missing quote for %s/%s — skipping",
                    pair.symbol_y, pair.symbol_x,
                )
                continue

            z_score = self._spread.compute_z_score(pair, mid_prices)
            rt_cost = self._spread.estimate_round_trip_cost(pair, quotes)

            event = self._signals.generate_signal(
                pair, z_score, current_date, rt_cost,
            )
            events.append(event)

        # Log summary
        active_signals = [
            e for e in events if e.signal != Signal.FLAT
        ]
        if active_signals:
            logger.info(
                "Step %s: %d signals from %d active pairs",
                current_date, len(active_signals), len(active_pairs),
            )

        return events

    def _compute_rolling_rebalance(
        self,
        quotes: dict[str, dict[str, float]],
    ) -> list[RebalanceResult]:
        """Compute rebalance orders for expired/refreshed pairs."""
        from stat_arb.execution.rebalancer import OpenPositionView

        assert self._rolling is not None  # noqa: S101
        assert self._rebalancer is not None  # noqa: S101

        # Build old_pairs and new_pairs dicts for the rebalancer
        old_pairs: dict[str, QualifiedPair] = {}
        new_pairs: dict[str, QualifiedPair] = {}

        # For expired keys: old pair exists, no new pair (-> FORCE_EXIT)
        for key in self._rolling.expired_keys:
            prev = self._rolling.get_prev_pair(key)
            if prev is not None:
                pair_key = f"{key[0]}/{key[1]}"
                old_pairs[pair_key] = prev

        # For refreshed keys: both old and new exist (-> ROLLOVER)
        for key in self._rolling.refreshed_keys:
            prev = self._rolling.get_prev_pair(key)
            if prev is not None:
                pair_key = f"{key[0]}/{key[1]}"
                old_pairs[pair_key] = prev
                # New pair is in active_pairs
                for p in self._rolling.active_pairs:
                    if (p.symbol_y, p.symbol_x) == key:
                        new_pairs[pair_key] = p
                        break

        # Build dummy position views (actual positions tracked by runner/backtest)
        # The engine provides the rebalance results; the caller executes them
        active_positions: dict[str, OpenPositionView] = {}
        for pair_key in list(old_pairs.keys()):
            if pair_key not in new_pairs:
                # Force exit — build a minimal view
                active_positions[pair_key] = OpenPositionView(
                    pair_key=pair_key,
                    direction=PositionDirection.LONG,
                    signed_qty_y=0,
                    signed_qty_x=0,
                    pair_id=0,
                )

        results = []

        # Force exits for expired
        for key in self._rolling.expired_keys:
            pair_key = f"{key[0]}/{key[1]}"
            results.append(RebalanceResult(
                pair_key=pair_key,
                action=RebalanceAction.FORCE_EXIT,
                orders=[],
                old_beta=old_pairs[pair_key].hedge_ratio if pair_key in old_pairs else None,
                new_beta=None,
                delta_qty_y=0,
                delta_qty_x=0,
                shares_traded=0,
            ))

        # Rollovers for refreshed
        for key in self._rolling.refreshed_keys:
            pair_key = f"{key[0]}/{key[1]}"
            if pair_key in old_pairs and pair_key in new_pairs:
                results.append(RebalanceResult(
                    pair_key=pair_key,
                    action=RebalanceAction.ROLLOVER,
                    orders=[],
                    old_beta=old_pairs[pair_key].hedge_ratio,
                    new_beta=new_pairs[pair_key].hedge_ratio,
                    delta_qty_y=0,
                    delta_qty_x=0,
                    shares_traded=0,
                ))

        return results

    @staticmethod
    def _mid_prices(
        quotes: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Derive mid prices from bid/ask quotes."""
        mids: dict[str, float] = {}
        for sym, q in quotes.items():
            bid = q.get("bid")
            ask = q.get("ask")
            if bid is not None and ask is not None:
                mids[sym] = (bid + ask) / 2.0
            elif "last" in q:
                mids[sym] = q["last"]
        return mids
