"""Main stat-arb trading engine orchestrator.

Ties together the walk-forward scheduler, spread computer, signal
generator, and execution layer into a single step-based loop.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

from stat_arb.config.constants import Signal, WindowPhase
from stat_arb.engine.signals import SignalEvent

if TYPE_CHECKING:
    from stat_arb.config.settings import SignalConfig, SizingConfig
    from stat_arb.data.universe import Universe
    from stat_arb.discovery.pair_discovery import PairDiscovery
    from stat_arb.engine.signals import SignalGenerator
    from stat_arb.engine.spread import SpreadComputer
    from stat_arb.engine.walk_forward import WalkForwardScheduler

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
    ) -> None:
        self._signal_config = signal_config
        self._sizing_config = sizing_config
        self._spread = spread_computer
        self._signals = signal_generator
        self._wf = walk_forward
        self._discovery = pair_discovery
        self._universe = universe
        self._formation_done: set[tuple[date, date]] = set()

    def step(
        self,
        current_date: date,
        quotes: dict[str, dict[str, float]],
    ) -> list[SignalEvent]:
        """Process one trading day.

        1. Determine the current walk-forward phase.
        2. If formation day and formation not yet run → run discovery.
        3. If trading day → compute Z-scores → generate signals.

        Args:
            current_date: Today's date.
            quotes: Real-time quotes keyed by symbol.  Each value must
                contain at least ``"bid"`` and ``"ask"`` (and optionally
                ``"last"``).  Mid prices are derived as ``(bid+ask)/2``.

        Returns:
            List of ``SignalEvent`` objects generated on this step.
        """
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

        # Build mid prices from quotes
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
