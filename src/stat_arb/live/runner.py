"""Live trading runner for the stat-arb system.

Executes one trading step per invocation (``run_once``) or loops
continuously with a sleep-until-next-close scheduler (``run_loop``).

Re-uses the same ``StatArbEngine`` and ``RiskManager`` instances as the
backtest path — no code duplication.

The runner:
  1. Fetches real-time quotes from the Schwab API (or uses DB prices for paper).
  2. Calls ``engine.step()`` to get signals.
  3. Sizes, risk-checks, and executes each signal through the configured broker.
  4. Tracks open positions and handles exits.
  5. Writes ``EngineEvent`` rows to the DB for TUI consumption.

Default broker is **paper** — real trading requires explicit ``--broker-mode=live``.
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from stat_arb.config.constants import (
    BrokerMode,
    EngineCommandType,
    EngineEventType,
    EventSeverity,
    PositionDirection,
    RebalanceAction,
    Signal,
)
from stat_arb.execution.order_builder import build_orders

if TYPE_CHECKING:
    from stat_arb.config.settings import SignalConfig
    from stat_arb.data.price_repo import PriceRepository
    from stat_arb.data.schwab_client import SchwabDataClient
    from stat_arb.data.universe import Universe
    from stat_arb.engine.stat_arb_engine import StatArbEngine
    from stat_arb.execution.broker_base import ExecutionBroker
    from stat_arb.execution.sizing import PositionSizer
    from stat_arb.risk.risk_manager import RiskManager
    from stat_arb.risk.structural_break import StructuralBreakMonitor

logger = logging.getLogger(__name__)

# US market close in UTC (16:00 ET = 21:00 UTC, 20:00 UTC in summer)
_MARKET_CLOSE_UTC_HOUR = 21
_POST_CLOSE_BUFFER_MINUTES = 30

# Heartbeat interval in seconds
_HEARTBEAT_INTERVAL_S = 60


class LiveRunner:
    """Live/paper trading runner.

    Args:
        engine: The StatArbEngine (same instance used in backtest).
        sizer: Position sizer.
        risk_manager: Portfolio risk gate.
        price_repo: Price repository for historical data lookups.
        universe: Tradable symbol universe.
        schwab_client: Schwab API client (None for DB-only paper mode).
        broker_mode: PAPER or LIVE.
        signal_config: Signal config for PaperBroker construction.
    """

    def __init__(
        self,
        engine: StatArbEngine,
        sizer: PositionSizer,
        risk_manager: RiskManager,
        price_repo: PriceRepository,
        universe: Universe,
        schwab_client: SchwabDataClient | None,
        broker_mode: BrokerMode,
        signal_config: SignalConfig,
        structural_break: StructuralBreakMonitor | None = None,
    ) -> None:
        self._engine = engine
        self._sizer = sizer
        self._risk = risk_manager
        self._price_repo = price_repo
        self._universe = universe
        self._schwab = schwab_client
        self._broker_mode = broker_mode
        self._broker = self._create_broker(broker_mode, schwab_client, signal_config)
        self._active_positions: dict[str, _LivePosition] = {}
        self._next_pair_id = 1
        self._shutdown = False
        self._last_heartbeat: float = 0.0
        self._structural_break = structural_break

    def run_once(self) -> int:
        """Execute a single trading step for today.

        Returns:
            Number of signals generated (including FLAT).
        """
        today = date.today()
        logger.info("=== Live step: %s (mode=%s) ===", today, self._broker_mode.value)

        self._post_event(
            EngineEventType.STATE_CHANGED, EventSeverity.INFO, "running",
        )
        self._post_event(
            EngineEventType.STEP_STARTED, EventSeverity.INFO,
            f"Scan started for {today}",
        )

        try:
            quotes = self._fetch_quotes()
            if not quotes:
                logger.warning("No quotes available — skipping step")
                self._post_event(
                    EngineEventType.STATE_CHANGED, EventSeverity.INFO, "idle",
                )
                return 0

            # Update paper broker quotes if applicable
            if hasattr(self._broker, "update_quotes"):
                self._broker.update_quotes(quotes)

            self._risk.reset_step_counters()
            self._update_pair_metrics(quotes)
            self._check_pnl_stops(quotes)
            self._check_structural_breaks(today)

            # Refresh earnings blackout cache
            if self._risk._earnings is not None:
                self._risk._earnings.refresh(self._universe.symbols, today)

            events = self._engine.step(today, quotes)

            # Process rebalance orders from rolling scheduler
            for rb in self._engine.pending_rebalance:
                self._handle_rebalance(rb)

            signals_processed = 0
            for event in events:
                if event.signal == Signal.FLAT:
                    continue

                signals_processed += 1
                pair_key = f"{event.pair.symbol_y}/{event.pair.symbol_x}"

                mid_y = self._mid_price(quotes, event.pair.symbol_y)
                mid_x = self._mid_price(quotes, event.pair.symbol_x)
                if mid_y is None or mid_x is None:
                    continue

                if event.signal in (Signal.LONG_SPREAD, Signal.SHORT_SPREAD):
                    self._handle_entry(event, pair_key, mid_y, mid_x)
                elif event.signal in (Signal.EXIT, Signal.STOP):
                    self._handle_exit(event, pair_key)

            # Update risk peak
            portfolio_value = self._broker.get_portfolio_value()
            self._risk.update_peak(portfolio_value)

            logger.info(
                "Step complete: %d signals, %d active positions, portfolio=$%.2f",
                signals_processed,
                len(self._active_positions),
                portfolio_value,
            )

            self._post_event(
                EngineEventType.STEP_COMPLETED, EventSeverity.INFO,
                f"Scan complete: {signals_processed} signals, "
                f"{len(self._active_positions)} pairs, ${portfolio_value:,.0f}",
                json.dumps({
                    "signals_count": signals_processed,
                    "active_pairs": len(self._active_positions),
                    "portfolio_value": portfolio_value,
                }),
            )
        except Exception as exc:
            self._post_event(
                EngineEventType.ERROR, EventSeverity.ERROR, str(exc),
            )
            self._post_event(
                EngineEventType.STATE_CHANGED, EventSeverity.INFO, "idle",
            )
            raise

        self._post_event(
            EngineEventType.STATE_CHANGED, EventSeverity.INFO, "idle",
        )
        return signals_processed

    def run_loop(self) -> None:
        """Run continuously, executing one step per trading day.

        Sleeps until after market close, runs a step, then sleeps again.
        Handles SIGINT/SIGTERM for graceful shutdown.
        """
        # Only register signal handlers from the main thread
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            "Starting live loop (mode=%s). Press Ctrl+C to stop.",
            self._broker_mode.value,
        )

        self._prune_old_events()
        self._post_event(
            EngineEventType.ENGINE_STARTED, EventSeverity.INFO,
            f"Engine started (mode={self._broker_mode.value})",
        )
        self._post_heartbeat()

        while not self._shutdown:
            now = datetime.now(UTC)

            # Compute next run time: today's post-close if we haven't run yet,
            # otherwise tomorrow's post-close
            next_run = self._next_run_time(now)
            wait_seconds = (next_run - now).total_seconds()

            if wait_seconds > 0:
                logger.info(
                    "Next step scheduled for %s (%.1f hours)",
                    next_run.strftime("%Y-%m-%d %H:%M UTC"),
                    wait_seconds / 3600,
                )
                # Sleep in short intervals for responsive shutdown
                while wait_seconds > 0 and not self._shutdown:
                    time.sleep(min(wait_seconds, 10.0))
                    now = datetime.now(UTC)
                    wait_seconds = (next_run - now).total_seconds()
                    self._maybe_heartbeat()
                    self._poll_commands()

            if self._shutdown:
                break

            # Skip weekends
            if next_run.weekday() >= 5:
                logger.info("Weekend — skipping to Monday")
                continue

            self._poll_commands()
            if self._shutdown:
                break

            try:
                self.run_once()
            except Exception:
                logger.exception("Error during live step — will retry next cycle")

        self._post_event(
            EngineEventType.ENGINE_STOPPED, EventSeverity.INFO,
            "Engine stopped",
        )
        logger.info("Live loop stopped.")

    # ------------------------------------------------------------------
    # Event writing
    # ------------------------------------------------------------------

    def _post_event(
        self,
        event_type: EngineEventType,
        severity: EventSeverity,
        message: str,
        detail_json: str | None = None,
    ) -> None:
        """Insert an EngineEvent row. Never crashes the engine."""
        try:
            from stat_arb.data.db import get_session
            from stat_arb.data.schemas import EngineEvent

            session = get_session()
            try:
                session.add(EngineEvent(
                    event_type=event_type.value,
                    severity=severity.value,
                    message=message[:500],
                    detail_json=detail_json,
                ))
                session.commit()
            finally:
                session.close()
        except Exception:
            logger.debug("Failed to write engine event", exc_info=True)

    def _post_heartbeat(self) -> None:
        """Post a heartbeat event and update the timestamp."""
        self._post_event(
            EngineEventType.HEARTBEAT, EventSeverity.INFO, "heartbeat",
        )
        self._last_heartbeat = time.monotonic()

    def _maybe_heartbeat(self) -> None:
        """Post a heartbeat if enough time has elapsed."""
        if time.monotonic() - self._last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            self._post_heartbeat()

    # ------------------------------------------------------------------
    # Command polling
    # ------------------------------------------------------------------

    def _poll_commands(self) -> None:
        """Check for unacknowledged commands from the TUI."""
        try:
            from sqlalchemy import select

            from stat_arb.data.db import get_session
            from stat_arb.data.schemas import EngineCommand

            session = get_session()
            try:
                stmt = select(EngineCommand).where(
                    EngineCommand.acknowledged.is_(False),
                )
                commands = session.execute(stmt).scalars().all()

                for cmd in commands:
                    if cmd.command == EngineCommandType.KILL_SWITCH:
                        self._risk._kill_switch_active = True
                        self._shutdown = True
                        self._post_event(
                            EngineEventType.KILL_SWITCH, EventSeverity.CRITICAL,
                            "Kill switch activated via TUI",
                        )
                        logger.critical("KILL SWITCH activated via DB command")

                    cmd.acknowledged = True

                if commands:
                    session.commit()
            finally:
                session.close()
        except Exception:
            logger.debug("Failed to poll commands", exc_info=True)

    # ------------------------------------------------------------------
    # Event pruning
    # ------------------------------------------------------------------

    def _prune_old_events(self, days: int = 7) -> None:
        """Delete engine events older than *days* to prevent unbounded growth."""
        try:
            from sqlalchemy import delete

            from stat_arb.data.db import get_session
            from stat_arb.data.schemas import EngineEvent

            cutoff = datetime.now(UTC) - timedelta(days=days)
            session = get_session()
            try:
                session.execute(
                    delete(EngineEvent).where(EngineEvent.created_at < cutoff),
                )
                session.commit()
            finally:
                session.close()
        except Exception:
            logger.debug("Failed to prune old events", exc_info=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_entry(
        self, event, pair_key: str, mid_y: float, mid_x: float,
    ) -> None:
        """Size, risk-check, and execute an entry signal."""
        if pair_key in self._active_positions:
            return  # already in position

        z_score = getattr(event, "z_score", 0.0)
        self._post_event(
            EngineEventType.SIGNAL, EventSeverity.WARNING,
            f"Signal: {pair_key} {event.signal.value} (z={z_score:+.2f})",
            json.dumps({
                "pair_key": pair_key,
                "signal_type": event.signal.value,
                "z_score": z_score,
            }),
        )

        size = self._sizer.size(mid_y, mid_x)
        pair_id = self._next_pair_id
        self._next_pair_id += 1

        decision = self._risk.check(
            event, size, self._broker, len(self._active_positions),
            current_date=date.today(),
        )
        if decision.decision.value != "approved":
            logger.info("REJECTED %s: %s", pair_key, decision.reason)
            return

        orders = build_orders(event, size, pair_id)
        fills = self._broker.submit_orders(orders)

        # Extract per-leg fill prices (skip zero/negative from Schwab)
        fill_prices: dict[str, float] = {}
        for fill in fills:
            if fill.price > 0:
                fill_prices[fill.symbol] = fill.price

        direction = (
            PositionDirection.LONG
            if event.signal == Signal.LONG_SPREAD
            else PositionDirection.SHORT
        )
        self._active_positions[pair_key] = _LivePosition(
            direction=direction,
            pair_id=pair_id,
            entry_date=date.today(),
            symbol_y=event.pair.symbol_y,
            symbol_x=event.pair.symbol_x,
            qty_y=size.qty_y,
            qty_x=size.qty_x,
            entry_price_y=fill_prices.get(event.pair.symbol_y, mid_y),
            entry_price_x=fill_prices.get(event.pair.symbol_x, mid_x),
            sector=event.pair.sector,
        )
        cohort_id = getattr(event.pair, "cohort_id", None)
        self._risk.register_pair(
            pair_id, event.pair.sector, cohort_id=cohort_id,
            gross_notional=size.gross_notional,
        )
        self._risk.record_entry()
        logger.info("ENTRY %s %s (pair_id=%d)", pair_key, direction.value, pair_id)

        self._post_event(
            EngineEventType.ORDER, EventSeverity.INFO,
            f"Order: {pair_key} ENTRY {event.pair.symbol_y}/{event.pair.symbol_x}",
            json.dumps({
                "pair_key": pair_key,
                "side": "ENTRY",
                "symbol": f"{event.pair.symbol_y}/{event.pair.symbol_x}",
            }),
        )

    def _handle_exit(self, event, pair_key: str) -> None:
        """Execute an exit/stop signal."""
        pos = self._active_positions.pop(pair_key, None)
        if pos is None:
            return

        z_score = getattr(event, "z_score", 0.0)
        self._post_event(
            EngineEventType.SIGNAL, EventSeverity.WARNING,
            f"Signal: {pair_key} {event.signal.value} (z={z_score:+.2f})",
            json.dumps({
                "pair_key": pair_key,
                "signal_type": event.signal.value,
                "z_score": z_score,
            }),
        )

        from stat_arb.execution.sizing import SizeResult

        size = SizeResult(
            qty_y=pos.qty_y, qty_x=pos.qty_x, notional_y=0, notional_x=0,
        )
        orders = build_orders(event, size, pos.pair_id, direction=pos.direction)
        self._broker.submit_orders(orders)
        self._risk.unregister_pair(pos.pair_id)

        exit_reason = getattr(event, "exit_reason", None)
        reason_str = exit_reason.value if exit_reason else event.signal.value
        logger.info("EXIT %s %s (pair_id=%d)", pair_key, reason_str, pos.pair_id)

        self._post_event(
            EngineEventType.ORDER, EventSeverity.INFO,
            f"Order: {pair_key} EXIT {reason_str}",
        )

    def _handle_rebalance(self, rb) -> None:
        """Execute a rebalance result from the rolling scheduler."""
        if rb.action == RebalanceAction.FORCE_EXIT:
            pos = self._active_positions.pop(rb.pair_key, None)
            if pos is None:
                return
            if rb.orders:
                self._broker.submit_orders(rb.orders)
            self._risk.unregister_pair(pos.pair_id)
            logger.info("REBALANCE FORCE_EXIT %s (pair_id=%d)", rb.pair_key, pos.pair_id)
            self._post_event(
                EngineEventType.ORDER, EventSeverity.INFO,
                f"Rebalance: {rb.pair_key} FORCE_EXIT",
            )
        elif rb.action == RebalanceAction.ROLLOVER:
            if rb.orders:
                self._broker.submit_orders(rb.orders)
            logger.info(
                "REBALANCE ROLLOVER %s beta=%.3f->%.3f",
                rb.pair_key,
                rb.old_beta or 0.0,
                rb.new_beta or 0.0,
            )
            self._post_event(
                EngineEventType.ORDER, EventSeverity.INFO,
                f"Rebalance: {rb.pair_key} ROLLOVER",
            )

    def _update_pair_metrics(
        self, quotes: dict[str, dict[str, float]],
    ) -> None:
        """Update pair PnL and notional for all active positions.

        PnL uses adverse (liquidation) prices: sell at bid, cover at ask.
        Notional uses mid prices for sector concentration tracking.
        """
        for pos in self._active_positions.values():
            q_y = quotes.get(pos.symbol_y)
            q_x = quotes.get(pos.symbol_x)
            if q_y is None or q_x is None:
                continue

            bid_y = q_y.get("bid", 0.0)
            ask_y = q_y.get("ask", 0.0)
            bid_x = q_x.get("bid", 0.0)
            ask_x = q_x.get("ask", 0.0)

            if pos.direction == PositionDirection.LONG:
                # Long spread: long Y, short X
                # Exit: sell Y at bid, cover X at ask
                pnl = (
                    pos.qty_y * (bid_y - pos.entry_price_y)
                    + pos.qty_x * (pos.entry_price_x - ask_x)
                )
            else:
                # Short spread: short Y, long X
                # Exit: cover Y at ask, sell X at bid
                pnl = (
                    pos.qty_y * (pos.entry_price_y - ask_y)
                    + pos.qty_x * (bid_x - pos.entry_price_x)
                )

            self._risk.update_pair_pnl(pos.pair_id, pnl)

            # Notional uses mid prices for sector concentration
            mid_y = (bid_y + ask_y) / 2.0
            mid_x = (bid_x + ask_x) / 2.0
            notional = pos.qty_y * mid_y + pos.qty_x * mid_x
            self._risk.update_pair_notional(pos.pair_id, notional)

    def _check_pnl_stops(
        self, quotes: dict[str, dict[str, float]],
    ) -> None:
        """Force-exit positions that have breached their per-pair PnL stop."""
        from types import SimpleNamespace

        from stat_arb.engine.signals import SignalEvent
        from stat_arb.execution.sizing import SizeResult

        breached = [
            (key, pos)
            for key, pos in self._active_positions.items()
            if self._risk.check_pair_pnl_stop(pos.pair_id)
        ]
        for pair_key, pos in breached:
            self._active_positions.pop(pair_key, None)
            size = SizeResult(
                qty_y=pos.qty_y, qty_x=pos.qty_x,
                notional_y=0.0, notional_x=0.0,
            )
            # Build a synthetic STOP event for order construction
            pair_ns = SimpleNamespace(
                symbol_y=pos.symbol_y, symbol_x=pos.symbol_x,
            )
            event = SignalEvent(
                signal=Signal.STOP,
                pair=pair_ns,  # type: ignore[arg-type]
                z_score=0.0,
                estimated_round_trip_cost=0.0,
            )
            orders = build_orders(event, size, pos.pair_id, direction=pos.direction)
            self._broker.submit_orders(orders)
            self._risk.unregister_pair(pos.pair_id)
            logger.warning(
                "PNL STOP %s (pair_id=%d, pnl=%.2f)",
                pair_key, pos.pair_id,
                self._risk._pair_pnl.get(pos.pair_id, 0.0),
            )
            self._post_event(
                EngineEventType.ORDER, EventSeverity.WARNING,
                f"PnL stop: {pair_key} force-exited",
            )

    def _check_structural_breaks(self, today: date) -> None:
        """Force-exit positions whose spread has structurally broken."""
        if self._structural_break is None:
            return

        import numpy as np

        from stat_arb.execution.sizing import SizeResult

        # Need structural_break_window days of history
        window = self._structural_break._window
        from datetime import timedelta

        start = today - timedelta(days=int(window * 2))  # safety margin for bdays

        breached: list[tuple[str, _LivePosition]] = []
        for pair_key, pos in self._active_positions.items():
            df = self._price_repo.get_close_prices(
                [pos.symbol_y, pos.symbol_x], start, today,
            )
            if df.empty or pos.symbol_y not in df.columns or pos.symbol_x not in df.columns:
                continue
            prices_y = df[pos.symbol_y].dropna().values.astype(np.float64)
            prices_x = df[pos.symbol_x].dropna().values.astype(np.float64)

            # Need the pair's QualifiedPair — look it up from the engine's active pairs
            qp = None
            for p in self._engine._wf.active_pairs:
                if p.symbol_y == pos.symbol_y and p.symbol_x == pos.symbol_x:
                    qp = p
                    break
            if qp is None:
                # Also check rolling scheduler if available
                if hasattr(self._engine, "_rolling") and self._engine._rolling:
                    for p in self._engine._rolling.active_pairs:
                        if p.symbol_y == pos.symbol_y and p.symbol_x == pos.symbol_x:
                            qp = p
                            break
            if qp is None:
                continue

            if self._structural_break.check_pair(qp, prices_y, prices_x):
                breached.append((pair_key, pos))

        for pair_key, pos in breached:
            self._active_positions.pop(pair_key, None)
            size = SizeResult(
                qty_y=pos.qty_y, qty_x=pos.qty_x,
                notional_y=0.0, notional_x=0.0,
            )
            from types import SimpleNamespace

            from stat_arb.engine.signals import SignalEvent

            pair_ns = SimpleNamespace(
                symbol_y=pos.symbol_y, symbol_x=pos.symbol_x,
            )
            event = SignalEvent(
                signal=Signal.STOP,
                pair=pair_ns,  # type: ignore[arg-type]
                z_score=0.0,
                estimated_round_trip_cost=0.0,
            )
            orders = build_orders(event, size, pos.pair_id, direction=pos.direction)
            self._broker.submit_orders(orders)
            self._risk.unregister_pair(pos.pair_id)
            logger.warning(
                "STRUCTURAL BREAK %s (pair_id=%d) — force-exited",
                pair_key, pos.pair_id,
            )
            self._post_event(
                EngineEventType.ORDER, EventSeverity.WARNING,
                f"Structural break: {pair_key} force-exited",
            )

    def _fetch_quotes(self) -> dict[str, dict[str, float]]:
        """Fetch real-time quotes from Schwab, or synthetic quotes from DB."""
        if self._schwab is not None:
            try:
                raw = self._schwab.fetch_batch_quotes(self._universe.symbols)
                quotes: dict[str, dict[str, float]] = {}
                for sym, data in raw.items():
                    bid = data.get("bidPrice", data.get("lastPrice", 0))
                    ask = data.get("askPrice", data.get("lastPrice", 0))
                    last = data.get("lastPrice", 0)
                    if bid and ask:
                        quotes[sym] = {"bid": float(bid), "ask": float(ask), "last": float(last)}
                return quotes
            except Exception:
                logger.exception("Failed to fetch quotes from Schwab")

        # Fallback: use DB close prices as synthetic bid=ask quotes
        today = date.today()
        df = self._price_repo.get_close_prices(self._universe.symbols, today, today)
        if df.empty:
            return {}
        quotes = {}
        for sym in df.columns:
            val = df[sym].iloc[0] if len(df) > 0 else None
            if val is not None:
                price = float(val)
                quotes[sym] = {"bid": price, "ask": price}
        return quotes

    @staticmethod
    def _create_broker(
        broker_mode: BrokerMode,
        schwab_client: SchwabDataClient | None,
        signal_config: SignalConfig,
    ) -> ExecutionBroker:
        """Instantiate the appropriate broker based on mode."""
        if broker_mode == BrokerMode.LIVE:
            from stat_arb.config.settings import SchwabBrokerConfig
            from stat_arb.execution.schwab_broker import LiveSchwabBroker

            if schwab_client is None:
                raise RuntimeError(
                    "Cannot use broker_mode=live without a valid Schwab client. "
                    "Check your SCHWAB_APP_KEY / SCHWAB_APP_SECRET."
                )
            return LiveSchwabBroker(schwab_client, SchwabBrokerConfig())

        # Default: paper broker
        from stat_arb.execution.paper_broker import PaperBroker

        return PaperBroker(signal_config)

    @staticmethod
    def _mid_price(
        quotes: dict[str, dict[str, float]], symbol: str,
    ) -> float | None:
        """Extract mid price from quotes dict."""
        q = quotes.get(symbol)
        if q is None:
            return None
        bid = q.get("bid")
        ask = q.get("ask")
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return q.get("last")

    @staticmethod
    def _next_run_time(now: datetime) -> datetime:
        """Compute the next post-market-close run time."""
        today_close = now.replace(
            hour=_MARKET_CLOSE_UTC_HOUR,
            minute=_POST_CLOSE_BUFFER_MINUTES,
            second=0,
            microsecond=0,
        )
        if now >= today_close:
            # Already past today's close — schedule for tomorrow
            target = today_close + timedelta(days=1)
        else:
            target = today_close

        # Skip weekends
        while target.weekday() >= 5:
            target += timedelta(days=1)

        return target

    def _signal_handler(self, signum: int, frame: object) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info("Received signal %d — shutting down after current step", signum)
        self._shutdown = True


class _LivePosition:
    """Tracking for an active live position with fill data."""

    __slots__ = (
        "direction", "pair_id", "entry_date",
        "symbol_y", "symbol_x", "qty_y", "qty_x",
        "entry_price_y", "entry_price_x", "sector",
    )

    def __init__(
        self,
        direction: PositionDirection,
        pair_id: int,
        entry_date: date,
        symbol_y: str = "",
        symbol_x: str = "",
        qty_y: int = 0,
        qty_x: int = 0,
        entry_price_y: float = 0.0,
        entry_price_x: float = 0.0,
        sector: str = "",
    ) -> None:
        self.direction = direction
        self.pair_id = pair_id
        self.entry_date = entry_date
        self.symbol_y = symbol_y
        self.symbol_x = symbol_x
        self.qty_y = qty_y
        self.qty_x = qty_x
        self.entry_price_y = entry_price_y
        self.entry_price_x = entry_price_x
        self.sector = sector
