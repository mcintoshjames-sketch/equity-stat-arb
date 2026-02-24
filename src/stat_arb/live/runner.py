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

Default broker is **paper** — real trading requires explicit ``--broker-mode=live``.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from stat_arb.config.constants import (
    BrokerMode,
    PositionDirection,
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

logger = logging.getLogger(__name__)

# US market close in UTC (16:00 ET = 21:00 UTC, 20:00 UTC in summer)
_MARKET_CLOSE_UTC_HOUR = 21
_POST_CLOSE_BUFFER_MINUTES = 30


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

    def run_once(self) -> int:
        """Execute a single trading step for today.

        Returns:
            Number of signals generated (including FLAT).
        """
        today = date.today()
        logger.info("=== Live step: %s (mode=%s) ===", today, self._broker_mode.value)

        quotes = self._fetch_quotes()
        if not quotes:
            logger.warning("No quotes available — skipping step")
            return 0

        # Update paper broker quotes if applicable
        if hasattr(self._broker, "update_quotes"):
            self._broker.update_quotes(quotes)

        events = self._engine.step(today, quotes)

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
        return signals_processed

    def run_loop(self) -> None:
        """Run continuously, executing one step per trading day.

        Sleeps until after market close, runs a step, then sleeps again.
        Handles SIGINT/SIGTERM for graceful shutdown.
        """
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            "Starting live loop (mode=%s). Press Ctrl+C to stop.",
            self._broker_mode.value,
        )

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
                    time.sleep(min(wait_seconds, 60.0))
                    now = datetime.now(UTC)
                    wait_seconds = (next_run - now).total_seconds()

            if self._shutdown:
                break

            # Skip weekends
            if next_run.weekday() >= 5:
                logger.info("Weekend — skipping to Monday")
                continue

            try:
                self.run_once()
            except Exception:
                logger.exception("Error during live step — will retry next cycle")

        logger.info("Live loop stopped.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_entry(
        self, event, pair_key: str, mid_y: float, mid_x: float,
    ) -> None:
        """Size, risk-check, and execute an entry signal."""
        if pair_key in self._active_positions:
            return  # already in position

        size = self._sizer.size(mid_y, mid_x)
        pair_id = self._next_pair_id
        self._next_pair_id += 1

        decision = self._risk.check(
            event, size, self._broker, len(self._active_positions),
        )
        if decision.decision.value != "approved":
            logger.info("REJECTED %s: %s", pair_key, decision.reason)
            return

        orders = build_orders(event, size, pair_id)
        self._broker.submit_orders(orders)

        direction = (
            PositionDirection.LONG
            if event.signal == Signal.LONG_SPREAD
            else PositionDirection.SHORT
        )
        self._active_positions[pair_key] = _LivePosition(
            direction=direction,
            pair_id=pair_id,
            entry_date=date.today(),
        )
        self._risk.register_pair(pair_id, event.pair.sector)
        logger.info("ENTRY %s %s (pair_id=%d)", pair_key, direction.value, pair_id)

    def _handle_exit(self, event, pair_key: str) -> None:
        """Execute an exit/stop signal."""
        pos = self._active_positions.pop(pair_key, None)
        if pos is None:
            return

        from stat_arb.execution.sizing import SizeResult

        size = SizeResult(qty_y=0, qty_x=0, notional_y=0, notional_x=0)
        orders = build_orders(event, size, pos.pair_id)
        self._broker.submit_orders(orders)
        self._risk.unregister_pair(pos.pair_id)
        logger.info("EXIT %s %s (pair_id=%d)", pair_key, event.signal.value, pos.pair_id)

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
    """Minimal tracking for an active live position."""

    __slots__ = ("direction", "pair_id", "entry_date")

    def __init__(
        self,
        direction: PositionDirection,
        pair_id: int,
        entry_date: date,
    ) -> None:
        self.direction = direction
        self.pair_id = pair_id
        self.entry_date = entry_date
