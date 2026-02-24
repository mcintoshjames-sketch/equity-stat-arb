"""Walk-forward window scheduler.

Generates non-overlapping formation → trading windows and coordinates
pair discovery for each formation period.  Enforces an execution buffer
so that signals from formation-end close prices are only eligible for
execution at the next trading-day open (no lookahead).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from stat_arb.config.constants import WindowPhase

if TYPE_CHECKING:
    from stat_arb.config.settings import WalkForwardConfig
    from stat_arb.data.universe import Universe
    from stat_arb.discovery.pair_discovery import PairDiscovery
    from stat_arb.discovery.pair_filter import QualifiedPair

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Window:
    """A single formation + trading window pair.

    Attributes:
        formation_start: First business day of the formation window.
        formation_end: Last business day of the formation window.
        trading_start: First business day of the trading window
            (one business day after formation_end — execution buffer).
        trading_end: Last business day of the trading window.
    """

    formation_start: date
    formation_end: date
    trading_start: date
    trading_end: date


class WalkForwardScheduler:
    """Generate and manage walk-forward formation/trading windows.

    Args:
        config: Walk-forward window durations.
    """

    def __init__(self, config: WalkForwardConfig) -> None:
        self._config = config
        self._windows: list[Window] = []
        self._active_pairs: list[QualifiedPair] = []
        self._current_window_idx: int = -1

    @property
    def active_pairs(self) -> list[QualifiedPair]:
        """Currently active qualified pairs for the trading window."""
        return list(self._active_pairs)

    @property
    def windows(self) -> list[Window]:
        """All generated windows."""
        return list(self._windows)

    def generate_windows(
        self,
        start_date: date,
        end_date: date,
    ) -> list[Window]:
        """Generate non-overlapping formation + trading windows.

        Each window consists of ``formation_days`` business days for
        estimation followed by ``trading_days`` business days for
        out-of-sample trading.  A one-business-day execution buffer
        separates formation end from trading start.

        Args:
            start_date: Earliest date to begin scheduling.
            end_date: Latest date (windows ending after this are excluded).

        Returns:
            List of ``Window`` objects in chronological order.
        """
        windows: list[Window] = []
        cursor = start_date
        f_days = self._config.formation_days
        t_days = self._config.trading_days

        while True:
            # Formation window: cursor + formation_days bdays
            bdays = pd.bdate_range(start=cursor, periods=f_days)
            if len(bdays) < f_days:
                break

            formation_start = bdays[0].date()
            formation_end = bdays[-1].date()

            # Trading window: next bday after formation_end + trading_days
            trading_bdays = pd.bdate_range(
                start=formation_end + pd.Timedelta(days=1),
                periods=t_days,
            )
            if len(trading_bdays) < t_days:
                break

            trading_start = trading_bdays[0].date()
            trading_end = trading_bdays[-1].date()

            if trading_end > end_date:
                break

            windows.append(Window(
                formation_start=formation_start,
                formation_end=formation_end,
                trading_start=trading_start,
                trading_end=trading_end,
            ))

            # Next window starts after trading ends
            cursor = trading_end + pd.Timedelta(days=1)

        self._windows = windows
        logger.info(
            "Generated %d walk-forward windows from %s to %s",
            len(windows), start_date, end_date,
        )
        return windows

    def current_phase(self, as_of_date: date) -> WindowPhase | None:
        """Determine which phase *as_of_date* falls in.

        Args:
            as_of_date: The date to classify.

        Returns:
            ``WindowPhase.FORMATION`` or ``WindowPhase.TRADING``,
            or ``None`` if the date is outside all windows.
        """
        for w in self._windows:
            if w.formation_start <= as_of_date <= w.formation_end:
                return WindowPhase.FORMATION
            if w.trading_start <= as_of_date <= w.trading_end:
                return WindowPhase.TRADING
        return None

    def current_window(self, as_of_date: date) -> Window | None:
        """Return the window that contains *as_of_date*, if any."""
        for w in self._windows:
            if w.formation_start <= as_of_date <= w.trading_end:
                return w
        return None

    def run_formation(
        self,
        window: Window,
        pair_discovery: PairDiscovery,
        universe: Universe,
    ) -> list[QualifiedPair]:
        """Execute pair discovery for a formation window.

        Calls ``PairDiscovery.discover()`` with the window's formation
        dates and stores the resulting pairs as active for the upcoming
        trading window.

        Args:
            window: The window whose formation period to process.
            pair_discovery: Discovery pipeline instance.
            universe: Tradable symbol universe.

        Returns:
            List of newly qualified pairs.
        """
        logger.info(
            "Running formation: %s → %s",
            window.formation_start, window.formation_end,
        )
        pairs = pair_discovery.discover(
            universe, window.formation_start, window.formation_end,
        )
        self._active_pairs = pairs
        logger.info(
            "Formation complete: %d pairs active for trading %s → %s",
            len(pairs), window.trading_start, window.trading_end,
        )
        return pairs
