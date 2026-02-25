"""Earnings blackout checker for pairs trading.

Blocks new entries and force-exits existing positions when either leg
of a pair has earnings within N business days.  Uses the FMP client's
two-layer cache to avoid rate-limit issues during backtests.

BMO safety: ``blackout_days=3`` means we exit at close of D-3, ensuring
the position is flat before any D-open (before-market-open) earnings print.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from stat_arb.data.fmp_client import FmpClient

logger = logging.getLogger(__name__)


class EarningsBlackout:
    """Earnings proximity checker for both legs of a pair.

    Args:
        fmp_client: FMP API client with cached earnings data.
        blackout_days: Business days before earnings to block/exit.
    """

    def __init__(self, fmp_client: FmpClient, blackout_days: int = 3) -> None:
        self._fmp = fmp_client
        self._blackout_days = blackout_days
        self._earnings_cache: dict[str, date | None] = {}

    def refresh(self, symbols: list[str], as_of: date) -> None:
        """Refresh earnings dates from FMP.  Call once per step."""
        self._earnings_cache = self._fmp.get_next_earnings(symbols, as_of)

    def is_blacked_out(self, symbol: str, as_of: date) -> bool:
        """True if *symbol* reports within ``blackout_days`` business days.

        BMO-safe: with ``blackout_days=3``, we exit at close of D-3.
        ``bdate_range(D-3, D)`` = ``[D-3, D-2, D-1, D]``, ``len - 1 = 3``.
        ``3 <= 3`` → blacked out.
        """
        earnings_date = self._earnings_cache.get(symbol)
        if earnings_date is None:
            return False
        if earnings_date <= as_of:
            return False  # past earnings, not upcoming
        bdays = pd.bdate_range(start=as_of, end=earnings_date)
        # bdate_range is inclusive both ends, so len-1 = bdays between
        return len(bdays) - 1 <= self._blackout_days

    def pair_blacked_out(
        self, symbol_y: str, symbol_x: str, as_of: date,
    ) -> str | None:
        """Returns the blacked-out symbol name, or None if clear."""
        if self.is_blacked_out(symbol_y, as_of):
            return symbol_y
        if self.is_blacked_out(symbol_x, as_of):
            return symbol_x
        return None
