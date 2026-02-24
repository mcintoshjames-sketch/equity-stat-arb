"""Structural break detection for active pairs.

Monitors whether the cointegrating relationship has broken down by
testing stationarity of the live spread using the ADF test.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from statsmodels.tsa.stattools import adfuller

if TYPE_CHECKING:
    from stat_arb.config.settings import RiskConfig
    from stat_arb.discovery.pair_filter import QualifiedPair

logger = logging.getLogger(__name__)


class StructuralBreakMonitor:
    """Detect structural breaks in pair cointegration.

    Computes the live spread using the pair's frozen formation parameters
    (β, α) and tests stationarity with ADF.  If the p-value exceeds the
    threshold, the spread is no longer stationary → structural break.

    Args:
        config: Risk configuration with ``structural_break_window`` and
            ``structural_break_pvalue`` fields.
    """

    def __init__(self, config: RiskConfig) -> None:
        self._window = config.structural_break_window
        self._pvalue_threshold = config.structural_break_pvalue

    def check_pair(
        self,
        pair: QualifiedPair,
        recent_prices_y: np.ndarray,
        recent_prices_x: np.ndarray,
    ) -> bool:
        """Test whether a pair's spread has become non-stationary.

        Args:
            pair: Qualified pair with frozen β/α.
            recent_prices_y: Recent Y-leg close prices.
            recent_prices_x: Recent X-leg close prices.

        Returns:
            ``True`` if the spread is no longer stationary (structural
            break detected), ``False`` if it remains stationary or if
            there is insufficient data.
        """
        n = min(len(recent_prices_y), len(recent_prices_x))
        if n < self._window:
            return False

        # Use the last `window` observations
        y = recent_prices_y[-self._window :]
        x = recent_prices_x[-self._window :]

        spread = y - pair.hedge_ratio * x - pair.intercept

        try:
            result = adfuller(spread, maxlag=1, autolag=None)
        except Exception:
            logger.warning(
                "ADF test failed for %s/%s — treating as no break",
                pair.symbol_y, pair.symbol_x,
            )
            return False

        pvalue = result[1]

        if pvalue > self._pvalue_threshold:
            logger.warning(
                "STRUCTURAL BREAK %s/%s: ADF p=%.4f > %.2f",
                pair.symbol_y, pair.symbol_x, pvalue, self._pvalue_threshold,
            )
            return True

        return False
