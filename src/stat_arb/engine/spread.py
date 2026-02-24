"""Spread computation and Z-score calculation using frozen formation params.

Computes the live spread (y − βx − α) and its Z-score from real-time
quotes, and estimates quote-width-based round-trip transaction costs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stat_arb.config.settings import SignalConfig, SizingConfig
    from stat_arb.discovery.pair_filter import QualifiedPair

logger = logging.getLogger(__name__)


class SpreadComputer:
    """Compute live spread, Z-score, and slippage from frozen pair parameters.

    Uses the formation-window hedge ratio, intercept, mean, and std
    that are frozen in the ``QualifiedPair`` — no re-estimation during
    the trading window.

    Args:
        signal_config: Signal configuration with slippage multiplier.
        sizing_config: Position sizing for dollar-based cost estimation.
    """

    def __init__(
        self,
        signal_config: SignalConfig,
        sizing_config: SizingConfig,
    ) -> None:
        self._signal_config = signal_config
        self._sizing_config = sizing_config

    def compute_z_score(
        self,
        pair: QualifiedPair,
        current_prices: dict[str, float],
    ) -> float:
        """Compute the current Z-score for a qualified pair.

        Args:
            pair: Frozen pair with formation-window parameters (β, α, μ, σ).
            current_prices: Mapping of symbol → mid price.

        Returns:
            Z-score: ``(spread − μ) / σ``.

        Raises:
            KeyError: If either leg's symbol is missing from *current_prices*.
        """
        price_y = current_prices[pair.symbol_y]
        price_x = current_prices[pair.symbol_x]

        spread = price_y - pair.hedge_ratio * price_x - pair.intercept
        z_score = (spread - pair.spread_mean) / pair.spread_std

        logger.debug(
            "%s/%s spread=%.4f z=%.3f (μ=%.4f σ=%.4f)",
            pair.symbol_y, pair.symbol_x,
            spread, z_score, pair.spread_mean, pair.spread_std,
        )
        return z_score

    def estimate_round_trip_cost(
        self,
        pair: QualifiedPair,
        quotes: dict[str, dict[str, float]],
    ) -> float:
        """Estimate round-trip transaction cost from quote widths.

        Uses quote-width-based slippage: fill at mid ± multiplier × half_spread
        for each leg, summed over both legs and a round trip (open + close).

        Cost model per leg per side::

            slippage = slippage_multiplier × (ask − bid) / 2

        Round trip for both legs::

            total = 2 × (slippage_y + slippage_x)

        Dollar cost is estimated at ``dollars_per_leg`` per leg.

        Args:
            pair: Qualified pair identifying the two legs.
            quotes: Mapping of symbol → {``"bid"``, ``"ask"``} dict.

        Returns:
            Estimated round-trip dollar cost.  Returns 0.0 if bid/ask
            data is unavailable for either leg.
        """
        mult = self._signal_config.slippage_multiplier
        dollars = self._sizing_config.dollars_per_leg
        total_cost = 0.0

        for sym in (pair.symbol_y, pair.symbol_x):
            q = quotes.get(sym)
            if q is None or "bid" not in q or "ask" not in q:
                logger.debug(
                    "No bid/ask for %s — slippage estimate unavailable", sym,
                )
                return 0.0

            bid, ask = q["bid"], q["ask"]
            mid = (bid + ask) / 2.0
            if mid <= 0:
                return 0.0

            half_spread = (ask - bid) / 2.0
            slippage_per_share = mult * half_spread
            shares = dollars / mid
            # Round trip: entry + exit = 2 × slippage
            total_cost += 2.0 * slippage_per_share * shares

        return total_cost
