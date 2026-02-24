"""Pair discovery orchestrator with parallel evaluation.

Coordinates the full discovery pipeline: fetch prices, pre-filter by
correlation, evaluate candidates in parallel via ``joblib``, and return
sorted qualified pairs.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from joblib import Parallel, delayed

from stat_arb.discovery.pair_filter import PairFilter, QualifiedPair

if TYPE_CHECKING:
    from stat_arb.config.settings import DiscoveryConfig
    from stat_arb.data.price_repo import PriceRepository
    from stat_arb.data.universe import Universe

logger = logging.getLogger(__name__)


class PairDiscovery:
    """Orchestrates cointegration-based pair discovery.

    Args:
        config: Discovery configuration with thresholds and parallelism settings.
        price_repo: Repository for fetching close prices.
    """

    def __init__(self, config: DiscoveryConfig, price_repo: PriceRepository) -> None:
        self._config = config
        self._price_repo = price_repo
        self._pair_filter = PairFilter(config)

    def discover(
        self,
        universe: Universe,
        formation_start: date,
        formation_end: date,
    ) -> list[QualifiedPair]:
        """Run the full discovery pipeline on the universe.

        Args:
            universe: Tradable symbol universe with sector mappings.
            formation_start: Start of the formation window (passed by caller).
            formation_end: End of the formation window (passed by caller).

        Returns:
            List of qualified pairs sorted by cointegration p-value (ascending).
        """
        # Step 1: Fetch close prices for all universe symbols
        prices_df = self._price_repo.get_close_prices(
            universe.symbols, formation_start, formation_end,
        )

        if prices_df.empty:
            logger.warning(
                "No price data for formation window %s to %s",
                formation_start, formation_end,
            )
            return []

        # Step 2: Generate all intra-sector pairs
        all_pairs = universe.sector_pairs
        logger.info("Total intra-sector candidate pairs: %d", len(all_pairs))

        # Step 3: Pre-filter by correlation
        candidates = []
        for sym_y, sym_x, sector in all_pairs:
            if sym_y not in prices_df.columns or sym_x not in prices_df.columns:
                continue

            y_prices = prices_df[sym_y].dropna()
            x_prices = prices_df[sym_x].dropna()

            # Align on common dates
            common_idx = y_prices.index.intersection(x_prices.index)
            if len(common_idx) < 60:
                continue

            y_aligned = y_prices.loc[common_idx]
            x_aligned = x_prices.loc[common_idx]

            corr = np.corrcoef(y_aligned.values, x_aligned.values)[0, 1]
            if corr >= self._config.min_correlation:
                candidates.append((sym_y, sym_x, sector, y_aligned, x_aligned))

        logger.info(
            "Correlation survivors: %d / %d (threshold=%.2f)",
            len(candidates), len(all_pairs), self._config.min_correlation,
        )

        if not candidates:
            return []

        # Step 4: Parallel evaluation through PairFilter
        results = Parallel(n_jobs=self._config.parallel_n_jobs)(
            delayed(self._pair_filter.evaluate)(
                sym_y, sym_x, sector, y_prices, x_prices,
                formation_start, formation_end,
            )
            for sym_y, sym_x, sector, y_prices, x_prices in candidates
        )

        qualified = [r for r in results if r is not None]

        # Sort by cointegration p-value (best first)
        qualified.sort(key=lambda p: p.coint_pvalue)

        logger.info(
            "Discovery complete: %d qualified pairs from %d candidates",
            len(qualified), len(candidates),
        )

        return qualified
