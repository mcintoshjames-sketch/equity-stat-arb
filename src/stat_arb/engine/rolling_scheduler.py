"""Rolling discovery scheduler for continuous pair refresh.

Instead of discrete non-overlapping walk-forward windows, the rolling
scheduler periodically re-scans for cointegrated pairs on a trailing
lookback.  Each pair carries an individual expiry date, allowing
overlapping cohorts and a continuously fresh portfolio.

Key behaviours:
- Re-discovered pairs get refreshed parameters and a reset expiry.
- ``expired_keys`` and ``refreshed_keys`` are populated each step
  for the engine to consume.
- ``_prev_pairs`` preserves old ``QualifiedPair`` for the rebalancer's
  ``old_pairs`` parameter.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from stat_arb.discovery.pair_filter import with_lifecycle

if TYPE_CHECKING:
    from stat_arb.config.settings import DiscoveryConfig, RollingSchedulerConfig
    from stat_arb.data.price_repo import PriceRepository
    from stat_arb.data.universe import Universe
    from stat_arb.discovery.pair_discovery import PairDiscovery
    from stat_arb.discovery.pair_filter import QualifiedPair

logger = logging.getLogger(__name__)


class RollingScheduler:
    """Rolling discovery scheduler with per-pair expiry.

    Args:
        config: Rolling scheduler configuration.
        discovery_config: Discovery pipeline parameters (for formation_days).
        pair_discovery: Pair discovery pipeline instance.
        universe: Tradable symbol universe.
        price_repo: Price repository for date calculations.
    """

    def __init__(
        self,
        config: RollingSchedulerConfig,
        discovery_config: DiscoveryConfig,
        pair_discovery: PairDiscovery,
        universe: Universe,
        price_repo: PriceRepository,
    ) -> None:
        self._config = config
        self._discovery_config = discovery_config
        self._discovery = pair_discovery
        self._universe = universe
        self._price_repo = price_repo

        self._active_pairs: dict[tuple[str, str], QualifiedPair] = {}
        self._prev_pairs: dict[tuple[str, str], QualifiedPair] = {}
        self._expired_keys: list[tuple[str, str]] = []
        self._refreshed_keys: list[tuple[str, str]] = []
        self._last_discovery: date | None = None
        self._cohort_counter: int = 0

    @property
    def active_pairs(self) -> list[QualifiedPair]:
        """Currently active qualified pairs."""
        return list(self._active_pairs.values())

    @property
    def expired_keys(self) -> list[tuple[str, str]]:
        """Pairs expired this step (consumed by engine for FORCE_EXIT)."""
        return list(self._expired_keys)

    @property
    def refreshed_keys(self) -> list[tuple[str, str]]:
        """Pairs whose params changed this step (consumed by engine for ROLLOVER)."""
        return list(self._refreshed_keys)

    def step(self, current_date: date) -> bool:
        """Called each trading day.  Returns True if discovery ran.

        Populates ``expired_keys`` and ``refreshed_keys`` for the
        engine to act on.
        """
        self._expired_keys = []
        self._refreshed_keys = []
        self._expire_pairs(current_date)
        if self._should_discover(current_date):
            self._run_discovery(current_date)
            return True
        return False

    def get_prev_pair(self, key: tuple[str, str]) -> QualifiedPair | None:
        """Get the previous QualifiedPair for a refreshed/expired key."""
        return self._prev_pairs.get(key)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_discover(self, current_date: date) -> bool:
        """Check whether enough business days have elapsed since last discovery."""
        if self._last_discovery is None:
            return True
        bdays = pd.bdate_range(
            start=self._last_discovery, end=current_date,
        )
        # bdate_range is inclusive of both ends, so subtract 1 for elapsed days
        return len(bdays) - 1 >= self._config.discovery_interval_days

    def _run_discovery(self, current_date: date) -> None:
        """Execute a discovery cycle and merge results into active pairs."""
        # Snapshot current pairs before overwriting
        self._prev_pairs = dict(self._active_pairs)

        # Compute formation window: trailing lookback ending at current_date
        bdays = pd.bdate_range(
            end=current_date,
            periods=self._config.formation_days,
        )
        formation_start = bdays[0].date()
        formation_end = bdays[-1].date()

        # Compute trading expiry
        expiry_bdays = pd.bdate_range(
            start=current_date, periods=self._config.trading_days + 1,
        )
        trading_expiry = expiry_bdays[-1].date()

        # Generate cohort ID
        self._cohort_counter += 1
        cohort_id = f"C{self._cohort_counter:04d}"

        logger.info(
            "Rolling discovery %s: formation %s→%s, expiry %s",
            cohort_id, formation_start, formation_end, trading_expiry,
        )

        new_pairs = self._discovery.discover(
            self._universe, formation_start, formation_end,
        )

        # Enrich with lifecycle and merge into active set
        added = 0
        for pair in new_pairs:
            if added >= self._config.max_cohort_pairs:
                break

            enriched = with_lifecycle(
                pair, discovery_date=current_date,
                trading_expiry=trading_expiry, cohort_id=cohort_id,
            )
            key = (pair.symbol_y, pair.symbol_x)

            if key in self._active_pairs:
                self._refreshed_keys.append(key)
            self._active_pairs[key] = enriched
            added += 1

        self._last_discovery = current_date

        logger.info(
            "Rolling discovery complete: %d added/refreshed, "
            "%d total active, %d refreshed",
            added, len(self._active_pairs), len(self._refreshed_keys),
        )

    def _expire_pairs(self, current_date: date) -> None:
        """Remove pairs past their trading_expiry."""
        expired = [
            k for k, p in self._active_pairs.items()
            if p.trading_expiry is not None and current_date > p.trading_expiry
        ]
        for k in expired:
            self._prev_pairs[k] = self._active_pairs.pop(k)
            self._expired_keys.append(k)

        if expired:
            logger.info(
                "Expired %d pairs: %s",
                len(expired),
                [f"{k[0]}/{k[1]}" for k in expired],
            )
