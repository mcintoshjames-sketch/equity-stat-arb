"""Portfolio-level risk gating for trade entry.

Checks drawdown kill-switch, pair count, gross exposure, sector
concentration, and edge-vs-slippage before approving new entries.
Exits and stops are always approved (risk reduction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from stat_arb.config.constants import RiskDecisionType, Signal

if TYPE_CHECKING:
    from stat_arb.config.settings import RiskConfig
    from stat_arb.engine.signals import SignalEvent
    from stat_arb.execution.broker_base import ExecutionBroker
    from stat_arb.execution.sizing import SizeResult
    from stat_arb.risk.earnings_blackout import EarningsBlackout

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskDecision:
    """Immutable risk check result.

    Attributes:
        decision: APPROVED or REJECTED.
        reason: Human-readable explanation.
    """

    decision: RiskDecisionType
    reason: str


class RiskManager:
    """Portfolio-level risk gate.

    Entry checks (in order):
    1. Drawdown kill-switch (permanent once triggered)
    2. Maximum active pair count
    3. Maximum gross exposure
    4. Maximum sector concentration
    5. Edge vs slippage

    Exits and stops are always approved.

    Args:
        config: Risk configuration with limits and thresholds.
    """

    def __init__(
        self,
        config: RiskConfig,
        earnings_blackout: EarningsBlackout | None = None,
    ) -> None:
        self._config = config
        self._earnings = earnings_blackout
        self._peak: float = 0.0
        self._kill_switch_active: bool = False
        self._pair_sectors: dict[int, str] = {}
        self._entries_this_step: int = 0
        self._pair_pnl: dict[int, float] = {}
        self._pair_notionals: dict[int, float] = {}
        self._pair_cohorts: dict[int, str] = {}

    def check(
        self,
        event: SignalEvent,
        size: SizeResult,
        broker: ExecutionBroker,
        active_pair_count: int,
        current_date: date | None = None,
    ) -> RiskDecision:
        """Evaluate whether a signal should be executed.

        Args:
            event: Signal event to evaluate.
            size: Proposed sizing for the trade.
            broker: Broker for exposure queries.
            active_pair_count: Number of currently active pairs.
            current_date: Today's date for earnings blackout check.

        Returns:
            ``RiskDecision`` with APPROVED or REJECTED and reason.
        """
        # Exits and stops are always approved (risk reduction)
        if event.signal in (Signal.EXIT, Signal.STOP):
            return RiskDecision(
                decision=RiskDecisionType.APPROVED,
                reason="exit/stop always approved",
            )

        # Not an entry signal — approve (FLAT signals shouldn't reach here)
        if event.signal not in (Signal.LONG_SPREAD, Signal.SHORT_SPREAD):
            return RiskDecision(
                decision=RiskDecisionType.APPROVED,
                reason="non-entry signal",
            )

        # --- Entry checks ---

        # 1. Drawdown kill-switch
        if self._kill_switch_active:
            return RiskDecision(
                decision=RiskDecisionType.REJECTED,
                reason="kill switch active — drawdown limit breached",
            )

        # 1a. Earnings blackout
        if self._earnings is not None and current_date is not None:
            blacked = self._earnings.pair_blacked_out(
                event.pair.symbol_y, event.pair.symbol_x, current_date,
            )
            if blacked:
                return RiskDecision(
                    decision=RiskDecisionType.REJECTED,
                    reason=f"earnings blackout for {blacked}",
                )

        # 1b. Max entries per step
        if self._entries_this_step >= self._config.max_entries_per_step:
            return RiskDecision(
                decision=RiskDecisionType.REJECTED,
                reason="max entries per step reached",
            )

        # 1c. Cohort concentration
        cohort_id = getattr(event.pair, "cohort_id", None)
        if cohort_id:
            cohort_count = sum(
                1 for c in self._pair_cohorts.values() if c == cohort_id
            )
            if cohort_count >= self._config.max_cohort_concentration:
                return RiskDecision(
                    decision=RiskDecisionType.REJECTED,
                    reason=f"cohort {cohort_id} at max concentration",
                )

        # 1d. Near-expiry guard
        trading_expiry = getattr(event.pair, "trading_expiry", None)
        if (
            trading_expiry is not None
            and current_date is not None
            and self._config.min_days_before_expiry > 0
        ):
            days_left = (trading_expiry - current_date).days
            if days_left < self._config.min_days_before_expiry:
                return RiskDecision(
                    decision=RiskDecisionType.REJECTED,
                    reason=(
                        f"only {days_left}d before expiry "
                        f"(min {self._config.min_days_before_expiry}d)"
                    ),
                )

        # 2. Max pairs
        if active_pair_count >= self._config.max_pairs:
            return RiskDecision(
                decision=RiskDecisionType.REJECTED,
                reason=f"max pairs ({self._config.max_pairs}) reached",
            )

        # 3. Max gross exposure
        current_gross = broker.get_gross_exposure()
        proposed_gross = current_gross + size.gross_notional
        if proposed_gross > self._config.max_gross_exposure:
            return RiskDecision(
                decision=RiskDecisionType.REJECTED,
                reason=(
                    f"gross exposure {proposed_gross:.0f} would exceed "
                    f"limit {self._config.max_gross_exposure:.0f}"
                ),
            )

        # 4. Sector concentration
        # Measured against max_gross_exposure to avoid bootstrap problem
        # where the first trade is always 100% of (currently empty) portfolio.
        sector = event.pair.sector
        sector_gross = self._sector_gross(sector, broker)
        total_gross = max(self._config.max_gross_exposure, 1.0)
        sector_pct = (sector_gross + size.gross_notional) / total_gross
        if sector_pct > self._config.max_sector_pct:
            return RiskDecision(
                decision=RiskDecisionType.REJECTED,
                reason=(
                    f"sector '{sector}' at {sector_pct:.1%} would exceed "
                    f"limit {self._config.max_sector_pct:.1%}"
                ),
            )

        # 5. Edge vs slippage
        if self._config.min_edge_over_slippage > 0:
            rt_cost = event.estimated_round_trip_cost
            if rt_cost > 0 and size.gross_notional > 0:
                edge_ratio = size.gross_notional / rt_cost
                if edge_ratio < self._config.min_edge_over_slippage:
                    return RiskDecision(
                        decision=RiskDecisionType.REJECTED,
                        reason=(
                            f"edge/slippage ratio {edge_ratio:.2f} below "
                            f"minimum {self._config.min_edge_over_slippage:.2f}"
                        ),
                    )

        return RiskDecision(
            decision=RiskDecisionType.APPROVED,
            reason="all risk checks passed",
        )

    def update_peak(self, portfolio_value: float) -> None:
        """Update high-water mark and check drawdown kill-switch.

        Args:
            portfolio_value: Current portfolio value.
        """
        if portfolio_value > self._peak:
            self._peak = portfolio_value

        dd = self.current_drawdown(portfolio_value)
        if dd >= self._config.max_drawdown_pct:
            if not self._kill_switch_active:
                logger.critical(
                    "KILL SWITCH ACTIVATED: drawdown %.2f%% >= limit %.2f%%",
                    dd * 100, self._config.max_drawdown_pct * 100,
                )
            self._kill_switch_active = True

    def current_drawdown(self, portfolio_value: float) -> float:
        """Compute current drawdown from the high-water mark.

        Args:
            portfolio_value: Current portfolio value.

        Returns:
            Drawdown as a positive fraction in [0, 1].
        """
        if self._peak <= 0:
            return 0.0
        return max(0.0, (self._peak - portfolio_value) / self._peak)

    def register_pair(
        self,
        pair_id: int,
        sector: str,
        cohort_id: str | None = None,
        gross_notional: float = 0.0,
    ) -> None:
        """Register a pair's sector, cohort, and notional for tracking.

        Args:
            pair_id: Unique pair identifier.
            sector: Sector classification.
            cohort_id: Optional discovery cohort identifier.
            gross_notional: Initial gross notional for the pair.
        """
        self._pair_sectors[pair_id] = sector
        if gross_notional > 0:
            self._pair_notionals[pair_id] = gross_notional
        if cohort_id:
            self._pair_cohorts[pair_id] = cohort_id

    def unregister_pair(self, pair_id: int) -> None:
        """Remove a pair from sector and cohort tracking.

        Args:
            pair_id: Unique pair identifier.
        """
        self._pair_sectors.pop(pair_id, None)
        self._pair_cohorts.pop(pair_id, None)
        self._pair_pnl.pop(pair_id, None)
        self._pair_notionals.pop(pair_id, None)

    def reset_step_counters(self) -> None:
        """Reset per-step entry counter. Call at start of each step."""
        self._entries_this_step = 0

    def record_entry(self) -> None:
        """Record a successful entry for per-step limiting."""
        self._entries_this_step += 1

    def update_pair_pnl(self, pair_id: int, unrealized_pnl: float) -> None:
        """Update the unrealized PnL for a pair.

        Args:
            pair_id: Unique pair identifier.
            unrealized_pnl: Current unrealized dollar PnL.
        """
        self._pair_pnl[pair_id] = unrealized_pnl

    def update_pair_notional(self, pair_id: int, gross_notional: float) -> None:
        """Update the current gross notional for a pair.

        Args:
            pair_id: Unique pair identifier.
            gross_notional: Current gross notional (sum of abs leg values).
        """
        self._pair_notionals[pair_id] = gross_notional

    def check_pair_pnl_stop(self, pair_id: int) -> bool:
        """Check whether a pair has breached its PnL stop.

        Returns:
            True if the pair should be force-closed.
        """
        pnl = self._pair_pnl.get(pair_id, 0.0)
        return pnl <= self._config.per_pair_pnl_stop

    def check_earnings_blackout(
        self, symbol_y: str, symbol_x: str, as_of: date,
    ) -> bool:
        """Check if either leg is within earnings blackout.

        Returns:
            True if the pair should be force-exited.
        """
        if self._earnings is None:
            return False
        return self._earnings.pair_blacked_out(symbol_y, symbol_x, as_of) is not None

    @property
    def kill_switch_active(self) -> bool:
        """Whether the drawdown kill-switch has been triggered."""
        return self._kill_switch_active

    def _sector_gross(self, sector: str, broker: ExecutionBroker) -> float:
        """Compute gross exposure for a given sector using tracked notionals.

        Falls back to count-based approximation when no notionals are
        tracked (backward compatibility).
        """
        if self._pair_notionals:
            return sum(
                self._pair_notionals.get(pid, 0.0)
                for pid, s in self._pair_sectors.items()
                if s == sector
            )
        # Fallback: count-based approximation
        total_pairs = len(self._pair_sectors)
        if total_pairs == 0:
            return 0.0
        sector_pairs = sum(1 for s in self._pair_sectors.values() if s == sector)
        return broker.get_gross_exposure() * (sector_pairs / total_pairs)
