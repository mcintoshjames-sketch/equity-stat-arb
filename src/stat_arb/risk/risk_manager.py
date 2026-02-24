"""Portfolio-level risk gating for trade entry.

Checks drawdown kill-switch, pair count, gross exposure, sector
concentration, and edge-vs-slippage before approving new entries.
Exits and stops are always approved (risk reduction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from stat_arb.config.constants import RiskDecisionType, Signal

if TYPE_CHECKING:
    from stat_arb.config.settings import RiskConfig
    from stat_arb.engine.signals import SignalEvent
    from stat_arb.execution.broker_base import ExecutionBroker
    from stat_arb.execution.sizing import SizeResult

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

    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        self._peak: float = 0.0
        self._kill_switch_active: bool = False
        self._pair_sectors: dict[int, str] = {}

    def check(
        self,
        event: SignalEvent,
        size: SizeResult,
        broker: ExecutionBroker,
        active_pair_count: int,
    ) -> RiskDecision:
        """Evaluate whether a signal should be executed.

        Args:
            event: Signal event to evaluate.
            size: Proposed sizing for the trade.
            broker: Broker for exposure queries.
            active_pair_count: Number of currently active pairs.

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

    def register_pair(self, pair_id: int, sector: str) -> None:
        """Register a pair's sector for concentration tracking.

        Args:
            pair_id: Unique pair identifier.
            sector: Sector classification.
        """
        self._pair_sectors[pair_id] = sector

    def unregister_pair(self, pair_id: int) -> None:
        """Remove a pair from sector tracking.

        Args:
            pair_id: Unique pair identifier.
        """
        self._pair_sectors.pop(pair_id, None)

    @property
    def kill_switch_active(self) -> bool:
        """Whether the drawdown kill-switch has been triggered."""
        return self._kill_switch_active

    def _sector_gross(self, sector: str, broker: ExecutionBroker) -> float:
        """Estimate gross exposure for a given sector.

        Approximates by counting pairs in the sector as a fraction of
        total gross exposure.
        """
        total_pairs = len(self._pair_sectors)
        if total_pairs == 0:
            return 0.0
        sector_pairs = sum(1 for s in self._pair_sectors.values() if s == sector)
        return broker.get_gross_exposure() * (sector_pairs / total_pairs)
