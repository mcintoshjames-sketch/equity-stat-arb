"""Marginal delta rebalancing for window transitions.

When a pair rolls over from Window N to Window N+1 with a new hedge
ratio (β), the physical inventory must be adjusted.  Instead of
closing and reopening (4× spread crossings), we compute the signed
delta between current holdings and β-weighted targets, then issue
only the marginal top-up or trim orders.

Capital constraint (``max_gross_per_pair``) is enforced after
rebalancing — if prices drifted upward, both legs may be trimmed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from stat_arb.config.constants import OrderSide, PositionDirection, RebalanceAction
from stat_arb.execution.broker_base import Order

if TYPE_CHECKING:
    from stat_arb.config.settings import SizingConfig
    from stat_arb.discovery.pair_filter import QualifiedPair

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RebalanceResult:
    """Outcome of reconciling one pair's inventory at a window transition.

    Attributes:
        pair_key: ``"symbol_y/symbol_x"`` identifier.
        action: ROLLOVER, FORCE_EXIT, or NO_CHANGE.
        orders: Marginal orders to execute (may be empty for NO_CHANGE).
        old_beta: Hedge ratio from the previous window (None for force-exit).
        new_beta: Hedge ratio from the new window (None for force-exit).
        delta_qty_y: Signed change in Y-leg shares.
        delta_qty_x: Signed change in X-leg shares.
        shares_traded: Total shares crossing the spread (|Δy| + |Δx|).
    """

    pair_key: str
    action: RebalanceAction
    orders: list[Order]
    old_beta: float | None
    new_beta: float | None
    delta_qty_y: int
    delta_qty_x: int
    shares_traded: int


@dataclass
class OpenPositionView:
    """Read-only snapshot of an active position for rebalancing.

    Attributes:
        pair_key: ``"symbol_y/symbol_x"`` identifier.
        direction: LONG or SHORT spread direction.
        signed_qty_y: Signed Y-leg shares (positive = long).
        signed_qty_x: Signed X-leg shares (positive = long).
        pair_id: Broker pair tracking identifier.
    """

    pair_key: str
    direction: PositionDirection
    signed_qty_y: int
    signed_qty_x: int
    pair_id: int


class InventoryRebalancer:
    """Compute marginal delta orders for window-transition inventory reconciliation.

    Uses β-weighted sizing: the Y leg is sized at ``dollars_per_leg``,
    and the X leg is derived as ``floor(β × qty_y)`` to maintain the
    cointegration hedge ratio in physical inventory.

    Args:
        config: Sizing configuration with ``dollars_per_leg`` and
            ``max_gross_per_pair``.
    """

    def __init__(self, config: SizingConfig) -> None:
        self._dollars_per_leg = config.dollars_per_leg
        self._max_gross = config.max_gross_per_pair

    def reconcile(
        self,
        active_positions: dict[str, OpenPositionView],
        old_pairs: dict[str, QualifiedPair],
        new_pairs: dict[str, QualifiedPair],
        mid_prices: dict[str, float],
    ) -> list[RebalanceResult]:
        """Reconcile all active positions against a new set of qualified pairs.

        Args:
            active_positions: Current inventory keyed by pair_key.
            old_pairs: QualifiedPairs from the expiring window.
            new_pairs: QualifiedPairs from the new window.
            mid_prices: Current mid prices for all symbols.

        Returns:
            List of ``RebalanceResult`` — one per active position.
        """
        results: list[RebalanceResult] = []

        for pair_key, pos in active_positions.items():
            if pair_key not in new_pairs:
                results.append(self._force_exit(pos))
            else:
                old_pair = old_pairs.get(pair_key)
                new_pair = new_pairs[pair_key]
                result = self._compute_delta(pos, old_pair, new_pair, mid_prices)
                results.append(result)

        return results

    def beta_target(
        self,
        mid_y: float,
        mid_x: float,
        beta: float,
    ) -> tuple[int, int]:
        """Compute β-weighted absolute target quantities for both legs.

        Y leg is sized at ``dollars_per_leg``, X leg is derived as
        ``floor(β × qty_y)``.  If gross notional exceeds ``max_gross_per_pair``,
        both legs are scaled down proportionally (X re-derived from scaled Y).

        Args:
            mid_y: Mid price for Y leg.
            mid_x: Mid price for X leg.
            beta: Hedge ratio from the new formation window.

        Returns:
            ``(abs_qty_y, abs_qty_x)`` — unsigned target quantities.
        """
        if mid_y <= 0 or mid_x <= 0:
            return 0, 0

        qty_y = math.floor(self._dollars_per_leg / mid_y)
        qty_x = math.floor(beta * qty_y)

        gross = qty_y * mid_y + qty_x * mid_x
        if gross > self._max_gross and gross > 0:
            scale = self._max_gross / gross
            qty_y = math.floor(qty_y * scale)
            qty_x = math.floor(beta * qty_y)

        return qty_y, qty_x

    def _compute_delta(
        self,
        pos: OpenPositionView,
        old_pair: QualifiedPair | None,
        new_pair: QualifiedPair,
        mid_prices: dict[str, float],
    ) -> RebalanceResult:
        """Compute marginal orders to align inventory with new β."""
        mid_y = mid_prices.get(new_pair.symbol_y)
        mid_x = mid_prices.get(new_pair.symbol_x)
        if mid_y is None or mid_x is None:
            logger.warning(
                "Missing prices for %s — skipping rebalance", pos.pair_key,
            )
            return RebalanceResult(
                pair_key=pos.pair_key,
                action=RebalanceAction.NO_CHANGE,
                orders=[],
                old_beta=old_pair.hedge_ratio if old_pair else None,
                new_beta=new_pair.hedge_ratio,
                delta_qty_y=0,
                delta_qty_x=0,
                shares_traded=0,
            )

        abs_qty_y, abs_qty_x = self.beta_target(mid_y, mid_x, new_pair.hedge_ratio)

        # Apply direction sign:
        #   LONG_SPREAD  → +Y (long), -X (short)
        #   SHORT_SPREAD → -Y (short), +X (long)
        sign = 1 if pos.direction == PositionDirection.LONG else -1
        tgt_y = sign * abs_qty_y
        tgt_x = -sign * abs_qty_x

        delta_y = tgt_y - pos.signed_qty_y
        delta_x = tgt_x - pos.signed_qty_x

        if delta_y == 0 and delta_x == 0:
            return RebalanceResult(
                pair_key=pos.pair_key,
                action=RebalanceAction.NO_CHANGE,
                orders=[],
                old_beta=old_pair.hedge_ratio if old_pair else None,
                new_beta=new_pair.hedge_ratio,
                delta_qty_y=0,
                delta_qty_x=0,
                shares_traded=0,
            )

        orders = self._build_delta_orders(
            new_pair, pos.pair_id, delta_y, delta_x,
        )

        logger.info(
            "REBALANCE %s β=%.3f→%.3f Δy=%+d Δx=%+d shares=%d",
            pos.pair_key,
            old_pair.hedge_ratio if old_pair else 0.0,
            new_pair.hedge_ratio,
            delta_y, delta_x,
            abs(delta_y) + abs(delta_x),
        )

        return RebalanceResult(
            pair_key=pos.pair_key,
            action=RebalanceAction.ROLLOVER,
            orders=orders,
            old_beta=old_pair.hedge_ratio if old_pair else None,
            new_beta=new_pair.hedge_ratio,
            delta_qty_y=delta_y,
            delta_qty_x=delta_x,
            shares_traded=abs(delta_y) + abs(delta_x),
        )

    def _force_exit(self, pos: OpenPositionView) -> RebalanceResult:
        """Generate full liquidation orders for a dropped pair."""
        orders: list[Order] = []
        pair_key_parts = pos.pair_key.split("/")
        symbol_y = pair_key_parts[0]
        symbol_x = pair_key_parts[1]

        if pos.signed_qty_y != 0:
            orders.append(Order(
                symbol=symbol_y,
                side=OrderSide.SELL if pos.signed_qty_y > 0 else OrderSide.BUY,
                quantity=abs(pos.signed_qty_y),
                pair_id=pos.pair_id,
                is_entry=False,
            ))
        if pos.signed_qty_x != 0:
            orders.append(Order(
                symbol=symbol_x,
                side=OrderSide.SELL if pos.signed_qty_x > 0 else OrderSide.BUY,
                quantity=abs(pos.signed_qty_x),
                pair_id=pos.pair_id,
                is_entry=False,
            ))

        logger.info(
            "FORCE EXIT %s: liquidating qty_y=%+d qty_x=%+d",
            pos.pair_key, pos.signed_qty_y, pos.signed_qty_x,
        )

        return RebalanceResult(
            pair_key=pos.pair_key,
            action=RebalanceAction.FORCE_EXIT,
            orders=orders,
            old_beta=None,
            new_beta=None,
            delta_qty_y=-pos.signed_qty_y,
            delta_qty_x=-pos.signed_qty_x,
            shares_traded=abs(pos.signed_qty_y) + abs(pos.signed_qty_x),
        )

    @staticmethod
    def _build_delta_orders(
        pair: QualifiedPair,
        pair_id: int,
        delta_y: int,
        delta_x: int,
    ) -> list[Order]:
        """Convert signed deltas into Order objects."""
        orders: list[Order] = []
        if delta_y != 0:
            orders.append(Order(
                symbol=pair.symbol_y,
                side=OrderSide.BUY if delta_y > 0 else OrderSide.SELL,
                quantity=abs(delta_y),
                pair_id=pair_id,
                is_entry=False,
            ))
        if delta_x != 0:
            orders.append(Order(
                symbol=pair.symbol_x,
                side=OrderSide.BUY if delta_x > 0 else OrderSide.SELL,
                quantity=abs(delta_x),
                pair_id=pair_id,
                is_entry=False,
            ))
        return orders
