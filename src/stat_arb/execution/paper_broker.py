"""Paper (simulated live) broker for forward testing.

Fills at mid +/- slippage with adverse fill logic.  Tracks positions
and cash for portfolio valuation.

Note: Position/cash tracking logic is duplicated with SimBroker.
This will be refactored into a shared base class in a future step.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from stat_arb.config.constants import OrderSide
from stat_arb.execution.broker_base import ExecutionBroker, Fill, Order

if TYPE_CHECKING:
    from stat_arb.config.settings import SignalConfig

logger = logging.getLogger(__name__)


class PaperBroker(ExecutionBroker):
    """Paper trading broker with adverse-fill slippage model.

    Args:
        signal_config: Signal configuration containing ``slippage_multiplier``.
        initial_cash: Starting cash balance.
    """

    def __init__(
        self,
        signal_config: SignalConfig,
        initial_cash: float = 50_000.0,
    ) -> None:
        self._slippage_mult = signal_config.slippage_multiplier
        self._cash = initial_cash
        self._positions: dict[str, int] = {}
        self._quotes: dict[str, dict[str, float]] = {}

    def update_quotes(self, quotes: dict[str, dict[str, float]]) -> None:
        """Set the current quote snapshot for fill pricing.

        Args:
            quotes: Mapping of symbol → {``"bid"``, ``"ask"``}.
        """
        self._quotes = quotes

    def submit_orders(self, orders: list[Order]) -> list[Fill]:
        """Execute orders against current quotes with adverse slippage.

        BUY fills at ``mid + slippage_mult × half_spread`` (pay more).
        SELL fills at ``mid - slippage_mult × half_spread`` (receive less).

        Args:
            orders: Orders to execute.

        Returns:
            List of fills for orders with available quotes.
        """
        fills: list[Fill] = []
        for order in orders:
            quote = self._quotes.get(order.symbol)
            if quote is None or "bid" not in quote or "ask" not in quote:
                logger.warning(
                    "No quote for %s — skipping order", order.symbol,
                )
                continue

            bid = quote["bid"]
            ask = quote["ask"]
            mid = (bid + ask) / 2.0
            half_spread = (ask - bid) / 2.0

            if order.side == OrderSide.BUY:
                fill_price = mid + self._slippage_mult * half_spread
            else:
                fill_price = mid - self._slippage_mult * half_spread

            # Update positions and cash
            signed_qty = (
                order.quantity if order.side == OrderSide.BUY else -order.quantity
            )
            self._positions[order.symbol] = (
                self._positions.get(order.symbol, 0) + signed_qty
            )
            self._cash -= signed_qty * fill_price

            fill = Fill(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                fill_time=datetime.now(UTC),
                broker_order_id=f"PAPER-{id(order)}",
                pair_id=order.pair_id,
                is_entry=order.is_entry,
            )
            fills.append(fill)

            logger.info(
                "PAPER FILL %s %s %d @ %.4f (pair=%d)",
                order.side.value, order.symbol, order.quantity,
                fill_price, order.pair_id,
            )

        return fills

    def get_portfolio_value(self) -> float:
        """Cash plus mark-to-market value of all positions."""
        mtm = 0.0
        for symbol, qty in self._positions.items():
            quote = self._quotes.get(symbol)
            if quote and "bid" in quote and "ask" in quote:
                mid = (quote["bid"] + quote["ask"]) / 2.0
                mtm += qty * mid
        return self._cash + mtm

    def get_gross_exposure(self) -> float:
        """Sum of absolute position values at mid prices."""
        exposure = 0.0
        for symbol, qty in self._positions.items():
            quote = self._quotes.get(symbol)
            if quote and "bid" in quote and "ask" in quote:
                mid = (quote["bid"] + quote["ask"]) / 2.0
                exposure += abs(qty) * mid
        return exposure
