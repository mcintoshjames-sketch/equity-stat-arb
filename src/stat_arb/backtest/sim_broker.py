"""Simulation broker for walk-forward backtesting.

Fills at close price +/- slippage bps.  Tracks positions and cash
for portfolio valuation during replay.

Note: Position/cash tracking logic is duplicated with PaperBroker.
This will be refactored into a shared base class in a future step.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

from stat_arb.config.constants import OrderSide
from stat_arb.execution.broker_base import ExecutionBroker, Fill, Order

logger = logging.getLogger(__name__)


class SimBroker(ExecutionBroker):
    """Backtest simulation broker with bps-based slippage.

    Args:
        slippage_bps: Slippage in basis points (e.g. 10 = 0.10%).
        initial_cash: Starting cash balance.
    """

    def __init__(
        self,
        slippage_bps: float = 10.0,
        initial_cash: float = 100_000.0,
    ) -> None:
        self._slippage_bps = slippage_bps
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._positions: dict[str, int] = {}
        self._current_date: date | None = None
        self._prices: dict[str, float] = {}

    def set_date(
        self,
        current_date: date,
        prices: dict[str, float],
    ) -> None:
        """Set replay state for the current simulation day.

        Args:
            current_date: The date being simulated.
            prices: Mapping of symbol → close price.
        """
        self._current_date = current_date
        self._prices = prices

    def submit_orders(self, orders: list[Order]) -> list[Fill]:
        """Execute orders against current close prices with bps slippage.

        BUY fills at ``close × (1 + bps/10000)`` (adverse).
        SELL fills at ``close × (1 - bps/10000)`` (adverse).

        Args:
            orders: Orders to execute.

        Returns:
            List of fills for orders with available prices.
        """
        fills: list[Fill] = []
        slip_mult = self._slippage_bps / 10_000.0

        for order in orders:
            close = self._prices.get(order.symbol)
            if close is None:
                logger.warning(
                    "No price for %s on %s — skipping",
                    order.symbol, self._current_date,
                )
                continue

            if order.side == OrderSide.BUY:
                fill_price = close * (1.0 + slip_mult)
            else:
                fill_price = close * (1.0 - slip_mult)

            # Update positions and cash
            signed_qty = (
                order.quantity if order.side == OrderSide.BUY else -order.quantity
            )
            self._positions[order.symbol] = (
                self._positions.get(order.symbol, 0) + signed_qty
            )
            self._cash -= signed_qty * fill_price

            fill_time = datetime(
                self._current_date.year,
                self._current_date.month,
                self._current_date.day,
                16, 0, 0,
                tzinfo=UTC,
            ) if self._current_date else datetime.now(UTC)

            fill = Fill(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                fill_time=fill_time,
                broker_order_id=f"SIM-{id(order)}",
                pair_id=order.pair_id,
                is_entry=order.is_entry,
            )
            fills.append(fill)

        return fills

    def get_portfolio_value(self) -> float:
        """Cash plus mark-to-market value of all positions."""
        mtm = 0.0
        for symbol, qty in self._positions.items():
            price = self._prices.get(symbol, 0.0)
            mtm += qty * price
        return self._cash + mtm

    def get_gross_exposure(self) -> float:
        """Sum of absolute position values at current prices."""
        exposure = 0.0
        for symbol, qty in self._positions.items():
            price = self._prices.get(symbol, 0.0)
            exposure += abs(qty) * price
        return exposure

    def reset(self) -> None:
        """Restore to initial state for a new backtest run."""
        self._cash = self._initial_cash
        self._positions.clear()
        self._current_date = None
        self._prices.clear()
