"""Execution broker abstract base class and order/fill data models.

Defines the ``Order`` and ``Fill`` frozen dataclasses shared by all broker
implementations, plus the ``ExecutionBroker`` ABC that paper, sim, and
live brokers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from stat_arb.config.constants import OrderSide


@dataclass(frozen=True)
class Order:
    """Immutable order to be submitted to a broker.

    Attributes:
        symbol: Ticker symbol.
        side: BUY or SELL.
        quantity: Number of shares (always positive).
        pair_id: Identifier linking this order to a pair.
        is_entry: True for entry trades, False for exits/stops.
    """

    symbol: str
    side: OrderSide
    quantity: int
    pair_id: int
    is_entry: bool


@dataclass(frozen=True)
class Fill:
    """Immutable fill record returned by a broker after execution.

    Attributes:
        symbol: Ticker symbol.
        side: BUY or SELL.
        quantity: Number of shares filled.
        price: Execution price per share.
        fill_time: Timestamp of fill.
        broker_order_id: Broker-assigned order identifier.
        pair_id: Identifier linking this fill to a pair.
        is_entry: True for entry trades, False for exits/stops.
    """

    symbol: str
    side: OrderSide
    quantity: int
    price: float
    fill_time: datetime
    broker_order_id: str
    pair_id: int
    is_entry: bool


class ExecutionBroker(ABC):
    """Abstract base class for execution brokers.

    Implementations must handle order submission, portfolio valuation,
    and gross exposure calculation.
    """

    @abstractmethod
    def submit_orders(self, orders: list[Order]) -> list[Fill]:
        """Submit a batch of orders and return fills.

        Args:
            orders: Orders to execute.

        Returns:
            List of fill records, one per successfully executed order.
        """

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Return the current total portfolio value (cash + positions)."""

    @abstractmethod
    def get_gross_exposure(self) -> float:
        """Return the current gross notional exposure (sum of |position × price|)."""
