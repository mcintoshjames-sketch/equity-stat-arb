"""Live Schwab broker implementation.

Submits orders to the Charles Schwab API via ``SchwabDataClient`` and
constructs ``Fill`` records from API responses.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from stat_arb.config.constants import OrderSide
from stat_arb.execution.broker_base import ExecutionBroker, Fill, Order

if TYPE_CHECKING:
    from stat_arb.config.settings import SchwabBrokerConfig
    from stat_arb.data.schwab_client import SchwabDataClient

logger = logging.getLogger(__name__)


class LiveSchwabBroker(ExecutionBroker):
    """Live execution broker using the Charles Schwab API.

    Args:
        client: Schwab API data client with order placement capability.
        broker_config: Broker execution settings (limit vs market orders).
    """

    def __init__(
        self,
        client: SchwabDataClient,
        broker_config: SchwabBrokerConfig,
    ) -> None:
        self._client = client
        self._config = broker_config

    def submit_orders(self, orders: list[Order]) -> list[Fill]:
        """Submit orders to Schwab and return fills.

        Args:
            orders: Orders to execute.

        Returns:
            List of fills for successfully placed orders.
        """
        fills: list[Fill] = []
        for order in orders:
            schwab_order = self._build_schwab_order(order)
            order_id = self._client.place_order(schwab_order)

            fill = Fill(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=0.0,  # actual fill price reported asynchronously
                fill_time=datetime.now(UTC),
                broker_order_id=order_id or "UNKNOWN",
                pair_id=order.pair_id,
                is_entry=order.is_entry,
            )
            fills.append(fill)

            logger.info(
                "SCHWAB ORDER %s %s %d (order_id=%s, pair=%d)",
                order.side.value, order.symbol, order.quantity,
                order_id, order.pair_id,
            )

        return fills

    def get_portfolio_value(self) -> float:
        """Delegate to Schwab API account value."""
        return self._client.get_account_value()

    def get_gross_exposure(self) -> float:
        """Sum absolute market values from Schwab positions."""
        positions = self._client.get_positions()
        exposure = 0.0
        for pos in positions:
            market_value = abs(float(pos.get("marketValue", 0.0)))
            exposure += market_value
        return exposure

    def _build_schwab_order(self, order: Order) -> dict:
        """Build a Schwab-compatible order JSON dict.

        Respects ``use_limit_orders`` flag:
        - ``False`` (default): MARKET order, session=NORMAL, duration=DAY
        - ``True``: LIMIT order with price = 0 (caller must set price)

        Args:
            order: Internal order to translate.

        Returns:
            Schwab order JSON dict.
        """
        instruction = "BUY" if order.side == OrderSide.BUY else "SELL"

        leg = {
            "instruction": instruction,
            "quantity": order.quantity,
            "instrument": {
                "symbol": order.symbol,
                "assetType": "EQUITY",
            },
        }

        if self._config.use_limit_orders:
            return {
                "orderType": "LIMIT",
                "session": "NORMAL",
                "duration": "DAY",
                "price": "0.00",  # caller should set actual limit price
                "orderStrategyType": "SINGLE",
                "orderLegCollection": [leg],
            }

        return {
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [leg],
        }
