"""Order construction from signal events and sizing results.

Translates a ``SignalEvent`` + ``SizeResult`` into a list of ``Order``
objects ready for broker submission.
"""

from __future__ import annotations

from stat_arb.config.constants import OrderSide, PositionDirection, Signal
from stat_arb.engine.signals import SignalEvent
from stat_arb.execution.broker_base import Order
from stat_arb.execution.sizing import SizeResult


def build_orders(
    event: SignalEvent,
    size: SizeResult,
    pair_id: int,
    direction: PositionDirection | None = None,
) -> list[Order]:
    """Build broker orders from a signal event and size result.

    Signal-to-order mapping:

    - **LONG_SPREAD** → BUY Y + SELL X (entry)
    - **SHORT_SPREAD** → SELL Y + BUY X (entry)
    - **EXIT / STOP** → reverse of entry (exit), direction-aware
    - **FLAT** → no orders

    Args:
        event: Signal event with signal type and pair info.
        size: Sizing result with share quantities per leg.
        pair_id: Database pair identifier for order tracking.
        direction: Original position direction (LONG or SHORT).
            Required for EXIT/STOP to reverse correctly.
            Defaults to LONG behavior when None.

    Returns:
        List of ``Order`` objects (0, 1, or 2 orders).
    """
    sig = event.signal
    pair = event.pair

    if sig == Signal.FLAT:
        return []

    if sig == Signal.LONG_SPREAD:
        return [
            Order(
                symbol=pair.symbol_y,
                side=OrderSide.BUY,
                quantity=size.qty_y,
                pair_id=pair_id,
                is_entry=True,
            ),
            Order(
                symbol=pair.symbol_x,
                side=OrderSide.SELL,
                quantity=size.qty_x,
                pair_id=pair_id,
                is_entry=True,
            ),
        ]

    if sig == Signal.SHORT_SPREAD:
        return [
            Order(
                symbol=pair.symbol_y,
                side=OrderSide.SELL,
                quantity=size.qty_y,
                pair_id=pair_id,
                is_entry=True,
            ),
            Order(
                symbol=pair.symbol_x,
                side=OrderSide.BUY,
                quantity=size.qty_x,
                pair_id=pair_id,
                is_entry=True,
            ),
        ]

    # EXIT or STOP — reverse the position (is_entry=False)
    if direction == PositionDirection.SHORT:
        # SHORT entry was SELL Y + BUY X → exit is BUY Y + SELL X
        return [
            Order(
                symbol=pair.symbol_y,
                side=OrderSide.BUY,
                quantity=size.qty_y,
                pair_id=pair_id,
                is_entry=False,
            ),
            Order(
                symbol=pair.symbol_x,
                side=OrderSide.SELL,
                quantity=size.qty_x,
                pair_id=pair_id,
                is_entry=False,
            ),
        ]

    # Default (LONG or None): LONG entry was BUY Y + SELL X → exit is SELL Y + BUY X
    return [
        Order(
            symbol=pair.symbol_y,
            side=OrderSide.SELL,
            quantity=size.qty_y,
            pair_id=pair_id,
            is_entry=False,
        ),
        Order(
            symbol=pair.symbol_x,
            side=OrderSide.BUY,
            quantity=size.qty_x,
            pair_id=pair_id,
            is_entry=False,
        ),
    ]
