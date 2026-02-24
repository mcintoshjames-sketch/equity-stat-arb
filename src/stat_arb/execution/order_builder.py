"""Order construction from signal events and sizing results.

Translates a ``SignalEvent`` + ``SizeResult`` into a list of ``Order``
objects ready for broker submission.
"""

from __future__ import annotations

from stat_arb.config.constants import OrderSide, Signal
from stat_arb.engine.signals import SignalEvent
from stat_arb.execution.broker_base import Order
from stat_arb.execution.sizing import SizeResult


def build_orders(
    event: SignalEvent,
    size: SizeResult,
    pair_id: int,
) -> list[Order]:
    """Build broker orders from a signal event and size result.

    Signal-to-order mapping:

    - **LONG_SPREAD** → BUY Y + SELL X (entry)
    - **SHORT_SPREAD** → SELL Y + BUY X (entry)
    - **EXIT / STOP** → reverse of entry (exit)
    - **FLAT** → no orders

    For EXIT/STOP, the caller must have tracked the original direction.
    EXIT reverses LONG_SPREAD (SELL Y + BUY X), STOP reverses similarly.

    Args:
        event: Signal event with signal type and pair info.
        size: Sizing result with share quantities per leg.
        pair_id: Database pair identifier for order tracking.

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
    # EXIT reverses: sell Y leg, buy X leg (assuming was long spread)
    # The caller pipeline must track direction; here we default to
    # closing a long-spread position.  For short-spread exits the
    # caller swaps before calling.
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
