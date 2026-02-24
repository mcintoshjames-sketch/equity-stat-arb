"""Position sizing for pair legs.

Computes share quantities per leg based on dollar allocation, respecting
the maximum gross notional constraint per pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stat_arb.config.settings import SizingConfig


@dataclass(frozen=True)
class SizeResult:
    """Immutable sizing output for a single pair trade.

    Attributes:
        qty_y: Share count for the Y leg.
        qty_x: Share count for the X leg.
        notional_y: Dollar notional for the Y leg.
        notional_x: Dollar notional for the X leg.
    """

    qty_y: int
    qty_x: int
    notional_y: float
    notional_x: float

    @property
    def gross_notional(self) -> float:
        """Combined notional across both legs."""
        return self.notional_y + self.notional_x


class PositionSizer:
    """Dollar-based position sizer for pair legs.

    Args:
        config: Sizing configuration with dollars_per_leg and max_gross_per_pair.
    """

    def __init__(self, config: SizingConfig) -> None:
        self._config = config

    def size(self, mid_y: float, mid_x: float) -> SizeResult:
        """Compute share quantities for a pair trade.

        Args:
            mid_y: Current mid price of the Y leg.
            mid_x: Current mid price of the X leg.

        Returns:
            ``SizeResult`` with floored share counts and notionals.

        Raises:
            ValueError: If either mid price is zero or negative.
        """
        if mid_y <= 0 or mid_x <= 0:
            raise ValueError(
                f"Mid prices must be positive, got mid_y={mid_y}, mid_x={mid_x}"
            )

        dollars = self._config.dollars_per_leg
        max_gross = self._config.max_gross_per_pair

        qty_y = math.floor(dollars / mid_y)
        qty_x = math.floor(dollars / mid_x)

        notional_y = qty_y * mid_y
        notional_x = qty_x * mid_x
        gross = notional_y + notional_x

        # Scale down if gross exceeds cap
        if gross > max_gross and gross > 0:
            scale = max_gross / gross
            qty_y = math.floor(qty_y * scale)
            qty_x = math.floor(qty_x * scale)
            notional_y = qty_y * mid_y
            notional_x = qty_x * mid_x

        return SizeResult(
            qty_y=qty_y,
            qty_x=qty_x,
            notional_y=notional_y,
            notional_x=notional_x,
        )
