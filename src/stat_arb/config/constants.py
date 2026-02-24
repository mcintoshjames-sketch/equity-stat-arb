"""Enums and constants for the stat arb system.

All domain-specific enums live here so that no module uses magic strings.
Import individual enums where needed::

    from stat_arb.config.constants import BrokerMode, Signal
"""

from enum import StrEnum


class BrokerMode(StrEnum):
    """Execution backend selection."""

    PAPER = "paper"
    LIVE = "live"
    SIM = "sim"  # backtest simulation


class Signal(StrEnum):
    """Trading signal types emitted by the signal generator."""

    LONG_SPREAD = "long_spread"    # Z < -entry_z → buy Y, sell X
    SHORT_SPREAD = "short_spread"  # Z > +entry_z → sell Y, buy X
    EXIT = "exit"                  # mean reversion complete or timeout
    STOP = "stop"                  # divergence stop triggered
    FLAT = "flat"                  # no position


class OrderSide(StrEnum):
    """Direction of a single-leg order."""

    BUY = "BUY"
    SELL = "SELL"


class WindowPhase(StrEnum):
    """Walk-forward window phase."""

    FORMATION = "formation"
    TRADING = "trading"


class PairStatus(StrEnum):
    """Lifecycle status of a discovered pair."""

    ACTIVE = "active"
    EXPIRED = "expired"
    STOPPED = "stopped"
    DECAYED = "decayed"


class PositionDirection(StrEnum):
    """Net direction of a pair position from the spread perspective."""

    LONG = "long"    # long spread: long Y, short X
    SHORT = "short"  # short spread: short Y, long X
    FLAT = "flat"    # no open position
