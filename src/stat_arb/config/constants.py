"""Enums and constants for the stat arb system."""

from enum import Enum, auto


class BrokerMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    SIM = "sim"  # backtest simulation


class Signal(str, Enum):
    LONG_SPREAD = "long_spread"    # Z < -entry_z → buy Y, sell X
    SHORT_SPREAD = "short_spread"  # Z > +entry_z → sell Y, buy X
    EXIT = "exit"                  # mean reversion complete or timeout
    STOP = "stop"                  # divergence stop triggered
    FLAT = "flat"                  # no position


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class WindowPhase(str, Enum):
    FORMATION = "formation"
    TRADING = "trading"


class PairStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    STOPPED = "stopped"
    DECAYED = "decayed"
