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


class RiskDecisionType(StrEnum):
    """Outcome of a risk check on a proposed trade."""

    APPROVED = "approved"
    REJECTED = "rejected"


class ExitReason(StrEnum):
    """Reason a pair position was closed."""

    MEAN_REVERSION = "mean_reversion"
    STOP_LOSS = "stop_loss"
    TIMEOUT = "timeout"
    STRUCTURAL_BREAK = "structural_break"
    KILL_SWITCH = "kill_switch"
    EARNINGS_BLACKOUT = "earnings_blackout"


class AlertSeverity(StrEnum):
    """Severity level for monitoring alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RebalanceAction(StrEnum):
    """Outcome of inventory reconciliation at a window transition."""

    ROLLOVER = "rollover"        # marginal delta rebalance
    FORCE_EXIT = "force_exit"    # pair dropped, full liquidation
    NO_CHANGE = "no_change"      # inventory already at target


class EngineEventType(StrEnum):
    """Types of events written by the engine to the DB."""

    HEARTBEAT = "heartbeat"
    STATE_CHANGED = "state_changed"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    SIGNAL = "signal"
    ORDER = "order"
    ERROR = "error"
    KILL_SWITCH = "kill_switch"
    ENGINE_STARTED = "engine_started"
    ENGINE_STOPPED = "engine_stopped"


class EventSeverity(StrEnum):
    """Severity level for engine event rows."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EngineCommandType(StrEnum):
    """Commands sent from the TUI to the engine via DB."""

    KILL_SWITCH = "kill_switch"
