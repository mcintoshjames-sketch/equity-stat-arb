"""Signal generation state machine for pairs trading.

Tracks per-pair position state and emits entry, exit, and stop signals
based on Z-score thresholds, time-based timeout, and divergence stops.
Logs estimated round-trip slippage with every signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from stat_arb.config.constants import ExitReason, PositionDirection, Signal

if TYPE_CHECKING:
    from stat_arb.config.settings import SignalConfig
    from stat_arb.discovery.pair_filter import QualifiedPair

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalEvent:
    """Immutable record of a signal emission.

    Attributes:
        signal: The signal type (entry, exit, stop, flat).
        pair: The qualified pair this signal pertains to.
        z_score: Z-score at the time the signal was generated.
        estimated_round_trip_cost: Dollar slippage estimate for the trade.
    """

    signal: Signal
    pair: QualifiedPair
    z_score: float
    estimated_round_trip_cost: float
    exit_reason: ExitReason | None = None


@dataclass
class _PairState:
    """Mutable per-pair position tracking."""

    direction: PositionDirection = PositionDirection.FLAT
    entry_date: date | None = None
    entry_z: float | None = None


class SignalGenerator:
    """Z-score-based signal state machine.

    Tracks the position direction for each pair and applies entry, exit,
    timeout, and stop rules.  Every emitted signal logs the estimated
    round-trip transaction cost so edge erosion is visible.

    Args:
        config: Signal configuration with Z-score thresholds and slippage.
    """

    def __init__(self, config: SignalConfig) -> None:
        self._config = config
        self._states: dict[tuple[str, str], _PairState] = {}

    def _get_state(self, pair: QualifiedPair) -> _PairState:
        key = (pair.symbol_y, pair.symbol_x)
        if key not in self._states:
            self._states[key] = _PairState()
        return self._states[key]

    def reset(self, pair: QualifiedPair) -> None:
        """Reset position state for a pair (e.g. at window rollover)."""
        key = (pair.symbol_y, pair.symbol_x)
        self._states.pop(key, None)

    def generate_signal(
        self,
        pair: QualifiedPair,
        z_score: float,
        current_date: date,
        estimated_round_trip_cost: float = 0.0,
    ) -> SignalEvent:
        """Evaluate the current Z-score and emit a signal.

        State transitions:

        - **FLAT → LONG_SPREAD**: ``z < -entry_z``
        - **FLAT → SHORT_SPREAD**: ``z > +entry_z``
        - **LONG/SHORT → EXIT**: ``|z| <= exit_z`` (mean reversion)
        - **LONG/SHORT → EXIT**: timeout (days held > mult × half_life)
        - **LONG/SHORT → STOP**: ``|z| > stop_z`` (divergence)

        Args:
            pair: Qualified pair with frozen formation parameters.
            z_score: Current Z-score from :class:`SpreadComputer`.
            current_date: Today's date for timeout calculation.
            estimated_round_trip_cost: Dollar slippage estimate.

        Returns:
            A ``SignalEvent`` with the resulting signal.
        """
        state = self._get_state(pair)
        cfg = self._config
        signal = Signal.FLAT

        if state.direction == PositionDirection.FLAT:
            signal = self._check_entry(z_score, cfg)
            if signal in (Signal.LONG_SPREAD, Signal.SHORT_SPREAD):
                direction = (
                    PositionDirection.LONG
                    if signal == Signal.LONG_SPREAD
                    else PositionDirection.SHORT
                )
                state.direction = direction
                state.entry_date = current_date
                state.entry_z = z_score
                self._log_signal(
                    pair, signal, z_score, estimated_round_trip_cost,
                )
                return SignalEvent(
                    signal=signal,
                    pair=pair,
                    z_score=z_score,
                    estimated_round_trip_cost=estimated_round_trip_cost,
                )
            return SignalEvent(
                signal=Signal.FLAT,
                pair=pair,
                z_score=z_score,
                estimated_round_trip_cost=0.0,
            )

        # Currently in a position — check exit / stop / timeout
        signal = self._check_exit(
            z_score, current_date, state, pair.half_life, cfg,
        )
        if signal in (Signal.EXIT, Signal.STOP):
            self._log_signal(
                pair, signal, z_score, estimated_round_trip_cost,
            )
            state.direction = PositionDirection.FLAT
            state.entry_date = None
            state.entry_z = None
            return SignalEvent(
                signal=signal,
                pair=pair,
                z_score=z_score,
                estimated_round_trip_cost=estimated_round_trip_cost,
            )

        # Hold — no signal change
        return SignalEvent(
            signal=Signal.FLAT,
            pair=pair,
            z_score=z_score,
            estimated_round_trip_cost=0.0,
        )

    @staticmethod
    def _check_entry(z_score: float, cfg: SignalConfig) -> Signal:
        if z_score < -cfg.entry_z:
            return Signal.LONG_SPREAD
        if z_score > cfg.entry_z:
            return Signal.SHORT_SPREAD
        return Signal.FLAT

    @staticmethod
    def _check_exit(
        z_score: float,
        current_date: date,
        state: _PairState,
        half_life: float,
        cfg: SignalConfig,
    ) -> Signal:
        # Hard stop: divergence beyond stop threshold
        if abs(z_score) > cfg.stop_z:
            return Signal.STOP

        # Profit target / mean reversion: spread reverted to near zero
        if abs(z_score) <= cfg.exit_z:
            return Signal.EXIT

        # Time-based timeout
        if state.entry_date is not None:
            days_held = (current_date - state.entry_date).days
            timeout_days = cfg.timeout_half_life_mult * half_life
            if days_held >= timeout_days:
                return Signal.EXIT

        return Signal.FLAT

    @staticmethod
    def _log_signal(
        pair: QualifiedPair,
        signal: Signal,
        z_score: float,
        cost: float,
    ) -> None:
        logger.info(
            "SIGNAL %s %s/%s z=%.3f hl=%.1fd est_rt_cost=$%.2f",
            signal.value, pair.symbol_y, pair.symbol_x,
            z_score, pair.half_life, cost,
        )
