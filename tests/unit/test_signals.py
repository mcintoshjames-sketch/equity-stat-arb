"""Tests for SignalGenerator state machine."""

from __future__ import annotations

from datetime import date, timedelta

from stat_arb.config.constants import Signal
from stat_arb.config.settings import SignalConfig
from stat_arb.discovery.pair_filter import QualifiedPair
from stat_arb.engine.signals import SignalGenerator


def _make_pair(**overrides: object) -> QualifiedPair:
    defaults = dict(
        symbol_y="AAA",
        symbol_x="BBB",
        sector="tech",
        formation_start=date(2023, 1, 2),
        formation_end=date(2023, 12, 29),
        hedge_ratio=1.2,
        intercept=5.0,
        spread_mean=0.0,
        spread_std=2.0,
        half_life=10.0,
        coint_pvalue=0.01,
        adf_pvalue=0.005,
        hurst=0.35,
    )
    defaults.update(overrides)
    return QualifiedPair(**defaults)  # type: ignore[arg-type]


def test_long_spread_entry() -> None:
    """Z < -entry_z should emit LONG_SPREAD."""
    sg = SignalGenerator(SignalConfig())
    pair = _make_pair()

    event = sg.generate_signal(pair, z_score=-2.5, current_date=date(2024, 1, 2))

    assert event.signal == Signal.LONG_SPREAD
    assert event.z_score == -2.5


def test_short_spread_entry() -> None:
    """Z > +entry_z should emit SHORT_SPREAD."""
    sg = SignalGenerator(SignalConfig())
    pair = _make_pair()

    event = sg.generate_signal(pair, z_score=2.5, current_date=date(2024, 1, 2))

    assert event.signal == Signal.SHORT_SPREAD


def test_flat_when_z_in_range() -> None:
    """Z between -entry_z and +entry_z should stay FLAT."""
    sg = SignalGenerator(SignalConfig())
    pair = _make_pair()

    event = sg.generate_signal(pair, z_score=1.0, current_date=date(2024, 1, 2))

    assert event.signal == Signal.FLAT


def test_exit_on_mean_reversion() -> None:
    """After entry, Z reaching exit_z band should emit EXIT."""
    sg = SignalGenerator(SignalConfig())
    pair = _make_pair()

    # Enter long spread
    sg.generate_signal(pair, z_score=-2.5, current_date=date(2024, 1, 2))

    # Z reverts to near zero
    event = sg.generate_signal(pair, z_score=0.3, current_date=date(2024, 1, 15))

    assert event.signal == Signal.EXIT


def test_stop_on_divergence() -> None:
    """Z beyond stop_z should emit STOP."""
    sg = SignalGenerator(SignalConfig())
    pair = _make_pair()

    # Enter long spread
    sg.generate_signal(pair, z_score=-2.5, current_date=date(2024, 1, 2))

    # Z diverges beyond stop
    event = sg.generate_signal(pair, z_score=-4.5, current_date=date(2024, 1, 10))

    assert event.signal == Signal.STOP


def test_timeout_exit() -> None:
    """Position held beyond timeout_half_life_mult × half_life should exit."""
    cfg = SignalConfig(timeout_half_life_mult=3.0)
    sg = SignalGenerator(cfg)
    pair = _make_pair(half_life=10.0)  # timeout = 3 * 10 = 30 days

    entry_date = date(2024, 1, 2)
    sg.generate_signal(pair, z_score=-2.5, current_date=entry_date)

    # Day 29: still holding (z in range, no exit)
    event = sg.generate_signal(
        pair, z_score=-1.5, current_date=entry_date + timedelta(days=29),
    )
    assert event.signal == Signal.FLAT

    # Day 30: timeout triggers
    event = sg.generate_signal(
        pair, z_score=-1.5, current_date=entry_date + timedelta(days=30),
    )
    assert event.signal == Signal.EXIT


def test_reentry_after_exit() -> None:
    """After EXIT, the state machine should allow re-entry."""
    sg = SignalGenerator(SignalConfig())
    pair = _make_pair()

    # Enter
    sg.generate_signal(pair, z_score=-2.5, current_date=date(2024, 1, 2))
    # Exit
    sg.generate_signal(pair, z_score=0.3, current_date=date(2024, 1, 15))
    # Re-enter
    event = sg.generate_signal(pair, z_score=2.5, current_date=date(2024, 2, 1))

    assert event.signal == Signal.SHORT_SPREAD


def test_reset_clears_state() -> None:
    """Reset should clear position state for a pair."""
    sg = SignalGenerator(SignalConfig())
    pair = _make_pair()

    sg.generate_signal(pair, z_score=-2.5, current_date=date(2024, 1, 2))
    sg.reset(pair)

    # Should be flat again — same Z doesn't keep position
    event = sg.generate_signal(pair, z_score=-1.0, current_date=date(2024, 1, 3))
    assert event.signal == Signal.FLAT
