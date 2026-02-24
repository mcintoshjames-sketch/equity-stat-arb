"""Tests for StatArbEngine orchestrator."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from stat_arb.config.constants import Signal
from stat_arb.config.settings import SignalConfig, SizingConfig, WalkForwardConfig
from stat_arb.discovery.pair_filter import QualifiedPair
from stat_arb.engine.signals import SignalGenerator
from stat_arb.engine.spread import SpreadComputer
from stat_arb.engine.stat_arb_engine import StatArbEngine
from stat_arb.engine.walk_forward import WalkForwardScheduler


def _make_pair(**overrides: object) -> QualifiedPair:
    defaults = dict(
        symbol_y="AAA",
        symbol_x="BBB",
        sector="tech",
        formation_start=date(2023, 1, 2),
        formation_end=date(2023, 12, 29),
        hedge_ratio=1.0,
        intercept=0.0,
        spread_mean=0.0,
        spread_std=5.0,
        half_life=10.0,
        coint_pvalue=0.01,
        adf_pvalue=0.005,
        hurst=0.35,
    )
    defaults.update(overrides)
    return QualifiedPair(**defaults)  # type: ignore[arg-type]


def _build_engine() -> tuple[StatArbEngine, WalkForwardScheduler]:
    """Build an engine wired with mocks for testing."""
    signal_config = SignalConfig()
    sizing_config = SizingConfig()
    wf_config = WalkForwardConfig(formation_days=60, trading_days=20)

    spread_computer = SpreadComputer(signal_config, sizing_config)
    signal_generator = SignalGenerator(signal_config)
    walk_forward = WalkForwardScheduler(wf_config)

    mock_discovery = MagicMock()
    mock_universe = MagicMock()

    engine = StatArbEngine(
        signal_config=signal_config,
        sizing_config=sizing_config,
        spread_computer=spread_computer,
        signal_generator=signal_generator,
        walk_forward=walk_forward,
        pair_discovery=mock_discovery,
        universe=mock_universe,
    )
    return engine, walk_forward


def test_step_outside_window_returns_empty() -> None:
    """Date outside all windows should return no events."""
    engine, wf = _build_engine()
    wf.generate_windows(date(2024, 1, 2), date(2024, 12, 31))

    events = engine.step(date(2020, 1, 1), {})
    assert events == []


def test_step_during_trading_generates_signals() -> None:
    """During trading window with active pairs, signals should be generated."""
    engine, wf = _build_engine()
    windows = wf.generate_windows(date(2023, 1, 2), date(2024, 12, 31))

    # Manually set active pairs (bypassing formation)
    pair = _make_pair()
    wf._active_pairs = [pair]

    trading_date = windows[0].trading_start
    quotes = {
        "AAA": {"bid": 119.0, "ask": 121.0},  # mid = 120
        "BBB": {"bid": 99.0, "ask": 101.0},   # mid = 100
    }

    events = engine.step(trading_date, quotes)

    assert len(events) == 1
    # spread = 120 - 1.0*100 - 0 = 20, z = (20-0)/5 = 4.0 → SHORT_SPREAD
    assert events[0].signal == Signal.SHORT_SPREAD
    assert events[0].z_score > 2.0


def test_step_during_formation_returns_empty() -> None:
    """Formation dates (non-end) should return no events."""
    engine, wf = _build_engine()
    windows = wf.generate_windows(date(2023, 1, 2), date(2024, 12, 31))

    # Step on first day of formation — not formation_end, so no discovery
    events = engine.step(windows[0].formation_start, {})
    assert events == []


def test_mid_prices_from_quotes() -> None:
    """_mid_prices should compute (bid+ask)/2."""
    mids = StatArbEngine._mid_prices({
        "X": {"bid": 99.0, "ask": 101.0},
        "Y": {"bid": 49.5, "ask": 50.5},
    })
    assert abs(mids["X"] - 100.0) < 1e-10
    assert abs(mids["Y"] - 50.0) < 1e-10


def test_mid_prices_fallback_to_last() -> None:
    """If no bid/ask, fall back to 'last' price."""
    mids = StatArbEngine._mid_prices({
        "X": {"last": 100.0},
    })
    assert abs(mids["X"] - 100.0) < 1e-10
