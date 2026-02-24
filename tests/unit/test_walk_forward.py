"""Tests for WalkForwardScheduler."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from stat_arb.config.constants import WindowPhase
from stat_arb.config.settings import WalkForwardConfig
from stat_arb.engine.walk_forward import WalkForwardScheduler


def test_generate_windows_basic() -> None:
    """Should produce at least one window for a 2-year range."""
    cfg = WalkForwardConfig(formation_days=252, trading_days=63)
    scheduler = WalkForwardScheduler(cfg)

    windows = scheduler.generate_windows(date(2022, 1, 3), date(2024, 12, 31))

    assert len(windows) >= 1
    w = windows[0]
    assert w.formation_start < w.formation_end
    assert w.formation_end < w.trading_start  # execution buffer
    assert w.trading_start <= w.trading_end


def test_windows_are_non_overlapping() -> None:
    """Windows must not overlap."""
    cfg = WalkForwardConfig(formation_days=60, trading_days=20)
    scheduler = WalkForwardScheduler(cfg)

    windows = scheduler.generate_windows(date(2022, 1, 3), date(2024, 12, 31))

    assert len(windows) >= 2
    for i in range(len(windows) - 1):
        assert windows[i].trading_end < windows[i + 1].formation_start


def test_execution_buffer() -> None:
    """Trading start must be at least one bday after formation end."""
    cfg = WalkForwardConfig(formation_days=60, trading_days=20)
    scheduler = WalkForwardScheduler(cfg)

    windows = scheduler.generate_windows(date(2023, 1, 2), date(2024, 12, 31))

    for w in windows:
        delta = (w.trading_start - w.formation_end).days
        # At least 1 calendar day gap (could be 3 if formation_end is Friday)
        assert delta >= 1


def test_current_phase_formation() -> None:
    """Date inside formation window returns FORMATION."""
    cfg = WalkForwardConfig(formation_days=60, trading_days=20)
    scheduler = WalkForwardScheduler(cfg)
    windows = scheduler.generate_windows(date(2023, 1, 2), date(2024, 12, 31))

    phase = scheduler.current_phase(windows[0].formation_start)
    assert phase == WindowPhase.FORMATION


def test_current_phase_trading() -> None:
    """Date inside trading window returns TRADING."""
    cfg = WalkForwardConfig(formation_days=60, trading_days=20)
    scheduler = WalkForwardScheduler(cfg)
    windows = scheduler.generate_windows(date(2023, 1, 2), date(2024, 12, 31))

    phase = scheduler.current_phase(windows[0].trading_start)
    assert phase == WindowPhase.TRADING


def test_current_phase_outside_returns_none() -> None:
    """Date outside all windows returns None."""
    cfg = WalkForwardConfig(formation_days=60, trading_days=20)
    scheduler = WalkForwardScheduler(cfg)
    scheduler.generate_windows(date(2023, 6, 1), date(2024, 6, 1))

    phase = scheduler.current_phase(date(2020, 1, 1))
    assert phase is None


def test_run_formation_stores_active_pairs() -> None:
    """run_formation should call discovery and store results."""
    cfg = WalkForwardConfig(formation_days=60, trading_days=20)
    scheduler = WalkForwardScheduler(cfg)
    windows = scheduler.generate_windows(date(2023, 1, 2), date(2024, 12, 31))

    mock_discovery = MagicMock()
    mock_discovery.discover.return_value = ["pair1", "pair2"]
    mock_universe = MagicMock()

    result = scheduler.run_formation(windows[0], mock_discovery, mock_universe)

    assert result == ["pair1", "pair2"]
    assert scheduler.active_pairs == ["pair1", "pair2"]
    mock_discovery.discover.assert_called_once_with(
        mock_universe,
        windows[0].formation_start,
        windows[0].formation_end,
    )
