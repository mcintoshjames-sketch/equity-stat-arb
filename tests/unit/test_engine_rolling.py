"""Tests for StatArbEngine rolling scheduler integration."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from stat_arb.config.constants import ExitReason, RebalanceAction, Signal
from stat_arb.config.settings import RiskConfig, SignalConfig, SizingConfig, WalkForwardConfig
from stat_arb.discovery.pair_filter import QualifiedPair, with_lifecycle
from stat_arb.engine.signals import SignalGenerator
from stat_arb.engine.spread import SpreadComputer
from stat_arb.engine.stat_arb_engine import StatArbEngine
from stat_arb.engine.walk_forward import WalkForwardScheduler
from stat_arb.risk.risk_manager import RiskManager


def _make_pair(sym_y: str = "AAA", sym_x: str = "BBB", **overrides) -> QualifiedPair:
    defaults = dict(
        symbol_y=sym_y,
        symbol_x=sym_x,
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


def _build_engine_with_rolling(
    rolling_scheduler: MagicMock,
    risk_manager: RiskManager | None = None,
) -> StatArbEngine:
    """Build an engine wired with a mock rolling scheduler."""
    signal_config = SignalConfig()
    sizing_config = SizingConfig()
    wf_config = WalkForwardConfig(formation_days=60, trading_days=20)

    spread_computer = SpreadComputer(signal_config, sizing_config)
    signal_generator = SignalGenerator(signal_config)
    walk_forward = WalkForwardScheduler(wf_config)

    mock_discovery = MagicMock()
    mock_universe = MagicMock()
    mock_rebalancer = MagicMock()

    engine = StatArbEngine(
        signal_config=signal_config,
        sizing_config=sizing_config,
        spread_computer=spread_computer,
        signal_generator=signal_generator,
        walk_forward=walk_forward,
        pair_discovery=mock_discovery,
        universe=mock_universe,
        risk_manager=risk_manager,
        rolling_scheduler=rolling_scheduler,
        rebalancer=mock_rebalancer,
    )
    return engine


def test_step_rolling_dispatches() -> None:
    """Engine should use _step_rolling when rolling_scheduler is set."""
    mock_rolling = MagicMock()
    mock_rolling.step.return_value = False
    mock_rolling.expired_keys = []
    mock_rolling.refreshed_keys = []
    mock_rolling.active_pairs = []

    engine = _build_engine_with_rolling(mock_rolling)

    events = engine.step(date(2024, 6, 3), {})
    mock_rolling.step.assert_called_once_with(date(2024, 6, 3))
    assert events == []


def test_step_walk_forward_when_no_rolling() -> None:
    """Engine should use walk-forward step when no rolling scheduler."""
    signal_config = SignalConfig()
    sizing_config = SizingConfig()
    wf_config = WalkForwardConfig(formation_days=60, trading_days=20)

    engine = StatArbEngine(
        signal_config=signal_config,
        sizing_config=sizing_config,
        spread_computer=SpreadComputer(signal_config, sizing_config),
        signal_generator=SignalGenerator(signal_config),
        walk_forward=WalkForwardScheduler(wf_config),
        pair_discovery=MagicMock(),
        universe=MagicMock(),
    )

    # No windows generated, should return empty
    events = engine.step(date(2024, 6, 3), {})
    assert events == []
    assert engine.pending_rebalance == []


def test_rolling_generates_signals() -> None:
    """Rolling mode should generate signals for active pairs."""
    pair = with_lifecycle(
        _make_pair(),
        discovery_date=date(2024, 6, 1),
        trading_expiry=date(2024, 9, 1),
        cohort_id="C0001",
    )

    mock_rolling = MagicMock()
    mock_rolling.step.return_value = False
    mock_rolling.expired_keys = []
    mock_rolling.refreshed_keys = []
    mock_rolling.active_pairs = [pair]

    engine = _build_engine_with_rolling(mock_rolling)

    quotes = {
        "AAA": {"bid": 119.0, "ask": 121.0},
        "BBB": {"bid": 99.0, "ask": 101.0},
    }

    events = engine.step(date(2024, 6, 3), quotes)
    assert len(events) == 1
    # spread = 120 - 1.0*100 - 0 = 20, z = (20-0)/5 = 4.0 -> SHORT_SPREAD (or STOP if >4)
    assert events[0].signal in (Signal.SHORT_SPREAD, Signal.STOP)


def test_rebalance_on_expiry() -> None:
    """pending_rebalance should include FORCE_EXIT for expired pairs."""
    pair = with_lifecycle(
        _make_pair(),
        discovery_date=date(2024, 6, 1),
        trading_expiry=date(2024, 9, 1),
        cohort_id="C0001",
    )

    mock_rolling = MagicMock()
    mock_rolling.step.return_value = True
    mock_rolling.expired_keys = [("AAA", "BBB")]
    mock_rolling.refreshed_keys = []
    mock_rolling.active_pairs = []
    mock_rolling.get_prev_pair.return_value = pair

    engine = _build_engine_with_rolling(mock_rolling)

    quotes = {
        "AAA": {"bid": 100.0, "ask": 100.0},
        "BBB": {"bid": 100.0, "ask": 100.0},
    }

    engine.step(date(2024, 9, 2), quotes)

    assert len(engine.pending_rebalance) == 1
    assert engine.pending_rebalance[0].action == RebalanceAction.FORCE_EXIT
    assert engine.pending_rebalance[0].pair_key == "AAA/BBB"


def test_rebalance_on_discovery_refresh() -> None:
    """pending_rebalance should include ROLLOVER for refreshed pairs."""
    pair_old = with_lifecycle(
        _make_pair(hedge_ratio=1.0),
        discovery_date=date(2024, 6, 1),
        trading_expiry=date(2024, 9, 1),
        cohort_id="C0001",
    )
    pair_new = with_lifecycle(
        _make_pair(hedge_ratio=1.5),
        discovery_date=date(2024, 7, 1),
        trading_expiry=date(2024, 10, 1),
        cohort_id="C0002",
    )

    mock_rolling = MagicMock()
    mock_rolling.step.return_value = True
    mock_rolling.expired_keys = []
    mock_rolling.refreshed_keys = [("AAA", "BBB")]
    mock_rolling.active_pairs = [pair_new]
    mock_rolling.get_prev_pair.return_value = pair_old

    engine = _build_engine_with_rolling(mock_rolling)

    quotes = {
        "AAA": {"bid": 100.0, "ask": 100.0},
        "BBB": {"bid": 100.0, "ask": 100.0},
    }

    engine.step(date(2024, 7, 1), quotes)

    rollovers = [
        r for r in engine.pending_rebalance
        if r.action == RebalanceAction.ROLLOVER
    ]
    assert len(rollovers) == 1
    assert rollovers[0].old_beta == 1.0
    assert rollovers[0].new_beta == 1.5


def test_pnl_stop_emits_signal() -> None:
    """Engine should emit STOP signal when PnL stop is breached."""
    pair = with_lifecycle(
        _make_pair(),
        discovery_date=date(2024, 6, 1),
        trading_expiry=date(2024, 9, 1),
        cohort_id="C0001",
    )

    mock_rolling = MagicMock()
    mock_rolling.step.return_value = False
    mock_rolling.expired_keys = []
    mock_rolling.refreshed_keys = []
    mock_rolling.active_pairs = [pair]

    risk_config = RiskConfig(per_pair_pnl_stop=-200.0)
    risk_manager = RiskManager(risk_config)

    # Set PnL below threshold
    pair_id = hash(("AAA", "BBB")) & 0x7FFFFFFF
    risk_manager.update_pair_pnl(pair_id, -300.0)

    engine = _build_engine_with_rolling(mock_rolling, risk_manager=risk_manager)

    quotes = {
        "AAA": {"bid": 100.0, "ask": 100.0},
        "BBB": {"bid": 100.0, "ask": 100.0},
    }

    events = engine.step(date(2024, 6, 3), quotes)

    stop_events = [e for e in events if e.signal == Signal.STOP]
    assert len(stop_events) >= 1
    assert stop_events[0].pair.symbol_y == "AAA"


def test_earnings_force_exit_emits_stop_with_reason() -> None:
    """Engine should emit STOP signal with EARNINGS_BLACKOUT reason."""
    from stat_arb.risk.earnings_blackout import EarningsBlackout

    pair = with_lifecycle(
        _make_pair(),
        discovery_date=date(2024, 6, 1),
        trading_expiry=date(2024, 9, 1),
        cohort_id="C0001",
    )

    mock_rolling = MagicMock()
    mock_rolling.step.return_value = False
    mock_rolling.expired_keys = []
    mock_rolling.refreshed_keys = []
    mock_rolling.active_pairs = [pair]

    # Set up earnings blackout: AAA earnings in 2 bdays
    mock_fmp = MagicMock()
    mock_fmp.get_next_earnings.return_value = {
        "AAA": date(2024, 6, 5),  # Wed, 2 bdays from Mon Jun 3
        "BBB": None,
    }
    blackout = EarningsBlackout(mock_fmp, blackout_days=3)
    blackout.refresh(["AAA", "BBB"], date(2024, 6, 3))

    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config, earnings_blackout=blackout)

    engine = _build_engine_with_rolling(mock_rolling, risk_manager=risk_manager)

    quotes = {
        "AAA": {"bid": 100.0, "ask": 100.0},
        "BBB": {"bid": 100.0, "ask": 100.0},
    }

    events = engine.step(date(2024, 6, 3), quotes)

    blackout_events = [
        e for e in events
        if e.signal == Signal.STOP and e.exit_reason == ExitReason.EARNINGS_BLACKOUT
    ]
    assert len(blackout_events) == 1
    assert blackout_events[0].pair.symbol_y == "AAA"
