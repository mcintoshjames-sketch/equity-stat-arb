"""Tests for earnings blackout integration in RiskManager."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from stat_arb.config.constants import RiskDecisionType, Signal
from stat_arb.config.settings import RiskConfig
from stat_arb.discovery.pair_filter import QualifiedPair
from stat_arb.engine.signals import SignalEvent
from stat_arb.execution.sizing import SizeResult
from stat_arb.risk.earnings_blackout import EarningsBlackout
from stat_arb.risk.risk_manager import RiskManager


def _make_pair(sym_y: str = "AAA", sym_x: str = "BBB") -> QualifiedPair:
    return QualifiedPair(
        symbol_y=sym_y,
        symbol_x=sym_x,
        sector="tech",
        formation_start=date(2023, 1, 2),
        formation_end=date(2023, 12, 29),
        hedge_ratio=1.0,
        intercept=0.0,
        spread_mean=0.0,
        spread_std=2.0,
        half_life=10.0,
        coint_pvalue=0.01,
        adf_pvalue=0.005,
        hurst=0.35,
    )


def _make_entry_event(pair: QualifiedPair | None = None) -> SignalEvent:
    return SignalEvent(
        signal=Signal.LONG_SPREAD,
        pair=pair or _make_pair(),
        z_score=-2.5,
        estimated_round_trip_cost=0.0,
    )


def _make_size() -> SizeResult:
    return SizeResult(qty_y=10, qty_x=10, notional_y=1500.0, notional_x=1500.0)


def test_earnings_blackout_rejects_entry() -> None:
    """Entry should be rejected when either leg is in blackout."""
    mock_fmp = MagicMock()
    mock_fmp.get_next_earnings.return_value = {
        "AAA": date(2024, 7, 24),  # within 3 bdays of Jul 22
        "BBB": None,
    }
    blackout = EarningsBlackout(mock_fmp, blackout_days=3)
    blackout.refresh(["AAA", "BBB"], date(2024, 7, 22))

    config = RiskConfig()
    rm = RiskManager(config, earnings_blackout=blackout)

    mock_broker = MagicMock()
    mock_broker.get_gross_exposure.return_value = 0.0

    event = _make_entry_event()
    decision = rm.check(event, _make_size(), mock_broker, 0, current_date=date(2024, 7, 22))
    assert decision.decision == RiskDecisionType.REJECTED
    assert "earnings blackout" in decision.reason
    assert "AAA" in decision.reason


def test_earnings_blackout_skipped_for_exits() -> None:
    """Exits are always approved regardless of blackout."""
    mock_fmp = MagicMock()
    mock_fmp.get_next_earnings.return_value = {"AAA": date(2024, 7, 24)}
    blackout = EarningsBlackout(mock_fmp, blackout_days=3)
    blackout.refresh(["AAA"], date(2024, 7, 22))

    config = RiskConfig()
    rm = RiskManager(config, earnings_blackout=blackout)

    mock_broker = MagicMock()

    exit_event = SignalEvent(
        signal=Signal.EXIT,
        pair=_make_pair(),
        z_score=0.3,
        estimated_round_trip_cost=0.0,
    )
    decision = rm.check(
        exit_event, _make_size(), mock_broker, 1, current_date=date(2024, 7, 22),
    )
    assert decision.decision == RiskDecisionType.APPROVED


def test_no_blackout_when_not_configured() -> None:
    """Without earnings blackout, entries should pass earnings-related checks."""
    config = RiskConfig()
    rm = RiskManager(config)  # no earnings_blackout

    mock_broker = MagicMock()
    mock_broker.get_gross_exposure.return_value = 0.0

    event = _make_entry_event()
    decision = rm.check(event, _make_size(), mock_broker, 0, current_date=date(2024, 7, 22))
    assert decision.decision == RiskDecisionType.APPROVED


def test_check_earnings_blackout_method() -> None:
    """check_earnings_blackout returns True when leg is blacked out."""
    mock_fmp = MagicMock()
    mock_fmp.get_next_earnings.return_value = {
        "AAA": date(2024, 7, 24),
        "BBB": None,
    }
    blackout = EarningsBlackout(mock_fmp, blackout_days=3)
    blackout.refresh(["AAA", "BBB"], date(2024, 7, 22))

    rm = RiskManager(RiskConfig(), earnings_blackout=blackout)

    assert rm.check_earnings_blackout("AAA", "BBB", date(2024, 7, 22)) is True
    assert rm.check_earnings_blackout("BBB", "AAA", date(2024, 7, 22)) is True  # checks both
    assert rm.check_earnings_blackout("BBB", "CCC", date(2024, 7, 22)) is False


def test_check_earnings_blackout_without_config() -> None:
    """check_earnings_blackout returns False when not configured."""
    rm = RiskManager(RiskConfig())
    assert rm.check_earnings_blackout("AAA", "BBB", date(2024, 7, 22)) is False
