"""Tests for rolling-mode risk manager enhancements."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from stat_arb.config.constants import RiskDecisionType, Signal
from stat_arb.config.settings import RiskConfig
from stat_arb.discovery.pair_filter import QualifiedPair
from stat_arb.engine.signals import SignalEvent
from stat_arb.execution.sizing import SizeResult
from stat_arb.risk.risk_manager import RiskManager


def _make_pair(**overrides) -> QualifiedPair:
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


def _make_event(signal: Signal, **pair_overrides) -> SignalEvent:
    pair = _make_pair(**pair_overrides)
    return SignalEvent(
        signal=signal,
        pair=pair,
        z_score=-2.5 if signal == Signal.LONG_SPREAD else 2.5,
        estimated_round_trip_cost=1.0,
    )


def _mock_broker(gross_exposure: float = 0.0) -> MagicMock:
    broker = MagicMock()
    broker.get_gross_exposure.return_value = gross_exposure
    broker.get_portfolio_value.return_value = 50_000.0
    return broker


def _size() -> SizeResult:
    return SizeResult(qty_y=10, qty_x=10, notional_y=500, notional_x=500)


class TestMaxEntriesPerStep:
    def test_rejects_after_limit(self) -> None:
        """Entries beyond max_entries_per_step should be rejected."""
        cfg = RiskConfig(max_entries_per_step=2, max_gross_exposure=100_000.0)
        rm = RiskManager(cfg)

        event = _make_event(Signal.LONG_SPREAD)
        broker = _mock_broker()

        # First two entries should pass
        rm.record_entry()
        rm.record_entry()

        # Third should be rejected
        decision = rm.check(event, _size(), broker, active_pair_count=2)
        assert decision.decision == RiskDecisionType.REJECTED
        assert "max entries per step" in decision.reason

    def test_allows_within_limit(self) -> None:
        """Entries within limit should be approved."""
        cfg = RiskConfig(max_entries_per_step=3, max_gross_exposure=100_000.0)
        rm = RiskManager(cfg)

        event = _make_event(Signal.LONG_SPREAD)
        broker = _mock_broker()

        # First entry
        decision = rm.check(event, _size(), broker, active_pair_count=0)
        assert decision.decision == RiskDecisionType.APPROVED

    def test_reset_clears_counter(self) -> None:
        """reset_step_counters should allow entries again."""
        cfg = RiskConfig(max_entries_per_step=1, max_gross_exposure=100_000.0)
        rm = RiskManager(cfg)

        event = _make_event(Signal.LONG_SPREAD)
        broker = _mock_broker()

        rm.record_entry()
        decision = rm.check(event, _size(), broker, active_pair_count=1)
        assert decision.decision == RiskDecisionType.REJECTED

        rm.reset_step_counters()
        decision = rm.check(event, _size(), broker, active_pair_count=1)
        assert decision.decision == RiskDecisionType.APPROVED


class TestPerPairPnlStop:
    def test_flagged_when_below_threshold(self) -> None:
        """Pair should be flagged when PnL is below the stop threshold."""
        cfg = RiskConfig(per_pair_pnl_stop=-200.0)
        rm = RiskManager(cfg)

        rm.update_pair_pnl(42, -250.0)
        assert rm.check_pair_pnl_stop(42) is True

    def test_not_flagged_when_above_threshold(self) -> None:
        """Pair should NOT be flagged when PnL is above the stop threshold."""
        cfg = RiskConfig(per_pair_pnl_stop=-200.0)
        rm = RiskManager(cfg)

        rm.update_pair_pnl(42, -100.0)
        assert rm.check_pair_pnl_stop(42) is False

    def test_unknown_pair_defaults_to_zero(self) -> None:
        """Unknown pair_id should default to 0.0 PnL (above any negative stop)."""
        cfg = RiskConfig(per_pair_pnl_stop=-200.0)
        rm = RiskManager(cfg)

        assert rm.check_pair_pnl_stop(999) is False

    def test_exactly_at_threshold(self) -> None:
        """PnL exactly at the threshold should trigger the stop."""
        cfg = RiskConfig(per_pair_pnl_stop=-200.0)
        rm = RiskManager(cfg)

        rm.update_pair_pnl(42, -200.0)
        assert rm.check_pair_pnl_stop(42) is True


class TestCohortConcentration:
    def test_rejects_when_cohort_at_max(self) -> None:
        """Entry should be rejected when cohort is at max concentration."""
        cfg = RiskConfig(
            max_cohort_concentration=2,
            max_gross_exposure=100_000.0,
        )
        rm = RiskManager(cfg)

        # Register 2 pairs in cohort C0001
        rm.register_pair(1, "tech", cohort_id="C0001")
        rm.register_pair(2, "tech", cohort_id="C0001")

        # Try to add a third from the same cohort
        event = _make_event(Signal.LONG_SPREAD, cohort_id="C0001")
        decision = rm.check(event, _size(), _mock_broker(), active_pair_count=2)
        assert decision.decision == RiskDecisionType.REJECTED
        assert "cohort" in decision.reason

    def test_allows_different_cohort(self) -> None:
        """Entry from a different cohort should be allowed."""
        cfg = RiskConfig(
            max_cohort_concentration=2,
            max_gross_exposure=100_000.0,
        )
        rm = RiskManager(cfg)

        rm.register_pair(1, "tech", cohort_id="C0001")
        rm.register_pair(2, "tech", cohort_id="C0001")

        # Different cohort should pass
        event = _make_event(Signal.LONG_SPREAD, cohort_id="C0002")
        decision = rm.check(event, _size(), _mock_broker(), active_pair_count=2)
        assert decision.decision == RiskDecisionType.APPROVED

    def test_no_cohort_skips_check(self) -> None:
        """Pair without cohort_id should skip the concentration check."""
        cfg = RiskConfig(
            max_cohort_concentration=1,
            max_gross_exposure=100_000.0,
        )
        rm = RiskManager(cfg)

        # No cohort_id on the pair
        event = _make_event(Signal.LONG_SPREAD)
        decision = rm.check(event, _size(), _mock_broker(), active_pair_count=0)
        assert decision.decision == RiskDecisionType.APPROVED

    def test_unregister_clears_cohort(self) -> None:
        """Unregistering a pair should clear its cohort tracking."""
        rm = RiskManager(RiskConfig())
        rm.register_pair(1, "tech", cohort_id="C0001")
        assert 1 in rm._pair_cohorts

        rm.unregister_pair(1)
        assert 1 not in rm._pair_cohorts
