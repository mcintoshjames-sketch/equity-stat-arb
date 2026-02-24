"""Tests for the reporting layer: metrics, alert manager."""

from __future__ import annotations

import pytest

from stat_arb.config.constants import AlertSeverity
from stat_arb.reporting.alerts import AlertManager
from stat_arb.reporting.metrics import (
    cagr,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_positive(self) -> None:
        """Positive returns → positive Sharpe."""
        returns = [0.01, 0.02, 0.01, 0.03, 0.01] * 20
        assert sharpe_ratio(returns) > 0

    def test_zero_std(self) -> None:
        """Identical returns → zero std → Sharpe = 0."""
        returns = [0.01] * 50
        assert sharpe_ratio(returns) == 0.0

    def test_empty(self) -> None:
        """Empty returns → 0."""
        assert sharpe_ratio([]) == 0.0

    def test_single_observation(self) -> None:
        """Single observation → 0."""
        assert sharpe_ratio([0.05]) == 0.0


class TestSortino:
    def test_all_positive(self) -> None:
        """All positive returns → zero downside → Sortino = 0."""
        returns = [0.01, 0.02, 0.03]
        assert sortino_ratio(returns) == 0.0

    def test_mixed_returns(self) -> None:
        """Mixed returns → finite Sortino."""
        returns = [0.02, -0.01, 0.03, -0.02, 0.01] * 20
        s = sortino_ratio(returns)
        assert s != 0.0


class TestMaxDrawdown:
    def test_monotonic_increasing(self) -> None:
        """Monotonically increasing equity → 0 drawdown."""
        curve = [100.0, 110.0, 120.0, 130.0]
        assert max_drawdown(curve) == 0.0

    def test_known_curve(self) -> None:
        """Known drawdown from peak-valley."""
        curve = [100.0, 200.0, 150.0, 180.0]
        # Peak=200, trough=150, dd=50/200=0.25
        assert max_drawdown(curve) == pytest.approx(0.25)

    def test_empty(self) -> None:
        """Empty curve → 0."""
        assert max_drawdown([]) == 0.0


class TestCAGR:
    def test_known_growth(self) -> None:
        """100 → 200 over 252 days = 100% CAGR."""
        curve = [100.0] + [200.0] * 252
        # 253 data points, 252 trading days
        result = cagr(curve, days=252)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_empty(self) -> None:
        """Empty curve → 0."""
        assert cagr([]) == 0.0


class TestWinRate:
    def test_basic(self) -> None:
        """3 out of 4 positive → 75%."""
        assert win_rate([10.0, 20.0, -5.0, 30.0]) == pytest.approx(0.75)

    def test_empty(self) -> None:
        """Empty → 0."""
        assert win_rate([]) == 0.0


class TestProfitFactor:
    def test_profitable(self) -> None:
        """Gross profit > gross loss → factor > 1."""
        pf = profit_factor([100.0, -50.0, 200.0, -25.0])
        # profit=300, loss=75 → 4.0
        assert pf == pytest.approx(4.0)

    def test_no_losses(self) -> None:
        """No losses → inf."""
        assert profit_factor([100.0, 200.0]) == float("inf")

    def test_no_profits(self) -> None:
        """No profits → 0."""
        assert profit_factor([-10.0, -20.0]) == 0.0

    def test_empty(self) -> None:
        """Empty → 0."""
        assert profit_factor([]) == 0.0


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class TestAlertManager:
    def test_drawdown_critical(self) -> None:
        """Drawdown at or above limit → CRITICAL."""
        am = AlertManager(
            max_drawdown_pct=0.10,
            max_gross_exposure=100_000.0,
            max_sector_pct=0.30,
        )
        alert = am.check_drawdown(0.10)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert "breached" in alert.message

    def test_drawdown_warning(self) -> None:
        """Drawdown approaching limit (>=75%) → WARNING."""
        am = AlertManager(
            max_drawdown_pct=0.10,
            max_gross_exposure=100_000.0,
            max_sector_pct=0.30,
        )
        alert = am.check_drawdown(0.08)  # 80% of limit
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING

    def test_drawdown_no_alert(self) -> None:
        """Low drawdown → no alert."""
        am = AlertManager(
            max_drawdown_pct=0.10,
            max_gross_exposure=100_000.0,
            max_sector_pct=0.30,
        )
        alert = am.check_drawdown(0.02)
        assert alert is None

    def test_exposure_critical(self) -> None:
        """Gross exposure above limit → CRITICAL."""
        am = AlertManager(
            max_drawdown_pct=0.10,
            max_gross_exposure=50_000.0,
            max_sector_pct=0.30,
        )
        alert = am.check_exposure(60_000.0)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL

    def test_structural_break_alert(self) -> None:
        """Structural break emits WARNING."""
        am = AlertManager(
            max_drawdown_pct=0.10,
            max_gross_exposure=100_000.0,
            max_sector_pct=0.30,
        )
        alert = am.check_structural_break("AAA/BBB")
        assert alert.severity == AlertSeverity.WARNING
        assert "AAA/BBB" in alert.message

    def test_get_alerts_and_clear(self) -> None:
        """Alerts accumulate and can be cleared."""
        am = AlertManager(
            max_drawdown_pct=0.10,
            max_gross_exposure=100_000.0,
            max_sector_pct=0.30,
        )
        am.check_drawdown(0.12)
        am.check_structural_break("X/Y")
        assert len(am.get_alerts()) == 2
        am.clear()
        assert len(am.get_alerts()) == 0

    def test_model_decay_high_fallback(self) -> None:
        """High Kalman fallback rate → WARNING."""
        am = AlertManager(
            max_drawdown_pct=0.10,
            max_gross_exposure=100_000.0,
            max_sector_pct=0.30,
        )
        alert = am.check_model_decay(fallback_rate=0.7, half_life_trend=0.0)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
