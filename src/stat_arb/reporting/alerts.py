"""Alert system for risk and model monitoring.

Checks drawdown, exposure, sector concentration, structural breaks,
and model decay, emitting structured alerts with severity levels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from stat_arb.config.constants import AlertSeverity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alert:
    """Immutable alert record.

    Attributes:
        severity: INFO, WARNING, or CRITICAL.
        category: Alert category (e.g. ``"drawdown"``, ``"exposure"``).
        message: Human-readable description.
        timestamp: When the alert was created.
        metadata: Optional key-value data for structured logging.
    """

    severity: AlertSeverity
    category: str
    message: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Risk and model alert monitor.

    Args:
        max_drawdown_pct: Drawdown threshold for alerts.
        max_gross_exposure: Gross exposure threshold for alerts.
        max_sector_pct: Sector concentration threshold for alerts.
    """

    def __init__(
        self,
        max_drawdown_pct: float,
        max_gross_exposure: float,
        max_sector_pct: float,
    ) -> None:
        self._max_drawdown_pct = max_drawdown_pct
        self._max_gross_exposure = max_gross_exposure
        self._max_sector_pct = max_sector_pct
        self._emitted: list[Alert] = []

    def check_drawdown(self, current_drawdown: float) -> Alert | None:
        """Check drawdown level and emit alert if thresholds breached.

        Args:
            current_drawdown: Current drawdown as a positive fraction.

        Returns:
            Alert if threshold breached, None otherwise.
        """
        if current_drawdown >= self._max_drawdown_pct:
            alert = Alert(
                severity=AlertSeverity.CRITICAL,
                category="drawdown",
                message=(
                    f"Drawdown {current_drawdown:.2%} breached limit "
                    f"{self._max_drawdown_pct:.2%}"
                ),
                timestamp=datetime.now(UTC),
                metadata={"drawdown": current_drawdown},
            )
            self._emit(alert)
            return alert

        if current_drawdown >= self._max_drawdown_pct * 0.75:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                category="drawdown",
                message=(
                    f"Drawdown {current_drawdown:.2%} approaching limit "
                    f"{self._max_drawdown_pct:.2%}"
                ),
                timestamp=datetime.now(UTC),
                metadata={"drawdown": current_drawdown},
            )
            self._emit(alert)
            return alert

        return None

    def check_exposure(self, gross_exposure: float) -> Alert | None:
        """Check gross exposure against limit.

        Args:
            gross_exposure: Current gross notional exposure.

        Returns:
            Alert if threshold breached, None otherwise.
        """
        if gross_exposure > self._max_gross_exposure:
            alert = Alert(
                severity=AlertSeverity.CRITICAL,
                category="exposure",
                message=(
                    f"Gross exposure ${gross_exposure:,.0f} exceeds limit "
                    f"${self._max_gross_exposure:,.0f}"
                ),
                timestamp=datetime.now(UTC),
                metadata={"gross_exposure": gross_exposure},
            )
            self._emit(alert)
            return alert
        return None

    def check_sector(
        self,
        sector: str,
        sector_pct: float,
    ) -> Alert | None:
        """Check sector concentration against limit.

        Args:
            sector: Sector name.
            sector_pct: Sector's fraction of gross exposure.

        Returns:
            Alert if threshold breached, None otherwise.
        """
        if sector_pct > self._max_sector_pct:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                category="sector",
                message=(
                    f"Sector '{sector}' at {sector_pct:.1%} exceeds limit "
                    f"{self._max_sector_pct:.1%}"
                ),
                timestamp=datetime.now(UTC),
                metadata={"sector": sector, "sector_pct": sector_pct},
            )
            self._emit(alert)
            return alert
        return None

    def check_structural_break(
        self,
        pair_key: str,
    ) -> Alert:
        """Emit a structural break alert.

        Args:
            pair_key: Identifier for the pair that broke.

        Returns:
            The emitted alert.
        """
        alert = Alert(
            severity=AlertSeverity.WARNING,
            category="structural_break",
            message=f"Structural break detected for {pair_key}",
            timestamp=datetime.now(UTC),
            metadata={"pair_key": pair_key},
        )
        self._emit(alert)
        return alert

    def check_model_decay(
        self,
        fallback_rate: float,
        half_life_trend: float,
    ) -> Alert | None:
        """Check model decay indicators.

        Args:
            fallback_rate: Fraction of Kalman fallbacks.
            half_life_trend: Half-life trend (positive = decay).

        Returns:
            Alert if decay detected, None otherwise.
        """
        if fallback_rate > 0.5:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                category="model_decay",
                message=(
                    f"Kalman fallback rate {fallback_rate:.0%} — "
                    "model parameters may be degrading"
                ),
                timestamp=datetime.now(UTC),
                metadata={
                    "fallback_rate": fallback_rate,
                    "half_life_trend": half_life_trend,
                },
            )
            self._emit(alert)
            return alert

        if half_life_trend > 5.0:
            alert = Alert(
                severity=AlertSeverity.INFO,
                category="model_decay",
                message=(
                    f"Half-life trend +{half_life_trend:.1f}d — "
                    "reversion slowing"
                ),
                timestamp=datetime.now(UTC),
                metadata={
                    "fallback_rate": fallback_rate,
                    "half_life_trend": half_life_trend,
                },
            )
            self._emit(alert)
            return alert

        return None

    def get_alerts(self) -> list[Alert]:
        """Return all emitted alerts."""
        return list(self._emitted)

    def clear(self) -> None:
        """Clear all emitted alerts."""
        self._emitted.clear()

    def _emit(self, alert: Alert) -> None:
        """Store and log an alert."""
        self._emitted.append(alert)
        log_fn = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.critical,
        }[alert.severity]
        log_fn("ALERT [%s] %s: %s", alert.severity, alert.category, alert.message)
