"""Data provider layer for the TUI dashboard.

Defines frozen display dataclasses and a protocol for data access.
``DbDataProvider`` reads from the SQLAlchemy DB and optionally the
Schwab client for token status.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sqlalchemy import asc, desc, select

from stat_arb.config.constants import EngineCommandType, EngineEventType, PairStatus
from stat_arb.data.db import get_session
from stat_arb.data.schemas import (
    DailyMetrics,
    DiscoveredPair,
    EngineCommand,
    EngineEvent,
    PairPosition,
    Trade,
)

if TYPE_CHECKING:
    from stat_arb.config.settings import RiskConfig
    from stat_arb.data.schwab_client import SchwabDataClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display dataclasses (frozen)
# ---------------------------------------------------------------------------

ACCESS_TOKEN_LIFETIME = 1800  # 30 minutes
REFRESH_TOKEN_LIFETIME = 604800  # 7 days

# Engine is considered alive if heartbeat is within this window
_LIVENESS_THRESHOLD_S = 90


@dataclass(frozen=True)
class PortfolioSnapshot:
    """End-of-day portfolio summary for dashboard display."""

    value: float
    daily_pnl: float
    gross_exposure: float
    active_pairs: int
    drawdown_pct: float


@dataclass(frozen=True)
class PairDisplayRow:
    """One row in the active-positions table."""

    pair_key: str
    sector: str
    direction: str
    z_score: float | None
    hedge_ratio: float
    half_life: float
    entry_date: date | None
    entry_z: float | None
    days_held: int | None
    spread_mean: float
    spread_std: float
    coint_pvalue: float
    adf_pvalue: float
    hurst: float
    intercept: float


@dataclass(frozen=True)
class TradeDisplayRow:
    """One row in the recent-trades log."""

    fill_time: datetime
    pair_key: str
    side: str
    symbol: str
    quantity: int
    price: float
    is_entry: bool


@dataclass(frozen=True)
class RiskUtilization:
    """Current risk utilisation vs configured limits."""

    pair_count: int
    pair_limit: int
    exposure: float
    exposure_limit: float
    drawdown_pct: float
    drawdown_limit: float
    kill_switch: bool
    sector_exposures: dict[str, float] = field(default_factory=dict)
    sector_limit: float = 0.30


@dataclass(frozen=True)
class TokenStatus:
    """Schwab OAuth token status."""

    access_issued: datetime | None
    refresh_issued: datetime | None
    access_remaining_s: float
    refresh_remaining_s: float
    access_valid: bool
    refresh_valid: bool


@dataclass(frozen=True)
class SystemStatus:
    """System connectivity and mode info."""

    broker_mode: str
    db_connected: bool
    schwab_connected: bool
    last_step_time: datetime | None
    earnings_blackout_enabled: bool = False


@dataclass(frozen=True)
class EngineEventRow:
    """Single engine event for activity feed display."""

    id: int
    event_type: str
    severity: str
    message: str
    detail_json: str | None
    created_at: datetime


@dataclass(frozen=True)
class EngineStatus:
    """Derived engine liveness and state from DB events."""

    is_alive: bool
    state: str
    last_heartbeat: datetime | None
    last_event: datetime | None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DashboardDataProvider(Protocol):
    """Protocol for dashboard data access."""

    def get_portfolio_snapshot(self) -> PortfolioSnapshot: ...

    def get_active_pairs(self) -> list[PairDisplayRow]: ...

    def get_recent_trades(self, limit: int = 20) -> list[TradeDisplayRow]: ...

    def get_risk_utilization(self) -> RiskUtilization: ...

    def get_token_status(self) -> TokenStatus | None: ...

    def get_system_status(self) -> SystemStatus: ...

    def get_recent_events(
        self, since_id: int | None = None, limit: int = 50,
    ) -> list[EngineEventRow]: ...

    def get_engine_status(self) -> EngineStatus: ...

    def send_kill_switch(self) -> None: ...


# ---------------------------------------------------------------------------
# DB-backed implementation
# ---------------------------------------------------------------------------


class DbDataProvider:
    """Reads dashboard data from the database and optional Schwab client.

    Args:
        risk_config: Risk limits for utilisation display.
        broker_mode: Current broker mode string.
        schwab_client: Optional Schwab client for token status.
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        broker_mode: str,
        schwab_client: SchwabDataClient | None = None,
        earnings_blackout_enabled: bool = False,
    ) -> None:
        self._risk_config = risk_config
        self._broker_mode = broker_mode
        self._schwab_client = schwab_client
        self._earnings_blackout_enabled = earnings_blackout_enabled

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Latest DailyMetrics row → PortfolioSnapshot."""
        session = get_session()
        try:
            row = session.execute(
                select(DailyMetrics).order_by(desc(DailyMetrics.trade_date)).limit(1)
            ).scalar_one_or_none()

            if row is None:
                return PortfolioSnapshot(
                    value=0.0,
                    daily_pnl=0.0,
                    gross_exposure=0.0,
                    active_pairs=0,
                    drawdown_pct=0.0,
                )

            return PortfolioSnapshot(
                value=row.portfolio_value,
                daily_pnl=row.daily_pnl,
                gross_exposure=row.gross_exposure,
                active_pairs=row.active_pairs,
                drawdown_pct=row.drawdown_pct,
            )
        finally:
            session.close()

    def get_active_pairs(self) -> list[PairDisplayRow]:
        """ACTIVE pairs with open positions → list of PairDisplayRow."""
        session = get_session()
        try:
            stmt = (
                select(DiscoveredPair, PairPosition)
                .join(PairPosition, DiscoveredPair.id == PairPosition.pair_id)
                .where(
                    DiscoveredPair.status == PairStatus.ACTIVE,
                    PairPosition.exit_date.is_(None),
                )
            )
            rows = session.execute(stmt).all()

            result: list[PairDisplayRow] = []
            today = date.today()
            for pair, pos in rows:
                days_held = (today - pos.entry_date).days if pos.entry_date else None
                direction = "long" if pos.signal == "long_spread" else "short"
                result.append(
                    PairDisplayRow(
                        pair_key=f"{pair.symbol_y}/{pair.symbol_x}",
                        sector=pair.sector,
                        direction=direction,
                        z_score=None,  # Would need live spread calc
                        hedge_ratio=pair.hedge_ratio,
                        half_life=pair.half_life,
                        entry_date=pos.entry_date,
                        entry_z=pos.entry_z,
                        days_held=days_held,
                        spread_mean=pair.spread_mean,
                        spread_std=pair.spread_std,
                        coint_pvalue=pair.coint_pvalue,
                        adf_pvalue=pair.adf_pvalue,
                        hurst=pair.hurst,
                        intercept=pair.intercept,
                    )
                )
            return result
        finally:
            session.close()

    def get_recent_trades(self, limit: int = 20) -> list[TradeDisplayRow]:
        """Recent trades ordered by fill_time DESC."""
        session = get_session()
        try:
            stmt = (
                select(Trade, DiscoveredPair)
                .join(DiscoveredPair, Trade.pair_id == DiscoveredPair.id)
                .order_by(desc(Trade.fill_time))
                .limit(limit)
            )
            rows = session.execute(stmt).all()

            return [
                TradeDisplayRow(
                    fill_time=trade.fill_time,
                    pair_key=f"{pair.symbol_y}/{pair.symbol_x}",
                    side=trade.side,
                    symbol=trade.symbol,
                    quantity=trade.quantity,
                    price=trade.price,
                    is_entry=trade.is_entry,
                )
                for trade, pair in rows
            ]
        finally:
            session.close()

    def get_risk_utilization(self) -> RiskUtilization:
        """Derive risk utilisation from latest DailyMetrics + RiskConfig."""
        snapshot = self.get_portfolio_snapshot()
        return RiskUtilization(
            pair_count=snapshot.active_pairs,
            pair_limit=self._risk_config.max_pairs,
            exposure=snapshot.gross_exposure,
            exposure_limit=self._risk_config.max_gross_exposure,
            drawdown_pct=snapshot.drawdown_pct,
            drawdown_limit=self._risk_config.max_drawdown_pct,
            kill_switch=False,  # Would need RiskManager state
            sector_limit=self._risk_config.max_sector_pct,
        )

    def get_token_status(self) -> TokenStatus | None:
        """Read Schwab token timestamps if client is available."""
        if self._schwab_client is None:
            return None

        try:
            tokens = self._schwab_client._client.tokens
            now = datetime.now(UTC)

            access_issued = getattr(tokens, "_access_token_issued", None)
            refresh_issued = getattr(tokens, "_refresh_token_issued", None)

            access_remaining = 0.0
            if access_issued is not None:
                elapsed = (now - access_issued).total_seconds()
                access_remaining = max(0.0, ACCESS_TOKEN_LIFETIME - elapsed)

            refresh_remaining = 0.0
            if refresh_issued is not None:
                elapsed = (now - refresh_issued).total_seconds()
                refresh_remaining = max(0.0, REFRESH_TOKEN_LIFETIME - elapsed)

            return TokenStatus(
                access_issued=access_issued,
                refresh_issued=refresh_issued,
                access_remaining_s=access_remaining,
                refresh_remaining_s=refresh_remaining,
                access_valid=access_remaining > 0,
                refresh_valid=refresh_remaining > 0,
            )
        except Exception:
            logger.exception("Failed to read token status")
            return None

    def get_system_status(self) -> SystemStatus:
        """Check system connectivity and mode."""
        db_ok = False
        last_step: datetime | None = None
        try:
            session = get_session()
            session.execute(select(DailyMetrics.id).limit(1))
            db_ok = True
            # Get most recent metrics date as proxy for last step
            row = session.execute(
                select(DailyMetrics.trade_date)
                .order_by(desc(DailyMetrics.trade_date))
                .limit(1)
            ).scalar_one_or_none()
            if row is not None:
                last_step = datetime.combine(row, datetime.min.time())
            session.close()
        except Exception:
            logger.debug("DB connectivity check failed", exc_info=True)

        schwab_ok = self._schwab_client is not None

        return SystemStatus(
            broker_mode=self._broker_mode,
            db_connected=db_ok,
            schwab_connected=schwab_ok,
            last_step_time=last_step,
            earnings_blackout_enabled=self._earnings_blackout_enabled,
        )

    # ------------------------------------------------------------------
    # Engine event methods (new — DB-based engine communication)
    # ------------------------------------------------------------------

    def get_recent_events(
        self, since_id: int | None = None, limit: int = 50,
    ) -> list[EngineEventRow]:
        """Fetch recent engine events from the DB.

        Args:
            since_id: If provided, returns events with id > since_id (ascending).
                      If None, returns the newest *limit* events (descending).
            limit: Maximum number of events to return.
        """
        session = get_session()
        try:
            if since_id is not None:
                stmt = (
                    select(EngineEvent)
                    .where(EngineEvent.id > since_id)
                    .order_by(asc(EngineEvent.id))
                    .limit(limit)
                )
            else:
                stmt = (
                    select(EngineEvent)
                    .order_by(desc(EngineEvent.id))
                    .limit(limit)
                )
            rows = session.execute(stmt).scalars().all()

            return [
                EngineEventRow(
                    id=row.id,
                    event_type=row.event_type,
                    severity=row.severity,
                    message=row.message,
                    detail_json=row.detail_json,
                    created_at=row.created_at,
                )
                for row in rows
            ]
        finally:
            session.close()

    def get_engine_status(self) -> EngineStatus:
        """Derive engine liveness from heartbeat and state_changed events."""
        session = get_session()
        try:
            # Last heartbeat
            hb_row = session.execute(
                select(EngineEvent)
                .where(EngineEvent.event_type == EngineEventType.HEARTBEAT)
                .order_by(desc(EngineEvent.id))
                .limit(1)
            ).scalar_one_or_none()

            last_heartbeat = hb_row.created_at if hb_row else None

            # Last state_changed
            state_row = session.execute(
                select(EngineEvent)
                .where(EngineEvent.event_type == EngineEventType.STATE_CHANGED)
                .order_by(desc(EngineEvent.id))
                .limit(1)
            ).scalar_one_or_none()

            state = state_row.message if state_row else "unknown"

            # Last event of any type
            last_row = session.execute(
                select(EngineEvent)
                .order_by(desc(EngineEvent.id))
                .limit(1)
            ).scalar_one_or_none()

            last_event = last_row.created_at if last_row else None

            # Liveness: heartbeat within threshold
            is_alive = False
            if last_heartbeat is not None:
                # Ensure timezone-aware comparison
                now = datetime.now(UTC)
                hb = last_heartbeat
                if hb.tzinfo is None:
                    hb = hb.replace(tzinfo=UTC)
                elapsed = (now - hb).total_seconds()
                is_alive = elapsed < _LIVENESS_THRESHOLD_S

            return EngineStatus(
                is_alive=is_alive,
                state=state,
                last_heartbeat=last_heartbeat,
                last_event=last_event,
            )
        finally:
            session.close()

    def send_kill_switch(self) -> None:
        """Insert a kill_switch command for the engine to pick up."""
        session = get_session()
        try:
            session.add(EngineCommand(
                command=EngineCommandType.KILL_SWITCH,
            ))
            session.commit()
        finally:
            session.close()
