"""SQLAlchemy 2.0 ORM models for the stat-arb system.

All domain entities are modelled here with proper foreign-key constraints,
relationships, and indexes.  The ``Base`` declarative base is imported by
``db.create_tables()`` to issue DDL.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from stat_arb.config.constants import PairStatus


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


class DailyPrice(Base):
    """OHLCV daily bar for a single equity symbol.

    Populated by :class:`PriceRepository` from Schwab API backfill or
    bulk CSV import.  The unique constraint on ``(symbol, trade_date)``
    prevents duplicate rows.
    """

    __tablename__ = "daily_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", name="uq_symbol_date"),
        Index("ix_daily_prices_symbol_date", "symbol", "trade_date"),
    )

    def __repr__(self) -> str:
        return f"<DailyPrice {self.symbol} {self.trade_date} c={self.close}>"


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------


class DiscoveredPair(Base):
    """A cointegrated pair found during a formation window.

    Stores the Engle-Granger test results and frozen spread parameters
    (hedge ratio, intercept, mean, std, half-life) that remain immutable
    throughout the subsequent trading window.
    """

    __tablename__ = "discovered_pairs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol_y: Mapped[str] = mapped_column(String(20), nullable=False)
    symbol_x: Mapped[str] = mapped_column(String(20), nullable=False)
    sector: Mapped[str] = mapped_column(String(50), nullable=False)
    formation_start: Mapped[date] = mapped_column(Date, nullable=False)
    formation_end: Mapped[date] = mapped_column(Date, nullable=False)

    # Frozen formation parameters
    hedge_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    intercept: Mapped[float] = mapped_column(Float, nullable=False)
    spread_mean: Mapped[float] = mapped_column(Float, nullable=False)
    spread_std: Mapped[float] = mapped_column(Float, nullable=False)
    half_life: Mapped[float] = mapped_column(Float, nullable=False)

    # Statistical test results
    coint_pvalue: Mapped[float] = mapped_column(Float, nullable=False)
    adf_pvalue: Mapped[float] = mapped_column(Float, nullable=False)
    hurst: Mapped[float] = mapped_column(Float, nullable=False)

    status: Mapped[str] = mapped_column(String(20), default=PairStatus.ACTIVE)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    trades: Mapped[list[Trade]] = relationship(
        back_populates="pair", lazy="selectin",
    )
    positions: Mapped[list[PairPosition]] = relationship(
        back_populates="pair", lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint(
            "symbol_y", "symbol_x", "formation_start", "formation_end",
            name="uq_pair_formation_window",
        ),
        Index("ix_pair_symbols_formation", "symbol_y", "symbol_x", "formation_start"),
    )

    def __repr__(self) -> str:
        return (
            f"<DiscoveredPair {self.symbol_y}/{self.symbol_x} "
            f"β={self.hedge_ratio:.3f} hl={self.half_life:.1f}d [{self.status}]>"
        )


# ---------------------------------------------------------------------------
# Trade execution
# ---------------------------------------------------------------------------


class Trade(Base):
    """A single-leg fill (buy or sell) belonging to a discovered pair.

    Each pair entry/exit generates two ``Trade`` rows (one per leg).
    """

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pair_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("discovered_pairs.id"), nullable=False,
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY / SELL
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    fill_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_entry: Mapped[bool] = mapped_column(default=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(50), nullable=True)

    pair: Mapped[DiscoveredPair] = relationship(back_populates="trades")

    __table_args__ = (
        Index("ix_trades_pair_id", "pair_id"),
    )

    def __repr__(self) -> str:
        return f"<Trade {self.side} {self.quantity} {self.symbol} @{self.price:.2f}>"


# ---------------------------------------------------------------------------
# Pair positions (spread-level)
# ---------------------------------------------------------------------------


class PairPosition(Base):
    """Lifecycle record for a spread position (entry → exit).

    Created when a signal opens a pair position, updated on close with
    ``exit_date``, ``exit_z``, ``pnl``, and ``exit_reason``.
    """

    __tablename__ = "pair_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pair_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("discovered_pairs.id"), nullable=False,
    )
    signal: Mapped[str] = mapped_column(String(20), nullable=False)
    entry_date: Mapped[date] = mapped_column(Date, nullable=False)
    exit_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    entry_z: Mapped[float] = mapped_column(Float, nullable=False)
    exit_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(30), nullable=True)

    pair: Mapped[DiscoveredPair] = relationship(back_populates="positions")

    __table_args__ = (
        Index("ix_pair_positions_pair_id", "pair_id"),
    )

    def __repr__(self) -> str:
        status = "open" if self.exit_date is None else f"closed pnl={self.pnl}"
        return f"<PairPosition {self.signal} z={self.entry_z:.2f} {status}>"


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------


class DailyMetrics(Base):
    """End-of-day portfolio snapshot for performance tracking.

    One row per trading day, recording portfolio value, PnL, exposure,
    and drawdown for the reporting dashboard.
    """

    __tablename__ = "daily_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)
    portfolio_value: Mapped[float] = mapped_column(Float, nullable=False)
    daily_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    gross_exposure: Mapped[float] = mapped_column(Float, nullable=False)
    active_pairs: Mapped[int] = mapped_column(Integer, nullable=False)
    drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False)

    def __repr__(self) -> str:
        return f"<DailyMetrics {self.trade_date} pnl={self.daily_pnl:+.2f}>"


# ---------------------------------------------------------------------------
# Backtest runs
# ---------------------------------------------------------------------------


class BacktestRun(Base):
    """Metadata and summary statistics for a completed backtest run.

    Stores the full config JSON so results can be reproduced and compared
    across parameter sweeps.
    """

    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    config_json: Mapped[str] = mapped_column(String, nullable=False)
    total_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_trades: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<BacktestRun {self.start_date}–{self.end_date} sharpe={self.sharpe}>"
