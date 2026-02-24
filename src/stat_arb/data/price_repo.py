"""DB-cached price repository with Schwab backfill on cache miss.

:class:`PriceRepository` is the single entry-point for historical close
prices throughout the system.  It queries the local database first and
transparently backfills missing symbols from the Schwab API when a
:class:`SchwabDataClient` is provided.

Bulk upserts use dialect-aware ``INSERT … ON CONFLICT DO UPDATE`` for
both SQLite and PostgreSQL, avoiding slow row-by-row ORM merges.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy import func, select

from stat_arb.data.db import get_engine, get_session
from stat_arb.data.schemas import DailyPrice

if TYPE_CHECKING:
    from stat_arb.data.schwab_client import SchwabDataClient

logger = logging.getLogger(__name__)

# Columns updated on conflict (everything except PK, symbol, trade_date)
_UPSERT_SET = {"open", "high", "low", "close", "volume"}


def _dialect_insert():
    """Return the dialect-specific ``insert`` function for the active engine.

    SQLite and PostgreSQL both support ``ON CONFLICT DO UPDATE`` but
    require different SQLAlchemy dialect imports.
    """
    engine = get_engine()
    dialect_name = engine.dialect.name

    if dialect_name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif dialect_name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        raise RuntimeError(f"Unsupported dialect for bulk upsert: {dialect_name}")

    return insert


class PriceRepository:
    """Serves close-price DataFrames, backfilling from Schwab when the DB lacks data.

    Args:
        schwab_client: Optional API client for automatic backfill.
            Pass ``None`` for offline / test usage.
    """

    def __init__(self, schwab_client: SchwabDataClient | None = None) -> None:
        self._schwab = schwab_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_close_prices(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return a pivot DataFrame of close prices indexed by date.

        Args:
            symbols: Ticker symbols to retrieve.
            start: First trade date (inclusive).
            end: Last trade date (inclusive).

        Returns:
            DataFrame with ``DatetimeIndex`` and one column per symbol.
            Missing symbols with no Schwab client return empty columns.
        """
        session = get_session()
        try:
            stmt = (
                select(DailyPrice.symbol, DailyPrice.trade_date, DailyPrice.close)
                .where(
                    DailyPrice.symbol.in_(symbols),
                    DailyPrice.trade_date >= start,
                    DailyPrice.trade_date <= end,
                )
                .order_by(DailyPrice.trade_date)
            )
            rows = session.execute(stmt).all()
        finally:
            session.close()

        if rows:
            df = pd.DataFrame(rows, columns=["symbol", "date", "close"])
            pivot = df.pivot(index="date", columns="symbol", values="close")
        else:
            pivot = pd.DataFrame()

        # Identify missing symbols and backfill
        found = set(pivot.columns) if not pivot.empty else set()
        missing = [s for s in symbols if s not in found]

        if missing and self._schwab is not None:
            logger.info("Backfilling %d symbols from Schwab: %s", len(missing), missing)
            for sym in missing:
                self._backfill_symbol(sym)

            # Re-query after backfill
            return self.get_close_prices(symbols, start, end)

        return pivot

    def get_date_range(self, symbol: str) -> tuple[date, date] | None:
        """Return the ``(min_date, max_date)`` available for a symbol, or ``None``.

        Useful for walk-forward window scheduling to determine data coverage.
        """
        session = get_session()
        try:
            stmt = select(
                func.min(DailyPrice.trade_date),
                func.max(DailyPrice.trade_date),
            ).where(DailyPrice.symbol == symbol)
            row = session.execute(stmt).one()
        finally:
            session.close()

        if row[0] is None:
            return None
        return (row[0], row[1])

    def upsert_prices(self, symbol: str, df: pd.DataFrame) -> int:
        """Bulk insert-or-update price data from a DataFrame.

        Uses dialect-aware ``INSERT … ON CONFLICT DO UPDATE`` for
        performance (single statement per batch, no row-by-row ORM round-trips).

        Args:
            symbol: Ticker symbol for all rows.
            df: DataFrame with ``DatetimeIndex`` and OHLCV columns.

        Returns:
            Number of rows processed.
        """
        if df.empty:
            return 0

        rows = _df_to_row_dicts(symbol, df)
        count = _bulk_upsert(rows)
        logger.info("Upserted %d price rows for %s", count, symbol)
        return count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _backfill_symbol(self, symbol: str) -> int:
        """Fetch 2-year history from Schwab and bulk-insert into the DB.

        Uses ``INSERT … ON CONFLICT DO UPDATE`` to handle duplicates
        gracefully in a single bulk statement.

        Returns:
            Number of rows persisted.
        """
        if self._schwab is None:
            return 0

        df = self._schwab.fetch_price_history(symbol, period_type="year", period=2)
        if df.empty:
            logger.warning("Schwab returned no data for %s", symbol)
            return 0

        rows = _df_to_row_dicts(symbol, df)
        count = _bulk_upsert(rows)
        logger.info("Backfilled %d rows for %s", count, symbol)
        return count


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _df_to_row_dicts(symbol: str, df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame with DatetimeIndex + OHLCV columns to row dicts."""
    rows: list[dict] = []
    for dt, row in df.iterrows():
        trade_date = dt.date() if hasattr(dt, "date") else dt
        rows.append({
            "symbol": symbol,
            "trade_date": trade_date,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
        })
    return rows


def _bulk_upsert(rows: list[dict]) -> int:
    """Execute a bulk INSERT … ON CONFLICT DO UPDATE for DailyPrice rows.

    On conflict with the ``(symbol, trade_date)`` unique constraint,
    updates OHLCV columns to the new values.

    Returns:
        Number of rows in the batch.
    """
    if not rows:
        return 0

    insert = _dialect_insert()
    stmt = insert(DailyPrice).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "trade_date"],
        set_={col: stmt.excluded[col] for col in _UPSERT_SET},
    )

    session = get_session()
    try:
        session.execute(stmt)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return len(rows)
