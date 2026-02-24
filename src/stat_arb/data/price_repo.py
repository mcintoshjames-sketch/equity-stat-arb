"""DB-cached price repository with Schwab backfill on cache miss.

:class:`PriceRepository` is the single entry-point for historical close
prices throughout the system.  It queries the local database first and
transparently backfills missing symbols from the Schwab API when a
:class:`SchwabDataClient` is provided.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy import func, select

from stat_arb.data.db import get_session
from stat_arb.data.schemas import DailyPrice

if TYPE_CHECKING:
    from stat_arb.data.schwab_client import SchwabDataClient

logger = logging.getLogger(__name__)


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
        """Insert or update price data from a DataFrame.

        Args:
            symbol: Ticker symbol for all rows.
            df: DataFrame with ``DatetimeIndex`` and OHLCV columns.

        Returns:
            Number of rows processed.
        """
        if df.empty:
            return 0

        session = get_session()
        count = 0
        try:
            for dt, row in df.iterrows():
                trade_date = dt.date() if hasattr(dt, "date") else dt
                existing = session.execute(
                    select(DailyPrice).where(
                        DailyPrice.symbol == symbol,
                        DailyPrice.trade_date == trade_date,
                    )
                ).scalar_one_or_none()

                if existing:
                    existing.open = float(row["open"])
                    existing.high = float(row["high"])
                    existing.low = float(row["low"])
                    existing.close = float(row["close"])
                    existing.volume = int(row["volume"])
                else:
                    session.add(DailyPrice(
                        symbol=symbol,
                        trade_date=trade_date,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"]),
                    ))
                count += 1
            session.commit()
            logger.info("Upserted %d price rows for %s", count, symbol)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _backfill_symbol(self, symbol: str) -> int:
        """Fetch 2-year history from Schwab and bulk-insert into the DB.

        Uses ``session.merge`` to handle duplicates gracefully (insert-or-update).

        Returns:
            Number of rows persisted.
        """
        if self._schwab is None:
            return 0

        df = self._schwab.fetch_price_history(symbol, period_type="year", period=2)
        if df.empty:
            logger.warning("Schwab returned no data for %s", symbol)
            return 0

        session = get_session()
        count = 0
        try:
            for dt, row in df.iterrows():
                price = DailyPrice(
                    symbol=symbol,
                    trade_date=dt.date(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                )
                session.merge(price)
                count += 1
            session.commit()
            logger.info("Backfilled %d rows for %s", count, symbol)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return count
