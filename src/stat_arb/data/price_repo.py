"""DB-cached price repository with Schwab backfill on miss."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy import select

from stat_arb.data.db import get_session
from stat_arb.data.schemas import DailyPrice

if TYPE_CHECKING:
    from stat_arb.data.schwab_client import SchwabDataClient

logger = logging.getLogger(__name__)


class PriceRepository:
    """Serves close-price DataFrames, backfilling from Schwab when the DB lacks data."""

    def __init__(self, schwab_client: SchwabDataClient | None = None) -> None:
        self._schwab = schwab_client

    def get_close_prices(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return a DataFrame of close prices indexed by date, one column per symbol.

        Missing symbols are backfilled from Schwab if a client is available.
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

        # Identify missing symbols
        found = set(pivot.columns) if not pivot.empty else set()
        missing = [s for s in symbols if s not in found]

        if missing and self._schwab is not None:
            logger.info("Backfilling %d symbols from Schwab: %s", len(missing), missing)
            for sym in missing:
                self._backfill_symbol(sym)

            # Re-query after backfill
            return self.get_close_prices(symbols, start, end)

        return pivot

    def _backfill_symbol(self, symbol: str) -> int:
        """Fetch full history from Schwab and persist to DB. Returns row count."""
        if self._schwab is None:
            return 0

        df = self._schwab.fetch_price_history(symbol, period_type="year", period=2)
        if df.empty:
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

    def upsert_prices(self, symbol: str, df: pd.DataFrame) -> int:
        """Insert or update price data from a DataFrame with DatetimeIndex and OHLCV columns."""
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
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return count
