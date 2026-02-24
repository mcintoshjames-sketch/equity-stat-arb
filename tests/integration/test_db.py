"""Integration tests for database and price repository."""

from datetime import date

import pandas as pd

from stat_arb.data.db import get_session
from stat_arb.data.price_repo import PriceRepository
from stat_arb.data.schemas import DailyPrice


class TestDatabase:
    def test_create_tables(self, db_engine):
        """Tables should be created without error."""
        from sqlalchemy import inspect
        inspector = inspect(db_engine)
        tables = inspector.get_table_names()
        assert "daily_prices" in tables
        assert "discovered_pairs" in tables
        assert "trades" in tables

    def test_insert_daily_price(self, db_session):
        price = DailyPrice(
            symbol="AAPL",
            trade_date=date(2024, 1, 2),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1_000_000,
        )
        db_session.add(price)
        db_session.commit()

        result = db_session.query(DailyPrice).filter_by(symbol="AAPL").first()
        assert result is not None
        assert result.close == 153.0


class TestPriceRepository:
    def test_get_close_prices_from_db(self, db_engine):
        session = get_session()
        # Insert test data
        for i, d in enumerate(pd.bdate_range("2024-01-02", periods=5)):
            session.add(DailyPrice(
                symbol="TEST",
                trade_date=d.date(),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=100_000,
            ))
        session.commit()
        session.close()

        repo = PriceRepository(schwab_client=None)
        df = repo.get_close_prices(
            ["TEST"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
        )
        assert not df.empty
        assert "TEST" in df.columns
        assert len(df) == 5

    def test_upsert_prices(self, db_engine):
        dates = pd.bdate_range("2024-06-01", periods=3)
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [103.0, 104.0, 105.0],
                "volume": [50000, 60000, 70000],
            },
            index=dates,
        )

        repo = PriceRepository(schwab_client=None)
        count = repo.upsert_prices("UPSERT_TEST", df)
        assert count == 3

        # Verify data
        result = repo.get_close_prices(
            ["UPSERT_TEST"],
            start=date(2024, 6, 1),
            end=date(2024, 6, 30),
        )
        assert len(result) == 3
