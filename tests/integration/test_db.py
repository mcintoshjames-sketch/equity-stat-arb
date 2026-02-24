"""Integration tests for database, ORM models, and price repository."""

from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd

from stat_arb.config.constants import PairStatus
from stat_arb.data.db import get_session
from stat_arb.data.price_repo import PriceRepository
from stat_arb.data.schemas import (
    DailyPrice,
    DiscoveredPair,
    PairPosition,
    Trade,
)


class TestDatabase:
    def test_create_tables(self, db_engine):
        """Tables should be created without error."""
        from sqlalchemy import inspect

        inspector = inspect(db_engine)
        tables = inspector.get_table_names()
        assert "daily_prices" in tables
        assert "discovered_pairs" in tables
        assert "trades" in tables
        assert "pair_positions" in tables
        assert "daily_metrics" in tables
        assert "backtest_runs" in tables

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


class TestDiscoveredPairSchema:
    def test_insert_discovered_pair(self, db_session):
        pair = DiscoveredPair(
            symbol_y="AAPL",
            symbol_x="MSFT",
            sector="technology",
            formation_start=date(2023, 1, 1),
            formation_end=date(2023, 12, 31),
            hedge_ratio=1.15,
            intercept=5.0,
            spread_mean=0.0,
            spread_std=1.2,
            half_life=12.5,
            coint_pvalue=0.01,
            adf_pvalue=0.02,
            hurst=0.35,
        )
        db_session.add(pair)
        db_session.commit()

        result = db_session.query(DiscoveredPair).first()
        assert result is not None
        assert result.symbol_y == "AAPL"
        assert result.symbol_x == "MSFT"
        assert result.hedge_ratio == 1.15
        assert result.status == PairStatus.ACTIVE

    def test_repr(self, db_session):
        pair = DiscoveredPair(
            symbol_y="AAPL", symbol_x="MSFT", sector="tech",
            formation_start=date(2023, 1, 1), formation_end=date(2023, 12, 31),
            hedge_ratio=1.150, intercept=5.0, spread_mean=0.0, spread_std=1.2,
            half_life=12.5, coint_pvalue=0.01, adf_pvalue=0.02, hurst=0.35,
        )
        r = repr(pair)
        assert "AAPL/MSFT" in r
        assert "1.150" in r


class TestForeignKeys:
    def test_trade_references_pair(self, db_session):
        """Trade.pair_id FK should link to DiscoveredPair."""
        pair = DiscoveredPair(
            symbol_y="AAPL", symbol_x="MSFT", sector="technology",
            formation_start=date(2023, 1, 1), formation_end=date(2023, 12, 31),
            hedge_ratio=1.15, intercept=5.0, spread_mean=0.0, spread_std=1.2,
            half_life=12.5, coint_pvalue=0.01, adf_pvalue=0.02, hurst=0.35,
        )
        db_session.add(pair)
        db_session.flush()  # get pair.id

        trade = Trade(
            pair_id=pair.id,
            symbol="AAPL",
            side="BUY",
            quantity=10,
            price=150.0,
            fill_time=datetime(2024, 1, 15, 10, 30),
        )
        db_session.add(trade)
        db_session.commit()

        # Verify FK relationship
        result = db_session.query(Trade).first()
        assert result is not None
        assert result.pair.symbol_y == "AAPL"

    def test_pair_position_references_pair(self, db_session):
        """PairPosition.pair_id FK should link to DiscoveredPair."""
        pair = DiscoveredPair(
            symbol_y="GS", symbol_x="JPM", sector="financials",
            formation_start=date(2023, 6, 1), formation_end=date(2024, 5, 31),
            hedge_ratio=0.95, intercept=2.0, spread_mean=0.0, spread_std=0.8,
            half_life=8.0, coint_pvalue=0.03, adf_pvalue=0.04, hurst=0.42,
        )
        db_session.add(pair)
        db_session.flush()

        position = PairPosition(
            pair_id=pair.id,
            signal="long_spread",
            entry_date=date(2024, 6, 5),
            entry_z=-2.3,
        )
        db_session.add(position)
        db_session.commit()

        result = db_session.query(PairPosition).first()
        assert result is not None
        assert result.pair.symbol_y == "GS"
        assert result.exit_date is None  # still open

    def test_relationship_back_populates(self, db_session):
        """DiscoveredPair should have .trades and .positions collections."""
        pair = DiscoveredPair(
            symbol_y="NVDA", symbol_x="AMD", sector="technology",
            formation_start=date(2023, 1, 1), formation_end=date(2023, 12, 31),
            hedge_ratio=1.5, intercept=0.0, spread_mean=0.0, spread_std=2.0,
            half_life=15.0, coint_pvalue=0.02, adf_pvalue=0.01, hurst=0.38,
        )
        db_session.add(pair)
        db_session.flush()

        db_session.add(Trade(
            pair_id=pair.id, symbol="NVDA", side="BUY",
            quantity=5, price=500.0, fill_time=datetime(2024, 1, 10, 9, 30),
        ))
        db_session.add(PairPosition(
            pair_id=pair.id, signal="long_spread",
            entry_date=date(2024, 1, 10), entry_z=-2.1,
        ))
        db_session.commit()

        # Refresh to load relationships
        db_session.refresh(pair)
        assert len(pair.trades) == 1
        assert len(pair.positions) == 1
        assert pair.trades[0].symbol == "NVDA"


class TestPriceRepository:
    def test_get_close_prices_from_db(self, db_engine):
        session = get_session()
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

        result = repo.get_close_prices(
            ["UPSERT_TEST"],
            start=date(2024, 6, 1),
            end=date(2024, 6, 30),
        )
        assert len(result) == 3

    def test_get_date_range(self, db_engine):
        session = get_session()
        for d in pd.bdate_range("2024-03-01", periods=10):
            session.add(DailyPrice(
                symbol="RANGE_TEST",
                trade_date=d.date(),
                open=100.0, high=101.0, low=99.0, close=100.0, volume=1000,
            ))
        session.commit()
        session.close()

        repo = PriceRepository(schwab_client=None)
        result = repo.get_date_range("RANGE_TEST")
        assert result is not None
        min_date, max_date = result
        assert min_date == date(2024, 3, 1)
        assert max_date == date(2024, 3, 14)

    def test_get_date_range_missing_symbol(self, db_engine):
        repo = PriceRepository(schwab_client=None)
        assert repo.get_date_range("NONEXISTENT") is None

    def test_backfill_from_schwab(self, db_engine):
        """PriceRepository should backfill missing symbols from Schwab."""
        dates = pd.bdate_range("2024-01-02", periods=5)
        mock_df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [105.0] * 5,
                "low": [99.0] * 5,
                "close": [103.0, 104.0, 105.0, 106.0, 107.0],
                "volume": [50000] * 5,
            },
            index=dates,
        )

        mock_schwab = MagicMock()
        mock_schwab.fetch_price_history.return_value = mock_df

        repo = PriceRepository(schwab_client=mock_schwab)
        df = repo.get_close_prices(
            ["BACKFILL_SYM"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
        )

        # Schwab should have been called for the missing symbol
        mock_schwab.fetch_price_history.assert_called_once_with(
            "BACKFILL_SYM", period_type="year", period=2,
        )
        assert not df.empty
        assert "BACKFILL_SYM" in df.columns
        assert len(df) == 5
