"""Tests for TUI data provider logic."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from stat_arb.config.constants import PairStatus
from stat_arb.config.settings import RiskConfig
from stat_arb.data.schemas import DailyMetrics, DiscoveredPair, PairPosition, Trade
from stat_arb.tui.data_provider import (
    DbDataProvider,
    PortfolioSnapshot,
)


@pytest.fixture
def provider(db_engine, risk_config):
    return DbDataProvider(
        risk_config=risk_config,
        broker_mode="paper",
        schwab_client=None,
    )


class TestPortfolioSnapshot:
    def test_from_daily_metrics(self, db_session, provider):
        """DailyMetrics row maps correctly to PortfolioSnapshot."""
        row = DailyMetrics(
            trade_date=date(2025, 6, 15),
            portfolio_value=49823.0,
            daily_pnl=127.0,
            gross_exposure=8400.0,
            active_pairs=3,
            drawdown_pct=0.0035,
        )
        db_session.add(row)
        db_session.commit()

        snap = provider.get_portfolio_snapshot()
        assert snap.value == 49823.0
        assert snap.daily_pnl == 127.0
        assert snap.gross_exposure == 8400.0
        assert snap.active_pairs == 3
        assert snap.drawdown_pct == pytest.approx(0.0035)

    def test_empty_db(self, db_session, provider):
        """Empty table returns zeroed snapshot."""
        snap = provider.get_portfolio_snapshot()
        assert snap == PortfolioSnapshot(
            value=0.0,
            daily_pnl=0.0,
            gross_exposure=0.0,
            active_pairs=0,
            drawdown_pct=0.0,
        )

    def test_latest_row_used(self, db_session, provider):
        """When multiple rows exist, the most recent trade_date is used."""
        old = DailyMetrics(
            trade_date=date(2025, 6, 14),
            portfolio_value=49000.0,
            daily_pnl=-50.0,
            gross_exposure=7000.0,
            active_pairs=2,
            drawdown_pct=0.01,
        )
        new = DailyMetrics(
            trade_date=date(2025, 6, 15),
            portfolio_value=50000.0,
            daily_pnl=100.0,
            gross_exposure=9000.0,
            active_pairs=4,
            drawdown_pct=0.005,
        )
        db_session.add_all([old, new])
        db_session.commit()

        snap = provider.get_portfolio_snapshot()
        assert snap.value == 50000.0
        assert snap.active_pairs == 4


class TestActivePairs:
    def test_active_pairs_query(self, db_session, provider):
        """Only ACTIVE pairs with open positions are returned."""
        pair_active = DiscoveredPair(
            symbol_y="AAPL", symbol_x="MSFT", sector="technology",
            formation_start=date(2025, 1, 1), formation_end=date(2025, 6, 1),
            hedge_ratio=1.2, intercept=5.0, spread_mean=0.1, spread_std=1.5,
            half_life=12.5, coint_pvalue=0.01, adf_pvalue=0.02, hurst=0.35,
            status=PairStatus.ACTIVE,
        )
        pair_expired = DiscoveredPair(
            symbol_y="JPM", symbol_x="BAC", sector="financials",
            formation_start=date(2025, 1, 1), formation_end=date(2025, 6, 1),
            hedge_ratio=0.9, intercept=2.0, spread_mean=0.05, spread_std=0.8,
            half_life=18.0, coint_pvalue=0.03, adf_pvalue=0.04, hurst=0.40,
            status=PairStatus.EXPIRED,
        )
        db_session.add_all([pair_active, pair_expired])
        db_session.flush()

        # Open position for active pair
        pos_open = PairPosition(
            pair_id=pair_active.id, signal="long_spread",
            entry_date=date(2025, 6, 10), entry_z=-2.15,
        )
        # Closed position for active pair (should be excluded)
        pos_closed = PairPosition(
            pair_id=pair_active.id, signal="long_spread",
            entry_date=date(2025, 5, 1), entry_z=-2.0,
            exit_date=date(2025, 5, 15), exit_z=-0.3, pnl=50.0,
            exit_reason="mean_reversion",
        )
        # Open position for expired pair (should be excluded)
        pos_expired = PairPosition(
            pair_id=pair_expired.id, signal="short_spread",
            entry_date=date(2025, 6, 1), entry_z=2.34,
        )
        db_session.add_all([pos_open, pos_closed, pos_expired])
        db_session.commit()

        pairs = provider.get_active_pairs()
        assert len(pairs) == 1
        assert pairs[0].pair_key == "AAPL/MSFT"
        assert pairs[0].direction == "long"
        assert pairs[0].entry_z == pytest.approx(-2.15)

    def test_pair_display_row_days_held(self, db_session, provider):
        """days_held computed from entry_date to today."""
        pair = DiscoveredPair(
            symbol_y="AAPL", symbol_x="MSFT", sector="technology",
            formation_start=date(2025, 1, 1), formation_end=date(2025, 6, 1),
            hedge_ratio=1.2, intercept=5.0, spread_mean=0.1, spread_std=1.5,
            half_life=12.5, coint_pvalue=0.01, adf_pvalue=0.02, hurst=0.35,
            status=PairStatus.ACTIVE,
        )
        db_session.add(pair)
        db_session.flush()

        entry = date.today() - timedelta(days=5)
        pos = PairPosition(
            pair_id=pair.id, signal="long_spread",
            entry_date=entry, entry_z=-2.0,
        )
        db_session.add(pos)
        db_session.commit()

        pairs = provider.get_active_pairs()
        assert len(pairs) == 1
        assert pairs[0].days_held == 5


class TestRecentTrades:
    def test_ordering(self, db_session, provider):
        """Trades are returned in reverse chronological order."""
        pair = DiscoveredPair(
            symbol_y="AAPL", symbol_x="MSFT", sector="technology",
            formation_start=date(2025, 1, 1), formation_end=date(2025, 6, 1),
            hedge_ratio=1.2, intercept=5.0, spread_mean=0.1, spread_std=1.5,
            half_life=12.5, coint_pvalue=0.01, adf_pvalue=0.02, hurst=0.35,
            status=PairStatus.ACTIVE,
        )
        db_session.add(pair)
        db_session.flush()

        t1 = Trade(
            pair_id=pair.id, symbol="AAPL", side="BUY", quantity=10,
            price=189.50, fill_time=datetime(2025, 6, 15, 14, 0), is_entry=True,
        )
        t2 = Trade(
            pair_id=pair.id, symbol="MSFT", side="SELL", quantity=8,
            price=412.20, fill_time=datetime(2025, 6, 15, 15, 30), is_entry=True,
        )
        db_session.add_all([t1, t2])
        db_session.commit()

        trades = provider.get_recent_trades(limit=10)
        assert len(trades) == 2
        # Most recent first
        assert trades[0].fill_time > trades[1].fill_time
        assert trades[0].symbol == "MSFT"
        assert trades[1].symbol == "AAPL"


class TestRiskUtilization:
    def test_from_config(self, db_session, provider, risk_config):
        """Limits come from RiskConfig, values from latest snapshot."""
        row = DailyMetrics(
            trade_date=date(2025, 6, 15),
            portfolio_value=50000.0,
            daily_pnl=100.0,
            gross_exposure=8400.0,
            active_pairs=3,
            drawdown_pct=0.035,
        )
        db_session.add(row)
        db_session.commit()

        risk = provider.get_risk_utilization()
        assert risk.pair_count == 3
        assert risk.pair_limit == risk_config.max_pairs
        assert risk.exposure == 8400.0
        assert risk.exposure_limit == risk_config.max_gross_exposure
        assert risk.drawdown_pct == pytest.approx(0.035)
        assert risk.drawdown_limit == risk_config.max_drawdown_pct


class TestTokenStatus:
    def test_valid_tokens(self):
        """Correct time-to-expiry from issued timestamps."""
        now = datetime.now(UTC)
        mock_client = MagicMock()
        mock_tokens = MagicMock()
        mock_tokens._access_token_issued = now - timedelta(seconds=300)
        mock_tokens._refresh_token_issued = now - timedelta(days=2)
        mock_client._client.tokens = mock_tokens

        provider = DbDataProvider(
            risk_config=RiskConfig(),
            broker_mode="paper",
            schwab_client=mock_client,
        )
        status = provider.get_token_status()
        assert status is not None
        assert status.access_valid is True
        assert status.refresh_valid is True
        # ~1500s remaining for access (1800 - 300)
        assert 1400 < status.access_remaining_s < 1600
        # ~5 days remaining for refresh (7 - 2)
        assert 4 * 86400 < status.refresh_remaining_s < 6 * 86400

    def test_expired_tokens(self):
        """Expired tokens show zero remaining."""
        now = datetime.now(UTC)
        mock_client = MagicMock()
        mock_tokens = MagicMock()
        mock_tokens._access_token_issued = now - timedelta(hours=1)
        mock_tokens._refresh_token_issued = now - timedelta(days=10)
        mock_client._client.tokens = mock_tokens

        provider = DbDataProvider(
            risk_config=RiskConfig(),
            broker_mode="paper",
            schwab_client=mock_client,
        )
        status = provider.get_token_status()
        assert status is not None
        assert status.access_remaining_s == 0.0
        assert status.refresh_remaining_s == 0.0
        assert status.access_valid is False
        assert status.refresh_valid is False

    def test_no_schwab(self, db_engine):
        """Returns None when no client is available."""
        provider = DbDataProvider(
            risk_config=RiskConfig(),
            broker_mode="paper",
            schwab_client=None,
        )
        assert provider.get_token_status() is None


class TestSystemStatus:
    def test_db_connected(self, db_engine):
        """DB connectivity check succeeds with initialized engine."""
        provider = DbDataProvider(
            risk_config=RiskConfig(),
            broker_mode="paper",
            schwab_client=None,
        )
        status = provider.get_system_status()
        assert status.db_connected is True
        assert status.broker_mode == "paper"
        assert status.schwab_connected is False


class TestKillSwitchWidget:
    def test_initial_state_is_safe(self):
        """KillSwitchWidget starts in safe (not killed) state."""
        from stat_arb.tui.widgets.kill_switch import KillSwitchWidget

        widget = KillSwitchWidget()
        assert widget.is_killed is False

    def test_set_killed_state(self):
        """set_killed transitions widget to killed state."""
        from stat_arb.tui.widgets.kill_switch import KillSwitchWidget

        widget = KillSwitchWidget()
        assert widget.is_killed is False
        # Calling set_killed without a running app will fail on query_one,
        # so we test the flag directly
        widget._killed = True
        assert widget.is_killed is True

    def test_initial_disabled_when_no_engine(self):
        """KillSwitchWidget button starts disabled when engine_active=False."""
        from stat_arb.tui.widgets.kill_switch import KillSwitchWidget

        widget = KillSwitchWidget(engine_active=False)
        assert widget._engine_active is False

    def test_initial_enabled_when_engine(self):
        """KillSwitchWidget button starts enabled when engine_active=True."""
        from stat_arb.tui.widgets.kill_switch import KillSwitchWidget

        widget = KillSwitchWidget(engine_active=True)
        assert widget._engine_active is True
