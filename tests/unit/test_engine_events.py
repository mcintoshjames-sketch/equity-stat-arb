"""Tests for LiveRunner DB event instrumentation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from stat_arb.config.constants import (
    BrokerMode,
    EngineCommandType,
    EngineEventType,
    EventSeverity,
)
from stat_arb.data.schemas import EngineCommand, EngineEvent
from stat_arb.live.runner import LiveRunner


def _make_runner(db_session) -> LiveRunner:
    """Create a LiveRunner with mocked dependencies and a real DB."""
    engine = MagicMock()
    engine.step.return_value = []
    sizer = MagicMock()
    risk_manager = MagicMock()
    risk_manager._kill_switch_active = False
    price_repo = MagicMock()
    universe = MagicMock()
    universe.symbols = []
    signal_config = MagicMock()

    with patch.object(LiveRunner, "_create_broker") as mock_broker_factory:
        mock_broker = MagicMock()
        mock_broker.get_portfolio_value.return_value = 50000.0
        mock_broker_factory.return_value = mock_broker

        runner = LiveRunner(
            engine=engine,
            sizer=sizer,
            risk_manager=risk_manager,
            price_repo=price_repo,
            universe=universe,
            schwab_client=None,
            broker_mode=BrokerMode.PAPER,
            signal_config=signal_config,
        )
    return runner


class TestPostEvent:
    def test_inserts_row(self, db_session):
        """_post_event inserts an EngineEvent row."""
        runner = _make_runner(db_session)
        runner._post_event(
            EngineEventType.ENGINE_STARTED, EventSeverity.INFO, "test message",
        )

        rows = db_session.query(EngineEvent).all()
        assert len(rows) == 1
        assert rows[0].event_type == "engine_started"
        assert rows[0].severity == "info"
        assert rows[0].message == "test message"

    def test_with_detail_json(self, db_session):
        """_post_event stores detail_json."""
        runner = _make_runner(db_session)
        runner._post_event(
            EngineEventType.STEP_COMPLETED, EventSeverity.INFO,
            "step done", '{"signals": 3}',
        )

        row = db_session.query(EngineEvent).one()
        assert row.detail_json == '{"signals": 3}'

    def test_truncates_long_message(self, db_session):
        """Messages longer than 500 chars are truncated."""
        runner = _make_runner(db_session)
        long_msg = "x" * 600
        runner._post_event(
            EngineEventType.ERROR, EventSeverity.ERROR, long_msg,
        )

        row = db_session.query(EngineEvent).one()
        assert len(row.message) == 500

    def test_never_crashes(self, db_session):
        """_post_event swallows DB errors gracefully."""
        runner = _make_runner(db_session)
        with patch("stat_arb.data.db.get_session", side_effect=RuntimeError("db down")):
            # Should not raise
            runner._post_event(
                EngineEventType.HEARTBEAT, EventSeverity.INFO, "ping",
            )


class TestPollCommands:
    def test_picks_up_kill_switch(self, db_session):
        """_poll_commands processes a kill_switch command."""
        runner = _make_runner(db_session)

        # Insert a kill switch command
        db_session.add(EngineCommand(
            command=EngineCommandType.KILL_SWITCH,
            acknowledged=False,
        ))
        db_session.commit()

        runner._poll_commands()

        assert runner._shutdown is True
        assert runner._risk._kill_switch_active is True

        # Command should be acknowledged
        cmd = db_session.query(EngineCommand).one()
        assert cmd.acknowledged is True

    def test_ignores_acknowledged_commands(self, db_session):
        """_poll_commands skips already-acknowledged commands."""
        runner = _make_runner(db_session)

        db_session.add(EngineCommand(
            command=EngineCommandType.KILL_SWITCH,
            acknowledged=True,
        ))
        db_session.commit()

        runner._poll_commands()
        assert runner._shutdown is False

    def test_never_crashes(self, db_session):
        """_poll_commands swallows DB errors gracefully."""
        runner = _make_runner(db_session)
        with patch("stat_arb.data.db.get_session", side_effect=RuntimeError("db down")):
            runner._poll_commands()
        assert runner._shutdown is False


class TestPruneOldEvents:
    def test_deletes_old_events(self, db_session):
        """Events older than the cutoff are removed."""
        runner = _make_runner(db_session)

        old = EngineEvent(
            event_type="heartbeat", severity="info", message="old",
            created_at=datetime.now(UTC) - timedelta(days=10),
        )
        recent = EngineEvent(
            event_type="heartbeat", severity="info", message="recent",
            created_at=datetime.now(UTC) - timedelta(hours=1),
        )
        db_session.add_all([old, recent])
        db_session.commit()

        runner._prune_old_events(days=7)

        # Re-query from a fresh session to see the pruned state
        from stat_arb.data.db import get_session

        session = get_session()
        rows = session.query(EngineEvent).all()
        session.close()
        assert len(rows) == 1
        assert rows[0].message == "recent"

    def test_never_crashes(self, db_session):
        """_prune_old_events swallows DB errors."""
        runner = _make_runner(db_session)
        with patch("stat_arb.data.db.get_session", side_effect=RuntimeError("db down")):
            runner._prune_old_events()
