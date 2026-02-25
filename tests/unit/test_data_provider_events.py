"""Tests for DbDataProvider engine event methods."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stat_arb.config.constants import EngineCommandType, EngineEventType
from stat_arb.data.schemas import EngineCommand, EngineEvent
from stat_arb.tui.data_provider import DbDataProvider


@pytest.fixture
def provider(db_engine, risk_config):
    return DbDataProvider(
        risk_config=risk_config,
        broker_mode="paper",
        schwab_client=None,
    )


class TestGetRecentEvents:
    def test_initial_load_desc(self, db_session, provider):
        """Without since_id, returns newest events in desc order."""
        for i in range(5):
            db_session.add(EngineEvent(
                event_type="step_completed", severity="info",
                message=f"event-{i}",
            ))
        db_session.commit()

        events = provider.get_recent_events(since_id=None, limit=3)
        assert len(events) == 3
        # Newest first (highest IDs)
        assert events[0].id > events[1].id > events[2].id

    def test_incremental_load_asc(self, db_session, provider):
        """With since_id, returns events with id > since_id in asc order."""
        for i in range(5):
            db_session.add(EngineEvent(
                event_type="step_completed", severity="info",
                message=f"event-{i}",
            ))
        db_session.commit()

        # Get initial load to find max id
        all_events = provider.get_recent_events(since_id=None, limit=50)
        min_id = min(e.id for e in all_events)
        mid_id = min_id + 2  # skip first 3

        # Incremental: get events after mid_id
        events = provider.get_recent_events(since_id=mid_id, limit=50)
        assert all(e.id > mid_id for e in events)
        # Should be in ascending order
        ids = [e.id for e in events]
        assert ids == sorted(ids)

    def test_empty_db(self, db_session, provider):
        """Returns empty list when no events exist."""
        events = provider.get_recent_events(since_id=None, limit=50)
        assert events == []

    def test_since_id_no_new(self, db_session, provider):
        """Returns empty list when no events after since_id."""
        db_session.add(EngineEvent(
            event_type="heartbeat", severity="info", message="ping",
        ))
        db_session.commit()

        all_events = provider.get_recent_events(since_id=None, limit=50)
        max_id = max(e.id for e in all_events)

        events = provider.get_recent_events(since_id=max_id, limit=50)
        assert events == []


class TestGetEngineStatus:
    def test_alive_with_recent_heartbeat(self, db_session, provider):
        """Engine is alive when heartbeat is within 90s."""
        db_session.add(EngineEvent(
            event_type=EngineEventType.HEARTBEAT, severity="info",
            message="heartbeat",
            created_at=datetime.now(UTC) - timedelta(seconds=30),
        ))
        db_session.add(EngineEvent(
            event_type=EngineEventType.STATE_CHANGED, severity="info",
            message="idle",
        ))
        db_session.commit()

        status = provider.get_engine_status()
        assert status.is_alive is True
        assert status.state == "idle"
        assert status.last_heartbeat is not None

    def test_dead_with_stale_heartbeat(self, db_session, provider):
        """Engine is not alive when heartbeat is older than 90s."""
        db_session.add(EngineEvent(
            event_type=EngineEventType.HEARTBEAT, severity="info",
            message="heartbeat",
            created_at=datetime.now(UTC) - timedelta(seconds=120),
        ))
        db_session.commit()

        status = provider.get_engine_status()
        assert status.is_alive is False

    def test_unknown_with_no_events(self, db_session, provider):
        """When no events exist, state is 'unknown' and not alive."""
        status = provider.get_engine_status()
        assert status.is_alive is False
        assert status.state == "unknown"
        assert status.last_heartbeat is None
        assert status.last_event is None


class TestSendKillSwitch:
    def test_inserts_command(self, db_session, provider):
        """send_kill_switch inserts an unacknowledged kill_switch command."""
        provider.send_kill_switch()

        cmd = db_session.query(EngineCommand).one()
        assert cmd.command == EngineCommandType.KILL_SWITCH
        assert cmd.acknowledged is False
