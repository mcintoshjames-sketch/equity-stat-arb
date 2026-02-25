"""Tests for ActivityFeed widget logic."""

from __future__ import annotations

from stat_arb.tui.widgets.activity_feed import ActivityFeed


class TestActivityFeed:
    def test_events_prepend_most_recent_first(self) -> None:
        """New events appear at the top of the list."""
        feed = ActivityFeed()
        feed._events.clear()
        feed.add_event("first", "info")
        feed.add_event("second", "info")

        texts = [t for t, _ in feed.events]
        # Most recent first
        assert "second" in texts[0]
        assert "first" in texts[1]

    def test_bounded_to_max(self) -> None:
        """Events beyond the max are discarded."""
        feed = ActivityFeed()
        feed._events.clear()
        for i in range(250):
            feed.add_event(f"event-{i}", "info")

        assert feed.event_count == 200  # _MAX_EVENTS

    def test_severity_colors_in_render(self) -> None:
        """Severity tags produce correct markup."""
        feed = ActivityFeed()
        feed._events.clear()
        feed.add_event("warn-msg", "warning")
        feed.add_event("err-msg", "error")
        feed.add_event("crit-msg", "critical")
        feed.add_event("info-msg", "info")

        rendered = feed._render_text()
        # info has no style tag
        assert "[yellow]" in rendered
        assert "[red]" in rendered
        assert "[red bold]" in rendered
