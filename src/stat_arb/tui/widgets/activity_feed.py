"""Real-time activity feed widget for the TUI dashboard.

Displays a bounded log of engine events, most recent first, with
severity-based colour coding.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime

from textual.widgets import Static

_MAX_EVENTS = 200
_DISPLAY_LIMIT = 50

_SEVERITY_MARKUP = {
    "info": "",
    "warning": "yellow",
    "error": "red",
    "critical": "red bold",
}


class ActivityFeed(Static):
    """Scrollable activity log with severity colours."""

    DEFAULT_CSS = """
    ActivityFeed {
        height: auto;
        padding: 1 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._events: deque[tuple[str, str]] = deque(maxlen=_MAX_EVENTS)

    def on_mount(self) -> None:
        self._refresh_display()

    def add_event(self, text: str, severity: str = "info") -> None:
        """Prepend a timestamped event to the feed.

        Args:
            text: Event description.
            severity: One of info, warning, error, critical.
        """
        ts = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
        self._events.appendleft((f"{ts} {text}", severity))
        self._refresh_display()

    def _refresh_display(self) -> None:
        self.update(self._render_text())

    def _render_text(self) -> str:
        """Build the markup string for display."""
        lines = ["[b]ACTIVITY[/b]"]
        if not self._events:
            lines.append("  No activity yet")
        else:
            for text, severity in list(self._events)[:_DISPLAY_LIMIT]:
                style = _SEVERITY_MARKUP.get(severity, "")
                if style:
                    lines.append(f"  [{style}]{text}[/{style}]")
                else:
                    lines.append(f"  {text}")
        return "\n".join(lines)

    @property
    def event_count(self) -> int:
        """Total events stored (up to max)."""
        return len(self._events)

    @property
    def events(self) -> list[tuple[str, str]]:
        """All stored events as (text, severity) pairs, most recent first."""
        return list(self._events)
