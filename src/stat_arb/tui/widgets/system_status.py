"""System status widget showing connectivity, mode, and engine state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from textual.widgets import Static

if TYPE_CHECKING:
    from stat_arb.tui.data_provider import SystemStatus, TokenStatus


_MODE_MARKUP = {
    "paper": "[green]PAPER[/green]",
    "live": "[red bold]LIVE[/red bold]",
    "sim": "SIM",
}

_ENGINE_STATE_MARKUP = {
    "running": "[green]running[/green]",
    "idle": "[yellow]idle[/yellow]",
    "stopped": "[red]stopped[/red]",
    "not running": "[dim]not running[/dim]",
}


class SystemStatusWidget(Static):
    """Displays system mode, DB/Schwab connectivity, engine state, and token summary."""

    DEFAULT_CSS = """
    SystemStatusWidget {
        height: auto;
        padding: 1 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._engine_state: str = "not running"
        self._next_scan_utc: datetime | None = None

    def on_mount(self) -> None:
        self.update("[b]SYSTEM STATUS[/b]\n  Loading...")

    def set_engine_state(self, state: str) -> None:
        """Update the engine state display."""
        self._engine_state = state

    def set_next_scan(self, next_scan_utc: datetime) -> None:
        """Update the next scan countdown target."""
        self._next_scan_utc = next_scan_utc

    def update_data(
        self,
        status: SystemStatus,
        token_status: TokenStatus | None = None,
    ) -> None:
        db_str = "[green]connected[/green]" if status.db_connected else "[red]disconnected[/red]"
        schwab_str = (
            "[green]connected[/green]" if status.schwab_connected
            else "[red]disconnected[/red]"
        )
        mode_str = _MODE_MARKUP.get(status.broker_mode, status.broker_mode)
        engine_str = _ENGINE_STATE_MARKUP.get(self._engine_state, self._engine_state)

        lines = [
            "[b]SYSTEM STATUS[/b]",
            f"  Mode:   {mode_str}",
            f"  DB:     {db_str}",
            f"  Schwab: {schwab_str}",
        ]

        # Only show engine info when engine is actually attached
        if self._engine_state != "not running":
            lines.insert(2, f"  Engine: {engine_str}")
            if self._next_scan_utc is not None:
                lines.append(f"  Next:   {self._format_countdown()}")

        if token_status is not None:
            from stat_arb.tui.widgets.token_status import _format_remaining

            lines.append(f"  Access:  {_format_remaining(token_status.access_remaining_s)}")
            lines.append(f"  Refresh: {_format_remaining(token_status.refresh_remaining_s)}")

        # Earnings blackout status
        if status.earnings_blackout_enabled:
            lines.append("  Earnings: [green]enabled[/green]")
        else:
            lines.append("  Earnings: [yellow]disabled (no FMP_API_KEY)[/yellow]")

        if status.last_step_time is not None:
            lines.append(f"  Last Step: {status.last_step_time:%Y-%m-%d %H:%M}")

        self.update("\n".join(lines))

    def _format_countdown(self) -> str:
        """Format remaining time until next scan."""
        if self._next_scan_utc is None:
            return "—"
        remaining = (self._next_scan_utc - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            return "imminent"
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        return f"{hours}h {minutes:02d}m"
