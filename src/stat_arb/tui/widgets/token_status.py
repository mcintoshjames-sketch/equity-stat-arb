"""Schwab token status widget with countdown."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Static

from stat_arb.tui.data_provider import ACCESS_TOKEN_LIFETIME, REFRESH_TOKEN_LIFETIME

if TYPE_CHECKING:
    from stat_arb.tui.data_provider import TokenStatus


def _format_remaining(seconds: float) -> str:
    """Format remaining seconds as human-readable string."""
    if seconds <= 0:
        return "EXPIRED"
    if seconds >= 86400:
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        return f"{days}d {hours}h"
    if seconds >= 3600:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


def _color_for_fraction(frac: float) -> str:
    """Return a rich color name based on remaining fraction."""
    if frac > 0.5:
        return "green"
    if frac > 0.1:
        return "yellow"
    return "red"


class TokenStatusWidget(Static):
    """Displays Schwab access/refresh token countdown."""

    DEFAULT_CSS = """
    TokenStatusWidget {
        height: auto;
        padding: 1 2;
    }
    """

    def on_mount(self) -> None:
        self.update("[b]TOKEN STATUS[/b]\n  No Schwab connection")

    def update_data(self, status: TokenStatus | None) -> None:
        if status is None:
            self.update("[b]TOKEN STATUS[/b]\n  No Schwab connection")
            return

        access_frac = status.access_remaining_s / ACCESS_TOKEN_LIFETIME
        refresh_frac = status.refresh_remaining_s / REFRESH_TOKEN_LIFETIME
        ac = _color_for_fraction(access_frac)
        rc = _color_for_fraction(refresh_frac)

        access_str = _format_remaining(status.access_remaining_s)
        refresh_str = _format_remaining(status.refresh_remaining_s)

        lines = [
            "[b]TOKEN STATUS[/b]",
            f"  Access:  [{ac}]{access_str}[/{ac}] remaining",
            f"  Refresh: [{rc}]{refresh_str}[/{rc}] remaining",
        ]
        self.update("\n".join(lines))
