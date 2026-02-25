"""Recent trades log widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Static

if TYPE_CHECKING:
    from stat_arb.tui.data_provider import TradeDisplayRow


class TradesLog(Static):
    """Displays recent trade fills in a compact log format."""

    DEFAULT_CSS = """
    TradesLog {
        height: auto;
        padding: 1 2;
    }
    """

    def on_mount(self) -> None:
        self.update("[b]RECENT TRADES[/b]\n  No trades")

    def update_data(self, trades: list[TradeDisplayRow]) -> None:
        lines = ["[b]RECENT TRADES[/b]"]
        if not trades:
            lines.append("  No trades")
        else:
            for t in trades[:10]:
                time_str = t.fill_time.strftime("%H:%M")
                tag = "ENT" if t.is_entry else "EXT"
                lines.append(
                    f"  {time_str} {t.pair_key:<12} {t.side:<4} "
                    f"{t.symbol:<6} {t.quantity}@{t.price:.2f} [{tag}]"
                )
        self.update("\n".join(lines))
