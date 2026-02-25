"""Pairs detail screen — full table with statistical parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Static

if TYPE_CHECKING:
    from stat_arb.tui.data_provider import DashboardDataProvider, PairDisplayRow

_COLUMNS = (
    "Pair", "Sector", "Dir", "Z-Score", "Beta", "Mu", "Sigma",
    "H.L.", "Coint p", "ADF p", "Hurst", "Entry Z", "Days",
)


class PairsScreen(Screen):
    """Detailed pairs view with full statistics."""

    BINDINGS = [
        ("r", "refresh_data", "Refresh"),
    ]

    def __init__(self, provider: DashboardDataProvider) -> None:
        super().__init__()
        self._provider = provider

    def compose(self) -> ComposeResult:
        with Vertical(id="pairs-table-panel"):
            yield Static("[b]PAIRS DETAIL[/b]")
            yield DataTable(id="pairs-detail-table")
        yield Static(id="pair-detail-panel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#pairs-detail-table", DataTable)
        for col in _COLUMNS:
            table.add_column(col, key=col)
        self._refresh_all()
        self.set_interval(30, self._refresh_all)

    def action_refresh_data(self) -> None:
        self._refresh_all()

    def _refresh_all(self) -> None:
        pairs = self._provider.get_active_pairs()
        table = self.query_one("#pairs-detail-table", DataTable)
        table.clear()
        for p in pairs:
            z_str = f"{p.z_score:+.2f}" if p.z_score is not None else "—"
            entry_z_str = f"{p.entry_z:+.2f}" if p.entry_z is not None else "—"
            days_str = str(p.days_held) if p.days_held is not None else "—"
            table.add_row(
                p.pair_key,
                p.sector,
                p.direction,
                z_str,
                f"{p.hedge_ratio:.3f}",
                f"{p.spread_mean:.4f}",
                f"{p.spread_std:.4f}",
                f"{p.half_life:.1f}d",
                f"{p.coint_pvalue:.3f}",
                f"{p.adf_pvalue:.3f}",
                f"{p.hurst:.3f}",
                entry_z_str,
                days_str,
            )

        detail = self.query_one("#pair-detail-panel", Static)
        if pairs:
            self._show_pair_detail(pairs[0], detail)
        else:
            detail.update("No active pairs")

    def _show_pair_detail(self, p: PairDisplayRow, widget: Static) -> None:
        """Show z-score position visual for a selected pair."""
        z = p.entry_z if p.entry_z is not None else 0.0
        # Render a number line from -4 to +4
        width = 40
        center = width // 2
        pos = int(center + (z / 4.0) * center)
        pos = max(0, min(width - 1, pos))
        line = list("-" * width)
        line[center] = "|"
        line[pos] = "*"

        lines = [
            f"[b]{p.pair_key}[/b] — {p.sector} — {p.direction}",
            f"  Z-Score Line:  -4 {''.join(line)} +4",
            f"  Entry Z: {z:+.2f}  |  Half-Life: {p.half_life:.1f}d"
            f"  |  Beta: {p.hedge_ratio:.3f}",
        ]
        widget.update("\n".join(lines))
