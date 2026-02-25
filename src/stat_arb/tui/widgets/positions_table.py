"""Active positions table widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import DataTable

if TYPE_CHECKING:
    from stat_arb.tui.data_provider import PairDisplayRow

_COLUMNS = ("Pair", "Dir", "Z-Score", "H.L.", "Entry Z", "Days", "Sector")


class PositionsTable(DataTable):
    """DataTable showing active pair positions."""

    DEFAULT_CSS = """
    PositionsTable {
        height: auto;
        min-height: 5;
        max-height: 16;
    }
    """

    def on_mount(self) -> None:
        for col in _COLUMNS:
            self.add_column(col, key=col)

    def update_data(self, pairs: list[PairDisplayRow]) -> None:
        self.clear()
        for p in pairs:
            z_str = f"{p.z_score:+.2f}" if p.z_score is not None else "—"
            entry_z_str = f"{p.entry_z:+.2f}" if p.entry_z is not None else "—"
            days_str = str(p.days_held) if p.days_held is not None else "—"
            self.add_row(
                p.pair_key,
                p.direction,
                z_str,
                f"{p.half_life:.1f}d",
                entry_z_str,
                days_str,
                p.sector,
            )
