"""Risk utilisation gauge widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Static

if TYPE_CHECKING:
    from stat_arb.tui.data_provider import RiskUtilization


def _bar(used: float, limit: float, width: int = 10) -> str:
    """Render a simple ASCII progress bar."""
    if limit <= 0:
        return "[" + "-" * width + "]"
    frac = min(used / limit, 1.0)
    filled = int(frac * width)
    return "[" + "=" * filled + "-" * (width - filled) + f"] {frac:.0%}"


class RiskGauge(Static):
    """Displays risk utilisation bars for pairs, exposure, drawdown."""

    DEFAULT_CSS = """
    RiskGauge {
        height: auto;
        padding: 1 2;
    }
    """

    def on_mount(self) -> None:
        self.update("[b]RISK UTILIZATION[/b]\n  No data")

    def update_data(self, risk: RiskUtilization) -> None:
        pairs_bar = _bar(risk.pair_count, risk.pair_limit)
        exp_bar = _bar(risk.exposure, risk.exposure_limit)
        dd_bar = _bar(risk.drawdown_pct, risk.drawdown_limit)
        kill = "ON" if risk.kill_switch else "OFF"

        lines = [
            "[b]RISK UTILIZATION[/b]",
            f"  Pairs:    {risk.pair_count}/{risk.pair_limit}  {pairs_bar}",
            f"  Exposure: ${risk.exposure:,.0f}/${risk.exposure_limit:,.0f}  {exp_bar}",
            f"  Drawdown: {risk.drawdown_pct:.2%}/{risk.drawdown_limit:.0%}  {dd_bar}",
            f"  Kill Switch: {kill}",
        ]
        self.update("\n".join(lines))
