"""Portfolio summary widget showing value, P&L, exposure, drawdown."""

from __future__ import annotations

from textual.widgets import Static


class PortfolioSummary(Static):
    """Displays portfolio value, daily P&L, drawdown, and gross exposure."""

    DEFAULT_CSS = """
    PortfolioSummary {
        height: auto;
        padding: 1 2;
    }
    """

    _value: float = 0.0
    _daily_pnl: float = 0.0
    _drawdown_pct: float = 0.0
    _gross_exposure: float = 0.0
    _active_pairs: int = 0

    def update_data(
        self,
        value: float,
        daily_pnl: float,
        drawdown_pct: float,
        gross_exposure: float,
        active_pairs: int,
    ) -> None:
        self._value = value
        self._daily_pnl = daily_pnl
        self._drawdown_pct = drawdown_pct
        self._gross_exposure = gross_exposure
        self._active_pairs = active_pairs
        self._render_content()

    def on_mount(self) -> None:
        self._render_content()

    def _render_content(self) -> None:
        pnl_sign = "+" if self._daily_pnl >= 0 else ""
        lines = [
            "[b]PORTFOLIO[/b]",
            f"  Value:       ${self._value:,.0f}",
            f"  Daily P&L:   {pnl_sign}${self._daily_pnl:,.0f}",
            f"  Drawdown:    {self._drawdown_pct:.2%}",
            f"  Gross Exp:   ${self._gross_exposure:,.0f}",
            f"  Active Pairs: {self._active_pairs}",
        ]
        self.update("\n".join(lines))
