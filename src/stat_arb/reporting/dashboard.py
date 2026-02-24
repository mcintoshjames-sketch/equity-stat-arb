"""Dashboard builder for portfolio performance summaries.

Aggregates equity curve, trade records, and risk metrics into a
structured summary dict and human-readable text output.
"""

from __future__ import annotations

from stat_arb.backtest.results import BacktestResult
from stat_arb.reporting import metrics


class DashboardBuilder:
    """Build portfolio performance dashboards from backtest results.

    Produces a structured dict with performance, trading, and risk
    sections, plus a human-readable text formatter.
    """

    def build(self, result: BacktestResult) -> dict:
        """Construct a performance summary from a backtest result.

        Args:
            result: Completed backtest result.

        Returns:
            Dict with ``performance``, ``trading``, and ``risk`` sections.
        """
        equity = result.equity_curve
        pnls = [t.pnl for t in result.trades]

        # Daily returns from equity curve
        daily_returns = []
        if len(equity) >= 2:
            daily_returns = [
                equity[i] / equity[i - 1] - 1.0
                for i in range(1, len(equity))
            ]

        return {
            "performance": {
                "total_return": result.total_return,
                "cagr": metrics.cagr(equity),
                "sharpe": metrics.sharpe_ratio(daily_returns),
                "sortino": metrics.sortino_ratio(daily_returns),
                "max_drawdown": metrics.max_drawdown(equity),
            },
            "trading": {
                "total_trades": result.total_trades,
                "win_rate": metrics.win_rate(pnls),
                "profit_factor": metrics.profit_factor(pnls),
                "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
            },
            "risk": {
                "max_drawdown": metrics.max_drawdown(equity),
                "start_date": str(result.start_date) if result.start_date else None,
                "end_date": str(result.end_date) if result.end_date else None,
            },
        }

    def format_text(self, summary: dict) -> str:
        """Format a summary dict as human-readable console output.

        Args:
            summary: Dict from ``build()``.

        Returns:
            Multi-line text string.
        """
        perf = summary["performance"]
        trad = summary["trading"]
        risk = summary["risk"]

        lines = [
            "=" * 50,
            "PORTFOLIO PERFORMANCE SUMMARY",
            "=" * 50,
            "",
            "Performance:",
            f"  Total Return:  {perf['total_return']:>10.2%}",
            f"  CAGR:          {perf['cagr']:>10.2%}",
            f"  Sharpe:        {perf['sharpe']:>10.2f}",
            f"  Sortino:       {perf['sortino']:>10.2f}",
            f"  Max Drawdown:  {perf['max_drawdown']:>10.2%}",
            "",
            "Trading:",
            f"  Total Trades:  {trad['total_trades']:>10d}",
            f"  Win Rate:      {trad['win_rate']:>10.2%}",
            f"  Profit Factor: {trad['profit_factor']:>10.2f}",
            f"  Avg P&L:       ${trad['avg_pnl']:>9.2f}",
            "",
            "Risk:",
            f"  Max Drawdown:  {risk['max_drawdown']:>10.2%}",
            f"  Period:        {risk['start_date']} to {risk['end_date']}",
            "=" * 50,
        ]
        return "\n".join(lines)
