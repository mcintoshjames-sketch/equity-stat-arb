"""Backtest result containers.

Provides ``TradeRecord`` for per-trade logging and ``BacktestResult``
for aggregated equity curve and performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from stat_arb.config.constants import ExitReason, Signal
from stat_arb.reporting import metrics


@dataclass(frozen=True)
class TradeRecord:
    """Immutable record of a single pair trade (entry to exit).

    Attributes:
        pair_key: String identifier for the pair (e.g. ``"AAPL/MSFT"``).
        signal: Entry signal type.
        entry_date: Date the position was opened.
        exit_date: Date the position was closed (or None if still open).
        entry_z: Z-score at entry.
        exit_z: Z-score at exit (or None if still open).
        pnl: Realised profit/loss in dollars.
        exit_reason: Why the position was closed.
    """

    pair_key: str
    signal: Signal
    entry_date: date
    exit_date: date | None
    entry_z: float
    exit_z: float | None
    pnl: float
    exit_reason: ExitReason | None


@dataclass
class BacktestResult:
    """Aggregated backtest output.

    Attributes:
        equity_curve: Daily portfolio values.
        trades: All completed trade records.
        config_json: Serialised configuration used for the run.
        start_date: First date in the backtest.
        end_date: Last date in the backtest.
    """

    equity_curve: list[float] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    config_json: str = ""
    start_date: date | None = None
    end_date: date | None = None

    @property
    def total_return(self) -> float:
        """Total return as a decimal (e.g. 0.12 = 12%)."""
        if len(self.equity_curve) < 2 or self.equity_curve[0] <= 0:
            return 0.0
        return self.equity_curve[-1] / self.equity_curve[0] - 1.0

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe ratio from equity curve."""
        if len(self.equity_curve) < 3:
            return 0.0
        returns = [
            self.equity_curve[i] / self.equity_curve[i - 1] - 1.0
            for i in range(1, len(self.equity_curve))
        ]
        return metrics.sharpe_ratio(returns)

    @property
    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown."""
        return metrics.max_drawdown(self.equity_curve)

    @property
    def total_trades(self) -> int:
        """Total number of completed trades."""
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        """Fraction of trades with positive P&L."""
        pnls = [t.pnl for t in self.trades]
        return metrics.win_rate(pnls)
