"""Pure portfolio performance metric functions.

Stateless, no DB or broker dependencies — safe to call from backtest
results, dashboards, or alert monitors.
"""

from __future__ import annotations

import math


def sharpe_ratio(
    daily_returns: list[float],
    rf: float = 0.0,
    days: int = 252,
) -> float:
    """Annualised Sharpe ratio.

    Args:
        daily_returns: Sequence of daily simple returns.
        rf: Daily risk-free rate (default 0).
        days: Trading days per year for annualisation.

    Returns:
        Annualised Sharpe.  Returns 0.0 if fewer than 2 observations
        or if standard deviation is zero.
    """
    n = len(daily_returns)
    if n < 2:
        return 0.0

    excess = [r - rf for r in daily_returns]
    mean = sum(excess) / n
    var = sum((x - mean) ** 2 for x in excess) / (n - 1)
    std = math.sqrt(var)
    if std == 0.0:
        return 0.0
    return (mean / std) * math.sqrt(days)


def sortino_ratio(
    daily_returns: list[float],
    rf: float = 0.0,
    days: int = 252,
) -> float:
    """Annualised Sortino ratio (downside deviation).

    Args:
        daily_returns: Sequence of daily simple returns.
        rf: Daily risk-free rate (default 0).
        days: Trading days per year for annualisation.

    Returns:
        Annualised Sortino.  Returns 0.0 if fewer than 2 observations
        or if downside deviation is zero.
    """
    n = len(daily_returns)
    if n < 2:
        return 0.0

    excess = [r - rf for r in daily_returns]
    mean = sum(excess) / n
    downside = [min(x, 0.0) ** 2 for x in excess]
    down_var = sum(downside) / (n - 1)
    down_std = math.sqrt(down_var)
    if down_std == 0.0:
        return 0.0
    return (mean / down_std) * math.sqrt(days)


def max_drawdown(equity_curve: list[float]) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction.

    Args:
        equity_curve: Sequence of portfolio values (not returns).

    Returns:
        Drawdown as a positive float in [0, 1].  Returns 0.0 for
        empty or monotonically increasing curves.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    worst = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > worst:
            worst = dd
    return worst


def cagr(equity_curve: list[float], days: int = 252) -> float:
    """Compound annual growth rate from an equity curve.

    Args:
        equity_curve: Sequence of portfolio values.
        days: Trading days per year for annualisation.

    Returns:
        CAGR as a decimal (e.g. 0.12 = 12%).  Returns 0.0 if the curve
        has fewer than 2 points or starts at zero.
    """
    n = len(equity_curve)
    if n < 2 or equity_curve[0] <= 0:
        return 0.0

    total_return = equity_curve[-1] / equity_curve[0]
    years = (n - 1) / days
    if years <= 0:
        return 0.0
    return total_return ** (1.0 / years) - 1.0


def win_rate(pnls: list[float]) -> float:
    """Fraction of trades with positive P&L.

    Args:
        pnls: Sequence of per-trade P&L values.

    Returns:
        Win rate in [0, 1].  Returns 0.0 for empty input.
    """
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


def profit_factor(pnls: list[float]) -> float:
    """Ratio of gross profits to gross losses.

    Args:
        pnls: Sequence of per-trade P&L values.

    Returns:
        Profit factor (> 1 is profitable).  Returns ``float('inf')``
        if there are no losses, 0.0 if there are no profits.
    """
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss
