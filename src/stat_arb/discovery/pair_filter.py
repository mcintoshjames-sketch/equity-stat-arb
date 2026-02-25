"""Multi-gate pair qualification filter.

Evaluates candidate pairs through cointegration, hedge ratio, OU half-life,
and Hurst exponent gates.  Only pairs passing all gates produce a frozen
:class:`QualifiedPair` dataclass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from stat_arb.discovery.cointegration import CointegrationTester
from stat_arb.discovery.hedge_ratio import HedgeRatioEstimator
from stat_arb.discovery.ou_process import estimate_ou_half_life

if TYPE_CHECKING:
    from stat_arb.config.settings import DiscoveryConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualifiedPair:
    """Frozen formation parameters for a qualified cointegrated pair.

    Fields align 1:1 with the ``DiscoveredPair`` ORM schema for
    direct persistence.  Lifecycle fields (``discovery_date``,
    ``trading_expiry``, ``cohort_id``) are optional and populated
    by the rolling scheduler.
    """

    symbol_y: str
    symbol_x: str
    sector: str
    formation_start: date
    formation_end: date
    hedge_ratio: float
    intercept: float
    spread_mean: float
    spread_std: float
    half_life: float
    coint_pvalue: float
    adf_pvalue: float
    hurst: float
    discovery_date: date | None = None
    trading_expiry: date | None = None
    cohort_id: str | None = None


def with_lifecycle(
    pair: QualifiedPair,
    discovery_date: date,
    trading_expiry: date,
    cohort_id: str,
) -> QualifiedPair:
    """Attach lifecycle metadata to a qualified pair via dataclass replace."""
    from dataclasses import replace
    return replace(
        pair,
        discovery_date=discovery_date,
        trading_expiry=trading_expiry,
        cohort_id=cohort_id,
    )


def _estimate_hurst(series: np.ndarray) -> float:
    """Estimate the Hurst exponent from variance scaling of spread levels.

    Uses the variance-ratio method: ``Var(X_{t+tau} - X_t) ~ tau^{2H}``.
    For a mean-reverting spread the variance saturates at large lags,
    giving H < 0.5.  For a random walk H ~ 0.5; for a trending spread
    H > 0.5.

    Args:
        series: 1-D array of spread level values.

    Returns:
        Hurst exponent.  H < 0.5 = mean-reverting, H > 0.5 = trending.
    """
    n = len(series)
    if n < 100:
        return 0.5  # insufficient data

    max_lag = min(n // 4, 100)
    lags = list(range(2, max_lag + 1))
    variances = []

    for lag in lags:
        diffs = series[lag:] - series[:-lag]
        variances.append(np.var(diffs))

    log_lags = np.log(lags)
    log_vars = np.log(variances)

    # log(Var) = 2H * log(tau) + c  →  slope / 2 = H
    coeffs = np.polyfit(log_lags, log_vars, 1)
    return float(coeffs[0] / 2.0)


class PairFilter:
    """Multi-gate evaluator for candidate pairs.

    Args:
        config: Discovery configuration with threshold parameters.
    """

    def __init__(self, config: DiscoveryConfig) -> None:
        self._config = config
        self._coint_tester = CointegrationTester(config)
        self._hedge_estimator = HedgeRatioEstimator(config)

    def evaluate(
        self,
        symbol_y: str,
        symbol_x: str,
        sector: str,
        y_prices: pd.Series,
        x_prices: pd.Series,
        formation_start: date,
        formation_end: date,
    ) -> QualifiedPair | None:
        """Run candidate pair through all qualification gates.

        Args:
            symbol_y: Y-leg ticker symbol.
            symbol_x: X-leg ticker symbol.
            sector: Sector classification for risk management.
            y_prices: Y-leg close price series.
            x_prices: X-leg close price series.
            formation_start: Start of formation window.
            formation_end: End of formation window.

        Returns:
            ``QualifiedPair`` if all gates pass, ``None`` otherwise.
        """
        # Gate 1: Dual Engle-Granger cointegration test (both directions)
        coint_result = self._coint_tester.test_pair(y_prices, x_prices)
        if coint_result is None:
            return None

        # If reverse direction had stronger ADF, swap legs
        if coint_result.swapped:
            symbol_y, symbol_x = symbol_x, symbol_y
            y_prices, x_prices = x_prices, y_prices

        # Gate 2: Hedge ratio estimation (Kalman → OLS fallback)
        hr_result = self._hedge_estimator.estimate(
            y_prices, x_prices, symbol_y, symbol_x,
        )
        if hr_result is None:
            return None

        # Gate 3: Compute spread and OU half-life
        spread = y_prices.values - hr_result.beta * x_prices.values - hr_result.intercept
        half_life = estimate_ou_half_life(spread)

        min_hl = self._config.min_half_life_days
        max_hl = self._config.max_half_life_days
        if half_life < min_hl or half_life > max_hl:
            logger.debug(
                "%s/%s rejected: half_life=%.1f outside [%d, %d]",
                symbol_y, symbol_x, half_life, min_hl, max_hl,
            )
            return None

        # Gate 4: Hurst exponent
        hurst = _estimate_hurst(spread)
        if hurst >= self._config.max_hurst:
            logger.debug(
                "%s/%s rejected: hurst=%.3f >= %.3f",
                symbol_y, symbol_x, hurst, self._config.max_hurst,
            )
            return None

        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread, ddof=1))

        return QualifiedPair(
            symbol_y=symbol_y,
            symbol_x=symbol_x,
            sector=sector,
            formation_start=formation_start,
            formation_end=formation_end,
            hedge_ratio=hr_result.beta,
            intercept=hr_result.intercept,
            spread_mean=spread_mean,
            spread_std=spread_std,
            half_life=half_life,
            coint_pvalue=coint_result.coint_pvalue,
            adf_pvalue=coint_result.adf_pvalue,
            hurst=hurst,
        )
