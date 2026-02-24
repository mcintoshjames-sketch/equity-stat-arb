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
    direct persistence.
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


def _estimate_hurst(series: np.ndarray) -> float:
    """Estimate the Hurst exponent via the rescaled range (R/S) method.

    Applied to the first differences of the spread so that
    H < 0.5 = mean-reverting (anti-persistent increments),
    H > 0.5 = trending (persistent increments).

    Args:
        series: 1-D array of spread values.

    Returns:
        Hurst exponent.  H < 0.5 = mean-reverting, H > 0.5 = trending.
    """
    # R/S on first differences to measure persistence of increments
    diffs = np.diff(series)
    n = len(diffs)
    if n < 20:
        return 0.5  # insufficient data

    max_k = min(n // 2, 100)
    sizes = []
    rs_values = []

    for size in range(10, max_k + 1):
        num_chunks = n // size
        if num_chunks < 1:
            continue

        rs_list = []
        for i in range(num_chunks):
            chunk = diffs[i * size : (i + 1) * size]
            mean = np.mean(chunk)
            devs = np.cumsum(chunk - mean)
            r = np.max(devs) - np.min(devs)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            sizes.append(size)
            rs_values.append(np.mean(rs_list))

    if len(sizes) < 2:
        return 0.5

    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)

    # OLS: log(R/S) = H * log(n) + c
    coeffs = np.polyfit(log_sizes, log_rs, 1)
    return float(coeffs[0])


class PairFilter:
    """Multi-gate evaluator for candidate pairs.

    Args:
        config: Discovery configuration with threshold parameters.
    """

    def __init__(self, config: DiscoveryConfig) -> None:
        self._config = config
        self._coint_tester = CointegrationTester(config)
        self._hedge_estimator = HedgeRatioEstimator()

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
        # Gate 1: Cointegration test
        coint_result = self._coint_tester.test_pair(y_prices, x_prices)
        if coint_result is None:
            return None

        # Gate 2: Hedge ratio estimation (Kalman → OLS fallback)
        hr_result = self._hedge_estimator.estimate(y_prices, x_prices)

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
