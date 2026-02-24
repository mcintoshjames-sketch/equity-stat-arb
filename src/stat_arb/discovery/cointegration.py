"""Engle-Granger cointegration testing for candidate pairs.

Uses ``statsmodels.tsa.stattools.coint`` for the Engle-Granger two-step test
and ``adfuller`` to verify stationarity of OLS residuals.  Tests both
regression directions (y-on-x and x-on-y) and keeps the direction with
the more negative ADF statistic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

if TYPE_CHECKING:
    from stat_arb.config.settings import DiscoveryConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CointegrationResult:
    """Frozen result of the Engle-Granger cointegration test."""

    coint_pvalue: float
    adf_pvalue: float
    adf_stat: float
    residuals: np.ndarray
    swapped: bool  # True when x-on-y direction had more negative ADF


class CointegrationTester:
    """Run Engle-Granger cointegration test on candidate pairs.

    Tests both regression directions and keeps the one whose OLS
    residuals have the more negative ADF statistic (stronger
    stationarity evidence).

    Args:
        config: Discovery configuration with p-value thresholds.
    """

    def __init__(self, config: DiscoveryConfig) -> None:
        self._config = config

    def test_pair(
        self, y: pd.Series, x: pd.Series,
    ) -> CointegrationResult | None:
        """Test whether two price series are cointegrated.

        Runs Engle-Granger in both directions (y-on-x and x-on-y) and
        returns the result with the more negative ADF statistic.

        Args:
            y: First price series (tentative Y-leg).
            x: Second price series (tentative X-leg).

        Returns:
            ``CointegrationResult`` if cointegrated, ``None`` otherwise.
        """
        result_fwd = self._test_direction(y, x, swapped=False)
        result_rev = self._test_direction(x, y, swapped=True)

        if result_fwd is None and result_rev is None:
            return None
        if result_fwd is not None and result_rev is None:
            return result_fwd
        if result_fwd is None and result_rev is not None:
            return result_rev

        # Both passed — keep direction with more negative ADF stat
        if result_fwd.adf_stat <= result_rev.adf_stat:
            return result_fwd
        return result_rev

    def _test_direction(
        self,
        y: pd.Series,
        x: pd.Series,
        *,
        swapped: bool,
    ) -> CointegrationResult | None:
        """Run Engle-Granger for a single regression direction."""
        _, coint_pvalue, _ = coint(y.values, x.values)

        if coint_pvalue > self._config.coint_pvalue:
            return None

        # OLS regression: y = beta * x + intercept
        x_const = add_constant(x.values)
        ols_result = OLS(y.values, x_const).fit()
        residuals = y.values - ols_result.predict(x_const)

        # ADF test on residuals
        adf_stat, adf_pvalue, *_ = adfuller(residuals, autolag="AIC")

        if adf_pvalue > self._config.adf_pvalue:
            return None

        return CointegrationResult(
            coint_pvalue=float(coint_pvalue),
            adf_pvalue=float(adf_pvalue),
            adf_stat=float(adf_stat),
            residuals=residuals,
            swapped=swapped,
        )
