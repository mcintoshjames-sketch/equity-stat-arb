"""Engle-Granger cointegration testing for candidate pairs.

Uses ``statsmodels.tsa.stattools.coint`` for the Engle-Granger two-step test
and ``adfuller`` to verify stationarity of OLS residuals.
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
    residuals: np.ndarray


class CointegrationTester:
    """Run Engle-Granger cointegration test on candidate pairs.

    Args:
        config: Discovery configuration with p-value thresholds.
    """

    def __init__(self, config: DiscoveryConfig) -> None:
        self._config = config

    def test_pair(self, y: pd.Series, x: pd.Series) -> CointegrationResult | None:
        """Test whether two price series are cointegrated.

        Args:
            y: Dependent (Y-leg) close prices.
            x: Independent (X-leg) close prices.

        Returns:
            ``CointegrationResult`` if cointegrated, ``None`` otherwise.
        """
        # Engle-Granger cointegration test
        coint_stat, coint_pvalue, _ = coint(y.values, x.values)

        if coint_pvalue > self._config.coint_pvalue:
            logger.debug(
                "Cointegration rejected: p=%.4f > %.4f",
                coint_pvalue, self._config.coint_pvalue,
            )
            return None

        # OLS regression: y = beta * x + intercept
        x_const = add_constant(x.values)
        ols_result = OLS(y.values, x_const).fit()
        residuals = y.values - ols_result.predict(x_const)

        # ADF test on residuals
        adf_stat, adf_pvalue, *_ = adfuller(residuals, autolag="AIC")

        if adf_pvalue > self._config.adf_pvalue:
            logger.debug("ADF rejected: p=%.4f > %.4f", adf_pvalue, self._config.adf_pvalue)
            return None

        return CointegrationResult(
            coint_pvalue=float(coint_pvalue),
            adf_pvalue=float(adf_pvalue),
            residuals=residuals,
        )
