"""Hedge ratio estimation via Kalman filter with rolling OLS fallback.

Primary estimator uses ``pykalman.KalmanFilter`` to track time-varying
beta and intercept.  Falls back to rolling OLS when Kalman EM fails to
converge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROLLING_WINDOW = 60  # days for OLS fallback


@dataclass(frozen=True)
class HedgeRatioResult:
    """Frozen result of hedge ratio estimation."""

    beta: float
    intercept: float
    beta_series: np.ndarray


class HedgeRatioEstimator:
    """Estimate time-varying hedge ratio between two price series."""

    def estimate(self, y: pd.Series, x: pd.Series) -> HedgeRatioResult:
        """Estimate hedge ratio using Kalman filter, falling back to rolling OLS.

        Args:
            y: Dependent (Y-leg) close prices.
            x: Independent (X-leg) close prices.

        Returns:
            ``HedgeRatioResult`` with final beta, intercept, and full beta series.
        """
        try:
            return self._kalman_estimate(y, x)
        except Exception:
            logger.warning("Kalman EM failed to converge — falling back to rolling OLS")
            return self._rolling_ols_estimate(y, x)

    def _kalman_estimate(self, y: pd.Series, x: pd.Series) -> HedgeRatioResult:
        """Kalman filter estimation of time-varying hedge ratio."""
        from pykalman import KalmanFilter

        n = len(y)
        # Observation matrix: [[x_t, 1]] for each timestep
        obs_mat = np.column_stack([x.values, np.ones(n)])
        obs_mat = obs_mat.reshape(n, 1, 2)

        kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=2,
            transition_matrices=np.eye(2),
            observation_matrices=obs_mat,
            initial_state_mean=np.array([1.0, 0.0]),
            initial_state_covariance=np.eye(2),
        )

        # EM to learn covariance parameters
        kf = kf.em(y.values, n_iter=10)

        # Smooth to get full state trajectory
        smoothed_state, _ = kf.smooth(y.values)

        beta_series = smoothed_state[:, 0]
        intercept_series = smoothed_state[:, 1]

        return HedgeRatioResult(
            beta=float(beta_series[-1]),
            intercept=float(intercept_series[-1]),
            beta_series=beta_series,
        )

    def _rolling_ols_estimate(self, y: pd.Series, x: pd.Series) -> HedgeRatioResult:
        """Rolling OLS fallback with 60-day window."""
        n = len(y)
        beta_series = np.full(n, np.nan)

        for i in range(n):
            start = max(0, i - _ROLLING_WINDOW + 1)
            if i - start + 1 < 10:  # need at least 10 observations
                continue
            y_win = y.values[start : i + 1]
            x_win = x.values[start : i + 1]
            x_const = np.column_stack([x_win, np.ones(len(x_win))])
            coeffs, _, _, _ = np.linalg.lstsq(x_const, y_win, rcond=None)
            beta_series[i] = coeffs[0]

        # Fill early NaNs with first valid value
        first_valid = np.where(~np.isnan(beta_series))[0]
        if len(first_valid) > 0:
            beta_series[: first_valid[0]] = beta_series[first_valid[0]]

        # Final OLS for beta and intercept
        x_const = np.column_stack([x.values, np.ones(n)])
        coeffs, _, _, _ = np.linalg.lstsq(x_const, y.values, rcond=None)

        return HedgeRatioResult(
            beta=float(coeffs[0]),
            intercept=float(coeffs[1]),
            beta_series=beta_series,
        )
