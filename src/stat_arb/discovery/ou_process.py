"""Ornstein-Uhlenbeck half-life estimation for mean-reverting spreads.

Estimates the speed of mean reversion by regressing spread changes on
lagged spread levels: dS_t = theta * S_{t-1} + epsilon.

First differences are winsorized at ±3σ (MAD-based) to dampen
ex-dividend price discontinuities without requiring a corporate
actions calendar.
"""

from __future__ import annotations

import numpy as np

# MAD-to-σ conversion factor for normally distributed data.
# For X ~ N(μ, σ²), MAD = σ × Φ⁻¹(0.75) ≈ σ × 0.6745,
# so σ ≈ MAD × 1 / 0.6745 ≈ MAD × 1.4826.
_MAD_NORMAL_SCALE: float = 1.4826

# Maximum fraction of observations that winsorization may affect.
# If clipping would touch more than this share of the sample, the
# tail mass is part of the genuine distribution (e.g. volatile but
# stationary dual-class spread), not corporate-action artifacts.
_MAX_CLIP_FRAC: float = 0.05


def estimate_ou_half_life(spread: np.ndarray) -> float:
    """Estimate the half-life of mean reversion for an OU process.

    Fits the discrete AR(1) model ``dS_t = theta * S_{t-1} + epsilon``
    and computes ``half_life = -ln(2) / theta``.

    First differences (dS_t) are winsorized at ±3 × σ_MAD to dampen
    ex-dividend discontinuities.  Lagged levels (S_{t-1}) are never
    modified so the mean-reverting structure is preserved.

    Args:
        spread: 1-D array of spread values (y - beta*x - intercept).

    Returns:
        Half-life in days.  Returns ``inf`` if theta >= 0 (no mean reversion).
    """
    # Both arrays have length n-1 and share the same index mapping:
    #   spread_lag[i]  = S_i
    #   spread_diff[i] = S_{i+1} - S_i
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)

    # Winsorize first differences only — levels are left untouched.
    spread_diff = _winsorize_diff(spread_diff)

    # OLS: dS = theta * S_{t-1}
    # theta = sum(S_{t-1} * dS) / sum(S_{t-1}^2)
    denom = np.dot(spread_lag, spread_lag)
    if denom == 0:
        return float("inf")

    theta = np.dot(spread_lag, spread_diff) / denom

    # Non-negative theta means no mean-reversion — spread is unit root or explosive
    if theta >= 0:
        return float("inf")

    half_life = -np.log(2) / theta
    return float(half_life)


def _winsorize_diff(
    diffs: np.ndarray,
    k: float = 3.0,
) -> np.ndarray:
    """Winsorize first differences at ±kσ using MAD-based robust scale.

    Uses Median Absolute Deviation instead of standard deviation so the
    scale estimate is not inflated by the very outliers being clipped.
    A safety valve skips winsorization entirely if more than 5% of
    observations would be clipped — protecting genuinely heavy-tailed
    but stationary processes from over-clipping.

    Args:
        diffs: 1-D array of spread first differences (dS_t).
        k: Number of robust σ for the clip boundary (default 3.0).

    Returns:
        Winsorized copy of *diffs*.  Never modifies the input array.
    """
    n = len(diffs)
    if n < 20:
        return diffs

    median_d = np.median(diffs)
    mad = np.median(np.abs(diffs - median_d))
    sigma_mad = mad * _MAD_NORMAL_SCALE

    # All diffs (nearly) identical — nothing to clip.
    if sigma_mad < 1e-15:
        return diffs

    clip = k * sigma_mad
    lower = median_d - clip
    upper = median_d + clip

    # Safety valve: if clipping would affect > _MAX_CLIP_FRAC of the
    # sample, the tails are genuine — skip winsorization.
    n_clipped = int(np.sum((diffs < lower) | (diffs > upper)))
    if n_clipped > n * _MAX_CLIP_FRAC:
        return diffs

    return np.clip(diffs, lower, upper)
