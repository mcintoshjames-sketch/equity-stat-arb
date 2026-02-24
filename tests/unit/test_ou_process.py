"""Tests for OU half-life estimation."""

from __future__ import annotations

import numpy as np

from stat_arb.discovery.ou_process import _winsorize_diff, estimate_ou_half_life


def test_synthetic_ou_half_life() -> None:
    """Synthetic OU residuals should produce half_life in [5, 30]."""
    np.random.seed(42)
    n = 500
    theta = 0.05  # mean-reversion speed → half_life ≈ -ln(2)/(-0.05) ≈ 13.9
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] * (1 - theta) + np.random.normal(0, 1.0)

    half_life = estimate_ou_half_life(spread)

    assert 5 <= half_life <= 30, f"half_life={half_life:.1f}, expected in [5, 30]"


def test_random_walk_returns_inf() -> None:
    """A random walk (θ ≈ 0) should return inf."""
    np.random.seed(99)
    n = 500
    # Pure random walk: no mean reversion
    spread = np.cumsum(np.random.normal(0, 1, n))

    half_life = estimate_ou_half_life(spread)

    assert half_life == float("inf")


# ---------------------------------------------------------------------------
# Winsorization risk-area tests
# ---------------------------------------------------------------------------


def test_dividend_outliers_dampened() -> None:
    """Injected dividend jumps should be clipped, improving hl accuracy.

    Risk 1 validation: winsorization operates only on dS_t (the outlier
    spikes) and not on S_{t-1} (mean-reverting levels).  We verify by
    comparing the winsorized result against a no-winsorization baseline
    on the same contaminated data.
    """
    rng = np.random.default_rng(42)
    n = 252
    theta = 0.07  # true hl ≈ -ln(2)/(-0.07) ≈ 9.9 days
    true_hl = -np.log(2) / (-theta)

    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] * (1 - theta) + rng.normal(0, 0.5)

    # Inject 4 quarterly "ex-dividend" drops (~10σ shocks in dS)
    for idx in [63, 126, 189, 250]:
        spread[idx] -= 5.0

    # Compute hl WITH winsorization (current implementation)
    hl_winsorized = estimate_ou_half_life(spread)

    # Compute hl WITHOUT winsorization (raw regression on contaminated data)
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)
    denom = np.dot(spread_lag, spread_lag)
    theta_raw = np.dot(spread_lag, spread_diff) / denom
    hl_raw = -np.log(2) / theta_raw if theta_raw < 0 else float("inf")

    # Winsorized hl should be closer to the true value than the raw hl.
    # Level shifts from dividends still affect S_{t-1}, so we can't
    # perfectly recover true_hl — but we should improve.
    error_winsorized = abs(hl_winsorized - true_hl)
    error_raw = abs(hl_raw - true_hl)
    assert error_winsorized < error_raw, (
        f"Winsorized hl ({hl_winsorized:.1f}) should be closer to true "
        f"({true_hl:.1f}) than raw ({hl_raw:.1f})"
    )


def test_winsorize_preserves_array_length() -> None:
    """Winsorizer must return the same length array (Risk 3: alignment)."""
    rng = np.random.default_rng(7)
    diffs = rng.normal(0, 1, 252)
    result = _winsorize_diff(diffs)
    assert len(result) == len(diffs)


def test_winsorize_skips_short_arrays() -> None:
    """Arrays shorter than 20 observations are returned unchanged."""
    diffs = np.array([0.1, -0.2, 0.3, -0.4, 100.0])
    result = _winsorize_diff(diffs)
    np.testing.assert_array_equal(result, diffs)


def test_winsorize_safety_valve_skips_heavy_tails() -> None:
    """If >5% of observations would be clipped, skip entirely (Risk 2).

    Protects genuinely volatile low-vol pairs (e.g. dual-class shares)
    from over-clipping.
    """
    rng = np.random.default_rng(99)
    # Cauchy-like heavy tails: ~25% of observations beyond ±3σ_MAD
    diffs = rng.standard_cauchy(200)
    result = _winsorize_diff(diffs)
    # Safety valve should fire — output unchanged
    np.testing.assert_array_equal(result, diffs)


def test_winsorize_does_not_modify_input() -> None:
    """Winsorizer must not mutate the input array."""
    rng = np.random.default_rng(10)
    diffs = rng.normal(0, 1, 100)
    diffs[50] = 50.0  # inject outlier
    original = diffs.copy()
    _winsorize_diff(diffs)
    np.testing.assert_array_equal(diffs, original)


def test_clean_ou_unchanged_by_winsorization() -> None:
    """A clean OU process should give effectively the same half-life.

    A normal sample may have 1-2 observations near ±3σ_MAD that get
    lightly clipped — this is expected and harmless.  We verify the
    *half-life estimate* is unchanged, not bit-identical arrays.
    """
    rng = np.random.default_rng(42)
    n = 500
    theta = 0.05
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] * (1 - theta) + rng.normal(0, 1.0)

    # Half-life with winsorization (current implementation)
    hl_winsorized = estimate_ou_half_life(spread)

    # Half-life without winsorization (raw regression)
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)
    denom = np.dot(spread_lag, spread_lag)
    theta_raw = np.dot(spread_lag, spread_diff) / denom
    hl_raw = -np.log(2) / theta_raw if theta_raw < 0 else float("inf")

    # Should be within 5% of each other on clean data
    assert abs(hl_winsorized - hl_raw) / hl_raw < 0.05, (
        f"hl_winsorized={hl_winsorized:.2f} vs hl_raw={hl_raw:.2f} "
        "differ by more than 5% on clean data"
    )
