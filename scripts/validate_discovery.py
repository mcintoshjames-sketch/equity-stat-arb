#!/usr/bin/env python3
"""Discovery layer validation script.

Runs the full discovery pipeline on synthetic data and prints summary
statistics to verify correctness of all gates.

Usage:
    python scripts/validate_discovery.py
"""

from __future__ import annotations

import sys
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# Ensure src/ is on the path
sys.path.insert(0, "src")

from stat_arb.config.settings import DiscoveryConfig  # noqa: E402
from stat_arb.data.universe import Universe  # noqa: E402
from stat_arb.discovery.pair_discovery import PairDiscovery  # noqa: E402


def _build_synthetic_prices(n: int = 500) -> pd.DataFrame:
    """Build synthetic prices with known cointegration structure."""
    np.random.seed(42)
    dates = pd.bdate_range(start="2022-01-01", periods=n)

    # Base random walk
    x_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))

    # Cointegrated with X: Y = 1.2*X + OU noise (theta=0.1, strong noise)
    theta_y = 0.1
    noise_y = np.zeros(n)
    for i in range(1, n):
        noise_y[i] = noise_y[i - 1] * (1 - theta_y) + np.random.normal(0, 5.0)
    y_prices = 1.2 * x_prices + 5.0 + noise_y

    # Another cointegrated pair: W = 0.8*X + OU noise (theta=0.15)
    theta_w = 0.15
    noise_w = np.zeros(n)
    for i in range(1, n):
        noise_w[i] = noise_w[i - 1] * (1 - theta_w) + np.random.normal(0, 4.0)
    w_prices = 0.8 * x_prices + 10.0 + noise_w

    # Independent random walk (not cointegrated)
    z_prices = 80 * np.exp(np.cumsum(np.random.normal(-0.0003, 0.025, n)))

    # Another independent random walk
    v_prices = 60 * np.exp(np.cumsum(np.random.normal(0.0002, 0.018, n)))

    df = pd.DataFrame(
        {
            "SYM_Y": y_prices,
            "SYM_X": x_prices,
            "SYM_W": w_prices,
            "SYM_Z": z_prices,
            "SYM_V": v_prices,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def main() -> None:
    universe = Universe(
        symbols=["SYM_Y", "SYM_X", "SYM_W", "SYM_Z", "SYM_V"],
        sector_map={
            "SYM_Y": "tech",
            "SYM_X": "tech",
            "SYM_W": "tech",
            "SYM_Z": "tech",
            "SYM_V": "tech",
        },
        sector_symbols={"tech": ["SYM_Y", "SYM_X", "SYM_W", "SYM_Z", "SYM_V"]},
    )

    prices_df = _build_synthetic_prices()
    repo = MagicMock()
    repo.get_close_prices.return_value = prices_df

    config = DiscoveryConfig(parallel_n_jobs=1)
    discovery = PairDiscovery(config, repo)

    formation_start = date(2022, 1, 3)
    formation_end = date(2023, 12, 29)

    print("=" * 60)
    print("DISCOVERY LAYER VALIDATION")
    print("=" * 60)
    print(f"Universe: {len(universe.symbols)} symbols")
    print(f"Sector pairs: {len(universe.sector_pairs)} candidates")
    print(f"Formation window: {formation_start} → {formation_end}")
    print(f"Config: coint_pvalue={config.coint_pvalue}, "
          f"adf_pvalue={config.adf_pvalue}")
    print(f"  half_life=[{config.min_half_life_days}, "
          f"{config.max_half_life_days}], max_hurst={config.max_hurst}")
    print(f"  use_ols_fallback={config.use_ols_fallback}")
    print()

    results = discovery.discover(universe, formation_start, formation_end)

    print(f"Qualified pairs: {len(results)}")
    print("-" * 60)

    if not results:
        print("WARNING: No pairs qualified — check gate thresholds")
        sys.exit(1)

    for i, pair in enumerate(results, 1):
        print(f"\n  Pair {i}: {pair.symbol_y} / {pair.symbol_x} "
              f"[{pair.sector}]")
        print(f"    hedge_ratio  = {pair.hedge_ratio:.4f}")
        print(f"    intercept    = {pair.intercept:.4f}")
        print(f"    spread_mean  = {pair.spread_mean:.4f}")
        print(f"    spread_std   = {pair.spread_std:.4f}")
        print(f"    half_life    = {pair.half_life:.1f} days")
        print(f"    coint_pvalue = {pair.coint_pvalue:.6f}")
        print(f"    adf_pvalue   = {pair.adf_pvalue:.6f}")
        print(f"    hurst        = {pair.hurst:.4f}")

        # Sanity checks
        ok = True
        if not (config.min_half_life_days
                <= pair.half_life
                <= config.max_half_life_days):
            print(f"    FAIL: half_life outside "
                  f"[{config.min_half_life_days}, "
                  f"{config.max_half_life_days}]")
            ok = False
        if pair.hurst >= config.max_hurst:
            print(f"    FAIL: hurst {pair.hurst:.4f} >= "
                  f"{config.max_hurst}")
            ok = False
        if pair.coint_pvalue > config.coint_pvalue:
            print(f"    FAIL: coint_pvalue > {config.coint_pvalue}")
            ok = False
        if pair.spread_std <= 0:
            print("    FAIL: spread_std <= 0")
            ok = False

        if ok:
            print("    PASS: all gates satisfied")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
