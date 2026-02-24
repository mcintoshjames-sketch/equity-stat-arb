"""Tests for SpreadComputer."""

from __future__ import annotations

from datetime import date

from stat_arb.config.settings import SignalConfig, SizingConfig
from stat_arb.discovery.pair_filter import QualifiedPair
from stat_arb.engine.spread import SpreadComputer


def _make_pair(**overrides: object) -> QualifiedPair:
    defaults = dict(
        symbol_y="AAA",
        symbol_x="BBB",
        sector="tech",
        formation_start=date(2023, 1, 2),
        formation_end=date(2023, 12, 29),
        hedge_ratio=1.2,
        intercept=5.0,
        spread_mean=0.0,
        spread_std=2.0,
        half_life=10.0,
        coint_pvalue=0.01,
        adf_pvalue=0.005,
        hurst=0.35,
    )
    defaults.update(overrides)
    return QualifiedPair(**defaults)  # type: ignore[arg-type]


def test_z_score_at_mean_is_zero() -> None:
    """When live spread equals formation mean, Z-score should be 0."""
    pair = _make_pair(hedge_ratio=1.0, intercept=0.0, spread_mean=10.0, spread_std=2.0)
    sc = SpreadComputer(SignalConfig(), SizingConfig())

    # spread = 110 - 1.0*100 - 0 = 10.0 = spread_mean → z = 0
    z = sc.compute_z_score(pair, {"AAA": 110.0, "BBB": 100.0})
    assert abs(z) < 1e-10


def test_z_score_positive_when_spread_above_mean() -> None:
    """Spread above mean should give positive Z-score."""
    pair = _make_pair(hedge_ratio=1.0, intercept=0.0, spread_mean=0.0, spread_std=5.0)
    sc = SpreadComputer(SignalConfig(), SizingConfig())

    # spread = 120 - 1.0*100 = 20, z = (20-0)/5 = 4.0
    z = sc.compute_z_score(pair, {"AAA": 120.0, "BBB": 100.0})
    assert abs(z - 4.0) < 1e-10


def test_z_score_uses_hedge_ratio_and_intercept() -> None:
    """Z-score must apply β and α correctly."""
    pair = _make_pair(
        hedge_ratio=1.5, intercept=3.0, spread_mean=2.0, spread_std=4.0,
    )
    sc = SpreadComputer(SignalConfig(), SizingConfig())

    # spread = 160 - 1.5*100 - 3.0 = 7.0, z = (7.0 - 2.0) / 4.0 = 1.25
    z = sc.compute_z_score(pair, {"AAA": 160.0, "BBB": 100.0})
    assert abs(z - 1.25) < 1e-10


def test_round_trip_cost_with_quotes() -> None:
    """Round-trip cost should reflect bid-ask width and slippage multiplier."""
    pair = _make_pair()
    cfg = SignalConfig(slippage_multiplier=0.5)
    sizing = SizingConfig(dollars_per_leg=1000.0)
    sc = SpreadComputer(cfg, sizing)

    quotes = {
        "AAA": {"bid": 99.90, "ask": 100.10},  # spread = 0.20
        "BBB": {"bid": 49.95, "ask": 50.05},   # spread = 0.10
    }
    cost = sc.estimate_round_trip_cost(pair, quotes)

    # AAA: half_spread=0.10, slip=0.5*0.10=0.05, shares=1000/100=10, leg=2*0.05*10=1.0
    # BBB: half_spread=0.05, slip=0.5*0.05=0.025, shares=1000/50=20, leg=2*0.025*20=1.0
    assert abs(cost - 2.0) < 0.01


def test_round_trip_cost_missing_quotes() -> None:
    """Missing bid/ask should return 0."""
    pair = _make_pair()
    sc = SpreadComputer(SignalConfig(), SizingConfig())
    cost = sc.estimate_round_trip_cost(pair, {"AAA": {"last": 100.0}})
    assert cost == 0.0


def test_adaptive_vol_disabled_uses_formation_std() -> None:
    """With adaptive_vol=False (default), Z uses frozen formation σ."""
    pair = _make_pair(hedge_ratio=1.0, intercept=0.0, spread_mean=0.0, spread_std=5.0)
    cfg = SignalConfig(adaptive_vol=False)
    sc = SpreadComputer(cfg, SizingConfig())

    prices = {"AAA": 120.0, "BBB": 100.0}
    # Feed many observations — should NOT affect Z-score
    for _ in range(30):
        z = sc.compute_z_score(pair, prices)

    # spread = 120-100 = 20, z = 20/5 = 4.0 (always uses formation std)
    assert abs(z - 4.0) < 1e-10


def test_adaptive_vol_uses_rolling_std() -> None:
    """With adaptive_vol=True and full window, Z uses rolling σ."""
    pair = _make_pair(hedge_ratio=1.0, intercept=0.0, spread_mean=0.0, spread_std=2.0)
    cfg = SignalConfig(adaptive_vol=True, adaptive_vol_window=5)
    sc = SpreadComputer(cfg, SizingConfig())

    # Feed 5 observations with known spreads to fill the window
    # spread = price_y - 1.0*price_x - 0 = price_y - price_x
    spreads = [10.0, 12.0, 14.0, 16.0, 18.0]
    for s in spreads[:-1]:
        sc.compute_z_score(pair, {"AAA": 100.0 + s, "BBB": 100.0})

    # 5th observation fills the window → rolling std kicks in
    z = sc.compute_z_score(pair, {"AAA": 100.0 + spreads[-1], "BBB": 100.0})

    # rolling std of [10, 12, 14, 16, 18] with ddof=1 ≈ 3.162
    # z = (18 - 0) / 3.162 ≈ 5.692 (vs 18/2 = 9.0 with formation std)
    assert z < 7.0  # much less than 9.0 (formation-std Z)
    assert z > 4.0  # but still large because spread is far from mean


def test_adaptive_vol_floor_prevents_explosion() -> None:
    """Rolling σ near zero should be floored at 50% of formation σ."""
    pair = _make_pair(hedge_ratio=1.0, intercept=0.0, spread_mean=0.0, spread_std=4.0)
    cfg = SignalConfig(adaptive_vol=True, adaptive_vol_window=5)
    sc = SpreadComputer(cfg, SizingConfig())

    # Feed 5 identical spreads → rolling std = 0
    for _ in range(5):
        sc.compute_z_score(pair, {"AAA": 110.0, "BBB": 100.0})

    # spread = 10.0, rolling_std = 0 → floored to 4.0 * 0.5 = 2.0
    # z = (10 - 0) / 2.0 = 5.0
    z = sc.compute_z_score(pair, {"AAA": 110.0, "BBB": 100.0})
    assert abs(z - 5.0) < 0.1
