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
