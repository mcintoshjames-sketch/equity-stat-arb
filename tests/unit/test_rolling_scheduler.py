"""Tests for the RollingScheduler."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from stat_arb.config.settings import DiscoveryConfig, RollingSchedulerConfig
from stat_arb.discovery.pair_filter import QualifiedPair
from stat_arb.engine.rolling_scheduler import RollingScheduler


def _make_pair(sym_y: str = "AAA", sym_x: str = "BBB", **overrides) -> QualifiedPair:
    defaults = dict(
        symbol_y=sym_y,
        symbol_x=sym_x,
        sector="tech",
        formation_start=date(2023, 1, 2),
        formation_end=date(2023, 12, 29),
        hedge_ratio=1.0,
        intercept=0.0,
        spread_mean=0.0,
        spread_std=2.0,
        half_life=10.0,
        coint_pvalue=0.01,
        adf_pvalue=0.005,
        hurst=0.35,
    )
    defaults.update(overrides)
    return QualifiedPair(**defaults)  # type: ignore[arg-type]


def _build_scheduler(
    discovery_interval: int = 21,
    trading_days: int = 63,
    max_cohort_pairs: int = 5,
    discovered_pairs: list[QualifiedPair] | None = None,
) -> RollingScheduler:
    """Build a RollingScheduler with a mocked discovery pipeline."""
    config = RollingSchedulerConfig(
        formation_days=60,
        trading_days=trading_days,
        discovery_interval_days=discovery_interval,
        max_cohort_pairs=max_cohort_pairs,
    )
    discovery_config = DiscoveryConfig(formation_days=60)
    mock_discovery = MagicMock()
    mock_discovery.discover.return_value = discovered_pairs or []
    mock_universe = MagicMock()
    mock_price_repo = MagicMock()

    scheduler = RollingScheduler(
        config=config,
        discovery_config=discovery_config,
        pair_discovery=mock_discovery,
        universe=mock_universe,
        price_repo=mock_price_repo,
    )
    return scheduler


def test_initial_discovery_runs_first_step() -> None:
    """First step() should trigger discovery since no last_discovery exists."""
    pair = _make_pair()
    scheduler = _build_scheduler(discovered_pairs=[pair])

    ran = scheduler.step(date(2024, 6, 3))
    assert ran is True
    assert len(scheduler.active_pairs) == 1
    assert scheduler.active_pairs[0].cohort_id is not None
    assert scheduler.active_pairs[0].discovery_date == date(2024, 6, 3)


def test_respects_interval() -> None:
    """Discovery should not re-trigger before the interval has elapsed."""
    pair = _make_pair()
    scheduler = _build_scheduler(
        discovery_interval=21,
        discovered_pairs=[pair],
    )

    # First step: runs discovery
    scheduler.step(date(2024, 6, 3))
    assert len(scheduler.active_pairs) == 1

    # Next day: should NOT run discovery (only 1 bday elapsed)
    ran = scheduler.step(date(2024, 6, 4))
    assert ran is False

    # 10 bdays later: still shouldn't run
    ran = scheduler.step(date(2024, 6, 17))
    assert ran is False


def test_discovery_runs_after_interval() -> None:
    """Discovery should run again after the interval elapses."""
    pair = _make_pair()
    scheduler = _build_scheduler(
        discovery_interval=5,
        trading_days=63,
        discovered_pairs=[pair],
    )

    # First step
    scheduler.step(date(2024, 6, 3))

    # 5 bdays later (Mon Jun 10)
    ran = scheduler.step(date(2024, 6, 10))
    assert ran is True


def test_pair_expiry() -> None:
    """Pairs should be removed after trading_expiry and appear in expired_keys."""
    pair = _make_pair()
    scheduler = _build_scheduler(
        trading_days=5,
        discovery_interval=5,
        discovered_pairs=[pair],
    )

    # Trigger first discovery
    scheduler.step(date(2024, 6, 3))
    assert len(scheduler.active_pairs) == 1

    # Get the expiry date from the enriched pair
    active = scheduler.active_pairs[0]
    expiry = active.trading_expiry

    # Now clear discovery return so future discoveries don't re-add the pair
    scheduler._discovery.discover.return_value = []

    # Step on the expiry date: pair still active (expiry is checked as >)
    scheduler.step(expiry)
    # Pair may or may not still be here depending on discovery re-run,
    # but we cleared the mock, so no new pairs are added.
    # _expire_pairs runs first (not expired yet since == not >),
    # then discovery runs but returns empty.
    assert len(scheduler.active_pairs) == 1

    # Step one day after expiry — pair should be expired
    next_bday = pd.bdate_range(start=expiry, periods=2)[1].date()
    scheduler.step(next_bday)

    assert len(scheduler.active_pairs) == 0
    assert len(scheduler.expired_keys) == 1
    assert scheduler.expired_keys[0] == ("AAA", "BBB")


def test_re_discovered_pair_refreshes_params() -> None:
    """Same pair getting re-discovered should refresh params and appear in refreshed_keys."""
    pair_v1 = _make_pair(hedge_ratio=1.0)
    pair_v2 = _make_pair(hedge_ratio=1.5)

    scheduler = _build_scheduler(
        discovery_interval=5,
        trading_days=63,
    )

    # First discovery with v1
    scheduler._discovery.discover.return_value = [pair_v1]
    scheduler.step(date(2024, 6, 3))
    assert scheduler.active_pairs[0].hedge_ratio == 1.0

    # Second discovery with v2
    scheduler._discovery.discover.return_value = [pair_v2]
    scheduler.step(date(2024, 6, 10))

    # Should have refreshed params
    assert scheduler.active_pairs[0].hedge_ratio == 1.5
    assert ("AAA", "BBB") in scheduler.refreshed_keys
    assert len(scheduler.active_pairs) == 1  # not duplicated


def test_max_cohort_pairs() -> None:
    """Should cap new pairs per discovery cycle at max_cohort_pairs."""
    pairs = [
        _make_pair(sym_y=f"Y{i}", sym_x=f"X{i}")
        for i in range(10)
    ]
    scheduler = _build_scheduler(
        max_cohort_pairs=3,
        discovered_pairs=pairs,
    )

    scheduler.step(date(2024, 6, 3))
    assert len(scheduler.active_pairs) == 3


def test_expired_pair_available_for_rebalancer() -> None:
    """get_prev_pair() should return old params for expired pairs."""
    pair = _make_pair(hedge_ratio=1.2)
    scheduler = _build_scheduler(
        trading_days=5,
        discovery_interval=5,
        discovered_pairs=[pair],
    )

    scheduler.step(date(2024, 6, 3))

    # Force expiry
    active = scheduler.active_pairs[0]
    expiry = active.trading_expiry
    next_bday = pd.bdate_range(start=expiry, periods=2)[1].date()
    scheduler._last_discovery = next_bday
    scheduler.step(next_bday)

    prev = scheduler.get_prev_pair(("AAA", "BBB"))
    assert prev is not None
    assert prev.hedge_ratio == 1.2
