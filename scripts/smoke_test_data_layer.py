#!/usr/bin/env python3
"""Smoke test for the data layer: Schwab API + DB round-trip.

Validates that SchwabDataClient, PriceRepository, and the SQLite DB
all work correctly with real-world API payloads.  Read-only — no orders
or trading activity.

Prerequisites:
    1. Set your Schwab app_key / app_secret in config/default.yaml
       (or ensure ~/.schwabdev/tokens.db has valid tokens).
    2. pip install -e ".[dev]"

Usage:
    python scripts/smoke_test_data_layer.py
"""

from __future__ import annotations

import sys
import traceback
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all smoke tests sequentially, printing pass/fail for each."""
    from stat_arb.config.settings import load_config
    from stat_arb.data.db import create_tables, init_db
    from stat_arb.data.price_repo import PriceRepository
    from stat_arb.data.schwab_client import SchwabDataClient
    from stat_arb.logging_config import setup_logging

    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    print(f"Loading config from: {config_path}")
    cfg = load_config(config_path)
    setup_logging(cfg.logging)

    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # Test 1: Schwab client creation (OAuth2 token refresh)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 1: SchwabDataClient creation (OAuth2 token check)")
    print("=" * 60)
    try:
        client = SchwabDataClient(cfg.schwab)
        account_hash = client.account_hash
        print(f"  Account hash: {account_hash[:8]}...")
        print("  \u2705 Client created and authenticated successfully")
        passed += 1
    except Exception as e:
        print(f"  \u274c FAILED: {e}")
        traceback.print_exc()
        failed += 1
        print("\n  Cannot proceed without a valid Schwab connection.")
        print(f"\nResults: {passed} passed, {failed} failed")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Test 2: fetch_price_history — 2 years of SPY daily bars
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 2: fetch_price_history('SPY', 2 years daily)")
    print("=" * 60)
    try:
        df_hist = client.fetch_price_history("SPY", period_type="year", period=2)
        assert not df_hist.empty, "DataFrame is empty — no candle data returned"
        print(f"  Rows: {len(df_hist)}")
        print(f"  Date range: {df_hist.index[0]} \u2192 {df_hist.index[-1]}")
        print(f"  Columns: {list(df_hist.columns)}")
        print("\n  Latest 3 bars:")
        print(df_hist.tail(3).to_string(max_cols=10))
        print("  \u2705 Price history fetched successfully")
        passed += 1
    except Exception as e:
        print(f"  \u274c FAILED: {e}")
        traceback.print_exc()
        failed += 1

    # ------------------------------------------------------------------
    # Test 3: fetch_batch_quotes — SPY, QQQ, AAPL
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 3: fetch_batch_quotes(['SPY', 'QQQ', 'AAPL'])")
    print("=" * 60)
    try:
        quotes = client.fetch_batch_quotes(["SPY", "QQQ", "AAPL"])
        assert len(quotes) > 0, "No quotes returned"
        for sym, q in quotes.items():
            bid = q.get("bidPrice", "N/A")
            ask = q.get("askPrice", "N/A")
            last = q.get("lastPrice", "N/A")
            mid = "N/A"
            if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
                mid = f"{(bid + ask) / 2:.2f}"
            print(f"  {sym}: bid={bid}  ask={ask}  last={last}  mid={mid}")
        print("  \u2705 Batch quotes fetched successfully")
        passed += 1
    except Exception as e:
        print(f"  \u274c FAILED: {e}")
        traceback.print_exc()
        failed += 1

    # ------------------------------------------------------------------
    # Test 4: DB init + backfill SPY via PriceRepository
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 4: DB init + PriceRepository backfill (SPY)")
    print("=" * 60)
    try:
        init_db(cfg.database)
        create_tables()
        print(f"  DB initialised: {cfg.database.url}")

        repo = PriceRepository(schwab_client=client)

        # Trigger backfill by requesting data for a symbol not yet in DB.
        # get_close_prices auto-backfills missing symbols via _backfill_symbol.
        df_prices = repo.get_close_prices(
            ["SPY"],
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
        )
        assert not df_prices.empty, "get_close_prices returned empty after backfill"
        print(f"  Rows returned: {len(df_prices)}")
        print(f"  Columns: {list(df_prices.columns)}")
        print("  \u2705 Backfill + DB round-trip successful")
        passed += 1
    except Exception as e:
        print(f"  \u274c FAILED: {e}")
        traceback.print_exc()
        failed += 1

    # ------------------------------------------------------------------
    # Test 5: get_date_range — verify DB coverage
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 5: get_date_range('SPY')")
    print("=" * 60)
    try:
        date_range = repo.get_date_range("SPY")
        assert date_range is not None, "No data found for SPY after backfill"
        min_date, max_date = date_range
        print(f"  Min date: {min_date}")
        print(f"  Max date: {max_date}")
        print(f"  Span: {(max_date - min_date).days} calendar days")
        print("  \u2705 Date range query successful")
        passed += 1
    except Exception as e:
        print(f"  \u274c FAILED: {e}")
        traceback.print_exc()
        failed += 1

    # ------------------------------------------------------------------
    # Test 6: get_close_prices — query from DB (no API call)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 6: get_close_prices('SPY', 2025-01-01 to 2025-02-24)")
    print("=" * 60)
    try:
        df_close = repo.get_close_prices(
            ["SPY"],
            start=date(2025, 1, 1),
            end=date(2025, 2, 24),
        )
        print(f"  Shape: {df_close.shape}")
        if not df_close.empty:
            print(f"  Date range: {df_close.index[0]} \u2192 {df_close.index[-1]}")
            print("\n  First 5 rows:")
            print(df_close.head().to_string())
            print("\n  Last 5 rows:")
            print(df_close.tail().to_string())
        else:
            print("  (empty — SPY data may not cover this date range yet)")
        print("  \u2705 Close price query successful")
        passed += 1
    except Exception as e:
        print(f"  \u274c FAILED: {e}")
        traceback.print_exc()
        failed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"\u2705 ALL {total} TESTS PASSED — data layer is ready for Step 4")
    else:
        print(f"\u274c {passed}/{total} passed, {failed} FAILED")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
