#!/usr/bin/env python3
"""Walk-forward backtest runner.

Wires up all dependencies from a YAML config and replays historical data
through the stat-arb engine, producing performance metrics and optionally
persisting results to the database.

Usage:
    python scripts/run_backtest.py --config config/default.yaml \
                                    --start 2023-01-01 \
                                    --end 2024-12-31 \
                                    --persist
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

# Ensure src/ is on the path
sys.path.insert(0, "src")

from stat_arb.backtest.sim_broker import SimBroker  # noqa: E402
from stat_arb.backtest.walk_forward_bt import WalkForwardBacktest  # noqa: E402
from stat_arb.config.settings import load_config  # noqa: E402
from stat_arb.data.db import create_tables, get_session, init_db  # noqa: E402
from stat_arb.data.price_repo import PriceRepository  # noqa: E402
from stat_arb.data.schemas import BacktestRun  # noqa: E402
from stat_arb.data.schwab_client import SchwabDataClient  # noqa: E402
from stat_arb.data.universe import load_universe  # noqa: E402
from stat_arb.discovery.pair_discovery import PairDiscovery  # noqa: E402
from stat_arb.engine.signals import SignalGenerator  # noqa: E402
from stat_arb.engine.spread import SpreadComputer  # noqa: E402
from stat_arb.engine.stat_arb_engine import StatArbEngine  # noqa: E402
from stat_arb.engine.walk_forward import WalkForwardScheduler  # noqa: E402
from stat_arb.execution.sizing import PositionSizer  # noqa: E402
from stat_arb.logging_config import setup_logging  # noqa: E402
from stat_arb.risk.risk_manager import RiskManager  # noqa: E402


def _parse_date(value: str) -> date:
    """Parse an ISO-format date string."""
    return date.fromisoformat(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a walk-forward backtest",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        required=True,
        help="Backtest start date (ISO format, e.g. 2023-01-01)",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        required=True,
        help="Backtest end date (ISO format, e.g. 2024-12-31)",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Save BacktestResult to the database",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 1. Load config
    cfg = load_config(args.config)

    # 2. Logging
    setup_logging(cfg.logging)

    # 3. Database
    init_db(cfg.database)
    create_tables()

    # 4. Universe
    universe = load_universe(cfg.universe)

    # 5. Price repository (Schwab client optional — falls back to DB-only)
    schwab_client: SchwabDataClient | None = None
    try:
        schwab_client = SchwabDataClient(cfg.schwab)
    except (ValueError, Exception) as exc:  # noqa: BLE001
        print(f"WARNING: Schwab client init failed ({exc}); running DB-only.")
        print("  Price data must already exist in the database.")
        print("  Update schwab.app_key / app_secret in your config to enable backfill.")
        print()
    price_repo = PriceRepository(schwab_client=schwab_client)

    # 6. Discovery
    pair_discovery = PairDiscovery(cfg.discovery, price_repo)

    # 7. Spread computer
    spread_computer = SpreadComputer(cfg.signal, cfg.sizing)

    # 8. Signal generator
    signal_generator = SignalGenerator(cfg.signal)

    # 9. Walk-forward scheduler + windows
    walk_forward = WalkForwardScheduler(cfg.walk_forward)
    windows = walk_forward.generate_windows(args.start, args.end)
    if not windows:
        print("ERROR: No walk-forward windows fit in the date range.")
        sys.exit(1)

    # 10. Engine
    engine = StatArbEngine(
        signal_config=cfg.signal,
        sizing_config=cfg.sizing,
        spread_computer=spread_computer,
        signal_generator=signal_generator,
        walk_forward=walk_forward,
        pair_discovery=pair_discovery,
        universe=universe,
    )

    # 11. Execution & risk
    sizer = PositionSizer(cfg.sizing)
    risk_manager = RiskManager(cfg.risk)
    sim_broker = SimBroker(slippage_bps=10.0)

    # 12. Backtest
    backtest = WalkForwardBacktest(
        engine=engine,
        price_repo=price_repo,
        risk_manager=risk_manager,
        sizer=sizer,
        sim_broker=sim_broker,
        universe=universe,
    )

    # 13. Run
    result = backtest.run(args.start, args.end)

    # --- Print summary ---
    print()
    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Period:       {args.start} → {args.end}")
    print(f"  Windows:      {len(windows)}")
    print(f"  Total Return: {result.total_return:+.2%}")
    print(f"  Sharpe Ratio: {result.sharpe:.3f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate:     {result.win_rate:.1%}")
    if result.equity_curve:
        print(f"  Final Equity: ${result.equity_curve[-1]:,.2f}")
    print()

    # Top 5 trades by |P&L|
    if result.trades:
        sorted_trades = sorted(result.trades, key=lambda t: abs(t.pnl), reverse=True)
        top = sorted_trades[:5]
        print("Top trades by |P&L|:")
        print(f"  {'Pair':<16} {'Signal':<14} {'Entry':<12} {'Exit':<12} {'P&L':>10}")
        print("  " + "-" * 64)
        for t in top:
            exit_dt = str(t.exit_date) if t.exit_date else "open"
            print(
                f"  {t.pair_key:<16} {t.signal.value:<14} "
                f"{t.entry_date!s:<12} {exit_dt:<12} {t.pnl:>+10.2f}"
            )
        print()

    # --- Persist ---
    if args.persist:
        session = get_session()
        try:
            run = BacktestRun(
                start_date=args.start,
                end_date=args.end,
                config_json=cfg.model_dump_json(),
                total_return=result.total_return,
                sharpe=result.sharpe,
                max_drawdown=result.max_drawdown,
                total_trades=result.total_trades,
            )
            session.add(run)
            session.commit()
            print(f"Persisted BacktestRun id={run.id}")
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


if __name__ == "__main__":
    main()
