"""Click CLI for the stat-arb pairs trading system.

Two commands:
  stat-arb run-backtest  — Walk-forward backtest over historical data.
  stat-arb run-live      — Execute one trading step (or loop) with a real/paper broker.

Both commands share the same dependency-wiring pattern: load config,
init DB, build StatArbEngine + RiskManager, then hand off to the
appropriate runner.
"""

from __future__ import annotations

import sys
from datetime import date

import click

# Ensure src/ is importable when running from the repo root.
sys.path.insert(0, "src")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_core(cfg):
    """Wire up the core engine components shared by backtest and live.

    Returns:
        (engine, walk_forward, sizer, risk_manager, price_repo, universe, schwab_client)
    """
    from stat_arb.data.db import create_tables, init_db
    from stat_arb.data.price_repo import PriceRepository
    from stat_arb.data.schwab_client import SchwabDataClient
    from stat_arb.data.universe import load_universe
    from stat_arb.discovery.pair_discovery import PairDiscovery
    from stat_arb.engine.signals import SignalGenerator
    from stat_arb.engine.spread import SpreadComputer
    from stat_arb.engine.stat_arb_engine import StatArbEngine
    from stat_arb.engine.walk_forward import WalkForwardScheduler
    from stat_arb.execution.sizing import PositionSizer
    from stat_arb.logging_config import setup_logging
    from stat_arb.risk.risk_manager import RiskManager

    setup_logging(cfg.logging)
    init_db(cfg.database)
    create_tables()

    universe = load_universe(cfg.universe)

    schwab_client: SchwabDataClient | None = None
    try:
        schwab_client = SchwabDataClient(cfg.schwab)
    except (ValueError, Exception) as exc:  # noqa: BLE001
        click.echo(f"WARNING: Schwab client init failed ({exc}); running DB-only.")
        click.echo("  Price data must already exist in the database.")
        click.echo()

    price_repo = PriceRepository(schwab_client=schwab_client)
    pair_discovery = PairDiscovery(cfg.discovery, price_repo)
    spread_computer = SpreadComputer(cfg.signal, cfg.sizing)
    signal_generator = SignalGenerator(cfg.signal)
    walk_forward = WalkForwardScheduler(cfg.walk_forward)
    sizer = PositionSizer(cfg.sizing)
    risk_manager = RiskManager(cfg.risk)

    engine = StatArbEngine(
        signal_config=cfg.signal,
        sizing_config=cfg.sizing,
        spread_computer=spread_computer,
        signal_generator=signal_generator,
        walk_forward=walk_forward,
        pair_discovery=pair_discovery,
        universe=universe,
    )

    return engine, walk_forward, sizer, risk_manager, price_repo, universe, schwab_client


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="stat-arb")
def cli() -> None:
    """Equity stat-arb pairs trading system."""


# ---------------------------------------------------------------------------
# run-backtest
# ---------------------------------------------------------------------------


@cli.command("run-backtest")
@click.option(
    "--config", "config_path",
    default="config/default.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
@click.option(
    "--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Backtest start date (YYYY-MM-DD).",
)
@click.option(
    "--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Backtest end date (YYYY-MM-DD).",
)
@click.option("--persist", is_flag=True, help="Save results to the database.")
def run_backtest(config_path: str, start, end, persist: bool) -> None:
    """Run a walk-forward backtest over historical data."""
    from stat_arb.backtest.sim_broker import SimBroker
    from stat_arb.backtest.walk_forward_bt import WalkForwardBacktest
    from stat_arb.config.settings import load_config

    start_date: date = start.date()
    end_date: date = end.date()

    cfg = load_config(config_path)
    engine, walk_forward, sizer, risk_manager, price_repo, universe, _ = _build_core(cfg)

    windows = walk_forward.generate_windows(start_date, end_date)
    if not windows:
        click.echo("ERROR: No walk-forward windows fit in the date range.")
        raise SystemExit(1)
    click.echo(f"Generated {len(windows)} walk-forward window(s)")

    sim_broker = SimBroker(slippage_bps=10.0)

    backtest = WalkForwardBacktest(
        engine=engine,
        price_repo=price_repo,
        risk_manager=risk_manager,
        sizer=sizer,
        sim_broker=sim_broker,
        universe=universe,
    )

    result = backtest.run(start_date, end_date)

    # --- Print summary ---
    click.echo()
    click.echo("=" * 60)
    click.echo("BACKTEST RESULTS")
    click.echo("=" * 60)
    click.echo(f"  Period:       {start_date} -> {end_date}")
    click.echo(f"  Total Return: {result.total_return:+.2%}")
    click.echo(f"  Sharpe Ratio: {result.sharpe:.3f}")
    click.echo(f"  Max Drawdown: {result.max_drawdown:.2%}")
    click.echo(f"  Total Trades: {result.total_trades}")
    click.echo(f"  Win Rate:     {result.win_rate:.1%}")
    if result.equity_curve:
        click.echo(f"  Final Equity: ${result.equity_curve[-1]:,.2f}")
    click.echo()

    if result.trades:
        sorted_trades = sorted(result.trades, key=lambda t: abs(t.pnl), reverse=True)
        top = sorted_trades[:5]
        click.echo("Top trades by |P&L|:")
        click.echo(f"  {'Pair':<16} {'Signal':<14} {'Entry':<12} {'Exit':<12} {'P&L':>10}")
        click.echo("  " + "-" * 64)
        for t in top:
            exit_dt = str(t.exit_date) if t.exit_date else "open"
            click.echo(
                f"  {t.pair_key:<16} {t.signal.value:<14} "
                f"{t.entry_date!s:<12} {exit_dt:<12} {t.pnl:>+10.2f}"
            )
        click.echo()

    if persist:
        _persist_backtest(cfg, start_date, end_date, result)


def _persist_backtest(cfg, start_date: date, end_date: date, result) -> None:
    """Save backtest results to the database."""
    from stat_arb.data.db import get_session
    from stat_arb.data.schemas import BacktestRun

    session = get_session()
    try:
        run = BacktestRun(
            start_date=start_date,
            end_date=end_date,
            config_json=cfg.model_dump_json(),
            total_return=result.total_return,
            sharpe=result.sharpe,
            max_drawdown=result.max_drawdown,
            total_trades=result.total_trades,
        )
        session.add(run)
        session.commit()
        click.echo(f"Persisted BacktestRun id={run.id}")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# run-live
# ---------------------------------------------------------------------------


@cli.command("run-live")
@click.option(
    "--config", "config_path",
    default="config/default.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
@click.option(
    "--loop", is_flag=True, default=False,
    help="Run continuously, sleeping until next market close between steps.",
)
@click.option(
    "--broker-mode", "broker_override",
    type=click.Choice(["paper", "live"], case_sensitive=False),
    default=None,
    help="Override broker_mode from config. Defaults to config value (paper).",
)
def run_live(config_path: str, loop: bool, broker_override: str | None) -> None:
    """Execute live trading steps (default: single step, then exit).

    Defaults to paper trading mode for safety.  Pass --broker-mode=live
    to use the real Schwab broker (requires valid API credentials).
    """
    from datetime import timedelta

    from stat_arb.config.constants import BrokerMode
    from stat_arb.config.settings import load_config
    from stat_arb.live.runner import LiveRunner

    cfg = load_config(config_path)
    engine, walk_forward, sizer, risk_manager, price_repo, universe, schwab_client = (
        _build_core(cfg)
    )

    # Generate walk-forward windows covering ~2 years ending today
    today = date.today()
    wf_start = today - timedelta(days=730)
    windows = walk_forward.generate_windows(wf_start, today)
    if not windows:
        click.echo("WARNING: No walk-forward windows generated — discovery won't run.")

    broker_mode = BrokerMode(broker_override) if broker_override else cfg.broker_mode

    # Safety guard: refuse live mode without explicit --broker-mode=live
    if broker_mode == BrokerMode.LIVE and broker_override != "live":
        click.echo(
            "ERROR: Config has broker_mode=live but --broker-mode=live was not "
            "explicitly passed on the command line. This is a safety guard to "
            "prevent accidental real trades.",
        )
        raise SystemExit(1)

    runner = LiveRunner(
        engine=engine,
        sizer=sizer,
        risk_manager=risk_manager,
        price_repo=price_repo,
        universe=universe,
        schwab_client=schwab_client,
        broker_mode=broker_mode,
        signal_config=cfg.signal,
    )

    if loop:
        runner.run_loop()
    else:
        runner.run_once()
