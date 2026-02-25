"""Main dashboard screen — overview of portfolio, risk, positions, activity, engine.

Engine state is derived entirely from DB polling — no in-process runner reference.
"""

from __future__ import annotations

import subprocess
import sys
from os import devnull
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Static

from stat_arb.config.constants import EngineEventType
from stat_arb.tui.screens.confirm_dialog import ConfirmKillSwitch
from stat_arb.tui.widgets.activity_feed import ActivityFeed
from stat_arb.tui.widgets.kill_switch import KillSwitchActivated, KillSwitchWidget
from stat_arb.tui.widgets.portfolio_summary import PortfolioSummary
from stat_arb.tui.widgets.positions_table import PositionsTable
from stat_arb.tui.widgets.risk_gauge import RiskGauge
from stat_arb.tui.widgets.system_status import SystemStatusWidget

if TYPE_CHECKING:
    from stat_arb.tui.data_provider import DashboardDataProvider


class MainScreen(Screen):
    """Dashboard overview screen (default)."""

    BINDINGS = [
        ("r", "refresh_data", "Refresh"),
        ("s", "start_engine", "Start Engine"),
        ("9", "emergency_stop", "Kill Switch"),
    ]

    def __init__(
        self,
        provider: DashboardDataProvider,
        broker_mode_str: str = "paper",
        config_path: str = "config/default.yaml",
    ) -> None:
        super().__init__()
        self._provider = provider
        self._broker_mode_str = broker_mode_str
        self._config_path = config_path
        self._last_event_id: int | None = None
        self._engine_alive = False
        self._engine_state = "not detected"
        self._engine_pid: int | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="top-row"):
            yield PortfolioSummary(id="portfolio-panel")
            yield RiskGauge(id="risk-panel")

        # Engine status bar — always present
        yield Static(self._engine_bar_text(), id="engine-bar")

        # Live mode warning banner
        if self._broker_mode_str == "live":
            yield Static(
                "[red bold on white] WARNING: LIVE MODE "
                "— REAL MONEY AT RISK [/red bold on white]",
                id="live-warning",
            )

        with Vertical(id="positions-panel"):
            yield Static("[b]ACTIVE POSITIONS[/b]")
            yield PositionsTable()

        with Horizontal(id="bottom-row"):
            yield ActivityFeed(id="activity-panel")
            with Vertical(id="system-kill-panel"):
                yield SystemStatusWidget(id="system-panel")
                yield KillSwitchWidget(
                    engine_active=True,
                    id="kill-switch-panel",
                )
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_all()
        self._load_initial_events()
        self.set_interval(30, self._refresh_all)
        self.set_interval(1, self._poll_engine_events)

    def action_refresh_data(self) -> None:
        self._refresh_all()

    # ------------------------------------------------------------------
    # Start engine
    # ------------------------------------------------------------------

    def action_start_engine(self) -> None:
        """Spawn the engine as a detached background process."""
        feed = self.query_one(ActivityFeed)

        if self._engine_alive:
            feed.add_event("Engine is already running", "warning")
            return

        if self._engine_pid is not None:
            feed.add_event(
                f"Engine already started (PID: {self._engine_pid}), waiting for heartbeat...",
                "warning",
            )
            return

        cmd = [
            sys.executable, "-m", "stat_arb",
            "run-live", "--loop", "--config", self._config_path,
        ]
        try:
            with open(devnull, "w") as dn:
                proc = subprocess.Popen(
                    cmd,
                    stdout=dn,
                    stderr=dn,
                    start_new_session=True,
                )
            self._engine_pid = proc.pid
            feed.add_event(
                f"Engine started (PID: {proc.pid})", "info",
            )
        except Exception as exc:
            feed.add_event(f"Failed to start engine: {exc}", "error")

    # ------------------------------------------------------------------
    # Kill switch flow
    # ------------------------------------------------------------------

    def action_emergency_stop(self) -> None:
        """Keybinding (9) for emergency stop."""
        ks = self.query_one(KillSwitchWidget)
        if ks.is_killed:
            return
        self._show_kill_confirm()

    def on_kill_switch_activated(self, message: KillSwitchActivated) -> None:
        """User pressed EMERGENCY STOP button — show confirmation."""
        self._show_kill_confirm()

    def _show_kill_confirm(self) -> None:
        """Push the kill switch confirmation dialog."""

        def _on_confirm(confirmed: bool) -> None:
            if confirmed:
                self._provider.send_kill_switch()
                self.query_one(ActivityFeed).add_event(
                    "Kill switch command sent", "critical",
                )

        self.app.push_screen(ConfirmKillSwitch(), _on_confirm)

    # ------------------------------------------------------------------
    # DB polling
    # ------------------------------------------------------------------

    def _load_initial_events(self) -> None:
        """Load the most recent events on startup (newest 50, desc)."""
        try:
            events = self._provider.get_recent_events(since_id=None, limit=50)
            feed = self.query_one(ActivityFeed)
            # events come in desc order (newest first) — add in reverse
            # so that the most recent ends up at the top of the feed
            for ev in reversed(events):
                if ev.event_type != EngineEventType.HEARTBEAT:
                    feed.add_event(ev.message, ev.severity)
                self._last_event_id = ev.id
        except Exception:
            pass

    def _poll_engine_events(self) -> None:
        """Poll for new engine events and update the display (runs every 1s)."""
        try:
            # Fetch new events since last seen
            events = self._provider.get_recent_events(
                since_id=self._last_event_id, limit=100,
            )
            feed = self.query_one(ActivityFeed)
            for ev in events:
                if ev.event_type != EngineEventType.HEARTBEAT:
                    feed.add_event(ev.message, ev.severity)
                self._last_event_id = ev.id
        except Exception:
            pass

        # Update engine status from DB
        try:
            status = self._provider.get_engine_status()
            self._engine_alive = status.is_alive
            self._engine_state = status.state if status.is_alive else "not detected"
            if status.is_alive:
                self._engine_pid = None  # clear startup guard
            self.query_one("#engine-bar", Static).update(self._engine_bar_text())
            self.query_one(SystemStatusWidget).set_engine_state(self._engine_state)

            # Update kill switch widget if kill_switch event detected
            if status.state == "killed":
                self.query_one(KillSwitchWidget).set_killed()
        except Exception:
            pass

        # Refresh token/system status
        try:
            token_status = self._provider.get_token_status()
            sys_status = self._provider.get_system_status()
            self.query_one(SystemStatusWidget).update_data(sys_status, token_status)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _engine_bar_text(self) -> str:
        """Build the engine status bar text."""
        mode = self._broker_mode_str.upper()
        if mode == "LIVE":
            mode_markup = f"[red bold]{mode}[/red bold]"
        elif mode == "PAPER":
            mode_markup = f"[green]{mode}[/green]"
        else:
            mode_markup = mode

        state_text = self._engine_state
        if not self._engine_alive:
            state_text += " [dim](press s to start)[/dim]"

        return (
            f"[b]Engine:[/b] {state_text} | "
            f"[b]Mode:[/b] {mode_markup}"
        )

    def _refresh_all(self) -> None:
        snap = self._provider.get_portfolio_snapshot()
        self.query_one(PortfolioSummary).update_data(
            value=snap.value,
            daily_pnl=snap.daily_pnl,
            drawdown_pct=snap.drawdown_pct,
            gross_exposure=snap.gross_exposure,
            active_pairs=snap.active_pairs,
        )

        risk = self._provider.get_risk_utilization()
        self.query_one(RiskGauge).update_data(risk)

        pairs = self._provider.get_active_pairs()
        self.query_one(PositionsTable).update_data(pairs)
