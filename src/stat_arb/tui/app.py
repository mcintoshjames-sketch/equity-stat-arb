"""StatArbDashboard — main Textual application with screen switching.

The dashboard is a passive DB monitor.  The engine runs independently
(via ``stat-arb run-live --loop``) and communicates through DB tables.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App

from stat_arb.tui.screens.help_screen import HelpScreen
from stat_arb.tui.screens.main_screen import MainScreen
from stat_arb.tui.screens.pairs_screen import PairsScreen
from stat_arb.tui.screens.token_screen import TokenScreen

if TYPE_CHECKING:
    from stat_arb.data.schwab_client import SchwabDataClient
    from stat_arb.tui.data_provider import DashboardDataProvider

_CSS_PATH = Path(__file__).parent / "styles" / "dashboard.tcss"

logger = logging.getLogger(__name__)


class StatArbDashboard(App):
    """Textual TUI dashboard for the stat-arb system."""

    TITLE = "Stat-Arb Dashboard"
    CSS_PATH = _CSS_PATH

    BINDINGS = [
        ("1", "screen_main", "Dashboard"),
        ("2", "screen_pairs", "Pairs"),
        ("3", "screen_tokens", "Tokens"),
        ("question_mark", "screen_help", "Help"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        provider: DashboardDataProvider,
        schwab_client: SchwabDataClient | None = None,
        callback_url: str = "https://127.0.0.1",
        app_key: str = "",
        broker_mode_str: str = "paper",
    ) -> None:
        super().__init__()
        self._provider = provider
        self._schwab_client = schwab_client
        self._callback_url = callback_url
        self._app_key = app_key
        self._broker_mode_str = broker_mode_str

    def on_mount(self) -> None:
        self.install_screen(
            MainScreen(
                self._provider,
                broker_mode_str=self._broker_mode_str,
            ),
            name="main",
        )
        self.install_screen(
            PairsScreen(self._provider), name="pairs",
        )
        self.install_screen(
            TokenScreen(
                self._provider,
                schwab_client=self._schwab_client,
                callback_url=self._callback_url,
                app_key=self._app_key,
            ),
            name="tokens",
        )
        self.install_screen(HelpScreen(), name="help")
        self.push_screen("main")

    # ------------------------------------------------------------------
    # Screen switching
    # ------------------------------------------------------------------

    def action_screen_main(self) -> None:
        self.switch_screen("main")

    def action_screen_pairs(self) -> None:
        self.switch_screen("pairs")

    def action_screen_tokens(self) -> None:
        self.switch_screen("tokens")

    def action_screen_help(self) -> None:
        self.switch_screen("help")

    def action_quit(self) -> None:
        self.exit()
