"""Token management screen — expiry countdown + re-auth flow."""

from __future__ import annotations

import urllib.parse
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Static
from textual.worker import Worker, WorkerState

from stat_arb.tui.data_provider import ACCESS_TOKEN_LIFETIME, REFRESH_TOKEN_LIFETIME
from stat_arb.tui.widgets.token_status import _color_for_fraction, _format_remaining

if TYPE_CHECKING:
    from stat_arb.data.schwab_client import SchwabDataClient
    from stat_arb.tui.data_provider import DashboardDataProvider


def _progress_bar(frac: float, width: int = 30) -> str:
    """Render a colored progress bar."""
    filled = int(frac * width)
    color = _color_for_fraction(frac)
    bar = "=" * filled + "-" * (width - filled)
    return f"[{color}][{bar}][/{color}]"


class TokenScreen(Screen):
    """Token management with countdown and re-auth."""

    BINDINGS = [
        ("r", "refresh_data", "Refresh"),
    ]

    def __init__(
        self,
        provider: DashboardDataProvider,
        schwab_client: SchwabDataClient | None = None,
        callback_url: str = "https://127.0.0.1",
        app_key: str = "",
    ) -> None:
        super().__init__()
        self._provider = provider
        self._schwab_client = schwab_client
        self._callback_url = callback_url
        self._app_key = app_key

    def compose(self) -> ComposeResult:
        yield Static(id="token-detail-panel")
        with Vertical(id="reauth-panel"):
            yield Static("[b]RE-AUTHENTICATION[/b]", id="reauth-title")
            yield Button("Start Re-Auth", id="start-reauth")
            yield Static("", id="auth-url-display")
            yield Input(placeholder="Paste callback URL here...", id="callback-input")
            yield Button("Submit", id="submit-callback", disabled=True)
            yield Static("", id="reauth-status")
        yield Footer()

    def on_mount(self) -> None:
        if not self._app_key:
            self.query_one("#start-reauth", Button).disabled = True
            self.query_one("#reauth-status", Static).update(
                "[dim]Re-auth unavailable — no Schwab credentials configured.\n"
                "Set SCHWAB_APP_KEY and SCHWAB_APP_SECRET env vars or update config.[/dim]"
            )
        self._refresh_tokens()
        self.set_interval(1, self._refresh_tokens)

    def action_refresh_data(self) -> None:
        self._refresh_tokens()

    def _refresh_tokens(self) -> None:
        status = self._provider.get_token_status()
        panel = self.query_one("#token-detail-panel", Static)

        if status is None:
            panel.update("[b]TOKEN STATUS[/b]\n\n  No Schwab connection available")
            return

        access_frac = status.access_remaining_s / ACCESS_TOKEN_LIFETIME
        refresh_frac = status.refresh_remaining_s / REFRESH_TOKEN_LIFETIME

        access_bar = _progress_bar(access_frac)
        refresh_bar = _progress_bar(refresh_frac)

        issued_a = status.access_issued.strftime("%H:%M:%S") if status.access_issued else "—"
        issued_r = (
            status.refresh_issued.strftime("%Y-%m-%d %H:%M") if status.refresh_issued else "—"
        )

        lines = [
            "[b]TOKEN STATUS[/b]",
            "",
            f"  Access Token   Issued: {issued_a}",
            f"    Remaining: {_format_remaining(status.access_remaining_s)}",
            f"    {access_bar}",
            "",
            f"  Refresh Token  Issued: {issued_r}",
            f"    Remaining: {_format_remaining(status.refresh_remaining_s)}",
            f"    {refresh_bar}",
        ]
        panel.update("\n".join(lines))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-reauth":
            self._start_reauth()
        elif event.button.id == "submit-callback":
            self._submit_callback()

    def _start_reauth(self) -> None:
        if not self._app_key:
            self.query_one("#reauth-status", Static).update(
                "[red]No app_key configured[/red]"
            )
            return

        auth_url = (
            f"https://api.schwabapi.com/v1/oauth/authorize"
            f"?client_id={self._app_key}"
            f"&redirect_uri={urllib.parse.quote(self._callback_url)}"
        )
        self.query_one("#auth-url-display", Static).update(
            f"[b]Open this URL in your browser:[/b]\n\n{auth_url}\n\n"
            "After authorizing, paste the callback URL below."
        )
        self.query_one("#submit-callback", Button).disabled = False
        self.query_one("#callback-input", Input).focus()

    def _submit_callback(self) -> None:
        callback_value = self.query_one("#callback-input", Input).value.strip()
        if not callback_value:
            self.query_one("#reauth-status", Static).update(
                "[red]Please paste the callback URL[/red]"
            )
            return

        # Extract auth code from callback URL
        try:
            parsed = urllib.parse.urlparse(callback_value)
            params = urllib.parse.parse_qs(parsed.query)
            code = params.get("code", [None])[0]
            if not code:
                raise ValueError("No 'code' parameter in URL")
        except Exception as exc:
            self.query_one("#reauth-status", Static).update(
                f"[red]Failed to parse callback URL: {exc}[/red]"
            )
            return

        self.query_one("#reauth-status", Static).update("Exchanging auth code...")
        self._exchange_code(code)

    def _exchange_code(self, code: str) -> None:
        """Exchange auth code for tokens in a worker thread."""
        if self._schwab_client is None:
            self.query_one("#reauth-status", Static).update(
                "[red]No Schwab client available[/red]"
            )
            return

        def do_exchange() -> str:
            tokens = self._schwab_client._client.tokens
            tokens._post_oauth_token("authorization_code", code)
            return "success"

        worker: Worker[str] = self.run_worker(do_exchange, thread=True)
        worker.on_state_changed = lambda state: self._on_exchange_done(worker)  # type: ignore[assignment]

    def _on_exchange_done(self, worker: Worker[str]) -> None:
        status_widget = self.query_one("#reauth-status", Static)
        if worker.state == WorkerState.SUCCESS:
            status_widget.update("[green]Token exchange successful![/green]")
            self._refresh_tokens()
        elif worker.state == WorkerState.ERROR:
            status_widget.update(
                f"[red]Token exchange failed: {worker.error}[/red]"
            )
