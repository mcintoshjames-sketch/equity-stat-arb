"""Confirmation dialogs for the TUI dashboard."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfirmKillSwitch(ModalScreen[bool]):
    """Modal confirmation before activating the kill switch.

    Returns True if the user confirms, False if cancelled.
    """

    DEFAULT_CSS = """
    ConfirmKillSwitch {
        align: center middle;
    }

    #confirm-dialog {
        width: 60;
        height: auto;
        border: thick $error;
        background: $surface;
        padding: 2 4;
    }

    #confirm-dialog Static {
        width: 100%;
        content-align: center middle;
        margin: 1 0;
    }

    #confirm-buttons {
        width: 100%;
        align: center middle;
        margin-top: 2;
    }

    #confirm-buttons Button {
        margin: 0 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static("[b red]KILL SWITCH CONFIRMATION[/b red]")
            yield Static(
                "This will halt ALL new entries.\n"
                "Existing positions will only be exited.\n\n"
                "Are you sure?"
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes - KILL", id="confirm-yes", variant="error")
                yield Button("No - Cancel", id="confirm-no", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")
