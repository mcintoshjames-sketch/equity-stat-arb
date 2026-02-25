"""Kill switch widget — status indicator and emergency stop button."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static


class KillSwitchActivated(Message):
    """Posted when the user clicks the emergency stop button."""


class KillSwitchWidget(Widget):
    """Displays kill switch status and an emergency stop button."""

    DEFAULT_CSS = """
    KillSwitchWidget {
        height: auto;
        padding: 1 2;
        layout: vertical;
    }
    """

    def __init__(self, engine_active: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._killed = False
        self._engine_active = engine_active

    def compose(self) -> ComposeResult:
        yield Static("[green]SAFE[/green]", id="kill-status")
        yield Button(
            "EMERGENCY STOP",
            id="kill-btn",
            variant="error",
            disabled=not self._engine_active,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "kill-btn" and not self._killed:
            self.post_message(KillSwitchActivated())

    def set_killed(self) -> None:
        """Update display to reflect killed state."""
        self._killed = True
        self.query_one("#kill-status", Static).update(
            "[red bold]KILLED[/red bold]"
        )
        self.query_one("#kill-btn", Button).disabled = True

    def enable(self) -> None:
        """Enable the kill switch button (engine is now running)."""
        self._engine_active = True
        self.query_one("#kill-btn", Button).disabled = False

    @property
    def is_killed(self) -> bool:
        return self._killed
