"""Help screen — strategy explainer, monitoring runbook, and glossary."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Static

_HELP_TEXT = """\
[b]HELP — Strategy, Runbook & Reference[/b]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[b u]How the Strategy Works[/b u]

  This system trades [b]equity pairs[/b] — two stocks whose prices move
  together over time.  When they temporarily diverge, the algo bets
  they will converge back.  This is called [b]statistical arbitrage[/b]
  (stat arb) or [b]pairs trading[/b].

  [b]1. Finding pairs (rolling discovery)[/b]
     Every 21 business days (~monthly) the algo scans ~60 stocks
     grouped by sector (tech, financials, healthcare, etc.) looking
     for pairs that are [b]cointegrated[/b] — statistically proven to
     share a long-run equilibrium.  Think of two dogs on leashes
     held by the same walker: they wander apart but always get
     pulled back together.

     The test used is Engle-Granger cointegration.  A pair passes
     if the cointegration p-value < 0.05, the ADF stationarity
     p-value < 0.05, and the Hurst exponent < 0.5 (confirming
     mean-reverting rather than trending behaviour).  The pair's
     half-life (how fast it reverts) must be between 5–30 days.

  [b]2. Measuring divergence (the spread)[/b]
     For a qualifying pair Y/X, the system fits Y = β·X + α using
     regression.  The [b]spread[/b] = Y − β·X − α.  This spread is
     normalized into a [b]z-score[/b] = (spread − μ) / σ, where μ and σ
     are the mean and standard deviation from the formation period.
     A z-score of −2 means the spread is 2 standard deviations
     below its historical mean — Y is cheap relative to X.

  [b]3. Trading rules[/b]
     [b]Entry:[/b]  When |z| > 2.0, open a position:
       • z < −2.0 → "long spread" (buy Y, sell X — betting Y
         will rise relative to X)
       • z > +2.0 → "short spread" (sell Y, buy X — betting Y
         will fall relative to X)

     [b]Exit:[/b]   When |z| < 0.5, the spread has reverted toward the
             mean — close the position and take profit.

     [b]Stop:[/b]   When |z| > 4.0, the spread has blown out further —
             the cointegration may have broken.  Close at a loss.

     [b]Timeout:[/b] If a position has been open longer than 3× the
               pair's half-life without exiting, force close it.

  [b]4. Position sizing[/b]
     Each leg gets $1,500 notional.  The X leg quantity is adjusted
     by the hedge ratio β so that the position is dollar-neutral
     (roughly equal dollar exposure on each side).  Max $3,000
     combined notional per pair.

  [b]5. Rolling discovery cycle[/b]
     The system uses a [b]rolling scheduler[/b]: every 21 business days
     (~monthly) it re-scans for cointegrated pairs over a trailing
     252-day lookback.  Each discovered pair gets a 63 business day
     (~1 quarter) trading lifetime.  Formation parameters (β, μ, σ)
     are [b]frozen[/b] for each pair — no lookahead bias.

     If a pair is re-discovered before its expiry, its parameters
     are refreshed (new β, μ, σ) and the trading window is reset.
     Pairs that are not re-discovered expire at the end of their
     trading window and are force-exited.  This keeps the portfolio
     fresh without waiting a full quarter between discovery cycles.

  [b]6. Earnings blackout[/b]
     When either leg of a pair has earnings within 3 business days,
     the system [b]blocks new entries[/b] and [b]force-exits[/b] existing
     positions.  This avoids overnight gap risk from earnings
     announcements (a 5-15% gap easily overwhelms the hedge ratio).
     Earnings dates are sourced from [b]Financial Modeling Prep (FMP)[/b].
     Set [b]FMP_API_KEY[/b] to enable; without it, no blackout checks run.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[b u]Monitoring Runbook[/b u]

  [b]Architecture[/b]
  The engine and dashboard are separate processes.  The engine runs
  headlessly via [b]stat-arb run-live --loop[/b] and writes events to
  the database.  This dashboard polls the DB every second and
  displays engine activity.  You can quit and reopen the dashboard
  at any time without affecting the engine.

  [b]Daily checks[/b]
    1. Open the dashboard and confirm the engine bar shows a state
       other than "not detected".  If it says "not detected" the
       engine process is not running or has crashed.

    2. Check the [b]Activity Feed[/b] for the most recent "Scan
       complete" message.  The engine runs once per day after
       market close (~16:30 ET / 21:30 UTC).  If the last scan
       is more than 1 day old on a weekday, investigate.

    3. Review [b]Portfolio[/b] panel: Value, Daily P&L, Drawdown.
       Drawdown approaching the 10% limit is a warning sign.

    4. Review [b]Risk Utilization[/b]:
       • Pairs: up to 10 active pairs is normal.
       • Exposure: should stay under $25,000 gross.
       • Drawdown: under 10%.  At 10% the kill switch
         auto-triggers and halts all new entries.
       • Sector: no single sector should exceed 30%.

    5. On the [b]Tokens[/b] screen (press 3), confirm Schwab access
       and refresh tokens are valid.  The access token refreshes
       every 30 minutes automatically.  The [b]refresh token[/b]
       expires after 7 days and requires manual re-authentication
       if it lapses.

  [b]Warning signs[/b]
    • [yellow]Yellow "Signal" events[/yellow] — normal, shows the algo is
      generating trade signals.
    • [red]Red "ERROR" events[/red] — the engine hit an exception during
      a step.  It will retry next cycle automatically, but
      repeated errors need investigation (check logs).
    • Drawdown climbing steadily — the strategy may be in a
      regime where cointegration relationships are breaking down.
      Consider whether to let it ride or activate the kill switch.
    • Many positions hitting the stop (z > 4.0) — structural
      breaks in pair relationships.  This can happen during
      earnings season, sector rotations, or macro shocks.

  [b]When to use the kill switch (press 9)[/b]
    The kill switch halts ALL new entries.  Existing positions
    continue to be monitored and will only be exited (via mean
    reversion, stop, or timeout).  Use it when:
    • Drawdown is approaching the 10% limit and you want to
      intervene before the automatic trigger.
    • You see a regime change (e.g. a market crash, sector
      shock, or news event) that invalidates the statistical
      relationships the algo relies on.
    • You need to take the system offline for maintenance.

    The kill switch is sent as a command via the database.  The
    engine picks it up within 10 seconds, acknowledges it, and
    stops entering new positions.  The engine process itself
    continues running (for exits) but will not open new trades.

    [b red]The kill switch cannot be undone from the dashboard.[/b red]
    To resume trading, restart the engine process.

  [b]Restarting the engine[/b]
    Press [b]s[/b] on the dashboard to start the engine as a background
    process.  Alternatively, from a terminal:
      [b]stat-arb run-live --loop[/b]              (paper mode)
      [b]stat-arb run-live --loop --broker-mode=live[/b]  (live mode)

    The dashboard will automatically reconnect within 1 second
    once the engine starts writing heartbeats again.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[b u]Keybindings[/b u]

  [b]1[/b]  Dashboard (main screen)
  [b]2[/b]  Pairs detail screen
  [b]3[/b]  Token management screen
  [b]?[/b]  This help screen
  [b]r[/b]  Refresh current screen data
  [b]s[/b]  Start engine (spawns background process)
  [b]9[/b]  Kill switch (sends command to engine via DB)
  [b]q[/b]  Quit dashboard (engine continues running)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[b u]Dashboard Reference[/b u]

[b]Positions Table[/b]
  [b]Pair[/b]       Y/X stock pair (e.g. AAPL/MSFT).
  [b]Dir[/b]        "long" (long Y, short X) or "short" (short Y, long X).
  [b]Z-Score[/b]    Current spread z-score: (spread − μ) / σ.
  [b]H.L.[/b]       Half-life in days.  Lower = faster mean reversion.
  [b]Entry Z[/b]    Z-score when the position was opened.
  [b]Days[/b]       Calendar days the position has been held.
  [b]Sector[/b]     Industry sector of the pair.

[b]Pairs Detail (screen 2)[/b]
  [b]Beta (β)[/b]   Hedge ratio from regression (Y = β·X + α).
  [b]Mu (μ)[/b]     Formation-period spread mean.
  [b]Sigma (σ)[/b]  Formation-period spread std deviation.
  [b]Coint p[/b]    Engle-Granger cointegration p-value (lower = better).
  [b]ADF p[/b]      ADF stationarity p-value (lower = better).
  [b]Hurst[/b]      Hurst exponent (< 0.5 = mean-reverting).
  [b]Intercept[/b]  Regression intercept (α).

[b]Portfolio Panel[/b]
  [b]Value[/b]        Total portfolio value (cash + positions).
  [b]Daily P&L[/b]    Today's profit/loss.
  [b]Drawdown[/b]     Peak-to-trough decline as % of high-water mark.
  [b]Gross Exp[/b]    Sum of absolute position values across all legs.
  [b]Active Pairs[/b] Number of pairs with open positions.

[b]Risk Utilization[/b]
  [b]Pairs[/b]      Current / max (10).
  [b]Exposure[/b]   Gross notional / max ($25,000).
  [b]Drawdown[/b]   Current % / max (10%).
  [b]Sector[/b]     Per-sector concentration / max (30%).
  [b]Kill Switch[/b] ON = all new entries halted, exits only.

[b]System Status[/b]
  [b]Mode[/b]       PAPER (simulated) or LIVE (real Schwab API).
  [b]Engine[/b]     Current state: running / idle / not detected.
  [b]DB[/b]         Database connectivity.
  [b]Schwab[/b]     Schwab API connectivity.
  [b]Access[/b]     OAuth access token time remaining (30 min lifetime).
  [b]Refresh[/b]    OAuth refresh token time remaining (7 day lifetime).

[b]Activity Feed[/b]
  Real-time log of engine events with severity colours:
    white = info, [yellow]yellow = warning[/yellow], \
[red]red = error[/red], [red bold]bold red = critical[/red bold]
"""


class HelpScreen(Screen):
    """Scrollable help / glossary screen."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("question_mark", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-scroll"):
            yield Static(_HELP_TEXT, id="help-text")
        yield Footer()

    def action_go_back(self) -> None:
        self.app.switch_screen("main")
