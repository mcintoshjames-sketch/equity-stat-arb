"""Tests for the EarningsBlackout checker."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from stat_arb.risk.earnings_blackout import EarningsBlackout


def _make_blackout(
    earnings_map: dict[str, date | None],
    blackout_days: int = 3,
) -> EarningsBlackout:
    """Build an EarningsBlackout with a pre-populated cache."""
    mock_fmp = MagicMock()
    mock_fmp.get_next_earnings.return_value = earnings_map
    blackout = EarningsBlackout(mock_fmp, blackout_days=blackout_days)
    blackout.refresh(list(earnings_map.keys()), date(2024, 7, 20))
    return blackout


def test_within_blackout_blocked() -> None:
    """2 bdays to earnings, blackout=3 → blocked."""
    # Friday Jul 19 → Monday Jul 22 earnings: bdays = [Jul 19, Jul 22], len-1=1
    # Actually testing: as_of=Jul 22 (Mon), earnings=Jul 24 (Wed)
    # bdays = [Jul 22, Jul 23, Jul 24], len-1 = 2, 2 <= 3 → blocked
    bo = _make_blackout({"AAPL": date(2024, 7, 24)}, blackout_days=3)
    assert bo.is_blacked_out("AAPL", date(2024, 7, 22)) is True


def test_outside_blackout_clear() -> None:
    """5 bdays to earnings, blackout=3 → not blocked."""
    # as_of=Jul 17 (Wed), earnings=Jul 24 (Wed)
    # bdays = [Jul 17, 18, 19, 22, 23, 24], len-1=5, 5 > 3 → clear
    bo = _make_blackout({"AAPL": date(2024, 7, 24)}, blackout_days=3)
    assert bo.is_blacked_out("AAPL", date(2024, 7, 17)) is False


def test_no_earnings_clear() -> None:
    """Symbol not in calendar → not blocked."""
    bo = _make_blackout({"AAPL": None}, blackout_days=3)
    assert bo.is_blacked_out("AAPL", date(2024, 7, 20)) is False


def test_past_earnings_ignored() -> None:
    """Earnings yesterday → not blocked (past, not upcoming)."""
    bo = _make_blackout({"AAPL": date(2024, 7, 19)}, blackout_days=3)
    assert bo.is_blacked_out("AAPL", date(2024, 7, 20)) is False


def test_pair_checks_both_legs() -> None:
    """Returns whichever leg is blacked out."""
    bo = _make_blackout({
        "AAPL": date(2024, 7, 24),  # within 3 bdays of Jul 22
        "MSFT": date(2024, 8, 15),  # far away
    }, blackout_days=3)
    result = bo.pair_blacked_out("AAPL", "MSFT", date(2024, 7, 22))
    assert result == "AAPL"

    # When neither is blacked out
    result = bo.pair_blacked_out("AAPL", "MSFT", date(2024, 7, 10))
    assert result is None


def test_bmo_safe() -> None:
    """Earnings on Monday BMO, blackout_days=1 → blocked on Friday."""
    # Earnings Mon Jul 22 BMO. as_of=Fri Jul 19.
    # bdays = [Jul 19, Jul 22], len-1=1, 1 <= 1 → blocked ✓
    # We are flat at Friday close, before Monday open.
    bo = _make_blackout({"AAPL": date(2024, 7, 22)}, blackout_days=1)
    assert bo.is_blacked_out("AAPL", date(2024, 7, 19)) is True


def test_exact_boundary() -> None:
    """Earnings exactly blackout_days bdays away → blocked."""
    # as_of=Mon Jul 22, earnings=Thu Jul 25
    # bdays = [Jul 22, 23, 24, 25], len-1=3, 3 <= 3 → blocked
    bo = _make_blackout({"AAPL": date(2024, 7, 25)}, blackout_days=3)
    assert bo.is_blacked_out("AAPL", date(2024, 7, 22)) is True

    # One day earlier (Jul 19 Fri), earnings Jul 25 Thu
    # bdays = [Jul 19, 22, 23, 24, 25], len-1=4, 4 > 3 → clear
    assert bo.is_blacked_out("AAPL", date(2024, 7, 19)) is False
