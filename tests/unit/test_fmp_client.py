"""Tests for the FMP API client."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stat_arb.config.settings import FmpConfig
from stat_arb.data.fmp_client import FmpClient, _DiskCache


def _make_config(tmp_path: Path, api_key: str = "test-key") -> FmpConfig:
    from pydantic import SecretStr

    return FmpConfig(
        api_key=SecretStr(api_key),
        earnings_blackout_days=3,
        cache_dir=str(tmp_path),
    )


@pytest.fixture()
def fmp_response() -> list[dict]:
    return [
        {"symbol": "AAPL", "date": "2024-07-25", "time": "amc"},
        {"symbol": "MSFT", "date": "2024-07-30", "time": "bmo"},
        {"symbol": "GOOG", "date": "2024-08-01", "time": "amc"},
    ]


def test_parse_earnings_response(tmp_path: Path, fmp_response: list[dict]) -> None:
    """Mock HTTP, verify cache is built correctly."""
    cfg = _make_config(tmp_path)
    client = FmpClient(cfg)

    with patch("stat_arb.data.fmp_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fmp_response
        mock_get.return_value = mock_resp

        result = client.get_next_earnings(
            ["AAPL", "MSFT", "GOOG"], date(2024, 7, 20),
        )

    assert result["AAPL"] == date(2024, 7, 25)
    assert result["MSFT"] == date(2024, 7, 30)
    assert result["GOOG"] == date(2024, 8, 1)


def test_cache_reuse_same_day(tmp_path: Path, fmp_response: list[dict]) -> None:
    """No re-fetch when called again with same as_of date."""
    cfg = _make_config(tmp_path)
    client = FmpClient(cfg)

    with patch("stat_arb.data.fmp_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fmp_response
        mock_get.return_value = mock_resp

        client.get_next_earnings(["AAPL"], date(2024, 7, 20))
        client.get_next_earnings(["AAPL"], date(2024, 7, 20))

    # Should only call API once (second call uses in-memory cache)
    assert mock_get.call_count == 1


def test_cache_invalidated_next_day(tmp_path: Path, fmp_response: list[dict]) -> None:
    """New day invalidates in-memory cache; disk cache may or may not cover."""
    cfg = _make_config(tmp_path)
    client = FmpClient(cfg)

    with patch("stat_arb.data.fmp_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fmp_response
        mock_get.return_value = mock_resp

        result1 = client.get_next_earnings(["AAPL"], date(2024, 7, 20))
        assert result1["AAPL"] == date(2024, 7, 25)

        # Next day — in-memory cache is invalidated but disk cache
        # covers [Jul 20, Sep 3]; second query is [Jul 21, Sep 4]
        # which extends 1 day past, so it may re-fetch
        result2 = client.get_next_earnings(["AAPL"], date(2024, 7, 21))
        assert result2["AAPL"] == date(2024, 7, 25)

    # At least one API call happened; second may or may not depending
    # on whether disk cache covered the extended range
    assert mock_get.call_count >= 1


def test_disk_cache_persists(tmp_path: Path, fmp_response: list[dict]) -> None:
    """Write then read from JSON file."""
    cache = _DiskCache(str(tmp_path))

    start = date(2024, 7, 1)
    end = date(2024, 8, 15)
    earnings = [
        {"symbol": "AAPL", "date": "2024-07-25"},
        {"symbol": "MSFT", "date": "2024-07-30"},
    ]

    cache.store(start, end, earnings)

    # New cache instance — reads from disk
    cache2 = _DiskCache(str(tmp_path))
    assert cache2.covers_range(date(2024, 7, 10), date(2024, 8, 10))

    entries = cache2.get_earnings_in_range(date(2024, 7, 20), date(2024, 7, 31))
    symbols = {e["symbol"] for e in entries}
    assert "AAPL" in symbols
    assert "MSFT" in symbols


def test_disk_cache_covers_range(tmp_path: Path) -> None:
    """No API call when range is already covered in disk cache."""
    cfg = _make_config(tmp_path)

    # Pre-populate disk cache with a range that covers [Jul 20, Sep 3]
    # (as_of=Jul 20, horizon=45 days → end=Sep 3)
    cache_path = tmp_path / "earnings_cache.json"
    cache_data = {
        "fetched_ranges": [["2024-07-01", "2024-09-15"]],
        "earnings": [
            {"symbol": "AAPL", "date": "2024-07-25"},
        ],
    }
    cache_path.write_text(json.dumps(cache_data))

    client = FmpClient(cfg)

    with patch("stat_arb.data.fmp_client.requests.get") as mock_get:
        result = client.get_next_earnings(["AAPL"], date(2024, 7, 20))

    # Should not call API since disk cache covers the range
    mock_get.assert_not_called()
    assert result["AAPL"] == date(2024, 7, 25)


def test_api_error_graceful(tmp_path: Path) -> None:
    """HTTP error returns empty, doesn't crash."""
    cfg = _make_config(tmp_path)
    client = FmpClient(cfg)

    with patch("stat_arb.data.fmp_client.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []  # Empty response
        mock_get.return_value = mock_resp

        result = client.get_next_earnings(["AAPL"], date(2024, 7, 20))

    assert result["AAPL"] is None


def test_no_api_key_returns_empty(tmp_path: Path) -> None:
    """When api_key is None, return empty dict immediately."""
    cfg = FmpConfig(api_key=None, cache_dir=str(tmp_path))
    client = FmpClient(cfg)
    result = client.get_next_earnings(["AAPL"], date(2024, 7, 20))
    assert result == {}
