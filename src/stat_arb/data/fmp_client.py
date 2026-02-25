"""Financial Modeling Prep (FMP) API client for earnings calendar data.

Provides a two-layer cached interface to the FMP earnings calendar endpoint:

1. **Persistent disk cache** — JSON file at ``{cache_dir}/earnings_cache.json``.
   Survives process restarts.  A multi-year backtest fetches from FMP once,
   then all simulated days hit disk.
2. **In-memory day cache** — ``dict[str, date | None]`` keyed by symbol,
   rebuilt from disk cache each simulated day.  Avoids re-parsing JSON on
   every ``is_blacked_out`` call.

The FMP free tier allows 250 requests/day.  The two-layer cache ensures
backtests never exhaust this limit.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

if TYPE_CHECKING:
    from stat_arb.config.settings import FmpConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://financialmodelingprep.com/stable"
_FETCH_HORIZON_DAYS = 45  # fetch this many calendar days ahead of as_of

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class FmpAPIError(Exception):
    """Raised when an FMP API call fails."""

    def __init__(self, endpoint: str, status_code: int, body: str = "") -> None:
        self.endpoint = endpoint
        self.status_code = status_code
        self.body = body
        super().__init__(f"{endpoint} returned HTTP {status_code}: {body}")

    @property
    def is_retryable(self) -> bool:
        return self.status_code == 429 or self.status_code >= 500


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, FmpAPIError) and exc.is_retryable


_api_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=1, max=16, jitter=2),
    stop=stop_after_attempt(4),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


class _DiskCache:
    """Persistent JSON cache for earnings data."""

    def __init__(self, cache_dir: str) -> None:
        self._path = Path(cache_dir) / "earnings_cache.json"
        self._data: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._data is not None:
            return self._data
        if self._path.exists():
            with self._path.open() as f:
                self._data = json.load(f)
        else:
            self._data = {"fetched_ranges": [], "earnings": []}
        return self._data

    def _save(self) -> None:
        if self._data is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as f:
            json.dump(self._data, f, indent=2)

    def covers_range(self, start: date, end: date) -> bool:
        """Check if any fetched range fully covers [start, end]."""
        data = self._load()
        start_s, end_s = start.isoformat(), end.isoformat()
        for r_start, r_end in data["fetched_ranges"]:
            if r_start <= start_s and r_end >= end_s:
                return True
        return False

    def get_earnings_in_range(self, start: date, end: date) -> list[dict[str, str]]:
        """Return earnings entries within [start, end]."""
        data = self._load()
        start_s, end_s = start.isoformat(), end.isoformat()
        return [
            e for e in data["earnings"]
            if start_s <= e["date"] <= end_s
        ]

    def store(self, start: date, end: date, earnings: list[dict[str, str]]) -> None:
        """Store fetched earnings and mark the range as covered."""
        data = self._load()
        data["fetched_ranges"].append([start.isoformat(), end.isoformat()])
        # Merge new earnings, dedup by (symbol, date)
        existing = {(e["symbol"], e["date"]) for e in data["earnings"]}
        for e in earnings:
            key = (e["symbol"], e["date"])
            if key not in existing:
                data["earnings"].append(e)
                existing.add(key)
        self._save()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class FmpClient:
    """Cached FMP API client for earnings calendar data.

    Args:
        config: ``FmpConfig`` subsection with API key and cache settings.
    """

    def __init__(self, config: FmpConfig) -> None:
        self._config = config
        self._api_key = config.api_key.get_secret_value() if config.api_key else None
        self._disk = _DiskCache(config.cache_dir)
        self._mem_cache: dict[str, date | None] = {}
        self._mem_cache_date: date | None = None

    def get_next_earnings(
        self,
        symbols: list[str],
        as_of: date,
    ) -> dict[str, date | None]:
        """Get the next earnings date for each symbol after ``as_of``.

        Returns a dict mapping symbol → next earnings date (or None if
        no upcoming earnings found in the cache/API response).

        Uses two-layer cache: in-memory (same-day fast path) and disk
        (persists across process restarts).
        """
        if self._api_key is None:
            return {}

        # Layer 1: in-memory cache (same day = instant return)
        if self._mem_cache_date == as_of and self._mem_cache:
            return self._mem_cache

        # Determine fetch range
        start = as_of
        end = as_of + timedelta(days=_FETCH_HORIZON_DAYS)

        # Layer 2: disk cache
        if self._disk.covers_range(start, end):
            raw_earnings = self._disk.get_earnings_in_range(start, end)
        else:
            # Layer 3: API fetch
            raw_earnings = self._fetch_earnings(start, end)
            self._disk.store(start, end, raw_earnings)

        # Build in-memory cache: symbol → nearest future earnings date
        self._mem_cache = self._build_mem_cache(symbols, raw_earnings, as_of)
        self._mem_cache_date = as_of
        return self._mem_cache

    @_api_retry
    def _fetch_earnings(self, start: date, end: date) -> list[dict[str, str]]:
        """Fetch earnings calendar from FMP API."""
        endpoint = "earnings-calendar"
        url = f"{_BASE_URL}/{endpoint}"
        params = {
            "from": start.isoformat(),
            "to": end.isoformat(),
            "apikey": self._api_key,
        }

        t0 = time.perf_counter()
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            logger.error("FMP API request failed: %s", exc)
            return []

        elapsed = time.perf_counter() - t0

        if resp.status_code != 200:
            raise FmpAPIError(endpoint, resp.status_code, resp.text[:500])

        data = resp.json()
        if not isinstance(data, list):
            logger.warning("FMP returned unexpected format: %s", type(data).__name__)
            return []

        earnings = [
            {"symbol": item["symbol"], "date": item["date"]}
            for item in data
            if "symbol" in item and "date" in item
        ]

        logger.info(
            "FMP: fetched %d earnings events for %s → %s (%.1fms)",
            len(earnings), start, end, elapsed * 1000,
        )
        return earnings

    @staticmethod
    def _build_mem_cache(
        symbols: list[str],
        raw_earnings: list[dict[str, str]],
        as_of: date,
    ) -> dict[str, date | None]:
        """Build symbol → nearest future earnings date mapping."""
        symbol_set = set(symbols)
        result: dict[str, date | None] = {s: None for s in symbols}

        for entry in raw_earnings:
            sym = entry["symbol"]
            if sym not in symbol_set:
                continue
            try:
                edate = date.fromisoformat(entry["date"])
            except (ValueError, TypeError):
                continue
            if edate <= as_of:
                continue
            # Keep nearest future date
            if result[sym] is None or edate < result[sym]:
                result[sym] = edate

        return result
