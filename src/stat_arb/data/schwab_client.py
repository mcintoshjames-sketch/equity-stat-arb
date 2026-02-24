"""Schwab API client wrapper around ``schwabdev``.

Provides a typed, retry-aware interface to the Charles Schwab brokerage API.
All methods return parsed Python objects (DataFrames, dicts) rather than raw
``requests.Response`` objects.

Authentication note:
    ``schwabdev`` handles OAuth2 token refresh internally via the SQLite
    ``tokens_db`` file specified in ``SchwabConfig``.  On first run it opens
    a browser for interactive authorisation; subsequent runs refresh
    automatically.  Tokens expire after 7 days of inactivity.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import pandas as pd
import schwabdev
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from stat_arb.config.settings import SchwabConfig

if TYPE_CHECKING:
    import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BATCH_QUOTE_LIMIT = 200  # Schwab's max symbols per quotes() call

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class SchwabAPIError(Exception):
    """Raised when a Schwab API call returns a non-200 status code."""

    def __init__(self, endpoint: str, status_code: int, body: str = "") -> None:
        self.endpoint = endpoint
        self.status_code = status_code
        self.body = body
        super().__init__(f"{endpoint} returned HTTP {status_code}: {body}")

    @property
    def is_retryable(self) -> bool:
        """True for transient errors (rate-limit or server errors)."""
        return self.status_code == 429 or self.status_code >= 500


def _check_response(resp: requests.Response, endpoint: str) -> None:
    """Validate HTTP 200 response, raising ``SchwabAPIError`` on failure."""
    if resp.status_code != 200:
        raise SchwabAPIError(endpoint, resp.status_code, resp.text[:500])


def _is_retryable(exc: BaseException) -> bool:
    """Tenacity predicate: retry only on transient API errors."""
    return isinstance(exc, SchwabAPIError) and exc.is_retryable


# Shared retry decorator for all API methods.
# Exponential backoff 1s→2s→4s with jitter, max 4 attempts.
_api_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=1, max=16, jitter=2),
    stop=stop_after_attempt(4),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SchwabDataClient:
    """Typed wrapper around ``schwabdev.Client`` for data access and order placement.

    All API calls include structured logging (endpoint, symbol, response time)
    and automatic retry with exponential backoff for transient errors (HTTP 429
    rate-limits and 5xx server errors).

    Args:
        config: ``SchwabConfig`` subsection with API credentials and paths.
    """

    def __init__(self, config: SchwabConfig) -> None:
        self._config = config
        self._client = schwabdev.Client(
            app_key=config.app_key.get_secret_value(),
            app_secret=config.app_secret.get_secret_value(),
            callback_url=config.callback_url,
            tokens_db=config.tokens_db,
        )
        self._account_hash: str | None = config.account_hash

    @property
    def account_hash(self) -> str:
        """Resolve the Schwab account hash, auto-detecting for single accounts."""
        if self._account_hash is None:
            resp = self._client.linked_accounts()
            _check_response(resp, "linked_accounts")
            accounts = resp.json()
            if not accounts:
                raise RuntimeError("No linked Schwab accounts found")
            self._account_hash = accounts[0]["hashValue"]
            logger.info("Auto-resolved account hash: %s...", self._account_hash[:8])
        return self._account_hash

    @_api_retry
    def fetch_price_history(
        self,
        symbol: str,
        period_type: str = "year",
        period: int = 1,
        frequency_type: str = "daily",
        frequency: int = 1,
    ) -> pd.DataFrame:
        """Fetch OHLCV daily bars for a single symbol.

        Args:
            symbol: Ticker symbol (e.g. ``"AAPL"``).
            period_type: ``"day"``, ``"month"``, ``"year"``, ``"ytd"``.
            period: Number of periods (e.g. 2 years).
            frequency_type: ``"daily"``, ``"weekly"``, ``"monthly"``.
            frequency: Frequency multiplier (usually 1).

        Returns:
            DataFrame with ``DatetimeIndex`` and columns
            ``[open, high, low, close, volume]``.  Empty DataFrame if
            no candle data is available.
        """
        endpoint = f"price_history({symbol})"
        t0 = time.perf_counter()

        resp = self._client.price_history(
            symbol,
            periodType=period_type,
            period=period,
            frequencyType=frequency_type,
            frequency=frequency,
        )
        _check_response(resp, endpoint)

        elapsed = time.perf_counter() - t0
        data = resp.json()
        candles = data.get("candles", [])

        if not candles:
            logger.warning("No candle data for %s (%.1fms)", symbol, elapsed * 1000)
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        # Schwab returns datetime as milliseconds since epoch
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df = df.set_index("datetime")
        df.index.name = "date"

        logger.debug(
            "Fetched %d bars for %s (%.1fms)", len(df), symbol, elapsed * 1000,
        )
        return df[["open", "high", "low", "close", "volume"]]

    @_api_retry
    def fetch_batch_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Fetch real-time quotes for multiple symbols.

        Automatically chunks requests into groups of 200 (Schwab's per-call
        limit) and merges results.

        Args:
            symbols: List of ticker symbols.

        Returns:
            Dict mapping symbol → quote data dict.
        """
        all_quotes: dict[str, dict] = {}

        for i in range(0, len(symbols), _BATCH_QUOTE_LIMIT):
            batch = symbols[i : i + _BATCH_QUOTE_LIMIT]
            endpoint = f"quotes(batch {i // _BATCH_QUOTE_LIMIT})"
            t0 = time.perf_counter()

            resp = self._client.quotes(batch)
            _check_response(resp, endpoint)

            elapsed = time.perf_counter() - t0
            data = resp.json()
            for sym, info in data.items():
                all_quotes[sym] = info.get("quote", {})

            logger.debug(
                "Fetched %d quotes in batch %d (%.1fms)",
                len(batch), i // _BATCH_QUOTE_LIMIT, elapsed * 1000,
            )

        return all_quotes

    @_api_retry
    def place_order(self, order: dict[str, Any]) -> str | None:
        """Place an order via the Schwab API.

        Args:
            order: Schwab order JSON dict (see ``order_builder.py``).

        Returns:
            Order ID parsed from the ``Location`` response header, or
            ``None`` if the header is absent.

        Raises:
            SchwabAPIError: If the order is rejected (non-200/201 status).
        """
        t0 = time.perf_counter()
        resp = self._client.place_order(self.account_hash, order)
        elapsed = time.perf_counter() - t0

        if resp.status_code not in (200, 201):
            raise SchwabAPIError("place_order", resp.status_code, resp.text[:500])

        # Location header contains the order ID at the end of the URL
        location = resp.headers.get("Location", "")
        order_id = location.rsplit("/", 1)[-1] if location else None
        logger.info("Order placed, id=%s (%.1fms)", order_id, elapsed * 1000)
        return order_id

    @_api_retry
    def get_positions(self) -> list[dict]:
        """Fetch current account positions.

        Returns:
            List of position dicts from the Schwab ``securitiesAccount``
            response.  Each dict contains ``instrument``, ``longQuantity``,
            ``shortQuantity``, ``marketValue``, etc.
        """
        t0 = time.perf_counter()
        resp = self._client.account_details(self.account_hash, fields="positions")
        _check_response(resp, "account_details(positions)")
        elapsed = time.perf_counter() - t0

        data = resp.json()
        positions = data["securitiesAccount"].get("positions", [])
        logger.debug("Fetched %d positions (%.1fms)", len(positions), elapsed * 1000)
        return positions

    @_api_retry
    def get_account_value(self) -> float:
        """Fetch the account's liquidation value.

        Returns:
            Total account liquidation value as a float.

        Note:
            Uses ``liquidationValue`` (not ``equity``) per the
            SCHWAB_API_PRIMER — this is the true total value if all
            positions were closed.
        """
        t0 = time.perf_counter()
        resp = self._client.account_details(self.account_hash)
        _check_response(resp, "account_details")
        elapsed = time.perf_counter() - t0

        data = resp.json()
        value = float(data["securitiesAccount"]["currentBalances"]["liquidationValue"])
        logger.debug("Account value: $%.2f (%.1fms)", value, elapsed * 1000)
        return value
