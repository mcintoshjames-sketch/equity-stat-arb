"""Schwab API client wrapper around schwabdev."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import schwabdev

from stat_arb.config.settings import SchwabConfig

logger = logging.getLogger(__name__)


class SchwabAPIError(Exception):
    def __init__(self, endpoint: str, status_code: int, body: str = ""):
        self.endpoint = endpoint
        self.status_code = status_code
        self.body = body
        super().__init__(f"{endpoint} returned HTTP {status_code}: {body}")


def _check_response(resp: Any, endpoint: str) -> None:
    if resp.status_code != 200:
        raise SchwabAPIError(endpoint, resp.status_code, resp.text[:500])


class SchwabDataClient:
    """Wrapper around schwabdev providing typed data access and order placement."""

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
        if self._account_hash is None:
            resp = self._client.linked_accounts()
            _check_response(resp, "linked_accounts")
            accounts = resp.json()
            if not accounts:
                raise RuntimeError("No linked Schwab accounts found")
            self._account_hash = accounts[0]["hashValue"]
            logger.info("Auto-resolved account hash: %s...", self._account_hash[:8])
        return self._account_hash

    def fetch_price_history(
        self,
        symbol: str,
        period_type: str = "year",
        period: int = 1,
        frequency_type: str = "daily",
        frequency: int = 1,
    ) -> pd.DataFrame:
        """Fetch OHLCV history for a symbol. Returns DataFrame with DatetimeIndex."""
        resp = self._client.price_history(
            symbol,
            periodType=period_type,
            period=period,
            frequencyType=frequency_type,
            frequency=frequency,
        )
        _check_response(resp, f"price_history({symbol})")
        data = resp.json()
        candles = data.get("candles", [])
        if not candles:
            logger.warning("No candle data for %s", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        # Schwab returns datetime as milliseconds since epoch
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df = df.set_index("datetime")
        df.index.name = "date"
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_batch_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Fetch quotes for multiple symbols (batched in groups of 200)."""
        all_quotes: dict[str, dict] = {}
        for i in range(0, len(symbols), 200):
            batch = symbols[i : i + 200]
            resp = self._client.quotes(batch)
            _check_response(resp, f"quotes(batch {i // 200})")
            data = resp.json()
            for sym, info in data.items():
                all_quotes[sym] = info.get("quote", {})
        return all_quotes

    def place_order(self, order: dict) -> str | None:
        """Place an order. Returns order ID from Location header if available."""
        resp = self._client.place_order(self.account_hash, order)
        if resp.status_code not in (200, 201):
            raise SchwabAPIError("place_order", resp.status_code, resp.text[:500])
        location = resp.headers.get("Location", "")
        # Location header contains the order ID at the end of the URL
        order_id = location.rsplit("/", 1)[-1] if location else None
        logger.info("Order placed, id=%s", order_id)
        return order_id

    def get_positions(self) -> list[dict]:
        """Get current account positions."""
        resp = self._client.account_details(self.account_hash, fields="positions")
        _check_response(resp, "account_details(positions)")
        data = resp.json()
        return data["securitiesAccount"].get("positions", [])

    def get_account_value(self) -> float:
        """Get liquidation value of the account."""
        resp = self._client.account_details(self.account_hash)
        _check_response(resp, "account_details")
        data = resp.json()
        return float(data["securitiesAccount"]["currentBalances"]["liquidationValue"])
