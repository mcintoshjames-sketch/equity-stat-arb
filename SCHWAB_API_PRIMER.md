# Schwab API Primer for Coding Agents

A practical guide to connecting to the Charles Schwab brokerage API using `schwabdev` (Python). Covers authentication, key endpoints, response formats, and gotchas.

---

## Prerequisites

### 1. Schwab Developer App

Register at [developer.schwab.com](https://developer.schwab.com):

1. Create an app to get an **App Key** and **App Secret**
2. Set the **callback URL** to `https://127.0.0.1` (localhost)
3. Wait for approval (can take 1-2 business days)

### 2. Install schwabdev

```bash
pip install "schwabdev>=3.0,<4.0"
```

### 3. Credentials File

Create `~/.schwab_greeks.env` (or your preferred location):

```env
SCHWAB_APP_KEY=your_app_key_here
SCHWAB_APP_SECRET=your_app_secret_here
SCHWAB_CALLBACK_URL=https://127.0.0.1
SCHWAB_ACCOUNT_HASH=               # optional — auto-detected for single accounts
```

Use `SecretStr` from Pydantic when loading credentials to avoid leaking secrets in logs:

```python
from pydantic import SecretStr

app_key: SecretStr = ...
client = schwabdev.Client(
    app_key=app_key.get_secret_value(),  # .get_secret_value() required
    ...
)
```

---

## Authentication

```python
import schwabdev

client = schwabdev.Client(
    app_key="YOUR_APP_KEY",
    app_secret="YOUR_APP_SECRET",
    callback_url="https://127.0.0.1",
    tokens_db="/path/to/tokens.db",   # SQLite file for token persistence
)
```

**First run**: `schwabdev` opens a browser for OAuth authorization. You log in to Schwab, authorize the app, and get redirected to the callback URL. Paste the full redirect URL back into the terminal prompt.

**Subsequent runs**: Token refresh is automatic. No manual intervention needed as long as `tokens.db` exists and the refresh token hasn't expired (Schwab refresh tokens last 7 days — if you don't run the app for 7+ days, you'll need to re-authorize).

### Token File Security

Set `0o600` permissions on the token DB. Reject symlinks. Keep it inside `$HOME`:

```python
from pathlib import Path
token_path = Path.home() / ".schwabdev" / "tokens.db"
token_path.parent.mkdir(parents=True, exist_ok=True)
token_path.touch(mode=0o600)
```

---

## Account Resolution

Every account endpoint requires an `account_hash` (an opaque string, NOT the account number).

```python
resp = client.linked_accounts()
accounts = resp.json()
# → [{"hashValue": "ABC123...", "accountNumber": "12345678"}, ...]

account_hash = accounts[0]["hashValue"]
```

For single-account users, auto-detect. For multi-account users, require explicit selection.

---

## Response Checking

**Every** Schwab API call returns a `requests.Response` object. Always check the status code before parsing:

```python
class SchwabAPIError(Exception):
    def __init__(self, endpoint: str, status_code: int, body: str = ""):
        self.endpoint = endpoint
        self.status_code = status_code
        self.body = body
        super().__init__(f"{endpoint} returned HTTP {status_code}: {body}")

def check_response(resp, endpoint: str) -> None:
    if resp.status_code != 200:
        raise SchwabAPIError(endpoint, resp.status_code, resp.text[:500])
```

Common errors:
- **401**: Token expired — `schwabdev` should auto-refresh, but sometimes requires re-auth
- **400**: Bad request — usually invalid parameter values (e.g., unsupported transaction type)
- **403**: App not approved or insufficient permissions
- **429**: Rate limited

---

## Key Endpoints

### Account Details & Balances

```python
resp = client.account_details(account_hash)
check_response(resp, "account_details")
data = resp.json()

balances = data["securitiesAccount"]["currentBalances"]
```

Key balance fields:

| Field | Meaning |
|-------|---------|
| `cashBalance` | Cash in the account |
| `equity` | Equity value (does NOT include full option mark-to-market) |
| `liquidationValue` | True total value if everything were closed — **use this for portfolio valuation** |
| `buyingPower` | Stock margin buying power = 2x SMA (NOT actual available cash) |
| `maintenanceRequirement` | Current margin maintenance requirement |
| `longMarketValue` | Total long position value |
| `shortMarketValue` | Total short position value (negative number) |
| `longOptionMarketValue` | Long options mark-to-market |
| `shortOptionMarketValue` | Short options mark-to-market (negative number) |

**Gotcha**: `equity` ≠ `liquidationValue`. For accounts with options, `liquidationValue` is the accurate total account value. Use it for XIRR terminal values, portfolio summaries, etc.

**Gotcha**: `buyingPower` is stock margin BP = 2 × SMA. To get actual margin requirement from a buying power impact: `margin = bp_impact / 2`.

### Positions

```python
resp = client.account_details(account_hash, fields="positions")
check_response(resp, "positions")
data = resp.json()

positions = data["securitiesAccount"].get("positions", [])
```

Each position:
```json
{
  "shortQuantity": 0,
  "averagePrice": 150.00,
  "longQuantity": 100,
  "marketValue": 15500.00,
  "instrument": {
    "assetType": "EQUITY",
    "symbol": "AAPL"
  }
}
```

**Gotcha**: Option positions **do not include Greeks** (delta, gamma, theta, vega are missing). You must fetch Greeks separately via the quotes API (see below).

**Gotcha**: Option positions sometimes omit `strikePrice` and `expirationDate`. Parse them from the OCC symbol as fallback:

```
OCC format: "AAPL  260320C00150000"
             ^^^^^^ underlying (6 chars, right-padded)
                    ^^^^^^ YYMMDD expiration
                          ^ C=call, P=put
                           ^^^^^^^^ strike × 1000 (3 implied decimals)
```

### Quotes (Single & Batch)

```python
# Single
resp = client.quote("AAPL")
data = resp.json()
quote = data["AAPL"]["quote"]
price = quote["lastPrice"]

# Batch (up to ~200 symbols per call)
resp = client.quotes(["AAPL", "TSLA", "NVDA"])
data = resp.json()
for symbol, info in data.items():
    quote = info["quote"]
```

Use batch quotes to fetch Greeks for option positions:

```python
option_symbols = [p.symbol for p in positions if p.asset_type == "OPTION"]
for i in range(0, len(option_symbols), 200):  # batch in groups of 200
    batch = option_symbols[i:i+200]
    resp = client.quotes(batch)
    data = resp.json()
    for sym, info in data.items():
        q = info["quote"]
        delta = q.get("delta")
        gamma = q.get("gamma")
        theta = q.get("theta")
        vega = q.get("vega")
```

### Option Chains

```python
resp = client.option_chains(
    "AAPL",
    contractType="ALL",          # ALL, CALL, or PUT
    strikeCount=20,              # strikes above + below ATM
    fromDate="2026-03-01",       # earliest expiration
    toDate="2026-06-01",         # latest expiration
    includeUnderlyingQuote=True,
)
check_response(resp, "option_chains")
data = resp.json()
```

Response structure:
```json
{
  "symbol": "AAPL",
  "status": "SUCCESS",
  "underlyingPrice": {"last": 150.0, "mark": 150.05},
  "callExpDateMap": {
    "2026-03-20:30": {        // expDate:DTE
      "150.0": [              // strike price
        {
          "putCall": "CALL",
          "symbol": "AAPL  260320C00150000",
          "bid": 5.20,
          "ask": 5.40,
          "delta": 0.52,
          "gamma": 0.03,
          "theta": -0.05,
          "vega": 0.25,
          "volatility": 30.5,     // ← PERCENTAGE, not decimal!
          "openInterest": 1500,
          "strikePrice": 150.0,
          "expirationDate": "2026-03-20"
        }
      ]
    }
  },
  "putExpDateMap": { /* same structure */ }
}
```

**CRITICAL Gotcha — IV Format**: Schwab returns implied volatility as a **percentage** (e.g., `30.5` means 30.5%). You **must** divide by 100 to get the decimal form (`0.305`) before using it in Black-Scholes, simulations, or any math:

```python
raw_vol = contract.get("volatility", 0.0) or 0.0
iv_decimal = raw_vol / 100.0  # 30.5 → 0.305
```

### Price History

```python
resp = client.price_history(
    "AAPL",
    periodType="year",
    period=1,
    frequencyType="daily",
    frequency=1,
)
check_response(resp, "price_history")
data = resp.json()

candles = data.get("candles", [])
# Each candle: {"open", "high", "low", "close", "volume", "datetime"}
# datetime is milliseconds since epoch
```

### Transactions

```python
from datetime import datetime

# Schwab requires datetime objects with microseconds for proper formatting
start_dt = datetime(2025, 1, 1, 0, 0, 0, 1000)  # note: microsecond=1000
end_dt = datetime(2025, 12, 31, 23, 59, 59, 999000)

resp = client.transactions(
    account_hash,
    startDate=start_dt,
    endDate=end_dt,
    types="TRADE,DIVIDEND_OR_INTEREST,RECEIVE_AND_DELIVER,"
          "ACH_RECEIPT,ACH_DISBURSEMENT,CASH_RECEIPT,CASH_DISBURSEMENT,"
          "WIRE_IN,WIRE_OUT,ELECTRONIC_FUND,JOURNAL",
)
check_response(resp, "transactions")
transactions = resp.json()  # list of transaction dicts
```

Each transaction:
```json
{
  "activityId": 12345678,
  "time": "2025-12-10T14:30:00+0000",
  "type": "TRADE",
  "netAmount": -1500.00,
  "description": "Bought 100 AAPL @ 150.00",
  "tradeDate": "2025-12-10T00:00:00+0000",
  "settlementDate": "2025-12-12T00:00:00+0000",
  "orderId": 9876543,
  "transferItems": [
    {
      "instrument": {
        "symbol": "AAPL",
        "assetType": "EQUITY"
      },
      "amount": 100.0,
      "cost": 15000.0,
      "price": 150.0,
      "positionEffect": "OPENING"
    }
  ]
}
```

**Gotcha — Date Format**: `schwabdev` expects `datetime` objects (NOT `date` or ISO strings). It internally formats them as `"2025-01-01T00:00:00.001Z"`. The microsecond is needed so the internal `[:-3]` slice produces correct milliseconds.

**Gotcha — 1-Year Limit**: The API rejects requests spanning more than 365 days. Chunk long ranges:

```python
from datetime import timedelta

current = start_date
while current <= end_date:
    chunk_end = min(current + timedelta(days=365), end_date)
    # fetch (current, chunk_end)
    current = chunk_end + timedelta(days=1)
```

**Gotcha — Valid Transaction Types**: Not all type strings are accepted. These work:
`TRADE`, `DIVIDEND_OR_INTEREST`, `RECEIVE_AND_DELIVER`, `ACH_RECEIPT`, `ACH_DISBURSEMENT`, `CASH_RECEIPT`, `CASH_DISBURSEMENT`, `WIRE_IN`, `WIRE_OUT`, `ELECTRONIC_FUND`, `JOURNAL`

These do **NOT** work (HTTP 400): `TRANSFER`

#### Transaction Type Reference

| Type | What It Is | External Flow? |
|------|-----------|----------------|
| `TRADE` | Buy/sell equities or options | No (internal) |
| `DIVIDEND_OR_INTEREST` | Dividends, interest payments | No (stays in account) |
| `RECEIVE_AND_DELIVER` | Option assignment/exercise, expiration | No (internal) |
| `CASH_RECEIPT` | External deposit (e.g., ACH from bank) | **Yes** |
| `CASH_DISBURSEMENT` | External withdrawal | **Yes** |
| `ACH_RECEIPT` | ACH deposit | **Yes** |
| `ACH_DISBURSEMENT` | ACH withdrawal | **Yes** |
| `WIRE_IN` / `WIRE_OUT` | Wire transfers | **Yes** |
| `ELECTRONIC_FUND` | Electronic fund transfer | **Yes** |
| `JOURNAL` | Internal ledger entries — **mixed** | **Only** if description contains "SCHWAB BANK" (real transfer) or "OVERDRAFT TO INVESTOR CHECKING" (cash swept out). Ignore `TRF FUNDS FRM/TO TYPE 1/2` (bookkeeping pairs that net to zero). |

### Order Preview

Preview a trade's impact on buying power without executing it:

```python
order = {
    "orderType": "LIMIT",
    "session": "NORMAL",
    "duration": "DAY",
    "orderStrategyType": "SINGLE",
    "price": "5.00",
    "orderLegCollection": [
        {
            "instruction": "SELL_TO_OPEN",
            "quantity": 1,
            "instrument": {
                "symbol": "AAPL  260320P00140000",
                "assetType": "OPTION",
            },
        }
    ],
}

resp = client.preview_order(account_hash, order)
if resp.status_code == 200:
    data = resp.json()
    projected_bp = (data.get("orderStrategy", {})
                        .get("orderBalance", {})
                        .get("projectedBuyingPower"))
```

This is read-only — no order is placed. Useful for reverse-engineering margin requirements.

---

## Common Patterns

### Minimal Working Example

```python
import schwabdev

# 1. Connect
client = schwabdev.Client(
    app_key="YOUR_KEY",
    app_secret="YOUR_SECRET",
    callback_url="https://127.0.0.1",
    tokens_db=str(Path.home() / ".schwabdev" / "tokens.db"),
)

# 2. Get account
accounts = client.linked_accounts().json()
account_hash = accounts[0]["hashValue"]

# 3. Get positions
resp = client.account_details(account_hash, fields="positions")
data = resp.json()
positions = data["securitiesAccount"].get("positions", [])
for p in positions:
    sym = p["instrument"]["symbol"]
    qty = p.get("longQuantity", 0) - p.get("shortQuantity", 0)
    val = p.get("marketValue", 0)
    print(f"{sym}: {qty} shares, ${val:,.2f}")

# 4. Get account value
balances = data["securitiesAccount"]["currentBalances"]
print(f"Liquidation value: ${balances['liquidationValue']:,.2f}")
```

### Robust Date Parsing

Schwab returns dates in inconsistent formats. Handle defensively:

```python
from datetime import date, datetime

def parse_schwab_datetime(s: str) -> datetime:
    """Parse Schwab's ISO datetime (may have Z suffix or timezone)."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime.now()

def parse_schwab_date(s: str) -> date | None:
    """Parse Schwab's date string (take first 10 chars)."""
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, AttributeError):
        return None
```

---

## Gotchas Summary

| Issue | Details |
|-------|---------|
| **IV is percentage** | Schwab returns `30.5` meaning 30.5%. Divide by 100 for decimal. |
| **Buying power is 2x** | `buyingPower` = 2 × SMA. Divide by 2 for actual margin. |
| **equity ≠ liquidationValue** | Use `liquidationValue` for true account value. |
| **Positions lack Greeks** | Fetch via `client.quotes()` separately. |
| **Option fields missing** | Parse strike/expiry from OCC symbol as fallback. |
| **Transaction dates** | Must be `datetime` objects with microseconds, not strings. |
| **1-year transaction limit** | Chunk requests into ≤365-day windows. |
| **TRANSFER type rejected** | HTTP 400 — don't include in `types` parameter. |
| **JOURNAL is mixed** | Only "SCHWAB BANK" journals are real external transfers. |
| **Batch quote limit** | ~200 symbols max per `client.quotes()` call. |
| **Token refresh** | Auto-refresh works, but tokens expire after 7 days of inactivity. |
| **Put delta sign** | Schwab may return positive put delta — negate it yourself. |
| **Vega convention** | Schwab vega = price change per 1 **percentage point** IV change, not per decimal unit. |
