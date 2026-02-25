"""Pydantic v2 configuration models and YAML loader.

Each config subsection is independently injectable — constructors accept their
specific subsection, never the full ``AppConfig``.  All models are frozen
(immutable) after construction to prevent accidental mutation at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator

from stat_arb.config.constants import BrokerMode

# ---------------------------------------------------------------------------
# Base with frozen immutability
# ---------------------------------------------------------------------------

_FROZEN = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Schwab API credentials
# ---------------------------------------------------------------------------


class SchwabConfig(BaseModel):
    """Charles Schwab API connection settings.

    ``app_key`` / ``app_secret`` are wrapped in ``SecretStr`` so they never
    leak into logs or repr output.  ``tokens_db`` supports ``~`` expansion.
    """

    model_config = _FROZEN

    app_key: SecretStr
    app_secret: SecretStr
    callback_url: str = "https://127.0.0.1"
    tokens_db: str = "~/.schwabdev/tokens.db"
    account_hash: str | None = None

    @field_validator("tokens_db")
    @classmethod
    def expand_home(cls, v: str) -> str:
        return str(Path(v).expanduser())


# ---------------------------------------------------------------------------
# Schwab broker execution settings
# ---------------------------------------------------------------------------


class SchwabBrokerConfig(BaseModel):
    """Schwab broker execution configuration.

    Attributes:
        use_limit_orders: Use LIMIT orders instead of MARKET orders.
            Safer for pairs trading where fills at a known price matter.
    """

    model_config = _FROZEN

    use_limit_orders: bool = False


# ---------------------------------------------------------------------------
# Universe / stock pool
# ---------------------------------------------------------------------------


class UniverseConfig(BaseModel):
    """Defines the tradable universe grouped by sector.

    Attributes:
        sectors: Mapping of sector name to list of ticker symbols.
        min_price: Minimum last price to include a symbol (liquidity filter).
        min_avg_volume: Minimum 20-day average volume (liquidity filter).
    """

    model_config = _FROZEN

    sectors: dict[str, list[str]]
    min_price: float = 10.0
    min_avg_volume: int = 1_000_000

    @field_validator("min_price")
    @classmethod
    def _positive_price(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("min_price must be positive")
        return v


# ---------------------------------------------------------------------------
# Pair discovery (Engle-Granger walk-forward)
# ---------------------------------------------------------------------------


class DiscoveryConfig(BaseModel):
    """Parameters for the cointegration-based pair discovery pipeline.

    Attributes:
        formation_days: Lookback window for the Engle-Granger test.
        coint_pvalue: Maximum p-value for Engle-Granger cointegration test.
        adf_pvalue: Maximum p-value for ADF stationarity test on residuals.
        min_half_life_days: Minimum OU half-life (filter out ultra-fast noise).
        max_half_life_days: Maximum OU half-life (filter out slow-reverting pairs).
        max_hurst: Maximum Hurst exponent (< 0.5 = mean-reverting).
        min_correlation: Pre-filter correlation threshold before running coint test.
        parallel_n_jobs: ``joblib`` concurrency; -1 = all cores.
    """

    model_config = _FROZEN

    formation_days: int = 252
    coint_pvalue: float = 0.05
    adf_pvalue: float = 0.05
    min_half_life_days: int = 5
    max_half_life_days: int = 30
    max_hurst: float = 0.5
    min_correlation: float = 0.78
    parallel_n_jobs: int = -1
    use_ols_fallback: bool = True
    min_common_obs: int = 60

    @field_validator("min_common_obs")
    @classmethod
    def _positive_obs(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("min_common_obs must be positive")
        return v

    @field_validator("coint_pvalue", "adf_pvalue")
    @classmethod
    def _pvalue_range(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("p-value must be in (0, 1)")
        return v

    @field_validator("max_hurst")
    @classmethod
    def _hurst_range(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("max_hurst must be in (0, 1]")
        return v

    @field_validator("min_half_life_days", "max_half_life_days")
    @classmethod
    def _positive_half_life(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("half-life days must be positive")
        return v

    @model_validator(mode="after")
    def _half_life_ordering(self) -> DiscoveryConfig:
        if self.min_half_life_days >= self.max_half_life_days:
            raise ValueError("min_half_life_days must be < max_half_life_days")
        return self


# ---------------------------------------------------------------------------
# Signal generation (z-score thresholds)
# ---------------------------------------------------------------------------


class SignalConfig(BaseModel):
    """Z-score thresholds for entry, exit, and stop-loss signals.

    Attributes:
        entry_z: Absolute z-score to open a position (long or short spread).
        exit_z: Absolute z-score to close — spread has mean-reverted enough.
        stop_z: Absolute z-score for divergence stop-loss.
        timeout_half_life_mult: Close after this many half-lives if not exited.
        slippage_multiplier: Fraction of quoted half-spread used as fill
            slippage.  0.5 = fill at mid ± 0.5 × half_spread (conservative
            default reflecting real-world bid-ask for less-liquid names).
    """

    model_config = _FROZEN

    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    timeout_half_life_mult: float = 3.0
    slippage_multiplier: float = 0.5
    adaptive_vol: bool = False
    adaptive_vol_window: int = 20

    @field_validator("adaptive_vol_window")
    @classmethod
    def _min_vol_window(cls, v: int) -> int:
        if v < 2:
            raise ValueError("adaptive_vol_window must be >= 2")
        return v

    @field_validator(
        "entry_z", "exit_z", "stop_z",
        "timeout_half_life_mult", "slippage_multiplier",
    )
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("z-score thresholds and timeout multiplier must be positive")
        return v

    @model_validator(mode="after")
    def _z_ordering(self) -> SignalConfig:
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be < entry_z")
        if self.entry_z >= self.stop_z:
            raise ValueError("entry_z must be < stop_z")
        return self


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


class SizingConfig(BaseModel):
    """Dollar-based position sizing per pair leg.

    Attributes:
        dollars_per_leg: Notional allocation to each leg of the pair.
        max_gross_per_pair: Maximum combined notional for both legs.
    """

    model_config = _FROZEN

    dollars_per_leg: float = 1500.0
    max_gross_per_pair: float = 3000.0

    @field_validator("dollars_per_leg", "max_gross_per_pair")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("sizing amounts must be positive")
        return v


# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------


class RiskConfig(BaseModel):
    """Portfolio-level risk constraints.

    Attributes:
        max_pairs: Maximum number of simultaneously active pairs.
        max_gross_exposure: Maximum total notional across all positions.
        max_sector_pct: Maximum fraction of gross exposure in one sector.
        max_drawdown_pct: Circuit breaker — halt trading at this drawdown.
        min_edge_over_slippage: Minimum expected edge above estimated slippage.
        max_entries_per_step: Maximum new entries allowed in a single step.
        per_pair_pnl_stop: Dollar loss threshold to force-close a pair.
        max_cohort_concentration: Maximum pairs from any single discovery cohort.
    """

    model_config = _FROZEN

    max_pairs: int = 10
    max_gross_exposure: float = 25_000.0
    max_sector_pct: float = 0.30
    max_drawdown_pct: float = 0.10
    min_edge_over_slippage: float = 0.0
    structural_break_window: int = 60
    structural_break_pvalue: float = 0.10
    max_entries_per_step: int = 3
    per_pair_pnl_stop: float = -200.0
    max_cohort_concentration: int = 5

    @field_validator("max_sector_pct", "max_drawdown_pct")
    @classmethod
    def _fraction_range(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("percentage must be in (0, 1]")
        return v

    @field_validator("max_pairs", "max_entries_per_step", "max_cohort_concentration")
    @classmethod
    def _positive_pairs(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_pairs must be positive")
        return v


# ---------------------------------------------------------------------------
# Rolling scheduler
# ---------------------------------------------------------------------------


class RollingSchedulerConfig(BaseModel):
    """Rolling discovery scheduler configuration.

    Attributes:
        formation_days: Trailing lookback window for cointegration discovery.
        trading_days: Per-pair trading lifetime in business days.
        discovery_interval_days: How often to re-run discovery (business days).
        max_cohort_pairs: Maximum new pairs discovered per cycle.
    """

    model_config = _FROZEN

    formation_days: int = 252
    trading_days: int = 63
    discovery_interval_days: int = 21
    max_cohort_pairs: int = 5

    @field_validator("formation_days", "trading_days",
                     "discovery_interval_days", "max_cohort_pairs")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("rolling scheduler fields must be positive")
        return v

    @model_validator(mode="after")
    def _interval_le_trading(self) -> RollingSchedulerConfig:
        if self.discovery_interval_days > self.trading_days:
            raise ValueError(
                "discovery_interval_days must be <= trading_days"
            )
        return self


# ---------------------------------------------------------------------------
# Walk-forward window
# ---------------------------------------------------------------------------


class FmpConfig(BaseModel):
    """Financial Modeling Prep API settings for earnings data.

    Attributes:
        api_key: FMP API key (optional — no blackout if absent).
        earnings_blackout_days: Business days before earnings to block/exit.
        cache_dir: Directory for persistent earnings cache (JSON).
    """

    model_config = _FROZEN

    api_key: SecretStr | None = None
    earnings_blackout_days: int = 3
    cache_dir: str = "~/.stat_arb"

    @field_validator("earnings_blackout_days")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("earnings_blackout_days must be >= 0")
        return v

    @field_validator("cache_dir")
    @classmethod
    def _expand_home(cls, v: str) -> str:
        return str(Path(v).expanduser())


# ---------------------------------------------------------------------------
# Walk-forward window
# ---------------------------------------------------------------------------


class WalkForwardConfig(BaseModel):
    """Walk-forward formation/trading window durations.

    Attributes:
        formation_days: Business days in the formation (estimation) window.
        trading_days: Business days in the trading (out-of-sample) window.
    """

    model_config = _FROZEN

    formation_days: int = 252
    trading_days: int = 63

    @field_validator("formation_days", "trading_days")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("window days must be positive")
        return v


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class DatabaseConfig(BaseModel):
    """SQLAlchemy database connection settings.

    Defaults to SQLite for development; set ``url`` to a PostgreSQL DSN
    for production (``postgresql+psycopg2://...``).
    """

    model_config = _FROZEN

    url: str = "sqlite:///stat_arb.db"
    echo: bool = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class LoggingConfig(BaseModel):
    """Application logging configuration.

    Attributes:
        level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Emit structured JSON logs when True.
        log_file: Optional path to a log file; ``None`` = console only.
    """

    model_config = _FROZEN

    level: str = "INFO"
    json_format: bool = True
    log_file: str | None = None


# ---------------------------------------------------------------------------
# Top-level application config
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Root configuration composing all subsections.

    Each subsection is independently injectable — pass ``config.discovery``
    to the discovery module, ``config.risk`` to the risk module, etc.
    Never pass the full ``AppConfig`` into a subsystem constructor.
    """

    model_config = _FROZEN

    schwab: SchwabConfig
    universe: UniverseConfig
    discovery: DiscoveryConfig = DiscoveryConfig()
    signal: SignalConfig = SignalConfig()
    sizing: SizingConfig = SizingConfig()
    risk: RiskConfig = RiskConfig()
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    rolling: RollingSchedulerConfig = RollingSchedulerConfig()
    fmp: FmpConfig = FmpConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
    broker_mode: BrokerMode = BrokerMode.PAPER


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> AppConfig:
    """Load and validate ``AppConfig`` from a YAML file.

    Environment variable overrides (applied after YAML loading):
      - ``SCHWAB_APP_KEY`` → ``schwab.app_key``
      - ``SCHWAB_APP_SECRET`` → ``schwab.app_secret``
      - ``SCHWAB_TOKENS_DB`` → ``schwab.tokens_db``
      - ``STAT_ARB_DB_URL`` → ``database.url``
      - ``BROKER_MODE`` → ``broker_mode``

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        Fully validated, immutable ``AppConfig`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        pydantic.ValidationError: If YAML contents fail validation.
    """
    import os

    path = Path(path)
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # Allow env var overrides for credentials and paths
    schwab = raw.get("schwab", {})
    if os.environ.get("SCHWAB_APP_KEY"):
        schwab["app_key"] = os.environ["SCHWAB_APP_KEY"]
    if os.environ.get("SCHWAB_APP_SECRET"):
        schwab["app_secret"] = os.environ["SCHWAB_APP_SECRET"]
    if os.environ.get("SCHWAB_TOKENS_DB"):
        schwab["tokens_db"] = os.environ["SCHWAB_TOKENS_DB"]

    # Allow env var override for database URL (e.g. Docker → /data/stat_arb.db)
    if os.environ.get("STAT_ARB_DB_URL"):
        db = raw.setdefault("database", {})
        db["url"] = os.environ["STAT_ARB_DB_URL"]

    # Allow env var override for broker mode
    if os.environ.get("BROKER_MODE"):
        raw["broker_mode"] = os.environ["BROKER_MODE"]

    # Allow env var override for FMP API key
    if os.environ.get("FMP_API_KEY"):
        fmp = raw.setdefault("fmp", {})
        fmp["api_key"] = os.environ["FMP_API_KEY"]

    return AppConfig(**raw)
