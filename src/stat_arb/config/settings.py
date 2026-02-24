"""Pydantic v2 configuration models and YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, SecretStr, field_validator

from stat_arb.config.constants import BrokerMode


class SchwabConfig(BaseModel):
    app_key: SecretStr
    app_secret: SecretStr
    callback_url: str = "https://127.0.0.1"
    tokens_db: str = "~/.schwabdev/tokens.db"
    account_hash: str | None = None

    @field_validator("tokens_db")
    @classmethod
    def expand_home(cls, v: str) -> str:
        return str(Path(v).expanduser())


class UniverseConfig(BaseModel):
    sectors: dict[str, list[str]]  # sector_name → [symbols]
    min_price: float = 10.0
    min_avg_volume: int = 500_000


class DiscoveryConfig(BaseModel):
    formation_days: int = 252
    coint_pvalue: float = 0.05
    adf_pvalue: float = 0.05
    min_half_life_days: int = 5
    max_half_life_days: int = 30
    max_hurst: float = 0.5
    min_correlation: float = 0.78  # pre-filter before coint test
    parallel_n_jobs: int = -1      # joblib parallelism (-1 = all cores)


class SignalConfig(BaseModel):
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    timeout_half_life_mult: float = 3.0


class SizingConfig(BaseModel):
    dollars_per_leg: float = 1500.0
    max_gross_per_pair: float = 3000.0


class RiskConfig(BaseModel):
    max_pairs: int = 10
    max_gross_exposure: float = 25_000.0
    max_sector_pct: float = 0.30
    max_drawdown_pct: float = 0.10
    min_edge_over_slippage: float = 0.0  # require edge > slippage


class WalkForwardConfig(BaseModel):
    formation_days: int = 252
    trading_days: int = 63  # ~3 months


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///stat_arb.db"
    echo: bool = False


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_format: bool = True
    log_file: str | None = None


class AppConfig(BaseModel):
    schwab: SchwabConfig
    universe: UniverseConfig
    discovery: DiscoveryConfig = DiscoveryConfig()
    signal: SignalConfig = SignalConfig()
    sizing: SizingConfig = SizingConfig()
    risk: RiskConfig = RiskConfig()
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
    broker_mode: BrokerMode = BrokerMode.PAPER


def load_config(path: str | Path) -> AppConfig:
    """Load AppConfig from a YAML file."""
    path = Path(path)
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return AppConfig(**raw)
