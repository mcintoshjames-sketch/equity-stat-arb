# Equity Stat Arb — Agent Instructions

## Project
Python-based equity pairs trading system. Discovers cointegrated pairs via walk-forward Engle-Granger testing, trades mean reversion through the Charles Schwab API (`schwabdev`).

## Build & Test
```bash
pip install -e ".[dev]"          # install with dev deps
pytest tests/unit/ -v            # unit tests (no I/O)
pytest tests/integration/ -v     # DB + mocked API tests
pytest tests/ -v                 # all tests
ruff check src/ tests/           # lint
```

## Architecture
- **Package**: `src/stat_arb/` — all imports are `from stat_arb.<module> import ...`
- **Config**: Pydantic v2 models in `config/settings.py`, loaded from `config/default.yaml`
- **Config injection**: constructors take their Pydantic config subsection, never the full `AppConfig`
- **Brokers**: Strategy pattern — `PaperBroker` / `LiveSchwabBroker` / `SimBroker` via `broker_mode` config
- **Formation params**: `QualifiedPair` is immutable — μ/σ/β frozen during trading window
- **DB**: SQLAlchemy 2.0 ORM, SQLite for dev, PostgreSQL for prod

## Conventions
- Python 3.11+, type hints on all public functions
- Enums in `config/constants.py`, not magic strings
- Schwab API: always `check_response()` after every call, batch quotes in groups of 200
- Tests: `tests/unit/` (pure logic), `tests/integration/` (DB + mock API), `tests/backtest/`
