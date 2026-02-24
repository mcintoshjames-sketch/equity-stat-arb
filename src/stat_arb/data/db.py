"""Database engine and session factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from stat_arb.config.settings import DatabaseConfig

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def init_db(config: DatabaseConfig) -> Engine:
    """Create the global engine and session factory. Returns the engine."""
    global _engine, _session_factory
    _engine = create_engine(config.url, echo=config.echo)
    _session_factory = sessionmaker(bind=_engine)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _engine


def get_session() -> Session:
    if _session_factory is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _session_factory()


def create_tables() -> None:
    """Create all ORM tables."""
    from stat_arb.data.schemas import Base

    engine = get_engine()
    Base.metadata.create_all(engine)
