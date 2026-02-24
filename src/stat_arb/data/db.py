"""Database engine and session factory.

Provides a thin initialisation layer over SQLAlchemy 2.0.  Call
:func:`init_db` once at application startup (or in test fixtures),
then use :func:`get_session` anywhere a database session is needed.

Pool settings (``pool_pre_ping``, ``pool_size``, ``pool_recycle``) are
applied for connection-pooled backends (PostgreSQL) and silently ignored
for SQLite.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from stat_arb.config.settings import DatabaseConfig

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def init_db(config: DatabaseConfig) -> Engine:
    """Create the global engine and session factory from ``DatabaseConfig``.

    For pooled backends (PostgreSQL), applies connection-health checks
    (``pool_pre_ping``), a 10-connection pool, and 1-hour recycle.
    SQLite ignores pool parameters automatically.

    Returns:
        The SQLAlchemy ``Engine`` instance.
    """
    global _engine, _session_factory

    is_sqlite = config.url.startswith("sqlite")

    connect_args: dict = {}
    pool_kwargs: dict = {}

    if is_sqlite:
        # SQLite needs check_same_thread=False for multi-threaded access
        connect_args["check_same_thread"] = False
    else:
        pool_kwargs.update(
            pool_pre_ping=True,
            pool_size=10,
            pool_recycle=3600,
        )

    _engine = create_engine(
        config.url,
        echo=config.echo,
        connect_args=connect_args,
        **pool_kwargs,
    )
    _session_factory = sessionmaker(bind=_engine)

    logger.info("Database initialised: %s", config.url.split("@")[-1])
    return _engine


def get_engine() -> Engine:
    """Return the singleton ``Engine``, or raise if :func:`init_db` was not called."""
    if _engine is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _engine


def get_session() -> Session:
    """Return a new ``Session`` from the factory.

    Callers are responsible for closing the session when done.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _session_factory()


def create_tables() -> None:
    """Issue ``CREATE TABLE`` DDL for all ORM models.

    Safe to call multiple times — SQLAlchemy's ``create_all`` is
    idempotent and will not drop/recreate existing tables.
    """
    from stat_arb.data.schemas import Base

    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created/verified")
