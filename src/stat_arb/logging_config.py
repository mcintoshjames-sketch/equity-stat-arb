"""Structured JSON logging configuration.

Provides a ``JSONFormatter`` that emits one-line JSON records suitable for
log aggregation (ELK, CloudWatch, etc.) and a ``setup_logging`` helper
that wires up console + optional file output from ``LoggingConfig``.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stat_arb.config.settings import LoggingConfig


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Fields: ``ts``, ``level``, ``logger``, ``msg``, and optionally ``exc``
    when an exception is attached to the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(config: LoggingConfig) -> None:
    """Configure the root logger from a ``LoggingConfig`` instance.

    * Clears existing handlers to allow re-initialisation.
    * Uses ``JSONFormatter`` when ``config.json_format`` is True.
    * Adds a ``FileHandler`` when ``config.log_file`` is set.
    * Silences noisy third-party loggers (urllib3, schwabdev).
    """
    root = logging.getLogger()
    root.setLevel(config.level.upper())

    # Clear existing handlers
    root.handlers.clear()

    if config.json_format:
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Silence noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("schwabdev").setLevel(logging.WARNING)
