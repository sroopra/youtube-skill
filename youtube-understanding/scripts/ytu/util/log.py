"""Logging: everything to stderr under the "ytu" logger; stdout is reserved for JSON."""

from __future__ import annotations

import logging
import sys


class _RedactingFormatter(logging.Formatter):
    """Formats the record (message, traceback, stack) and then scrubs secrets from the result."""

    def format(self, record: logging.LogRecord) -> str:
        from ..proc import redact  # lazy: proc imports errors, keep log.py import-light
        return redact(super().format(record))


def setup_logging(verbosity: int = 0) -> logging.Logger:
    """0 -> WARNING, 1 (-v) -> INFO, 2+ (-vv) -> DEBUG."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logger = logging.getLogger("ytu")
    logger.setLevel(level)
    if not any(getattr(h, "_ytu", False) for h in logger.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(_RedactingFormatter("ytu %(levelname)s %(name)s: %(message)s"))
        h._ytu = True  # type: ignore[attr-defined]
        logger.addHandler(h)
    return logger
