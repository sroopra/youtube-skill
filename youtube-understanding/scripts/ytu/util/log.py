"""Logging: everything to stderr under the "ytu" logger; stdout is reserved for JSON."""

from __future__ import annotations

import logging
import sys


def setup_logging(verbosity: int = 0) -> logging.Logger:
    """0 -> WARNING, 1 (-v) -> INFO, 2+ (-vv) -> DEBUG."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logger = logging.getLogger("ytu")
    logger.setLevel(level)
    if not any(getattr(h, "_ytu", False) for h in logger.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("ytu %(levelname)s %(name)s: %(message)s"))
        h._ytu = True  # type: ignore[attr-defined]
        logger.addHandler(h)
    return logger
