"""Retry with exponential backoff + jitter (research 4.13): 1-2-4-8 s, max 5 attempts, cap 60 s.

Callers decide retryability via ``should_retry(exc) -> bool``; never retry 400/402/403/404/quota_exceeded.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")
log = logging.getLogger("ytu.retry")


def backoff_delays(attempts: int = 5, base: float = 1.0, cap: float = 60.0, jitter: float = 0.25,
                   rng: Callable[[], float] = random.random) -> list[float]:
    """Delays between attempts (len = attempts - 1)."""
    return [min(cap, base * 2 ** i) * (1 + jitter * rng()) for i in range(max(0, attempts - 1))]


def retry(fn: Callable[[], T], *, should_retry: Callable[[BaseException], bool], attempts: int = 5,
          base: float = 1.0, cap: float = 60.0, sleep: Callable[[float], None] = time.sleep,
          what: str = "call") -> T:
    delays = backoff_delays(attempts, base, cap)
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - re-raised below unless retryable
            if i == attempts - 1 or not should_retry(e):
                raise
            d = min(cap, delays[i])
            log.warning("%s failed (%s); retry %d/%d in %.1fs", what, type(e).__name__, i + 1, attempts - 1, d)
            sleep(d)
    raise AssertionError("unreachable")
