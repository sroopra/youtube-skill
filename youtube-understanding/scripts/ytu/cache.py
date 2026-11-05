"""Content-addressed cache.

Layout: ``<root>/<namespace>/<key[:2]>/<key><suffix>`` where
``key = sha256(canonical_json(parts))`` and ``parts`` is a JSON-able dict that
fully describes the input (video id, provider, tool version, config, prompt...).

JSON entries are stored as an envelope ``{"created_at": epoch, "parts": parts, "value": value}``.
File entries (frames, audio) are stored as-is; TTL uses mtime.
``force=True`` makes every ``get*`` miss (writes still happen) -> --force-refresh.
Never put secrets or signed googlevideo/timedtext URLs in ``parts`` or values.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .util.paths import atomic_write_bytes, atomic_write_text

log = logging.getLogger("ytu.cache")

# namespace -> TTL seconds (None = forever)
NAMESPACES: dict[str, float | None] = {
    "metadata": 24 * 3600,
    "transcript": None,
    "frames": None,
    "gemini": None,
    "probe": 7 * 24 * 3600,   # feature-detection results, e.g. agentic_unavailable
    "usage": None,            # free-tier hours tracker
}
_NS_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_SUFFIX_RE = re.compile(r"^\.[a-z0-9]{1,8}$")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def cache_key(parts: dict) -> str:
    return hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, root: Path, force: bool = False):
        self.root = Path(root)
        self.force = force

    def _ns(self, namespace: str) -> str:
        if not _NS_RE.match(namespace):
            raise ValueError(f"bad cache namespace {namespace!r}")
        return namespace

    def path_for(self, namespace: str, parts: dict, suffix: str = ".json") -> Path:
        if not _SUFFIX_RE.match(suffix):
            raise ValueError(f"bad cache suffix {suffix!r}")
        key = cache_key(parts)
        return self.root / self._ns(namespace) / key[:2] / f"{key}{suffix}"

    def _expired(self, namespace: str, created: float) -> bool:
        ttl = NAMESPACES.get(namespace)
        return ttl is not None and (time.time() - created) > ttl

    # JSON ---------------------------------------------------------------------
    def get(self, namespace: str, parts: dict) -> Any | None:
        """Return cached value or None (miss, expired, corrupt or force)."""
        if self.force:
            return None
        p = self.path_for(namespace, parts)
        try:
            env = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            log.warning("corrupt cache entry %s; ignoring", p)
            return None
        if self._expired(namespace, float(env.get("created_at", 0))):
            return None
        return env.get("value")

    def put(self, namespace: str, parts: dict, value: Any) -> Path:
        p = self.path_for(namespace, parts)
        env = {"created_at": time.time(), "parts": parts, "value": value}
        return atomic_write_text(p, json.dumps(env, ensure_ascii=False, allow_nan=False), mode=0o600)

    # Files --------------------------------------------------------------------
    def get_file(self, namespace: str, parts: dict, suffix: str) -> Path | None:
        if self.force:
            return None
        p = self.path_for(namespace, parts, suffix)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return None
        return None if self._expired(namespace, mtime) else p

    def put_file(self, namespace: str, parts: dict, src: Path | bytes, suffix: str, move: bool = True) -> Path:
        """Store bytes or an existing file (moved by default) atomically; returns cache path."""
        p = self.path_for(namespace, parts, suffix)
        if isinstance(src, (bytes, bytearray)):
            return atomic_write_bytes(p, bytes(src))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp")
        (shutil.move if move else shutil.copy2)(str(src), str(tmp))
        tmp.replace(p)
        return p
