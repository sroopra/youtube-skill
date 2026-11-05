"""Safe filesystem helpers: filenames from untrusted text, containment checks, atomic writes."""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from pathlib import Path

from ..errors import BadInput

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_YT_PY = Path(__file__).resolve().parents[2] / "yt.py"


def yt_cmd() -> str:
    """Runnable command prefix for host-agent instructions: `uv run <abs>/scripts/yt.py`."""
    import shlex
    return f"uv run {shlex.quote(str(_YT_PY))}"


def safe_component(name: str, max_len: int = 80, default: str = "item") -> str:
    """Reduce arbitrary (untrusted) text to a single safe path component."""
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = _UNSAFE.sub("-", s).strip(".-_")[:max_len].strip(".-_")
    return s or default


def ensure_within(base: Path, target: Path) -> Path:
    """Resolve ``target`` and raise BadInput unless it is inside ``base``."""
    b, t = Path(base).resolve(), Path(target).resolve()
    if t != b and b not in t.parents:
        raise BadInput(f"path escapes {b}: {t}")
    return t


def frame_filename(abs_ts: float, ext: str = "jpg") -> str:
    """754.2 -> "t0754200.jpg" (milliseconds, zero-padded to 7)."""
    return f"t{int(round(abs_ts * 1000)):07d}.{ext}"


def atomic_write_bytes(path: Path, data: bytes, mode: int = 0o644) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def atomic_write_text(path: Path, text: str, mode: int = 0o644) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"), mode)
