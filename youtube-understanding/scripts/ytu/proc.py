"""Subprocess wrapper. ALL external commands go through :func:`run` (tests monkeypatch
``ytu.proc.run``). No shell, always a timeout, output captured, secrets redacted.

yt-dlp invocation decision (see :func:`ytdlp_cmd`):
  1. ``<sys.executable> -I -m yt_dlp`` when yt_dlp is importable in the current env
     (the normal case: scripts/yt.py's PEP 723 deps pin it). Same pinned version,
     no extra resolver round-trip, ``-I`` keeps cwd / PYTHON* env out of sys.path.
  2. else ``uvx --from "yt-dlp[default,deno,curl-cffi]==<YTDLP_VERSION>" yt-dlp``.
  3. else MissingDependency.
Always followed by YTDLP_BASE_ARGS (--ignore-config etc.).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .errors import BadInput, MissingDependency, ProcessFailed, SubprocessTimeout

log = logging.getLogger("ytu.proc")

YTDLP_VERSION = "2026.10.26"
YTDLP_SPEC = f"yt-dlp[default,deno,curl-cffi]=={YTDLP_VERSION}"
YTDLP_BASE_ARGS = ["--ignore-config", "--no-playlist", "--no-progress", "--color", "never"]

_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|COOKIE|CREDENTIAL|AUTH)", re.I)


@dataclass
class RunResult:
    cmd: list[str]
    returncode: int
    stdout: str | bytes
    stderr: str
    duration_s: float


def _secret_values(*envs: Mapping[str, str] | None) -> list[str]:
    vals = set()
    for env in envs:
        for k, v in (env or {}).items():
            if v and len(v) >= 6 and _SECRET_NAME.search(k):
                vals.add(v)
    return sorted(vals, key=len, reverse=True)


_EXTRA_SECRETS: set[str] = set()
_GOOGLEVIDEO = re.compile(r"https?://[^\s\"'<>]*googlevideo\.com/[^\s\"'<>]*", re.I)
_SECRET_PARAM = re.compile(r"(?i)\b(key|token|api_key|sig|lsig|signature|ip|ei|n|pot)=([^&\s\"']+)")


def add_secret(value: str | None) -> None:
    """Register a secret (e.g. a key loaded from a file) so every redact() call scrubs it."""
    if value and len(value) >= 6:
        _EXTRA_SECRETS.add(value)


def redact(text: str, env: Mapping[str, str] | None = None, secrets=()) -> str:
    """Replace secret values (secret-named vars in os.environ + ``env``, add_secret() values,
    ``secrets``), whole signed googlevideo.com stream URLs, and secret/IP-bearing query params."""
    if not text:
        return text
    extra = {v for v in (*_EXTRA_SECRETS, *(secrets or ())) if v and len(v) >= 6}
    for v in sorted(set(_secret_values(os.environ, env)) | extra, key=len, reverse=True):
        text = text.replace(v, "***")
    text = _GOOGLEVIDEO.sub("<googlevideo-url>", text)
    return _SECRET_PARAM.sub(r"\1=***", text)


def child_env(env: Mapping[str, str] | None = None, scrub_secrets: bool = True) -> dict[str, str]:
    """os.environ overlaid with ``env``; secret-looking vars dropped unless explicitly passed in ``env``."""
    base = {k: v for k, v in os.environ.items() if not (scrub_secrets and _SECRET_NAME.search(k))}
    base.update(env or {})
    return base


def run(cmd: Sequence[str], *, timeout: float, env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None, check: bool = True, input: str | bytes | None = None,
        text: bool = True, scrub_secrets: bool = True) -> RunResult:
    """Run ``cmd`` (list, no shell). Raises:
    MissingDependency (exe not found), SubprocessTimeout, ProcessFailed (rc != 0 and check).
    ``text=False`` returns stdout as bytes (stderr is always decoded, errors replaced)."""
    if isinstance(cmd, (str, bytes)) or not cmd:
        raise BadInput("proc.run needs a non-empty argument list (no shell strings)")
    cmd = [str(c) for c in cmd]
    log.debug("run: %s", redact(" ".join(cmd), env))
    data = input.encode("utf-8") if isinstance(input, str) else input
    t0 = time.monotonic()
    try:
        kw = {"input": data} if data is not None else {"stdin": subprocess.DEVNULL}
        cp = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=cwd,
                            env=child_env(env, scrub_secrets), **kw)
    except FileNotFoundError:
        raise MissingDependency(f"executable not found: {cmd[0]}", f"Install {Path(cmd[0]).name} (see `yt.py doctor`).") from None
    except subprocess.TimeoutExpired:
        raise SubprocessTimeout(f"{Path(cmd[0]).name} timed out after {timeout:.0f}s", timeout_s=timeout) from None
    stderr = redact(cp.stderr.decode("utf-8", "replace"), env)
    stdout: str | bytes = cp.stdout.decode("utf-8", "replace") if text else cp.stdout
    res = RunResult(cmd, cp.returncode, stdout, stderr, time.monotonic() - t0)
    if check and cp.returncode != 0:
        raise ProcessFailed(f"{Path(cmd[0]).name} exited {cp.returncode}", cmd0=Path(cmd[0]).name,
                            returncode=cp.returncode, stderr_tail=stderr[-2000:])
    return res


def which(name: str) -> str | None:
    return shutil.which(name)


def require(name: str, remediation: str | None = None) -> str:
    p = which(name)
    if not p:
        raise MissingDependency(f"{name} not found on PATH", remediation or f"Install {name} and retry.")
    return p


def ytdlp_available_inprocess() -> bool:
    return importlib.util.find_spec("yt_dlp") is not None


def ytdlp_cmd(*args: str) -> list[str]:
    """Full argv for yt-dlp: [launcher..., *YTDLP_BASE_ARGS, *args]. Pass cookies etc. in args."""
    if ytdlp_available_inprocess():
        launcher = [sys.executable, "-I", "-m", "yt_dlp"]
    elif which("uvx"):
        launcher = ["uvx", "--from", YTDLP_SPEC, "yt-dlp"]
    else:
        raise MissingDependency("yt-dlp not available",
                                "Run via `uv run scripts/yt.py ...` (installs pinned deps) or install uv: "
                                "https://docs.astral.sh/uv/")
    return [*launcher, *YTDLP_BASE_ARGS, *args]


def ytdlp_env() -> dict[str, str]:
    """Extra env for yt-dlp: put the current venv's bin dir first on PATH so the
    `deno` wheel binary (yt-dlp[deno] extra) is found as the JS runtime."""
    bindir = str(Path(sys.executable).parent)
    return {"PATH": os.pathsep.join([bindir, os.environ.get("PATH", "")])}
