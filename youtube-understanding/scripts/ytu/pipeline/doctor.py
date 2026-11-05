"""`doctor` command: environment diagnostics (research 4.13). Never prints the API key."""

from __future__ import annotations

import importlib.util
import platform
import socket
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .. import __version__, config, proc
from ..config import Config

BINARIES = ("ffmpeg", "ffprobe", "uv", "uvx", "deno", "node")
HOSTS = ("www.youtube.com", "generativelanguage.googleapis.com")
SANDBOX_HINT = ("Allow outbound HTTPS to this host: Claude Code `sandbox.network.allowedDomains`; "
                "Codex `network_access = true` in [sandbox_workspace_write]; Copilot cloud agent custom allowlist.")


def _platform() -> str:
    return f"{platform.system().lower()}-{platform.machine().lower()}"


def _importable(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):  # parent package missing
        return False


def _cache_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".doctor-"):
            pass
        return True
    except OSError:
        return False


def probe_host(host: str, timeout: float) -> dict:
    """DNS + HTTPS HEAD. Any HTTP status counts as reachable (4xx still proves connectivity)."""
    try:
        socket.getaddrinfo(host, 443)
    except OSError as e:
        return {"ok": False, "error": f"DNS failed: {e}", "remediation": SANDBOX_HINT}
    req = urllib.request.Request(f"https://{host}/", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"ok": True, "status": r.status}
    except urllib.error.HTTPError as e:
        return {"ok": True, "status": e.code}
    except (urllib.error.URLError, OSError) as e:
        return {"ok": False, "error": f"HTTPS failed: {getattr(e, 'reason', e)}", "remediation": SANDBOX_HINT}


def run_doctor(cfg: Config, *, network: bool = False) -> dict:
    from ..providers import asr

    bins = {b: proc.which(b) for b in BINARIES}
    if proc.ytdlp_available_inprocess():
        launcher = "in-process"
    elif bins["uvx"]:
        launcher = "uvx"
    else:
        launcher = None
    warnings: list[str] = []
    if launcher is None:
        warnings.append("yt-dlp unavailable: run via `uv run scripts/yt.py` or install uv")
    if not (bins["ffmpeg"] and bins["ffprobe"]):
        warnings.append("ffmpeg/ffprobe missing: frames and local ASR disabled (brew install ffmpeg / apt install ffmpeg)")
    if not (bins["deno"] or bins["node"]) and launcher != "in-process":
        warnings.append("no JS runtime (deno/node): yt-dlp may miss formats; the in-process yt-dlp[deno] extra fixes this")
    if not bins["uv"]:
        warnings.append("uv missing: on-demand local ASR (`uv run --with ...`) unavailable")
    ks = config.key_status()
    if not ks["present"]:
        warnings.append("no Gemini key: transcript + host-frames mode only (looked in " + ", ".join(ks["looked"]) + ")")
    writable = _cache_writable(Path(cfg.cache_dir))
    if not writable:
        warnings.append(f"cache dir not writable: {cfg.cache_dir} (set YTU_CACHE_DIR)")
    lang = cfg.lang or "en"
    out = {
        "python": sys.version.split()[0], "platform": _platform(), "ytu_version": __version__,
        "yt_dlp": {"launcher": launcher, "version": proc.YTDLP_VERSION},
        "binaries": bins, "gemini_key": ks,
        "google_genai": _importable("google.genai"),
        "youtube_transcript_api": _importable("youtube_transcript_api"),
        "cache_dir": str(cfg.cache_dir), "cache_writable": writable,
        "asr_backend": asr.select_backend(platform.system().lower(), platform.machine().lower(), lang),
        "strategy": cfg.strategy, "models": cfg.models, "transcript_providers": cfg.transcript_providers,
        "warnings": warnings,
    }
    if network:
        out["network"] = {h: probe_host(h, cfg.timeout("http")) for h in HOSTS}
        warnings += [f"network: {h}: {r['error']}" for h, r in out["network"].items() if not r["ok"]]
    return out
