"""Configuration and API-key loading.

Precedence (low -> high): defaults < ~/.config/youtube-understanding/config.toml
(override path: YTU_CONFIG_FILE) < environment < CLI overrides.

config.toml holds NON-SECRET settings only (keys equal Config field names, e.g.
``models = ["gemini-3.8-flash"]``, ``max_cost_usd = 1.0``, ``[timeouts] metadata = 90``).
Secrets are loaded only via :func:`get_gemini_key`.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import stat
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .errors import BadInput, MissingKey

log = logging.getLogger("ytu.config")

DEFAULT_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite"]
DEFAULT_TRANSCRIPT_PROVIDERS = ["ytdlp", "yta", "asr"]
KNOWN_TRANSCRIPT_PROVIDERS = {"ytdlp", "yta", "asr"}
STRATEGIES = ("auto", "transcript", "gemini", "local")
KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
DEFAULT_TIMEOUTS = {
    "metadata": 60.0,      # yt-dlp -J
    "subtitles": 60.0,     # one caption track download
    "frame": 90.0,         # one remote ffmpeg seek
    "section": 600.0,      # --download-sections clip
    "asr": 3600.0,         # local ASR run
    "gemini": 900.0,       # one Gemini request incl. polling
    "http": 30.0,          # misc HTTP probes
}


def _config_home(env: Mapping[str, str]) -> Path:
    return Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "youtube-understanding"


def default_env_file(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env["YTU_ENV_FILE"]).expanduser() if env.get("YTU_ENV_FILE") else _config_home(env) / ".env"


def default_config_file(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env["YTU_CONFIG_FILE"]).expanduser() if env.get("YTU_CONFIG_FILE") else _config_home(env) / "config.toml"


def _writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=p, prefix=".probe-"):
            pass
        return True
    except OSError:
        return False


def resolve_cache_dir(env: Mapping[str, str] | None = None, check: bool = True) -> Path:
    """${YTU_CACHE_DIR:-${XDG_CACHE_HOME:-~/.cache}/youtube-understanding}; falls back to
    $TMPDIR/youtube-understanding-<uid> when not writable (sandboxes)."""
    env = os.environ if env is None else env
    if env.get("YTU_CACHE_DIR"):
        p = Path(env["YTU_CACHE_DIR"]).expanduser()
    else:
        p = Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser() / "youtube-understanding"
    if not check or _writable_dir(p):
        return p
    fb = Path(env.get("TMPDIR") or tempfile.gettempdir()) / f"youtube-understanding-{os.getuid()}"
    log.warning("cache dir %s not writable; using %s", p, fb)
    try:
        fb.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:
        pass
    if not private_dir_ok(fb):  # predictable name in a shared dir: never trust one we do not own
        fb = Path(tempfile.mkdtemp(prefix="youtube-understanding-"))
        log.warning("fallback cache dir not private (owner/mode/symlink); using fresh %s", fb)
    return fb


def cache_tmp_root(cache_dir: Path) -> Path:
    """<cache>/tmp created 0700; refuses a symlink or a dir owned by someone else."""
    p = Path(cache_dir) / "tmp"
    p.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = os.lstat(p)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise BadInput(f"cache temp dir {p} is a symlink or not owned by you",
                       "Remove it or set YTU_CACHE_DIR to a private directory.")
    return p


def private_dir_ok(p: Path) -> bool:
    """True iff ``p`` is a real directory (not a symlink) owned by us with no group/other access."""
    try:
        st = os.lstat(p)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and not st.st_mode & 0o077


def _split_list(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass
class Config:
    cache_dir: Path
    out_root: Path = Path("youtube-knowledge")        # package dir = out_root/<video_id> unless --out
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    transcript_providers: list[str] = field(default_factory=lambda: list(DEFAULT_TRANSCRIPT_PROVIDERS))
    strategy: str = "auto"                            # auto|transcript|gemini|local
    lang: str | None = None                           # preferred transcript language (base code)
    max_cost_usd: float = 0.50
    max_input_tokens: int = 800_000
    max_frames: int = 24
    frame_height: int = 720
    timeouts: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TIMEOUTS))
    force_refresh: bool = False
    dry_run: bool = False
    gemini_tier: str = "paid"                         # "free" enables the 8 h/day local tracker
    cookies: Path | None = None                       # passthrough to yt-dlp, opt-in only
    retry_attempts: int = 5

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise BadInput(f"unknown strategy {self.strategy!r}", f"Use one of: {', '.join(STRATEGIES)}.")
        bad = [p for p in self.transcript_providers if p not in KNOWN_TRANSCRIPT_PROVIDERS]
        if bad:
            raise BadInput(f"unknown transcript provider(s): {bad}",
                           f"Known: {sorted(KNOWN_TRANSCRIPT_PROVIDERS)}.")
        if not self.models:
            raise BadInput("empty model list", "Set YTU_GEMINI_MODELS or --models.")
        if self.max_cost_usd < 0 or self.max_input_tokens <= 0:
            raise BadInput("budget limits must be positive")

    def timeout(self, name: str) -> float:
        return float(self.timeouts.get(name, DEFAULT_TIMEOUTS.get(name, 120.0)))

    def out_dir_for(self, video_id: str) -> Path:
        return self.out_root / video_id

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @classmethod
    def load(cls, overrides: Mapping | None = None, env: Mapping[str, str] | None = None,
             config_file: Path | None = None) -> "Config":
        """Build a Config. ``overrides`` (CLI) keys = field names; None values are ignored."""
        env = os.environ if env is None else env
        vals: dict = {}
        cf = config_file or default_config_file(env)
        if cf.is_file():
            try:
                data = tomllib.loads(cf.read_text(encoding="utf-8"))
            except (tomllib.TOMLDecodeError, OSError) as e:
                raise BadInput(f"invalid config file {cf}: {e}", "Fix or remove the config.toml.") from None
            names = {f.name for f in dataclasses.fields(cls)}
            for k, v in data.items():
                if k in names:
                    vals[k] = v
                else:
                    log.warning("ignoring unknown config key %r in %s", k, cf)
        # environment
        if env.get("YTU_GEMINI_MODELS"):
            vals["models"] = _split_list(env["YTU_GEMINI_MODELS"])
        if env.get("YTU_GEMINI_MODEL"):  # single model pinned first, ladder keeps the rest
            first = env["YTU_GEMINI_MODEL"].strip()
            vals["models"] = [first] + [m for m in vals.get("models", DEFAULT_MODELS) if m != first]
        if env.get("YTU_TRANSCRIPT_PROVIDERS"):
            vals["transcript_providers"] = _split_list(env["YTU_TRANSCRIPT_PROVIDERS"])
        for key, name, conv in (("YTU_MAX_COST_USD", "max_cost_usd", float),
                                ("YTU_MAX_INPUT_TOKENS", "max_input_tokens", int),
                                ("YTU_STRATEGY", "strategy", str),
                                ("YTU_OUT_DIR", "out_root", Path),
                                ("YTU_GEMINI_TIER", "gemini_tier", str)):
            if env.get(key):
                try:
                    vals[name] = conv(env[key])
                except ValueError:
                    raise BadInput(f"invalid {key}={env[key]!r}") from None
        timeouts = dict(DEFAULT_TIMEOUTS)
        timeouts.update(vals.pop("timeouts", {}) or {})
        for name in DEFAULT_TIMEOUTS:
            ev = env.get(f"YTU_TIMEOUT_{name.upper()}")
            if ev:
                try:
                    timeouts[name] = float(ev)
                except ValueError:
                    raise BadInput(f"invalid YTU_TIMEOUT_{name.upper()}={ev!r}") from None
        if env.get("YTU_FORCE_REFRESH") in ("1", "true", "yes"):
            vals["force_refresh"] = True
        # CLI
        for k, v in (overrides or {}).items():
            if v is None:
                continue
            if k == "timeouts":
                timeouts.update(v)
            else:
                vals[k] = v
        vals["timeouts"] = timeouts
        if "cache_dir" in vals:
            vals["cache_dir"] = Path(vals["cache_dir"]).expanduser()
        else:
            vals["cache_dir"] = resolve_cache_dir(env)
        for k in ("out_root", "cookies"):
            if vals.get(k) is not None:
                vals[k] = Path(vals[k]).expanduser()
        return cls(**vals)


# --- API key ------------------------------------------------------------------------

def parse_env_file(text: str) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, optional ``export ``, # comments,
    single/double-quoted values (no interpolation, no multi-line)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, val = line.partition("=")
        key = key.strip()
        if not sep or not key.replace("_", "").isalnum():
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        out[key] = val
    return out


def _key_lookup(env: Mapping[str, str], env_file: Path) -> tuple[str | None, str | None]:
    for name in KEY_ENV_VARS:
        if env.get(name, "").strip():
            return env[name].strip(), f"env:{name}"
    try:
        st = env_file.stat()
    except OSError:
        return None, None
    if not stat.S_ISREG(st.st_mode):
        return None, None
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        log.warning("%s is readable by group/other; run: chmod 600 %s", env_file, env_file)
    try:
        data = parse_env_file(env_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        log.warning("could not read %s", env_file)
        return None, None
    for name in KEY_ENV_VARS:
        if data.get(name, "").strip():
            from . import proc  # a file key is not in os.environ: register it so every error path scrubs it
            proc.add_secret(data[name].strip())
            return data[name].strip(), f"file:{env_file}"
    return None, None


def get_gemini_key(env: Mapping[str, str] | None = None, env_file: Path | None = None) -> str | None:
    """Return the Gemini API key or None. Never logs or prints the value."""
    env = os.environ if env is None else env
    return _key_lookup(env, env_file or default_env_file(env))[0]


def key_status(env: Mapping[str, str] | None = None, env_file: Path | None = None) -> dict:
    """For `doctor`: {"present": bool, "source": "env:GEMINI_API_KEY"|"file:<path>"|None, "looked": [...]}."""
    env = os.environ if env is None else env
    ef = env_file or default_env_file(env)
    key, source = _key_lookup(env, ef)
    return {"present": key is not None, "source": source, "looked": [*(f"env:{n}" for n in KEY_ENV_VARS), str(ef)]}


def require_gemini_key(env: Mapping[str, str] | None = None, env_file: Path | None = None) -> str:
    env = os.environ if env is None else env
    ef = env_file or default_env_file(env)
    key = _key_lookup(env, ef)[0]
    if key is None:
        raise MissingKey(f"Gemini API key missing (looked in: {', '.join(KEY_ENV_VARS)}, {ef})")
    return key
