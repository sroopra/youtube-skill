"""Gemini multimodal provider via the Interactions API (google-genai 2.34.0; research 2.A, 4.4-4.6).

``google.genai`` is imported lazily inside methods (transcript-only runs never import it).
Key: ``config.require_gemini_key()`` -> ``genai.Client(api_key=...)``; never logged, cached,
or included in errors (error text is redacted and the key value replaced).

Request shape (Interactions):
    client.interactions.create(
        model=..., input=build_input(req),
        response_format={"type": "text", "mime_type": "application/json", "schema": req.schema},
        generation_config={"thinking_level": ..., "max_output_tokens": ...},
        system_instruction=prompts.SYSTEM_INSTRUCTION,
        previous_interaction_id=..., background=True (agentic / > 10 min))   # None fields omitted
    No temperature/top_p/top_k. Text = output_text; agentic_confirmed = any processing_call step.
    Poll ``interactions.get(id)`` while status == "in_progress" (5 s; bounded by cfg.timeout("gemini")).
    Fallback when the client has no ``interactions`` attribute: ``models.generate_content`` with
    :func:`build_generate_content` (static only).

Errors: retry 408/429 rate-limit/500/502/503/504 + transport errors (1-2-4-8 s + jitter, cap 60 s,
    cfg.retry_attempts, server retry hints honoured; SDK retries disabled-ish via attempts=1);
    404 model_not_found or 429 quota_exceeded -> next model in cfg.models (recorded in warnings);
    400 with agentic -> probe cache ``{"agentic": model}`` = unavailable, raise AgenticUnavailable
    (caller falls back to static); 401/403 -> AuthError; 402 -> AuthError (billing);
    private/unlisted/unavailable video -> VideoUnavailable; exhausted ladder -> NetworkError.
Cost: estimate_cost() before any call; > cfg.max_cost_usd / cfg.max_input_tokens -> OverBudget.
    cfg.dry_run -> no call. gemini_tier "free" -> local 8 h/day YouTube tracker (namespace "usage").
Output: JSON parsed, times coerced to absolute seconds (window-relative detected + shifted),
    validated against req.schema (minimal checker); invalid -> one repair call, then data=None
    + warning. Every string scanned with detect_injection -> injection_flags.
Cache: namespace "gemini", parts = request_cache_parts(req); only schema-valid results cached.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import logging
import math
import re
import time
from typing import Any

from .. import config as _config
from .. import proc
from ..cache import Cache, cache_key, canonical_json
from ..config import Config
from ..errors import (AuthError, BadInput, InternalError, MissingDependency, MissingKey, NetworkError, OverBudget,
                      SubprocessTimeout, VideoUnavailable, YTUError)
from ..models import AnalyzeRequest, AnalyzeResult, Availability, CostEstimate, Window
from ..prompts import SYSTEM_INSTRUCTION
from ..urls import parse_ts
from ..util.retry import backoff_delays
from ..util.sanitize import detect_injection

log = logging.getLogger("ytu.gemini")

SDK_VERSION = "2.34.0"
# The only host that ever receives the key (see _get_client).
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/"
PRICE_TABLE_DATE = "2026-11-05"
# USD per 1M tokens (paid tier, research 2.A pricing row). Update with date.
# 3.8/3.6 Flash double on 2027-01-01; 4 Argon doubles on 2026-12-01.
PRICES = {
    "gemini-3.8-flash": {"in": 0.75, "out": 3.75},
    "gemini-3.6-flash": {"in": 0.75, "out": 3.75},
    "gemini-3.5-flash-lite": {"in": 0.30, "out": 2.50},
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
    "gemini-4-argon": {"in": 2.00, "out": 10.00},
    "gemini-3.1-pro-preview": {"in": 2.00, "out": 12.00},
}
TOKENS_PER_FRAME = {"low": 70, "medium": 70, "high": 280}
AUDIO_TOKENS_PER_S = 32
AGENTIC_MODELS = {"gemini-4-argon", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"}
RESOLUTIONS = ("low", "medium", "high")
THINKING_LEVELS = ("minimal", "low", "medium", "high")
POLL_S = 5.0
BACKGROUND_OVER_S = 600.0
FREE_TIER_YT_S_PER_DAY = 8 * 3600
TIME_KEYS = ("t", "start", "end")
TIME_SLACK_S = 5.0
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
MAX_HINT_S = 60.0

_MODEL_NOT_FOUND = re.compile(r"model_not_found|is not found for API version|models?/\S+ (is )?not found|"
                              r"unknown model|model\b.{0,40}\bnot (found|exist)", re.I)
_QUOTA = re.compile(r"quota_exceeded|per[ _-]?day|daily", re.I)
_VIDEO = re.compile(r"\bprivate\b|\bunlisted\b|video\b.{0,40}\b(unavailable|not (found|available|accessible))|"
                    r"cannot (access|fetch|retrieve)\b.{0,30}\bvideo", re.I)
_TRANSPORT = re.compile(r"Timeout|Connect|Network|Protocol|NoResponse|ReadError|WriteError", re.I)


class AgenticUnavailable(YTUError):
    """Agentic processing rejected / not observed for this model; caller should use static windows."""

    exit_code = 1
    default_remediation = "Use static processing with --start/--end windows (done automatically by `ask`)."


# --- estimates ---------------------------------------------------------------------

def estimate_tokens(duration_s: float, resolution: str = "low", fps: float = 1.0) -> int:
    """duration_s * (TOKENS_PER_FRAME[resolution] * fps + AUDIO_TOKENS_PER_S), rounded up
    (~102 tok/s low, ~312 tok/s high at 1 fps)."""
    fps = 1.0 if fps is None else float(fps)
    tpf = TOKENS_PER_FRAME.get(resolution, TOKENS_PER_FRAME["high"])
    return math.ceil(max(0.0, float(duration_s)) * (tpf * fps + AUDIO_TOKENS_PER_S))


def estimate_cost(model: str, duration_s: float, resolution: str = "low", fps: float = 1.0,
                  prompt_tokens: int = 2000, output_tokens: int = 8192) -> CostEstimate:
    """CostEstimate(model, input_tokens, output_tokens, usd, basis). Unknown model -> most expensive
    known price, stated in basis."""
    fps = 1.0 if fps is None else fps
    price, note = PRICES.get(model), ""
    if price is None:
        price = max(PRICES.values(), key=lambda p: (p["in"], p["out"]))
        note = " (unknown model: using highest known price)"
    in_tok = estimate_tokens(duration_s, resolution, fps) + int(prompt_tokens)
    usd = in_tok / 1e6 * price["in"] + output_tokens / 1e6 * price["out"]
    tpf = TOKENS_PER_FRAME.get(resolution, TOKENS_PER_FRAME["high"])
    basis = (f"{duration_s:g}s x ({tpf} tok/frame x {fps:g} fps + {AUDIO_TOKENS_PER_S} audio tok/s) + "
             f"{prompt_tokens} prompt tok; ${price['in']}/${price['out']} per 1M in/out; "
             f"prices {PRICE_TABLE_DATE}{note}")
    return CostEstimate(model, in_tok, int(output_tokens), round(usd, 6), basis)


# --- request building (pure) ---------------------------------------------------------

def _offset(seconds: float) -> str:
    s = f"{float(seconds):.3f}".rstrip("0").rstrip(".")
    return f"{s}s"


def _schema_hash(schema: dict | None) -> str | None:
    return hashlib.sha256(canonical_json(schema).encode()).hexdigest() if schema is not None else None


def request_cache_parts(req: AnalyzeRequest) -> dict:
    """Research 4.5 key: everything that changes the model's answer (never the key)."""
    static = req.mode == "static"
    return {
        "v": 1,
        "video_id": req.video_id,
        "provider": GeminiProvider.name,
        "model": req.model,
        "mode": req.mode,
        "resolution": req.resolution,
        "fps": req.fps if static else None,
        "start_offset": _offset(req.window.start) if static and req.window else None,
        "end_offset": _offset(req.window.end) if static and req.window else None,
        "prompt_template_id": req.prompt_template_id,
        "user_question": req.user_question,
        "schema_hash": _schema_hash(req.schema),
        "prompt_hash": hashlib.sha256(req.prompt.encode()).hexdigest(),
        "thinking_level": req.thinking_level,
        "max_output_tokens": req.max_output_tokens,
        "previous_interaction_id": req.previous_interaction_id,
        "sdk_version": SDK_VERSION,
    }


def _processing(req: AnalyzeRequest) -> str | dict:
    if req.mode == "agentic":
        return "agentic"
    if req.window is None and req.fps is None:
        return "static"
    p: dict[str, Any] = {"type": "static"}
    if req.window is not None:
        p["start_offset"], p["end_offset"] = _offset(req.window.start), _offset(req.window.end)
    if req.fps is not None:
        p["fps"] = req.fps
    return p


def build_input(req: AnalyzeRequest) -> list[dict]:
    """Pure: the ``input`` list for interactions.create."""
    return [{"type": "text", "text": req.prompt},
            {"type": "video", "uri": req.url, "resolution": req.resolution, "processing": _processing(req)}]


def build_create_kwargs(req: AnalyzeRequest, model: str | None = None) -> dict:
    """Pure: full kwargs for ``client.interactions.create`` (None fields omitted)."""
    kw: dict[str, Any] = {
        "model": model or req.model,
        "input": build_input(req),
        "generation_config": {"thinking_level": req.thinking_level, "max_output_tokens": req.max_output_tokens},
        "system_instruction": SYSTEM_INSTRUCTION,
    }
    if req.schema is not None:
        kw["response_format"] = {"type": "text", "mime_type": "application/json", "schema": req.schema}
    if req.previous_interaction_id:
        kw["previous_interaction_id"] = req.previous_interaction_id
    if req.mode == "agentic" or _media_seconds(req) > BACKGROUND_OVER_S:
        kw["background"] = True
    return kw


def build_generate_content(req: AnalyzeRequest, model: str | None = None) -> dict:
    """Pure: kwargs for legacy ``client.models.generate_content`` (static only; research 2.A types)."""
    if req.mode != "static":
        raise AgenticUnavailable("agentic processing is not available via generateContent")
    vm: dict[str, Any] = {}
    if req.window is not None:
        vm["start_offset"], vm["end_offset"] = _offset(req.window.start), _offset(req.window.end)
    if req.fps is not None:
        vm["fps"] = req.fps
    video: dict[str, Any] = {"file_data": {"file_uri": req.url, "mime_type": "video/mp4"}}
    if vm:
        video["video_metadata"] = vm
    config: dict[str, Any] = {
        "system_instruction": SYSTEM_INSTRUCTION,
        "max_output_tokens": req.max_output_tokens,
        "thinking_config": {"thinking_level": req.thinking_level.upper()},
        "media_resolution": f"MEDIA_RESOLUTION_{req.resolution.upper()}",
    }
    if req.schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = req.schema
    return {"model": model or req.model,
            "contents": [{"role": "user", "parts": [video, {"text": req.prompt}]}],
            "config": config}


def _media_seconds(req: AnalyzeRequest) -> float:
    if req.mode == "static" and req.window is not None:
        return req.window.duration
    return float(req.duration_s or 0.0)


# --- output handling (pure) ----------------------------------------------------------

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _type_ok(value: Any, t: str) -> bool:
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool) or (
            isinstance(value, float) and value.is_integer())
    py = _TYPES.get(t)
    return True if py is None else isinstance(value, py)


def validate_json(value: Any, schema: dict | None, path: str = "$") -> list[str]:
    """Minimal JSON-schema check (type, enum, required, properties, items). [] = valid."""
    if not schema:
        return []
    errs: list[str] = []
    t = schema.get("type")
    types_ = t if isinstance(t, list) else [t] if t else []
    if types_ and not any(_type_ok(value, x) for x in types_):
        return [f"{path}: expected {'|'.join(types_)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, dict):
        errs += [f"{path}.{k}: missing" for k in schema.get("required", []) if k not in value]
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                errs += validate_json(value[k], sub, f"{path}.{k}")
    elif isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errs += validate_json(v, schema["items"], f"{path}[{i}]")
    return errs


def parse_json_text(text: str) -> tuple[Any, str | None]:
    """json.loads with ```json fences stripped. Returns (value, error)."""
    s = (text or "").strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*)\n```$", s, re.S)
    if m:
        s = m.group(1)
    try:
        return json.loads(s), None
    except ValueError as e:
        return None, f"invalid JSON: {e}"


def _walk(obj: Any, path: str = "$"):
    """Yield (container, key, value, path) for every leaf/child."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield obj, k, v, f"{path}.{k}"
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield obj, i, v, f"{path}[{i}]"
            yield from _walk(v, f"{path}[{i}]")


def normalize_times(data: Any, window: Window | None, duration_s: float | None = None) -> list[str]:
    """In place: coerce time fields (t/start/end) to absolute float seconds. Strings like "12:34"
    are parsed; unparseable ones are left as-is (validation will flag). If every time fits the
    clip length but not the absolute window, the model used window-relative times -> shift by
    window.start. Out-of-range times are kept and reported (never invented or clamped)."""
    warns: list[str] = []
    refs = [(c, k, p) for c, k, v, p in _walk(data) if isinstance(c, dict) and k in TIME_KEYS
            and v is not None and not isinstance(v, (dict, list, bool))]
    for c, k, p in refs:
        if isinstance(c[k], str):
            try:
                c[k] = parse_ts(c[k].strip())
            except YTUError:
                warns.append(f"{p}: unparseable time {c[k]!r}")
    nums = [(c, k, p) for c, k, p in refs if isinstance(c[k], (int, float))]
    if not nums:
        return warns
    vals = [float(c[k]) for c, k, _ in nums]
    if window is not None and window.start > TIME_SLACK_S:
        in_abs = all(window.start - TIME_SLACK_S <= v <= window.end + TIME_SLACK_S for v in vals)
        in_rel = all(-TIME_SLACK_S <= v <= window.duration + TIME_SLACK_S for v in vals)
        if not in_abs and in_rel:
            for c, k, _ in nums:
                c[k] = float(c[k]) + window.start
            warns.append(f"model returned window-relative times; shifted by {window.start:g}s")
    lo, hi = (window.start, window.end) if window else (0.0, duration_s)
    bad = [p for c, k, p in nums if float(c[k]) < lo - TIME_SLACK_S or (hi is not None and float(c[k]) > hi + TIME_SLACK_S)]
    if bad:
        warns.append(f"{len(bad)} time(s) outside {lo:g}-{hi if hi is not None else '?'}s: {', '.join(bad[:10])}")
    for c, k, _ in nums:
        c[k] = float(c[k])
    return warns


def flag_injection(data: Any, raw_text: str = "") -> list[dict]:
    """[{path, patterns}] for every output string matching sanitize.INJECTION_PATTERNS."""
    flags = [{"path": p, "patterns": pats} for _, _, v, p in _walk(data)
             if isinstance(v, str) and (pats := detect_injection(v))]
    if data is None and (pats := detect_injection(raw_text)):
        flags.append({"path": "raw_text", "patterns": pats})
    return flags


# --- error classification ----------------------------------------------------------------

def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _status_of(e: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        v = getattr(e, attr, None)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


def _text_of(e: BaseException) -> str:
    bits = [str(e)]
    for attr in ("body", "status", "details", "message"):
        v = getattr(e, attr, None)
        if v:
            bits.append(v if isinstance(v, str) else json.dumps(v, default=str))
    return " ".join(bits)


def _retry_hint(e: BaseException, text: str) -> float | None:
    headers = getattr(e, "headers", None)
    if headers is None:
        headers = getattr(getattr(e, "response", None), "headers", None)
    try:
        ra = headers.get("retry-after") or headers.get("Retry-After") if headers else None
        if ra:
            return min(MAX_HINT_S, float(ra))
    except (TypeError, ValueError, AttributeError):
        pass
    m = re.search(r"retry_?delay\W{0,4}(\d+(?:\.\d+)?)s", text, re.I)
    return min(MAX_HINT_S, float(m.group(1))) if m else None


@dataclasses.dataclass
class _Failure:
    kind: str                 # "retry" | "next_model" | "agentic" | "fatal"
    error: YTUError
    hint: float | None = None


class _TransientInteraction(NetworkError):
    """A background interaction that ended "failed" with a retryable code (5xx/UNAVAILABLE/...)."""


_TRANSIENT_CODES = {"UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "ABORTED",
                    *(str(c) for c in RETRY_STATUS)}


def classify_error(e: BaseException, *, agentic: bool, model: str, scrub: str | None = None) -> _Failure:
    """Map an SDK/transport exception to a retry/fallback decision + typed error."""
    if isinstance(e, _TransientInteraction):
        return _Failure("retry", NetworkError(e.message, e.remediation, **e.details))
    if isinstance(e, YTUError):
        return _Failure("fatal", e)
    status = _status_of(e)
    text = _text_of(e)
    safe = proc.redact(text.replace(scrub, "***") if scrub else text)[:500]
    msg = f"Gemini {model}: {status or type(e).__name__}: {safe}"
    if status is None:
        if isinstance(e, (OSError, TimeoutError)) or _TRANSPORT.search(type(e).__name__):
            return _Failure("retry", NetworkError(f"Gemini transport error ({type(e).__name__}): {safe}"))
        return _Failure("fatal", InternalError(f"Gemini SDK error ({type(e).__name__}): {safe}"))
    if status == 429:
        if _QUOTA.search(text):
            return _Failure("next_model", NetworkError(
                msg, "Gemini quota exhausted for this model/project (RPD resets midnight Pacific). Check "
                     "https://aistudio.google.com/rate-limit, try later, or set --models.", status=status))
        return _Failure("retry", NetworkError(msg, "Gemini rate limit; wait and retry.", status=status),
                        _retry_hint(e, text))
    if status in RETRY_STATUS:
        return _Failure("retry", NetworkError(msg, "Gemini service error; retry later.", status=status),
                        _retry_hint(e, text))
    if status == 404:
        if _MODEL_NOT_FOUND.search(text):
            return _Failure("next_model", BadInput(msg, "Model not available; set --models / YTU_GEMINI_MODELS.",
                                                   status=status))
        return _Failure("fatal", VideoUnavailable(msg, status=status))
    if status in (401, 403):
        if _VIDEO.search(text):
            return _Failure("fatal", VideoUnavailable(msg, "Gemini only accepts public YouTube videos.",
                                                      status=status))
        return _Failure("fatal", AuthError(
            msg, "Check GEMINI_API_KEY is valid, the Generative Language API is enabled for its project, "
                 "and your region is supported.", status=status))
    if status == 402:
        return _Failure("fatal", AuthError(msg, "Billing required/insufficient for this model; check "
                                                "billing in Google AI Studio or use a free-tier model.",
                                           status=status))
    if status == 400 and _VIDEO.search(text):
        return _Failure("fatal", VideoUnavailable(msg, "Gemini only accepts public YouTube videos "
                                                       "(private/unlisted rejected).", status=status))
    if status == 400 and agentic:
        return _Failure("agentic", AgenticUnavailable(msg, status=status))
    return _Failure("fatal", BadInput(msg, "Gemini rejected the request; check the schema/window/fps "
                                           "values. Re-run with -v.", status=status))


# --- provider --------------------------------------------------------------------------

def agentic_probe_parts(model: str) -> dict:
    return {"agentic": model}


def agentic_unavailable(cache: Cache, model: str) -> bool:
    """True if a previous call recorded agentic as unavailable for ``model`` (probe TTL 7 d)."""
    v = cache.get("probe", agentic_probe_parts(model))
    return bool(v and v.get("unavailable"))


def _pacific_date() -> str:
    try:
        from zoneinfo import ZoneInfo
        return _dt.datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    except Exception:  # noqa: BLE001 - tzdata missing
        return (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=8)).date().isoformat()


def _usage_dict(u: Any) -> dict:
    if u is None:
        return {}
    if hasattr(u, "model_dump"):
        d = u.model_dump(exclude_none=True, mode="json")
    elif isinstance(u, dict):
        d = dict(u)
    else:
        d = {k: v for k, v in vars(u).items() if not k.startswith("_") and v is not None}
    inp = d.get("total_input_tokens", d.get("prompt_token_count"))
    if isinstance(inp, int) and isinstance(d.get("total_tool_use_tokens"), int):
        inp += d["total_tool_use_tokens"]  # agentic: video segments arrive as tool-use tokens (live-verified)
    d.setdefault("input_tokens", inp)
    d.setdefault("output_tokens", d.get("total_output_tokens", d.get("candidates_token_count")))
    d.setdefault("total_tokens", d.get("total_token_count"))
    return {k: v for k, v in d.items() if v is not None}


class GeminiProvider:
    name = "gemini-interactions"

    def __init__(self, cfg: Config, cache: Cache):
        self.cfg, self.cache = cfg, cache
        self._client: Any = None
        self._key_scrub: str | None = None
        self._sleep = time.sleep
        self._clock = time.monotonic

    # -- availability / client --------------------------------------------------------
    def available(self) -> Availability:
        """ok iff a key is found and google.genai importable. No network."""
        if _config.get_gemini_key() is None:
            return Availability(False, "Gemini API key missing", MissingKey.default_remediation)
        try:
            import google.genai  # noqa: F401
        except ImportError:
            return Availability(False, "google-genai not importable",
                                f"Run via `uv run scripts/yt.py` (PEP 723 pins google-genai=={SDK_VERSION}).")
        return Availability(True)

    def _get_client(self) -> Any:
        if self._client is None:
            key = _config.require_gemini_key()
            try:
                from google import genai
                from google.genai import types
            except ImportError as e:
                raise MissingDependency("google-genai not importable",
                                        f"Run via `uv run scripts/yt.py` (google-genai=={SDK_VERSION}).") from e
            self._key_scrub = key
            # Our own retry loop owns backoff; keep SDK retries minimal (attempts=1).
            # Pin the endpoint: without base_url and vertexai=False, the SDK reads
            # GOOGLE_GEMINI_BASE_URL / GOOGLE_GENAI_USE_VERTEXAI from the environment, so one
            # env var could send the key to another host.
            self._client = genai.Client(api_key=key, vertexai=False, http_options=types.HttpOptions(
                base_url=GEMINI_BASE_URL,
                timeout=int(self.cfg.timeout("gemini") * 1000),
                retry_options=types.HttpRetryOptions(attempts=1)))
        return self._client

    # -- budget -----------------------------------------------------------------------
    def estimate(self, req: AnalyzeRequest, model: str | None = None) -> CostEstimate:
        secs = _media_seconds(req)
        if secs <= 0:
            raise BadInput("cannot estimate Gemini cost: no window and no video duration",
                           "Pass AnalyzeRequest.window or duration_s.")
        fps = req.fps if req.mode == "static" else 1.0
        est = estimate_cost(model or req.model, secs, req.resolution, fps or 1.0,
                            prompt_tokens=len(req.prompt) // 4 + len(SYSTEM_INSTRUCTION) // 4 + 300,
                            output_tokens=req.max_output_tokens)
        if req.mode == "agentic":
            est.basis += " [agentic: static upper-bound estimate]"
        return est

    def _check_budget(self, est: CostEstimate) -> None:
        if est.usd > self.cfg.max_cost_usd or est.input_tokens > self.cfg.max_input_tokens:
            raise OverBudget(f"Gemini estimate ${est.usd:.4f} / {est.input_tokens} input tokens exceeds "
                             f"limit ${self.cfg.max_cost_usd} / {self.cfg.max_input_tokens} tokens",
                             estimate=est.to_dict())

    def _free_tier_parts(self) -> dict:
        return {"tier": "free", "date": _pacific_date(), "what": "youtube-seconds"}

    def _check_free_tier(self, secs: float) -> None:
        if self.cfg.gemini_tier != "free":
            return
        used = float((self.cache.get("usage", self._free_tier_parts()) or {}).get("seconds", 0.0))
        if used + secs > FREE_TIER_YT_S_PER_DAY:
            raise OverBudget(f"free-tier YouTube allowance: {used / 3600:.2f} h used today + {secs / 3600:.2f} h "
                             f"> 8 h/day", "Wait until midnight Pacific, narrow the window, or set "
                                           "YTU_GEMINI_TIER=paid.")

    def _record_free_tier(self, secs: float) -> None:
        if self.cfg.gemini_tier == "free":
            parts = self._free_tier_parts()
            # read without force so --force-refresh does not reset the counter
            used = float((Cache(self.cache.root).get("usage", parts) or {}).get("seconds", 0.0))
            self.cache.put("usage", parts, {"seconds": used + secs})

    # -- main -------------------------------------------------------------------------
    def analyze(self, req: AnalyzeRequest) -> AnalyzeResult:
        self._validate(req)
        parts = request_cache_parts(req)
        rhash = cache_key(parts)
        hit = self.cache.get("gemini", parts)
        if hit is not None:
            res = AnalyzeResult.from_dict(hit)
            res.cached = True
            return res
        if req.mode == "agentic" and agentic_unavailable(self.cache, req.model):
            raise AgenticUnavailable(f"agentic processing previously unavailable for {req.model} (probe cache)")

        warnings: list[str] = []
        est = self.estimate(req)
        if self.cfg.dry_run:
            try:
                self._check_budget(est)
            except OverBudget as e:
                warnings.append(e.message)
            return AnalyzeResult(None, "", {}, req.model, req.mode, False, rhash, est, warnings=warnings)
        self._check_budget(est)
        self._check_free_tier(_media_seconds(req))

        model, out, est = self._call_ladder(req, est, warnings)
        data, warns = self._process(req, out["text"])
        warnings += warns
        if data is None and req.schema is not None:
            data, out, est = self._repair(req, model, est, out, warnings)

        agentic_ok = "processing_call" in out["step_types"]
        if req.mode == "agentic" and not agentic_ok:
            self.cache.put("probe", agentic_probe_parts(model),
                           {"unavailable": True, "reason": "no processing_call step observed"})
            warnings.append("agentic requested but no processing_call step observed; recorded agentic_unavailable")
        usage = out["usage"]
        actual_in = usage.get("input_tokens")
        if isinstance(actual_in, (int, float)) and est.input_tokens and \
                abs(actual_in - est.input_tokens) / est.input_tokens > 0.30:
            warnings.append(f"input tokens: estimate {est.input_tokens} vs actual {actual_in} (>30% off)")
        result = AnalyzeResult(
            data=data, raw_text=out["text"], usage=usage, model=model, mode=req.mode,
            agentic_confirmed=agentic_ok, request_hash=rhash, cost_estimate=est,
            interaction_id=out["id"], warnings=warnings, injection_flags=flag_injection(data, out["text"]))
        self._record_free_tier(_media_seconds(req))
        if data is not None or req.schema is None:
            # key by the model that answered: a fallback answer must not shadow the requested model later
            self.cache.put("gemini", request_cache_parts(dataclasses.replace(req, model=model)), result.to_dict())
            if model != req.model:
                warnings.append(f"answered by fallback model {model}; not cached under {req.model}")
        return result

    def _validate(self, req: AnalyzeRequest) -> None:
        if req.mode not in ("static", "agentic"):
            raise BadInput(f"unknown Gemini mode {req.mode!r}", "Use static or agentic.")
        if req.resolution not in RESOLUTIONS:
            raise BadInput(f"bad resolution {req.resolution!r}", f"Use one of {RESOLUTIONS}.")
        if req.thinking_level not in THINKING_LEVELS:
            raise BadInput(f"bad thinking_level {req.thinking_level!r}", f"Use one of {THINKING_LEVELS}.")
        if req.mode == "static" and req.fps is not None and not 0 < req.fps <= 24:
            raise BadInput(f"fps {req.fps} out of range (0, 24]")
        if req.mode == "agentic" and (req.window is not None or req.fps is not None):
            raise BadInput("window/fps are static-only; agentic takes the whole video",
                           "Drop window/fps or use mode='static'.")

    def _ladder(self, req: AnalyzeRequest) -> list[str]:
        return [req.model] + [m for m in self.cfg.models if m != req.model]

    def _call_ladder(self, req: AnalyzeRequest, est: CostEstimate, warnings: list[str]):
        client = self._get_client()
        last: YTUError | None = None
        for i, model in enumerate(self._ladder(req)):
            if req.mode == "agentic" and model not in AGENTIC_MODELS:
                warnings.append(f"skipped {model}: agentic not supported")
                continue
            if i:
                est = self.estimate(req, model)
                self._check_budget(est)
            try:
                return model, self._call_with_retry(client, req, model), est
            except _NextModel as nm:
                last = nm.error
                warnings.append(f"{model} unavailable ({type(nm.error).__name__}); trying next model")
                log.warning("%s unavailable; trying next model", model)
        if req.mode == "agentic" and last is None:
            raise AgenticUnavailable("no model in the ladder supports agentic processing")
        raise last or NetworkError("Gemini: no usable model", "Set --models / YTU_GEMINI_MODELS.")

    def _call_with_retry(self, client: Any, req: AnalyzeRequest, model: str) -> dict:
        attempts = max(1, self.cfg.retry_attempts)
        delays = backoff_delays(attempts)
        for i in range(attempts):
            try:
                return self._call_once(client, req, model)
            except Exception as e:  # noqa: BLE001 - classified below
                f = classify_error(e, agentic=req.mode == "agentic", model=model, scrub=self._key_scrub)
                if f.kind == "next_model":
                    raise _NextModel(f.error) from None
                if f.kind == "agentic":
                    self.cache.put("probe", agentic_probe_parts(model),
                                   {"unavailable": True, "reason": f.error.message[:200]})
                    raise f.error from None
                if f.kind == "retry" and i < attempts - 1:
                    d = min(MAX_HINT_S, max(delays[i], f.hint or 0.0))
                    log.warning("Gemini %s failed (%s); retry %d/%d in %.1fs", model, f.error.message[:120],
                                i + 1, attempts - 1, d)
                    self._sleep(d)
                    continue
                raise f.error from None
        raise AssertionError("unreachable")

    def _call_once(self, client: Any, req: AnalyzeRequest, model: str) -> dict:
        if not hasattr(client, "interactions"):
            resp = client.models.generate_content(**build_generate_content(req, model))
            return {"text": _get(resp, "text") or "", "usage": _usage_dict(_get(resp, "usage_metadata")),
                    "id": None, "step_types": set()}
        ia = client.interactions.create(**build_create_kwargs(req, model))
        deadline = self._clock() + self.cfg.timeout("gemini")
        while _get(ia, "status") == "in_progress":
            if self._clock() > deadline:
                try:
                    client.interactions.cancel(_get(ia, "id"))
                except Exception:  # noqa: BLE001 - best effort
                    pass
                raise SubprocessTimeout(f"Gemini interaction still in progress after {self.cfg.timeout('gemini'):g}s",
                                        "Raise YTU_TIMEOUT_GEMINI or narrow the window.")
            self._sleep(POLL_S)
            ia = client.interactions.get(_get(ia, "id"))
        status = _get(ia, "status")
        if status in ("failed", "cancelled", "requires_action"):
            raw = _get(ia, "errors") or []
            errs = [f"{_get(x, 'code')}: {_get(x, 'message')}" for x in raw]
            text = proc.redact("; ".join(errs), secrets=[self._key_scrub])[:500]
            transient = status == "failed" and any(
                _get(x, "code") in RETRY_STATUS or str(_get(x, "code")).upper() in _TRANSIENT_CODES
                or str(_get(x, "status") or "").upper() in _TRANSIENT_CODES for x in raw)
            raise (_TransientInteraction if transient else NetworkError)(
                f"Gemini interaction {status}: {text or 'no error detail'}", "Retry later; re-run with -v.",
                status=status)
        steps = _get(ia, "steps") or []
        return {"text": _get(ia, "output_text") or "", "usage": _usage_dict(_get(ia, "usage")),
                "id": _get(ia, "id"), "step_types": {_get(s, "type") for s in steps},
                "status": status}

    def _process(self, req: AnalyzeRequest, text: str) -> tuple[Any, list[str]]:
        data, err = parse_json_text(text)
        if err:
            return (text if req.schema is None else None), [f"model output: {err}"]
        warns = normalize_times(data, req.window if req.mode == "static" else None, req.duration_s)
        errs = validate_json(data, req.schema)
        if errs:
            return None, warns + [f"schema: {'; '.join(errs[:10])}"]
        return data, warns

    def _repair(self, req: AnalyzeRequest, model: str, est: CostEstimate, out: dict, warnings: list[str]):
        if est.usd * 2 > self.cfg.max_cost_usd:
            warnings.append("output invalid; repair call skipped (would exceed --max-cost-usd)")
            return None, out, est
        problems = "; ".join(w for w in warnings if w.startswith(("schema:", "model output:")))[:1000]
        fix = dataclasses.replace(req, model=model, prompt=req.prompt + (
            "\n\nYour previous reply did not match the response schema (" + problems + "). "
            "Reply again with ONLY valid JSON matching the schema. Do not invent content to fill fields; "
            "use empty lists and `uncertainties` instead."))
        client = self._get_client()
        out2 = self._call_with_retry(client, fix, model)
        data, warns = self._process(req, out2["text"])
        warnings += [f"repair: {w}" for w in warns]
        if data is None:
            warnings.append("output still invalid after one repair attempt; data=None")
        u1, u2 = out["usage"], out2["usage"]
        out2["usage"] = {**u2, "calls": 2, **{k: u1[k] + u2[k] for k in ("input_tokens", "output_tokens", "total_tokens")
                                              if isinstance(u1.get(k), int) and isinstance(u2.get(k), int)}}
        out2["step_types"] = out["step_types"] | out2["step_types"]
        return data, out2, dataclasses.replace(est, usd=round(est.usd * 2, 6),
                                               basis=est.basis + " [x2: repair call]")


class _NextModel(Exception):
    def __init__(self, error: YTUError):
        super().__init__(error.message)
        self.error = error
