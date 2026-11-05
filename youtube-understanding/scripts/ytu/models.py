"""Shared data model. All times are absolute seconds from video start (float).

Every dataclass has ``to_dict()`` (JSON-safe, via dataclasses.asdict) and
``from_dict(d)`` (tolerant: unknown keys ignored, missing optional keys default).

ID conventions: segments ``s0001``, frames ``F0001``, evidence ``E0001``
(1-based, zero-padded to 4; wider if > 9999). Use :func:`make_id`.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Literal


def make_id(prefix: str, n: int) -> str:
    """make_id("s", 1) -> "s0001". ``n`` is 1-based."""
    return f"{prefix}{n:04d}"


class JsonMixin:
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, d: dict | None):
        if d is None:
            return None
        hints = typing.get_type_hints(cls)
        kwargs = {}
        for f in dataclasses.fields(cls):  # type: ignore[arg-type]
            if f.name in d:
                kwargs[f.name] = _coerce(hints[f.name], d[f.name])
        return cls(**kwargs)


def _coerce(tp, value):
    if value is None:
        return None
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        for arg in typing.get_args(tp):
            if arg is type(None):
                continue
            return _coerce(arg, value)
    if origin is list:
        (inner,) = typing.get_args(tp) or (Any,)
        return [_coerce(inner, v) for v in value]
    if origin is tuple:
        return tuple(value)
    if isinstance(tp, type) and issubclass(tp, JsonMixin) and isinstance(value, dict):
        return tp.from_dict(value)
    return value


# --- Video identity & metadata ------------------------------------------------------

@dataclass
class VideoRef(JsonMixin):
    video_id: str                      # 11-char YouTube id
    canonical_url: str                 # https://www.youtube.com/watch?v=<id>
    start_s: float | None = None       # timestamp from the URL (&t=, start=), if any


@dataclass
class Chapter(JsonMixin):
    start: float
    end: float
    title: str                         # untrusted text


TrackKind = Literal["manual", "auto", "generated", "asr"]


@dataclass
class TrackRef(JsonMixin):
    source: str                        # "ytdlp" | "yta" | "asr-parakeet" | "asr-mlx" | "asr-fw" | "asr-gemini"
    code: str                          # yt-dlp key e.g. "en", "en-orig", "de"; asr: "asr:<model>"
    lang: str | None                   # base language e.g. "en" (None if unknown)
    kind: str = "manual"               # TrackKind
    name: str | None = None            # display name, e.g. "English (Original)"
    formats: list[str] = field(default_factory=list)  # available exts: json3, srt, vtt, ...

    @property
    def is_generated(self) -> bool:
        return self.kind != "manual"


@dataclass
class VideoMeta(JsonMixin):
    video_id: str
    title: str                         # untrusted
    duration: float | None             # None for live streams
    channel: str | None = None
    channel_id: str | None = None
    upload_date: str | None = None     # YYYYMMDD
    description: str | None = None     # untrusted
    language: str | None = None        # yt-dlp info.language (may be None)
    availability: str | None = None    # public|unlisted|private|needs_auth|subscriber_only|premium_only|None
    live_status: str | None = None     # not_live|is_live|is_upcoming|was_live|post_live|None
    age_limit: int | None = None
    view_count: int | None = None
    webpage_url: str | None = None
    chapters: list[Chapter] = field(default_factory=list)
    tracks: list[TrackRef] = field(default_factory=list)  # manual + auto inventory (live_chat excluded)
    yt_dlp_version: str | None = None
    fetched_at: str | None = None      # ISO-8601 UTC


# --- Transcript --------------------------------------------------------------------

@dataclass
class Word(JsonMixin):
    w: str
    start: float
    end: float
    conf: float | None = None


@dataclass
class Segment(JsonMixin):
    id: str                            # "s0001"
    start: float
    end: float
    text: str                          # untrusted
    words: list[Word] | None = None


@dataclass
class Transcript(JsonMixin):
    video_id: str
    segments: list[Segment]
    source: str                        # provider name: "ytdlp" | "yta" | "asr-*"
    track: TrackRef | None = None
    is_generated: bool = False
    language: str | None = None
    provenance: dict = field(default_factory=dict)  # {tool, tool_version, fetched_at, format, window?, model?}

    def text_between(self, start: float, end: float) -> list[Segment]:
        return [s for s in self.segments if s.end > start and s.start < end]


@dataclass
class Window(JsonMixin):
    start: float
    end: float
    reason: str | None = None          # "chapter:<n>" | "fixed" | "question" | "user"

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid window {self.start}-{self.end}")

    @property
    def duration(self) -> float:
        return self.end - self.start


# --- Frames ------------------------------------------------------------------------

@dataclass
class Frame(JsonMixin):
    id: str                            # "F0001" (assigned by pipeline; providers may leave "")
    path: str                          # absolute path (cache) or package-relative "screenshots/t0754200.jpg"
    abs_ts: float                      # absolute seconds in the video
    selection_reason: str              # transcript-cue|gemini-event|interval|scene|user-question|keyframe
    sha256: str
    width: int
    height: int
    pts: float | None = None
    method: str | None = None          # "remote-seek" | "section-download"


# --- Multimodal --------------------------------------------------------------------

@dataclass
class CostEstimate(JsonMixin):
    model: str
    input_tokens: int
    output_tokens: int
    usd: float
    basis: str = ""                    # human-readable formula / price table date


@dataclass
class AnalyzeRequest(JsonMixin):
    video_id: str
    url: str                           # canonical watch URL
    prompt: str                        # full prompt (untrusted parts already wrapped)
    model: str
    mode: str = "static"               # "static" | "agentic"
    schema: dict | None = None         # JSON schema for response_format
    window: Window | None = None       # static only
    fps: float | None = None           # static only, (0, 24]
    resolution: str = "low"            # "low" | "medium" | "high"
    prompt_template_id: str = ""       # e.g. "overview@1"
    user_question: str | None = None
    max_output_tokens: int = 8192
    thinking_level: str = "low"        # "low" | "medium" | "high" (3.8 Flash rejects "minimal")
    previous_interaction_id: str | None = None
    duration_s: float | None = None    # video duration; needed for the cost estimate when window is None


@dataclass
class AnalyzeResult(JsonMixin):
    data: Any                          # parsed JSON (schema-validated) or None
    raw_text: str                      # untrusted model output
    usage: dict                        # {input_tokens, output_tokens, total_tokens, ...} as returned
    model: str
    mode: str
    agentic_confirmed: bool            # True only if a processing_call step was observed
    request_hash: str                  # cache key of the request
    cost_estimate: CostEstimate | None = None
    interaction_id: str | None = None
    cached: bool = False
    warnings: list[str] = field(default_factory=list)        # ladder fallbacks, time fixes, schema issues
    injection_flags: list[dict] = field(default_factory=list)  # [{path, patterns}] from detect_injection


@dataclass
class Plan(JsonMixin):
    """Output of pipeline.strategy.plan()."""
    strategy: str                      # resolved: "transcript" | "gemini" | "local"
    transcript_providers: list[str]
    multimodal: str                    # "gemini-static" | "gemini-agentic" | "host-frames" | "none"
    model: str | None = None
    resolution: str | None = None
    fps: float | None = None
    windows: list[Window] = field(default_factory=list)
    estimate: CostEstimate | None = None
    reasons: list[str] = field(default_factory=list)


# --- Evidence ----------------------------------------------------------------------

@dataclass
class EvidenceItem(JsonMixin):
    id: str                            # "E0001"
    claim: str
    kind: str                          # speech|code|diagram|slide|ui|inference
    t_start: float
    t_end: float
    timestamp: str                     # fmt_ts(t_start)
    url: str                           # ts_link(video_id, t_start)
    transcript_segment_ids: list[str] = field(default_factory=list)
    frame_ids: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)   # "captions:manual" | "captions:auto" | "gemini:static:<model>" | "host-vision" | "asr:<model>"
    status: str = "unverified"         # verified|corroborated|single-source|unverified|contradicted
    confidence: str = "low"            # high|medium|low
    notes: str = ""
    injection_suspect: bool = False
    demonstrated: bool | None = None   # workflow B: True = shown/said in video, False = inferred


@dataclass
class Gap(JsonMixin):
    t_start: float
    t_end: float
    reason: str


@dataclass
class Evidence(JsonMixin):
    video_id: str
    source_hash: str = ""
    items: list[EvidenceItem] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    schema_version: str = "1"


@dataclass
class Availability(JsonMixin):
    ok: bool
    reason: str = ""
    remediation: str = ""
