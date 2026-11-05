"""Provider interfaces (research 4.3) and the lazy registry.

Constructor convention for every provider class: ``Cls(cfg: Config, cache: Cache)``.
Providers must:
  - run external tools only via ``ytu.proc.run`` (with ``cfg.timeout(<name>)``),
  - raise only ``ytu.errors.YTUError`` subclasses (never bare exceptions to the CLI),
  - never fabricate text/timestamps: return what the source returned, or raise,
  - cache via ``cache.get/put`` with keys that include tool version (see CONTRACT.md),
    and honour ``cache.force``,
  - log to the ``ytu.<module>`` logger (stderr); never print to stdout,
  - never log/persist API keys or signed media URLs.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..models import (AnalyzeRequest, AnalyzeResult, Availability, Frame, TrackRef, Transcript,
                      VideoMeta, Window)

if TYPE_CHECKING:
    from ..cache import Cache
    from ..config import Config


@runtime_checkable
class TranscriptProvider(Protocol):
    """Caption / ASR source.

    name: "ytdlp" | "yta" | "asr-parakeet" | "asr-mlx" | "asr-fw" | "asr-gemini"
    """

    name: str

    def available(self) -> Availability:
        """Cheap local check (deps importable / binaries on PATH / key present). No network."""
        ...

    def inventory(self, meta: VideoMeta) -> list[TrackRef]:
        """Tracks this provider could fetch for ``meta``. ytdlp: from meta.tracks (no network).
        yta: may call transcript list (network). asr: one synthetic TrackRef(code="asr:<model>")."""
        ...

    def fetch(self, video_id: str, track: TrackRef | None, window: Window | None = None) -> Transcript:
        """Fetch one track (``None`` = provider's own best pick) and return segments with ids
        s0001.. in time order, absolute seconds, deduped (no rolling-caption duplicates).
        ``window`` limits the result (ASR: only that audio section is transcribed; captions:
        filter segments overlapping the window, ids keep full-track numbering).
        Raises VideoUnavailable / Blocked / NetworkError / MissingDependency / ProcessFailed."""
        ...


@runtime_checkable
class MultimodalProvider(Protocol):
    """name: "gemini-interactions" | "gemini-generatecontent" | "host-frames"."""

    name: str

    def available(self) -> Availability:
        ...

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResult:
        """One model call (with retry/backoff and model-ladder fallback on 404/quota).
        Must check the cost estimate against cfg.max_cost_usd / cfg.max_input_tokens BEFORE
        calling (raise OverBudget) and honour cfg.dry_run (no call; return result with
        data=None, raw_text="", cost_estimate set). Results cached in namespace "gemini"
        keyed per research 4.5."""
        ...


@runtime_checkable
class FrameProvider(Protocol):
    """Local frame extraction (ffmpeg). name: "ffmpeg-remote"."""

    name: str

    def available(self) -> Availability:
        ...

    def grab(self, video_id: str, t: float, height: int = 720, reason: str = "user-question") -> Frame:
        """One JPEG at absolute time ``t`` via remote seek (yt-dlp -g + ffmpeg -ss before -i).
        Stored in cache namespace "frames"; Frame.path = absolute cache path, id = ""."""
        ...

    def window(self, video_id: str, start: float, end: float, strategy: str = "interval",
               every_s: float = 10.0, max_frames: int = 24, height: int = 720) -> list[Frame]:
        """Frames in [start, end]. strategy: "interval" (every ``every_s``), "keyframes",
        "scene" (best-effort; may return few/none for screencasts). Never more than max_frames."""
        ...


# --- registry ---------------------------------------------------------------------
# name -> "module:Class" relative to ytu.providers; imported lazily so optional deps
# (google-genai, ASR) are only imported when used.

TRANSCRIPT_PROVIDERS = {
    "ytdlp": "captions:YtDlpCaptions",
    "yta": "captions:YtaCaptions",
    "asr": "asr:AsrProvider",
}
MULTIMODAL_PROVIDERS = {
    "gemini": "gemini:GeminiProvider",
}
FRAME_PROVIDERS = {
    "ffmpeg": "frames:FfmpegFrames",
}


def _load(table: dict[str, str], name: str):
    from ..errors import BadInput

    if name not in table:
        raise BadInput(f"unknown provider {name!r}", f"Known: {', '.join(sorted(table))}.")
    mod, cls = table[name].split(":")
    return getattr(importlib.import_module(f"ytu.providers.{mod}"), cls)


def get_transcript_provider(name: str, cfg: "Config", cache: "Cache") -> TranscriptProvider:
    return _load(TRANSCRIPT_PROVIDERS, name)(cfg, cache)


def get_multimodal_provider(name: str, cfg: "Config", cache: "Cache") -> MultimodalProvider:
    return _load(MULTIMODAL_PROVIDERS, name)(cfg, cache)


def get_frame_provider(cfg: "Config", cache: "Cache", name: str = "ffmpeg") -> FrameProvider:
    return _load(FRAME_PROVIDERS, name)(cfg, cache)
