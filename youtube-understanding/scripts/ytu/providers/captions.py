"""Caption transcript providers (research 2.B, 4.4 step 2).

YtDlpCaptions (name "ytdlp"):
    Download exactly ONE track as json3 into a fresh temp dir (tempfile.mkdtemp under
    cfg.cache_dir / "tmp"), parse, delete the temp dir.
        flag = "--write-subs" if track.kind == "manual" else "--write-auto-subs"
        proc.run(proc.ytdlp_cmd("--skip-download", flag, "--sub-langs", track.code,
                 "--sub-format", "json3", "-o", f"{tmp}/%(id)s.%(ext)s", *cookie_args(cfg),
                 canonical_url(video_id)),
                 timeout=cfg.timeout("subtitles"), env=proc.ytdlp_env(), check=False)
    NEVER pass regex/multi-lang --sub-langs ("en.*" pulls translated tracks -> HTTP 429).
    rc != 0 -> errors.classify_ytdlp_error. Missing output file -> ProcessFailed.
    The json3 file is read with json.loads (data only; never executed).

YtaCaptions (name "yta"): youtube-transcript-api 1.2.4
    YouTubeTranscriptApi().list(video_id) -> pick an original track (never .translate());
    .fetch() snippets {text,start,duration} -> segments. Exceptions are mapped to typed errors
    (see _map_yta_error). The library is imported lazily.

Cache: namespace "transcript", parts {"video_id", "source": name, "track": track.code, "kind",
    "tool": "yt-dlp"|"youtube-transcript-api", "tool_version", "v": 1}; value Transcript.to_dict()
    (full track; window filtering applied after cache read).

Segment ids: make_id("s", n) in time order over the FULL track.
Transcript.provenance: {"tool", "tool_version", "format": "json3"|"yta", "fetched_at", "track_name"}
(+ "window" when filtered).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import shutil
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .. import proc
from ..cache import Cache
from ..config import Config
from ..errors import (BadInput, Blocked, NetworkError, ProcessFailed, VideoUnavailable,
                      classify_ytdlp_error)
from ..models import Availability, Segment, TrackRef, Transcript, VideoMeta, VideoRef, Window, Word, make_id
from ..urls import canonical_url
from ..util.sanitize import strip_control
from .ytdlp_meta import cookie_args, fetch_metadata

log = logging.getLogger("ytu.captions")

SCHEMA_V = 1
YTA_VERSION = "1.2.4"
NO_CAPTIONS_HINT = "No usable captions; try --strategy local (ASR) or another --transcript-providers order."
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", strip_control(text or "")).strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- pure parsing -------------------------------------------------------------------

def _conf(seg: dict) -> float | None:
    # acAsrConf is 0 on every observed track (research 2.B) -> treat 0/missing as unknown.
    # Non-zero values are assumed to be on a 0-255 scale (undocumented).
    v = seg.get("acAsrConf")
    if not isinstance(v, (int, float)) or v <= 0:
        return None
    return min(1.0, float(v) / 255.0)


def parse_json3(data: dict) -> list[Segment]:
    """Pure: json3 -> segments (one per event with text). Newline-only ``aAppend`` events and
    events without ``segs`` are skipped; an event's end is clamped to the next event's start
    (rolling auto captions overlap). Word timings when the event has >1 seg with tOffsetMs."""
    events = []
    for ev in (data or {}).get("events") or []:
        segs = ev.get("segs")
        if not segs or "tStartMs" not in ev:
            continue
        text = _clean("".join(str(s.get("utf8", "")) for s in segs))
        if not text:
            continue
        start_ms = int(ev["tStartMs"])
        events.append((start_ms, start_ms + int(ev.get("dDurationMs") or 0), text, segs))
    events.sort(key=lambda e: e[0])
    out: list[Segment] = []
    for i, (start_ms, end_ms, text, segs) in enumerate(events):
        if i + 1 < len(events) and start_ms < events[i + 1][0] < end_ms:
            end_ms = events[i + 1][0]
        start, end = start_ms / 1000, max(end_ms, start_ms) / 1000
        out.append(Segment(id=make_id("s", i + 1), start=start, end=end, text=text,
                           words=_words(start_ms, end_ms, segs)))
    return out


def _words(start_ms: int, end_ms: int, segs: list[dict]) -> list[Word] | None:
    if len(segs) < 2 or not any("tOffsetMs" in s for s in segs):
        return None
    raw = [(start_ms + int(s.get("tOffsetMs") or 0), _clean(str(s.get("utf8", ""))), _conf(s)) for s in segs]
    raw = [r for r in raw if r[1]]
    words = []
    for j, (t, w, conf) in enumerate(raw):
        nxt = raw[j + 1][0] if j + 1 < len(raw) else end_ms
        words.append(Word(w=w, start=t / 1000, end=max(nxt, t) / 1000, conf=conf))
    return words or None


def _norm(text: str) -> str:
    return _WS.sub(" ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def dedupe_segments(segments: list[Segment]) -> list[Segment]:
    """Pure: drop consecutive time-overlapping segments whose text duplicates or is contained in
    its neighbour (rolling captions); keeps the longer text, unions the time span, renumbers ids.
    Non-overlapping repeats (a phrase genuinely said twice) are kept."""
    out: list[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        if out:
            prev = out[-1]
            a, b = _norm(prev.text), _norm(seg.text)
            overlaps = seg.start < prev.end
            if overlaps and b and (b == a or b in a):
                out[-1] = replace(prev, end=max(prev.end, seg.end))
                continue
            if overlaps and a and a in b:
                out[-1] = replace(seg, start=prev.start, end=max(prev.end, seg.end))
                continue
        out.append(seg)
    return [replace(s, id=make_id("s", i + 1)) for i, s in enumerate(out)]


def _clamp(segments: list[Segment]) -> list[Segment]:
    """End of each segment clamped to the next segment's start (overlapping auto snippets)."""
    out = list(segments)
    for i in range(len(out) - 1):
        nxt = out[i + 1].start
        if out[i].start < nxt < out[i].end:
            out[i] = replace(out[i], end=nxt)
    return out


def _apply_window(t: Transcript, window: Window | None) -> Transcript:
    if window is None:
        return t
    return replace(t, segments=t.text_between(window.start, window.end),
                   provenance={**t.provenance, "window": window.to_dict()})


def _cached(cache: Cache, parts: dict) -> Transcript | None:
    hit = cache.get("transcript", parts)
    return Transcript.from_dict(hit) if hit else None


# --- yt-dlp -------------------------------------------------------------------------

class YtDlpCaptions:
    name = "ytdlp"

    def __init__(self, cfg: Config, cache: Cache):
        self.cfg, self.cache = cfg, cache

    def available(self) -> Availability:
        if proc.ytdlp_available_inprocess() or proc.which("uvx"):
            return Availability(True)
        return Availability(False, "yt-dlp not importable and uvx not on PATH",
                            "Run via `uv run scripts/yt.py ...` or install uv: https://docs.astral.sh/uv/")

    def inventory(self, meta: VideoMeta) -> list[TrackRef]:
        """meta.tracks with "json3" in formats (no network)."""
        return [t for t in meta.tracks if t.source == "ytdlp" and "json3" in t.formats and t.code != "live_chat"]

    def _pick(self, video_id: str) -> TrackRef:
        from ..pipeline.strategy import pick_track  # lazy: strategy imports providers

        meta = fetch_metadata(self.cfg, self.cache, VideoRef(video_id, canonical_url(video_id)))
        track = pick_track(meta, self.cfg.lang)
        if track is None:
            raise VideoUnavailable(f"no original caption track for {video_id}", NO_CAPTIONS_HINT)
        return track

    def fetch(self, video_id: str, track: TrackRef | None, window: Window | None = None) -> Transcript:
        if track is None:
            track = self._pick(video_id)
        if not track.code or track.code == "live_chat" or any(c in track.code for c in ",*.^$|[]()"):
            raise BadInput(f"refusing caption track code {track.code!r}", "Pass one exact track code (e.g. en, en-orig).")
        parts = {"video_id": video_id, "source": self.name, "track": track.code, "kind": track.kind,
                 "tool": "yt-dlp", "tool_version": proc.YTDLP_VERSION, "v": SCHEMA_V}
        t = _cached(self.cache, parts)
        if t is None:
            t = self._download(video_id, track)
            self.cache.put("transcript", parts, t.to_dict())
        return _apply_window(t, window)

    def _download(self, video_id: str, track: TrackRef) -> Transcript:
        from ..config import cache_tmp_root
        tmp = Path(tempfile.mkdtemp(prefix="subs-", dir=cache_tmp_root(self.cfg.cache_dir)))
        try:
            flag = "--write-subs" if track.kind == "manual" else "--write-auto-subs"
            res = proc.run(proc.ytdlp_cmd("--skip-download", flag, "--sub-langs", track.code,
                                          "--sub-format", "json3", "-o", f"{tmp}/%(id)s.%(ext)s",
                                          *cookie_args(self.cfg), canonical_url(video_id)),
                           timeout=self.cfg.timeout("subtitles"), env=proc.ytdlp_env(), check=False)
            if res.returncode:
                raise classify_ytdlp_error(res.stderr, res.returncode)
            files = sorted(tmp.glob("*.json3"))
            if not files:
                raise ProcessFailed(f"yt-dlp wrote no json3 file for track {track.code!r}",
                                    NO_CAPTIONS_HINT, stderr_tail=(res.stderr or "")[-800:])
            try:
                data = json.loads(files[0].read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise ProcessFailed(f"unparseable json3 for track {track.code!r}") from None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        segments = dedupe_segments(parse_json3(data))
        if not segments:
            raise VideoUnavailable(f"caption track {track.code!r} is empty", NO_CAPTIONS_HINT)
        return Transcript(video_id=video_id, segments=segments, source=self.name, track=track,
                          is_generated=track.is_generated, language=track.lang,
                          provenance={"tool": "yt-dlp", "tool_version": proc.YTDLP_VERSION, "format": "json3",
                                      "fetched_at": _now(), "track_name": track.name})


# --- youtube-transcript-api ---------------------------------------------------------

def _map_yta_error(exc: Exception, video_id: str):
    """Map youtube-transcript-api exceptions to YTUError (returned, caller raises)."""
    import youtube_transcript_api as yta

    try:
        text = str(exc).strip()
    except Exception:  # noqa: BLE001 - library __str__ can fail on odd instances
        text = ""
    first = (text.splitlines() or [""])[0][:300]
    msg = f"youtube-transcript-api: {type(exc).__name__}: {first}".rstrip(": ")
    if isinstance(exc, (yta.RequestBlocked, yta.PoTokenRequired)):   # IpBlocked subclasses RequestBlocked
        return Blocked(msg, video_id=video_id)
    if isinstance(exc, yta.AgeRestricted):
        return VideoUnavailable(msg, "Age-restricted: youtube-transcript-api cannot fetch it; use the ytdlp "
                                     "provider (optionally --cookies FILE).", video_id=video_id)
    if isinstance(exc, (yta.TranscriptsDisabled, yta.NoTranscriptFound)):
        return VideoUnavailable(msg, NO_CAPTIONS_HINT, video_id=video_id)
    if isinstance(exc, (yta.VideoUnavailable, yta.VideoUnplayable)):
        return VideoUnavailable(msg, video_id=video_id)
    if isinstance(exc, yta.InvalidVideoId):
        return BadInput(msg, "Pass a YouTube URL or 11-character video id.")
    if isinstance(exc, yta.YouTubeRequestFailed):
        return NetworkError(msg, video_id=video_id)
    if isinstance(exc, yta.YouTubeTranscriptApiException):
        return ProcessFailed(msg, video_id=video_id)
    if isinstance(exc, (OSError, ConnectionError)) or type(exc).__module__.startswith(("requests", "urllib3")):
        return NetworkError(msg, video_id=video_id)
    return None


class YtaCaptions:
    name = "yta"

    def __init__(self, cfg: Config, cache: Cache):
        self.cfg, self.cache = cfg, cache

    def available(self) -> Availability:
        if importlib.util.find_spec("youtube_transcript_api") is None:
            return Availability(False, "youtube-transcript-api not installed",
                                "Run via `uv run scripts/yt.py ...` (pins youtube-transcript-api==1.2.4).")
        return Availability(True)

    def _call(self, video_id: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - mapped to typed errors below
            mapped = _map_yta_error(exc, video_id)
            if mapped is None:
                raise
            raise mapped from None

    def _list(self, video_id: str):
        from youtube_transcript_api import YouTubeTranscriptApi

        return self._call(video_id, lambda: list(YouTubeTranscriptApi().list(video_id)))

    @staticmethod
    def _ref(tr) -> TrackRef:
        code = tr.language_code
        return TrackRef(source="yta", code=code, lang=(code or "").split("-")[0] or None,
                        kind="generated" if tr.is_generated else "manual", name=tr.language, formats=["yta"])

    def inventory(self, meta: VideoMeta) -> list[TrackRef]:
        """Network: lists the video's original tracks (translations are never listed)."""
        return [self._ref(tr) for tr in self._list(meta.video_id)]

    def _choose(self, transcripts: list, track: TrackRef | None):
        if track is not None:
            want_gen = track.is_generated
            code = track.code[: -len("-orig")] if track.code.endswith("-orig") else track.code
            exact = [t for t in transcripts if t.language_code == code]
            exact.sort(key=lambda t: t.is_generated != want_gen)
            return exact[0] if exact else None
        lang = self.cfg.lang or "en"

        def rank(t):
            base = (t.language_code or "").split("-")[0]
            return (t.is_generated, base != lang, base != "en")

        return min(transcripts, key=rank) if transcripts else None

    def fetch(self, video_id: str, track: TrackRef | None, window: Window | None = None) -> Transcript:
        base = {"video_id": video_id, "source": self.name, "tool": "youtube-transcript-api",
                "tool_version": YTA_VERSION, "v": SCHEMA_V}
        if track is not None and track.source == "yta":
            parts = {**base, "track": track.code, "kind": track.kind}
            t = _cached(self.cache, parts)
            if t is not None:
                return _apply_window(t, window)
        transcripts = self._list(video_id)
        chosen = self._choose(transcripts, track)
        if chosen is None:
            raise VideoUnavailable(f"no matching caption track via youtube-transcript-api for {video_id}",
                                   NO_CAPTIONS_HINT, wanted=track.code if track else None)
        ref = self._ref(chosen)
        parts = {**base, "track": ref.code, "kind": ref.kind}
        t = _cached(self.cache, parts)
        if t is None:
            fetched = self._call(video_id, lambda: list(chosen.fetch()))
            segs = []
            for sn in fetched:
                text = _clean(getattr(sn, "text", ""))
                if not text:
                    continue
                start = float(sn.start)
                segs.append(Segment(id="", start=start, end=start + max(0.0, float(sn.duration or 0)), text=text))
            segments = _clamp(dedupe_segments(segs))
            if not segments:
                raise VideoUnavailable(f"caption track {ref.code!r} is empty", NO_CAPTIONS_HINT)
            t = Transcript(video_id=video_id, segments=segments, source=self.name, track=ref,
                           is_generated=ref.is_generated, language=ref.lang,
                           provenance={"tool": "youtube-transcript-api", "tool_version": YTA_VERSION,
                                       "format": "yta", "fetched_at": _now(), "track_name": ref.name})
            self.cache.put("transcript", parts, t.to_dict())
        return _apply_window(t, window)
