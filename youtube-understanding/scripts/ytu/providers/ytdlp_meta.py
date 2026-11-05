"""Video metadata via ``yt-dlp -J --skip-download`` (research 2.B, 4.4 step 1).

Command (always via proc):
    proc.run(proc.ytdlp_cmd("-J", "--skip-download", *cookie_args(cfg), ref.canonical_url),
             timeout=cfg.timeout("metadata"), env=proc.ytdlp_env(), check=False)
    rc != 0  -> raise errors.classify_ytdlp_error(res.stderr, res.returncode)
    stdout   -> json.loads -> parse_info()

Cache: namespace "metadata" (TTL 24 h), parts {"video_id": id, "yt_dlp": proc.YTDLP_VERSION, "v": 1};
store ``VideoMeta.to_dict()``. Signed track URLs are never stored (TrackRef has no url field).
Cookies (opt-in): ``cfg.cookies`` -> ``--cookies FILE``; ``cfg.cookies_from_browser`` (if the
config ever grows that field) -> ``--cookies-from-browser SPEC``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from .. import proc
from ..cache import Cache
from ..config import Config
from ..errors import BadInput, ProcessFailed, classify_ytdlp_error
from ..models import Chapter, TrackRef, VideoMeta, VideoRef
from ..util.sanitize import strip_control

log = logging.getLogger("ytu.ytdlp_meta")

SCHEMA_V = 1
_DROP_TRACK_KEYS = {"live_chat"}


def cookie_args(cfg: Config) -> list[str]:
    """["--cookies", FILE] and/or ["--cookies-from-browser", SPEC]; [] when not configured."""
    args: list[str] = []
    if getattr(cfg, "cookies", None):
        args += ["--cookies", str(cfg.cookies)]
    browser = getattr(cfg, "cookies_from_browser", None)
    if browser:
        args += ["--cookies-from-browser", str(browser)]
    return args


def _base_lang(code: str) -> str | None:
    if not code:
        return None
    if code.endswith("-orig"):
        code = code[: -len("-orig")]
    return code.split("-")[0] or None


def _tracks(table: dict | None, kind: str) -> list[TrackRef]:
    out = []
    for code, entries in (table or {}).items():
        if code in _DROP_TRACK_KEYS or not isinstance(entries, list):
            continue
        entries = [e for e in entries if isinstance(e, dict)]
        out.append(TrackRef(source="ytdlp", code=code, lang=_base_lang(code), kind=kind,
                            name=(entries[0].get("name") if entries else None),
                            formats=[e["ext"] for e in entries if e.get("ext")]))
    return out


def _chapters(raw: list | None, duration: float | None) -> list[Chapter]:
    raw = [c for c in (raw or []) if isinstance(c, dict) and c.get("start_time") is not None]
    out = []
    for i, c in enumerate(raw):
        start = float(c["start_time"])
        end = c.get("end_time")
        if end is None:
            end = raw[i + 1]["start_time"] if i + 1 < len(raw) else duration
        if end is None:
            continue
        out.append(Chapter(start=start, end=float(end), title=strip_control(str(c.get("title") or ""))))
    return out


def parse_info(info: dict) -> VideoMeta:
    """Pure: yt-dlp info dict -> VideoMeta. Manual subs kind="manual", automatic_captions
    kind="auto"; ``live_chat`` dropped; translated auto keys kept (strategy never picks them)."""
    if not isinstance(info, dict) or not info.get("id"):
        raise ProcessFailed("yt-dlp returned no video info", "Re-run with -v; the URL may not be a single video.")
    if info.get("_type") == "playlist":
        raise BadInput("URL resolved to a playlist", "Pass a single video URL.")
    dur = info.get("duration")
    duration = float(dur) if dur not in (None, "") else None
    desc = info.get("description")
    return VideoMeta(
        video_id=info["id"],
        title=strip_control(str(info.get("title") or "")),
        duration=duration,
        channel=info.get("channel") or info.get("uploader"),
        channel_id=info.get("channel_id"),
        upload_date=info.get("upload_date"),
        description=strip_control(desc) if isinstance(desc, str) else None,
        language=info.get("language"),
        availability=info.get("availability"),
        live_status=info.get("live_status"),
        age_limit=info.get("age_limit"),
        view_count=info.get("view_count"),
        webpage_url=info.get("webpage_url"),
        chapters=_chapters(info.get("chapters"), duration),
        tracks=_tracks(info.get("subtitles"), "manual") + _tracks(info.get("automatic_captions"), "auto"),
        yt_dlp_version=proc.YTDLP_VERSION,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _cache_parts(video_id: str) -> dict:
    return {"video_id": video_id, "yt_dlp": proc.YTDLP_VERSION, "v": SCHEMA_V}


def fetch_metadata(cfg: Config, cache: Cache, ref: VideoRef) -> VideoMeta:
    """Cached metadata for ``ref``. Live/availability policy is strategy.check_video_ok's job;
    yt-dlp failures (blocked, private, network...) raise typed errors via classify_ytdlp_error."""
    parts = _cache_parts(ref.video_id)
    hit = cache.get("metadata", parts)
    if hit:
        log.debug("metadata cache hit %s", ref.video_id)
        return VideoMeta.from_dict(hit)
    res = proc.run(proc.ytdlp_cmd("-J", "--skip-download", *cookie_args(cfg), ref.canonical_url),
                   timeout=cfg.timeout("metadata"), env=proc.ytdlp_env(), check=False)
    if res.returncode:
        raise classify_ytdlp_error(res.stderr, res.returncode)
    try:
        info = json.loads(res.stdout)
    except (TypeError, ValueError):
        raise ProcessFailed("yt-dlp -J returned invalid JSON", stderr_tail=(res.stderr or "")[-800:]) from None
    meta = parse_info(info)
    if meta.video_id != ref.video_id:
        log.warning("yt-dlp returned id %s for %s", meta.video_id, ref.video_id)
    cache.put("metadata", parts, meta.to_dict())
    return meta
