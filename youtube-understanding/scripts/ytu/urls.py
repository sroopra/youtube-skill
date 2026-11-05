"""YouTube URL parsing/normalization and timestamp helpers (stdlib only, no network)."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

from .errors import BadInput
from .models import VideoRef

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtube-nocookie.com", "www.youtube-nocookie.com", "gaming.youtube.com",
}
_SHORT_HOSTS = {"youtu.be", "www.youtu.be"}
_PATH_ID_PREFIXES = ("shorts", "embed", "live", "v", "e")
_CHANNEL_PREFIXES = ("channel", "c", "user")

_HMS_RE = re.compile(r"^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s?)?$")


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def is_video_id(s: str) -> bool:
    return bool(_ID_RE.match(s or ""))


def parse_url(raw: str) -> VideoRef:
    """Parse any supported YouTube URL form (or a bare 11-char id) into a VideoRef.

    Raises BadInput for playlists without a video, channels, non-YouTube hosts,
    malformed ids.
    """
    s = (raw or "").strip()
    if not s:
        raise BadInput("empty URL", "Pass a YouTube URL or 11-character video id.")
    if is_video_id(s):
        return VideoRef(s, canonical_url(s), None)
    if "://" not in s:
        s = "https://" + s
    try:
        parts = urlsplit(s)
    except ValueError as e:
        raise BadInput(f"unparseable URL: {e}") from None
    if parts.scheme not in ("http", "https"):
        raise BadInput(f"unsupported URL scheme: {parts.scheme}", "Use an https://www.youtube.com/watch?v=... URL.")
    host = (parts.hostname or "").lower()
    query = parse_qs(parts.query)
    frag = parse_qs(parts.fragment) if "=" in parts.fragment else {}
    segs = [p for p in parts.path.split("/") if p]

    vid = None
    if host in _SHORT_HOSTS:
        vid = segs[0] if segs else None
    elif host in _YT_HOSTS:
        if segs[:1] == ["watch"] or (not segs and "v" in query):
            vid = (query.get("v") or [None])[0]
        elif len(segs) >= 2 and segs[0] in _PATH_ID_PREFIXES:
            vid = segs[1]
        elif segs[:1] == ["playlist"] or ("list" in query and "v" not in query):
            raise BadInput("playlist URL without a video id",
                           "Playlists are not supported; pass a single video URL (watch?v=...).")
        elif segs and (segs[0].startswith("@") or segs[0] in _CHANNEL_PREFIXES):
            raise BadInput("channel URL, not a video",
                           "Open the channel/live page in a browser and pass the specific watch?v=... URL.")
        elif segs[:1] == ["attribution_link"] and "u" in query:
            return parse_url("https://www.youtube.com" + query["u"][0])
    else:
        raise BadInput(f"not a YouTube URL: host {host or '?'}", "Only youtube.com / youtu.be URLs are supported.")

    if not vid or not is_video_id(vid):
        raise BadInput(f"could not find a valid 11-character video id in {raw!r}",
                       "Pass a URL like https://www.youtube.com/watch?v=VIDEO_ID or the bare id.")

    start = None
    for key in ("t", "start", "time_continue"):
        val = (query.get(key) or frag.get(key) or [None])[0]
        if val:
            try:
                start = parse_ts(val)
            except BadInput:
                start = None
            break
    return VideoRef(vid, canonical_url(vid), start)


def parse_ts(s: str | int | float) -> float:
    """"754" | "754s" | "754.5" | "12:34" | "1:02:03" | "12m34s" | "1h2m3s" -> seconds."""
    if isinstance(s, (int, float)):
        if s < 0:
            raise BadInput(f"negative timestamp: {s}")
        return float(s)
    t = (s or "").strip().lower()
    if not t:
        raise BadInput("empty timestamp")
    if ":" in t:
        parts = t.split(":")
        if len(parts) > 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
            raise BadInput(f"bad timestamp {s!r}", "Use SS, MM:SS, HH:MM:SS or 1h2m3s.")
        if any(float(p) >= 60 for p in parts[1:]):  # MM:SS / HH:MM:SS: minutes and seconds < 60
            raise BadInput(f"bad timestamp {s!r}: minute/second fields must be < 60",
                           "Use SS, MM:SS, HH:MM:SS or 1h2m3s.")
        total = 0.0
        for p in parts:
            total = total * 60 + float(p)
        return total
    m = _HMS_RE.match(t)
    if not m or not any(m.groups()):
        raise BadInput(f"bad timestamp {s!r}", "Use SS, MM:SS, HH:MM:SS or 1h2m3s.")
    h, mi, se = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + se


def fmt_ts(seconds: float) -> str:
    """754.9 -> "12:34"; 3723 -> "01:02:03" (floor to whole seconds)."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def ts_link(video_id: str, seconds: float) -> str:
    """Citation link: https://www.youtube.com/watch?v=ID&t=754s (floored)."""
    return f"{canonical_url(video_id)}&t={max(0, int(seconds))}s"
