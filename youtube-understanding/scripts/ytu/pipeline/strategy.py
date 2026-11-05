"""Auto strategy (research 4.4), transcript selection, and the `inspect` / `transcript` commands.

Strategy resolution (``plan``):
    cfg.strategy "transcript" -> multimodal "none"        (captions only, never Gemini)
                 "local"      -> multimodal "host-frames" (captions/ASR + frames for the host agent)
                 "gemini"     -> Gemini required (MissingKey without a key)
                 "auto"       -> Gemini when a key is present, else "host-frames"
Gemini sizing: <=600 s static, high res if dur*312 < max_input_tokens; <=3 h static low if
dur*102 < max_input_tokens; otherwise chapter/fixed windows (make_windows). ``ask`` prefers
agentic unless the probe cache recorded it as unavailable for the model.
"""

from __future__ import annotations

import json
import logging
import math

from .. import config as config_mod
from ..cache import Cache
from ..config import Config
from ..errors import (BadInput, Blocked, LiveStream, MissingDependency, MissingKey, NetworkError,
                      ProcessFailed, VideoUnavailable, YTUError)
from ..models import CostEstimate, Plan, TrackRef, Transcript, VideoMeta, VideoRef, Window  # noqa: F401
from ..urls import fmt_ts, ts_link

log = logging.getLogger("ytu.strategy")

VISUAL_CUES = ("as you can see", "this code", "on the screen", "on screen", "diagram", "slide",
               "let me show", "look at", "here we have", "this function", "terminal")

SHORT_VIDEO_S = 600
LONG_VIDEO_S = 3 * 3600
TOKENS_PER_S_HIGH = 312   # 280 tok/frame @1 fps + 32 audio
TOKENS_PER_S_LOW = 102    # 70 tok/frame @1 fps + 32 audio
HANDOFF_CHARS = 50_000


# --- video checks & track choice -----------------------------------------------------

def check_video_ok(meta: VideoMeta, need_gemini: bool) -> None:
    """Raise LiveStream if live_status in {"is_live","is_upcoming"} or duration is None.
    Raise VideoUnavailable if availability in {"private","subscriber_only","premium_only"}.
    If need_gemini and availability not in {"public", None} -> VideoUnavailable."""
    if meta.live_status in ("is_live", "is_upcoming") or meta.duration is None:
        raise LiveStream(f"{meta.video_id} is a live/upcoming stream (no fixed duration)")
    if meta.availability in ("private", "subscriber_only", "premium_only"):
        raise VideoUnavailable(f"{meta.video_id} is {meta.availability}",
                               "Only public/unlisted videos are supported (members-only needs --cookies).")
    if need_gemini and meta.availability not in ("public", None):
        raise VideoUnavailable(f"{meta.video_id} is {meta.availability}; Gemini only accepts public videos",
                               "Gemini only accepts public videos; use --strategy transcript|local")


def clamp_window(window: Window | None, duration: float | None) -> Window | None:
    """Clamp a user window to the video end (BadInput if it starts at/after the end)."""
    if window is None or not duration:
        return window
    if window.start >= duration:
        raise BadInput(f"window starts after the video ends ({fmt_ts(duration)})")
    return Window(window.start, float(duration), window.reason) if window.end > duration else window


def _base(code: str | None) -> str | None:
    return code.split("-")[0].lower() if code else None


def pick_track(meta: VideoMeta, lang: str | None = None) -> TrackRef | None:
    """Exactly one track (research 4.4 step 2):
    subtitles[lang] -> subtitles[en] -> subtitles[any] -> auto[f"{lang}-orig"] -> auto[lang]
    (only if no "-orig" key exists at all) -> None (caller falls through to yta/ASR).
    Never translated auto tracks, never "live_chat"."""
    lang = (lang or _base(meta.language) or "en").lower()
    tracks = [t for t in meta.tracks if t.code != "live_chat"]
    manual = [t for t in tracks if t.kind == "manual"]
    auto = [t for t in tracks if t.kind != "manual"]
    for want in (lang, "en"):
        exact = [t for t in manual if t.code.lower() == want]
        based = [t for t in manual if _base(t.code) == want]
        if exact or based:
            return (exact or based)[0]
    if manual:
        return manual[0]
    origs = [t for t in auto if t.code.endswith("-orig")]
    if origs:
        for want in (lang, _base(meta.language)):
            hit = [t for t in origs if want and t.code.lower() == f"{want}-orig"]
            if hit:
                return hit[0]
        return origs[0] if len(origs) == 1 else sorted(origs, key=lambda t: t.code)[0]
    hit = [t for t in auto if t.code.lower() == lang]
    return hit[0] if hit else None


def get_transcript(cfg: Config, cache: Cache, meta: VideoMeta, *, lang: str | None = None,
                   window: Window | None = None) -> Transcript:
    """Try cfg.transcript_providers in order; skip unavailable providers; on caption-level
    failures try the next. "asr" only if cfg.strategy in {"auto","local"}.
    All fail -> raise the most specific error collected (Blocked > VideoUnavailable > others)."""
    from ..providers import base

    errors: list[YTUError] = []
    tried: list[str] = []
    for name in cfg.transcript_providers:
        if name == "asr" and cfg.strategy not in ("auto", "local"):
            continue
        prov = base.get_transcript_provider(name, cfg, cache)
        av = prov.available()
        if not av.ok:
            log.info("transcript provider %s unavailable: %s", name, av.reason)
            errors.append(MissingDependency(f"{name}: {av.reason}", av.remediation or None))
            continue
        track = None
        if name == "ytdlp":
            track = pick_track(meta, lang or cfg.lang)
            if track is None:
                log.info("no suitable yt-dlp caption track for %s", meta.video_id)
                errors.append(VideoUnavailable(f"no caption track for {meta.video_id}",
                                               "No captions; try --strategy local (ASR)."))
                continue
        elif name == "asr":  # ASR needs the spoken language to pick an engine (parakeet = European only)
            track = TrackRef(source="asr", code="asr", lang=_base(lang or cfg.lang or meta.language), kind="asr")
        tried.append(name)
        try:
            tr = prov.fetch(meta.video_id, track, window)
        except LiveStream:
            raise
        except (VideoUnavailable, NetworkError, ProcessFailed, MissingDependency) as e:
            log.warning("transcript provider %s failed: %s", name, e.message)
            errors.append(e)
            continue
        if not tr.segments and window is None:
            errors.append(VideoUnavailable(f"{name}: empty transcript"))
            continue
        return tr
    for cls in (Blocked, VideoUnavailable, NetworkError, ProcessFailed, MissingDependency):
        for e in errors:
            if isinstance(e, cls):
                e.details.setdefault("providers_tried", tried)
                raise e
    raise VideoUnavailable(f"no transcript source available for {meta.video_id}",
                           "Enable a provider (--transcript-providers ytdlp,yta,asr) or use --strategy local.",
                           providers_tried=tried)


# --- windows & cues ----------------------------------------------------------------

def make_windows(meta: VideoMeta, max_window_s: float = 1200, min_window_s: float = 120,
                 fixed_s: float = 900, overlap_s: float = 15) -> list[Window]:
    """Research 4.11: chapters (merge < min, split > max) else fixed windows with overlap."""
    dur = meta.duration
    if not dur or dur <= 0:
        return []
    chapters = sorted((c for c in meta.chapters if c.start < dur), key=lambda c: c.start)
    groups: list[list[float]] = []  # [start, end, first_chapter_no]
    for n, c in enumerate(chapters, 1):
        s, e = max(0.0, float(c.start)), min(float(c.end or dur), dur)
        if e <= s:
            continue
        if groups and (groups[-1][1] - groups[-1][0]) < min_window_s:
            groups[-1][1] = e
        else:
            groups.append([s, e, n])
    if groups:  # chapters should cover the whole video; cover any head/tail they miss
        groups[0][0] = 0.0
        if dur - groups[-1][1] > 1.0:
            groups.append([groups[-1][1], dur, 0])
    if len(groups) > 1 and groups[-1][1] - groups[-1][0] < min_window_s:
        last = groups.pop()
        groups[-1][1] = last[1]
    out: list[Window] = []
    if groups:
        for s, e, n in groups:
            parts = max(1, math.ceil((e - s) / max_window_s))
            step = (e - s) / parts
            for i in range(parts):
                out.append(Window(round(s + i * step, 3), round(s + (i + 1) * step if i < parts - 1 else e, 3),
                                  f"chapter:{int(n)}" if n else "fixed"))
        return out
    if dur <= fixed_s + min_window_s:
        return [Window(0.0, float(dur), "fixed")]
    s = 0.0
    while s < dur:
        e = min(dur, s + fixed_s)
        if dur - e < min_window_s:
            e = dur
        out.append(Window(round(s, 3), round(e, 3), "fixed"))
        if e >= dur:
            break
        s = e - overlap_s
    return out


def visual_cues(transcript: Transcript) -> list[tuple[float, str, str]]:
    """[(t, segment_id, cue)] for segments containing VISUAL_CUES (case-insensitive); first cue per segment."""
    out = []
    for s in transcript.segments:
        low = s.text.lower()
        for cue in VISUAL_CUES:
            if cue in low:
                out.append((s.start, s.id, cue))
                break
    return out


# --- plan ----------------------------------------------------------------------------

def _agentic_unavailable(cfg: Config, model: str) -> bool:
    try:
        v = Cache(cfg.cache_dir, force=cfg.force_refresh).get("probe", {"agentic": model})
    except (OSError, ValueError):
        return False
    return bool(v and v.get("unavailable"))


def estimate_windows(model: str, durations: list[float], resolution: str, fps: float) -> CostEstimate:
    from ..providers import gemini

    parts = [gemini.estimate_cost(model, d, resolution, fps) for d in durations]
    if len(parts) == 1:
        return parts[0]
    return CostEstimate(model, sum(p.input_tokens for p in parts), sum(p.output_tokens for p in parts),
                        round(sum(p.usd for p in parts), 6),
                        f"sum of {len(parts)} windows; " + (parts[0].basis if parts else ""))


def plan(cfg: Config, meta: VideoMeta, *, command: str, key_present: bool,
         question: str | None = None, window: Window | None = None,
         resolution: str | None = None) -> Plan:
    """Research 4.4 step 3 + 4.6. command in {"analyze","ask","package"}. Records every decision in
    reasons[]. ``resolution`` forces a media resolution (CLI --resolution). Never calls a paid API."""
    providers = [p for p in cfg.transcript_providers if p != "asr" or cfg.strategy in ("auto", "local")]
    reasons: list[str] = []
    strat = cfg.strategy
    if strat == "transcript":
        return Plan("transcript", providers, "none", reasons=["--strategy transcript: captions only, no Gemini"])
    if strat == "local":
        return Plan("local", providers, "host-frames",
                    reasons=["--strategy local: transcript + targeted frames for the host agent's vision"])
    if strat == "gemini" and not key_present:
        config_mod.require_gemini_key()  # raises MissingKey listing where it looked
        raise MissingKey("Gemini API key missing")
    if strat == "auto" and not key_present:
        return Plan("local", providers, "host-frames",
                    reasons=["auto: no Gemini key -> transcript + host-agent frames (no paid calls)"])
    reasons.append(f"{strat}: Gemini key present -> Gemini path")
    model = cfg.models[0]
    dur = float(meta.duration or 0)
    span = window.duration if window else dur
    max_in = cfg.max_input_tokens
    fps = 1.0
    windows: list[Window] = []
    if command == "ask" and window is None and not _agentic_unavailable(cfg, model):
        multimodal, res = "gemini-agentic", resolution or "low"
        reasons.append("ask: agentic first (falls back to static clips if unavailable)")
        reasons.append("agentic token use is not knowable up front; estimate = static over the whole video "
                       "(upper bound); over budget -> static clips around the question")
    else:
        if command == "ask" and window is not None:
            reasons.append("ask with an explicit window/--at: static clip of that span only (no whole-video agentic)")
        elif command == "ask":
            reasons.append(f"ask: agentic recorded unavailable for {model} (probe cache) -> static clips")
        multimodal = "gemini-static"
        if window:
            windows = [window]
            res = resolution or ("high" if span <= SHORT_VIDEO_S and span * TOKENS_PER_S_HIGH < max_in else "low")
            reasons.append(f"window {fmt_ts(window.start)}-{fmt_ts(window.end)} ({span:.0f}s) -> static clip, {res}")
        elif dur <= SHORT_VIDEO_S:
            res = resolution or ("high" if dur * TOKENS_PER_S_HIGH < max_in else "low")
            reasons.append(f"duration {dur:.0f}s <= {SHORT_VIDEO_S}s -> whole-video static, {res}")
        elif dur <= LONG_VIDEO_S and dur * TOKENS_PER_S_LOW < max_in:
            res = resolution or "low"
            reasons.append(f"duration {dur:.0f}s <= 3h and ~{int(dur * TOKENS_PER_S_LOW)} tok < {max_in} "
                           f"-> whole-video static, {res}")
        else:
            res = resolution or "low"
            windows = make_windows(meta)
            src = "chapters" if any(w.reason and w.reason.startswith("chapter") for w in windows) else "fixed 15-min"
            reasons.append(f"duration {dur:.0f}s too long for one static call -> {len(windows)} {src} windows, {res}")
    durations = [w.duration for w in windows] or [span]
    est = estimate_windows(model, durations, res, fps)
    if est.usd > cfg.max_cost_usd and res != "low" and resolution is None:
        res = "low"
        est = estimate_windows(model, durations, res, fps)
        reasons.append(f"downscaled to low resolution to fit max_cost_usd {cfg.max_cost_usd}")
    if multimodal == "gemini-static" and resolution is None and \
            over_budget(cfg, Plan("gemini", providers, multimodal, windows=windows, estimate=est)):
        lower = estimate_windows(model, durations, res, 0.5)
        if not over_budget(cfg, Plan("gemini", providers, multimodal, windows=windows, estimate=lower)):
            fps, est = 0.5, lower
            reasons.append(f"lowered fps to 0.5 to fit the budget (${cfg.max_cost_usd}, {max_in} tok/request)")
    if question:
        reasons.append("user question included in the prompt")
    p = Plan("gemini", providers, multimodal, model=model, resolution=res, fps=fps, windows=windows,
             estimate=est, reasons=reasons)
    why = over_budget(cfg, p)
    if why:
        reasons.append(f"{why}: the call will be refused (OverBudget) unless limits are raised")
    return p


def auto_degrade(cfg: Config, meta: VideoMeta, p: Plan, *, budget: bool = True) -> tuple[Plan, str | None]:
    """--strategy auto only: when the Gemini plan cannot run (non-public video, or over budget after
    downscaling) fall back to transcript + host-agent frames instead of failing. Returns
    (plan, uncertainty or None). Explicit --strategy gemini keeps the hard refusal."""
    if cfg.strategy != "auto" or not p.multimodal.startswith("gemini"):
        return p, None
    if meta.availability not in ("public", None):
        why = (f"Gemini skipped: video is {meta.availability} (Gemini only accepts public videos); "
               "used transcript + host-agent frames")
    elif budget and (ob := over_budget(cfg, p)):
        why = (f"Gemini skipped for cost ({ob}); used transcript + host-agent frames. Raise --max-cost-usd / "
               "--max-input-tokens, or narrow with --start/--end, to use Gemini")
    else:
        return p, None
    q = Plan("local", p.transcript_providers, "host-frames",
             reasons=p.reasons + [f"auto: {why}"])
    return q, why


def over_budget(cfg: Config, p: Plan) -> str | None:
    """Reason string when the plan's estimate exceeds the configured budget, else None."""
    e = p.estimate
    if e is None:
        return None
    per_req = e.input_tokens / max(1, len(p.windows))
    if e.usd > cfg.max_cost_usd:
        return f"estimated ${e.usd:.4f} > --max-cost-usd {cfg.max_cost_usd}"
    if per_req > cfg.max_input_tokens:
        return f"estimated {int(per_req)} input tokens/request > --max-input-tokens {cfg.max_input_tokens}"
    return None


# --- commands ------------------------------------------------------------------------

def _injection_flags(fields: list[tuple[str, str | None]]) -> list[dict]:
    from ..util.sanitize import detect_injection
    return [{"field": name, "patterns": pats} for name, text in fields if (pats := detect_injection(text or ""))]


def _untrusted(flags: list[dict]) -> dict:
    """Marker for raw video text handed to the host agent (titles, descriptions, captions)."""
    from ..util.sanitize import UNTRUSTED_NOTICE
    return {"untrusted": True, "content_notice": UNTRUSTED_NOTICE, "injection_flags": flags}


def key_present() -> bool:
    return config_mod.get_gemini_key() is not None


def run_inspect(cfg: Config, cache: Cache, ref: VideoRef) -> dict:
    """CLI `inspect`: metadata (description truncated to 2000 chars), chosen track, plan.
    No transcript download, no Gemini call."""
    from ..providers import ytdlp_meta

    meta = ytdlp_meta.fetch_metadata(cfg, cache, ref)
    video = meta.to_dict()
    desc = video.pop("description", None) or ""
    video["description_excerpt"] = desc[:2000]
    video["description_truncated"] = len(desc) > 2000
    for c in video.get("chapters", []):
        c["timestamp"], c["url"] = fmt_ts(c["start"]), ts_link(meta.video_id, c["start"])
    for t in video.get("tracks", []):
        t.pop("formats", None)
    kp = key_present()
    flags = _injection_flags([("title", meta.title), ("description", desc[:2000])] +
                             [(f"chapter:{n}", c.title) for n, c in enumerate(meta.chapters, 1)])
    out: dict = {"video": video, "chosen_track": None, "key_present": kp, **_untrusted(flags)}
    t = pick_track(meta, cfg.lang)
    out["chosen_track"] = t.to_dict() if t else None
    try:
        out["plan"] = plan(cfg, meta, command="analyze", key_present=kp).to_dict()
    except YTUError as e:
        out["plan"] = None
        out["plan_error"] = e.to_dict()
    warnings = []
    try:
        check_video_ok(meta, need_gemini=False)
    except YTUError as e:
        warnings.append(e.message)
    out["warnings"] = warnings
    return out


def _seg_line(s, fmt: str, n: int) -> str:
    if fmt == "txt":
        return f"[{fmt_ts(s.start)}] {s.text}\n"
    from .package import srt_time
    return f"{n}\n{srt_time(s.start)} --> {srt_time(max(s.end, s.start + 0.001))}\n{s.text}\n\n"


def run_transcript(cfg: Config, cache: Cache, ref: VideoRef, *, lang: str | None = None,
                   window: Window | None = None, fmt: str = "json", max_chars: int = HANDOFF_CHARS,
                   offset: int = 0) -> dict:
    """CLI `transcript`. fmt json -> {"video_id","source","track","is_generated","language",
    "segments":[...with "timestamp","url"]}; fmt srt|txt -> {"video_id","source","format","text"}.
    Output bounded at a segment boundary: {"truncated": true, "next_offset": <segment index>}."""
    from ..providers import ytdlp_meta

    if fmt == "text":
        fmt = "txt"
    if fmt not in ("json", "srt", "txt"):
        raise BadInput(f"unknown format {fmt!r}", "Use json, srt or txt.")
    if offset < 0 or max_chars <= 0:
        raise BadInput("--offset must be >= 0 and --max-chars > 0")
    meta = ytdlp_meta.fetch_metadata(cfg, cache, ref)
    check_video_ok(meta, need_gemini=False)
    tr = get_transcript(cfg, cache, meta, lang=lang, window=window)
    segs = tr.text_between(window.start, window.end) if window else tr.segments
    out: dict = {"video_id": meta.video_id, "source": tr.source,
                 "track": tr.track.to_dict() if tr.track else None, "is_generated": tr.is_generated,
                 "language": tr.language, "total_segments": len(segs), "offset": offset}
    used, items, text_parts = 0, [], []
    idx = offset
    for idx in range(offset, len(segs)):
        s = segs[idx]
        if fmt == "json":
            d = {**s.to_dict(), "timestamp": fmt_ts(s.start), "url": ts_link(meta.video_id, s.start)}
            if d.get("words") is None:
                d.pop("words", None)
            size = len(json.dumps(d, ensure_ascii=False, separators=(",", ":"))) + 1  # as emitted (compact)
        else:
            d = _seg_line(s, fmt, idx + 1)
            size = len(d)
        if used + size > max_chars and (items or text_parts):
            out["truncated"], out["next_offset"] = True, idx
            break
        used += size
        (items if fmt == "json" else text_parts).append(d)
    else:
        out["truncated"] = False
    if fmt == "json":
        out["segments"] = items
    else:
        out["format"], out["text"] = fmt, "".join(text_parts)
    shown = segs[offset:offset + (len(items) if fmt == "json" else len(text_parts))]
    out.update(_untrusted(_injection_flags([(x.id, x.text) for x in shown])))
    if tr.is_generated:
        out["note"] = "auto-generated captions: wording, names and punctuation may be wrong"
    return out
