"""Workflows A (understand) and B (tutorial) — `analyze` command (research 4.10, 4.11).

Flow: fetch_metadata -> plan -> check_video_ok -> get_transcript ->
  multimodal "gemini-*": per window AnalyzeRequest(prompt with wrap_untrusted(title/description/
      chapter titles/transcript excerpt), schema=EXTRACTION_SCHEMA[mode]) -> evidence items
      (sources "gemini:static:<model>"); model timestamps snapped to transcript segments
      (> 5 s off -> flagged); speech claims corroborated against the transcript via
      evidence.match_speech; code without a frame stays "unverified".
  "host-frames"/"none": transcript-only digest (no model call): chapters, transcript chunks
      (<= 50k chars each), visual cues, suggested frame timestamps and HOST-AGENT tasks.
      NO summary is invented by the script: summary is None, the host agent writes it.
cfg.dry_run -> {"plan", "dry_run": true} without any paid call.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field

from ..cache import Cache
from ..config import Config
from ..errors import NetworkError, OverBudget, ProcessFailed, YTUError
from ..models import (AnalyzeRequest, AnalyzeResult, Evidence, EvidenceItem, Gap, Plan, Transcript, VideoMeta,
                      VideoRef, Window)
from ..urls import canonical_url, fmt_ts, parse_ts, ts_link
from ..util.paths import yt_cmd
from ..util.sanitize import UNTRUSTED_NOTICE, detect_injection, wrap_untrusted
from . import evidence as ev
from . import strategy

log = logging.getLogger("ytu.analyze")

MODES = ("understand", "tutorial")
HANDOFF_CHARS = 50_000
PROMPT_TRANSCRIPT_CHARS = 30_000
SPEECH_SEARCH_S = 60.0
NO_SPEECH_GAP_S = 30.0

# Numeric seconds: the Gemini provider normalises "t"/"start"/"end" to floats before schema validation.
_TS = {"type": "number", "description": "ABSOLUTE seconds from the start of the video, e.g. 754.2"}
_ITEM = {"type": "object", "properties": {"text": {"type": "string"}, "t": _TS,
                                          "quote": {"type": "string", "description": "verbatim words spoken, if any"},
                                          "demonstrated": {"type": "boolean"}},
         "required": ["text", "t"]}
_BASE_PROPS = {
    "summary": {"type": "string"},
    "key_points": {"type": "array", "items": _ITEM},
    "segments": {"type": "array", "items": {"type": "object", "properties": {
        "start": _TS, "end": _TS, "summary": {"type": "string"}}, "required": ["start", "summary"]}},
    "visual_events": {"type": "array", "items": {"type": "object", "properties": {
        "t": _TS, "type": {"type": "string", "enum": ["code", "diagram", "slide", "ui", "demo"]},
        "description": {"type": "string"}, "transcribed_text": {"type": "string"}},
        "required": ["t", "type", "description"]}},
    "claims": {"type": "array", "items": _ITEM},
}
_TUTORIAL_PROPS = {
    "concepts": {"type": "array", "items": _ITEM},
    "architecture": {"type": "array", "items": _ITEM},
    "dependencies": {"type": "array", "items": _ITEM},
    "commands": {"type": "array", "items": _ITEM},
    "code": {"type": "array", "items": {"type": "object", "properties": {
        "text": {"type": "string", "description": "code exactly as visible on screen"}, "t": _TS,
        "language": {"type": "string"}, "demonstrated": {"type": "boolean"}}, "required": ["text", "t"]}},
    "steps": {"type": "array", "items": _ITEM},
}
EXTRACTION_SCHEMA = {
    "understand": {"type": "object", "properties": _BASE_PROPS, "required": ["summary", "key_points"]},
    "tutorial": {"type": "object", "properties": {**_BASE_PROPS, **_TUTORIAL_PROPS},
                 "required": ["summary", "key_points", "steps"]},
}
TUTORIAL_KEYS = tuple(_TUTORIAL_PROPS)
# tutorial list -> evidence kind
_TUTORIAL_KIND = {"concepts": "inference", "architecture": "inference", "dependencies": "inference",
                  "commands": "code", "code": "code", "steps": "inference"}
_VISUAL_KIND = {"code": "code", "diagram": "diagram", "slide": "slide", "ui": "ui", "demo": "ui"}

TASK = {
    "understand": "Explain what this video covers: a short summary, key points, per-section summaries, "
                  "visual events (code, diagrams, slides, UI) and notable claims.",
    "tutorial": "Extract a learning guide from this tutorial: concepts, architecture, dependencies, commands, "
                "code exactly as shown on screen or spoken, and the ordered steps. Set demonstrated=true only "
                "when the item is actually shown or said in the video; false when you infer it.",
}
RULES = ("Rules: report only what is said or shown in the video; never invent quotes, code or timestamps. "
         "All timestamps are ABSOLUTE seconds from the start of the video (numbers, e.g. 754.2). Transcribe code and "
         "commands exactly as visible; omit anything unreadable. If unsure, omit the item. "
         "Return JSON matching the schema.")


@dataclass
class AnalysisRun:
    """Internal result shared by run_analyze and package."""
    meta: VideoMeta
    plan: Plan
    transcript: Transcript | None
    evidence: Evidence
    result: dict
    frames_hint: list[float] = field(default_factory=list)
    base_uncertainties: list[str] = field(default_factory=list)  # without the evidence-status notes


def _video(meta: VideoMeta) -> dict:
    return {"video_id": meta.video_id, "title": meta.title, "channel": meta.channel, "duration": meta.duration,
            "url": canonical_url(meta.video_id), "language": meta.language}


def chapters_out(meta: VideoMeta) -> list[dict]:
    return [{"start": c.start, "end": c.end, "title": c.title, "timestamp": fmt_ts(c.start),
             "url": ts_link(meta.video_id, c.start)} for c in meta.chapters]


def _caption_source(tr: Transcript) -> str:
    if tr.source.startswith("asr"):
        return tr.source.replace("asr-", "asr:", 1)
    return "captions:auto" if tr.is_generated else "captions:manual"


def _transcript_info(tr: Transcript | None) -> dict | None:
    if tr is None:
        return None
    return {"source": tr.source, "track": tr.track.code if tr.track else None, "is_generated": tr.is_generated,
            "language": tr.language, "segments": len(tr.segments)}


def chunk_transcript(tr: Transcript, max_chars: int = HANDOFF_CHARS) -> list[dict]:
    """Split "[MM:SS] text" lines into chunks <= max_chars at segment boundaries."""
    chunks, cur, size, first = [], [], 0, 0
    for i, s in enumerate(tr.segments):
        line = f"[{fmt_ts(s.start)}] {s.text}\n"
        if cur and size + len(line) > max_chars:
            chunks.append((first, i, "".join(cur)))
            cur, size, first = [], 0, i
        cur.append(line)
        size += len(line)
    if cur:
        chunks.append((first, len(tr.segments), "".join(cur)))
    out = []
    for n, (a, b, text) in enumerate(chunks):
        segs = tr.segments[a:b]
        out.append({"index": n, "offset": a, "segment_ids": [segs[0].id, segs[-1].id], "start": segs[0].start,
                    "end": segs[-1].end, "timestamp": fmt_ts(segs[0].start), "chars": len(text), "text": text})
    return out


def speech_gaps(tr: Transcript, duration: float | None, min_gap: float = NO_SPEECH_GAP_S,
                start: float = 0.0) -> list[Gap]:
    """Silent stretches in [start, duration]; ``start`` is the window start for windowed runs."""
    gaps, prev = [], start
    for s in tr.segments:
        if s.end <= start:
            continue
        if s.start - prev >= min_gap:
            gaps.append(Gap(prev, s.start, "no speech in captions (possible visual-only section)"))
        prev = max(prev, s.end)
    if duration and duration - prev >= min_gap:
        gaps.append(Gap(prev, duration, "no speech in captions (possible visual-only section)"))
    return gaps


def suggest_frames(meta: VideoMeta, cues: list[tuple[float, str, str]], extra: list[float] = (),
                   gaps: list[Gap] = (), limit: int = 24, min_sep: float = 10.0) -> list[float]:
    """Visual-cue times (+2 s), Gemini visual events, chapter starts (+5 s) and no-speech midpoints,
    deduped by ``min_sep`` seconds, capped at ``limit``, clamped inside the video."""
    dur = meta.duration or 0
    cand = [round(t, 1) for t in extra]
    cand += [round(t + 2.0, 1) for t, _, _ in cues]
    cand += [round((g.t_start + g.t_end) / 2, 1) for g in gaps]
    cand += [round(c.start + 5.0, 1) for c in meta.chapters]
    out: list[float] = []
    for t in cand:  # priority order: model events, cues, gaps, chapters
        t = min(max(0.0, t), max(0.0, dur - 0.5)) if dur else max(0.0, t)
        if all(abs(t - o) >= min_sep for o in out):
            out.append(t)
        if len(out) >= limit:
            break
    return sorted(out)


# --- Gemini path ---------------------------------------------------------------------

def build_prompt(meta: VideoMeta, tr: Transcript | None, mode: str, window: Window | None,
                 question: str | None = None) -> str:
    where = (f"Analyze only {fmt_ts(window.start)}-{fmt_ts(window.end)} of the video." if window
             else "Analyze the whole video.")
    chap = "\n".join(f"{fmt_ts(c.start)} {c.title}" for c in meta.chapters) or "(none)"
    parts = [TASK[mode], where, RULES, UNTRUSTED_NOTICE,
             wrap_untrusted("metadata", f"title: {meta.title}\nchannel: {meta.channel or ''}\nchapters:\n{chap}"),
             wrap_untrusted("description", (meta.description or "")[:2000])]
    if tr is not None:
        segs = tr.text_between(window.start, window.end) if window else tr.segments
        text = "".join(f"[{fmt_ts(s.start)}] {s.text}\n" for s in segs)
        if len(text) > PROMPT_TRANSCRIPT_CHARS:
            text = text[:PROMPT_TRANSCRIPT_CHARS] + "\n[transcript excerpt truncated]"
        parts.append("Transcript excerpt (captions, may contain errors):")
        parts.append(wrap_untrusted("captions", text))
    if question:
        parts.append("Focus on this user question (answer only from the video):")
        parts.append(wrap_untrusted("user-question", question))
    return "\n\n".join(parts)


def _parse_model_ts(raw) -> float | None:
    try:
        return parse_ts(str(raw).strip())
    except (YTUError, ValueError, TypeError):
        return None


def window_shift(raws, window: Window | None, slack: float = 5.0) -> float:
    """Batch rule (same as the Gemini provider's normalize_times): if NONE of the parsed times fit
    the absolute window but ALL fit the clip length, the model used window-relative times ->
    return window.start (the offset to add to every one); else 0. Never decided per item."""
    vals = [v for v in (_parse_model_ts(r) for r in raws) if v is not None]
    if not window or window.start <= slack or not vals:
        return 0.0
    in_abs = all(window.start - slack <= v <= window.end + slack for v in vals)
    in_rel = all(-slack <= v <= window.duration + slack for v in vals)
    return window.start if not in_abs and in_rel else 0.0


def model_time(raw, window: Window | None, duration: float | None,
               shift: float = 0.0) -> tuple[float | None, str]:
    """Parse a model timestamp (absolute unless the batch ``shift`` from window_shift says otherwise).
    Times outside the analysed window are rejected (None), never moved or kept as a false citation."""
    t = _parse_model_ts(raw)
    if t is None:
        return None, f"unparseable model timestamp {str(raw)[:20]!r}"
    note = ""
    if shift:
        t, note = t + shift, "model timestamps looked window-relative; offset added"
    if duration and t > duration + 5:
        return None, f"model timestamp {fmt_ts(t)} beyond video end"
    if window and not (window.start - 5 <= t <= window.end + 5):
        return None, (f"model timestamp {fmt_ts(t)} outside the analysed window "
                      f"{fmt_ts(window.start)}-{fmt_ts(window.end)}")
    return t, note


class _Builder:
    """Turns model JSON into evidence items with snapping, corroboration and dedupe."""

    def __init__(self, meta: VideoMeta, tr: Transcript | None, model: str, mode: str):
        self.meta, self.tr, self.items = meta, tr, []
        self.source = f"gemini:{mode}:{model}"
        self.cap_source = _caption_source(tr) if tr else None
        self.dropped = 0
        self.out_of_window = 0

    def add(self, text: str, raw_t, kind: str, window: Window | None, *, quote: str | None = None,
            demonstrated: bool | None = None, note: str = "", shift: float = 0.0) -> EvidenceItem | None:
        text = (text or "").strip()
        if not text:
            return None
        t, tnote = model_time(raw_t, window, self.meta.duration, shift)
        if t is None:
            self.dropped += 1
            if "outside the analysed window" in tnote:
                self.out_of_window += 1
            log.info("dropping model item without usable timestamp: %s", tnote)
            return None
        notes = [n for n in (note, tnote) if n]
        seg, delta = ev.snap_to_segment(t, self.tr)
        seg_ids: list[str] = []
        if seg is not None and delta <= ev.SNAP_TOLERANCE_S:
            t = seg.start if delta > 0 else t
            seg_ids = [seg.id]
        elif seg is not None:
            notes.append(f"model timestamp {delta:.0f}s from nearest transcript segment; not snapped")
        status, sources, conf = "single-source", [self.source], "medium"
        if kind in ("speech", "inference") and self.tr is not None:
            near = self.tr.text_between(t - SPEECH_SEARCH_S, t + SPEECH_SEARCH_S)
            hit = ev.match_speech(quote or text, near)
            if hit:
                status, conf = "corroborated", "high"
                sources.append(self.cap_source)
                seg_ids = hit
                first = next(s for s in near if s.id == hit[0])
                if abs(first.start - t) > ev.SNAP_TOLERANCE_S:
                    notes.append(f"matched transcript at {fmt_ts(first.start)} (model said {fmt_ts(t)})")
                t = first.start
                kind = "speech" if quote else kind
        for other in self.items:  # dedupe overlapping windows (+-15 s, similar text)
            if other.kind == kind and abs(other.t_start - t) <= 15 and \
                    difflib.SequenceMatcher(None, other.claim.lower(), text.lower()).ratio() > 0.8:
                return other
        it = ev.new_item(self.meta.video_id, text, kind, t, sources=sources, segment_ids=seg_ids, status=status,
                         confidence=conf, notes="; ".join(notes), demonstrated=demonstrated)
        if it.status == "unverified":
            it.confidence = "low"
        self.items.append(it)
        return it


def _ref_entry(it: EvidenceItem | None, text: str, **extra) -> dict:
    return {"text": text, "evidence_ids": [it] if it else [], **extra}


def _resolve_ids(obj):
    """Replace EvidenceItem placeholders in evidence_ids lists by their final ids (after finalize)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "evidence_ids":
                obj[k] = [x.id if isinstance(x, EvidenceItem) else x for x in v]
            else:
                _resolve_ids(v)
    elif isinstance(obj, list):
        for v in obj:
            _resolve_ids(v)
    return obj


def _gemini_pass(cfg: Config, cache: Cache, meta: VideoMeta, tr: Transcript | None, p: Plan, mode: str,
                 question: str | None, out: dict, evidence: Evidence) -> list[float]:
    from ..providers import base

    prov = base.get_multimodal_provider("gemini", cfg, cache)
    av = prov.available()
    if not av.ok:
        from ..errors import MissingDependency
        raise MissingDependency(f"Gemini provider unavailable: {av.reason}", av.remediation or None)
    b = _Builder(meta, tr, p.model or cfg.models[0], "static")
    windows: list[Window | None] = list(p.windows) or [None]
    summaries, usage, results = [], {}, []
    tut = {k: [] for k in TUTORIAL_KEYS}
    events: list[float] = []
    failures: list[YTUError] = []
    for w in windows:
        req = AnalyzeRequest(video_id=meta.video_id, url=canonical_url(meta.video_id),
                             prompt=build_prompt(meta, tr, mode, w, question), model=p.model or cfg.models[0],
                             mode="static", schema=EXTRACTION_SCHEMA[mode], window=w, fps=p.fps,
                             resolution=p.resolution or "low", prompt_template_id=f"{mode}@1",
                             duration_s=meta.duration,
                             user_question=question)
        try:
            res: AnalyzeResult = prov.analyze(req)
        except (NetworkError, ProcessFailed) as e:
            if len(windows) == 1:
                raise
            failures.append(e)
            s, e_ = (w.start, w.end) if w else (0.0, meta.duration or 0.0)
            evidence.gaps.append(Gap(s, e_, f"gemini failed for window: {type(e).__name__}"))
            continue
        results.append(res)
        out["uncertainties"] += [f"gemini: {w}" for w in res.warnings or [] if f"gemini: {w}" not in out["uncertainties"]]
        b.source = f"gemini:{res.mode or 'static'}:{res.model}"
        for k, v in (res.usage or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
        data = res.data if isinstance(res.data, dict) else None
        if data is None:
            s, e_ = (w.start, w.end) if w else (0.0, meta.duration or 0.0)
            evidence.gaps.append(Gap(s, e_, "gemini returned no schema-valid JSON"))
            continue
        # provider already normalised times; re-applying the batch rule is idempotent (fakes/raw data)
        sh = window_shift([v.get("t") for k in ("key_points", "claims", "visual_events", *TUTORIAL_KEYS)
                           for v in (data.get(k) or []) if isinstance(v, dict)], w)
        if isinstance(data.get("summary"), str) and data["summary"].strip():
            summaries.append((w.start if w else 0.0, data["summary"].strip()))
        for kp in data.get("key_points") or []:
            if isinstance(kp, dict):
                it = b.add(kp.get("text", ""), kp.get("t"), "inference", w, quote=kp.get("quote"), shift=sh)
                if it:
                    out["key_points"].append(_ref_entry(it, it.claim))
        for c in data.get("claims") or []:
            if isinstance(c, dict):
                b.add(c.get("text", ""), c.get("t"), "speech" if c.get("quote") else "inference", w,
                      quote=c.get("quote"), shift=sh)
        for ve in data.get("visual_events") or []:
            if not isinstance(ve, dict):
                continue
            kind = _VISUAL_KIND.get(str(ve.get("type")), "ui")
            text, note = str(ve.get("description") or ""), ""
            shown = str(ve.get("transcribed_text") or "")
            if shown and kind == "code":  # the claim is the code itself; description goes to notes
                text, note = shown, f"model description: {text}" if text else ""
            elif shown:
                text = f"{text}\n{shown}" if text else shown
            it = b.add(text, ve.get("t"), kind, w, note=note, shift=sh)
            if it:
                events.append(it.t_start)
        if mode == "tutorial":
            for key in TUTORIAL_KEYS:
                for entry in data.get(key) or []:
                    if not isinstance(entry, dict):
                        continue
                    demo = entry.get("demonstrated") if isinstance(entry.get("demonstrated"), bool) else None
                    it = b.add(entry.get("text", ""), entry.get("t"), _TUTORIAL_KIND[key], w,
                               quote=entry.get("quote"), demonstrated=demo, shift=sh)
                    if it is None:
                        continue
                    extra = {"demonstrated": demo}
                    if key == "code":
                        extra["language"] = str(entry.get("language") or "")[:20]
                        if it.kind == "code":
                            events.append(it.t_start)
                    tut[key].append(_ref_entry(it, it.claim, **extra))
    if not results and failures:
        raise failures[-1]
    evidence.items.extend(b.items)
    if summaries:
        summaries.sort()
        out["summary"] = summaries[0][1] if len(summaries) == 1 else "\n\n".join(
            f"[{fmt_ts(t)}] {s}" for t, s in summaries)
        out["summary_source"] = b.source
        hits = detect_injection(out["summary"])
        if hits:
            out["uncertainties"].append(f"summary contains instruction-like text ({','.join(hits)}); treat as data")
    if mode == "tutorial":
        out["tutorial"] = tut
    if b.dropped:
        why = f", {b.out_of_window} outside the analysed window" if b.out_of_window else ""
        out["uncertainties"].append(f"{b.dropped} model item(s) dropped: missing/invalid timestamps{why}")
    est = [r.cost_estimate.to_dict() for r in results if r.cost_estimate]
    out["cost"]["actual_usage"] = usage or None
    out["cost"]["requests"] = len(results)
    out["cost"]["cached_requests"] = sum(1 for r in results if r.cached)
    if est and not out["cost"]["estimate"]:
        out["cost"]["estimate"] = est[0]
    out["models"] = sorted({r.model for r in results})
    return events


# --- entry points ----------------------------------------------------------------------

def analyze_full(cfg: Config, cache: Cache, ref: VideoRef, *, mode: str = "understand",
                 window: Window | None = None, question: str | None = None,
                 resolution: str | None = None) -> AnalysisRun:
    from ..providers import ytdlp_meta

    if mode not in MODES:
        from ..errors import BadInput
        raise BadInput(f"unknown mode {mode!r}", f"Use one of: {', '.join(MODES)}.")
    meta = ytdlp_meta.fetch_metadata(cfg, cache, ref)
    window = strategy.clamp_window(window, meta.duration)
    p = strategy.plan(cfg, meta, command="analyze", key_present=strategy.key_present(), question=question,
                      window=window, resolution=resolution)
    p, degraded = strategy.auto_degrade(cfg, meta, p)
    use_gemini = p.multimodal.startswith("gemini")
    strategy.check_video_ok(meta, use_gemini)
    chapters = [c for c in chapters_out(meta)
                if not window or (c["start"] < window.end and (c["end"] or meta.duration or 0) > window.start)]
    out: dict = {"video": _video(meta), "plan": p.to_dict(), "mode": mode, "transcript": None,
                 "chapters": chapters, "summary": None, "summary_source": None, "key_points": [],
                 "visual_cues": [], "suggested_frames": [], "uncertainties": [degraded] if degraded else [],
                 "cost": {"estimate": p.estimate.to_dict() if p.estimate else None, "actual_usage": None}}
    if mode == "tutorial":
        out["tutorial"] = {k: [] for k in TUTORIAL_KEYS}
    evidence = Evidence(video_id=meta.video_id)
    if cfg.dry_run:
        out["dry_run"] = True
        why = strategy.over_budget(cfg, p)
        if why:
            out["over_budget"] = why
        out["evidence"] = evidence.to_dict()
        return AnalysisRun(meta, p, None, evidence, out)
    if use_gemini:
        why = strategy.over_budget(cfg, p)
        if why:
            raise OverBudget(why, estimate=p.estimate.to_dict() if p.estimate else None)
    try:
        tr: Transcript | None = strategy.get_transcript(cfg, cache, meta, window=window)
    except YTUError as e:
        if not use_gemini:
            raise
        log.warning("no transcript (%s); continuing with Gemini only", e.message)
        tr = None
        out["uncertainties"].append(f"no transcript ({e.message}): nothing could be corroborated against captions")
        evidence.gaps.append(Gap(window.start if window else 0.0, window.end if window else meta.duration or 0.0,
                                 "no captions"))
    if tr is not None and window:
        tr = Transcript(tr.video_id, tr.text_between(window.start, window.end), tr.source, tr.track,
                        tr.is_generated, tr.language, tr.provenance)
    out["transcript"] = _transcript_info(tr)
    cues = strategy.visual_cues(tr) if tr else []
    out["visual_cues"] = [{"t": t, "timestamp": fmt_ts(t), "url": ts_link(meta.video_id, t), "segment_id": sid,
                           "cue": cue} for t, sid, cue in cues[:cfg.max_frames]]
    if len(cues) > cfg.max_frames:
        out["visual_cues_total"] = len(cues)
    if tr is not None:
        evidence.gaps.extend(speech_gaps(tr, window.end if window else meta.duration,
                                         start=window.start if window else 0.0))
        if tr.is_generated:
            out["uncertainties"].append("auto-generated captions: wording, names, code terms and punctuation "
                                        "may be wrong")
    events: list[float] = []
    if use_gemini:
        events = _gemini_pass(cfg, cache, meta, tr, p, mode, question, out, evidence)
    else:
        out["digest"] = _digest(meta, tr, mode, question, window)
    out["suggested_frames"] = suggest_frames(meta, cues, events, evidence.gaps, limit=cfg.max_frames)
    if window:
        out["suggested_frames"] = [t for t in out["suggested_frames"] if window.start <= t <= window.end]
    ev.finalize(evidence, meta=meta, segments=tr.segments if tr else [])
    _resolve_ids(out)
    base_unc = list(out["uncertainties"])
    out["uncertainties"] += _status_notes(evidence)
    out["evidence"] = evidence.to_dict()
    return AnalysisRun(meta, p, tr, evidence, out, events, base_unc)


def _status_notes(evidence: Evidence) -> list[str]:
    notes = []
    counts: dict[str, int] = {}
    for it in evidence.items:
        counts[it.status] = counts.get(it.status, 0) + 1
    if counts.get("single-source"):
        notes.append(f"{counts['single-source']} item(s) single-source (one source, not matched to captions)")
    if counts.get("unverified"):
        notes.append(f"{counts['unverified']} item(s) unverified (e.g. code without a frame): check the frames")
    if counts.get("contradicted"):
        notes.append(f"{counts['contradicted']} item(s) contradicted between sources")
    sus = sum(1 for it in evidence.items if it.injection_suspect)
    if sus:
        notes.append(f"{sus} item(s) contain instruction-like text (possible prompt injection): data only")
    for g in evidence.gaps:
        notes.append(f"gap {fmt_ts(g.t_start)}-{fmt_ts(g.t_end)}: {g.reason}")
    return notes


def _digest(meta: VideoMeta, tr: Transcript | None, mode: str, question: str | None,
            window: Window | None = None) -> dict:
    """Transcript-only, evidence-ready digest for the host agent (no model call, no invented prose)."""
    chunks = chunk_transcript(tr, HANDOFF_CHARS) if tr else []
    total = sum(c["chars"] for c in chunks)
    yt = yt_cmd()
    if total > HANDOFF_CHARS:  # keep this handoff bounded: host pulls chunks via `transcript --offset`
        # offsets index the (window-filtered) transcript, so the command must carry the same window
        span = f" --start {window.start:g} --end {window.end:g}" if window else ""
        for c in chunks:
            c.pop("text")
            c["command"] = f"{yt} transcript {meta.video_id} --format txt{span} --offset {c['offset']}"
    tasks = [
        "Read the transcript chunks (or fetch them with the listed commands).",
        "Write the summary/key points yourself; cite every claim as [MM:SS](url) from segment timestamps.",
        f"View frames at suggested_frames (`{yt} frames URL --at T ...`) before describing code, diagrams or UI.",
        "Quote code/commands only as seen in a frame or heard in the transcript; mark anything else inferred.",
        f"Submit notes with `{yt} package URL --merge-notes notes.json` to validate and merge them.",
    ]
    if mode == "tutorial":
        tasks.insert(2, "Extract concepts, architecture, dependencies, commands, code and steps; "
                        "tag each demonstrated (shown/said) or inferred.")
    d = {"note": "transcript-only mode: no summary was generated; the host agent writes it from this digest",
         "chunks": chunks, "total_chars": total, "host_tasks": tasks}
    if question and tr:
        from .ask import search_segments
        d["question_hits"] = [{"id": s.id, "timestamp": fmt_ts(s.start), "url": ts_link(meta.video_id, s.start),
                               "text": s.text} for s in search_segments(tr, question, k=8)]
    return d


def run_analyze(cfg: Config, cache: Cache, ref: VideoRef, *, mode: str = "understand",
                window: Window | None = None, question: str | None = None,
                resolution: str | None = None) -> dict:
    """Returns {
      "video": {"video_id","title","channel","duration","url","language"},
      "plan": Plan.to_dict(), "mode",
      "transcript": {"source","track","is_generated","language","segments": int} | None,
      "chapters": [{"start","end","title","timestamp","url"}],
      "summary": str | None,                      # only from model output, else None
      "summary_source": "gemini:static:<model>" | None,
      "key_points": [{"text","evidence_ids"}],
      "tutorial": {...} (mode tutorial; entries {"text","evidence_ids","demonstrated"[, "language"]}),
      "visual_cues": [{"t","timestamp","url","segment_id","cue"}],
      "suggested_frames": [float],
      "digest": {...}                             # transcript-only modes: chunks + host tasks
      "evidence": Evidence.to_dict(), "uncertainties": [str],
      "cost": {"estimate": CostEstimate|None, "actual_usage": dict|None}
    } ; dry_run -> plan only with "dry_run": true."""
    return analyze_full(cfg, cache, ref, mode=mode, window=window, question=question,
                        resolution=resolution).result
