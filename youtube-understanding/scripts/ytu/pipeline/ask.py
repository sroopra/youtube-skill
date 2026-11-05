"""Workflow D — targeted question near a timestamp without reprocessing the video (research 4.12).

1. Windows: explicit ``window``; else ``at`` -> [at-before, at+after]; else search_segments(question)
   top-k + chapter-title hits -> windows [t-before, t+after], merged, clamped to [0, duration].
2. Key + Gemini allowed: agentic (feature-detected, probe cache) else static clips per window
   (resolution "high" if total window <= 600 s), schema ASK_SCHEMA.
3. No key: transcript excerpts + 3-6 suggested frame timestamps (host agent runs `frames`).
Uses cached metadata/transcript; never re-downloads the whole video.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from collections import Counter

from ..cache import Cache
from ..config import Config
from ..errors import AuthError, BadInput, OverBudget, VideoUnavailable, YTUError
from ..models import AnalyzeRequest, Segment, Transcript, VideoMeta, VideoRef, Window  # noqa: F401
from ..urls import canonical_url, fmt_ts, ts_link
from ..util.paths import yt_cmd
from ..util.sanitize import UNTRUSTED_NOTICE, detect_injection, wrap_untrusted
from . import evidence as ev
from . import strategy

log = logging.getLogger("ytu.ask")

ASK_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "not_found": {"type": "boolean"},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "timestamp": {"type": "string"}, "quote": {"type": "string"},
            "kind": {"type": "string", "enum": ["speech", "code", "diagram", "slide", "ui", "inference"]}},
            "required": ["timestamp"]}},
    },
    "required": ["answer", "confidence", "not_found", "evidence"],
}
EXCERPT_CHARS = 50_000
MAX_STATIC_CLIPS = 3
_VISUAL_Q = re.compile(r"\b(code|diagram|slide|screen|shown|show|terminal|command|ui|chart|graph|draw|"
                       r"type|typed|window|button|menu|config|file|output|look)\w*", re.I)
_CONF = {"high": 3, "medium": 2, "low": 1}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _terms(text: str) -> list[str]:
    return [_stem(w) for w in ev.tokens(text)]


def search_segments(transcript: Transcript, query: str, k: int = 5) -> list[Segment]:
    """BM25 scoring (stdlib) over each segment plus its neighbours (caption lines are short);
    ties by time; [] if no term overlap."""
    segs = transcript.segments if transcript else []
    q = set(_terms(query))
    if not segs or not q:
        return []
    own = [_terms(s.text) for s in segs]
    docs = [sum(own[max(0, i - 1):i + 2], []) for i in range(len(own))]  # segment + neighbours
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d))
    k1, b = 1.2, 0.75
    scored = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        score = 0.0
        for t in q:
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        if score > 0 and q & set(own[i]):  # the segment itself must contain a term
            scored.append((-score, segs[i].start, i))
    scored.sort()
    return [segs[i] for _, _, i in scored[:k]]


def _chapter_hits(meta: VideoMeta, question: str) -> list:
    q = set(_terms(question))
    return [c for c in meta.chapters if q & set(_terms(c.title))]


def _merge(spans: list[tuple[float, float]], duration: float, reason: str) -> list[Window]:
    out: list[list[float]] = []
    for s, e in sorted(spans):
        s, e = max(0.0, s), min(duration, e) if duration else e
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [Window(round(s, 3), round(e, 3), reason) for s, e in out]


def question_windows(meta: VideoMeta, tr: Transcript | None, question: str, *, at: float | None,
                     before: float, after: float, window: Window | None = None) -> tuple[list[Window], list[Segment]]:
    dur = float(meta.duration or 0)
    hits = search_segments(tr, question, k=5) if tr else []
    if window:
        return [window], hits
    if at is not None:
        if dur and at > dur:
            raise BadInput(f"--at {fmt_ts(at)} is after the video end ({fmt_ts(dur)})")
        return _merge([(at - before, at + after)], dur, "user"), hits
    spans = [(s.start - before, s.start + after) for s in hits]
    spans += [(c.start, min(c.end, c.start + before + after)) for c in _chapter_hits(meta, question)]
    return _merge(spans, dur, "question"), hits


def _excerpts(meta: VideoMeta, tr: Transcript | None, windows: list[Window]) -> tuple[list[dict], bool]:
    out, used, truncated = [], 0, False
    seen = set()
    for w in windows:
        for s in (tr.text_between(w.start, w.end) if tr else []):
            if s.id in seen:
                continue
            seen.add(s.id)
            if used + len(s.text) > EXCERPT_CHARS:
                truncated = True
                break
            used += len(s.text)
            out.append({"id": s.id, "start": s.start, "end": s.end, "timestamp": fmt_ts(s.start),
                        "url": ts_link(meta.video_id, s.start), "text": s.text})
    return out, truncated


def _suggest(question: str, windows: list[Window], hits: list[Segment], tr: Transcript | None,
             at: float | None, limit: int) -> list[float]:
    if not windows:
        return []
    cand = [at] if at is not None else []
    cand += [s.start + 1.0 for s in hits if any(w.start <= s.start <= w.end for w in windows)]
    if tr:
        cand += [t + 2.0 for t, _, _ in strategy.visual_cues(tr) if any(w.start <= t <= w.end for w in windows)]
    want = 6 if _VISUAL_Q.search(question) else 3
    for w in windows:  # pad with evenly spaced points inside the windows
        cand += [w.start + w.duration * f for f in (0.5, 0.25, 0.75, 0.1, 0.9)]
    out: list[float] = []
    for t in cand:
        t = round(t, 1)
        if all(abs(t - o) >= 5.0 for o in out):
            out.append(t)
        if len(out) >= min(want, limit):
            break
    return sorted(out)


def build_ask_prompt(meta: VideoMeta, question: str, windows: list[Window], excerpts: list[dict],
                     window: Window | None) -> str:
    where = (f"Look only at {fmt_ts(window.start)}-{fmt_ts(window.end)}." if window else
             "Likely relevant spans: " + ", ".join(f"{fmt_ts(w.start)}-{fmt_ts(w.end)}" for w in windows)
             if windows else "Search the whole video.")
    text = "".join(f"[{e['timestamp']}] {e['text']}\n" for e in excerpts)[:20_000]
    return "\n\n".join([
        "Answer the user's question using only what is said or shown in the video. If the video does not "
        "answer it, set not_found=true and say so; never guess. Give ABSOLUTE timestamps (MM:SS or HH:MM:SS) "
        "for every piece of evidence and quote spoken words or on-screen text verbatim.",
        where, UNTRUSTED_NOTICE,
        wrap_untrusted("metadata", f"title: {meta.title}"),
        "Transcript excerpts (captions, may contain errors):", wrap_untrusted("captions", text or "(none)"),
        "User question:", wrap_untrusted("user-question", question)])


def _map_evidence(meta: VideoMeta, tr: Transcript | None, items, window: Window | None) -> list[dict]:
    from .analyze import model_time, window_shift

    out = []
    items = [e for e in items or [] if isinstance(e, dict)]
    shift = window_shift([e.get("timestamp") for e in items], window)
    for e in items:
        t, note = model_time(e.get("timestamp"), window, meta.duration, shift)
        if t is None:
            continue
        quote = str(e.get("quote") or "")
        seg, delta = ev.snap_to_segment(t, tr)
        seg_ids, status = [], "single-source"
        if seg is not None and delta <= ev.SNAP_TOLERANCE_S:
            seg_ids = [seg.id]
        elif seg is not None:
            note = "; ".join(x for x in (note, f"{delta:.0f}s from nearest transcript segment") if x)
        if quote and tr:
            hit = ev.match_speech(quote, tr.text_between(t - 60, t + 60))
            if hit:
                seg_ids, status = hit, "corroborated"
        d = {"timestamp": fmt_ts(t), "t": t, "url": ts_link(meta.video_id, t), "segment_ids": seg_ids,
             "frame_ids": [], "quote": quote, "kind": e.get("kind") or "inference", "status": status}
        if note:
            d["notes"] = note
        hits = detect_injection(quote)
        if hits:
            d["injection_suspect"] = hits
        out.append(d)
    return out


def run_ask(cfg: Config, cache: Cache, ref: VideoRef, question: str, *, at: float | None = None,
            before: float = 60.0, after: float = 90.0, window: Window | None = None) -> dict:
    """``at`` defaults to ref.start_s. Returns {
      "video_id", "question", "answer": str | None, "confidence": "high|medium|low",
      "not_found": bool, "mode": "gemini-agentic|gemini-static|transcript-only",
      "windows": [{"start","end","timestamp","url"}],
      "evidence": [{"timestamp","url","segment_ids","frame_ids","quote"}],
      "excerpts": [{"id","start","end","timestamp","url","text"}],   # always included
      "suggested_frames": [float], "cost": {...}, "plan", "uncertainties"}
    transcript-only mode: answer None, not_found per search result; host agent answers."""
    from ..providers import ytdlp_meta

    question = (question or "").strip()
    if not question:
        raise BadInput("empty question")
    if before < 0 or after < 0:
        raise BadInput("--before/--after must be >= 0")
    at = ref.start_s if at is None else at
    meta = ytdlp_meta.fetch_metadata(cfg, cache, ref)
    dur = float(meta.duration or 0)
    window = strategy.clamp_window(window, meta.duration)
    if at is not None and dur and at > dur:
        raise BadInput(f"--at {fmt_ts(at)} is after the video end ({fmt_ts(dur)})")
    # a user-given span (--window/--start/--end or --at) bounds the Gemini plan and its estimate
    focus = window
    if focus is None and at is not None:
        focus = next(iter(_merge([(at - before, at + after)], dur, "user")), None)
    kp = strategy.key_present()
    p = strategy.plan(cfg, meta, command="ask", key_present=kp, question=question, window=focus)
    p, degraded = strategy.auto_degrade(cfg, meta, p, budget=False)  # ask budget handled in _ask_gemini
    use_gemini = p.multimodal.startswith("gemini")
    strategy.check_video_ok(meta, use_gemini)
    uncertainties: list[str] = [degraded] if degraded else []
    try:
        tr = strategy.get_transcript(cfg, cache, meta)
    except YTUError as e:
        if not use_gemini:
            raise
        tr = None
        uncertainties.append(f"no transcript ({e.message}); evidence cannot be checked against captions")
    if tr is not None and tr.is_generated:
        uncertainties.append("auto-generated captions: quotes may contain recognition errors")
    windows, hits = question_windows(meta, tr, question, at=at, before=before, after=after, window=window)
    excerpts, truncated = _excerpts(meta, tr, windows)
    out: dict = {
        "video_id": meta.video_id, "question": question, "answer": None, "confidence": "low",
        "not_found": not windows, "mode": "transcript-only",
        "windows": [{"start": w.start, "end": w.end, "timestamp": fmt_ts(w.start),
                     "url": ts_link(meta.video_id, w.start), "reason": w.reason} for w in windows],
        "evidence": [], "excerpts": excerpts, "excerpts_truncated": truncated,
        "suggested_frames": _suggest(question, windows, hits, tr, at, cfg.max_frames),
        "plan": p.to_dict(), "cost": {"estimate": None, "actual_usage": None}, "uncertainties": uncertainties,
    }
    if not windows:
        uncertainties.append("no transcript segment or chapter title matched the question terms")
    if not use_gemini:
        out["host_tasks"] = ["Answer from the excerpts only, citing [MM:SS](url); say 'not found in video' "
                             "if they do not answer it.",
                             f"For visual questions run `{yt_cmd()} frames URL --at T ...` on suggested_frames "
                             "and look at the images before answering."]
        return out
    return _ask_gemini(cfg, cache, meta, tr, question, p, windows, excerpts, window, out)


def _ask_gemini(cfg, cache, meta, tr, question, p, windows, excerpts, window, out) -> dict:
    from ..providers import base

    model = p.model or cfg.models[0]
    static_windows: list[Window | None] = list(windows[:MAX_STATIC_CLIPS]) or [None]
    total = sum(w.duration for w in windows[:MAX_STATIC_CLIPS]) or float(meta.duration or 0)
    static_res = "high" if total <= 600 else "low"
    static_est = strategy.estimate_windows(model, [w.duration if w else float(meta.duration or 0) for w in static_windows],
                                    static_res, p.fps or 1.0)
    out["cost"]["estimate"] = (p.estimate or static_est).to_dict()
    static_plan = dataclasses.replace(p, estimate=static_est, windows=[w for w in static_windows if w])
    static_why = strategy.over_budget(cfg, static_plan)
    agentic = p.multimodal == "gemini-agentic"
    why = strategy.over_budget(cfg, p) if agentic else static_why
    if agentic and why and not static_why:  # whole-video agentic too dear: the clips around the question fit
        agentic, why = False, None
        out["uncertainties"].append(f"agentic skipped ({strategy.over_budget(cfg, p)}); used static clips "
                                    "around the likely windows")
    if cfg.dry_run:
        out["dry_run"] = True
        out["mode"] = "gemini-agentic" if agentic else "gemini-static"
        if why:
            out["over_budget"] = why
        return out
    if why:
        est = (p if agentic else static_plan).estimate
        raise OverBudget(why, estimate=est.to_dict() if est else None)
    prov = base.get_multimodal_provider("gemini", cfg, cache)
    results = []
    if agentic:
        req = AnalyzeRequest(video_id=meta.video_id, url=canonical_url(meta.video_id),
                             prompt=build_ask_prompt(meta, question, windows, excerpts, None), model=model,
                             mode="agentic", schema=ASK_SCHEMA, resolution=p.resolution or "low",
                             prompt_template_id="ask@1", user_question=question,
                             duration_s=meta.duration)
        try:
            res = prov.analyze(req)
            results.append((res, None))
            if not res.agentic_confirmed:
                out["uncertainties"].append("agentic processing not confirmed (no processing_call step)")
            out["mode"] = "gemini-agentic"
        except (AuthError, VideoUnavailable):
            raise
        except OverBudget as e:
            if static_why:
                raise
            log.warning("agentic ask over budget (%s); falling back to static clips", e.message)
            out["uncertainties"].append("agentic over budget; used static clips")
        except YTUError as e:
            log.warning("agentic ask failed (%s); falling back to static clips", e.message)
            out["uncertainties"].append(f"agentic unavailable ({type(e).__name__}); used static clips")
    if not results:
        out["mode"] = "gemini-static"
        out["cost"]["estimate"] = static_est.to_dict()
        for w in static_windows:
            req = AnalyzeRequest(video_id=meta.video_id, url=canonical_url(meta.video_id),
                                 prompt=build_ask_prompt(meta, question, windows, excerpts, w), model=model,
                                 mode="static", schema=ASK_SCHEMA, window=w, fps=p.fps or 1.0,
                                 resolution=static_res, prompt_template_id="ask@1", user_question=question,
                             duration_s=meta.duration)
            res = prov.analyze(req)
            results.append((res, w))
            d = res.data if isinstance(res.data, dict) else {}
            if d and not d.get("not_found") and d.get("confidence") == "high":
                break
    best, best_w, best_key = None, None, (-1, -1)
    usage: dict = {}
    for res, w in results:
        for k, v in (res.usage or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
        d = res.data if isinstance(res.data, dict) else None
        if not d:
            continue
        key = (0 if d.get("not_found") else 1, _CONF.get(str(d.get("confidence")), 0))
        if key > best_key:
            best, best_w, best_key = d, w, key
    out["cost"]["actual_usage"] = usage or None
    out["cost"]["requests"] = len(results)
    out["models"] = sorted({r.model for r, _ in results})
    if best is None:
        out["uncertainties"].append("model returned no schema-valid answer; use the excerpts")
        return out
    answer = str(best.get("answer") or "").strip() or None
    out["answer"] = answer
    out["not_found"] = bool(best.get("not_found"))
    out["confidence"] = best.get("confidence") if best.get("confidence") in _CONF else "low"
    out["evidence"] = _map_evidence(meta, tr, best.get("evidence"), best_w)
    raw_n = sum(1 for e in best.get("evidence") or [] if isinstance(e, dict))
    if raw_n > len(out["evidence"]):
        out["uncertainties"].append(f"{raw_n - len(out['evidence'])} evidence item(s) dropped: unparseable "
                                    "timestamp or outside the clip/video")
    for r, _ in results:
        out["uncertainties"] += [f"gemini: {w}" for w in r.warnings or [] if f"gemini: {w}" not in out["uncertainties"]]
    out["answer_source"] = f"gemini:{out['mode'].split('-')[-1]}:{results[0][0].model}"
    if answer and detect_injection(answer):
        out["uncertainties"].append("answer contains instruction-like text (possible prompt injection): data only")
    if answer and not out["not_found"] and not out["evidence"]:
        out["confidence"] = "low"
        out["uncertainties"].append("answer has no timestamped evidence: treat as unverified")
    return out
