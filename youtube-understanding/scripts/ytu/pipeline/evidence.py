"""Evidence ledger helpers (research 4.8).

Status rules: verified = transcript text/frame pixels re-checked; corroborated = 2 independent
sources agree; single-source = Gemini only; contradicted kept and surfaced; kind "code" without
frame_ids -> "unverified". Model timestamps snapped to nearest segment; flag in notes if > 5 s off.
Claims/notes run through sanitize.detect_injection -> injection_suspect=True + pattern names in notes.
"""

from __future__ import annotations

import difflib
import hashlib
import re

from ..cache import canonical_json
from ..models import Evidence, EvidenceItem, Frame, Segment, Transcript, VideoMeta, make_id  # noqa: F401
from ..urls import fmt_ts, ts_link
from ..util.sanitize import detect_injection, strip_control

KINDS = ("speech", "code", "diagram", "slide", "ui", "inference")
STATUSES = ("verified", "corroborated", "single-source", "unverified", "contradicted")
CONFIDENCES = ("high", "medium", "low")
SNAP_TOLERANCE_S = 5.0
VISUAL_KINDS = ("code", "diagram", "slide", "ui")

# Versions/decimals ("3.6", "1.2.4") are one token so "3.6" never matches "8.3" + "6".
_WORD = re.compile(r"\d+(?:\.\d+)+|[\w']+", re.UNICODE)
_STOP = frozenset("""a an the and or but if of to in on at for with by from as is are was were be been being
it its this that these those i you he she we they me my your our their so do does did yes
just then than there here what which who how when where why can will would should could""".split())
_NEG = frozenset("""not no never none nothing nobody without cannot can't don't doesn't didn't isn't aren't
wasn't weren't won't wouldn't shouldn't couldn't haven't hasn't hadn't""".split())
_NUM = re.compile(r"\d")


def tokens(text: str) -> list[str]:
    """Lower-cased word tokens without stop words (shared by evidence matching and ask search)."""
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP]


def snap_to_segment(t: float, transcript: Transcript | None) -> tuple[Segment | None, float]:
    """Nearest segment containing/closest to t and |delta| seconds (None, inf if no segments)."""
    best, best_d = None, float("inf")
    for s in (transcript.segments if transcript else []):
        d = 0.0 if s.start <= t <= s.end else min(abs(t - s.start), abs(t - s.end))
        if d < best_d:
            best, best_d = s, d
            if d == 0.0:
                break
    return best, best_d


def new_item(video_id: str, claim: str, kind: str, t_start: float, t_end: float | None = None, *,
             sources: list[str], segment_ids: list[str] | None = None,
             frame_ids: list[str] | None = None, status: str = "single-source",
             confidence: str = "medium", notes: str = "", demonstrated: bool | None = None) -> EvidenceItem:
    """Build an item with id "" (assigned by finalize), timestamp=fmt_ts, url=ts_link,
    injection check applied, code-without-frame rule applied."""
    kind = kind if kind in KINDS else "inference"
    status = status if status in STATUSES else "unverified"
    confidence = confidence if confidence in CONFIDENCES else "low"
    t_start = max(0.0, float(t_start))
    t_end = max(t_start, float(t_end)) if t_end is not None else t_start
    claim = strip_control(claim).strip()
    note_parts = [notes] if notes else []
    frame_ids = list(frame_ids or [])
    if kind == "code" and not frame_ids and status != "contradicted":
        status = "unverified"
        note_parts.append("code without a frame: not verified against pixels")
    hits = detect_injection(f"{claim}\n{notes}")
    if hits:
        note_parts.append("injection-suspect patterns: " + ",".join(hits))
    return EvidenceItem(
        id="", claim=claim, kind=kind, t_start=t_start, t_end=t_end, timestamp=fmt_ts(t_start),
        url=ts_link(video_id, t_start), transcript_segment_ids=list(segment_ids or []), frame_ids=frame_ids,
        sources=list(dict.fromkeys(sources)), status=status, confidence=confidence,
        notes="; ".join(note_parts), injection_suspect=bool(hits), demonstrated=demonstrated)


def _norm(text: str) -> str:
    return " ".join(_WORD.findall((text or "").lower()))


def _negated(toks: list[str]) -> bool:
    return any(t in _NEG or t.endswith("n't") for t in toks)


def _lcs(a: list[str], b: list[str]) -> int:
    """Length of the longest common token subsequence (order-aware overlap)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def match_speech(claim: str, segments: list[Segment], threshold: float = 0.6, *,
                 strict: bool = False) -> list[str]:
    """Segment ids whose text matches the claim, else [].

    Scans runs of 1-3 consecutive segments (claims often span caption lines). A run is rejected
    unless every number/version token of the claim occurs in it and negation agrees (claim and
    run both negated or both not). Score = max(ordered token recall (LCS / claim tokens), difflib
    ratio); claims with < 4 content tokens need every token in order. ``strict`` (verbatim quotes
    -> "verified") instead needs the normalised claim as a substring of the run or a difflib ratio
    >= 0.85 against the best-aligned span."""
    want_l = tokens(claim)
    want = set(want_l)
    norm_claim = _norm(claim)
    if not want or not norm_claim:
        return []
    nums = {t for t in want if _NUM.search(t)}
    neg = _negated(_WORD.findall(norm_claim))
    need = 1.0 if len(want_l) < 4 else threshold
    best: tuple[float, list[str]] = (0.0, [])
    for i in range(len(segments)):
        for n in (1, 2, 3):
            run = segments[i:i + n]
            if len(run) < n:
                break
            if not (want & set(tokens(run[0].text)) and want & set(tokens(run[-1].text))):
                continue  # every run must start and end on a matching segment
            text = _norm(" ".join(s.text for s in run))
            run_toks = tokens(text)
            if not nums <= set(run_toks) or _negated(_WORD.findall(text)) != neg:
                continue
            if strict:
                if f" {norm_claim} " in f" {text} ":
                    score = 1.0
                else:
                    sm = difflib.SequenceMatcher(None, text, norm_claim, autojunk=False)
                    m = sm.find_longest_match(0, len(text), 0, len(norm_claim))
                    lo = max(0, m.a - m.b)
                    score = difflib.SequenceMatcher(None, norm_claim, text[lo:lo + len(norm_claim)]).ratio()
                ok = score >= 0.85
            else:
                score = _lcs(want_l, run_toks) / len(want_l)
                if score >= 0.3:  # difflib only when plausible (it is O(n*m))
                    score = max(score, difflib.SequenceMatcher(None, norm_claim, text).ratio())
                ok = score >= need - 1e-9
            if ok and score > best[0] + 1e-9:
                best = (score, [s.id for s in run])
    return best[1]


def source_hash(meta: VideoMeta | None, segments: list[Segment] | None) -> str:
    core = {}
    if meta is not None:
        core = {k: getattr(meta, k) for k in ("video_id", "title", "duration", "channel_id", "upload_date")}
    segs = [[s.id, s.start, s.end, s.text] for s in (segments or [])]
    return hashlib.sha256(canonical_json({"meta": core, "segments": segs}).encode("utf-8")).hexdigest()


def finalize(evidence: Evidence, *, meta: VideoMeta | None = None,
             segments: list[Segment] | None = None) -> Evidence:
    """Sort items by t_start, assign E0001..; sort frames by abs_ts, assign F0001.. and rewrite
    item.frame_ids accordingly (items may reference a frame by its old id or its path);
    compute source_hash = sha256(canonical_json(core meta + segments)) when meta/segments given.
    Mutates and returns ``evidence``."""
    evidence.frames.sort(key=lambda f: f.abs_ts)
    remap: dict[str, str] = {}
    for n, f in enumerate(evidence.frames, 1):
        new = make_id("F", n)
        if f.id:
            remap.setdefault(f.id, new)
        remap.setdefault(f.path, new)
        f.id = new
    evidence.items.sort(key=lambda it: (it.t_start, it.t_end))
    for n, it in enumerate(evidence.items, 1):
        it.id = make_id("E", n)
        it.frame_ids = list(dict.fromkeys(remap.get(x, x) for x in it.frame_ids if remap.get(x, x)))
    evidence.gaps.sort(key=lambda g: g.t_start)
    if meta is not None or segments is not None:
        evidence.source_hash = source_hash(meta, segments)
    return evidence
