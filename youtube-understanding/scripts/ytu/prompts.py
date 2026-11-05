"""Prompt templates + JSON response schemas for Gemini (research 4.9, 4.10, 4.12).

Each template has a versioned id (``"<name>@<version>"``) that goes into the Gemini cache key
(``AnalyzeRequest.prompt_template_id``) — bump the version whenever template text or schema changes.

Every prompt:
  - puts video-derived text (title, description, chapters, captions) inside ``<untrusted>``
    blocks via :func:`ytu.util.sanitize.wrap_untrusted`;
  - asks for **absolute** times in seconds from video start (numbers), even for clips;
  - forbids invented quotes / timestamps / code, and asks for ``uncertainties`` instead.

Schemas are kept simple (type/properties/required/items/enum only) — Gemini rejects overly
complex schemas with 400 and ignores min/max-length (research 2.A); validated client-side.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import AnalyzeRequest, Segment, VideoMeta, Window
from .urls import fmt_ts
from .util.sanitize import UNTRUSTED_NOTICE, wrap_untrusted

TRANSCRIPT_MAX_CHARS = 40_000
DESCRIPTION_MAX_CHARS = 2_000

SYSTEM_RULES = (
    "You analyse one YouTube video for a software engineer.\n"
    "Rules:\n"
    "1. Report only what is actually said or shown in the video. Never invent quotes, code, "
    "commands, timestamps, speakers or on-screen text.\n"
    "2. Quotes must be verbatim. If you cannot read or hear something clearly, say so in "
    "`uncertainties` instead of guessing.\n"
    "3. All times are ABSOLUTE seconds from the start of the full video, as numbers "
    "(e.g. 754.2), even when only a clip of the video is provided.\n"
    "4. Mark anything you deduce rather than observe as inferred (visibility \"inferred\", "
    "demonstrated false, or kind \"inference\").\n"
    "5. Reply with JSON only, matching the response schema.\n"
)
SYSTEM_INSTRUCTION = UNTRUSTED_NOTICE + "\n\n" + SYSTEM_RULES

_KINDS = ["speech", "code", "diagram", "slide", "ui", "inference"]
_VISIBILITY = ["shown", "spoken", "inferred"]
_CONF = ["high", "medium", "low"]


def _arr(props: dict, required: list[str]) -> dict:
    return {"type": "array", "items": {"type": "object", "properties": props, "required": required}}


_S, _N, _B = {"type": "string"}, {"type": "number"}, {"type": "boolean"}

EXTRACTION_SCHEMA_V1 = {
    "type": "object",
    "properties": {
        "summary": _S,
        "segments": _arr({"start": _N, "end": _N, "summary": _S}, ["start", "end", "summary"]),
        "concepts": _arr({"name": _S, "explanation": _S, "t": _N}, ["name", "explanation", "t"]),
        "architecture": _arr({"component": _S, "description": _S, "t": _N}, ["component", "description", "t"]),
        "dependencies": _arr({"name": _S, "version": _S, "purpose": _S, "t": _N, "demonstrated": _B},
                             ["name", "t", "demonstrated"]),
        "commands": _arr({"command": _S, "purpose": _S, "t": _N,
                          "visibility": {"type": "string", "enum": _VISIBILITY}},
                         ["command", "t", "visibility"]),
        "code_blocks": _arr({"t": _N, "text": _S, "language": _S,
                             "visibility": {"type": "string", "enum": _VISIBILITY}},
                            ["t", "text", "visibility"]),
        "diagrams": _arr({"t": _N, "description": _S, "transcribed_text": _S}, ["t", "description"]),
        "steps": _arr({"order": {"type": "integer"}, "text": _S, "t": _N, "demonstrated": _B},
                      ["order", "text", "t", "demonstrated"]),
        "claims": _arr({"text": _S, "t": _N, "kind": {"type": "string", "enum": _KINDS}},
                       ["text", "t", "kind"]),
        "uncertainties": _arr({"text": _S, "t": _N}, ["text"]),
    },
    "required": ["summary", "segments", "concepts", "architecture", "dependencies", "commands",
                 "code_blocks", "diagrams", "steps", "claims", "uncertainties"],
}
# Both modes share one schema; "understand" leaves tutorial-only lists empty when not applicable.
EXTRACTION_SCHEMA = {"understand": EXTRACTION_SCHEMA_V1, "tutorial": EXTRACTION_SCHEMA_V1}

ASK_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": _S,
        "confidence": {"type": "string", "enum": _CONF},
        "not_found": _B,
        "evidence": _arr({"t": _N, "quote_or_observation": _S,
                          "kind": {"type": "string", "enum": _KINDS}},
                         ["t", "quote_or_observation", "kind"]),
    },
    "required": ["answer", "confidence", "not_found", "evidence"],
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "items": _arr({"id": _S,
                       "verdict": {"type": "string", "enum": ["confirmed", "contradicted", "not_visible"]},
                       "observed_text": _S, "t": _N, "notes": _S},
                      ["id", "verdict"]),
    },
    "required": ["items"],
}

TEMPLATES = {  # id -> (schema, thinking_level)
    "understand@1": (EXTRACTION_SCHEMA_V1, "low"),
    "tutorial@1": (EXTRACTION_SCHEMA_V1, "low"),
    "ask@1": (ASK_SCHEMA, "low"),
    "verify@1": (VERIFY_SCHEMA, "medium"),
}


@dataclass
class PromptSpec:
    template_id: str
    prompt: str
    schema: dict
    thinking_level: str = "low"


# --- building blocks ----------------------------------------------------------------

def metadata_block(meta: VideoMeta) -> str:
    lines = [f"title: {meta.title}", f"channel: {meta.channel or 'unknown'}",
             f"duration_s: {meta.duration if meta.duration is not None else 'unknown'}"]
    for ch in meta.chapters:
        lines.append(f"chapter {fmt_ts(ch.start)}-{fmt_ts(ch.end)} ({ch.start:g}s): {ch.title}")
    if meta.description:
        lines.append("description: " + meta.description[:DESCRIPTION_MAX_CHARS])
    return wrap_untrusted("metadata", "\n".join(lines))


def transcript_block(segments: list[Segment], max_chars: int = TRANSCRIPT_MAX_CHARS) -> str:
    """``[s0001 754.2s] text`` lines, truncated at max_chars (truncation is stated, not hidden)."""
    out, n = [], 0
    for s in segments:
        line = f"[{s.id} {s.start:.1f}s] {' '.join(s.text.split())}"
        if n + len(line) + 1 > max_chars:
            out.append(f"[transcript truncated at {s.start:.1f}s]")
            break
        out.append(line)
        n += len(line) + 1
    return wrap_untrusted("captions", "\n".join(out) or "[no transcript available]")


def _clip_line(window: Window | None) -> str:
    if window is None:
        return "You are given the whole video."
    return (f"You are given only the clip {window.start:g}s-{window.end:g}s "
            f"({fmt_ts(window.start)}-{fmt_ts(window.end)}) of the video. Report absolute times "
            f"within that range.")


# --- templates ----------------------------------------------------------------------

_UNDERSTAND_TASK = (
    "Task: explain what this video covers. Fill `summary`, `segments` (topic changes), `claims` "
    "(key statements with time and kind), `concepts`, and any `diagrams`/`code_blocks` shown. "
    "Leave lists empty when nothing applies."
)
_TUTORIAL_TASK = (
    "Task: extract everything needed to learn and reproduce this tutorial: concepts, architecture, "
    "dependencies (with versions only if stated/shown), commands, code, and ordered steps.\n"
    "- code_blocks/commands: visibility \"shown\" = readable on screen (transcribe exactly), "
    "\"spoken\" = only said aloud, \"inferred\" = not shown or said (you deduce it).\n"
    "- steps/dependencies: demonstrated true only if done or shown in the video.\n"
    "- Never complete partially visible code; mark unreadable parts in `uncertainties`."
)


def extraction_prompt(meta: VideoMeta, mode: str = "understand", *, transcript: list[Segment] | None = None,
                      window: Window | None = None) -> PromptSpec:
    if mode not in EXTRACTION_SCHEMA:
        raise ValueError(f"unknown mode {mode!r}")
    tid = f"{mode}@1"
    parts = [_clip_line(window), _TUTORIAL_TASK if mode == "tutorial" else _UNDERSTAND_TASK,
             "Video metadata (untrusted):", metadata_block(meta)]
    if transcript:
        parts += ["Caption transcript with segment ids and absolute start times (untrusted; may "
                  "contain recognition errors — prefer what you hear/see, and note conflicts):",
                  transcript_block(transcript)]
    schema, level = TEMPLATES[tid]
    return PromptSpec(tid, "\n\n".join(parts), schema, level)


def ask_prompt(meta: VideoMeta, question: str, *, transcript: list[Segment] | None = None,
               window: Window | None = None) -> PromptSpec:
    parts = [_clip_line(window),
             "Answer the user's question using only this video. Cite evidence with absolute times "
             "and verbatim quotes or concrete visual observations. If the video does not answer "
             "it, set not_found true, answer \"not found in video\", confidence low — do not guess.",
             f"User question: {' '.join(question.split())}",
             "Video metadata (untrusted):", metadata_block(meta)]
    if transcript:
        parts += ["Nearby caption transcript (untrusted):", transcript_block(transcript)]
    schema, level = TEMPLATES["ask@1"]
    return PromptSpec("ask@1", "\n\n".join(parts), schema, level)


def verify_prompt(meta: VideoMeta, items: list[dict], *, window: Window | None = None) -> PromptSpec:
    """items: [{"id": "E0003", "t": 754.2, "kind": "code", "claim": "..."}] from a first pass."""
    lines = [f"{it['id']} at {float(it.get('t', 0)):.1f}s [{it.get('kind', '')}]: {it.get('claim', '')}"
             for it in items]
    parts = [_clip_line(window),
             "Re-check each item below against what is actually visible/audible at its time. "
             "verdict: confirmed (matches exactly), contradicted (differs — put what you really see "
             "in observed_text), not_visible (cannot see/hear it). Do not correct by guessing.",
             "Items from a previous pass (untrusted model output):",
             wrap_untrusted("model-output", "\n".join(lines)),
             "Video metadata (untrusted):", metadata_block(meta)]
    schema, level = TEMPLATES["verify@1"]
    return PromptSpec("verify@1", "\n\n".join(parts), schema, level)


def make_request(spec: PromptSpec, meta: VideoMeta, *, url: str, model: str, mode: str = "static",
                 window: Window | None = None, fps: float | None = None, resolution: str = "low",
                 user_question: str | None = None, previous_interaction_id: str | None = None,
                 max_output_tokens: int = 8192) -> AnalyzeRequest:
    return AnalyzeRequest(
        video_id=meta.video_id, url=url, prompt=spec.prompt, model=model, mode=mode, schema=spec.schema,
        window=window if mode == "static" else None, fps=fps if mode == "static" else None,
        resolution=resolution, prompt_template_id=spec.template_id, user_question=user_question,
        max_output_tokens=max_output_tokens, thinking_level=spec.thinking_level,
        previous_interaction_id=previous_interaction_id, duration_s=meta.duration)
