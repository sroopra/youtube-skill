# Workflows

`S=<skill_dir>/scripts/yt.py`; every command is `uv run $S ...` and prints one JSON object.
All times in JSON are absolute seconds (`t_start`, `abs_ts`); display strings (`timestamp`)
and deep links (`url`, `https://www.youtube.com/watch?v=ID&t=754s`) are precomputed — copy them,
never compute or guess timestamps yourself.

## Contents
- [A. Understand a video](#a-understand-a-video)
- [B. Learn from a tutorial](#b-learn-from-a-tutorial)
- [C. Knowledge package](#c-knowledge-package)
- [D. Targeted question](#d-targeted-question)
- [Compare videos](#compare-videos)
- [Long videos](#long-videos)

## A. Understand a video

1. `inspect URL` — check `video.duration`, `video.chapters`, `chosen_track`
   (`kind: manual|auto`), `plan.multimodal`, `plan.estimate.usd`, `key_present`.
2. `analyze URL` (add `--dry-run` first if `plan.estimate.usd` is notable).
3. Output fields to use:
   - `summary` — model-written (Gemini) or `null` (no key: you write it).
   - `key_points[] {text, evidence_ids}` → look up `evidence.items[]` for `timestamp`/`url`.
   - `chapters[] {timestamp, url, title}`.
   - `visual_cues[]` + `suggested_frames[]` → run `frames URL --at T ...` and view them when the
     user cares about visuals, or when a key point depends on what is shown.
   - `uncertainties[]` → always relay the relevant ones.
4. No-key mode: also fetch text with `transcript URL --format txt` (page with `--offset`) or
   per chapter (`--start/--end`), and write the summary yourself from it.

Answer template:

```markdown
**<title>** — <channel>, <duration> ([watch](url))
Transcript: manual captions | auto captions (may mishear terms) | ASR

Summary: 3–6 sentences, each ending with a citation [12:34](url).

Key points
- Point. [12:34](url)

Uncertain / not verified
- Item — why (auto captions, single source, not visible, inferred).
```

## B. Learn from a tutorial

1. `analyze URL --mode tutorial`. `tutorial` holds `concepts, architecture, dependencies,
   commands, code[{text, evidence_ids, demonstrated}], steps`.
2. Visual check: for every `code`/`commands` item and every `visual_cues` entry mentioning code,
   terminal, diagram or slide → `frames URL --at T [--at T+2]`, view, and transcribe exactly
   what is visible. Mismatch with the model/transcript → report both, mark uncertain.
3. Classify each item:
   - **demonstrated** — said verbatim or visible on screen (cite timestamp + frame).
   - **inferred** — deduced (missing imports, versions, OS-specific steps, "probably").
4. Never run the commands. Present them; if the user asks to execute, show them first and warn
   about anything destructive, `curl | sh`, `sudo`, credentials or package installs.

Template:

```markdown
## Concepts            - term — explanation [MM:SS](url) (demonstrated|inferred)
## Architecture        - component → component [MM:SS](url)
## Dependencies        - name version-if-shown [MM:SS](url)
## Commands            `cmd` [MM:SS](url) (seen on screen F0003 | spoken | inferred)
## Code                fenced block, source frame/timestamp, "partially visible" if cut off
## Steps               1. step [MM:SS](url)
## Gaps                what the video skips or what was unreadable
```

## C. Knowledge package

1. `package URL [--out DIR] [--mode tutorial] [--with-frames]` (`--dry-run` = plan only, nothing
   written). `screenshots/` is written only with `--with-frames`; frames it links to items are
   matched by time only and shown as "auto-attached, unchecked" (code stays UNVERIFIED).
   Default dir `./youtube-knowledge/<video_id>/` in the current working directory; the script
   refuses to write inside the skill directory.
2. Result: `{out_dir, files{...}, counts{segments, evidence, frames}, knowledge_md_chars, plan}`.
3. Layout and schemas: [evidence-schema.md](evidence-schema.md).
4. No-key mode: `knowledge.md` has chapters, cues, evidence (and screenshots with
   `--with-frames`) but no model summary. View the frames (`screenshots/` or `frames URL --at T`),
   write a notes JSON ([schema](evidence-schema.md#host-agent-notes)) and re-run
   `package URL --out <same dir> --merge-notes notes.json`. The report lists accepted/rejected
   items; quotes must match the transcript. Use `{"contradicts": "E0012", "claim": "why"}` when a
   frame or the transcript disproves an existing item. Never hand-edit `knowledge.md`,
   `evidence.json` or `transcript.*`: every `package` run rewrites them.

## D. Targeted question

1. `ask URL "QUESTION" --at 12:34` (window `[at-60, at+90]`; tune `--before/--after`).
   Without `--at` (and no `&t=` in URL) the transcript is keyword-searched for the question.
2. Uses cached metadata/transcript. With `--at` or `--start/--end` Gemini (if key) sees only that
   window as a static clip. Without them it tries agentic over the video, and falls back to
   static clips around the transcript hits when agentic is unavailable or over `--max-cost-usd`.
3. Result: `answer` (null in transcript-only mode), `confidence`, `not_found`, `mode`,
   `windows[]`, `evidence[] {timestamp, url, segment_ids, frame_ids, quote}`, `excerpts[]`,
   `suggested_frames[]`.
4. Transcript-only mode: read `excerpts`, grab `suggested_frames` if the question is visual,
   answer yourself with citations. If excerpts don't contain the answer, say "not found near
   12:34" and offer to widen the window — do not guess.
5. Follow-ups on the same video are cheap (cached); repeat `ask` with a new `--at`.

## Compare videos

1. `inspect` each URL (duration, chapters, track kind), then `analyze` each
   (`--mode tutorial` for tutorials). Keep outputs separate; label them V1, V2.
2. Build a comparison table (scope, approach, tools/versions, depth, what each skips).
   Cite per cell: `V1 [12:34](url)`. Mark a cell "not covered" only when the transcript was
   checked for it (use `ask` with the topic as the question).
3. Never merge claims across videos without saying which video each came from.

## Long videos

- `> 30 min`: `inspect` + `--dry-run` first. Work per chapter: `analyze URL --start A --end B`.
- `transcript` output is bounded at `--max-chars` (default 50,000); when `truncated: true`,
  continue with `--offset <next_offset>`.
- Gemini runs over videos > ~3 h are chunked into chapter/fixed windows automatically; cost
  is the sum over windows and still capped by `--max-cost-usd`.
