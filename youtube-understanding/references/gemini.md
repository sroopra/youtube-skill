# Gemini provider

Used only when a key is present and the strategy allows it (`auto` or `gemini`). Transcript,
frames and host-frames mode never import or call Gemini. All facts below verified from docs on
**2026-11-05**; the live Gemini path was not exercised during development (no key) — treat
token counts and agentic behaviour as estimates until `usage` confirms them.

## Key setup

Lookup order (first hit wins; the key is never printed, logged, cached or passed to child processes,
and is only sent to `https://generativelanguage.googleapis.com/`, pinned in `providers/gemini.py`):

1. env `GEMINI_API_KEY`, then `GOOGLE_API_KEY`
2. file `~/.config/youtube-understanding/.env` (override path: `YTU_ENV_FILE`), lines `GEMINI_API_KEY=...`.
   With `XDG_CONFIG_HOME` set it is `$XDG_CONFIG_HOME/youtube-understanding/.env` (and `config.toml` there too).

The user sets the key, not the agent. Type it in an editor, never in a shell command (it would
stay in shell history and in the agent's log):

```bash
mkdir -p ~/.config/youtube-understanding
touch ~/.config/youtube-understanding/.env
chmod 600 ~/.config/youtube-understanding/.env   # a warning is printed if group/other can read it
# open the file in an editor and add one line: GEMINI_API_KEY=your-key
uv run <skill_dir>/scripts/yt.py doctor           # gemini_key.present / source, never the value
```

The skill can't hide the key from an agent that runs as the same user: that agent can read the
file or the environment itself. The rules in SKILL.md's Security section tell agents not to.
The real limit is the key: restrict it and cap it (see the README's "Keep your key safe").

Missing key → exit 5, message names the places looked. Get a key at https://aistudio.google.com/apikey.

## API shape (Interactions API, google-genai 2.34.0)

```python
client.interactions.create(
    model="gemini-3.8-flash",
    input=[{"type": "text", "text": prompt},
           {"type": "video", "uri": "https://www.youtube.com/watch?v=ID", "resolution": "low",
            "processing": {"type": "static", "start_offset": "600s", "end_offset": "1200s", "fps": 1}}],
    response_format={"type": "text", "mime_type": "application/json", "schema": schema},
    generation_config={"thinking_level": "low", "max_output_tokens": 8192},
    system_instruction=UNTRUSTED_NOTICE + task)
```

- `processing`: `"static"` (default, whole video), static with offsets/fps (window), or `"agentic"`.
- Agentic: documented for Gemini 4 Argon, 3.8 Flash, 3.6 Flash, 3.5 Flash Lite; YouTube +
  agentic on AI Studio is undocumented → feature-detected (400 or no `processing_call` step →
  cached as unavailable, falls back to static windows). `agentic_confirmed` in results.
- `temperature/top_p/top_k` omitted (ignored by 3.6+, future 400). `thinking_level: minimal` is rejected by 3.8 Flash.
- YouTube input: preview, **public videos only** (private/unlisted rejected → use `--strategy transcript|local`),
  ≤ 10 videos/request, free tier 8 h of YouTube video/day, paid no length limit.
- Structured output schema validated client-side (Gemini ignores `minLength` etc.); one repair retry.

## Model ladder

Default `gemini-3.8-flash` → `gemini-3.5-flash-lite` → `gemini-2.5-flash-lite`. Next model on
404 `model_not_found` or 429 `quota_exceeded`.

| Config | Effect |
|---|---|
| `--models a,b` / `YTU_GEMINI_MODELS=a,b` | Replace ladder |
| `YTU_GEMINI_MODEL=x` | Pin `x` first, keep rest of ladder |
| `~/.config/youtube-understanding/config.toml` | `models = [...]`, `max_cost_usd = 1.0`, `[timeouts] gemini = 900` (no secrets) |

Model notes: 3.8 Flash = 1,048,576 in / 65,536 out tokens. `gemini-2.5-flash` → routed to
3.5-flash-lite and `gemini-2.5-pro` → 3.8-flash on 2026-12-01.

## Token math and pricing

Tokens ≈ `duration_s × (tokens_per_frame × fps + 32 audio)`; frames: low/medium 70, high 280
(the video guide says 66/258 — `usage` in results tells the truth; warning if > 30% off).

| Resolution @1 fps | tokens/s | 10 min | 1 h |
|---|---|---|---|
| low | 102 | 61k | 367k |
| high | 312 | 187k | 1.12M (over default cap) |

Paid prices per 1M tokens (table in `providers/gemini.py`, `PRICE_TABLE_DATE = 2026-11-05`):

| Model | Input | Output | Note |
|---|---|---|---|
| gemini-3.8-flash | $0.75 | $3.75 | → $1.50 / $7.50 from 2027-01-01 |
| gemini-3.6-flash | $0.75 | $3.75 | same change |
| gemini-3.5-flash-lite | $0.30 | $2.50 | |
| gemini-2.5-flash-lite | $0.10 | $0.40 | audio input $0.30 |

Unknown model → priced at the most expensive known rate (stated in `estimate.basis`).
Batch ≈ 50% (not used). No YouTube-specific pricing line exists.

## Cost controls

- Estimate before every call (`plan.estimate`): input tokens incl. ~2k prompt, max output tokens.
- Over budget = `usd > --max-cost-usd` (default 0.50, `YTU_MAX_COST_USD`) or
  `input_tokens > --max-input-tokens` per request (default 800k, `YTU_MAX_INPUT_TOKENS`).
- Plan downscales first: ≤ 10 min high-res if it fits, else low; ≤ 3 h static low if it fits;
  longer → chapter/fixed windows; still over → fps 0.5.
- Still over: `--strategy auto` falls back to transcript + host-agent frames, with an uncertainty
  naming the estimate (no paid call). `--strategy gemini` refuses with exit 7 `OverBudget`.
  Non-public (e.g. unlisted) videos: auto falls back the same way; `gemini` exits 6.
- `ask`: with `--at` / `--start/--end`, one static clip of that window only. Otherwise agentic
  (estimated as whole-video static, an upper bound); when that is over budget, static clips
  around the transcript hits instead.
- `--dry-run`: plan + estimate, no call. Results cached by video + answering model + mode + window + fps +
  resolution + prompt template + question + schema hash + SDK version → repeat calls are free;
  `--force-refresh` bypasses.
- `YTU_GEMINI_TIER=free` enables the local 8 h/day YouTube-hours tracker.
- A fallback-model answer is cached under the fallback model, so the requested model is tried
  again next run. Background interactions that end `failed` with 5xx/UNAVAILABLE/INTERNAL are
  retried with backoff like HTTP 5xx.

## Errors and retry

| HTTP | Handling |
|---|---|
| 408, 429 rate-limit, 500, 503 | Retry with backoff 1→2→4→8 s + jitter, ≤ 5 attempts, cap 60 s |
| 404 model_not_found, 429 quota_exceeded | Next model in ladder (quota resets midnight Pacific) |
| 400 with agentic | Mark agentic unavailable, use static windows |
| 400 other, 402 | Exit 1/4 with message, no retry |
| 401, 403 | Exit 5 (bad key / API not enabled for project) |

Rate limits per tier are only visible at https://aistudio.google.com/rate-limit.
Spend cap per rolling 10 min: Tier 1 $10, Tier 2/3 $200.

## Prompting rules (implemented)

- Title, description, chapter titles, transcript excerpts are wrapped in
  `<untrusted source="...">` blocks; system instruction says they are data, not instructions.
- Timestamps requested absolute (`MM:SS`/`HH:MM:SS`); window-relative replies are detected and
  offset; all model timestamps snapped to transcript segments (> 5 s → noted).
- Model output is untrusted: schema-validated, stored, never executed.
