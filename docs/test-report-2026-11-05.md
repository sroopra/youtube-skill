# youtube-understanding: integration & end-to-end test results

Date: 2026-11-05. Host: macOS darwin-arm64, Python 3.13.11, yt-dlp 2026.10.26 (in-process), ffmpeg (Homebrew), uv.
Gemini key: **present** (detected via `config.key_status()`, source = user `.env` file; value never printed).
All live commands were run as `uv run youtube-understanding/scripts/yt.py ...`
with `YTU_CACHE_DIR` set to a fresh `mktemp -d` scratch directory, so the first run of each command was a cold cache.
stdout and stderr were captured to files and the real exit code was recorded for every command (nothing piped through `tail`).

## 1. Unit / integration suite

| Run | Result |
|---|---|
| Before integration fixes | `uv run pytest`: 441 passed |
| After fixes (final) | `uv run pytest`: **445 passed**, 0 failed, ~1.8 s. No network, subprocess or paid API calls (all mocked) |

Requirement coverage (`tests/test_requirements.py` unless noted):

| Requirement | Test(s) |
|---|---|
| Manual captions | `test_manual_captions_end_to_end` |
| Auto captions (`-orig` preferred, flagged generated) | `test_auto_captions_prefers_orig_and_flags_generated` |
| No captions → ASR / Gemini / clear error | `test_no_captions_falls_back_to_asr`, `test_no_captions_with_key_ask_still_uses_gemini`, `test_no_captions_transcript_strategy_clear_error` |
| Code-on-screen tutorial | `test_code_on_screen_tutorial_real_gemini_provider` |
| Diagram / presentation, no key | `test_diagram_presentation_no_key_points_host_to_frames` |
| Long-form chunking | `test_long_form_chunking_by_chapters` |
| Unsupported / inaccessible URL | `test_unsupported_url_exit_2`, `test_inaccessible_video_typed_exit` (now includes the live wording "This video is unavailable") |
| Missing API key | `test_missing_key_gemini_strategy` |
| Provider errors & timeouts | `test_gemini_error_never_leaks_key`, `test_gemini_503_retried_then_succeeds`, `test_ytdlp_timeout_falls_through_to_next_provider`, `test_all_providers_fail_reports_network_error`, `test_metadata_timeout_exit_4` |
| Cache repeat | `test_second_run_hits_cache`, `test_gemini_result_cached` |
| Multilingual track selection | `test_multilingual_track_selection` (parametrized) |
| Injection text in transcript | `test_injection_in_transcript_is_wrapped_and_flagged`, `test_injection_in_transcript_no_key_ask` |
| New regressions from this run | `test_analyze.py::test_speech_gaps_windowed_ignores_time_before_window`, `test_package.py::test_entries_drops_verbatim_repeats`, `test_package.py::test_host_note_confirming_model_item_is_folded_not_duplicated` |

## 2. Live end-to-end runs (OBSERVED)

| # | Command (abridged) | Exit | Time | Observation |
|---|---|---|---|---|
| 1 | `doctor --network` | 0 | 0.6 s | ok=true. yt-dlp in-process 2026.10.26; ffmpeg/ffprobe/uv/deno/node found; key present (source only); youtube.com 200, googleapis 404 (reachable); ASR backend `parakeet` |
| 2 | `curl -sS -o /dev/null -w '%{http_code}' --max-time 15` youtube.com / googleapis | curl 0 / 0 | – | HTTP 200 / 404 |
| 3 | `inspect jNQXAC9IVRw` | 0 | 3.0 s | "Me at the zoo", 19 s, 3 chapters; chosen track `en` **manual** (manual `en` + `de` available) |
| 4 | `inspect zjkBMFhNj_g` | 0 | 3.8 s | 3588 s, 21 chapters, 157 tracks; chosen `en-orig` **auto** ("English (Original)"), not a translation |
| 5 | `inspect 9mmVa6O-hzQ` | 0 | 3.6 s | German, 759 s, 4 chapters; chosen `de-orig` auto |
| 6 | `inspect dD_KwXU9meA` | 0 | 3.7 s | Japanese, 409 s, 0 chapters; chosen `ja-orig` auto |
| 7 | `transcript youtu.be/jNQXAC9IVRw --format text` | 0 | 3.3 s | 6 segments, `is_generated=false` (text below) |
| 8 | `transcript zjkBMFhNj_g --format json` | 0 | 4.8 s | 1704 segments total; page of 80, `truncated=true`, `next_offset=80`; note "auto-generated captions..." |
| 9 | `transcript 9mmVa6O-hzQ --window 0:00-1:00` | 0 | 3.6 s | `de-orig`; "[00:00] in diesem video lernst du in den / nächsten zehn minuten ... programmiersprache python" |
| 10 | `transcript dD_KwXU9meA --window 0:00-1:00` | 0 | 3.5 s | `ja-orig`; "[00:05] プログラミング講座やっていきたいとおもいます". ASR misrecognition visible: channel name あきちょん rendered as 秋緒ん |
| 11 | repeat of #8 | 0 | **0.41 s** | cache hit (4.84 s → 0.41 s); output byte-identical |
| 12 | #8 with `--force-refresh` | 0 | 7.9 s | cache bypassed; same output |
| 13 | `ask zjkBMFhNj_g "...two parameters files..." --at 0:30 --strategy transcript` | 0 | 0.4 s | transcript-only: `answer=null`, window 00:00–02:00, verbatim excerpts with timestamped links (host answers from excerpts) |
| 14 | `ask zjkBMFhNj_g&t=1550s "What two variables predict..." --dry-run` | 0 | 0.4 s | plan `gemini-agentic`, window 24:50–27:20 (chapter "LLM Scaling Laws" 25:43), estimate $0.31 (static whole-video upper bound), no call |
| 15 | same, live | 0 | 11.4 s | answer "N (number of parameters) and D (amount of text)", confidence high. Evidence: speech quote at 25:56 **matches transcript** s0741–s0745 ("...predictable function of only two variables you need to know n the number of parameters in the network and D the amount of text..."); slide quote at 25:46. Actual usage 8633 input tokens (7105 tool-use) + 273 output ≈ $0.0075 |
| 16 | `frames rfscVS0vtbw --at 54:40,56:37,57:01` | 0 | 51.7 s | 3 JPEGs at 1280×720. Remote seek returned 403, so the code fell back to short section downloads (stderr warning). Frame at 57:01 viewed: PyCharm, `result = int(num1) + int(num2)`, `ValueError: invalid literal for int() with base 10: '4.3'` |
| 17 | `analyze rfscVS0vtbw --mode tutorial --window 52:39-58:27 --dry-run` | 0 | 3.7 s | gemini-static, high res, 348 s, est. 110,576 in / 8,192 out tokens = **$0.1137** |
| 18 | `analyze zjkBMFhNj_g --dry-run` | 0 | 0.1 s | whole-video static, low res, est. 367,976 tokens = **$0.3067**; reason "3588s <= 3h and ~365976 tok < 800000" |
| 19 | `analyze jNQXAC9IVRw --strategy gemini` (live) | 0 | 9.3 s | gemini-static gemini-3.8-flash. Summary mentions elephants and long trunks. 4 evidence items snapped to segments s0001–s0006. Usage 6187 in / 439 out (est. 7928 / 8192 output cap) |
| 20 | repeat of #19 | 0 | **0.35 s** | `cached_requests: 1`; otherwise identical |
| 21 | `package rfscVS0vtbw --mode tutorial --window 52:39-58:27 --with-frames --out examples/rfscVS0vtbw` | 0 | 140.5 s | 59 segments, 23 evidence items, 8 screenshots; Gemini static (cost est. $0.114). Time dominated by 8 section downloads after the remote-seek 403 |
| 22 | same + `--merge-notes host-notes.json` (before fix) | 0 | 0.5 s | 12/12 notes accepted, but frames were duplicated (16) and code blocks repeated. **Bug, fixed (see §4)** |
| 23 | same, after fix | 0 | 0.5 s | 12/12 accepted, 0 rejected; 32 items, **8 frames**; 3 code blocks now `verified` with frames; 9 verified / 11 corroborated / 12 single-source |
| 24 | `inspect watch?v=xxxxxxxxxxx` (before fix) | **1** | 1.4 s | ProcessFailed: "This video is unavailable". Wrong type. **Bug, fixed** |
| 25 | same, after fix | **6** | 1.0 s | `VideoUnavailable` (typed exit 6) |
| 26 | `inspect https://vimeo.com/76979871` | 2 | 0.1 s | BadInput "not a YouTube URL: host vimeo.com" + remediation |
| 27 | `inspect youtube.com/playlist?list=...` | 2 | 0.1 s | BadInput "playlist URL without a video id" |
| 28 | `analyze jNQXAC9IVRw --strategy gemini` with `YTU_ENV_FILE=<missing>` | 5 | 0.1 s | MissingKey; message lists only where it looked; no key material |
| 29 | `transcript jNQXAC9IVRw --format srt`, no key | 0 | 0.1 s | works without Gemini |
| 30 | `analyze jNQXAC9IVRw`, no key | 0 | 0.1 s | plan `local`; summary null; suggested frames [5.0, 18.5] for host |
| 31 | grep `AIza` across all captured stdout/stderr | – | – | 0 matches |

### Transcript ground truth check

`jNQXAC9IVRw` ("Me at the zoo", widely documented as the first YouTube upload, about elephants):

```
[00:01] All right, so here we are, in front of the elephants
[00:05] the cool thing about these guys is that they have really...
[00:07] really really long trunks
[00:12] and that's cool
[00:14] (baaaaaaaaaaahhh!!)
[00:16] and that's pretty much all there is to say
```

Matches the known content: "in front of the elephants" and "really, really long trunks". Gemini summary (#19) is consistent with this. It names the speaker "Jawed Karim"; that name is not in the transcript (channel = `jawed`), so treat it as model inference.

`zjkBMFhNj_g` scaling-laws answer (#15): checked against caption segments s0737–s0745 at 25:47–26:03; the quoted speech is verbatim.

`rfscVS0vtbw` example: host viewed all 8 screenshots. Code text in `knowledge.md` matches the frames: `result = num1 + num2`, `int(num1) + int(num2)`, `float(num1) + float(num2)`. ValueError text matches the frames at 56:57 and 57:26. Transcript says "five plus 8.3 is definitely not 58.3" (54:44).

## 3. Example knowledge package

`examples/rfscVS0vtbw/`: `metadata.json`, `transcript.srt`, `transcript.json`, `evidence.json`, `knowledge.md`, `screenshots/` (8 JPEGs), `host-notes.json`.

- Source: freeCodeCamp "Learn Python - Full Course for Beginners", window 52:39–58:27 (chapter "Building a Basic Calculator").
- The host notes were written after viewing every screenshot. They only claim what is in the frames or transcript.
- Inference is marked: the "9.5" output is labelled INFERRED because no host-viewed frame shows it.
- The host summary flags the model's "Python 3.6" dependency as unconfirmed, since the console path is truncated in the frames.

## 4. Failures found and fixed in this run

| Bug | Fix | Test |
|---|---|---|
| yt-dlp "This video is unavailable" (live wording) gave `ProcessFailed` exit 1 | `errors._YTDLP_PATTERNS` also matches `video is unavailable` → `VideoUnavailable` exit 6 | `test_inaccessible_video_typed_exit` new case |
| Windowed package reported a false gap "00:00–52:39 no speech" | `analyze.speech_gaps(..., start=window.start)` | `test_speech_gaps_windowed_ignores_time_before_window` |
| Steps repeated verbatim in "Actionable guidance" when the model emitted duplicates | `package._entries` drops identical lines | `test_entries_drops_verbatim_repeats` |
| Host notes pointing at `screenshots/<t>.jpg` duplicated frames; confirming notes duplicated items/code blocks | `merge_notes` reuses an existing Frame with the same sha256. A note with the same kind and claim within 5 s is folded into the existing item (sources/ids unioned, stronger status kept) | `test_host_note_confirming_model_item_is_folded_not_duplicated` |

## 5. Validation, install, discovery (OBSERVED)

| Step | Result |
|---|---|
| `uvx --from skills-ref==0.1.1 agentskills validate <abs skill dir>` | rc 0, "Valid skill" |
| Symlink `~/.claude/skills/youtube-understanding` | created → repo skill dir (target was absent) |
| Symlink `~/.agents/skills/youtube-understanding` | created → repo skill dir (target was absent) |
| `claude -p "List the names of available skills that relate to YouTube. Reply with names only." --max-turns 2` (timeout 180 s) | rc 0, 8 s, output: `youtube-understanding` |
| `.gitignore` | added (.env*, caches, `__pycache__`, `.venv`, `youtube-knowledge/`). No git add/commit run |

## 6. NOT tested live (untested / mocked only)

- Local ASR fallback on a real no-caption video in this run (parakeet was run live by the media agent on a 5–15 s window; mlx-whisper and faster-whisper were mocked only).
- yta (youtube-transcript-api) provider in this run: ytdlp succeeded first every time.
- Bot-check / 429 / members-only / age-restricted / cookies paths (mocked only).
- Gemini agentic → static fallback, model-ladder fallback on 404/429, repair call, free-tier tracker (mocked only).
- Long-video multi-window Gemini analyze (>3 h or >800k tokens) (plan logic unit-tested only).
- `frames --frame-strategy scene|keyframes` in this run (media agent ran them live earlier).
- Codex / Gemini CLI / Copilot discovery of the `~/.agents/skills` symlink.
- Linux.

## 7. Remaining gaps / observations

- **Remote seek denied (403)** on rfscVS0vtbw. Each frame then needs a section download (about 15–17 s per frame), so `package --with-frames` took 140 s for 8 frames.
- The `package` JSON output has no `actual_usage` / cost actuals; `analyze` and `ask` do.
- The agentic `ask` dry-run estimate ($0.31) is a whole-video static upper bound; the actual cost was about $0.0075.
- Model-written visual claims repeat on-screen text inside the claim (e.g. "...58.3. Enter a number: 5 ..."). Cosmetic.
- Gemini evidence `segment_ids` can cover only part of a multi-segment quote (s0743–45 cited; the quote spans s0741–45).
- Gemini placed a concept sentence under "Dependencies" in the rfsc package. Section placement is model judgement and is not validated.

## 8. Post-review fixes

Four review lenses (correctness, security, spec, agent-ux) raised 34 issues, 3 of them duplicates. Each was checked against the code before it was fixed. Nothing was committed.

### Results

| Check | Result |
|---|---|
| `uv run pytest` | **472 passed**, 0 failed, ~1.6 s (was 445). All network, subprocess and paid API calls are mocked |
| `uvx --from skills-ref==0.1.1 agentskills validate <abs skill dir>` | rc 0, "Valid skill" |
| `grep AIza` over all live stdout/stderr from this pass | 0 matches |

### Fixes (with regression tests)

| Issue | Fix | Test |
|---|---|---|
| `match_speech` used bag-of-tokens recall. "Python 3.6" was corroborated by "8.3" + "6", and "always use sudo" matched "never use sudo" | Versions and decimals are now one token, and every number must match exactly. Negation must agree on both sides. Matching uses ordered recall (LCS); claims under 4 tokens need every token. A host `quote` is `verified` only when it matches near-verbatim (substring, or ratio >= 0.85) | `test_evidence.py::test_match_speech_numbers_versions_negation`, `test_match_speech_strict_quotes` |
| `ask --at` on a long video raised OverBudget, because the estimate covered the whole video | An `--at` window or an explicit window now goes into `plan()` as a static clip of that span. When the agentic estimate is over budget but the static clips fit, the static clips run. OverBudget raised by the provider during agentic also falls back to clips | `test_ask.py::test_at_on_long_video_default_budget_uses_static`, `test_at_uses_static_clip_only`, `test_agentic_overbudget_from_provider_falls_back`, `test_gemini_dry_run_and_budget` |
| Agentic `ask` ignored `--start/--end` | An explicit window never uses agentic. It runs a static clip with the "Look only at" prompt | `test_explicit_window_never_agentic` |
| `model_time` shifted each out-of-window time by itself (500 became 1100) | The window-relative shift is now one decision per batch (`window_shift`). Times outside the analysed window are **dropped** and counted in `uncertainties`; they are never moved and never cited | `test_analyze.py::test_window_shift_is_batch_rule`, `test_out_of_window_model_times_dropped_not_cited` |
| Auto mode with a key exited 7 on videos longer than ~1h40m | `plan` tries fps 0.5 after low resolution. If still over budget, `strategy.auto_degrade` switches to transcript + host frames and adds an uncertainty that names the estimate and how to raise it. Explicit `--strategy gemini` still refuses | `test_auto_long_video_default_budget_degrades[2,4]`, `test_over_budget_refused_for_explicit_gemini` |
| Auto mode on an unlisted video exited 6 | Auto now degrades to transcript + host frames. `--strategy gemini` still exits 6 | `test_auto_unlisted_falls_back_to_transcript` |
| ASR never received a language, so parakeet ran on Japanese | `get_transcript` passes `lang or cfg.lang or meta.language` to ASR | `test_strategy.py::test_get_transcript_asr_gets_language`, updated `test_no_captions_falls_back_to_asr` |
| Windowed digest chunk commands had no window, so offsets pointed into the wrong span | Commands now include `--start S --end E`, plus the absolute `uv run /abs/scripts/yt.py` | `test_window_clamped_chapters_filtered_digest_commands` |
| Window end past the video end gave a false gap and an inflated estimate | `strategy.clamp_window` runs before `plan()` in analyze, ask and package | same test |
| Gemini cache was keyed by the requested model, so fallback answers stuck | Results are now cached under the model that answered | `test_model_ladder_fallback` (extended) |
| A `failed` background interaction with 5xx/UNAVAILABLE/INTERNAL was fatal | `_TransientInteraction` is classified as `retry` and gets backoff | `test_failed_interaction_transient_is_retried`, `test_failed_interaction_not_transient_is_fatal` |
| Fallback cache dir `$TMPDIR/youtube-understanding-<uid>` was trusted even when owned by another user or mode 0777 | Owner, mode and symlink are now checked with lstat, else a fresh `mkdtemp`. `<cache>/tmp` refuses symlinks and dirs owned by others | `test_config.py::test_cache_dir_fallback_rejects_foreign_or_open_dir`, `test_cache_tmp_root_refuses_symlink` |
| `redact()` missed `lsig/ip/ei/n/pot` and whole googlevideo URLs | Whole `*.googlevideo.com/...` URLs are replaced, and the param list is extended | `test_proc.py::test_redact_googlevideo_and_params` |
| A key loaded from the .env file was not scrubbed by the generic redaction | `config._key_lookup` registers it with `proc.add_secret()`, so every `redact()` call scrubs it | `test_file_key_registered_for_redaction` |
| `inspect`/`transcript` returned raw untrusted text with no marker | Output now has `untrusted: true`, `content_notice` and `injection_flags[{field, patterns}]`. `formats` is dropped from inspect tracks | `test_inspect_and_transcript_mark_untrusted` |
| `md_escape` left bare URLs that GFM auto-links | Bare URLs are wrapped in code spans with the text unchanged | `test_sanitize.py::test_md_escape` |
| `attach_frames` marked time-matched frames as supporting code and visual claims | Auto-attached frames are labelled "auto-attached, unchecked" and the status does not change, so code stays UNVERIFIED until a host note confirms it. A new note form, `{"contradicts": "E0012", "claim": why}`, marks an item `contradicted` (kept and surfaced) | `test_package_gemini_with_frames`, `test_merge_notes_contradicts` |
| `frames --at` past the end gave exit 1 plus a "network" remediation | Times are checked against cached metadata, giving BadInput exit 2 that states the duration | `test_frames.py::test_run_frames_past_end_is_bad_input` |
| Argparse errors printed nothing on stdout | `_JsonArgParser.error` writes `{ok:false,error:{type:BadInput,...}}` and exits 2 | `test_cli.py::test_argparse_error_is_json` |
| stdout JSON was indented (25-35% overhead), and `--max-chars` measured a different encoding | Output is compact by default, with `--pretty` to indent. The transcript page is sized the same way it is emitted | `test_output_compact_unless_pretty` |
| `99:99:99` / `1:75` accepted | Minute and second fields must be < 60 | `test_urls.py::test_colon_fields_out_of_range` |
| Windowed analyze returned all chapters and uncapped visual_cues | Only chapters that overlap the window are returned. Cues are capped at `--max-frames`, with `visual_cues_total` | `test_window_clamped_chapters_filtered_digest_commands` |
| Provider warnings (e.g. "27 time(s) outside window") were never surfaced | analyze and ask add them to `uncertainties` with the prefix `gemini:` | covered by live run below |
| Docs: no `--with-frames` / `--merge-notes`, hand-edit advice, "Gemini untested live", XDG path | SKILL.md (Workflow C row and Commands block), workflows.md C/D, evidence-schema.md (new "Host-agent notes" section and speech-matching rules), gemini.md (budget and degrade behaviour, XDG), README (Limitations, XDG) | `test_docs.py` (passes) |

### Skipped

| Issue | Reason |
|---|---|
| `util/retry.py` unused for yt-dlp | yt-dlp has its own retries, and the provider chain already falls through on failure. Adding a second retry layer would slow down bot-check failures. Left as is |
| transcript default format `json` → `txt` | Compact output removed most of the overhead (indent). Changing the default would break the documented JSON contract. `--format txt` is still recommended in SKILL.md |

### Live re-runs (OBSERVED, `set -euo pipefail`, mktemp dir under scratchpad, real exit codes)

| # | Command | rc | Time | Result |
|---|---|---|---|---|
| 32 | `frames jNQXAC9IVRw --at 1:00` | 2 | – | `BadInput "--at 01:00 is at/after the video end (00:19)"` (was exit 1 + network remediation) |
| 33 | `ask jNQXAC9IVRw q --at banana` / `--at 1:75` | 2 / 2 | – | stdout JSON `BadInput` with `--help` remediation; usage on stderr |
| 34 | `analyze rfscVS0vtbw --strategy transcript --start 1:00:00 --end 2:30:00` | 0 | – | chunk commands `uv run /…/scripts/yt.py transcript rfscVS0vtbw --format txt --start 3600 --end 9000 --offset 465`. Running that command returned `[01:46:01] when the if statement is true…`, which matches the chunk start 01:46:01. 12 overlapping chapters (was 35); cues 24 of 35 |
| 35 | `analyze rfscVS0vtbw --dry-run` (key present, default $0.50) | 0 | – | `host-frames`; uncertainty "Gemini skipped for cost (estimated $2.3204 > --max-cost-usd 0.5)…" (was exit 7) |
| 36 | `ask rfscVS0vtbw "what is the calculator" --at 53:20 --dry-run` | 0 | – | `gemini-static`, window 52:20–54:50, est. $0.0673, no over_budget (was OverBudget $1.27) |
| 37 | `ask rfscVS0vtbw … --start 52:30 --end 58:20 --dry-run` | 0 | – | `gemini-static`, est. $0.1141 (was agentic whole video) |
| 38 | `analyze jNQXAC9IVRw --strategy transcript --start 0:05 --end 10:00` | 0 | – | no gaps past 00:19 (window clamped) |
| 39 | `inspect rfscVS0vtbw` / `transcript rfscVS0vtbw` | 0 / 0 | – | `untrusted: true`, `injection_flags` (transcript: 1 flagged segment); inspect 12.6 KB (was 21 KB); transcript page 50.6 KB for `--max-chars 50000` (was 60 KB) |
| 40 | `ask "zjkBMFhNj_g&t=1550s" "What two variables…"` (live, paid) | 0 | 21 s | now `gemini-static` on 24:50–27:20 only. Answer "N (number of parameters) and D (amount of text)", confidence high, 2 evidence items corroborated against s0743–s0745. Usage 44,875 in / 223 out ≈ $0.035 (agentic in #15 cost ≈ $0.0075, but would be refused on longer videos) |
| 41 | `package rfscVS0vtbw --mode tutorial --window 52:39-58:27 --with-frames --merge-notes …` into a temp dir | 0 | 180 s | Gemini cache miss → **new paid static call** (104,081 in). The model returned every time in 1:27:19–1:35:44, outside the 52:39–58:27 clip. The provider warned "27 time(s) outside 3159-3507s". Before the drop fix these would have been cited at wrong timestamps. **Bug found and fixed**: out-of-window items are now dropped and counted, and provider warnings are surfaced |
| 42 | `analyze rfscVS0vtbw --mode tutorial --window 52:39-58:27 --force-refresh` (live, paid) | 0 | 39 s | fresh answer, all times inside the window, 24 items, no time warnings. 104,081 in / 2,199 out ≈ $0.086 |
| 43 | `package …` (2 passes: write screenshots, then `--merge-notes host-notes.json`) into a temp dir | 0 / 0 | 101 s / 0.4 s | 31 items, 8 frames; 9/12 notes accepted. 3 rejected because the notes name screenshots that this run's frame times did not produce. **"Python 3.6" (E0011) is now `single-source`, not corroborated.** Auto-attached frames render as "frames (auto-attached, unchecked)" |

Paid Gemini spend in this pass: about $0.12 (#40, #41, #42).

### Example package status (regenerated 2026-11-05)

`examples/rfscVS0vtbw/` was regenerated with the post-review code (old copy kept in the session scratchpad):

| Step | Result |
|---|---|
| `package rfscVS0vtbw --mode tutorial --window 52:39-58:27 --with-frames --out examples/rfscVS0vtbw` | rc 0; 59 segments, 24 evidence items, 8 frames. Gemini result served from the local cache (no new cache file written), $0 |
| Host review | all 8 screenshots viewed; one extra frame at 57:38 grabbed with `frames --at` because the 57:35 frame did not yet show the `9.5` output |
| `host-notes.json` rewritten | 12 notes with near-verbatim quotes and frame paths |
| `package … --merge-notes host-notes.json` | rc 0; 12/12 accepted, 0 rejected; final statuses 9 verified, 12 corroborated, 12 single-source; 9 frames |

Checks against the screenshots:
- Gemini's E0019 "4.3" input: consistent with speech at 56:49 ("we'll add 4.3 and 5.5"); the frame shows only the second input 5.5 because the first line scrolled away.
- "Python 3.6" (E0011): single-source inference. The interpreter path is cut off at `Python.framework/Versi`, recorded as host note E0016.
- Renderer bug found and fixed: inference items were labelled *demonstrated* when the model claimed so. `package._tag` now always labels `kind: inference` as *inferred* (regression test `test_inference_is_never_labelled_demonstrated`).

### Remaining gaps

- Gemini clip timestamps are not reliable: one of two identical static calls returned times about 35 minutes off. They are now dropped rather than cited, but such a run yields a transcript-only result.
- `ask` with `--at` now always uses a static clip. That is cheaper and bounded on long videos, but costs more than agentic on short ones (about $0.035 vs $0.0075 in #40 vs #15).
- Mock-only (unchanged): model-ladder fallback, agentic→static fallback, repair call, retry of a `failed` interaction, free-tier tracker.
