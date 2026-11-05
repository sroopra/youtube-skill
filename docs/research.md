# youtube-understanding — Research & Architecture

Verification date for every claim in this document: **2026-11-05**.
Environment used for live tests: macOS arm64, uv 0.9.17, Python 3.12/3.13, ffmpeg 9.0.2 (Homebrew), residential IP. **No `GEMINI_API_KEY` was available**, so no Gemini call was executed; every Gemini claim is doc- or SDK-source-based.

Status legend used throughout:

| Status | Meaning |
|---|---|
| **verified-live-test** | Command/code actually run on 2026-11-05 (or by a researcher earlier in this session) and output observed |
| **verified-doc** | Read in official docs / official repo source on 2026-11-05 |
| **community-report** | Third-party README, issue, blog, search snippet |
| **assumption** | Inference or prior knowledge; not confirmed today |

Where an original research claim and its later verification pass conflicted, the **verified** version is used here and the correction is noted.

---

## 1. Technologies and implementations evaluated

### 1.1 Multimodal video understanding APIs
| Technology | Role considered | Outcome |
|---|---|---|
| Google Gemini API — **Interactions API** (`client.interactions.create`, `POST /v1beta/interactions`) | Primary: native YouTube URL ingestion, static + agentic processing | **Selected (primary multimodal provider)** |
| Google Gemini API — `generateContent` (legacy, "remains fully supported") | Fallback request shape | Selected as secondary code path |
| Vertex AI / Gemini Enterprise Agent Platform | YouTube + agentic explicitly documented, needs GCP/ADC | Optional future provider, not v1 |
| Anthropic Claude API (vision) | Frames-as-images + transcript; no video input | Used implicitly via host agent viewing frames |
| OpenAI Responses API | Official images-and-vision guide lists only PNG/JPEG/WEBP/non-animated GIF (`input_image`); no `input_video` / video in official docs (re-checked 2026-11-05) | Frames + transcript only; not a provider in v1 |
| TwelveLabs (Pegasus 1.6, released 2026-10-06; 1.5 default since 2026-08-18) | Video-to-text; sources = asset upload (≤4 GB direct, ≤10 GB multipart), URL, HLS/base64 (≤2 h); sync `/analyze` ≤1 h; YouTube page URLs **not documented** | Rejected for v1 (needs media re-hosting) |
| YouTube Data API v3 `captions.download` | Official captions endpoint | Rejected: "requires the user to have permission to edit the video" (OAuth `youtube.force-ssl`), 200 quota units/call → useless for third-party videos |

### 1.2 Transcript extraction
| Technology | Version | Outcome |
|---|---|---|
| yt-dlp (+ `yt-dlp-ejs`, deno PyPI wheel, curl_cffi) | 2026.10.26 / ejs 0.8.0 / deno 2.9.8 / curl_cffi 0.17.0 | **Primary** (metadata + captions + media) |
| youtube-transcript-api | 1.2.4 | **Secondary** caption source |
| pytubefix | 11.2.1 | Rejected (works; heavier, duplicates yt-dlp) |

### 1.3 Local / cloud ASR fallback
| Technology | Version | Outcome |
|---|---|---|
| mlx-whisper | 0.4.3 (mlx 0.32.4) | **Selected** on darwin-arm64 |
| faster-whisper | 1.2.1 (CTranslate2 4.8.5) | **Selected** on Linux/Intel, with `av<19` pin |
| parakeet-mlx (NVIDIA `parakeet-tdt-0.6b-v3`) | 0.7.0 (2026-11-04, Apache-2.0; model CC-BY-4.0) | **Optional fast path** on darwin-arm64 for the 25 European languages only (no ja/zh); punctuated output, token timestamps + confidence. Live-tested |
| WhisperX | 3.8.6 (BSD-2-Clause, Py <3.14) | Not v1 (heavy: torch + pyannote); reference for word alignment + diarization |
| pyannote.audio `speaker-diarization-community-1` | 4.0.7 | Optional diarization only; gated HF model → needs HF token + accepting conditions; not v1 |
| openai-whisper | 20250625 | Last resort only |
| whisper.cpp (`whisper-cli`) | n/a (not installed) | Rejected (needs compile/system install) |
| Gemini audio (via Gemini provider) | — | Cloud ASR option when key present (assumption: works; not tested) |
| Groq STT (`whisper-large-v3-turbo`) | — | Optional cloud ASR with word timestamps |
| OpenAI STT (`gpt-transcribe`; older models deprecated) | — | Not recommended (no documented word timestamps) |
| Deepgram nova-3 | — | Optional, not v1 |

### 1.4 Visual processing
| Technology | Outcome |
|---|---|
| ffmpeg (input-seek single frames, `select='gt(scene,T)'`, `fps=`, `thumbnail`, `-skip_frame nokey`, `mpdecimate`) | **Selected** |
| yt-dlp `-g` (resolve stream URL) + remote ffmpeg seek | **Selected** for targeted screenshots |
| yt-dlp `--download-sections` | **Selected** for window analysis |
| PySceneDetect 0.7.2 | Not needed (ffmpeg equivalent); revisit `HashDetector` for screencasts |
| Tesseract OCR | Optional only (not installed; poor on code/dark themes — assumption, not benchmarked) |
| ocrmac 1.0.1 (Apple Vision framework, MIT) | **Optional** macOS-only OCR pre-pass for code frames (live-tested; see §2.C); RapidOCR 3.10.0 (Apache-2.0) = cross-platform candidate, untested |
| Host-agent vision (Claude Code / Codex / etc. viewing JPEGs) | **Selected** for OCR/diagram reading in no-key mode |

### 1.5 Existing skills / MCP servers / pipelines surveyed
anthropics/skills, anthropics/claude-plugins-official, intellectronica/agent-skills `youtube-transcript`, other skills.sh caption wrappers, kimtaeyoon83/mcp-server-youtube-transcript, jkawamoto/mcp-youtube-transcript, anaisbetts/mcp-youtube, kevinwatt/yt-dlp-mcp, ZubeidHendricks/youtube-mcp-server, kirbah/mcp-youtube, coyaSONG/youtube-mcp-server, tubeagentkit/youtube-mcp, steipete/summarize, byjlw/video-analyzer, haithamelmengad/popcorn, kwacky1/video-understanding-mcp, godovasik/gemini-video-analysis, DerYUYU/gemini-video-mcp, mrzhangguoguo/gemini-video-analysis-skill, ludmila-omlopes/agent-skills, samihalawa/video-analysis-mcp, adamanz/gemini-video-mcp-server, Jayne040622/video-to-knowledge-skill, shepnerd/vidify, Contendeo (hosted), official MCP registry.

**Verdict: build.** Nothing covers transcript + native multimodal + frames + evidence-linked knowledge package in a portable Python/uv skill.

### 1.6 Agent Skills packaging
agentskills.io specification + `skills-ref` validator; client discovery rules for Claude Code, Codex, GitHub Copilot (CLI / VS Code / cloud agent), Gemini CLI, Antigravity, Cursor, OpenCode; sandbox/network behaviour for Claude Code, Codex, Copilot cloud agent, Gemini CLI.

---

## 2. Evidence (verification date 2026-11-05)

### 2.A Gemini API

#### Verified (live test here)
| Claim | Evidence | Source |
|---|---|---|
| Old ai.google.dev doc URLs 301 → aistudio.google.com/docs/*; `?generate-content=true` variant still renders only Interactions examples | `curl -sI`: `/gemini-api/docs/video-understanding` → `aistudio.google.com/docs/video-understanding`; `/gemini-api/docs/generate-content/video-understanding` → `…?generate-content=true`; `/api/interactions-api` → `aistudio.google.com/api-reference/interactions-api` | https://ai.google.dev/gemini-api/docs/video-understanding , https://aistudio.google.com/docs/video-understanding?generate-content=true |
| google-genai latest = **2.34.0** (uploaded 2026-10-29T06:43:21), `requires_python >=3.11`; `Client(...).interactions` exists | PyPI JSON; `uv run --with google-genai==2.34.0` (Py 3.12) introspection | https://pypi.org/pypi/google-genai/json |
| SDK 2.34.0 `Interaction.model_fields` includes `output_text, output_image, output_audio, output_video, steps, status, id, usage, previous_interaction_id` | introspection | same |
| SDK `VideoContent` fields: `data, mime_type, name, processing, resolution, type, uri` | `google/genai/_gaos/types/interactions/videocontent.py` in 2.34.0 | installed package |
| SDK 2.34.0 method signatures: `client.models.generate_content(*, model: str, contents=…, config: GenerateContentConfig\|None)`; `client.files.upload(*, file: str\|PathLike\|IOBase, config: UploadFileConfig\|None)`; `client.interactions` (`GeminiNextGenInteractions`) methods `create, get, cancel, delete`; `interactions.create(*, timeout=…, automatic_continuation=…, extra_headers/query/body, **body)` | `inspect.signature`, `uv run --with google-genai==2.34.0` | installed package |
| SDK `CreateModelInteraction` body fields: `model, input, generation_config, response_format, response_mime_type, response_modalities, system_instruction, tools, safety_settings, previous_interaction_id, store, stream, background, cached_content, continuation_token, service_tier, webhook_config, environment, labels`. `service_tier` ∈ `flex \| standard \| priority \| deferred`. `response_format` type = `ResponseFormat \| list[ResponseFormat]`; `TextResponseFormat{type:"text", mime_type, schema}`; docstring: `minLength, maxLength, minProperties, maxProperties, uniqueItems, contains, minContains, maxContains` ignored; `minItems`/`maxItems` enforced; too-complex schemas → 400 INVALID_ARGUMENT. **No `step_list` field** | `google/genai/_gaos/types/interactions/createmodelinteraction.py`, `textresponseformat.py` | installed package |
| `types.HttpRetryOptions` fields exist: `attempts, initial_delay, max_delay, exp_base, jitter, http_status_codes` (resolves earlier assumption) | `HttpRetryOptions.model_fields` | installed package |
| generateContent types: `VideoMetadata{start_offset: str, end_offset: str, fps: float}` (fps default 1.0, valid range (0.0, 24.0]); `FileData{display_name, file_uri, mime_type}`; `Part.media_resolution: PartMediaResolution{level, num_tokens}` (level adds `ULTRA_HIGH`); config `MediaResolution` = UNSPECIFIED/LOW/MEDIUM/HIGH; **`Part.media_processing: MediaProcessing{MEDIA_PROCESSING_UNSPECIFIED, STATIC, AGENTIC}`** | grep of `types.py` on main | https://raw.githubusercontent.com/googleapis/python-genai/main/google/genai/types.py |

#### Verified (docs)
| Claim | Source |
|---|---|
| Interactions API needs google-genai / @google/genai **≥ 2.3.0**; GA "as of June 2026"; "recommended interface for all new projects"; generateContent "considered legacy" but "remains fully supported" | https://aistudio.google.com/docs/interactions |
| YouTube call: `client.interactions.create(model="gemini-3.8-flash", input=[{"type":"text","text":…},{"type":"video","uri":"https://www.youtube.com/watch?v=9hE5-98ZeCg"}])` → `interaction.output_text`. REST: `POST https://generativelanguage.googleapis.com/v1beta/interactions` | https://aistudio.google.com/docs/video-understanding |
| `output_text` is an SDK convenience — not in REST `Interaction` schema; REST clients read text from `steps`. Status enum: `in_progress, requires_action, completed, failed, cancelled, incomplete, requires_continuation` | https://aistudio.google.com/api-reference/interactions-api |
| VideoContent: `type:"video"`, `uri`, `data`, `mime_type` (video/mp4, mpeg, mpg, mov, avi, x-flv, webm, wmv, 3gpp), `name`, `resolution` (low/medium/high/ultra_high), `processing` = `"static"` \| `"agentic"` \| `{type:"static", start_offset, end_offset, fps}`. Offsets are strings `"10.5s"`, non-negative, end > start. Per-part key is **`resolution`** (guide prose says `media_resolution`) | https://aistudio.google.com/api-reference/interactions-api |
| Agentic (`"processing":"agentic"`) supported on **Gemini 4 Argon, 3.8 Flash, 3.6 Flash, 3.5 Flash Lite**. Static is default. Offsets + fps are static-only. Agentic adds `processing_call` / `processing_result` steps (`ProcessingCallStep`, `ProcessingResultStep`); "up to 88% fewer total tokens" on long-form; may cost more / higher TTFT on clips < 5 min or broad queries | https://aistudio.google.com/docs/video-understanding |
| Background polling pattern: `while interaction.status == "in_progress": time.sleep(5); interaction = client.interactions.get(interaction.id)` | same |
| AI Studio agentic example uses **only** `client.files.upload(file="path/to/lecture.mp4")` — YouTube+agentic not stated (re-read 2026-11-05: still neither stated nor excluded; YouTube section silent on agentic; no max length for agentic). Step shapes: `processing_call{type, id, signature}` ("model requested a video segment or audio transcript"), `processing_result{type, call_id, signature}`. Static mode: "Timestamps are added every second" | same |
| `gemini-3.8-flash` model page: input **1,048,576**, output **65,536** tokens; inputs Text/Image/Video/Audio/PDF; output Text; "Stable", latest update September 2026; thinking levels **low/medium/high — `minimal` not supported (returns error)**; supports Batch, Caching, Flex + Priority inference, Structured outputs, Function calling, URL context, Search grounding; no Live API. Agentic video not listed in its feature list (listed on video page instead) | https://aistudio.google.com/docs/models/gemini-3.8-flash |
| Audio guide: "Speaker diarization (identifying different speakers)" via **prompt + schema** (no dedicated param); max **9.5 h audio per prompt**; 32 tok/s (1 min = 1,920 tok); timestamps `MM:SS` via prompt. Its transcription example passes a **YouTube URL as `{"type":"video","uri":…,"mime_type":"video/mp4"}`** with `response_format=response_schema` (bare JSON schema, segments `{speaker, timestamp, content, language, emotion}`) — first doc example combining YouTube + structured output; bare-schema shape conflicts with SDK `TextResponseFormat` → use typed `{type:"text", mime_type:"application/json", schema}` | https://aistudio.google.com/docs/audio |
| Files API: ≤ **2 GB/file**, **20 GB/project**, auto-deleted after **48 h**, free; "Always use the Files API when" total request > **100 MB** (PDF 50 MB); user uploads not downloadable; REST resumable upload `${BASE_URL}/upload/v1beta/files` (`x-goog-upload-url` header). Page shows Interactions example with `{"type":"audio","uri":…,"mime_type":…}` only | https://aistudio.google.com/docs/files |
| Rate limits page lists **no RPM/TPM/RPD numbers** — "can be viewed in Google AI Studio" (https://aistudio.google.com/rate-limit). Batch enqueued tokens T1/T2/T3: 3.8 Flash 3M/400M/1B; 3.5 Flash-Lite & 2.5 Flash-Lite 10M/500M/1B; 4 Argon 5M/500M/1B. Spend cap per rolling 10 min: Free N/A, T1 $10, T2 $200, T3 $200. Tiers: T1 = linked billing; T2 = $100 paid + 3 days; T3 = $1,000 + 30 days. RPD resets midnight Pacific | https://aistudio.google.com/docs/rate-limits |
| Vertex lists agentic sources "YouTube URLs", "Cloud Storage URIs", "Inline Base64"; generateContent `"mediaProcessing": "AGENTIC"` (Python `media_processing="AGENTIC"`), v1beta1 only; "Only one YouTube video URL is supported per request" | https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/video-understanding |
| YouTube on Gemini API: preview, no charge; public only (private/unlisted rejected); free tier 8 h/day; paid no length limit; ≤ 10 videos/request on 2.5+ (1 earlier); "Pricing and rate limits may change" | https://aistudio.google.com/docs/video-understanding |
| Token math: default 1 FPS; **default resolution = low**: ~100 tok/s low, ~300 tok/s high (66 / 258 tok/frame + audio 32 tok/s) | https://aistudio.google.com/docs/video-understanding |
| Media-resolution page (Gemini 3/4 video): unspecified = low = medium = **70** tok/frame, high = **280**, ultra_high N/A for video | https://aistudio.google.com/docs/media-resolution |
| 1M-context models: "up to 3 hours long by default (at low media resolution), or up to 1 hour long at high" | https://aistudio.google.com/docs/video-understanding |
| Inline video limit inconsistent: table "<100MB, <1min" vs prose "under 20MB total request size"; Files page says use Files API above 100 MB → still conservative: inline only < 20 MB, else Files API | same + https://aistudio.google.com/docs/files |
| Models: `gemini-3.8-flash` (New), `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite` (stable), `gemini-3-flash-preview`, `gemini-3.1-pro-preview`, `gemini-4-argon` (1M ctx, text/image/audio/video/PDF), `gemini-2.5-flash-lite` (served until further notice). `gemini-2.5-flash` → routes to 3.5-flash-lite 2026-12-01; `gemini-2.5-pro` → 3.8-flash 2026-12-01. Shut down: 2.0-flash, 2.0-flash-lite, 3-pro-preview, 3.1-flash-lite-preview. New projects: "3.5 Flash-Lite or 3.8 Flash" | https://aistudio.google.com/docs/models |
| Pricing per 1M tokens (paid): 3.8 Flash & 3.6 Flash $0.75 in / $3.75 out through 2026-12-31 → $1.50 / $7.50 from 2027-01-01; 3.5 Flash-Lite $0.30 / $2.50; 2.5 Flash-Lite $0.10 ($0.30 audio) / $0.40; 4 Argon $2 / $10 through 2026-11-30 → $4 / $20; 3.1 Pro Preview $2/$4 in, $12/$18 out (≤/>200k). Batch ≈ 50%. Free tier: 3.8, 3.6, 3.5-Lite, 2.5 Flash, 2.5-Lite. 3.8 Flash: one input rate (no audio split); **Flex** input $0.375 → $0.75 (2027-01-01); **Priority** input $1.35 → $2.70; context caching $0.075 → $0.15. 3.5 Flash-Lite $0.30 "text / image / video / audio", caching not available on free tier. Full page (101,809 chars, incl. tail) read: **no YouTube-specific pricing line** | https://aistudio.google.com/docs/pricing |
| Timestamps in prompts: "timestamps of the form MM:SS" e.g. `01:15` | https://aistudio.google.com/docs/video-understanding |
| Structured output: `response_format={"type":"text","mime_type":"application/json","schema": Model.model_json_schema()}`; Pydantic `min_length`/`maxLength` stripped → validate client-side; `generation_config`: `thinking_level` (minimal/low/medium/high — **3.8 Flash rejects `minimal`**, see model page), `thinking_summaries`, `max_output_tokens`, `seed`, `stop_sequences`, `video_config` | https://aistudio.google.com/docs/structured-output , https://aistudio.google.com/api-reference/interactions-api |
| Gemini 4 / 3.6+ currently **ignore** `temperature/top_p/top_k` with deprecation headers; "will return 400 errors in a future release" | https://aistudio.google.com/docs/troubleshooting |
| Retry: 429 rate_limit / 408 / 500 / 503 retryable (1s→2s→4s→8s + jitter); not retryable 400, 402, 403, 404 (model_not_found → next model), 429 `quota_exceeded` (RPD resets midnight Pacific); SDK auto-retries up to 4×, max 60 s; limits per project | https://aistudio.google.com/docs/api-errors , https://aistudio.google.com/docs/rate-limits |
| Claude vision: images only; ≤100 images/request on 200k-ctx models, 600 otherwise; >20 images → each dim ≤ 2000–3000 px rule; 8000×8000 max; 128 MB request; 1920×1080 ≈ 1,560 tokens (standard tier) | https://platform.claude.com/docs/en/build-with-claude/vision |

#### Community reports
| Claim | Source |
|---|---|
| ~~`gemini-3.8-flash` 1M context (reseller listing only)~~ — superseded: now verified-doc (1,048,576) on official model page | freellm.net |
| Agentic on Interactions: 85,480 → 7,416 tokens (~91%) on a 15-min YouTube video; `processing` on generateContent → `400 INVALID_ARGUMENT: Unknown name "processing"` | https://github.com/DerYUYU/gemini-video-mcp |
| godovasik skill calls `/v1beta/interactions` with `processing:"agentic"` and default model `gemini-3.8-flash` | https://github.com/godovasik/gemini-video-analysis |
| OpenAI native video for GPT-5.4/5.5 (blogs only); openai-node #1778 is a feature request for `input_video`; gateways (Vercel AI Gateway, NanoGPT) expose `input_video` but route to Gemini etc. — not OpenAI direct | https://github.com/openai/openai-node/issues/1778 , https://vercel.com/docs/ai-gateway/inputs-and-tools/video-input |
| TwelveLabs Pegasus ~$1.75/h (older doc; new plan-based billing from 2026-10-15 per release notes); version conflict resolved: 1.6 released 2026-10-06, 1.5 default since 2026-08-18 | https://docs.twelvelabs.io/v1.3/docs/guides/analyze-videos-and-images , https://docs.twelvelabs.io/docs/get-started/release-notes (verified-doc) |

#### Assumptions / unverified
- Agentic processing accepts YouTube URIs on **Gemini API (AI Studio)** — undocumented; Vertex documents it. Cannot test (no key).
- Structured output (`response_format`) works together with video parts (audio guide shows YouTube + schema, but untested and with a shape that disagrees with the SDK type), and with agentic (no doc example).
- HH:MM:SS timestamps work for > 1 h videos (docs only ever show `MM:SS`).
- `step_list` stateless mechanism (guide prose) — absent from API ref and from SDK 2.34.0 `CreateModelInteraction` fields (verified); treat as doc error, use `previous_interaction_id` / `store`.
- Gemini speaker labels via prompt are model-inferred, not acoustic diarization — accuracy unknown.
- Actual RPM/RPD per model/tier: only visible in AI Studio dashboard (needs account) — not documented publicly.
- `service_tier:"flex"` semantics (latency/SLA) on Interactions — field verified in SDK, behaviour not read.
- Network hosts: `generativelanguage.googleapis.com` (verified as REST base); upload uses `${BASE_URL}/upload/v1beta/files` — BASE_URL undefined in docs, assumed same host.

### 2.B Transcripts

#### Verified (live test here)
| Claim | Evidence |
|---|---|
| yt-dlp 2026.10.26 (2026-10-26T22:25:24Z, Py ≥3.11); yt-dlp-ejs 0.8.0 (2026-03-17, Py ≥3.10); youtube-transcript-api 1.2.4 (2026-01-29, Py ≥3.8,<3.15, deps requests + defusedxml); pytubefix 11.2.1 (2026-10-16) | PyPI JSON https://pypi.org/pypi/yt-dlp/json etc. |
| `uvx --from "yt-dlp[default,deno,curl-cffi]" yt-dlp -v …` → `JS runtimes: deno-2.9.8`, `Request Handlers: urllib, requests, websockets, curl_cffi`, `[jsc:deno] Solving JS challenges using deno`, `challenge solver lib script v0.8.0 (source: python package, variant: minified)`, no warnings | run on zjkBMFhNj_g |
| Extras: `default` (incl. `yt-dlp-ejs==0.8.0`), `deno` (`deno>=2.9.8`), `curl-cffi` (`>=0.5.10,<0.18`), `secretstorage`, `pin`, `pin-curl-cffi` (0.17.0), `pin-deno` (2.9.8), `pin-secretstorage` | https://pypi.org/pypi/yt-dlp/json requires_dist |
| No runtime: exit 0 but `WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; …` ; node 25 / bun on PATH are **not** used by default | `uvx yt-dlp -J --skip-download jNQXAC9IVRw` |
| `--sub-format srt` is natively offered for both manual and auto tracks (json3, srv1–3, ttml, srt, vtt) | `--list-subs` |
| Video facts (see §5 / brief list): jNQXAC9IVRw manual en+de, 3 chapters, `language` None; zjkBMFhNj_g 3588 s, 21 chapters, `language` "en", `subtitles={}`, 157 auto keys incl. single `en-orig` ("English (Original)", `__yt_dlp_client: tv_samsung`) | `-J --skip-download` |
| json3 shape: `{wireMagic, pens, wsWinStyles, wpWinPositions, events:[{tStartMs, dDurationMs, wWinId, aAppend?, segs:[{utf8, tOffsetMs?, acAsrConf}]}]}`; Karpathy: 3408 events, 13854 word segs, 1703 `aAppend:1` newline events; `acAsrConf` = 0 everywhere | researcher run |
| VTT → SRT conversion of auto captions yields rolling-duplicate cues | researcher run |
| Translated tracks blocked: yt-dlp `HTTP Error 429` on `en-de`, `de` (ja video), `ja` (de video); youtube-transcript-api `.translate('de').fetch()` → IpBlocked (re-reproduced today), while originals fetched same minute | `scratchpad/probe-transcripts/out` |
| `--sub-langs "en.*,en"` regex pulls `en-de` etc. → 429 | researcher run |
| `-J` exposes 40 keys incl. `chapters [{start_time,end_time,title}]`, `heatmap`, `language`, `availability`, `live_status`, `subtitles`, `automatic_captions`; track entry `{ext, url, name, impersonate, __yt_dlp_client}` | zoo `-J` |
| youtube-transcript-api 1.2.4 signatures: `YouTubeTranscriptApi(proxy_config=None, http_client=None)`; `.fetch(video_id, languages=('en',), preserve_formatting=False)`; `.list(video_id)`; `WebshareProxyConfig(proxy_username, proxy_password, filter_ip_locations=None, retries_when_blocked=10, domain_name='p.webshare.io', proxy_port=80)`; `GenericProxyConfig(http_url=None, https_url=None)`; exceptions incl. `TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, VideoUnplayable, IpBlocked, RequestBlocked, AgeRestricted, PoTokenRequired, InvalidVideoId, YouTubeRequestFailed` | introspection |
| youtube-transcript-api lists one "generated" track (no `-orig` distinction); auto snippets overlap in time | live |
| `uvx --from youtube-transcript-api youtube_transcript_api jNQXAC9IVRw --format srt` → valid SRT | live |
| pytubefix: `YouTube(url).captions` → `en`, `de`, `a.en`; `generate_srt_captions()`; 2.9–4.3 s/video | live |
| ASR audio path: `-f "ba[ext=m4a]/ba/b" -x --audio-format mp3 --audio-quality 5 --postprocessor-args "ffmpeg:-ac 1 -ar 16000"` on 9mmVa6O-hzQ → 11.72 MiB m4a → 3.9 MB 16 kHz mono mp3, 11.6 s total, no PO token | live |
| `[pot] PO Token Providers: none`, no PO-token warnings in any test (residential IP) | live |
| **Live streams** (2026-11-05): `@LofiGirl/live` → `4t4AQ15yj1M`, `@NASA/live` → `ROboUHfPUrU`: `live_status:"is_live"`, `is_live:true`, **`duration: None`**, `language: None`, no chapters. LofiGirl `subtitles` = `{"live_chat": …}` (**`live_chat` appears as a subtitle key** — must be excluded from track pick); NASA `automatic_captions` = `{"en":[vtt]}` only (no json3/srt, no `-orig`). Ended stream `jfKfPfyJRdk` → `ERROR: This live stream recording is not available.` | `uvx --from "yt-dlp[default,deno,curl-cffi]==2026.10.26" yt-dlp -J --skip-download` |
| **Age-restricted** `HtVdAasjOgU` (yt-dlp test video, Witcher 3 trailer): yt-dlp **without cookies** → `age_limit:18`, `availability:"needs_auth"`, 36 formats, manual subs (ar, zh-CN, cs, en, fr…), 160 auto keys; `--write-subs --sub-langs en --sub-format srt` downloaded (35 B, one blank cue — track content itself trivial). youtube-transcript-api 1.2.4 → "This video is age-restricted… Cookie Authentication is temporarily unsupported" (fails) | `scratchpad/critic/age` |

#### Verified (docs)
| Claim | Source |
|---|---|
| JS runtime requirement since 2025.11.12 (#15012). EJS wiki minimums: deno 2.3.0 (only default-enabled), node 22.0.0, bun 1.2.11–1.3.14 (deprecated), QuickJS 2023-12-9 / QuickJS-NG (`qjs`). Priority deno > node > quickjs > bun; to force node with deno present: `--no-js-runtimes --js-runtimes node` | https://github.com/yt-dlp/yt-dlp/issues/15012 , https://github.com/yt-dlp/yt-dlp/wiki/EJS |
| Default clients `tv_samsung,visionos,web` (no runtime → web, tv_samsung omitted); extractor args `youtube:player_client`, `skip=hls,dash,translated_subs`, `po_token`, `fetch_pot`, `pot_trace`, `player_skip`, `lang` | https://raw.githubusercontent.com/yt-dlp/yt-dlp/master/README.md (#### youtube) |
| PO Token Guide (edited 2026-07-12): only GVS + Subs enforced; `web` needs Subs; visionos/tv_samsung not in table | https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide |
| Cookies: export from fresh private window after visiting robots.txt; account ban risk; guest ~300 videos/h, account ~2000/h; `-t sleep` = `--sleep-subtitles 5 --sleep-requests 0.75` | https://github.com/yt-dlp/yt-dlp/wiki/Extractors , FAQ |
| youtube-transcript-api: cloud IPs (AWS/GCP/Azure) mostly blocked; residential proxies workaround; cookie auth currently broken | https://raw.githubusercontent.com/jdepoix/youtube-transcript-api/master/README.md |
| YouTube ToS (effective 2023-11-29), Permissions & Restrictions: no accessing/reproducing/**downloading** content and no access by automated means (robots, botnets, scrapers; translated from de: "Mit automatisierten Verfahren (z. B. Robotern, Botnets oder Scrapern)") except public search engines per robots.txt, or as permitted by the Service, YouTube written permission, or applicable law → legal/ToS risk is on the **user**; skill must not ship cookies, must not bulk-crawl, and should keep media downloads transient (cache text/frames, delete media) | https://www.youtube.com/static?template=terms (rendered de locale; quoted items 1 and 3) |
| YouTube Data API `captions.download` needs OAuth scope `youtube.force-ssl`/`youtubepartner` and "permission to edit the video"; 200 units; `tfmt` sbv/scc/srt/ttml/vtt → not a third-party caption source | https://developers.google.com/youtube/v3/docs/captions/download |

#### Community reports
- youtube-transcript-api #592 `PoTokenRequired` when baseUrl has `exp=xpe` — https://github.com/jdepoix/youtube-transcript-api/issues/592 (not reproduced).
- "Sign in to confirm you're not a bot" on datacenter IPs — https://www.technetexperts.com/fix-yt-dlp-cloud-signin-error/ .
- youtube-transcript-api #612 (429 retry doesn't rotate IP), #618, #603.

#### Assumptions / unverified
- `<lang>-orig` = original ASR, plain `<lang>` = machine translation: **observed live, not documented** — could change.
- Signed timedtext URLs expire ~6 h (`expire=`).
- Behaviour on CI/cloud IPs (likely blocked).
- deno PyPI wheel availability for Linux aarch64 / musl.
- Age-restricted via yt-dlp works unauthenticated **today** for one test video (client-dependent; may regress). Members-only / private: untested (expect `availability` `subscriber_only` / `private` → exit 6).
- Live auto-captions (`en` VTT on a live stream) content/latency: not downloaded.

### 2.C Audio / visual processing

#### Verified (live test here)
| Claim | Evidence |
|---|---|
| tesseract, whisper-cli, whisper-cpp not installed; ffmpeg 9.0.2 present | `which` |
| `--download-sections "*00:01:00-00:01:40" --force-keyframes-at-cuts` on rfscVS0vtbw → 40.000 s, 854×480, start_time 0, 27 s wall (re-encode); without flag (aqz-KE-bpKQ, 45 s) → 24.5 s wall | researcher run |
| Remote frame: `U=$(yt-dlp -q -f "bv*[height<=720][ext=mp4]" -g URL \| head -1); ffmpeg -ss 01:30:00 -i "$U" -frames:v 1 -q:v 2 out.jpg` → 1280×720, 37 KB, 4.6 s; code legible at 720p | `probe-av/frames/remote_5400s.jpg` |
| Remote 120 s window stream ≈ 58 s/pass (single sample) | `probe-av/sc2` |
| Scene detection `select='gt(scene,0.3)',showinfo -fps_mode vfr` → 11 cuts on BBB 45 s; screencast (freeCodeCamp + webcam overlay) 0 frames at 0.3/0.1/0.05; max scene score 0.007 | `probe-av/scenes`, `scores.txt` |
| `mpdecimate` with defaults (hi=768, lo=320, frac=0.33) kept 120/120 frames on screencast; local ffmpeg 9.0.2 lacks `mode`/`min` options shown in online docs | `ffmpeg -h filter=mpdecimate` |
| `fps=1/10`, `thumbnail=300`, `-skip_frame nokey` (16 keyframes / 45 s) all work | `probe-av/frames` |
| PySceneDetect 0.7.2 via `uvx scenedetect` (opencv-python is a **core dep**; no `[opencv]`/`[opencv-headless]` extra exists) → 11 scenes | `probe-av/psd`, https://pypi.org/pypi/scenedetect/0.7.2/json |
| mlx-whisper 0.4.3: `mlx_whisper clip20.wav --model mlx-community/whisper-small-mlx --word-timestamps True --output-format json` → 55 s cold / 1.9 s warm for 20 s audio; signature `transcribe(audio: str\|np.ndarray\|mx.array, *, path_or_hf_repo='mlx-community/whisper-tiny', …, word_timestamps=False, clip_timestamps='0', hallucination_silence_threshold=None)`; heavy deps (torch, numba, scipy) | live |
| faster-whisper 1.2.1 + av 19.0.2 → `TypeError: open() got an unexpected keyword argument 'metadata_errors'`; works with `av<19` (18.1.0); av 19 needs Py ≥3.12 so Py 3.11 resolves 18.x | live; https://github.com/SYSTRAN/faster-whisper/pull/1495 (merged 2026-09-30, unreleased) |
| faster-whisper small int8 CPU: 20 s audio in 4.1 s, `language_probability` 0.996, word timestamps | live |
| openai-whisper 20250625 installs via uvx (~24 s) | live |
| parakeet-mlx 0.7.0: `uvx --from parakeet-mlx==0.7.0 parakeet-mlx clip20.wav --output-format json` → cold **65.7 s** (incl. model download `mlx-community/parakeet-tdt-0.6b-v3`), warm **3.1 s wall** incl. uvx startup; punctuated + capitalised English; JSON `{text, sentences[{text,start,end,duration,confidence,tokens[{text,start,end,duration,confidence}]}]}` — tokens are **sub-word** (`" P"`, `"yt"`, `"h"`) → must merge to words; `--highlight-words` SRT emits one cue per token (unusable as transcript.srt) | `scratchpad/critic/parakeet` |
| ocrmac 1.0.1 (Apple Vision, `recognition_level='accurate'`) on 720p code frame `remote_5400s.jpg`: 0.18 s warm (27 s first run incl. install); lines `def sayhi():`, `print ("Hello User")`, `sayhi()`, `Giraffe Academy` — conf 1.0/1.0/0.5; **inserted a space** (`print ("…")` vs on-screen `print("…")`) and **drops indentation**, but bbox x (0.034 vs 0.131) lets indentation be reconstructed. Host-agent vision read the same frame correctly | `scratchpad/critic/scripts/ocr.py`, `bbox.py` |

#### Verified (docs)
| Claim | Source |
|---|---|
| ffmpeg `-ss` before `-i` seeks to nearest seek point and, with `-accurate_seek` (default) when transcoding, decodes+discards up to position → frame-accurate for JPEG output | https://ffmpeg.org/ffmpeg.html |
| `select` `scene` variable is 0–1 (distinct from `scdet` threshold 0–100) | https://ffmpeg.org/ffmpeg-filters.html |
| mpdecimate's main use is very-low-bitrate encoding | https://ffmpeg.org/ffmpeg-filters.html#mpdecimate |
| faster-whisper benchmarks (13 min audio, i7-12700K 8 thr): small fp32 2m37s, int8 1m42s, int8 batch8 51s; whisper.cpp fp32 2m05s; openai fp32 6m58s. `vad_filter` removes silence >2 s; `BatchedInferencePipeline(model).transcribe(..., batch_size=16)`; PyAV bundles FFmpeg | https://github.com/SYSTRAN/faster-whisper |
| whisper.cpp needs CMake build, 16-bit WAV, `-ml 1`, `--vad` | https://github.com/ggml-org/whisper.cpp |
| OpenAI STT: `gpt-transcribe` $0.0045/min; whisper-1 / gpt-4o-transcribe / mini / diarize deprecated, shutdown 2027-02-26; `timestamp_granularities[]` only for whisper-1; 25 MB | https://developers.openai.com/api/docs/guides/speech-to-text |
| Groq: whisper-large-v3-turbo $0.04/h, large-v3 $0.111/h; 25 MB free / 100 MB dev; `timestamp_granularities[]` segment\|word with verbose_json | https://console.groq.com/docs/speech-to-text |
| Deepgram nova-3 $0.0043/min mono; 2 GB max; words[] with start/end/confidence | https://developers.deepgram.com/docs/pre-recorded-audio , https://deepgram.com/pricing |
| Tesseract: ≥300 dpi, psm 4/6/11 | https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html |
| parakeet-tdt-0.6b-v3: 25 European languages (no ja/zh), auto language detection, punctuation/capitalisation, word+segment timestamps, CC-BY-4.0; ~24 min full-attention on A100, up to 3 h local attention. parakeet-mlx default model, `--chunk-duration` default 120 s, `--output-format txt\|srt\|vtt\|json\|all`, Python `from_pretrained(...)`, `model.transcribe(path, chunk_duration=…, overlap_duration=…)`; needs ffmpeg | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 , https://github.com/senstella/parakeet-mlx |
| pyannote `speaker-diarization-community-1`: gated (accept conditions + HF token), `Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=…)`; 31 s per hour audio on H100; paid `precision-2` runs on pyannoteAI servers | https://github.com/pyannote/pyannote-audio |

#### Assumptions / unverified
- Multimodal LLM > tesseract for code/diagrams (not benchmarked).
- googlevideo URLs expire within hours and are IP-bound; throttling at many seeks/video untested.
- Section start snaps to keyframe without `--force-keyframes-at-cuts` → must measure with ffprobe.
- Crop-to-content / perceptual-hash dedupe effective for webcam-overlay screencasts.
- mlx-whisper on Linux (`mlx[cpu]`/`mlx[cuda]` wheels exist) — untested.
- CTranslate2 has no Metal backend (CPU on Mac).

### 2.D Agent Skills packaging

#### Verified (live test here)
- PyYAML 6.0.2: unquoted `description: Use this skill when: the user asks` → `ScannerError`; quoted parses.
- `curl -sI developers.openai.com/codex/skills` → 308 → https://learn.chatgpt.com/docs/build-skills.
- Local: `~/.agents/.skill-lock.json` (multi-agent installer used), `~/.claude/skills` (15 skills), `~/.codex/skills/.system` present; codex/gemini/cursor-agent/opencode not on PATH.
- **skills-ref validator executed**: PyPI `skills-ref` latest **0.1.1** (Py ≥3.11); executable is **`agentskills`** (not `skills-ref`): `uvx --from skills-ref==0.1.1 agentskills validate <dir>` (also `read-properties`, `to-prompt`). Sample skill with `name, description, license, compatibility, metadata{version}` → `Valid skill`, rc 0. Adding `when_to_use` → `Unexpected fields in frontmatter: when_to_use. Only ['allowed-tools', 'compatibility', 'description', 'license', 'metadata', 'name'] are allowed.`, rc 1 → Claude-Code-only keys must stay out of SKILL.md.

#### Verified (docs)
| Claim | Source |
|---|---|
| Frontmatter keys: `name` (1–64, lowercase alnum + `-`, = dir name), `description` (1–1024), `license`, `compatibility` (1–500), `metadata` (str→str), `allowed-tools` (experimental) | https://agentskills.io/specification |
| SKILL.md < 500 lines / < 5000 tokens; references one level deep; tell agent when to read each file | https://agentskills.io/skill-creation/best-practices.md |
| Scripts: PEP 723 + `uv run`, non-interactive, `--help`, JSON stdout, stderr diagnostics, distinct exit codes, idempotent, bounded output (harness truncation 10–30K chars) | https://agentskills.io/skill-creation/using-scripts.md |
| skills-ref validator rejects keys outside the 6; strictyaml; `split('---', 2)`; skills-ref Py ≥3.11 (PyPI latest 0.1.1, not 0.1.0 — corrected) | https://raw.githubusercontent.com/agentskills/agentskills/main/skills-ref/src/skills_ref/validator.py |
| Claude.ai / Skills API upload hard-errors on extra keys; name must not contain `anthropic`/`claude`, no XML tags; < 30 MB | https://platform.claude.com/docs/en/api/skills-guide , https://code.claude.com/docs/en/skills |
| Claude Code: `~/.claude/skills`, `.claude/skills` (no `.agents/skills`); symlinks OK; `${CLAUDE_SKILL_DIR}`; cwd is project not skill dir | https://code.claude.com/docs/en/skills |
| Claude Code sandbox off by default; `sandbox.network.allowedDomains` empty by default; writes cwd + `$TMPDIR` + `--add-dir` | https://code.claude.com/docs/en/sandboxing |
| Codex: `.agents/skills` (repo), `~/.agents/skills` (user), `/etc/codex/skills`; network off in workspace-write → `[sandbox_workspace_write] network_access = true`; optional `agents/openai.yaml` | https://learn.chatgpt.com/docs/build-skills , https://learn.chatgpt.com/docs/agent-approvals-security |
| Copilot: `.github/skills`, `.claude/skills`, `.agents/skills`, `~/.copilot/skills`, `~/.agents/skills`; cloud-agent firewall allowlist excludes youtube/googlevideo/generativelanguage | https://docs.github.com/en/copilot/concepts/agents/about-agent-skills , https://docs.github.com/en/copilot/reference/copilot-allowlist-reference |
| Gemini CLI: `~/.gemini/skills`, `~/.agents/skills`, `.agents/skills`; Antigravity `.agents/skills`, `~/.gemini/config/skills` | https://geminicli.com/docs/cli/skills/ , https://antigravity.google/docs/skills |
| Cursor and OpenCode read `.agents/skills`, `~/.agents/skills`, `.claude/skills`, `~/.claude/skills` | https://cursor.com/docs/context/skills , https://opencode.ai/docs/skills/ |
| **Claude Code** detail: locations enterprise / `~/.claude/skills` / `.claude/skills` (+ parent dirs to repo root) / nested `<subdir>/.claude/skills` (on demand) / `--add-dir` / plugin `<plugin>/skills/` (namespaced `/plugin:skill`). Precedence enterprise > personal > project. Live reload of skill dirs (new top-level dir → `/reload-skills`). `description`+`when_to_use` truncated at **1,536 chars** in listing. Extra CC-only keys (`when_to_use, argument-hint, arguments, disable-model-invocation, user-invocable, disallowed-tools, model, effort, context, agent, background, hooks, paths, shell`); unknown keys silently ignored. Install via plugin: `/plugin marketplace add <owner/repo>` then `/plugin install <plugin>@<marketplace>` | https://code.claude.com/docs/en/skills |
| **Codex** detail: REPO `.agents/skills` in every dir from cwd up to repo root; USER `$HOME/.agents/skills`; ADMIN `/etc/codex/skills`; SYSTEM bundled (`skill-creator`, `skill-installer`). Docs **do not mention `~/.codex/skills`**. Duplicates not merged. Symlinks followed. Install: `$skill-installer <name>` (curated; can fetch other repos on request); no separate CLI install documented. Invoke `/skills` or `$name`; implicit via `description`. `agents/openai.yaml`: `interface{display_name, short_description, icon_small, icon_large, brand_color, default_prompt}`, `policy.allow_implicit_invocation` (default true), `dependencies.tools`. Disable: `[[skills.config]]` with `path` or `name` + `enabled = false` in `~/.codex/config.toml` (last match wins; restart) | https://learn.chatgpt.com/docs/build-skills |
| **Gemini CLI** detail: precedence built-in < extension < user (`~/.gemini/skills` or `~/.agents/skills`) < workspace (`.gemini/skills` or `.agents/skills`); `.agents/skills` beats `.gemini/skills` within a tier. Install `gemini skills install https://github.com/user/repo.git [--path <subdir>] [--scope user\|workspace] [--consent]`; `gemini skills list --all`; `gemini skills uninstall <name> --scope workspace`; in-session `/skills list\|link <path>\|enable\|disable\|reload`. **Activation requires user approval prompt** (`--consent` only skips install-time confirmation) | https://geminicli.com/docs/cli/skills/ |
| **Copilot CLI** detail: in-session `/skills list`, `/skills` (enable/disable picker), `/skills info NAME`, `/skills add`, `/skills reload`, `/skills remove DIR`; terminal `copilot skill list\|add <FILE\|URL\|DIRECTORY>\|enable\|disable`. Cloud sandbox: personal skills via dotfiles repo `.copilot/skills` | https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-skills |
| **`gh skill install <OWNER/REPO> [<skill[@version]>]`** flags `--agent` (48 values incl. `github-copilot` (non-interactive default), `claude-code`, `codex`, `gemini-cli`, `cursor`, `opencode`, `antigravity`, `universal`), `--scope project\|user` (default project), `--dir`, `--pin <tag\|sha>`, `--all`, `--force`, `--from-local`, `--allow-hidden-dirs`, `--upstream`. Project scope: Copilot, Cursor, Codex, Gemini CLI, Antigravity, Amp, Cline, OpenCode, Warp **share `.agents/skills`** → one cross-agent install command | https://cli.github.com/manual/gh_skill_install |
| Vercel `npx skills add <owner/repo>` with `-a/--agent <agents…>` (`'*'` = all; ids `claude-code`, `codex`, `cursor`, `gemini-cli`, …) — community-report (wiki via search snippet; agent count varies 41–75+ by version) | https://github.com/vercel-labs/skills/wiki |

#### Assumptions / unverified
- ~~skills-ref validate untested~~ → resolved (PyPI 0.1.1, exe `agentskills`, run above).
- Whether Codex still scans `~/.codex/skills` (directory exists locally; docs silent).
- Gemini CLI activation-consent prompt in non-interactive/headless mode (`gemini -p`) — behaviour not documented in skills page.
- Exact network host list for sandbox allowlists (youtube.com, *.googlevideo.com, *.ytimg.com, generativelanguage.googleapis.com, pypi.org, files.pythonhosted.org).
- Codex workspace-write writable roots (is `~/.cache/uv` writable?).

### 2.E Existing solutions

#### Verified (live test here)
- No YouTube/video skill in anthropics/skills (187,535★, pushed 2026-10-28) or anthropics/claude-plugins-official (`gh api …/git/trees/main?recursive=1`).

#### Verified (docs) — repo READMEs / GitHub metadata
| Project | Stars / last push / license | What it does | Gap vs our goal |
|---|---|---|---|
| intellectronica/agent-skills `youtube-transcript` | 299★, 2026-04-25, CC0; 3.4K installs (highest seen) | youtube-transcript-api, `[MM:SS] text` | Transcript only |
| kimtaeyoon83/mcp-server-youtube-transcript | 603★, 2026-07-21, MIT, TS | `get_transcript(url, lang, include_timestamps, strip_ads)`, optional TwelveLabs (direct mp4 only) | No frames; open 429 issue #39 |
| jkawamoto/mcp-youtube-transcript | 495★, 2026-11-04, MIT, Py | transcript/timed/languages/info, 50,000-char pages, proxy options | No frames |
| anaisbetts/mcp-youtube | 552★, 2026-08-16, MIT | one yt-dlp sub download | Minimal |
| kevinwatt/yt-dlp-mcp | 299★, 2026-10-29, MIT | 10 `ytdlp_*` tools | No vision |
| coyaSONG/youtube-mcp-server | 27★, 2026-10-26, MIT | `research-video` timestamp-cited excerpts, `&t=65s` links | Text only |
| steipete/summarize | 6,787★, 2026-11-04, MIT, Node ≥24 | caption chain (youtubei → captionTracks json3 → yt-dlp → Apify), ASR chain, `--slides` scene frames, `--slides-ocr`, JSON envelope | Not Python skill; no evidence.json / VLM code reading |
| byjlw/video-analyzer | 1,609★, Apache-2.0 | frames + Whisper + vision LLM | Local files, heavy |
| haithamelmengad/popcorn | 7★, 2026-02-05, MIT, TS | scene frames + whisper backends, videoType presets (Claude Code MCP plugin) | Local files |
| ZubeidHendricks/youtube-mcp-server | 593★, 2026-08-08, MIT, TS | repo description: "MCP Server for YouTube API, enabling video management, Shorts creation, and advanced analy[tics]" (README not read) | API-oriented; no frames (assumption) |
| kirbah/mcp-youtube | 33★, 2026-11-01, MIT, TS | description: "token-optimized, structured data for your LLMs" (README not read) | Text only (assumption) |
| tubeagentkit/youtube-mcp | 24★, 2026-10-30, MIT, JS | transcripts, search, channels/playlists | Text only |
| mrzhangguoguo/gemini-video-analysis-skill | 7★, 2026-06-26, MIT | Gemini-native transcripts, storyboards, keyframes, HTML report | Closest overlap; small, unproven; no evidence ledger seen |
| ludmila-omlopes/agent-skills `youtube-gemini-analyzer` | 1★, 2026-07-24, **no license** | Gemini YouTube analyzer skill | No license → cannot reuse code |
| kwacky1/video-understanding-mcp | 0★, MIT | frame provenance (PTS, timebase, reason, sha256), max 24 frames, cache key incl. model sha256 | Local only, early |
| godovasik/gemini-video-analysis | 0★, MIT | Interactions static/agentic, `gemini-3.8-flash` | Unproven |
| DerYUYU/gemini-video-mcp | 1★, MIT | analyze_video, model fallback on 429/503 | Unproven |
| Jayne040622/video-to-knowledge-skill | 1★, MIT | manifest, media hash, segments/frames JSON, `source_changed` refusal | Chinese platforms, whisper.cpp |
| shepnerd/vidify | 8★, Apache-2.0 | evidence `{asr_segment_ids, frame_ids}` | Needs self-hosted vLLM |
| samihalawa/video-analysis-mcp (2★, 2025-09-10, **no license**) / adamanz/gemini-video-mcp-server (0★, 2025-12-27, MIT) | stale | — | Excluded |

Dependency licenses (GitHub `license.spdx_id`, 2026-11-05): yt-dlp **Unlicense**, yt-dlp/ejs Unlicense, youtube-transcript-api MIT, pytubefix MIT, googleapis/python-genai Apache-2.0, faster-whisper MIT, mlx-examples (mlx-whisper) MIT, openai/whisper MIT, parakeet-mlx Apache-2.0 (model weights CC-BY-4.0 → attribution), whisperX BSD-2-Clause, PySceneDetect BSD-3-Clause, ocrmac MIT, deno MIT, curl_cffi MIT. ffmpeg = system binary (LGPL/GPL depending on build) — not redistributed by the skill. All compatible with an MIT/Apache-2.0 skill.

#### Community reports
- Install counts of other skills.sh YouTube skills (1–27); ClawHub "suspicious patterns" flag on youtube-transcriber.

---

## 3. Comparison

Scores are qualitative (High / Med / Low) unless numbers are given. Cost assumes paid tier, 2026 prices.

| Approach | Accuracy (speech) | Accuracy (on-screen code / diagrams) | Cost (1 h video) | Speed | Reliability | Maintenance | Status of evidence |
|---|---|---|---|---|---|---|---|
| **Gemini direct-URL, static** (`processing:"static"`, low res, 1 fps) | High (audio 32 tok/s included) — assumption, untested | Med at low res (70 tok/frame); High at `resolution:"high"` for short clipped windows | ~367k tok ≈ **$0.28** on 3.8 Flash (low); ~1.1M tok at high → exceeds 1M ctx, must clip; YouTube ingest itself "no charge" (preview) | Single call; latency unknown (untested) | Public videos only; preview feature "may change"; free tier 8 h/day | Low (one SDK) but fast-moving model IDs (deprecations 2026-12-01) | verified-doc, not live |
| **Gemini agentic** (`processing:"agentic"`) | High; "~7% higher quality" claim | High for targeted questions (model navigates) | "up to 88% fewer tokens" on long-form; per-second cost variable; more tokens on <5 min clips | Higher TTFT; use `background=True` + poll | **YouTube+agentic on Gemini API undocumented**; needs feature-detect + static fallback | Low–Med | verified-doc (Vertex documents YouTube); community 91% anecdote |
| **Transcript via yt-dlp** (`-J` + one json3/srt track) | Manual: High; auto ASR: Med (no punctuation in de; poor names in ja) | None | $0 | 2–4 s metadata + 2–4 s track (deno cached; first run ~12 s) | High from residential IP; translated tracks 429; datacenter IP bot-check risk | Med: yt-dlp releases frequently; JS runtime requirement | verified-live-test |
| **Transcript via youtube-transcript-api** | Same caption data; no `-orig` distinction; overlapping snippets | None | $0 | ~0.2–1.4 s | Cloud IPs blocked; cookie auth broken; #592 PoToken risk | Low–Med | verified-live-test + doc |
| **Local ASR fast path** (parakeet-mlx, darwin-arm64, 25 EU langs) | Good + punctuated (single 20 s English clip) | None | $0 | warm 3.1 s / 20 s audio incl. startup; cold 66 s (model download) | Apple Silicon only; no ja/zh; sub-word tokens need merging | Low–Med | verified-live-test (1 clip) |
| **Local OCR pre-pass** (ocrmac / Apple Vision) | n/a | Med–High text, loses indentation + spacing quirks; layout recoverable from bbox | $0 | 0.18 s / frame warm | macOS only | Low | verified-live-test (1 frame) |
| **Local ASR fallback** (mlx-whisper / faster-whisper) | Good (whisper-small: lang prob 0.996 on test); speaker labels none | None | $0 compute; ~0.3 MB/min audio download | mlx warm 1.9 s / 20 s audio; faster-whisper int8 4.1 s / 20 s (≈ 12 min per 1 h CPU, extrapolated) | Good; heavy installs (torch 132 MiB etc.); faster-whisper needs `av<19` | Med (pins) | verified-live-test |
| **Frame extraction + multimodal frames** (ffmpeg + host vision or Gemini images) | n/a (pair with transcript) | High at 720p for code (observed legible); diagrams depend on viewer model | Host vision: no extra API cost; Claude ≈ 1.2–1.6k tok/frame | ~5 s per targeted remote frame; section download ~25 s / 45 s clip | Scene detection fails on screencasts (score 0.007); googlevideo URL expiry | Med (ffmpeg filter quirks) | verified-live-test |
| **Existing MCP / skills** | Transcript-only for most | None except summarize `--slides` (+tesseract) | $0 – hosted credits | Varies | Many open 429 / "not working" issues | Out of our control | verified-doc |

---

## 4. Recommended architecture

### 4.1 Rationale (one paragraph)
Captions are free, fast and exact-timestamped, so every run starts with metadata + captions via yt-dlp. Gemini's native YouTube ingestion is the only option that "watches" the video (speech + visuals) without downloading it, so it is the primary multimodal path when `GEMINI_API_KEY` exists, used **targeted** (clipped static windows or agentic) rather than blanket over long videos. Without a key, the host agent's own vision on targeted ffmpeg frames replaces Gemini. Everything is recorded as evidence with provenance so claims can be verified against transcript spans and frames.

### 4.2 Layout
```
youtube-understanding/
  SKILL.md                      # <500 lines; 6 spec keys only
  scripts/yt.py                 # PEP 723 entrypoint (subcommands)
  scripts/yt_skill/...          # package: providers, cache, package writer
  references/gemini.md          # prompting, shapes, limits, errors
  references/troubleshooting.md # 429, bot-check, sandbox network, cookies
  references/evidence-schema.md
  assets/evidence.schema.json
  assets/knowledge.template.md
  agents/openai.yaml            # optional Codex UI metadata
```
`requires-python >=3.11` (yt-dlp, google-genai, skills-ref all need it).

### 4.3 Provider interfaces
```python
class TranscriptProvider(Protocol):
    name: str                      # "ytdlp" | "yta" | "asr-parakeet" (darwin-arm64, EU langs) | "asr-mlx" | "asr-fw" | "asr-gemini" (speaker labels via prompt) | "asr-groq"
    def available(self) -> Availability: ...           # deps, keys, network probe
    def inventory(self, meta: VideoMeta) -> list[TrackRef]: ...
    def fetch(self, video_id: str, track: TrackRef | None, window: Window | None) -> Transcript
    # Transcript = segments[{id, start, end, text, words?[{w,start,end,conf?}]}] + provenance

class MultimodalProvider(Protocol):
    name: str                      # "gemini-interactions" | "gemini-generatecontent" | "host-frames"
    def available(self) -> Availability: ...
    def analyze(self, req: AnalyzeRequest) -> AnalyzeResult
    # AnalyzeRequest: url, prompt, schema, mode(static|agentic), window(start,end)?, fps?, resolution, model
    # AnalyzeResult: parsed JSON, raw text, usage tokens, steps summary (agentic_confirmed: bool), model, request hash

class FrameProvider(Protocol):     # local fallback / verification
    def grab(self, video_id: str, t: float, height: int = 720) -> Frame          # remote seek
    def window(self, video_id: str, start: float, end: float, strategy: str) -> list[Frame]
    # Frame: path, abs_ts, pts, selection_reason, sha256, width, height
```

### 4.4 Auto strategy rules (`--strategy auto`)
1. **Always**: `yt-dlp -J --skip-download` → metadata, chapters, availability, language, track inventory. Abort with clear error if `availability != "public"` *and* Gemini path requested (Gemini rejects private/unlisted); captions/frames path may still work for unlisted. `live_status == "is_live"` / `duration is None` → refuse with exit 6 ("live stream; retry after it ends") — verified live: no duration, no usable captions. `availability == "needs_auth"` (age-gated) → yt-dlp path only (youtube-transcript-api fails; Gemini behaviour unknown).
2. **Transcript track pick** (exactly one): `subtitles[lang]` → `subtitles[any]` (prefer `en`) → `automatic_captions[f"{lang}-orig"]` → `automatic_captions[lang]` → youtube-transcript-api → ASR. `lang` = `info.language` base (may be None → `en` → any `*-orig`; for multi-`-orig` dubbed videos pick the one matching `info.language`). Never request translations. Always drop the `live_chat` key (seen inside `subtitles` on a live stream; on replays = assumption).
3. **Multimodal decision**:
   - No `GEMINI_API_KEY` → `host-frames` (transcript + targeted frames for host agent to view).
   - Key present, duration ≤ 10 min → Gemini static, `resolution:"high"` if est. tokens (`dur × 312`) < 800k else low.
   - Key present, 10 min – ~3 h → Gemini static low for overview *only if* `dur × 102 < 800k`; otherwise chunk (4.9). For a **question** (`ask`), prefer agentic first, then static window.
   - > 3 h → never whole-video static; chapters/windows only, or agentic.
   - Agentic: feature-detect — if call errors (400) or no `processing_call` step appears, record `agentic_unavailable` in cache and fall back to static window.
4. **Visual need detection** (when to grab frames): transcript cues ("as you can see", "this code", "on the screen", "diagram", "slide", "let me show"), chapter titles, low-word-density stretches, Gemini-returned `visual_events` timestamps, user question about code/diagrams.
5. **Model ladder**: `gemini-3.8-flash` → `gemini-3.5-flash-lite` (on 404 `model_not_found`, cost mode `--budget`, or quota) → `gemini-2.5-flash-lite` (budget, only if needed). Env override `YTU_GEMINI_MODEL`.

### 4.5 Caching
Root: `${YTU_CACHE_DIR:-${XDG_CACHE_HOME:-~/.cache}/youtube-understanding}` with fallback to `$TMPDIR` if not writable (sandboxes).
| Artifact | Key |
|---|---|
| metadata | `video_id` + yt-dlp version (TTL 24 h; views/likes change) |
| transcript | `video_id` + source + track code (`en-orig`, `de`, `asr:mlx-whisper-small`) + tool version |
| frames | `video_id` + abs_ts (ms) + height + extraction method |
| gemini result | sha256 of canonical JSON `{video_id, provider, model, mode, resolution, fps, start_offset, end_offset, prompt_template_id+version, user_question, schema_hash, sdk_version}` |
Never cache signed googlevideo/timedtext URLs beyond the run.

### 4.6 Cost controls
- Pre-flight estimate printed in JSON: static `tokens ≈ dur_s × (70|280 + 32) / fps-adjusted`; `$ = tokens/1e6 × price(model)` (price table in `references/gemini.md`, dated).
- `--max-cost-usd` (default 0.50) and `--max-input-tokens` (default 800k); refuse or downscale (low res → lower fps → clip) before calling.
- `--dry-run` prints plan + estimate, makes no Gemini call.
- Record actual `usage` from the response; warn when estimate and actual differ > 30% (resolves 66 vs 70 tok/frame question).
- Track free-tier YouTube hours/day locally (8 h cap) when `YTU_GEMINI_TIER=free`.
- Omit `temperature/top_p/top_k`; use `thinking_level:"low"` for extraction, `"medium"` for verification; set `max_output_tokens`.

### 4.7 Output package (`--out`, default `./youtube-knowledge/<video_id>/`)
```
metadata.json      # yt-dlp subset + provenance {transcript_source, track, is_generated, yt_dlp_version,
                   #   js_runtime, client, providers_used, models, cost_estimate, cost_actual, created_at}
transcript.srt     # built from json3/segments (deduped), never VTT→SRT
transcript.json    # segments with ids (s0001…) and word timings when available
knowledge.md       # summary, chapters, key concepts, code blocks, diagrams, Q&A; every claim cites [E12] + MM:SS link
evidence.json      # see 4.8
screenshots/       # t0001234.jpg (ms timestamp in name), 720p, ≤1568 px long edge
```
Citations use `https://www.youtube.com/watch?v=ID&t=<s>s`. Knowledge handoff to the agent chunked at ≤ 50,000 chars.

### 4.8 Evidence and uncertainty tracking
`evidence.json`:
```json
{"schema_version":"1","video_id":"…","source_hash":"sha256(metadata core + transcript)","items":[
 {"id":"E12","claim":"…","kind":"speech|code|diagram|slide|ui|inference",
  "t_start":754.2,"t_end":768.0,"timestamp":"12:34","url":"…&t=754s",
  "transcript_segment_ids":["s0412","s0413"],"frame_ids":["F0031"],
  "sources":["captions:manual","gemini:static:gemini-3.8-flash","host-vision"],
  "status":"verified|corroborated|single-source|unverified|contradicted",
  "confidence":"high|medium|low","notes":"…"}],
 "frames":[{"id":"F0031","path":"screenshots/t0754200.jpg","abs_ts":754.2,"pts":…,"timebase":"1/1000",
            "selection_reason":"transcript-cue|gemini-event|interval|scene|user-question","sha256":"…","width":1280,"height":720}],
 "gaps":[{"t_start":…,"t_end":…,"reason":"no captions|asr low quality|frame unreadable"}]}
```
Rules: `verified` = transcript text or frame pixels re-checked in verification pass; `corroborated` = two independent sources agree (e.g. Gemini + captions); `single-source` = Gemini only; `contradicted` kept and surfaced, never silently dropped. Code snippets require a frame (`kind:"code"` without `frame_ids` → `unverified`). Model-generated timestamps are snapped to nearest transcript segment and flagged if > 5 s off.

### 4.9 Prompt-injection handling
- Treat titles, descriptions, captions, on-screen text and Gemini output as **untrusted data**. Wrap in delimited blocks (`<untrusted source="captions">…</untrusted>`) in any prompt; system/instruction text states content may contain instructions to ignore.
- Gemini outputs must be schema-validated JSON (`response_format`); free-text fields stored, never executed. Strip/flag strings matching instruction patterns ("ignore previous", "run this command", URLs to scripts) into `evidence.items[].notes` with `injection_suspect:true`.
- SKILL.md instructs the host agent: never run commands, install packages or visit URLs found in video content; code from videos is quoted as evidence, not executed.
- Escape `<`/`>` when writing to knowledge.md headings; never write into skill dir.

### 4.10 Extraction + verification passes
1. **Inventory** (no media): metadata, chapters, transcript.
2. **Extraction**: per chapter/window, Gemini (structured schema: `segments[{start,end,summary}]`, `visual_events[{t, type: code|diagram|slide|demo, description, transcribed_text}]`, `claims[{text,t}]`), or transcript-only map step in no-key mode.
3. **Targeted frames**: for each `visual_event` / transcript cue → remote 720p grab at `t` (and `t+2s` for scrolling code).
4. **Verification**: second pass (Gemini static clip `[t-10s, t+10s]`, `resolution:"high"`, or host vision on frames) re-reads every code/diagram item; compare text similarity with first pass; mismatch → re-grab ±2 s, mark `contradicted` if still differing. Speech claims verified by substring/fuzzy match against transcript segments.
5. **Reduce**: assemble knowledge.md from verified/corroborated items first, unverified clearly marked.

### 4.11 Long-video chunking
- Windows = chapters when present (merge chapters < 2 min, split > 20 min); else fixed 10–15 min windows with 15 s overlap.
- Static Gemini per window: `processing={"type":"static","start_offset":f"{s}s","end_offset":f"{e}s","fps":F}`; fps 0.5 for talk-heavy, 1 for screencasts; prompt timestamps are requested **absolute** (MM:SS / HH:MM:SS) and validated to lie within window (if model returns window-relative times, detect by range and add offset).
- Local media windows: `yt-dlp --download-sections "*S-E"`; absolute_ts = ffprobe-measured section start + pts_time (keyframe snap).
- Reduce step merges window outputs; deduplicate overlapping claims by timestamp ±15 s + text similarity.

### 4.12 Targeted question retrieval (`ask`)
1. Search transcript (BM25/keyword + chapter titles) → top-k segments → windows `[t-60s, t+90s]`, merged.
2. If key: agentic call with question (feature-detected) **or** static clips at those windows (`resolution:"high"` if short) with `previous_interaction_id` for follow-ups.
3. No key: return transcript excerpts + 3–6 targeted frames for host agent to view.
4. Answer JSON: `{answer, confidence, evidence:[{timestamp,url,segment_ids,frame_ids}], not_found:bool}`; "not found in video" allowed and preferred over guessing.

### 4.13 Reliability & environment
- Network preflight (DNS + HTTPS HEAD to youtube.com and generativelanguage.googleapis.com) with per-client remediation messages (Codex `network_access = true`; Claude Code `sandbox.network.allowedDomains`; Copilot cloud custom allowlist).
- Retry: 429 rate_limit/408/500/503, 1-2-4-8 s + jitter, max 5, cap 60 s; never retry 400/402/403/404/quota_exceeded.
- yt-dlp errors → typed exit codes (2 bad args, 3 missing dep, 4 network/blocked, 5 auth/missing key, 6 video unavailable/private, 7 budget exceeded).
- `--cookies FILE` / `--cookies-from-browser` passthrough only after a bot-check error, with ban-risk warning; `--sleep` preset for batches.
- ToS: YouTube ToS restricts downloading and automated access (see §2.B) → single-video, user-initiated use only; no bulk/channel crawling; delete downloaded media after frame/ASR extraction; document in SKILL.md.
- ASR order: captions → (darwin-arm64 and lang in parakeet-v3 set) parakeet-mlx → mlx-whisper / faster-whisper → Gemini audio (if key) → openai-whisper. Diarization: none by default; `--speakers` uses Gemini prompt-based speaker labels (marked `inference`), pyannote not in v1.
- OCR: on macOS optionally run ocrmac on code frames as a cheap cross-check against VLM reading (indentation from bbox x); disagreement → `corroborated`/`contradicted` logic in 4.8.

### 4.14 Distribution / install
- Ship as plain Agent Skill dir (6 spec keys only; validated with `uvx --from skills-ref agentskills validate`).
- Cross-agent install: `gh skill install <owner>/<repo> youtube-understanding --agent <id> --scope user` (writes `.agents/skills` for Copilot/Codex/Gemini CLI/Cursor/OpenCode at project scope); Gemini CLI `gemini skills install <git-url> --path youtube-understanding`; Codex `$skill-installer` or copy to `~/.agents/skills`; Claude Code copy/symlink to `~/.claude/skills/youtube-understanding` or ship a plugin (`/plugin install`).
- Claude-Code-only frontmatter (`when_to_use`, `disable-model-invocation`, …) **must not** be added — fails `agentskills validate` and claude.ai upload.

---

## 5. Known limitations and unresolved questions

### Limitations
- **Gemini multimodal path is entirely unverified live** (no `GEMINI_API_KEY`): request shapes, YouTube ingestion latency, structured output with video, token accounting, error bodies.
- Gemini YouTube ingestion is a **preview**: public videos only, free tier 8 h/day, terms and pricing "may change".
- Model churn: `gemini-2.5-flash`/`2.5-pro` reroute 2026-12-01; 3.8 Flash price doubles 2027-01-01; 4 Argon price doubles 2026-12-01.
- Captions: YouTube translated tracks 429/IpBlocked from this IP; `-orig` semantics undocumented; ASR captions have no punctuation (de) and no speaker labels; `acAsrConf` always 0.
- Datacenter/CI IPs untested; both youtube-transcript-api README and community reports predict blocks.
- Scene detection does not work for screencasts with webcam overlays (single test video); mpdecimate also failed.
- Local ASR installs are heavy (torch, mlx-metal); faster-whisper broken with PyAV 19 on Py ≥3.12 until a release > 1.2.1.
- No diarization in any caption source; Gemini offers prompt-based speaker labels (documented, untested); pyannote community-1 gated (HF token), not evaluated.
- Live streams unsupported (verified: `duration None`, only `live_chat`/VTT tracks). Age-restricted: yt-dlp OK unauthenticated on 1 video, youtube-transcript-api fails. Members-only/private: untested.
- YouTube ToS restricts downloading/automated access; compliance responsibility on user.
- parakeet-mlx: Apple Silicon only, 25 European languages; OCR pre-pass (ocrmac) macOS only — Linux has no live-tested local OCR.
- Gemini RPM/RPD per model not publicly documented (AI Studio dashboard only).

### Unresolved questions
1. Does `processing:"agentic"` accept YouTube URIs on the Gemini API (AI Studio)? (Vertex: yes.)
2. Does `Part.media_processing=AGENTIC` work on Gemini API generateContent (SDK has it; DerYUYU reports `processing` rejected on generateContent — different field name)?
3. Does `response_format` JSON schema work with video parts and with agentic? (Audio guide shows YouTube + bare schema — doc-level yes for static; shape vs SDK `TextResponseFormat` unresolved; agentic undocumented.)
4. Tokens per frame for 3.8 Flash at default: 66/258 (video guide, re-read today) vs 70/280 (media-resolution page)? Measure via `usage`.
5. ~~`gemini-3.8-flash` context window~~ → **resolved**: 1,048,576 in / 65,536 out (model page).
6. Stateless multi-turn: `step_list` (guide prose) — **not in SDK request fields** (verified); use `store` / `previous_interaction_id`.
7. Inline video limit: 20 MB (video prose) vs 100 MB (video table + Files page) — still conflicting; Files API limits **resolved** (2 GB / 20 GB / 48 h).
8. Are window timestamps returned absolute or relative when `start_offset` is set?
9. ~~Pricing page tail~~ → **resolved**: no YouTube pricing line.
10. PO-token enforcement spreading to default clients (visionos/tv_samsung)?
11. Perceptual-hash / crop dedupe for screencasts; PySceneDetect `HashDetector`.
12. googlevideo URL lifetime and throttling under many remote seeks.
13. ~~skills-ref validation~~ → **resolved** (0.1.1, `agentskills validate`). Codex `~/.codex/skills` still undocumented.
14. deno PyPI wheel coverage on Linux aarch64/musl.
15. Gemini behaviour on age-restricted (`needs_auth`) public videos.
16. Gemini CLI skill activation consent in headless `gemini -p` runs.
17. Cross-platform local OCR for Linux (RapidOCR untested).

---

## Appendix: live-tested video IDs
| ID | Content | Used for |
|---|---|---|
| `jNQXAC9IVRw` | "Me at the zoo", 19 s, manual en+de, 3 chapters, `language` None | manual captions, `-J` fields, regex 429 pitfall, yta SRT |
| `zjkBMFhNj_g` | Karpathy "[1hr Talk] Intro to LLMs", 3588 s, 21 chapters, en, auto-only `en-orig` | json3 word timings, yta generated, translation block |
| `kCc8FmEb1nY` | Karpathy "Let's build GPT", 6980 s, 30 chapters, en-US, 12 `*-orig` keys (dubbed audio) | multi-orig edge case |
| `9mmVa6O-hzQ` | "PYTHON LERNEN in 10 Minuten", 759 s, de, 4 chapters, auto-only `de-orig` | German ASR, audio-only ASR download |
| `hYfYO2mEUnk` | Kurzgesagt German making-of, 603 s, manual `de` + `de-orig`, 6 chapters | manual non-English |
| `dD_KwXU9meA` | Japanese programming lecture #1, 409 s, ja, auto-only `ja-orig` | Japanese ASR, `skip=translated_subs` |
| `8jPQjjsBbIc`, `arj7oStGLkU` | TED talks, many manual languages | multi-language manual |
| `aircAruvnKk` | 3Blue1Brown, many manual languages | multi-language manual |
| `rfscVS0vtbw` | freeCodeCamp Python course (~4.5 h screencast + webcam) | section download, screencast scene/mpdecimate failure; likely the remote 01:30:00 frame test (ID not recorded in that log) |
| `aqz-KE-bpKQ` | Big Buck Bunny | scene detection, PySceneDetect, keyframes |
| `e6vPt_e9sRw` (de 6.3 h), `FFrrShS7LZw` (de 72 min), `tCMl1AWfhQQ` (ja 2.5 h, 23 chapters) | metadata only | long-video candidates |
| `9hE5-98ZeCg` | Google's doc example URL | not tested (no key) |
| `4t4AQ15yj1M` (@LofiGirl/live), `ROboUHfPUrU` (@NASA/live), `jfKfPfyJRdk` (ended) | live streams | `live_status`, `duration None`, `live_chat` key, ended-stream error |
| `HtVdAasjOgU` | age-restricted (18+) trailer | yt-dlp unauthenticated OK; youtube-transcript-api AgeRestricted |
