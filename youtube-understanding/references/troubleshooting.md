# Troubleshooting

Always start with `uv run <skill_dir>/scripts/yt.py doctor --network` (binaries, key presence,
cache dir, ASR backend, DNS/HTTPS to youtube.com + googleapis.com). Error JSON:
`{"ok": false, "error": {"type", "message", "remediation", ...}}` — follow `remediation`.
Add `-vv` for debug logs on stderr (secrets and signed URLs are redacted).

## Exit codes

| Code | Type | Typical cause | Fix |
|---|---|---|---|
| 1 | InternalError / ProcessFailed | Bug, unexpected tool output | Re-run with `-vv`; report stderr |
| 2 | BadInput | Bad URL/timestamp, `--start` without `--end`, `--out` inside skill dir | Fix arguments |
| 3 | MissingDependency | `ffmpeg`, `uv`/`uvx` missing | `brew install ffmpeg uv` / `apt install ffmpeg` + https://docs.astral.sh/uv/ |
| 4 | NetworkError / Blocked / SubprocessTimeout | No network, sandbox, YouTube bot check, 429 | See below |
| 5 | MissingKey / AuthError | Gemini key absent or rejected | [gemini.md](gemini.md#key-setup) or `--strategy transcript` |
| 6 | VideoUnavailable / LiveStream | Private, removed, region-locked, members-only, live | See below |
| 7 | OverBudget | Estimate > `--max-cost-usd` / `--max-input-tokens` | `--dry-run`, `--start/--end`, `ask`, raise limit if user agrees |

## Network and sandboxes

Hosts needed: `www.youtube.com`, `youtube.com`, `*.googlevideo.com` (frames/audio),
`*.ytimg.com`, `generativelanguage.googleapis.com` (Gemini), plus `pypi.org` and
`files.pythonhosted.org` for the first `uv run` (dependency install). Host list beyond
youtube.com/googleapis.com is best-effort.

| Client | Enable network |
|---|---|
| Codex | `~/.codex/config.toml`: `[sandbox_workspace_write]` `network_access = true`, restart. If `~/.cache/uv` is not writable set `UV_CACHE_DIR` to a writable dir |
| Claude Code (sandbox on) | `sandbox.network.allowedDomains` in settings: add the hosts above |
| Copilot cloud agent | Firewall allowlist excludes youtube/googlevideo/generativelanguage → add custom allowlist entries |
| Gemini CLI | Approve the skill activation prompt; network is normally open |

Cache dir not writable (sandbox) → automatic fallback to `$TMPDIR/youtube-understanding-<uid>`;
override with `--cache-dir DIR` or `YTU_CACHE_DIR`.

## YouTube blocks (exit 4, Blocked)

- "Sign in to confirm you're not a bot", HTTP 429/403: common on datacenter/CI/VPN IPs.
  1. Wait and retry later; avoid many requests in a row.
  2. Use cached results (`transcript`, `ask` reuse cache; don't pass `--force-refresh`).
  3. Last resort, user-supplied only: `--cookies cookies.txt` (Netscape format exported from a
     logged-in browser). Warn the user: automated use can get the account flagged.
- Translated caption tracks are never requested (they trigger 429 quickly).
- yt-dlp needs a JS runtime for YouTube; `deno` is bundled via the `yt-dlp[deno]` extra.
  `doctor` shows `binaries.deno`. Outdated yt-dlp is the most common cause of sudden failures —
  the pinned version is in `scripts/yt.py`; bump it there and in `pyproject.toml`.

## Video problems (exit 6)

| Message | Meaning / fix |
|---|---|
| Live stream | Unsupported (no duration, no usable captions). Retry after the stream ends and the replay is processed |
| Private / removed / unavailable | Nothing to do; ask the user for another link |
| Age-restricted | Captions usually still work unauthenticated; else `--cookies` (ban risk). youtube-transcript-api fails on these |
| Members-only | Needs `--cookies` from a member account |
| Gemini only accepts public videos | Unlisted/needs-auth video with Gemini strategy → `--strategy transcript` or `local` |

## No transcript

Order tried: yt-dlp captions → youtube-transcript-api → local ASR. If all fail:

- `--strategy transcript` never runs ASR — use `auto` or `local`.
- ASR installs on demand via `uvx` (first run downloads hundreds of MB to GBs and models):
  `parakeet-mlx` (Apple Silicon, 25 European languages), `mlx-whisper` (Apple Silicon, other
  languages), `faster-whisper` (Linux/Intel; pinned with `av<19`). `doctor` shows `asr_backend`.
  Timeout `YTU_TIMEOUT_ASR` (default 3600 s). Media is deleted after transcription.
- With a Gemini key, `analyze` can still work from the video itself; mark spoken quotes as
  single-source.
- `--lang xx` selects a different original-language track; auto-translated tracks are not used.

## Frames

- `ffmpeg` required (exit 3). Frames are grabbed by seeking the remote stream (no full download);
  each grab ≤ `YTU_TIMEOUT_FRAME` (90 s).
- Signed stream URLs expire; a retry fetches a fresh one. They are never cached or logged.
- `--frame-strategy keyframes|scene` may be unsupported (BadInput) — use `interval` or `--at`.
- Scene detection is unreliable for screencasts with webcam overlays; prefer `--at` at transcript cues.
- More than `--max-frames` (24) requested → capped.

## Timeouts

Defaults (seconds): metadata 60, subtitles 60, frame 90, section 600, asr 3600, gemini 900,
http 30. Override: `YTU_TIMEOUT_<NAME>=SECONDS` or `[timeouts]` in
`~/.config/youtube-understanding/config.toml`.

## Stale or wrong results

- `--force-refresh` ignores all caches for that run (metadata TTL is 24 h, the rest is permanent).
- Delete the cache: `rm -rf "${YTU_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/youtube-understanding}"`.
