# youtube-understanding

[![CI](https://github.com/sroopra/youtube-skill/actions/workflows/ci.yml/badge.svg)](https://github.com/sroopra/youtube-skill/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Give your coding agent a YouTube link. It reads the video and answers with timestamps.

Works in every agent that reads the [Agent Skills standard](https://agentskills.io): Claude Code, Codex, Gemini CLI, GitHub Copilot, pi, Cursor, OpenCode. MIT.

## What it does

- Summarizes a video. Every point links to the second in the video where it is said.
- Pulls the commands, code and steps out of a tutorial.
- Answers a question about one part of a video, for example "what happens at 25:00?".
- Writes notes to disk that another agent can use later.
- Takes screenshots, so the agent can read code and diagrams on screen.

It does not invent content. If the video does not show or say something, the notes mark it as a guess or leave it out.

## Example

You type:

> Watch https://youtu.be/rfscVS0vtbw from 52:39 to 58:27 and give me the code.

The agent replies (shortened from [`examples/rfscVS0vtbw/knowledge.md`](examples/rfscVS0vtbw/knowledge.md), see [`examples/README.md`](examples/README.md) for why the transcript and screenshots are not in the repo):

> The presenter reads two numbers with `input()` and adds them. The first run prints `58.3` for 5 + 8.3, because `input()` returns text ([54:40](https://www.youtube.com/watch?v=rfscVS0vtbw&t=3280s)). He fixes it with `float()` ([57:25](https://www.youtube.com/watch?v=rfscVS0vtbw&t=3445s)):
>
> ```python
> num1 = input("Enter a number: ")
> num2 = input("Enter another number: ")
> result = float(num1) + float(num2)
> print(result)
> ```
>
> Python version: not readable on screen (inferred).

## Install

```bash
npx skills add sroopra/youtube-skill
```

The installer asks which agents to install for and whether to install for this project or for your user. It works with Claude Code, Codex, Gemini CLI, GitHub Copilot, pi, Cursor, OpenCode and many more. It needs Node 22.20 or newer. Restart your agent afterwards.

The skill also needs two programs that no installer adds for you:

| Need | Why | Get it |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Installs Python and all libraries on the first run | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| ffmpeg | Takes screenshots | `brew install ffmpeg` or `apt install ffmpeg` |

macOS and Linux only. Windows is not tested.

Check that everything works. Ask your agent: "Run the youtube-understanding doctor check." Or run it yourself from the folder the skill was installed to:

```bash
uv run <skill-folder>/scripts/yt.py doctor --network
```

The first run downloads the libraries. It takes about 30 seconds.

Update: `npx skills update`. Remove: `npx skills remove`.

### Other ways to install

Use one of these instead of `npx`, not as well. Two installs make the agent load the skill twice.

| Agent | Command |
|---|---|
| Claude Code (plugin) | `claude plugin marketplace add sroopra/youtube-skill`, then `claude plugin install youtube-understanding@youtube-skill` |
| pi (package) | `pi install git:github.com/sroopra/youtube-skill` |
| Gemini CLI | `gemini skills install https://github.com/sroopra/youtube-skill --path youtube-understanding --consent` |
| Any agent, without Node | `gh skill install sroopra/youtube-skill` (gh 2.90 or newer) |

Update the Claude Code plugin with `claude plugin marketplace update youtube-skill`, then `claude plugin update youtube-understanding@youtube-skill`. Update the pi package with `pi update`.

### Working on the skill itself

Clone the repo and link the folder, so your edits are live:

```bash
ln -s "$PWD/youtube-understanding" ~/.claude/skills/youtube-understanding   # Claude Code
ln -s "$PWD/youtube-understanding" ~/.agents/skills/youtube-understanding   # all other agents
```

### Codex network access

Codex blocks network access by default. Add this to `~/.codex/config.toml`:

```toml
[sandbox_workspace_write]
network_access = true
```

## Gemini key (optional)

Without a key, the skill reads the transcript. Your agent then does the thinking, at no extra cost.

With a key, Google Gemini also watches the video. Use a key when the video shows code, slides or diagrams that nobody reads out loud.

| | No key | With key |
|---|---|---|
| Spoken content | yes | yes |
| Code and slides on screen | screenshots only, your agent reads them | yes |
| Cost | free | about $0.01 to $0.30 for a video up to 1 hour |
| Private videos | no | no |

To add a key:

1. Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Open the file `~/.config/youtube-understanding/.env` in your editor.
3. Add one line: `GEMINI_API_KEY=your-key`
4. Run `chmod 600 ~/.config/youtube-understanding/.env`

Type the key in an editor. If you paste it into a shell command, it stays in your shell history.

The skill also reads `GEMINI_API_KEY` or `GOOGLE_API_KEY` from the environment. The environment wins over the file.

The skill never prints the key, writes it to the cache, or puts it in the notes. `doctor` only shows whether it found a key.

## Use it

Talk to your agent. You do not have to run commands.

- "Summarize this video: https://youtu.be/..."
- "Extract all code examples from this tutorial."
- "What does the speaker say about Kafka partitions at around minute 25?"
- "Compare these two tutorials: URL1 URL2"
- "Make reusable notes from this video in ./notes"

The agent calls `scripts/yt.py`. You can call it yourself too:

| Command | What you get | Time |
|---|---|---|
| `yt.py transcript URL` | Transcript with timestamps | 3 to 5 s |
| `yt.py analyze URL --dry-run` | The plan and the cost, nothing is paid | 1 s |
| `yt.py analyze URL` | Summary data, Gemini if a key exists | 10 to 40 s |
| `yt.py ask URL "question" --at 25:00` | Answer from the part near 25:00 only | 10 to 20 s |
| `yt.py frames URL --at 12:34` | A screenshot as a JPEG file | 5 to 20 s |
| `yt.py package URL --with-frames --out ./notes` | Full notes folder | 1 to 3 min |

Write each command as `uv run <skill-folder>/scripts/yt.py ...`. Output is JSON.

Add `--window 10:00-25:00` to work on one part of a long video. It is faster and cheaper.

## The notes folder

`package` writes this:

```
notes/
  knowledge.md     summary, concepts, code, steps, with timestamp links
  evidence.json    every claim, its source and how sure it is
  transcript.srt   the original captions
  transcript.json  the same, with segment ids
  metadata.json    title, channel, length, cost
  screenshots/     JPEGs named by time, t3280000.jpg = 54:40
```

Each claim has a status:

| Status | Meaning |
|---|---|
| verified | Checked against the transcript or a screenshot |
| corroborated | Two sources agree, for example Gemini and the captions |
| single-source | Only one source says it |
| contradicted | Sources disagree. The claim is kept and marked. |

## Cost and safety limits

- The skill never makes a Gemini call that it estimates above $0.50. Change this with `--max-cost-usd`.
- If a video is too long for the limit, the default mode falls back to the transcript and tells you.
- `--dry-run` shows the cost. It makes no paid call.
- Every result is cached. A second run of the same request is free and takes under 1 second.
- Add `--force-refresh` to skip the cache.
- The skill treats video text as data. It tells the agent never to run commands that a video shows or says.

## Known limits

- YouTube can block cloud servers, CI machines and VPNs. If you see "Sign in to confirm you're not a bot", run it from your own computer.
- Live streams and private videos do not work.
- Auto captions have no speaker names and often get technical words wrong.
- On some videos a screenshot takes about 17 s, because YouTube refuses direct seeking.
- Gemini sometimes returns a timestamp outside the part you asked for. The skill drops those items and tells you how many.
- A video without captions needs local speech-to-text. It downloads several hundred MB on first use and is not tested end to end.

Full test log: [`docs/test-report-2026-11-05.md`](docs/test-report-2026-11-05.md). Research and design choices: [`docs/research.md`](docs/research.md).

## Settings

Defaults work. Change them with environment variables or `~/.config/youtube-understanding/config.toml`.

| Variable | Default | Effect |
|---|---|---|
| `YTU_GEMINI_MODEL` | `gemini-3.8-flash` | First Gemini model to try |
| `YTU_MAX_COST_USD` | `0.50` | Cost limit per run |
| `YTU_CACHE_DIR` | `~/.cache/youtube-understanding` | Where results are cached |
| `YTU_ENV_FILE` | `~/.config/youtube-understanding/.env` | Where the key file is |

## Development

```bash
cd youtube-understanding
uv run --locked pytest
uvx --from skills-ref==0.1.1 agentskills validate "$PWD"
```

Tests use no network and no paid calls. CI runs both commands, plus `uv run --script scripts/yt.py --help` to check the dependency block in `yt.py`, on Ubuntu and macOS, with Python 3.11 and 3.13, for every push to `main` and every pull request (`.github/workflows/ci.yml`).

## License

MIT, see [`LICENSE`](LICENSE).
