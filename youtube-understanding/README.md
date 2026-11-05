# youtube-understanding

This folder is the skill. Your agent reads `SKILL.md`.

Install, Gemini key setup and usage: see the [README in the repo root](../README.md).

| Path | Content |
|---|---|
| `SKILL.md` | Instructions for the agent |
| `scripts/yt.py` | The command line tool, run with `uv run` |
| `scripts/ytu/` | The code |
| `references/` | Details the agent reads when it needs them |
| `tests/` | Unit tests, no network |

Run the tests:

```bash
uv run pytest -q
uvx --from skills-ref==0.1.1 agentskills validate "$PWD"
```
