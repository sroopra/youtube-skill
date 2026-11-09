"""Copy [project].dependencies and requires-python from pyproject.toml into the PEP 723 block of yt.py.

Dependabot updates pyproject.toml and uv.lock but not PEP 723 inline metadata
(dependabot/dependabot-core#11946), so .github/workflows/dependabot-sync.yml runs this on
Dependabot PRs. Standard library only: it runs before any dependency is installed.

    python3 .github/scripts/sync_pep723.py           # rewrite yt.py if it differs
    python3 .github/scripts/sync_pep723.py --check   # exit 1 if it differs
"""

import sys
import tomllib
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2] / "youtube-understanding"
SCRIPT = PROJECT / "scripts" / "yt.py"


def render_block(requires_python: str, deps: list[str]) -> list[str]:
    lines = ["# /// script", f'# requires-python = "{requires_python}"', "# dependencies = ["]
    lines += [f'#     "{d}",' for d in deps]
    return lines + ["# ]", "# ///"]


def main() -> int:
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    text = SCRIPT.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = lines.index("# /// script")
    end = lines.index("# ///", start + 1)
    new = lines[:start] + render_block(project["requires-python"], project["dependencies"]) + lines[end + 1:]
    new_text = "\n".join(new) + ("\n" if text.endswith("\n") else "")
    if new_text == text:
        print("yt.py PEP 723 block already matches pyproject.toml")
        return 0
    if "--check" in sys.argv[1:]:
        print("yt.py PEP 723 block differs from pyproject.toml; run .github/scripts/sync_pep723.py")
        return 1
    SCRIPT.write_text(new_text, encoding="utf-8")
    print("updated yt.py PEP 723 block from pyproject.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
