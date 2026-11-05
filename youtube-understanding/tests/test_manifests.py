"""The PEP 723 block in scripts/yt.py (what users install) must match pyproject.toml (what tests use)."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pep723_block() -> dict:
    lines = (ROOT / "scripts" / "yt.py").read_text(encoding="utf-8").splitlines()
    start = lines.index("# /// script")
    end = lines.index("# ///", start + 1)
    body = "\n".join(line.removeprefix("#").removeprefix(" ") for line in lines[start + 1 : end])
    return tomllib.loads(body)


def _project_table() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_pep723_dependencies_match_pyproject():
    assert sorted(_pep723_block()["dependencies"]) == sorted(_project_table()["dependencies"])


def test_pep723_requires_python_matches_pyproject():
    assert _pep723_block()["requires-python"] == _project_table()["requires-python"]
