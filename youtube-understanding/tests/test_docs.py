"""Docs consistency: SKILL.md frontmatter (agentskills spec), links, CLI surface, evidence schema."""

import dataclasses
import json
import re
from pathlib import Path

import pytest

from ytu import cli
from ytu.models import EvidenceItem, Evidence, Frame, Gap

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "SKILL.md"
SPEC_KEYS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}


def _frontmatter() -> tuple[dict, str]:
    text = SKILL.read_text(encoding="utf-8")
    _, fm, body = text.split("---", 2)
    out = {}
    for line in fm.strip().splitlines():
        if line.startswith(" "):
            continue  # nested metadata values
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out, body


def test_frontmatter_spec_keys_only():
    fm, _ = _frontmatter()
    assert set(fm) <= SPEC_KEYS
    assert fm["name"] == "youtube-understanding" == ROOT.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fm["name"])
    assert fm["license"] == "MIT"


def test_description_and_compatibility_quoted_and_bounded():
    fm, _ = _frontmatter()
    for key, limit in (("description", 1024), ("compatibility", 500)):
        v = fm[key]
        assert v.startswith('"') and v.endswith('"'), f"{key} must be quoted (colons in YAML)"
        assert 1 <= len(v[1:-1]) <= limit
    d = fm["description"].lower()
    for word in ("youtube", "summarize", "tutorial", "compare", "timestamp", "code"):
        assert word in d


def test_body_concise():
    _, body = _frontmatter()
    assert len(body.splitlines()) < 200


def _md_links(path: Path) -> list[str]:
    return re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("doc", ["SKILL.md", "README.md", "references/workflows.md",
                                 "references/evidence-schema.md", "references/gemini.md",
                                 "references/troubleshooting.md"])
def test_relative_links_resolve(doc):
    path = ROOT / doc
    for link in _md_links(path):
        if link.startswith(("http://", "https://")) or not re.search(r"\.(md|json|ya?ml|py)$", link):
            continue  # external or template placeholder like (url)
        assert (path.parent / link).exists(), f"{doc}: broken link {link}"


def _subcommands() -> dict:
    parser = cli.build_parser()
    action = next(a for a in parser._actions if a.dest == "command")
    return action.choices


def test_skill_commands_match_cli():
    subs = _subcommands()
    text = SKILL.read_text(encoding="utf-8")
    block = text.split("## Commands", 1)[1].split("```", 2)[1]
    for line in block.splitlines():
        m = re.match(r"uv run \$S (\w+)(.*)", line.strip())
        if not m:
            continue
        cmd, rest = m.groups()
        assert cmd in subs, cmd
        opts = {o for a in subs[cmd]._actions for o in a.option_strings}
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", rest):
            assert flag in opts, f"{cmd} {flag}"


def test_common_flags_in_skill_exist():
    opts = {o for a in _subcommands()["analyze"]._actions for o in a.option_strings}
    text = SKILL.read_text(encoding="utf-8")
    common = text.split("Common flags:", 1)[1].split("\n\n", 1)[0]
    for flag in re.findall(r"(--[a-z][a-z-]*)", common):
        assert flag in opts, flag


def _schema() -> dict:
    return json.loads((ROOT / "assets" / "evidence.schema.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name,cls", [("item", EvidenceItem), ("frame", Frame), ("gap", Gap)])
def test_schema_matches_models(name, cls):
    sub = _schema()["$defs"][name]
    fields = {f.name for f in dataclasses.fields(cls)}
    assert set(sub["properties"]) == fields
    assert set(sub["required"]) <= fields


def test_schema_top_level_matches_evidence():
    s = _schema()
    assert set(s["properties"]) == {f.name for f in dataclasses.fields(Evidence)}


def test_schema_patterns_accept_model_output():
    from ytu.urls import fmt_ts, ts_link
    from ytu.models import make_id
    item = _schema()["$defs"]["item"]["properties"]
    assert re.search(item["timestamp"]["pattern"], fmt_ts(754.2))
    assert re.search(item["timestamp"]["pattern"], fmt_ts(3723))
    assert re.search(item["url"]["pattern"], ts_link("jNQXAC9IVRw", 754.2))
    assert re.search(item["id"]["pattern"], make_id("E", 1))
    assert re.search(item["transcript_segment_ids"]["items"]["pattern"], make_id("s", 12))
