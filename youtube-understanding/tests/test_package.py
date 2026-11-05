import dataclasses
import json
from pathlib import Path

import pytest

from pipeline_fakes import DEFAULT_LINES, FakeFrames, FakeGemini, install, meta, transcript
from test_analyze import GEMINI_DATA
from ytu.errors import BadInput
from ytu.models import Chapter, Segment
from ytu.pipeline import package
from ytu.urls import parse_url

REF = parse_url("jNQXAC9IVRw")


def test_srt_time_and_srt():
    assert package.srt_time(754.2) == "00:12:34,200"
    assert package.srt_time(3723.0005) == "01:02:03,001" or package.srt_time(3723.0005) == "01:02:03,000"
    assert package.srt_time(-1) == "00:00:00,000"
    srt = package.segments_to_srt([Segment("s0001", 0, 0, "a"), Segment("s0002", 1, 2, "  "),
                                   Segment("s0003", 2.5, 4, "b\n c")])
    assert srt == "1\n00:00:00,000 --> 00:00:00,001\na\n\n2\n00:00:02,500 --> 00:00:04,000\nb\nc\n"


def test_resolve_out_dir_safety(cfg, tmp_path):
    with pytest.raises(BadInput):
        package.resolve_out_dir(cfg, "x", package.SKILL_DIR / "out")
    with pytest.raises(BadInput):
        package.resolve_out_dir(cfg, "x", package.SKILL_DIR / "scripts" / ".." / "x")
    with pytest.raises(BadInput):
        package.resolve_out_dir(cfg, "x", Path("/"))
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(BadInput):
        package.resolve_out_dir(cfg, "x", f)
    link = tmp_path / "link"
    link.symlink_to(package.SKILL_DIR)
    with pytest.raises(BadInput):
        package.resolve_out_dir(cfg, "x", link / "pkg")
    assert package.resolve_out_dir(cfg, "vid", None) == (cfg.out_root / "vid").resolve()


def _files(out):
    d = Path(out["out_dir"])
    return {n: d / p for n, p in out["files"].items()}


def test_package_no_key_skeleton(monkeypatch, cfg, cache, tmp_path):
    install(monkeypatch, tr=transcript(DEFAULT_LINES, generated=True))
    out = package.run_package(cfg, cache, REF, out_dir=tmp_path / "pkg")
    f = _files(out)
    assert set(f) >= {"metadata.json", "transcript.srt", "transcript.json", "knowledge.md", "evidence.json"}
    md = f["knowledge.md"].read_text()
    for h in ("## Source & metadata", "## Executive summary", "## Main concepts", "## Technical details",
              "## Visual observations", "## Architecture & implementation", "## Code", "## Commands",
              "## Verified vs uncertain", "## Timestamps", "## Actionable guidance", "## HOST-AGENT TODO"):
        assert h in md
    assert "Transcript-derived outline" in md and "Demo \\<b\\>video\\</b\\>" in md  # title escaped
    assert out["summary_present"] is False and "merge-notes" in out["next"]
    meta_json = json.loads(f["metadata.json"].read_text())
    assert meta_json["provenance"]["transcript_source"] == "ytdlp" and meta_json["provenance"]["is_generated"]
    assert "description" not in meta_json["video"]
    assert json.loads(f["transcript.json"].read_text())[0]["id"] == "s0001"
    assert f["transcript.srt"].read_text().startswith("1\n00:00:00,000 --> 00:00:04,000\nwelcome")
    assert out["counts"] == {"segments": 6, "evidence": 0, "frames": 0}


def test_package_dry_run_writes_nothing(monkeypatch, cfg, cache, tmp_path):
    install(monkeypatch, tr=transcript(DEFAULT_LINES))
    out = package.run_package(dataclasses.replace(cfg, dry_run=True), cache, REF, out_dir=tmp_path / "pkg")
    assert out["dry_run"] and not (tmp_path / "pkg").exists()


def test_package_gemini_with_frames(monkeypatch, cfg, cache, tmp_path):
    fr = FakeFrames(tmp_path)
    install(monkeypatch, m=meta(chapters=[Chapter(0, 300, "All")]), tr=transcript(DEFAULT_LINES), key=True,
            gemini=FakeGemini([GEMINI_DATA]), frames=fr)
    out = package.run_package(cfg, cache, REF, out_dir=tmp_path / "pkg", mode="tutorial", with_frames=True)
    f = _files(out)
    evj = json.loads(f["evidence.json"].read_text())
    assert evj["frames"] and all(fr_["path"].startswith("screenshots/t") for fr_ in evj["frames"])
    assert all((Path(out["out_dir"]) / x["path"]).is_file() for x in evj["frames"])
    code = [i for i in evj["items"] if i["kind"] == "code"]
    # time-proximity frames are linked but unchecked: code stays unverified (regression)
    assert code and all(i["frame_ids"] and i["status"] == "unverified" and package.AUTO_FRAME_NOTE in i["notes"]
                        for i in code)
    md = f["knowledge.md"].read_text()
    assert "frames (auto-attached, unchecked): F0" in md and "UNVERIFIED: no frame checked" in md
    assert "A docker compose tutorial." in md and "```yaml" in md and "[E0" in md
    assert "INJECTION-SUSPECT" in md and "HOST-AGENT TODO" not in md
    assert "![F0001" in md and "**Chapters**" in md
    prov = json.loads(f["metadata.json"].read_text())["provenance"]
    assert "gemini-interactions" in prov["providers_used"] and "ffmpeg-remote" in prov["providers_used"]
    assert len(fr.grabs) <= cfg.max_frames


def test_package_frame_failures_become_gaps(monkeypatch, cfg, cache, tmp_path):
    fr = FakeFrames(tmp_path, fail=True)
    install(monkeypatch, tr=transcript(DEFAULT_LINES), frames=fr)
    out = package.run_package(cfg, cache, REF, out_dir=tmp_path / "pkg", with_frames=True)
    assert len(fr.grabs) == 3 and any("stopped" in w for w in out["warnings"])
    evj = json.loads(_files(out)["evidence.json"].read_text())
    assert any(g["reason"].startswith("frame unavailable") for g in evj["gaps"])


def test_package_unverified_code_label(monkeypatch, cfg, cache, tmp_path):
    install(monkeypatch, tr=transcript(DEFAULT_LINES), key=True, gemini=FakeGemini([GEMINI_DATA]))
    out = package.run_package(cfg, cache, REF, out_dir=tmp_path / "pkg", mode="tutorial")
    md = _files(out)["knowledge.md"].read_text()
    assert "UNVERIFIED: no frame checked" in md


def test_merge_notes(monkeypatch, cfg, cache, tmp_path):
    install(monkeypatch, tr=transcript(DEFAULT_LINES))
    shot = cfg.cache_dir / "frames" / "ab" / "t0042000.jpg"
    shot.parent.mkdir(parents=True)
    shot.write_bytes(b"\xff\xd8\xff\xe0data")
    outside = tmp_path / "evil.jpg"
    outside.write_bytes(b"\xff\xd8\xff\xe0data")
    notes = {"summary": "Host summary of a compose tutorial.", "items": [
        {"claim": "Install deps with pip", "kind": "speech", "t_start": "00:05",
         "quote": "we install the dependencies with pip", "section": "steps", "demonstrated": True},
        {"claim": "Compose file defines a db service", "kind": "code", "t_start": "00:42",
         "frames": [str(shot)], "section": "code", "language": "yaml"},
        {"claim": "fabricated quote", "kind": "speech", "t_start": "00:05", "quote": "we use kubernetes and helm"},
        {"claim": "too late", "kind": "speech", "t_start": "99:00"},
        {"claim": "far from speech", "kind": "inference", "t_start": "02:45"},
        {"claim": "bad frame", "kind": "ui", "t_start": "00:41", "frames": [str(outside)]},
        {"claim": "bad kind", "kind": "dance", "t_start": 1},
        "not an object",
    ]}
    p = tmp_path / "notes.json"
    p.write_text(json.dumps(notes))
    out = package.run_package(cfg, cache, REF, out_dir=tmp_path / "pkg", mode="tutorial", merge_notes_path=p)
    rep = out["merge"]
    assert rep["accepted"] == 2 and rep["summary"] is True
    assert sorted(r["index"] for r in rep["rejected"]) == [2, 3, 4, 5, 6, 7]
    f = _files(out)
    evj = json.loads(f["evidence.json"].read_text())
    by_claim = {i["claim"]: i for i in evj["items"]}
    assert by_claim["Install deps with pip"]["status"] == "verified"
    assert by_claim["Install deps with pip"]["transcript_segment_ids"] == ["s0002"]
    code = by_claim["Compose file defines a db service"]
    assert code["frame_ids"] == ["F0001"] and "host-vision" in code["sources"] and code["status"] == "single-source"
    assert evj["frames"][0]["path"] == "screenshots/t0042000.jpg" and evj["frames"][0]["abs_ts"] == 42.0
    md = f["knowledge.md"].read_text()
    assert "Host summary of a compose tutorial." in md and "*Source: host-agent*" in md
    assert "(demonstrated, verified)" in md and "```yaml" in md
    assert "host-agent" in json.loads(f["metadata.json"].read_text())["provenance"]["providers_used"]


def test_merge_notes_contradicts(monkeypatch, cfg, cache, tmp_path):
    install(monkeypatch, tr=transcript(DEFAULT_LINES), key=True, gemini=FakeGemini([GEMINI_DATA, GEMINI_DATA]))
    first = package.run_package(cfg, cache, REF, out_dir=tmp_path / "pkg", mode="tutorial")
    items = json.loads(_files(first)["evidence.json"].read_text())["items"]
    target = next(i for i in items if i["claim"] == "database is postgres")
    p = tmp_path / "notes.json"
    p.write_text(json.dumps({"items": [{"contradicts": target["id"], "claim": "frame shows MySQL, not Postgres"},
                                       {"contradicts": "E9999", "claim": "x"}]}))
    out = package.run_package(cfg, cache, REF, out_dir=tmp_path / "pkg", mode="tutorial", merge_notes_path=p)
    assert out["merge"]["accepted"] == 1 and [r["index"] for r in out["merge"]["rejected"]] == [1]
    evj = json.loads(_files(out)["evidence.json"].read_text())
    it = next(i for i in evj["items"] if i["claim"] == "database is postgres")  # kept, not deleted
    assert it["status"] == "contradicted" and "frame shows MySQL" in it["notes"] and "host-agent" in it["sources"]
    assert any("contradicted" in u for u in evj.get("uncertainties", []) or []) or \
        "contradicted" in _files(out)["knowledge.md"].read_text()


def test_load_notes_errors(tmp_path):
    p = tmp_path / "n.json"
    p.write_text("{nope")
    with pytest.raises(BadInput):
        package.load_notes(p)
    p.write_text("[]")
    with pytest.raises(BadInput):
        package.load_notes(p)
    with pytest.raises(BadInput):
        package.load_notes(tmp_path / "missing.json")


def test_stale_screenshots_removed_only_tool_named(monkeypatch, cfg, cache, tmp_path):
    install(monkeypatch, tr=transcript(DEFAULT_LINES))
    shots = tmp_path / "pkg" / "screenshots"
    shots.mkdir(parents=True)
    (shots / "t0001000.jpg").write_bytes(b"\xff\xd8\xff")
    (shots / "mine.jpg").write_bytes(b"keep")
    package.run_package(cfg, cache, REF, out_dir=tmp_path / "pkg")
    assert not (shots / "t0001000.jpg").exists() and (shots / "mine.jpg").exists()


def test_entries_drops_verbatim_repeats():
    lines = package._entries([{"text": "Do X", "evidence_ids": []}, {"text": "Do X", "evidence_ids": []},
                              {"text": "Do Y", "evidence_ids": []}], {}, set())
    assert len(lines) == 2


def test_host_note_confirming_model_item_is_folded_not_duplicated(cfg, tmp_path):
    from ytu.models import Evidence, Frame
    from ytu.pipeline import evidence as ev
    out = tmp_path / "pkg"
    (out / "screenshots").mkdir(parents=True)
    jpg = b"\xff\xd8\xff\xe0same-image"
    (out / "screenshots" / "t0042000.jpg").write_bytes(jpg)
    import hashlib
    shot = Frame(id="F0001", path="/cache/frames/ab/x.jpg", abs_ts=42.0, selection_reason="gemini-event",
                 sha256=hashlib.sha256(jpg).hexdigest(), width=1, height=1)
    model_item = ev.new_item("abc", "x = 1\nprint(x)", "code", 42.0, sources=["gemini:static:m"],
                             frame_ids=["F0001"], status="single-source")
    evidence = Evidence(video_id="abc", items=[model_item], frames=[shot])
    analysis = {"tutorial": {"code": [{"text": model_item.claim, "evidence_ids": [model_item]}]}}
    notes = {"items": [{"claim": "x = 1\n  print(x)", "kind": "code", "t_start": 43,
                        "frames": ["screenshots/t0042000.jpg"], "section": "code"}]}
    rep = package.merge_notes(cfg, meta(), None, analysis, evidence, notes, out)
    assert rep["accepted"] == 1 and rep["rejected"] == []
    assert len(evidence.items) == 1 and len(evidence.frames) == 1      # no duplicate item or frame
    assert "host-agent" in model_item.sources and model_item.notes.endswith("confirmed by host-agent note")
    assert len(analysis["tutorial"]["code"]) == 1


def test_inference_is_never_labelled_demonstrated():
    from ytu.models import EvidenceItem
    it = EvidenceItem(id="E0001", claim="Python 3.6", kind="inference", t_start=1.0, t_end=2.0,
                      timestamp="00:01", url="u", status="single-source", demonstrated=True)
    tag = package._tag(it)
    assert "demonstrated" not in tag
    assert "inferred" in tag
    shown = dataclasses.replace(it, kind="code")
    assert "demonstrated" in package._tag(shown)
