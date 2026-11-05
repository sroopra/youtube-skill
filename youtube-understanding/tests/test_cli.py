import json

import pytest

from pipeline_fakes import DEFAULT_LINES, install, transcript
from ytu import cli
from ytu.errors import MissingKey


def run(capsys, argv):
    rc = cli.main(argv)
    return rc, json.loads(capsys.readouterr().out)


def test_parser_new_flags():
    p = cli.build_parser()
    a = p.parse_args(["analyze", "jNQXAC9IVRw", "--window", "1:00-2:30", "--model", "m1", "--resolution", "high",
                      "--question", "why?", "--force", "--dry-run", "--max-cost-usd", "1"])
    assert a.window == (60.0, 150.0) and a.model == "m1" and a.force_refresh is True and a.dry_run is True
    a = p.parse_args(["frames", "x", "--at", "1:00,2:00", "--at", "5", "--cues"])
    assert a.at == [[60.0, 120.0], [5.0]] and a.cues
    a = p.parse_args(["package", "x", "--with-frames", "--merge-notes", "n.json", "--out", "o"])
    assert a.with_frames and str(a.merge_notes) == "n.json"
    a = p.parse_args(["transcript", "x", "--format", "text"])
    assert a.fmt == "text"
    with pytest.raises(SystemExit):
        p.parse_args(["analyze", "x", "--window", "nope"])


def test_model_pinned_first(tmp_path):
    a = cli.build_parser().parse_args(["inspect", "x", "--model", "gemini-2.5-flash-lite",
                                       "--cache-dir", str(tmp_path)])
    cfg = cli.config_from_args(a)
    assert cfg.models[0] == "gemini-2.5-flash-lite" and cfg.models.count("gemini-2.5-flash-lite") == 1


def test_window_and_start_conflict(capsys):
    rc, out = run(capsys, ["transcript", "jNQXAC9IVRw", "--window", "1-2", "--start", "1", "--end", "3"])
    assert rc == 2 and "either --window" in out["error"]["message"]


def test_transcript_command(monkeypatch, capsys, tmp_path):
    install(monkeypatch, tr=transcript(DEFAULT_LINES))
    rc, out = run(capsys, ["transcript", "https://youtu.be/jNQXAC9IVRw", "--format", "txt", "--window", "0:39-1:00",
                           "--cache-dir", str(tmp_path)])
    assert rc == 0 and out["ok"] and out["text"].startswith("[00:40] as you can see")


def test_analyze_no_key(monkeypatch, capsys, tmp_path):
    install(monkeypatch, tr=transcript(DEFAULT_LINES))
    rc, out = run(capsys, ["analyze", "jNQXAC9IVRw", "--mode", "tutorial", "--cache-dir", str(tmp_path)])
    assert rc == 0 and out["summary"] is None and out["digest"]["host_tasks"]


def test_analyze_strategy_gemini_without_key_exit_5(monkeypatch, capsys, tmp_path):
    install(monkeypatch, tr=transcript(DEFAULT_LINES))
    rc, out = run(capsys, ["analyze", "jNQXAC9IVRw", "--strategy", "gemini", "--cache-dir", str(tmp_path)])
    assert rc == MissingKey.exit_code and out["error"]["type"] == "MissingKey"
    assert "test-key" not in json.dumps(out)


def test_ask_uses_url_timestamp(monkeypatch, capsys, tmp_path):
    install(monkeypatch, tr=transcript(DEFAULT_LINES))
    rc, out = run(capsys, ["ask", "https://www.youtube.com/watch?v=jNQXAC9IVRw&t=2m", "what happens?",
                           "--before", "5", "--after", "10", "--cache-dir", str(tmp_path)])
    assert rc == 0 and out["windows"][0]["start"] == 115 and out["mode"] == "transcript-only"


def test_frames_cues_and_at(monkeypatch, capsys, tmp_path):
    install(monkeypatch, tr=transcript(DEFAULT_LINES))
    seen = {}

    def fake_run_frames(cfg, cache, ref, **kw):
        seen.update(kw)
        return {"video_id": ref.video_id, "frames": []}
    from ytu.providers import frames
    monkeypatch.setattr(frames, "run_frames", fake_run_frames)
    rc, out = run(capsys, ["frames", "jNQXAC9IVRw", "--cues", "--at", "10", "--cache-dir", str(tmp_path)])
    assert rc == 0 and seen["at"] == [10.0, 42.0] and seen["start"] is None
    rc, out = run(capsys, ["frames", "jNQXAC9IVRw", "--window", "0-30", "--every", "5", "--cache-dir", str(tmp_path)])
    assert rc == 0 and (seen["start"], seen["end"], seen["every_s"]) == (0.0, 30.0, 5.0)
    rc, out = run(capsys, ["frames", "jNQXAC9IVRw", "--at", "1", "--window", "0-30", "--cache-dir", str(tmp_path)])
    assert rc == 2


def test_package_refuses_skill_dir(monkeypatch, capsys, tmp_path):
    from ytu.pipeline import package
    install(monkeypatch, tr=transcript(DEFAULT_LINES))
    rc, out = run(capsys, ["package", "jNQXAC9IVRw", "--out", str(package.SKILL_DIR / "pkg"),
                           "--cache-dir", str(tmp_path)])
    assert rc == 2 and "skill directory" in out["error"]["message"]
    rc, out = run(capsys, ["package", "jNQXAC9IVRw", "--out", str(tmp_path / "pkg"), "--cache-dir", str(tmp_path)])
    assert rc == 0 and out["files"]["knowledge.md"] == "knowledge.md"


def test_argparse_error_is_json(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["ask", "jNQXAC9IVRw", "q", "--at", "banana"])
    cap = capsys.readouterr()
    out = json.loads(cap.out)
    assert e.value.code == 2 and out["ok"] is False and out["error"]["type"] == "BadInput"
    assert "banana" in out["error"]["message"] and "--help" in out["error"]["remediation"]
    assert "usage:" in cap.err


def test_output_compact_unless_pretty():
    assert cli.dumps({"a": [1, 2]}) == '{"a":[1,2]}'
    assert "\n  " in cli.dumps({"a": [1, 2]}, pretty=True)
