import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ytu import cli
from ytu.errors import YTUError
from ytu.models import (AnalyzeResult, CostEstimate, Evidence, EvidenceItem, Frame, Segment, TrackRef,
                        Transcript, VideoMeta, Window, Word, make_id)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "yt.py"

MODULES = ["ytu", "ytu.cli", "ytu.models", "ytu.errors", "ytu.urls", "ytu.config", "ytu.cache", "ytu.proc",
           "ytu.util.sanitize", "ytu.util.paths", "ytu.util.retry", "ytu.util.log", "ytu.providers.base",
           "ytu.providers.ytdlp_meta", "ytu.providers.captions", "ytu.providers.gemini", "ytu.providers.frames",
           "ytu.providers.asr", "ytu.pipeline.strategy", "ytu.pipeline.analyze", "ytu.pipeline.ask",
           "ytu.pipeline.package", "ytu.pipeline.evidence", "ytu.pipeline.doctor"]


@pytest.mark.parametrize("mod", MODULES)
def test_import(mod):
    importlib.import_module(mod)


def test_gemini_sdk_not_imported_by_scaffold():
    """Fresh interpreter: other tests may legitimately import the SDK (provider.available())."""
    code = ("import importlib, sys; " + "; ".join(f"importlib.import_module({m!r})" for m in MODULES)
            + "; print('google.genai' in sys.modules)")
    src = str(Path(__file__).resolve().parents[1] / "scripts")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PYTHONPATH": src})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


def test_registry_classes_exist():
    from ytu.providers import base
    for table in (base.TRANSCRIPT_PROVIDERS, base.MULTIMODAL_PROVIDERS, base.FRAME_PROVIDERS):
        for name in table:
            assert isinstance(base._load(table, name), type)


def test_script_help():
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    for c in ("doctor", "inspect", "transcript", "analyze", "ask", "frames", "package"):
        assert c in r.stdout


def test_cli_bad_url_json_error(capsys):
    rc = cli.main(["inspect", "https://vimeo.com/1"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 2 and out["ok"] is False and out["error"]["type"] == "BadInput"


def test_cli_window_validation(capsys):
    rc = cli.main(["transcript", "jNQXAC9IVRw", "--start", "10"])
    assert rc == 2
    rc = cli.main(["frames", "jNQXAC9IVRw"])
    assert rc == 2


def test_cli_maps_ytu_error(monkeypatch, capsys):
    def boom(args):
        from ytu.errors import OverBudget
        raise OverBudget("too expensive")
    monkeypatch.setattr(cli, "dispatch", boom)
    assert cli.main(["inspect", "jNQXAC9IVRw"]) == 7
    assert json.loads(capsys.readouterr().out)["error"]["remediation"]


def test_error_codes():
    from ytu import errors as e
    codes = {e.BadInput: 2, e.MissingDependency: 3, e.NetworkError: 4, e.Blocked: 4, e.SubprocessTimeout: 4,
             e.AuthError: 5, e.MissingKey: 5, e.VideoUnavailable: 6, e.LiveStream: 6, e.OverBudget: 7,
             e.InternalError: 1, e.ProcessFailed: 1}
    for cls, code in codes.items():
        err = cls("m")
        assert isinstance(err, YTUError) and err.exit_code == code and err.remediation
        assert err.to_dict()["exit_code"] == code


def test_models_roundtrip():
    tr = TrackRef(source="ytdlp", code="en-orig", lang="en", kind="auto", formats=["json3"])
    t = Transcript(video_id="x", segments=[Segment(make_id("s", 1), 0.0, 1.5, "hi", [Word("hi", 0.0, 1.5)])],
                   source="ytdlp", track=tr, is_generated=True, language="en")
    t2 = Transcript.from_dict(json.loads(json.dumps(t.to_dict())))
    assert t2 == t and isinstance(t2.segments[0].words[0], Word) and t2.track.is_generated
    m = VideoMeta(video_id="x", title="t", duration=None, tracks=[tr])
    assert VideoMeta.from_dict({**m.to_dict(), "unknown": 1}) == m
    ev = Evidence(video_id="x", items=[EvidenceItem("E0001", "c", "speech", 1, 2, "00:01", "u")],
                  frames=[Frame("F0001", "p", 1.0, "interval", "h", 1, 1)])
    assert Evidence.from_dict(ev.to_dict()) == ev
    r = AnalyzeResult(data={"a": 1}, raw_text="", usage={}, model="m", mode="static", agentic_confirmed=False,
                      request_hash="h", cost_estimate=CostEstimate("m", 1, 1, 0.1))
    assert AnalyzeResult.from_dict(r.to_dict()).cost_estimate.usd == 0.1
    assert make_id("E", 12) == "E0012" and make_id("F", 12345) == "F12345"
    with pytest.raises(ValueError):
        Window(5, 5)
