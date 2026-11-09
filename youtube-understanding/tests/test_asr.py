from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ytu import proc
from ytu.errors import MissingDependency, ProcessFailed, VideoUnavailable
from ytu.models import TrackRef, VideoMeta, Window
from ytu.providers import asr as asr_mod
from ytu.providers.asr import AsrProvider, select_backend, worker_segments

sys.path.insert(0, str(Path(asr_mod.__file__).resolve().parent.parent))
import asr_worker  # noqa: E402  (standalone script; pure helpers only)

VID = "jNQXAC9IVRw"
WORKER_OUT = {"engine": "faster-whisper", "model": "small", "language": "en", "segments": [
    {"start": 4.0, "end": 6.0, "text": " cool thing ", "words": [{"w": "cool", "start": 4.0, "end": 4.5, "conf": 0.9}]},
    {"start": 0.5, "end": 3.0, "text": "All right, so here we are", "words": []},
    {"start": 7.0, "end": 7.5, "text": "  ", "words": []},
]}


@pytest.mark.parametrize("system,machine,lang,want", [
    ("darwin", "arm64", "en", "parakeet"),
    ("darwin", "arm64", None, "parakeet"),
    ("darwin", "arm64", "de-DE", "parakeet"),
    ("darwin", "arm64", "ja", "mlx-whisper"),
    ("darwin", "x86_64", "en", "faster-whisper"),
    ("linux", "x86_64", "en", "faster-whisper"),
    ("linux", "aarch64", "en", "faster-whisper"),
])
def test_select_backend(system, machine, lang, want):
    assert select_backend(system, machine, lang) == want


def test_worker_segments_offsets_sorts_and_drops_empty():
    segs = worker_segments(WORKER_OUT, offset=100.0)
    assert [(s.id, s.start, s.end, s.text) for s in segs] == [
        ("s0001", 100.5, 103.0, "All right, so here we are"),
        ("s0002", 104.0, 106.0, "cool thing")]
    assert segs[0].words is None and segs[1].words[0].start == 104.0


def test_merge_parakeet_tokens():
    toks = [{"text": " P", "start": 0.0, "end": 0.1, "confidence": 0.9},
            {"text": "yt", "start": 0.1, "end": 0.2, "confidence": 0.8},
            {"text": "hon", "start": 0.2, "end": 0.3, "confidence": 0.95},
            {"text": " is", "start": 0.4, "end": 0.5, "confidence": 1.0}]
    assert asr_worker.merge_tokens(toks) == [
        {"w": "Python", "start": 0.0, "end": 0.3, "conf": 0.8},
        {"w": "is", "start": 0.4, "end": 0.5, "conf": 1.0}]


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(asr_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(asr_mod.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(proc, "which", lambda n: f"/usr/bin/{n}")
    monkeypatch.setattr(proc, "ytdlp_available_inprocess", lambda: True)


def rules(fake_run, out=WORKER_OUT, asr_rc=0):
    def ytdlp(cmd):
        if "-x" not in cmd:
            return False
        Path(cmd[cmd.index("-o") + 1].replace("%(ext)s", "mp3")).write_bytes(b"ID3fake")
        return True

    def worker(cmd):
        if cmd[:2] != ["uv", "run"]:
            return False
        if asr_rc == 0:
            Path(cmd[-1]).write_text(json.dumps(out))
        return True
    fake_run.on(ytdlp)
    fake_run.on(worker, returncode=asr_rc, stderr="engine crashed" if asr_rc else "")


def test_fetch_windowed_asr(fake_run, linux, cfg, cache):
    rules(fake_run)
    t = AsrProvider(cfg, cache).fetch(VID, TrackRef("asr-fw", "asr:x", "en", kind="asr"), Window(60, 90))
    dl, run = fake_run.calls[0][0], fake_run.calls[1][0]
    assert "*60.000-90.000" in dl and "ffmpeg:-ac 1 -ar 16000" in dl
    assert run[:3] == ["uv", "run", "--no-project"]
    for spec in asr_mod.BACKENDS["faster-whisper"][1]:
        assert run[run.index(spec) - 1] == "--with"
    i = run.index("python")
    assert run[i + 1] == "-I" and run[i + 2].endswith("asr_worker.py")
    assert run[run.index("--lang") + 1] == "en" and run[run.index("--model") + 1] == "small"
    assert fake_run.calls[1][1]["timeout"] == cfg.timeout("asr")
    assert t.is_generated and t.source == "asr-fw" and t.track.kind == "asr"
    assert t.provenance["source"] == "asr:faster-whisper:small" == t.track.code
    assert t.segments[0].start == 60.5 and t.segments[0].id == "s0001"
    assert not any((cfg.cache_dir / "tmp").iterdir())   # audio deleted
    n = len(fake_run.calls)
    t2 = AsrProvider(cfg, cache).fetch(VID, TrackRef("asr-fw", "asr:x", "en", kind="asr"), Window(60, 90))
    assert len(fake_run.calls) == n and t2.to_dict() == t.to_dict()


def test_fetch_env_overrides(fake_run, linux, cfg, cache, monkeypatch):
    monkeypatch.setenv("YTU_ASR_BACKEND", "mlx-whisper")
    monkeypatch.setenv("YTU_ASR_MODEL", "mlx-community/whisper-large-v3-turbo")
    rules(fake_run)
    t = AsrProvider(cfg, cache).fetch(VID, None)
    run = fake_run.calls[1][0]
    assert asr_mod.BACKENDS["mlx-whisper"][1][0] in run and "--download-sections" not in fake_run.calls[0][0]
    assert t.track.code == "asr:mlx-whisper:mlx-community/whisper-large-v3-turbo"
    assert t.language == "en"   # detected by engine when not given


def test_fetch_worker_failure_cleans_up(fake_run, linux, cfg, cache):
    rules(fake_run, asr_rc=1)
    with pytest.raises(ProcessFailed) as ei:
        AsrProvider(cfg, cache).fetch(VID, None)
    assert "engine crashed" in ei.value.details["stderr_tail"]
    assert not any((cfg.cache_dir / "tmp").iterdir())


def test_fetch_unavailable_video(fake_run, linux, cfg, cache):
    fake_run.on(lambda c: "-x" in c, returncode=1, stderr="ERROR: [youtube] x: Private video")
    with pytest.raises(VideoUnavailable):
        AsrProvider(cfg, cache).fetch(VID, None)


def test_missing_uv(monkeypatch, cfg, cache):
    monkeypatch.setattr(proc, "which", lambda n: None)
    p = AsrProvider(cfg, cache)
    assert not p.available().ok
    with pytest.raises(MissingDependency):
        p.fetch(VID, None)


def test_inventory(linux, cfg, cache):
    meta = VideoMeta(video_id=VID, title="t", duration=19.0, language="en-US")
    [tr] = AsrProvider(cfg, cache).inventory(meta)
    assert tr.kind == "asr" and tr.lang == "en" and tr.code == "asr:faster-whisper:small" and tr.is_generated


@pytest.mark.parametrize("backend, engine", [
    ("parakeet", "parakeet-mlx"), ("mlx-whisper", "mlx-whisper"), ("faster-whisper", "faster-whisper")])
def test_engine_pins_come_from_requirements_files(backend, engine):
    # Pins live in requirements files so Dependabot can bump them; the engine is pinned exactly and first.
    pkgs = asr_mod.BACKENDS[backend][1]
    assert pkgs[0].startswith(f"{engine}==") and all(p and not p.startswith("#") for p in pkgs)


def test_faster_whisper_keeps_av_cap():
    assert "av<19" in asr_mod.BACKENDS["faster-whisper"][1]
