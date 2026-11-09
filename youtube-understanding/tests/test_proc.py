import subprocess
import sys

import pytest

from ytu import proc
from ytu.errors import (BadInput, Blocked, LiveStream, MissingDependency, NetworkError, ProcessFailed,
                        SubprocessTimeout, VideoUnavailable, classify_ytdlp_error)
from ytu.util.retry import backoff_delays, retry

SECRET = "AIzaTESTSECRETVALUE123"


class _CP:
    def __init__(self, rc=0, out=b"", err=b""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _patch_subprocess(monkeypatch, result=None, exc=None):
    seen = {}

    def fake(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        if exc:
            raise exc
        return result

    monkeypatch.setattr(subprocess, "run", fake)
    return seen


def test_run_success_no_shell(monkeypatch):
    seen = _patch_subprocess(monkeypatch, _CP(0, b"hello", b""))
    r = proc.run(["echo", "hi"], timeout=5)
    assert r.stdout == "hello" and r.returncode == 0
    assert seen["cmd"] == ["echo", "hi"] and seen["kw"]["timeout"] == 5
    assert "shell" not in seen["kw"]
    assert seen["kw"]["stdin"] is subprocess.DEVNULL


def test_run_bytes_and_input(monkeypatch):
    seen = _patch_subprocess(monkeypatch, _CP(0, b"\xff\xd8", b""))
    r = proc.run(["x"], timeout=1, text=False, input="abc")
    assert r.stdout == b"\xff\xd8" and seen["kw"]["input"] == b"abc"


def test_run_rejects_string_cmd():
    with pytest.raises(BadInput):
        proc.run("echo hi", timeout=1)
    with pytest.raises(BadInput):
        proc.run([], timeout=1)


def test_run_missing_exe(monkeypatch):
    _patch_subprocess(monkeypatch, exc=FileNotFoundError())
    with pytest.raises(MissingDependency) as e:
        proc.run(["ffmpeg", "-version"], timeout=1)
    assert e.value.exit_code == 3


def test_run_timeout(monkeypatch):
    _patch_subprocess(monkeypatch, exc=subprocess.TimeoutExpired(["x"], 3))
    with pytest.raises(SubprocessTimeout) as e:
        proc.run(["x"], timeout=3)
    assert e.value.exit_code == 4


def test_run_failure_redacts(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    _patch_subprocess(monkeypatch, _CP(1, b"", f"boom {SECRET} url?key=zzz&sig=abc".encode()))
    with pytest.raises(ProcessFailed) as e:
        proc.run(["tool"], timeout=1)
    tail = e.value.details["stderr_tail"]
    assert SECRET not in tail and "zzz" not in tail and "***" in tail
    r = proc.run(["tool"], timeout=1, check=False)
    assert r.returncode == 1 and SECRET not in r.stderr


def test_child_env_scrubs_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("HARMLESS", "1")
    env = proc.child_env({"EXTRA": "x"})
    assert "GEMINI_API_KEY" not in env and env["HARMLESS"] == "1" and env["EXTRA"] == "x"
    assert proc.child_env({"MY_TOKEN": "t"})["MY_TOKEN"] == "t"  # explicit pass-through
    assert "GEMINI_API_KEY" in proc.child_env(scrub_secrets=False)


def test_real_subprocess_roundtrip():
    r = proc.run([sys.executable, "-I", "-c", "print('ok')"], timeout=30)
    assert r.stdout.strip() == "ok"


def test_ytdlp_cmd_inprocess(monkeypatch):
    monkeypatch.setattr(proc, "ytdlp_available_inprocess", lambda: True)
    cmd = proc.ytdlp_cmd("-J", "URL")
    assert cmd[:4] == [sys.executable, "-I", "-m", "yt_dlp"]
    assert cmd[-2:] == ["-J", "URL"] and "--ignore-config" in cmd


def test_ytdlp_cmd_uvx_fallback(monkeypatch):
    monkeypatch.setattr(proc, "ytdlp_available_inprocess", lambda: False)
    monkeypatch.setattr(proc, "which", lambda n: "/usr/bin/uvx" if n == "uvx" else None)
    assert proc.ytdlp_cmd("-J")[:4] == ["uvx", "--from", proc.YTDLP_SPEC, "yt-dlp"]
    monkeypatch.setattr(proc, "which", lambda n: None)
    with pytest.raises(MissingDependency):
        proc.ytdlp_cmd()


def test_ytdlp_env_prepends_bin():
    assert proc.ytdlp_env()["PATH"].split(":")[0] == str(__import__("pathlib").Path(sys.executable).parent)


def test_require(monkeypatch):
    monkeypatch.setattr(proc, "which", lambda n: None)
    with pytest.raises(MissingDependency, match="ffmpeg"):
        proc.require("ffmpeg")


def test_fake_run_fixture(fake_run):
    fake_run.on("-J", stdout='{"id": "x"}')
    assert proc.run(["yt-dlp", "-J"], timeout=1).stdout == '{"id": "x"}'
    assert fake_run.calls[0][0] == ["yt-dlp", "-J"]
    with pytest.raises(AssertionError):
        proc.run(["other"], timeout=1)


@pytest.mark.parametrize("stderr,cls", [
    ("ERROR: [youtube] x: Private video. Sign in if you've been granted access", VideoUnavailable),
    ("ERROR: [youtube] x: Video unavailable", VideoUnavailable),
    ("ERROR: [youtube] x: This live stream recording is not available.", LiveStream),
    ("ERROR: [youtube] x: Sign in to confirm you're not a bot.", Blocked),
    ("ERROR: Unable to download subtitles: HTTP Error 429: Too Many Requests", Blocked),
    ("ERROR: Sign in to confirm your age.", VideoUnavailable),
    ("ERROR: [Errno 8] nodename nor servname provided", NetworkError),
    ("ERROR: something odd", ProcessFailed),
])
def test_classify_ytdlp_error(stderr, cls):
    e = classify_ytdlp_error("WARNING: noise\n" + stderr, 1)
    assert type(e) is cls and e.remediation
    assert "ERROR" in e.message


def test_backoff_and_retry():
    d = backoff_delays(5, rng=lambda: 0.0)
    assert d == [1, 2, 4, 8]
    assert max(backoff_delays(10, rng=lambda: 0.0)) == 60
    slept, n = [], {"i": 0}

    def flaky():
        n["i"] += 1
        if n["i"] < 3:
            raise NetworkError("x")
        return "ok"

    assert retry(flaky, should_retry=lambda e: isinstance(e, NetworkError), sleep=slept.append) == "ok"
    assert len(slept) == 2
    with pytest.raises(BadInput):
        retry(lambda: (_ for _ in ()).throw(BadInput("no")), should_retry=lambda e: False, sleep=slept.append)
    with pytest.raises(NetworkError):
        retry(lambda: (_ for _ in ()).throw(NetworkError("x")), should_retry=lambda e: True, attempts=2,
              sleep=lambda s: None)


def test_redact_googlevideo_and_params():
    url = ("https://rr3---sn-x.googlevideo.com/videoplayback?expire=1&ei=abc&ip=203.0.113.9"
           "&sig=X&lsig=Y&n=zzz")
    out = proc.redact(f"Error opening input file {url}.")
    assert "203.0.113.9" not in out and "lsig=Y" not in out and "<googlevideo-url>" in out
    assert proc.redact("x?ip=203.0.113.9&lsig=Y&zip=1") == "x?ip=***&lsig=***&zip=1"
    assert proc.redact("text with secretvalue99", secrets=["secretvalue99"]) == "text with ***"


def test_log_handler_redacts_messages_and_tracebacks(monkeypatch):
    # -vv logs unhandled exceptions with exc_info; the traceback must not carry the key.
    import logging
    import sys

    from ytu.util.log import setup_logging

    key = "AIzaFAKE0000000000000000000000000000000"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    handler = next(h for h in setup_logging().handlers if getattr(h, "_ytu", False))
    try:
        raise RuntimeError(f"request failed for key {key}")
    except RuntimeError:
        record = logging.LogRecord("ytu.test", logging.DEBUG, __file__, 1, "unhandled %s", (key,),
                                   sys.exc_info())
    out = handler.format(record)
    assert "RuntimeError" in out and key not in out
