"""Shared fixtures. Network is blocked for every test; subprocesses are faked via ``fake_run``."""

from __future__ import annotations

import socket
from typing import Callable

import pytest

from ytu import proc
from ytu.proc import RunResult


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Hermetic env: fake HOME/XDG dirs, no API keys, no YTU_* overrides, no sockets."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    import os
    for k in list(os.environ):
        if k.startswith("YTU_") or k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(k, raising=False)

    def _no_net(*a, **k):
        raise RuntimeError("network access in unit tests is forbidden; mock it")

    monkeypatch.setattr(socket.socket, "connect", _no_net)
    monkeypatch.setattr(socket, "create_connection", _no_net)
    monkeypatch.setattr(socket, "getaddrinfo", _no_net)


class FakeRun:
    """Replaces ytu.proc.run. ``on(predicate_or_substring, stdout=..., stderr=..., returncode=..., raises=...)``
    registers a response; unmatched commands raise AssertionError. ``calls`` records (cmd, kwargs)."""

    def __init__(self):
        self.calls: list[tuple[list[str], dict]] = []
        self._rules: list[tuple[Callable[[list[str]], bool], dict]] = []

    def on(self, match, *, stdout="", stderr="", returncode=0, raises: BaseException | None = None):
        pred = match if callable(match) else (lambda cmd, s=match: s in " ".join(cmd))
        self._rules.append((pred, dict(stdout=stdout, stderr=stderr, returncode=returncode, raises=raises)))
        return self

    def __call__(self, cmd, *, timeout, env=None, cwd=None, check=True, input=None, text=True, scrub_secrets=True):
        cmd = [str(c) for c in cmd]
        self.calls.append((cmd, dict(timeout=timeout, env=env, cwd=cwd, check=check)))
        for pred, r in self._rules:
            if pred(cmd):
                if r["raises"] is not None:
                    raise r["raises"]
                res = RunResult(cmd, r["returncode"], r["stdout"], r["stderr"], 0.0)
                if check and r["returncode"] != 0:
                    from ytu.errors import ProcessFailed
                    raise ProcessFailed(f"{cmd[0]} exited {r['returncode']}", returncode=r["returncode"],
                                        stderr_tail=r["stderr"])
                return res
        raise AssertionError(f"unexpected subprocess call: {cmd}")


@pytest.fixture
def fake_run(monkeypatch) -> FakeRun:
    fr = FakeRun()
    monkeypatch.setattr(proc, "run", fr)
    return fr


@pytest.fixture
def cfg(tmp_path):
    from ytu.config import Config
    return Config.load({"cache_dir": tmp_path / "cache", "out_root": tmp_path / "out"})


@pytest.fixture
def cache(cfg):
    from ytu.cache import Cache
    return Cache(cfg.cache_dir)
