import json
import os
import time

import pytest

from ytu.cache import Cache, cache_key, canonical_json
from ytu.util.paths import ensure_within, frame_filename, safe_component
from ytu.errors import BadInput

PARTS = {"video_id": "jNQXAC9IVRw", "provider": "ytdlp", "track": "en"}


def test_key_is_order_independent():
    assert cache_key({"a": 1, "b": [1, 2]}) == cache_key({"b": [1, 2], "a": 1})
    assert cache_key({"a": 1}) != cache_key({"a": 2})
    assert canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_put_get_roundtrip(tmp_path):
    c = Cache(tmp_path)
    assert c.get("transcript", PARTS) is None
    p = c.put("transcript", PARTS, {"segments": [1, 2]})
    assert p.is_file() and p.parent.parent.name == "transcript"
    assert p.name == cache_key(PARTS) + ".json"
    assert c.get("transcript", PARTS) == {"segments": [1, 2]}
    assert oct(p.stat().st_mode & 0o777) == "0o600"
    assert not [f for f in p.parent.iterdir() if f.name.endswith(".tmp")]


def test_force_bypasses_reads_but_writes(tmp_path):
    Cache(tmp_path).put("gemini", PARTS, 1)
    forced = Cache(tmp_path, force=True)
    assert forced.get("gemini", PARTS) is None
    forced.put("gemini", PARTS, 2)
    assert Cache(tmp_path).get("gemini", PARTS) == 2


def test_metadata_ttl(tmp_path, monkeypatch):
    c = Cache(tmp_path)
    c.put("metadata", PARTS, "m")
    assert c.get("metadata", PARTS) == "m"
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 25 * 3600)
    assert c.get("metadata", PARTS) is None
    c.put("transcript", PARTS, "t")
    assert c.get("transcript", PARTS) == "t"  # no TTL


def test_corrupt_entry_is_miss(tmp_path):
    c = Cache(tmp_path)
    p = c.put("transcript", PARTS, 1)
    p.write_text("{not json")
    assert c.get("transcript", PARTS) is None


def test_files(tmp_path):
    c = Cache(tmp_path / "c")
    assert c.get_file("frames", PARTS, ".jpg") is None
    p = c.put_file("frames", PARTS, b"\xff\xd8data", ".jpg")
    assert c.get_file("frames", PARTS, ".jpg") == p and p.read_bytes() == b"\xff\xd8data"
    src = tmp_path / "src.jpg"
    src.write_bytes(b"x")
    p2 = c.put_file("frames", {"k": 2}, src, ".jpg")
    assert p2.read_bytes() == b"x" and not src.exists()
    src.write_bytes(b"y")
    c.put_file("frames", {"k": 3}, src, ".jpg", move=False)
    assert src.exists()
    assert Cache(tmp_path / "c", force=True).get_file("frames", PARTS, ".jpg") is None


@pytest.mark.parametrize("ns", ["../x", "Meta", "", "a/b"])
def test_bad_namespace(tmp_path, ns):
    with pytest.raises(ValueError):
        Cache(tmp_path).path_for(ns, PARTS)


@pytest.mark.parametrize("suffix", ["jpg", "../x", ".j/p", ".toolongsuffix"])
def test_bad_suffix(tmp_path, suffix):
    with pytest.raises(ValueError):
        Cache(tmp_path).path_for("frames", PARTS, suffix)


def test_envelope_contents(tmp_path):
    p = Cache(tmp_path).put("probe", PARTS, {"agentic_unavailable": True})
    env = json.loads(p.read_text())
    assert env["parts"] == PARTS and "created_at" in env


def test_safe_component():
    assert safe_component("../../etc/passwd") == "etc-passwd"
    assert safe_component("Héllo wörld!") == "Hello-world"
    assert safe_component("...") == "item"
    assert len(safe_component("a" * 500)) <= 80
    assert "/" not in safe_component("a/b\\c\x00d")


def test_ensure_within(tmp_path):
    assert ensure_within(tmp_path, tmp_path / "a" / "b") == (tmp_path / "a" / "b").resolve()
    with pytest.raises(BadInput):
        ensure_within(tmp_path / "a", tmp_path / "a" / ".." / "b")
    link = tmp_path / "link"
    os.symlink("/etc", link)
    with pytest.raises(BadInput):
        ensure_within(tmp_path, link / "hosts")


def test_frame_filename():
    assert frame_filename(754.2) == "t0754200.jpg"
    assert frame_filename(0) == "t0000000.jpg"
