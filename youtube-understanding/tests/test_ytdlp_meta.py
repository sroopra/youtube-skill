import json
from pathlib import Path

import pytest

from ytu import proc
from ytu.config import Config
from ytu.errors import Blocked, LiveStream, NetworkError, ProcessFailed, VideoUnavailable
from ytu.models import VideoMeta
from ytu.providers import ytdlp_meta
from ytu.providers.ytdlp_meta import cookie_args, fetch_metadata, parse_info
from ytu.urls import parse_url

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _launcher(monkeypatch):
    monkeypatch.setattr(proc, "ytdlp_available_inprocess", lambda: True)


def test_parse_info_zoo_manual_tracks_and_chapters():
    m = parse_info(load("info_jNQXAC9IVRw.json"))
    assert m.video_id == "jNQXAC9IVRw" and m.title == "Me at the zoo"
    assert m.duration == 19.0 and isinstance(m.duration, float)
    assert m.language is None and m.availability == "public" and m.live_status == "not_live"
    assert [(c.start, c.end, c.title) for c in m.chapters] == [(0, 5, "Intro"), (5, 17, "The cool thing"), (17, 19, "End")]
    manual = {t.code: t for t in m.tracks if t.kind == "manual"}
    assert set(manual) == {"en", "de"}
    assert manual["en"].name == "English" and "json3" in manual["en"].formats and manual["en"].lang == "en"
    assert {t.code for t in m.tracks if t.kind == "auto"} == {"en", "de"}
    assert all(t.source == "ytdlp" for t in m.tracks)
    assert m.yt_dlp_version == proc.YTDLP_VERSION and m.fetched_at.endswith("+00:00")


def test_parse_info_auto_orig_lang():
    m = parse_info(load("info_zjkBMFhNj_g.json"))
    assert m.language == "en" and m.duration == 3588.0
    orig = [t for t in m.tracks if t.code == "en-orig"]
    assert orig and orig[0].kind == "auto" and orig[0].lang == "en" and orig[0].name == "English (Original)"
    assert not [t for t in m.tracks if t.kind == "manual"]


def test_parse_info_drops_live_chat_and_handles_live():
    info = {"id": "4t4AQ15yj1M", "title": "lofi", "duration": None, "live_status": "is_live",
            "subtitles": {"live_chat": [{"ext": "json", "url": "x"}]},
            "automatic_captions": {"en": [{"ext": "vtt", "name": "English"}]}, "chapters": None}
    m = parse_info(info)
    assert m.duration is None and m.live_status == "is_live" and m.chapters == []
    assert [(t.code, t.kind, t.formats) for t in m.tracks] == [("en", "auto", ["vtt"])]


def test_parse_info_chapter_missing_end_and_control_chars():
    info = {"id": "abcdefghijk", "title": "a‮b\x07", "duration": 100,
            "chapters": [{"start_time": 0, "title": "x"}, {"start_time": 40, "title": "y"}]}
    m = parse_info(info)
    assert [(c.start, c.end) for c in m.chapters] == [(0, 40), (40, 100.0)]
    assert m.title == "ab"


def test_parse_info_never_stores_urls():
    m = parse_info({"id": "abcdefghijk", "title": "t",
                    "subtitles": {"en": [{"ext": "json3", "url": "https://www.youtube.com/api/timedtext?sig=SECRET"}]}})
    assert "timedtext" not in json.dumps(m.to_dict())


def test_parse_info_rejects_garbage():
    with pytest.raises(ProcessFailed):
        parse_info({})


def test_cookie_args(tmp_path):
    cfg = Config.load({"cache_dir": tmp_path})
    assert cookie_args(cfg) == []
    cfg.cookies = tmp_path / "c.txt"
    assert cookie_args(cfg) == ["--cookies", str(tmp_path / "c.txt")]
    cfg.cookies = None
    cfg.cookies_from_browser = "firefox"  # optional future field
    assert cookie_args(cfg) == ["--cookies-from-browser", "firefox"]


def test_fetch_metadata_runs_and_caches(fake_run, cfg, cache):
    fake_run.on("-J", stdout=json.dumps(load("info_jNQXAC9IVRw.json")))
    ref = parse_url("https://youtu.be/jNQXAC9IVRw")
    m = fetch_metadata(cfg, cache, ref)
    assert isinstance(m, VideoMeta) and m.title == "Me at the zoo"
    argv, kw = fake_run.calls[0]
    assert "--skip-download" in argv and argv[-1] == "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    assert "--ignore-config" in argv and "--cookies" not in argv
    assert kw["timeout"] == cfg.timeout("metadata") and kw["check"] is False
    m2 = fetch_metadata(cfg, cache, ref)
    assert len(fake_run.calls) == 1 and m2 == m
    cache.force = True
    fetch_metadata(cfg, cache, ref)
    assert len(fake_run.calls) == 2


def test_fetch_metadata_passes_cookies(fake_run, cfg, cache, tmp_path):
    cfg.cookies = tmp_path / "cookies.txt"
    fake_run.on("-J", stdout=json.dumps(load("info_jNQXAC9IVRw.json")))
    fetch_metadata(cfg, cache, parse_url("jNQXAC9IVRw"))
    argv = fake_run.calls[0][0]
    assert argv[argv.index("--cookies") + 1] == str(tmp_path / "cookies.txt")


@pytest.mark.parametrize("stderr,exc", [
    ("ERROR: [youtube] x: Sign in to confirm you’re not a bot. Use --cookies-from-browser", Blocked),
    ("ERROR: [youtube] x: Private video. Sign in if you've been granted access", VideoUnavailable),
    ("ERROR: [youtube] x: Video unavailable", VideoUnavailable),
    ("ERROR: [youtube] x: Join this channel to get access to members-only content", VideoUnavailable),
    ("ERROR: [youtube] x: Sign in to confirm your age.", VideoUnavailable),
    ("ERROR: [youtube] x: This live stream recording is not available.", LiveStream),
    ("ERROR: unable to download webpage: HTTP Error 429: Too Many Requests", Blocked),
    ("ERROR: [Errno 8] nodename nor servname provided, or not known", NetworkError),
    ("ERROR: something odd", ProcessFailed),
])
def test_fetch_metadata_error_mapping(fake_run, cfg, cache, stderr, exc):
    fake_run.on("-J", returncode=1, stderr=stderr)
    with pytest.raises(exc) as ei:
        fetch_metadata(cfg, cache, parse_url("jNQXAC9IVRw"))
    assert ei.value.remediation
    assert cache.get("metadata", ytdlp_meta._cache_parts("jNQXAC9IVRw")) is None


def test_fetch_metadata_member_remediation_mentions_cookies(fake_run, cfg, cache):
    fake_run.on("-J", returncode=1, stderr="ERROR: members-only content")
    with pytest.raises(VideoUnavailable) as ei:
        fetch_metadata(cfg, cache, parse_url("jNQXAC9IVRw"))
    assert "--cookies" in ei.value.remediation


def test_fetch_metadata_bad_json(fake_run, cfg, cache):
    fake_run.on("-J", stdout="not json")
    with pytest.raises(ProcessFailed):
        fetch_metadata(cfg, cache, parse_url("jNQXAC9IVRw"))
