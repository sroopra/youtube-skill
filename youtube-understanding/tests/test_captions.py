import json
import sys
import types
from pathlib import Path

import pytest

from ytu import proc
from ytu.errors import BadInput, Blocked, NetworkError, ProcessFailed, VideoUnavailable
from ytu.models import Segment, TrackRef, Window
from ytu.providers import captions
from ytu.providers.captions import YtaCaptions, YtDlpCaptions, dedupe_segments, parse_json3
from ytu.providers.ytdlp_meta import parse_info

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _launcher(monkeypatch):
    monkeypatch.setattr(proc, "ytdlp_available_inprocess", lambda: True)


# --- parse_json3 --------------------------------------------------------------------

def test_parse_json3_manual_zoo():
    segs = parse_json3(load("jNQXAC9IVRw.en.json3"))
    assert [s.id for s in segs] == [f"s000{i}" for i in range(1, 7)]
    assert segs[0].start == 1.2 and segs[0].end == pytest.approx(3.36)
    assert segs[0].text == "All right, so here we are, in front of the elephants"  # newline collapsed
    assert segs[-1].text == "and that's pretty much all there is to say"
    assert all(s.words is None for s in segs)
    assert all(a.end <= b.start for a, b in zip(segs, segs[1:]))


def test_parse_json3_auto_words_and_clamping():
    segs = parse_json3(load("zjkBMFhNj_g.en-orig.json3"))
    first = segs[0]
    assert first.text == "hi everyone so recently I gave a"
    assert first.start == 0.16 and first.end == 2.28  # clamped to next event (2.28), not 4.24
    assert [w.w for w in first.words][:3] == ["hi", "everyone", "so"]
    assert first.words[1].start == pytest.approx(0.32) and first.words[0].end == pytest.approx(0.32)
    assert first.words[-1].end == pytest.approx(2.28)
    assert all(w.conf is None for w in first.words)  # acAsrConf == 0 -> unknown
    assert segs[1].text.startswith("30-minute talk")
    assert not any(s.text == "" for s in segs)  # aAppend newline events dropped
    assert all(a.end <= b.start for a, b in zip(segs, segs[1:]))
    assert [s.id for s in segs] == [f"s{i:04d}" for i in range(1, len(segs) + 1)]


def test_parse_json3_skips_empty_and_sorts():
    data = {"events": [
        {"tStartMs": 5000, "dDurationMs": 1000, "segs": [{"utf8": "b"}]},
        {"tStartMs": 0, "dDurationMs": 99999, "id": 1},
        {"tStartMs": 1000, "dDurationMs": 1000, "segs": [{"utf8": "  \n "}]},
        {"tStartMs": 2000, "dDurationMs": 1000, "segs": [{"utf8": "a"}]},
    ]}
    segs = parse_json3(data)
    assert [(s.id, s.text, s.start) for s in segs] == [("s0001", "a", 2.0), ("s0002", "b", 5.0)]
    assert parse_json3({}) == []


def test_parse_json3_conf_scale():
    data = {"events": [{"tStartMs": 0, "dDurationMs": 1000,
                        "segs": [{"utf8": "a", "acAsrConf": 255}, {"utf8": " b", "tOffsetMs": 500, "acAsrConf": 0}]}]}
    w = parse_json3(data)[0].words
    assert w[0].conf == 1.0 and w[1].conf is None


# --- dedupe ------------------------------------------------------------------------

def S(i, a, b, t):
    return Segment(id=f"s{i:04d}", start=a, end=b, text=t)


def test_dedupe_rolling_duplicates():
    segs = [S(1, 0, 3, "hello world"), S(2, 2, 5, "hello world"), S(3, 4, 6, "world"),
            S(4, 5.5, 8, "world and more"), S(5, 9, 10, "next")]
    out = dedupe_segments(segs)
    assert [(s.id, s.text, s.start, s.end) for s in out] == [
        ("s0001", "hello world", 0, 6), ("s0002", "world and more", 5.5, 8), ("s0003", "next", 9, 10)]


def test_dedupe_keeps_non_overlapping_repeats():
    segs = [S(1, 0, 1, "yes"), S(2, 1, 2, "yes"), S(3, 2, 3, "Yes!")]
    assert len(dedupe_segments(segs)) == 3


def test_dedupe_extends_when_contained_text_grows():
    out = dedupe_segments([S(1, 0, 2, "the cat"), S(2, 1, 4, "the cat sat")])
    assert [(s.text, s.start, s.end) for s in out] == [("the cat sat", 0, 4)]


# --- YtDlpCaptions ------------------------------------------------------------------

def _writes_json3(fixture, code):
    """fake_run predicate+side effect: write fixture into the -o directory yt-dlp would use."""
    def pred(cmd):
        if "--sub-format" not in cmd:
            return False
        out = cmd[cmd.index("-o") + 1]
        vid = cmd[-1].rsplit("=", 1)[-1]
        Path(out.replace("%(id)s.%(ext)s", f"{vid}.{code}.json3")).write_text((FIX / fixture).read_text())
        return True
    return pred


def test_ytdlp_inventory_json3_only():
    meta = parse_info(load("info_jNQXAC9IVRw.json"))
    inv = YtDlpCaptions(None, None).inventory(meta)
    assert {(t.code, t.kind) for t in inv} == {("en", "manual"), ("de", "manual")}  # auto en/de are vtt-only


def test_ytdlp_fetch_manual(fake_run, cfg, cache):
    fake_run.on(_writes_json3("jNQXAC9IVRw.en.json3", "en"))
    track = TrackRef(source="ytdlp", code="en", lang="en", kind="manual", name="English", formats=["json3"])
    p = YtDlpCaptions(cfg, cache)
    t = p.fetch("jNQXAC9IVRw", track)
    argv, kw = fake_run.calls[0]
    assert "--write-subs" in argv and "--write-auto-subs" not in argv
    assert argv[argv.index("--sub-langs") + 1] == "en" and argv[argv.index("--sub-format") + 1] == "json3"
    assert kw["timeout"] == cfg.timeout("subtitles") and kw["check"] is False
    assert t.source == "ytdlp" and not t.is_generated and t.language == "en" and len(t.segments) == 6
    assert t.provenance["format"] == "json3" and t.provenance["tool_version"] == proc.YTDLP_VERSION
    tmp_root = cfg.cache_dir / "tmp"
    assert list(tmp_root.iterdir()) == []  # temp dir removed
    # cached; window filter keeps full-track ids
    w = p.fetch("jNQXAC9IVRw", track, Window(6, 13))
    assert len(fake_run.calls) == 1
    assert [s.id for s in w.segments] == ["s0002", "s0003", "s0004"]
    assert w.provenance["window"] == {"start": 6, "end": 13, "reason": None}


def test_ytdlp_fetch_auto_orig(fake_run, cfg, cache):
    fake_run.on(_writes_json3("zjkBMFhNj_g.en-orig.json3", "en-orig"))
    track = TrackRef(source="ytdlp", code="en-orig", lang="en", kind="auto", formats=["json3"])
    t = YtDlpCaptions(cfg, cache).fetch("zjkBMFhNj_g", track)
    argv = fake_run.calls[0][0]
    assert "--write-auto-subs" in argv and argv[argv.index("--sub-langs") + 1] == "en-orig"
    assert t.is_generated and t.segments[0].words


def test_ytdlp_manual_and_auto_same_code_cached_separately(fake_run, cfg, cache):
    fake_run.on(_writes_json3("jNQXAC9IVRw.en.json3", "en"))
    p = YtDlpCaptions(cfg, cache)
    p.fetch("jNQXAC9IVRw", TrackRef(source="ytdlp", code="en", lang="en", kind="manual"))
    p.fetch("jNQXAC9IVRw", TrackRef(source="ytdlp", code="en", lang="en", kind="auto"))
    assert len(fake_run.calls) == 2


def test_ytdlp_refuses_regex_and_live_chat(cfg, cache):
    p = YtDlpCaptions(cfg, cache)
    for code in ("en.*", "en,de", "live_chat", ""):
        with pytest.raises(BadInput):
            p.fetch("jNQXAC9IVRw", TrackRef(source="ytdlp", code=code, lang="en"))


def test_ytdlp_no_output_file(fake_run, cfg, cache):
    fake_run.on("--sub-format", stderr="WARNING: There are no subtitles for the requested languages")
    with pytest.raises(ProcessFailed):
        YtDlpCaptions(cfg, cache).fetch("jNQXAC9IVRw", TrackRef(source="ytdlp", code="fr", lang="fr"))
    assert list((cfg.cache_dir / "tmp").iterdir()) == []


def test_ytdlp_blocked(fake_run, cfg, cache):
    fake_run.on("--sub-format", returncode=1, stderr="ERROR: Unable to download video subtitles: HTTP Error 429: Too Many Requests")
    with pytest.raises(Blocked):
        YtDlpCaptions(cfg, cache).fetch("jNQXAC9IVRw", TrackRef(source="ytdlp", code="en", lang="en"))


def test_ytdlp_cookies_passthrough(fake_run, cfg, cache, tmp_path):
    cfg.cookies = tmp_path / "c.txt"
    fake_run.on(_writes_json3("jNQXAC9IVRw.en.json3", "en"))
    YtDlpCaptions(cfg, cache).fetch("jNQXAC9IVRw", TrackRef(source="ytdlp", code="en", lang="en"))
    assert "--cookies" in fake_run.calls[0][0]


def test_ytdlp_fetch_without_track_uses_pick(fake_run, cfg, cache, monkeypatch):
    from ytu.pipeline import strategy
    fake_run.on("-J", stdout=json.dumps(load("info_jNQXAC9IVRw.json")))
    fake_run.on(_writes_json3("jNQXAC9IVRw.en.json3", "en"))
    monkeypatch.setattr(strategy, "pick_track", lambda meta, lang=None: next(t for t in meta.tracks if t.code == "en"))
    t = YtDlpCaptions(cfg, cache).fetch("jNQXAC9IVRw", None)
    assert t.track.code == "en"
    monkeypatch.setattr(strategy, "pick_track", lambda meta, lang=None: None)
    with pytest.raises(VideoUnavailable):
        YtDlpCaptions(cfg, cache).fetch("jNQXAC9IVRw", None)


def test_ytdlp_available(monkeypatch, cfg, cache):
    assert YtDlpCaptions(cfg, cache).available().ok
    monkeypatch.setattr(proc, "ytdlp_available_inprocess", lambda: False)
    monkeypatch.setattr(proc, "which", lambda n: None)
    a = YtDlpCaptions(cfg, cache).available()
    assert not a.ok and a.remediation


# --- YtaCaptions --------------------------------------------------------------------

class FakeSnippet:
    def __init__(self, text, start, duration):
        self.text, self.start, self.duration = text, start, duration


class FakeTr:
    def __init__(self, code, generated, snippets=(), exc=None):
        self.language_code, self.is_generated = code, generated
        self.language = code.upper() + (" (auto-generated)" if generated else "")
        self._snips, self._exc, self.fetched = list(snippets), exc, 0

    def fetch(self, preserve_formatting=False):
        self.fetched += 1
        if self._exc:
            raise self._exc
        return iter(self._snips)

    def translate(self, code):  # must never be called
        raise AssertionError("translate called")


@pytest.fixture
def fake_yta(monkeypatch):
    import youtube_transcript_api as yta
    state = {"tracks": [], "list_exc": None, "lists": 0}

    class FakeApi:
        def __init__(self, *a, **k):
            pass

        def list(self, video_id):
            state["lists"] += 1
            if state["list_exc"]:
                raise state["list_exc"]
            return iter(state["tracks"])

    monkeypatch.setattr(yta, "YouTubeTranscriptApi", FakeApi)
    return state


SNIPS = [FakeSnippet("hi everyone", 0.16, 4.0), FakeSnippet("hi everyone", 2.0, 2.0),
         FakeSnippet("so recently [Music]\nI gave", 4.1, 3.0), FakeSnippet("  ", 7, 1)]


def test_yta_fetch_generated_and_cache(fake_yta, cfg, cache):
    gen = FakeTr("en", True, SNIPS)
    fake_yta["tracks"] = [FakeTr("de", True, SNIPS), gen]
    p = YtaCaptions(cfg, cache)
    t = p.fetch("zjkBMFhNj_g", None)
    assert t.source == "yta" and t.is_generated and t.track.kind == "generated" and t.track.code == "en"
    assert [(s.id, s.text) for s in t.segments] == [("s0001", "hi everyone"), ("s0002", "so recently [Music] I gave")]
    assert t.segments[0].end == pytest.approx(4.1)  # overlap clamped to next start
    assert t.provenance["tool"] == "youtube-transcript-api"
    t2 = p.fetch("zjkBMFhNj_g", TrackRef(source="yta", code="en", lang="en", kind="generated"), Window(3, 10))
    assert gen.fetched == 1 and [s.id for s in t2.segments] == ["s0001", "s0002"]


def test_yta_prefers_manual_and_maps_orig_code(fake_yta, cfg, cache):
    man, gen = FakeTr("en", False, SNIPS), FakeTr("en", True, SNIPS)
    fake_yta["tracks"] = [gen, FakeTr("fr", False, SNIPS), man]
    assert YtaCaptions(cfg, cache).fetch("x" * 11, None).track.kind == "manual" and man.fetched == 1
    t = YtaCaptions(cfg, cache).fetch("x" * 11, TrackRef(source="ytdlp", code="en-orig", lang="en", kind="auto"))
    assert t.track.kind == "generated" and gen.fetched == 1


def test_yta_no_matching_track(fake_yta, cfg, cache):
    fake_yta["tracks"] = [FakeTr("de", False, SNIPS)]
    with pytest.raises(VideoUnavailable):
        YtaCaptions(cfg, cache).fetch("x" * 11, TrackRef(source="yta", code="ja", lang="ja"))


def test_yta_inventory(fake_yta, cfg, cache):
    fake_yta["tracks"] = [FakeTr("en", False), FakeTr("de", True)]
    meta = parse_info(load("info_jNQXAC9IVRw.json"))
    inv = YtaCaptions(cfg, cache).inventory(meta)
    assert [(t.source, t.code, t.kind) for t in inv] == [("yta", "en", "manual"), ("yta", "de", "generated")]


def _yta_exc(name):
    import youtube_transcript_api as yta
    cls = getattr(yta, name)
    exc = cls.__new__(cls)
    Exception.__init__(exc, f"{name} happened\nlong details")
    exc.video_id = "x" * 11
    return exc


@pytest.mark.parametrize("name,err", [
    ("IpBlocked", Blocked), ("RequestBlocked", Blocked), ("PoTokenRequired", Blocked),
    ("VideoUnavailable", VideoUnavailable), ("VideoUnplayable", VideoUnavailable),
    ("AgeRestricted", VideoUnavailable), ("TranscriptsDisabled", VideoUnavailable),
    ("NoTranscriptFound", VideoUnavailable), ("InvalidVideoId", BadInput),
    ("YouTubeRequestFailed", NetworkError), ("YouTubeDataUnparsable", ProcessFailed),
])
def test_yta_error_mapping(fake_yta, cfg, cache, name, err):
    fake_yta["list_exc"] = _yta_exc(name)
    with pytest.raises(err) as ei:
        YtaCaptions(cfg, cache).fetch("x" * 11, None)
    assert type(ei.value) is err
    assert "long details" not in ei.value.message and ei.value.remediation


def test_yta_disabled_remediation_suggests_asr(fake_yta, cfg, cache):
    fake_yta["list_exc"] = _yta_exc("TranscriptsDisabled")
    with pytest.raises(VideoUnavailable) as ei:
        YtaCaptions(cfg, cache).fetch("x" * 11, None)
    assert "--strategy local" in ei.value.remediation


def test_yta_fetch_error_mapped(fake_yta, cfg, cache):
    fake_yta["tracks"] = [FakeTr("en", True, exc=_yta_exc("IpBlocked"))]
    with pytest.raises(Blocked):
        YtaCaptions(cfg, cache).fetch("x" * 11, None)


def test_yta_unknown_exception_propagates(fake_yta, cfg, cache):
    fake_yta["list_exc"] = KeyError("weird")
    with pytest.raises(KeyError):
        YtaCaptions(cfg, cache).fetch("x" * 11, None)


def test_yta_available(monkeypatch, cfg, cache):
    assert YtaCaptions(cfg, cache).available().ok
    monkeypatch.setattr(captions.importlib.util, "find_spec", lambda n: None)
    assert not YtaCaptions(cfg, cache).available().ok
