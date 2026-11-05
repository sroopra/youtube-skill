import dataclasses

import pytest

from pipeline_fakes import DEFAULT_LINES, FakeGemini, install, meta, transcript
from ytu.cache import Cache
from ytu.errors import BadInput, OverBudget, ProcessFailed, VideoUnavailable
from ytu.models import Chapter, Window
from ytu.pipeline import ask
from ytu.urls import parse_url

REF = parse_url("jNQXAC9IVRw")


def test_search_segments_bm25():
    tr = transcript(DEFAULT_LINES)
    hits = ask.search_segments(tr, "Which database is used?")
    assert hits[0].id == "s0004"
    assert ask.search_segments(tr, "how to install dependencies")[0].id == "s0002"
    assert ask.search_segments(tr, "kubernetes") == []
    assert ask.search_segments(tr, "the of and") == []  # stop words only
    assert ask.search_segments(transcript([]), "x") == []
    assert len(ask.search_segments(tr, "docker compose", k=1)) == 1


def test_no_key_at_window_and_excerpts(monkeypatch, cfg, cache, fake_run):
    install(monkeypatch, tr=transcript(DEFAULT_LINES))
    out = ask.run_ask(cfg, cache, REF, "what is on screen?", at=42)
    assert out["mode"] == "transcript-only" and out["answer"] is None and not out["not_found"]
    assert out["windows"][0]["start"] == 0 and out["windows"][0]["end"] == 132
    assert [e["id"] for e in out["excerpts"]][:2] == ["s0001", "s0002"]
    assert 3 <= len(out["suggested_frames"]) <= 6 and 42 in out["suggested_frames"]
    assert out["host_tasks"] and not fake_run.calls


def test_no_key_search_windows_merge_and_not_found(monkeypatch, cfg, cache):
    install(monkeypatch, m=meta(chapters=[Chapter(0, 100, "Intro"), Chapter(100, 300, "Running compose")]),
            tr=transcript(DEFAULT_LINES))
    out = ask.run_ask(cfg, cache, REF, "which database runs here", before=10, after=20)
    assert out["windows"][0]["start"] == 35.0 and out["windows"][0]["reason"] == "question"
    out = ask.run_ask(cfg, cache, REF, "running")  # chapter-title match
    assert any(w["start"] <= 100 <= w["end"] for w in out["windows"])
    out = ask.run_ask(cfg, cache, REF, "kubernetes helm")
    assert out["not_found"] and out["windows"] == [] and out["excerpts"] == [] and out["suggested_frames"] == []


def test_url_timestamp_default_and_bad_at(monkeypatch, cfg, cache):
    install(monkeypatch, tr=transcript(DEFAULT_LINES))
    out = ask.run_ask(cfg, cache, parse_url("https://youtu.be/jNQXAC9IVRw?t=120"), "what now", before=5, after=5)
    assert out["windows"][0]["start"] == 115
    with pytest.raises(BadInput):
        ask.run_ask(cfg, cache, REF, "x", at=9999)
    with pytest.raises(BadInput):
        ask.run_ask(cfg, cache, REF, "  ")


def test_explicit_window(monkeypatch, cfg, cache):
    install(monkeypatch, tr=transcript(DEFAULT_LINES))
    out = ask.run_ask(cfg, cache, REF, "anything", window=Window(100, 130, "user"))
    assert [e["id"] for e in out["excerpts"]] == ["s0005"]


ANSWER = {"answer": "Postgres.", "confidence": "high", "not_found": False,
          "evidence": [{"timestamp": "00:46", "quote": "the database runs in postgres", "kind": "speech"},
                       {"timestamp": "nonsense"}]}


def test_gemini_agentic(monkeypatch, cfg, cache):
    g = FakeGemini([ANSWER])
    install(monkeypatch, tr=transcript(DEFAULT_LINES), key=True, gemini=g)
    out = ask.run_ask(cfg, cache, REF, "which database?")
    assert out["mode"] == "gemini-agentic" and g.requests[0].mode == "agentic"
    assert g.requests[0].schema is ask.ASK_SCHEMA and '<untrusted source="user-question">' in g.requests[0].prompt
    assert out["answer"] == "Postgres." and out["confidence"] == "high"
    assert out["evidence"] == [{"timestamp": "00:46", "t": 46.0, "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw&t=46s",
                                "segment_ids": ["s0004"], "frame_ids": [], "quote": "the database runs in postgres",
                                "kind": "speech", "status": "corroborated"}]


def test_gemini_agentic_falls_back_to_static(monkeypatch, cfg, cache):
    g = FakeGemini([ANSWER], raises={"agentic": ProcessFailed("400 processing")})
    install(monkeypatch, tr=transcript(DEFAULT_LINES), key=True, gemini=g)
    out = ask.run_ask(cfg, cache, REF, "which database?")
    assert out["mode"] == "gemini-static" and [r.mode for r in g.requests] == ["agentic", "static"]
    assert g.requests[1].window is not None and g.requests[1].resolution == "high"
    assert any("agentic unavailable" in u for u in out["uncertainties"])


def test_at_uses_static_clip_only(monkeypatch, cfg, cache):
    g = FakeGemini([ANSWER])
    install(monkeypatch, tr=transcript(DEFAULT_LINES), key=True, gemini=g)
    out = ask.run_ask(cfg, cache, REF, "which database?", at=46)
    assert out["mode"] == "gemini-static" and [r.mode for r in g.requests] == ["static"]
    st = g.requests[0]
    assert st.window.start == 0 and st.window.end == 136 and st.resolution == "high"


def test_at_on_long_video_default_budget_uses_static(monkeypatch, cfg, cache):
    # regression: 2 h video + default $0.50 used to raise OverBudget from the whole-video estimate
    g = FakeGemini([ANSWER])
    install(monkeypatch, m=meta(duration=7200), tr=transcript(DEFAULT_LINES), key=True, gemini=g)
    out = ask.run_ask(cfg, cache, REF, "which database?", at=754)
    assert out["mode"] == "gemini-static" and len(g.requests) == 1 and g.requests[0].window.duration == 150


def test_explicit_window_never_agentic(monkeypatch, cfg, cache):
    g = FakeGemini([ANSWER])
    install(monkeypatch, m=meta(duration=16000), tr=transcript(DEFAULT_LINES), key=True, gemini=g)
    out = ask.run_ask(cfg, cache, REF, "which database?", window=Window(3150, 3500))
    assert [r.mode for r in g.requests] == ["static"] and g.requests[0].window == Window(3150, 3500)
    assert "Look only at 52:30-58:20" in g.requests[0].prompt and out["mode"] == "gemini-static"


def test_gemini_static_when_probe_says_unavailable(monkeypatch, cfg, cache):
    Cache(cfg.cache_dir).put("probe", {"agentic": cfg.models[0]}, {"unavailable": True})
    nf = {"answer": "", "confidence": "low", "not_found": True, "evidence": []}
    g = FakeGemini([nf, nf])
    install(monkeypatch, tr=transcript(DEFAULT_LINES), key=True, gemini=g)
    out = ask.run_ask(cfg, cache, REF, "docker compose install", before=5, after=5)
    assert {r.mode for r in g.requests} == {"static"} and len(g.requests) == len(out["windows"][:3])
    assert out["not_found"] and out["answer"] is None


def test_gemini_errors_that_must_not_fall_back(monkeypatch, cfg, cache):
    g = FakeGemini(raises={"agentic": VideoUnavailable("private")})
    install(monkeypatch, tr=transcript(DEFAULT_LINES), key=True, gemini=g)
    with pytest.raises(VideoUnavailable):
        ask.run_ask(cfg, cache, REF, "q")


def test_gemini_dry_run_and_budget(monkeypatch, cfg, cache):
    g = FakeGemini([ANSWER])
    install(monkeypatch, m=meta(duration=8 * 3600), tr=transcript(DEFAULT_LINES), key=True, gemini=g)
    out = ask.run_ask(dataclasses.replace(cfg, dry_run=True), cache, REF, "which database?")
    # whole-video agentic is over budget, the clips around the hits are not -> static, no refusal
    assert out["dry_run"] and out["mode"] == "gemini-static" and "over_budget" not in out and not g.requests
    assert any("agentic skipped" in u for u in out["uncertainties"]) and out["excerpts"]
    tiny = dataclasses.replace(cfg, max_cost_usd=0.0001)
    out = ask.run_ask(dataclasses.replace(tiny, dry_run=True), cache, REF, "which database?")
    assert out["over_budget"] and not g.requests
    with pytest.raises(OverBudget):
        ask.run_ask(tiny, cache, REF, "which database?")


def test_agentic_overbudget_from_provider_falls_back(monkeypatch, cfg, cache):
    g = FakeGemini([ANSWER], raises={"agentic": OverBudget("provider estimate")})
    install(monkeypatch, tr=transcript(DEFAULT_LINES), key=True, gemini=g)
    out = ask.run_ask(cfg, cache, REF, "which database?")
    assert [r.mode for r in g.requests] == ["agentic", "static"] and out["mode"] == "gemini-static"


def test_answer_without_evidence_is_low_confidence(monkeypatch, cfg, cache):
    g = FakeGemini([{"answer": "Yes", "confidence": "high", "not_found": False, "evidence": []}])
    install(monkeypatch, tr=transcript(DEFAULT_LINES), key=True, gemini=g)
    out = ask.run_ask(cfg, cache, REF, "database?")
    assert out["confidence"] == "low" and any("no timestamped evidence" in u for u in out["uncertainties"])
