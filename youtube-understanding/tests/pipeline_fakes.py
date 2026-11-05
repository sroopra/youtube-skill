"""Shared fakes for pipeline tests (strategy/analyze/ask/package/cli). No network, no subprocess."""

from __future__ import annotations

from ytu.models import (AnalyzeResult, Availability, Chapter, CostEstimate, Frame, Segment, TrackRef, Transcript,
                        VideoMeta, make_id)

VID = "jNQXAC9IVRw"


def meta(duration=300.0, chapters=None, tracks=None, **kw) -> VideoMeta:
    d = dict(video_id=VID, title="Demo <b>video</b>", duration=duration, channel="chan", language="en",
             availability="public", live_status="not_live", description="desc", upload_date="20260101",
             chapters=chapters or [], yt_dlp_version="2026.10.26",
             tracks=tracks if tracks is not None else [TrackRef("ytdlp", "en", "en", "manual", formats=["json3"])])
    d.update(kw)
    return VideoMeta(**d)


def transcript(texts: list[tuple[float, float, str]], generated=False) -> Transcript:
    segs = [Segment(make_id("s", i), s, e, t) for i, (s, e, t) in enumerate(texts, 1)]
    tr = TrackRef("ytdlp", "en", "en", "auto" if generated else "manual")
    return Transcript(VID, segs, "ytdlp", tr, generated, "en", {"tool": "yt-dlp"})


DEFAULT_LINES = [
    (0.0, 4.0, "welcome to the tutorial about docker compose"),
    (4.0, 9.0, "first we install the dependencies with pip"),
    (40.0, 45.0, "as you can see this code defines the service"),
    (45.0, 50.0, "the database runs in postgres"),
    (120.0, 126.0, "now run docker compose up to start everything"),
    (200.0, 205.0, "thanks for watching"),
]


def fake_estimate(model, duration_s, resolution="low", fps=1.0, prompt_tokens=2000, output_tokens=8192):
    per = {"low": 102, "medium": 102, "high": 312}[resolution]
    tokens = int(duration_s * per) + prompt_tokens
    return CostEstimate(model, tokens, output_tokens, round(tokens / 1e6 * 0.75 + output_tokens / 1e6 * 3.75, 6),
                        "fake")


class FakeGemini:
    name = "gemini-interactions"

    def __init__(self, responses=None, raises=None):
        self.responses = list(responses or [])
        self.raises = raises or {}
        self.requests = []

    def available(self):
        return Availability(True)

    def analyze(self, req):
        self.requests.append(req)
        if req.mode in self.raises:
            raise self.raises[req.mode]
        data = self.responses.pop(0) if self.responses else None
        return AnalyzeResult(data=data, raw_text="", usage={"input_tokens": 100, "output_tokens": 10},
                             model=req.model, mode=req.mode, agentic_confirmed=req.mode == "agentic",
                             request_hash="h", cost_estimate=CostEstimate(req.model, 1, 1, 0.01))


class FakeFrames:
    name = "ffmpeg-remote"

    def __init__(self, tmp_path, fail=False):
        self.tmp, self.fail, self.grabs = tmp_path, fail, []

    def available(self):
        return Availability(True)

    def grab(self, video_id, t, height=720, reason="user-question"):
        from ytu.errors import NetworkError
        self.grabs.append(t)
        if self.fail:
            raise NetworkError("seek failed")
        p = self.tmp / f"frame-{int(t * 1000)}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
        return Frame("", str(p), t, reason, "sha", 1280, 720, method="remote-seek")


def install(monkeypatch, *, m=None, tr=None, key=False, gemini=None, frames=None):
    """Patch metadata, transcript, cost estimate, key presence and providers."""
    from ytu.pipeline import strategy
    from ytu.providers import base, gemini as gem, ytdlp_meta

    m = m or meta()
    monkeypatch.setattr(ytdlp_meta, "fetch_metadata", lambda cfg, cache, ref: m)
    if tr is not None:
        def _get(cfg, cache, meta_, *, lang=None, window=None):
            if isinstance(tr, Exception):
                raise tr
            return tr
        monkeypatch.setattr(strategy, "get_transcript", _get)
    monkeypatch.setattr(gem, "estimate_cost", fake_estimate)
    monkeypatch.setattr(strategy, "key_present", lambda: key)
    if gemini is not None:
        monkeypatch.setattr(base, "get_multimodal_provider", lambda name, cfg, cache: gemini)
    if frames is not None:
        monkeypatch.setattr(base, "get_frame_provider", lambda cfg, cache, name="ffmpeg": frames)
    return m
