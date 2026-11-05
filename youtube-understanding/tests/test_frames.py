from __future__ import annotations

from pathlib import Path

import pytest

from ytu import proc
from ytu.errors import Blocked, BadInput, MissingDependency, ProcessFailed
from ytu.models import VideoRef
from ytu.providers import frames as fr
from ytu.providers.frames import (FfmpegFrames, jpeg_size, parse_showinfo, run_frames, thumb_distance)

VID = "jNQXAC9IVRw"
SIGNED = "https://rr1---sn.googlevideo.com/videoplayback?expire=1&sig=SECRETSIG"


def make_jpeg(w=1280, h=720, tag=b"") -> bytes:
    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0\x00\x11\x08" + h.to_bytes(2, "big") + w.to_bytes(2, "big") + b"\x03" + b"\x01\x22\x00" * 3
    return b"\xff\xd8" + app0 + sof + b"\xff\xda\x00\x02" + tag + b"\xff\xd9"


@pytest.fixture
def tools(monkeypatch):
    monkeypatch.setattr(proc, "which", lambda n: f"/usr/bin/{n}")
    monkeypatch.setattr(proc, "ytdlp_available_inprocess", lambda: True)


def ffmpeg_writes(data_for=lambda cmd: make_jpeg()):
    """Predicate for fake_run: matches an ffmpeg seek and writes the output JPEG (side effect)."""
    def pred(cmd):
        if cmd[0] != "ffmpeg" or "-frames:v" not in cmd:
            return False
        Path(cmd[-1]).write_bytes(data_for(cmd))
        return True
    return pred


def thumbs(fake_run, value=b"\x00" * 256):
    fake_run.on(lambda c: c[0] == "ffmpeg" and "rawvideo" in c, stdout=value)


# --- pure ---------------------------------------------------------------------------------

def test_jpeg_size_reads_sof():
    assert jpeg_size(make_jpeg(320, 240)) == (320, 240)


def test_jpeg_size_skips_fill_and_rst_markers():
    data = make_jpeg(640, 360)
    data = data[:2] + b"\xff\xff" + data[2:]
    assert jpeg_size(data) == (640, 360)


@pytest.mark.parametrize("bad", [b"", b"PNG....", b"\xff\xd8\xff\xda\x00\x02\xff\xd9", b"\xff\xd8\x00\x00"])
def test_jpeg_size_rejects_non_jpeg(bad):
    with pytest.raises(ValueError):
        jpeg_size(bad)


def test_parse_showinfo():
    err = ("[Parsed_showinfo_1 @ 0x1] config in time_base: 1/15360\n"
           "[Parsed_showinfo_1 @ 0x1] n:   0 pts: 249856 pts_time:16.2667 duration: 512\n"
           "[Parsed_showinfo_1 @ 0x1] n:   1 pts: 309248 pts_time:20.1333 duration: 512\n"
           "frame=    2 fps=0.0\n")
    assert parse_showinfo(err) == [16.2667, 20.1333]


def test_thumb_distance():
    assert thumb_distance(b"\x00" * 4, b"\x04" * 4) == 4.0
    assert thumb_distance(b"\x00", b"") == float("inf")


# --- grab -----------------------------------------------------------------------------------

def test_grab_remote_seek_caches_and_never_stores_url(fake_run, tools, cfg, cache):
    fake_run.on(lambda c: "-g" in c, stdout=SIGNED + "\n")
    fake_run.on(ffmpeg_writes())
    f = FfmpegFrames(cfg, cache).grab(VID, 10.0)
    assert (f.width, f.height, f.method, f.abs_ts, f.id) == (1280, 720, "remote-seek", 10.0, "")
    assert Path(f.path).is_file() and Path(f.path).is_relative_to(cfg.cache_dir / "frames")
    ff = fake_run.calls[1][0]
    assert ff[ff.index("-ss") + 1] == "10.000" and ff.index("-ss") < ff.index("-i")
    assert fake_run.calls[0][0][fake_run.calls[0][0].index("-f") + 1].startswith("bv*[height<=720][ext=mp4]")
    # signed URL never persisted in cache
    for p in cfg.cache_dir.rglob("*"):
        if p.is_file():
            assert b"SECRETSIG" not in p.read_bytes()
    assert not any((cfg.cache_dir / "tmp").iterdir())   # temp dir cleaned
    # second call: cache hit, no subprocess
    n = len(fake_run.calls)
    f2 = FfmpegFrames(cfg, cache).grab(VID, 10.0)
    assert f2.sha256 == f.sha256 and len(fake_run.calls) == n


def test_multiple_grabs_reuse_one_stream_url(fake_run, tools, cfg, cache):
    fake_run.on(lambda c: "-g" in c, stdout=SIGNED)
    fake_run.on(ffmpeg_writes())
    fp = FfmpegFrames(cfg, cache)
    for t in (1, 2, 3):
        fp.grab(VID, t)
    assert sum("-g" in c for c, _ in fake_run.calls) == 1


def test_grab_reresolves_url_once_on_403(fake_run, tools, cfg, cache):
    state = {"n": 0}

    def pred(cmd):
        if cmd[0] != "ffmpeg":
            return False
        state["n"] += 1
        if state["n"] > 1:
            Path(cmd[-1]).write_bytes(make_jpeg())
        return state["n"] == 1
    fake_run.on(lambda c: "-g" in c, stdout=SIGNED)
    fake_run.on(pred, returncode=1, stderr="HTTP error 403 Forbidden")
    fake_run.on(lambda c: c[0] == "ffmpeg")
    f = FfmpegFrames(cfg, cache).grab(VID, 5)
    assert f.width == 1280
    assert sum("-g" in c for c, _ in fake_run.calls) == 2


def test_grab_falls_back_to_section_download_when_seek_denied(fake_run, tools, cfg, cache):
    """Stream URLs that 403 even after re-resolving -> yt-dlp 2 s section, first frame; later grabs skip seeking."""
    def section(cmd):
        if "--download-sections" not in cmd:
            return False
        Path(cmd[cmd.index("-o") + 1].replace("%(ext)s", "mp4")).write_bytes(b"fake mp4")
        return True

    def decode(cmd):
        if cmd[0] != "ffmpeg" or "-ss" in cmd:
            return False
        Path(cmd[-1]).write_bytes(make_jpeg(640, 360))
        return True
    fake_run.on(lambda c: "-g" in c, stdout=SIGNED)
    fake_run.on(lambda c: c[0] == "ffmpeg" and "-ss" in c, returncode=1, stderr="Server returned 403 Forbidden")
    fake_run.on(section)
    fake_run.on(decode)
    fp = FfmpegFrames(cfg, cache)
    f = fp.grab(VID, 3159)
    assert (f.method, f.abs_ts, f.width) == ("section-download", 3159, 640)
    sect = [c for c, _ in fake_run.calls if "--download-sections" in c]
    assert sect and "*3159.000-3161.000" in sect[0]
    n_seeks = sum("-ss" in c for c, _ in fake_run.calls)
    fp.grab(VID, 3300)
    assert sum("-ss" in c for c, _ in fake_run.calls) == n_seeks       # no further remote seeks
    assert not any((cfg.cache_dir / "tmp").iterdir())                    # clips deleted
    assert FfmpegFrames(cfg, cache).grab(VID, 3159).method == "section-download"  # cache hit


def test_grab_past_end_is_bad_input(fake_run, tools, cfg, cache):
    fake_run.on(lambda c: "-g" in c, stdout=SIGNED)
    fake_run.on(lambda c: c[0] == "ffmpeg")      # rc 0 but no file written
    with pytest.raises(BadInput, match="past the end"):
        FfmpegFrames(cfg, cache).grab(VID, 9999)


def test_grab_ffmpeg_failure(fake_run, tools, cfg, cache):
    fake_run.on(lambda c: "-g" in c, stdout=SIGNED)
    fake_run.on(lambda c: c[0] == "ffmpeg", returncode=1, stderr="boom")
    with pytest.raises(ProcessFailed):
        FfmpegFrames(cfg, cache).grab(VID, 1)


def test_ytdlp_bot_check_is_blocked(fake_run, tools, cfg, cache):
    fake_run.on(lambda c: "-g" in c, returncode=1, stderr="ERROR: [youtube] x: Sign in to confirm you're not a bot")
    with pytest.raises(Blocked):
        FfmpegFrames(cfg, cache).grab(VID, 1)


def test_missing_ffmpeg(monkeypatch, cfg, cache):
    monkeypatch.setattr(proc, "which", lambda n: None)
    fp = FfmpegFrames(cfg, cache)
    assert not fp.available().ok
    with pytest.raises(MissingDependency):
        fp.grab(VID, 1)


def test_cookies_passed_only_when_configured(fake_run, tools, cfg, cache, tmp_path):
    fake_run.on(lambda c: "-g" in c, stdout=SIGNED)
    fake_run.on(ffmpeg_writes())
    FfmpegFrames(cfg, cache).grab(VID, 1)
    assert "--cookies" not in fake_run.calls[0][0]
    cfg.cookies = tmp_path / "c.txt"
    FfmpegFrames(cfg, cache).grab(VID, 2)
    assert "--cookies" in fake_run.calls[2][0]


# --- window ---------------------------------------------------------------------------------

def test_window_interval_bounded_and_deduped(fake_run, tools, cfg, cache):
    fake_run.on(lambda c: "-g" in c, stdout=SIGNED)
    fake_run.on(ffmpeg_writes(lambda c: make_jpeg(tag=c[c.index("-ss") + 1].encode())))
    # thumbnails: frames at 0 and 10 identical, 20 different
    def thumb(cmd):
        return cmd[0] == "ffmpeg" and "rawvideo" in cmd
    seen = []

    def thumb_pred(cmd):
        if not thumb(cmd):
            return False
        seen.append(cmd)
        return len(seen) == 3
    fake_run.on(thumb_pred, stdout=b"\xff" * 256)
    thumbs(fake_run)
    frames = FfmpegFrames(cfg, cache).window(VID, 0, 100, every_s=10, max_frames=3)
    assert [f.abs_ts for f in frames] == [0.0, 20.0]
    assert all(f.selection_reason == "interval" for f in frames)
    # listing cached
    n = len(fake_run.calls)
    assert len(FfmpegFrames(cfg, cache).window(VID, 0, 100, every_s=10, max_frames=3)) == 2
    assert len(fake_run.calls) == n


def scene_rules(fake_run, start_time="0.000000", pts=(1.5, 4.0)):
    def ytdlp_section(cmd):
        if "--download-sections" not in cmd:
            return False
        out = Path(cmd[cmd.index("-o") + 1].replace("%(ext)s", "mp4"))
        out.write_bytes(b"fake mp4")
        return True

    def ffmpeg_scene(cmd):
        if cmd[0] != "ffmpeg" or "-fps_mode" not in cmd:
            return False
        pattern = cmd[-1]
        for i in range(len(pts)):
            Path(pattern.replace("%05d", f"{i + 1:05d}")).write_bytes(make_jpeg(tag=bytes([i])))
        return True
    info = "".join(f"[Parsed_showinfo_1 @ 0x1] n:{i:4d} pts: 1 pts_time:{p} duration: 1\n" for i, p in enumerate(pts))
    fake_run.on(ytdlp_section)
    fake_run.on(lambda c: c[0] == "ffprobe", stdout=start_time + "\n")
    fake_run.on(ffmpeg_scene, stderr=info)


def test_window_scene_uses_section_offset(fake_run, tools, cfg, cache):
    scene_rules(fake_run)
    fp = FfmpegFrames(cfg, cache)
    vals = iter([b"\x00" * 256, b"\xff" * 256])
    fp._thumb = lambda p: next(vals)
    frames = fp.window(VID, 100, 130, strategy="scene")
    sect = next(c for c, _ in fake_run.calls if "--download-sections" in c)
    assert "*100.000-130.000" in sect
    assert [(f.abs_ts, f.pts, f.selection_reason, f.method) for f in frames] == \
        [(101.5, 1.5, "scene", "section-download"), (104.0, 4.0, "scene", "section-download")]
    assert not any((cfg.cache_dir / "tmp").iterdir())   # clip deleted


def test_window_scene_subtracts_clip_start_time(fake_run, tools, cfg, cache):
    scene_rules(fake_run, start_time="0.500000", pts=(1.5,))
    thumbs(fake_run)
    frames = FfmpegFrames(cfg, cache).window(VID, 100, 130, strategy="scene")
    assert frames[0].abs_ts == 101.0


def test_window_scene_none_falls_back_to_interval(fake_run, tools, cfg, cache):
    fake_run.on(lambda c: "--download-sections" in c and
                Path(c[c.index("-o") + 1].replace("%(ext)s", "mp4")).write_bytes(b"x") >= 0)
    fake_run.on(lambda c: c[0] == "ffprobe", stdout="0\n")
    fake_run.on(lambda c: c[0] == "ffmpeg" and "-fps_mode" in c, returncode=234,
                stderr="Nothing was written into output file")
    fake_run.on(lambda c: "-g" in c, stdout=SIGNED)
    fake_run.on(ffmpeg_writes(lambda c: make_jpeg(tag=c[c.index("-ss") + 1].encode())))
    vals = iter([bytes([i * 50]) * 256 for i in range(5)])
    fp = FfmpegFrames(cfg, cache)
    fp._thumb = lambda p: next(vals)
    frames = fp.window(VID, 0, 30, strategy="scene", every_s=10)
    assert [f.selection_reason for f in frames] == ["interval"] * 3


def test_window_bad_args(cfg, cache):
    fp = FfmpegFrames(cfg, cache)
    with pytest.raises(BadInput):
        fp.window(VID, 10, 5)
    with pytest.raises(BadInput):
        fp.window(VID, 0, 5, strategy="magic")


# --- run_frames -------------------------------------------------------------------------------

def test_run_frames_ids_sorted_and_links(monkeypatch, fake_run, tools, cfg, cache):
    from pipeline_fakes import meta
    from ytu.providers import ytdlp_meta
    monkeypatch.setattr(ytdlp_meta, "fetch_metadata", lambda *a, **k: meta(duration=300.0))
    fake_run.on(lambda c: "-g" in c, stdout=SIGNED)
    fake_run.on(ffmpeg_writes(lambda c: make_jpeg(tag=c[c.index("-ss") + 1].encode())))
    ref = VideoRef(VID, f"https://www.youtube.com/watch?v={VID}")
    out = run_frames(cfg, cache, ref, at=[75.5, 10.0, 10.0])
    assert [(f["id"], f["abs_ts"], f["timestamp"]) for f in out["frames"]] == \
        [("F0001", 10.0, "00:10"), ("F0002", 75.5, "01:15")]
    assert out["frames"][1]["url"].endswith("&t=75s")
    assert "SECRETSIG" not in str(out)


def test_run_frames_requires_exactly_one_mode(cfg, cache):
    ref = VideoRef(VID, f"https://www.youtube.com/watch?v={VID}")
    with pytest.raises(BadInput):
        run_frames(cfg, cache, ref)
    with pytest.raises(BadInput):
        run_frames(cfg, cache, ref, at=[1.0], start=0, end=5)
    with pytest.raises(BadInput, match="max-frames"):
        run_frames(cfg, cache, ref, at=[1, 2, 3], max_frames=2)


def test_subsample():
    assert fr._subsample(list(range(10)), 3) == [0, 3, 6]
    assert fr._subsample([1, 2], 5) == [1, 2]


def test_run_frames_past_end_is_bad_input(monkeypatch, cfg, cache):
    from pipeline_fakes import meta
    from ytu.providers import ytdlp_meta
    monkeypatch.setattr(ytdlp_meta, "fetch_metadata", lambda *a, **k: meta(duration=19.0))
    ref = VideoRef(VID, f"https://www.youtube.com/watch?v={VID}")
    with pytest.raises(BadInput, match=r"--at 01:00 is at/after the video end \(00:19\)"):
        run_frames(cfg, cache, ref, at=[5.0, 60.0])
