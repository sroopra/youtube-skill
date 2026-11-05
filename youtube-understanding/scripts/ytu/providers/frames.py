"""Frame extraction with ffmpeg (research 2.C, 4.10 step 3, 4.11).

Two methods:

remote-seek (``grab``; ``window(strategy="interval")``) -- no media download:
    url = first line of ``yt-dlp -q -f "bv*[height<=H][ext=mp4]/bv*[height<=H]/b" -g URL``
    ffmpeg -ss T -i url -frames:v 1 ...  (input seek; accurate when transcoding)
    The signed googlevideo URL lives only in this provider instance's memory for the
    current run: never cached, never logged (proc redacts ``sig=``). One URL is reused
    for every timestamp of a video; on an HTTP 403 (expired) it is re-resolved once.

section-download (``window(strategy="scene"|"keyframes")``):
    yt-dlp --download-sections "*S-E" -> temp clip (deleted afterwards), then
    ffmpeg select/showinfo picks frames; their ``pts_time`` is parsed from showinfo.
    Absolute ts = S + pts_time - clip_start_time (clip start measured with ffprobe).
    Verified live (jNQXAC9IVRw, section 7-15 s, keyframes at 5.4/11.2): yt-dlp's
    stream-copy cut puts the pre-roll at negative pts behind an MP4 edit list, so the
    decoded pts 0 is exactly S (frame hash identical to a remote seek at 7.0 s) and
    ffprobe reports start_time 0.

Near-duplicate frames inside a window are dropped using a 16x16 grayscale thumbnail
(ffmpeg rawvideo, no Pillow): mean absolute pixel difference < DEDUPE_THRESHOLD vs the
last kept frame. Cache namespace "frames": one JPEG per (video_id, t_ms, height, method),
plus a JSON listing per window request. Width/height come from the JPEG SOF marker.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import tempfile
from pathlib import Path

from .. import proc
from ..cache import Cache
from ..config import Config
from ..errors import BadInput, MissingDependency, ProcessFailed, YTUError, classify_ytdlp_error
from ..models import Availability, Frame, VideoRef, make_id
from ..urls import canonical_url, fmt_ts, ts_link

log = logging.getLogger("ytu.frames")

FFMPEG_REMEDIATION = "brew install ffmpeg (macOS) / sudo apt install ffmpeg (Debian/Ubuntu)"
SCALE_VF = "scale='min(1568,iw)':-2"   # long edge <= 1568 px (vision-model friendly)
SCENE_THRESHOLD = 0.3                   # ffmpeg select scene score, 0..1
DEDUPE_THRESHOLD = 3.0                  # mean |diff| on 0..255 grayscale 16x16 thumbnails
STRATEGIES = ("interval", "keyframes", "scene")
_SHOWINFO_PTS = re.compile(r"\bn:\s*(\d+)\b.*?\bpts_time:\s*(-?[\d.]+)")
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_size(data: bytes) -> tuple[int, int]:
    """Pure: (width, height) from JPEG SOFn marker; raise ValueError if not a JPEG."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG (missing SOI)")
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            raise ValueError("corrupt JPEG marker stream")
        marker = data[i + 1]
        if marker == 0xFF:          # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:   # standalone markers
            i += 2
            continue
        seglen = int.from_bytes(data[i + 2:i + 4], "big")
        if marker in _SOF:
            if i + 9 > n:
                break
            h = int.from_bytes(data[i + 5:i + 7], "big")
            w = int.from_bytes(data[i + 7:i + 9], "big")
            return w, h
        if marker == 0xDA or seglen < 2:   # start of scan before SOF
            break
        i += 2 + seglen
    raise ValueError("no SOF marker found in JPEG")


def _cookie_args(cfg: Config) -> list[str]:
    return ["--cookies", str(cfg.cookies)] if cfg.cookies else []


def _ytdlp(cfg: Config, args: list[str], timeout_name: str):
    """Run yt-dlp; typed error on failure."""
    res = proc.run(proc.ytdlp_cmd(*args), timeout=cfg.timeout(timeout_name), env=proc.ytdlp_env(), check=False)
    if res.returncode:
        raise classify_ytdlp_error(res.stderr, res.returncode)
    return res


def _tmpdir(cfg: Config) -> Path:
    from ..config import cache_tmp_root
    return Path(tempfile.mkdtemp(prefix="frames-", dir=cache_tmp_root(cfg.cache_dir)))


def _video_format(height: int) -> str:
    return f"bv*[height<={height}][ext=mp4]/bv*[height<={height}]/b"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _frame_from_file(path: Path, t: float, reason: str, method: str, pts: float | None = None) -> Frame:
    data = path.read_bytes()
    try:
        w, h = jpeg_size(data)
    except ValueError as e:
        raise ProcessFailed(f"ffmpeg wrote an unreadable JPEG at {fmt_ts(t)}: {e}") from None
    return Frame(id="", path=str(path), abs_ts=round(t, 3), selection_reason=reason, sha256=_sha256(data),
                 width=w, height=h, pts=pts, method=method)


def _frame_parts(video_id: str, t: float, height: int, method: str) -> dict:
    return {"video_id": video_id, "t_ms": int(round(t * 1000)), "height": height, "method": method, "v": 1}


class FfmpegFrames:
    name = "ffmpeg-remote"

    def __init__(self, cfg: Config, cache: Cache):
        self.cfg, self.cache = cfg, cache
        self._urls: dict[tuple[str, int], str] = {}   # in-memory only, this run
        self._seek_denied: set[str] = set()            # videos whose stream URLs 403 on remote seek

    # -- checks ------------------------------------------------------------------
    def available(self) -> Availability:
        """ok iff ffmpeg on PATH (proc.which) and yt-dlp launcher resolvable."""
        if not proc.which("ffmpeg"):
            return Availability(False, "ffmpeg not found on PATH", FFMPEG_REMEDIATION)
        if not (proc.ytdlp_available_inprocess() or proc.which("uvx")):
            return Availability(False, "yt-dlp not available",
                                "Run via `uv run scripts/yt.py ...` or install uv (https://docs.astral.sh/uv/).")
        return Availability(True)

    def _require_ffmpeg(self, *names: str) -> None:
        for n in ("ffmpeg", *names):
            if not proc.which(n):
                raise MissingDependency(f"{n} not found on PATH", FFMPEG_REMEDIATION)

    # -- remote seek ---------------------------------------------------------------
    def _stream_url(self, video_id: str, height: int, refresh: bool = False) -> str:
        key = (video_id, height)
        if refresh or key not in self._urls:
            res = _ytdlp(self.cfg, ["-q", "-f", _video_format(height), "-g", *_cookie_args(self.cfg),
                                    canonical_url(video_id)], "metadata")
            urls = [ln.strip() for ln in str(res.stdout).splitlines() if ln.strip().startswith("http")]
            if not urls:
                raise ProcessFailed("yt-dlp -g returned no stream URL",
                                    "Re-run with -v; the video may have no video stream at this height.")
            self._urls[key] = urls[0]
        return self._urls[key]

    def _seek(self, url: str, t: float, out: Path):
        return proc.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", url,
                         "-frames:v", "1", "-vf", SCALE_VF, "-q:v", "2", "-y", str(out)],
                        timeout=self.cfg.timeout("frame"), check=False)

    def grab(self, video_id: str, t: float, height: int = 720, reason: str = "user-question") -> Frame:
        if t < 0:
            raise BadInput(f"negative timestamp {t}")
        parts = _frame_parts(video_id, t, height, "remote-seek")
        for method in ("remote-seek", "section-download"):
            hit = self.cache.get_file("frames", _frame_parts(video_id, t, height, method), ".jpg")
            if hit:
                return _frame_from_file(hit, t, reason, method)
        self._require_ffmpeg()
        if video_id in self._seek_denied:
            return self._grab_section(video_id, t, height, reason)
        tmp = _tmpdir(self.cfg)
        try:
            out = tmp / "frame.jpg"
            res = self._seek(self._stream_url(video_id, height), t, out)
            if res.returncode and "403" in res.stderr:     # signed URL expired -> resolve once more
                log.info("stream URL rejected (403); re-resolving")
                res = self._seek(self._stream_url(video_id, height, refresh=True), t, out)
            if res.returncode and "403" in res.stderr:
                # Some player clients' URLs reject plain HTTP range reads; yt-dlp's own downloader
                # still works, so fall back to a 2 s section download for this video.
                log.warning("remote seek denied (403) for %s; using short section downloads", video_id)
                self._seek_denied.add(video_id)
                return self._grab_section(video_id, t, height, reason)
            if res.returncode:
                raise ProcessFailed(f"ffmpeg failed to grab a frame at {fmt_ts(t)}",
                                    "Re-run with -v; check network access to *.googlevideo.com.",
                                    returncode=res.returncode, stderr_tail=res.stderr[-1500:])
            if not out.exists() or out.stat().st_size == 0:
                raise BadInput(f"no frame decoded at {fmt_ts(t)} (past the end of the video?)",
                               "Use a timestamp within the video duration (`yt.py inspect URL`).")
            path = self.cache.put_file("frames", parts, out, ".jpg")
            return _frame_from_file(path, t, reason, "remote-seek")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _grab_section(self, video_id: str, t: float, height: int, reason: str) -> Frame:
        """Fallback for grab(): download [t, t+2 s] with yt-dlp and keep its first frame.
        yt-dlp cuts sections with an output-side seek, so the first displayed frame is at ``t``."""
        tmp = _tmpdir(self.cfg)
        try:
            _ytdlp(self.cfg, ["-q", "-f", _video_format(height), "--download-sections", f"*{t:.3f}-{t + 2:.3f}",
                              *_cookie_args(self.cfg), "-o", str(tmp / "clip.%(ext)s"), canonical_url(video_id)],
                   "section")
            clips = [p for p in tmp.iterdir() if p.name.startswith("clip.") and not p.name.endswith(".part")]
            if not clips:
                raise BadInput(f"no frame decoded at {fmt_ts(t)} (past the end of the video?)",
                               "Use a timestamp within the video duration (`yt.py inspect URL`).")
            out = tmp / "frame.jpg"
            res = proc.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(clips[0]),
                            "-frames:v", "1", "-vf", SCALE_VF, "-q:v", "2", "-y", str(out)],
                           timeout=self.cfg.timeout("frame"), check=False)
            if res.returncode or not out.exists() or out.stat().st_size == 0:
                raise ProcessFailed(f"ffmpeg failed to decode the section at {fmt_ts(t)}", "Re-run with -v.",
                                    returncode=res.returncode, stderr_tail=res.stderr[-1500:])
            path = self.cache.put_file("frames", _frame_parts(video_id, t, height, "section-download"), out, ".jpg")
            return _frame_from_file(path, t, reason, "section-download")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)   # downloaded media is transient

    # -- windows -------------------------------------------------------------------
    def window(self, video_id: str, start: float, end: float, strategy: str = "interval",
               every_s: float = 10.0, max_frames: int = 24, height: int = 720) -> list[Frame]:
        if strategy not in STRATEGIES:
            raise BadInput(f"unknown frame strategy {strategy!r}", f"Use one of: {', '.join(STRATEGIES)}.")
        if start < 0 or end <= start:
            raise BadInput(f"bad window {start}-{end}", "Need 0 <= start < end.")
        if every_s <= 0 or max_frames < 1:
            raise BadInput("every_s must be > 0 and max_frames >= 1")
        lparts = {"video_id": video_id, "start": round(start, 3), "end": round(end, 3), "strategy": strategy,
                  "every_s": every_s, "max_frames": max_frames, "height": height,
                  "scene": SCENE_THRESHOLD, "dedupe": DEDUPE_THRESHOLD, "v": 1}
        cached = self.cache.get("frames", lparts)
        if cached is not None and all(Path(f["path"]).exists() for f in cached):
            return [Frame.from_dict(f) for f in cached]
        self._require_ffmpeg()
        if strategy == "interval":
            frames = self._interval(video_id, start, end, every_s, max_frames, height)
        else:
            frames = self._section(video_id, start, end, strategy, max_frames, height)
            if not frames:
                log.warning("%s detection found no frames in %s-%s (common for screencasts); "
                            "falling back to interval sampling", strategy, fmt_ts(start), fmt_ts(end))
                frames = self._interval(video_id, start, end, every_s, max_frames, height)
        frames = self._dedupe(frames)
        self.cache.put("frames", lparts, [f.to_dict() for f in frames])
        return frames

    def _interval(self, video_id, start, end, every_s, max_frames, height) -> list[Frame]:
        times, t = [], start
        while t < end and len(times) < max_frames:
            times.append(round(t, 3))
            t += every_s
        return [self.grab(video_id, t, height, reason="interval") for t in times]

    def _section(self, video_id, start, end, strategy, max_frames, height) -> list[Frame]:
        self._require_ffmpeg("ffprobe")
        tmp = _tmpdir(self.cfg)
        try:
            _ytdlp(self.cfg, ["-q", "-f", _video_format(height), "--download-sections", f"*{start:.3f}-{end:.3f}",
                              *_cookie_args(self.cfg), "-o", str(tmp / "clip.%(ext)s"), canonical_url(video_id)],
                   "section")
            clips = [p for p in tmp.iterdir() if p.name.startswith("clip.") and not p.name.endswith(".part")]
            if not clips:
                raise ProcessFailed("yt-dlp --download-sections produced no file", "Re-run with -v.")
            clip = clips[0]
            offset = start - self._clip_start(clip)
            sel = f"select='gt(scene,{SCENE_THRESHOLD})'," if strategy == "scene" else ""
            pre = ["-skip_frame", "nokey"] if strategy == "keyframes" else []
            res = proc.run(["ffmpeg", "-nostdin", "-hide_banner", *pre, "-i", str(clip),
                            "-vf", f"{sel}showinfo,{SCALE_VF}", "-fps_mode", "vfr", "-q:v", "2",
                            "-y", str(tmp / "out%05d.jpg")],
                           timeout=self.cfg.timeout("section"), check=False)
            pts = parse_showinfo(res.stderr)
            if res.returncode:
                if not pts:     # nothing selected: ffmpeg 9 fails to open the encoder with 0 frames
                    return []
                raise ProcessFailed(f"ffmpeg {strategy} extraction failed", "Re-run with -v.",
                                    returncode=res.returncode, stderr_tail=res.stderr[-1500:])
            picks = [(i, p) for i, p in enumerate(pts) if p >= 0 and offset + p < end]
            picks = _subsample(picks, max_frames)
            method = "section-download"
            frames = []
            for i, p in picks:
                src = tmp / f"out{i + 1:05d}.jpg"
                if not src.exists():
                    continue
                t = round(offset + p, 3)
                path = self.cache.put_file("frames", _frame_parts(video_id, t, height, method), src, ".jpg")
                frames.append(_frame_from_file(path, t, strategy if strategy == "scene" else "keyframe",
                                               method, pts=round(p, 3)))
            return frames
        finally:
            shutil.rmtree(tmp, ignore_errors=True)   # downloaded media is transient

    def _clip_start(self, clip: Path) -> float:
        res = proc.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of",
                        "default=nw=1:nk=1", str(clip)], timeout=self.cfg.timeout("frame"), check=False)
        try:
            return float(str(res.stdout).strip().splitlines()[0])
        except (ValueError, IndexError):
            log.debug("ffprobe gave no start_time; assuming 0")
            return 0.0

    def _thumb(self, path: str) -> bytes | None:
        res = proc.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path,
                        "-vf", "scale=16:16,format=gray", "-f", "rawvideo", "-"],
                       timeout=self.cfg.timeout("frame"), check=False, text=False)
        data = res.stdout if isinstance(res.stdout, bytes) else b""
        return data if res.returncode == 0 and len(data) == 256 else None

    def _dedupe(self, frames: list[Frame]) -> list[Frame]:
        kept, last = [], None
        for f in sorted(frames, key=lambda f: f.abs_ts):
            th = self._thumb(f.path)
            if th is not None and last is not None and thumb_distance(th, last) < DEDUPE_THRESHOLD:
                log.debug("dropping near-duplicate frame at %s", fmt_ts(f.abs_ts))
                continue
            kept.append(f)
            if th is not None:
                last = th
        return kept


def parse_showinfo(stderr: str) -> list[float]:
    """Pure: pts_time of each frame reported by ffmpeg's showinfo filter, in output order."""
    out: dict[int, float] = {}
    for line in stderr.splitlines():
        if "showinfo" not in line:
            continue
        m = _SHOWINFO_PTS.search(line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return [out[k] for k in sorted(out)]


def thumb_distance(a: bytes, b: bytes) -> float:
    """Pure: mean absolute difference of two equal-length grayscale byte strings."""
    if len(a) != len(b) or not a:
        return float("inf")
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _subsample(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _duration(cfg: Config, cache: Cache, ref: VideoRef) -> float | None:
    """Video duration from (cached) metadata; None when unknown (the grab then reports its own error)."""
    from . import ytdlp_meta
    try:
        return ytdlp_meta.fetch_metadata(cfg, cache, ref).duration
    except YTUError as e:
        log.info("no metadata for the duration check: %s", e.message)
        return None


def run_frames(cfg: Config, cache: Cache, ref: VideoRef, *, at: list[float] | None = None,
               start: float | None = None, end: float | None = None, strategy: str = "interval",
               every_s: float = 10.0, height: int = 720, max_frames: int = 24) -> dict:
    """CLI `frames`. Exactly one of ``at`` or (start, end) (else BadInput).
    Returns {"video_id", "frames": [Frame.to_dict() with ids F0001.. by abs_ts, plus
    "timestamp": fmt_ts, "url": ts_link]}. Paths are absolute cache paths for the host agent to view."""
    has_window = start is not None or end is not None
    if bool(at) == has_window:
        raise BadInput("frames needs exactly one of --at TS (repeatable) or --start/--end")
    if height < 144 or height > 2160:
        raise BadInput(f"bad --height {height}", "Use 144..2160 (default 720).")
    fp = FfmpegFrames(cfg, cache)
    if at:
        times = sorted({round(float(t), 3) for t in at})
        if len(times) > max_frames:
            raise BadInput(f"{len(times)} timestamps exceeds --max-frames {max_frames}",
                           "Pass fewer --at values or raise --max-frames.")
        dur = _duration(cfg, cache, ref)
        late = [t for t in times if dur and t >= dur]
        if late:  # ffmpeg's failure here looks like a network error; say what is wrong instead
            raise BadInput(f"--at {', '.join(fmt_ts(t) for t in late)} is at/after the video end ({fmt_ts(dur)})",
                           "Use timestamps within the video duration (`yt.py inspect URL`).")
        frames = [fp.grab(ref.video_id, t, height, reason="user-question") for t in times]
    else:
        if start is None or end is None:
            raise BadInput("--start and --end are both required for a window")
        frames = fp.window(ref.video_id, start, end, strategy=strategy, every_s=every_s,
                           max_frames=max_frames, height=height)
    out = []
    for n, f in enumerate(sorted(frames, key=lambda f: f.abs_ts), 1):
        f.id = make_id("F", n)
        out.append({**f.to_dict(), "timestamp": fmt_ts(f.abs_ts), "url": ts_link(ref.video_id, f.abs_ts)})
    return {"video_id": ref.video_id, "frames": out}
