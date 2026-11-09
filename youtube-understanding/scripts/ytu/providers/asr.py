"""On-demand local ASR fallback (research 2.C, 4.13). NEVER a default dependency.

Backend selection (select_backend):
    darwin + arm64 and lang in PARAKEET_LANGS (or lang None) -> "parakeet"      (parakeet-mlx)
    darwin + arm64 otherwise                                   -> "mlx-whisper"  (mlx-whisper)
    else                                                       -> "faster-whisper" (faster-whisper, "av<19")
Engine pins live in scripts/ytu/asr_requirements/requirements-<backend>.txt so Dependabot can update them.
Overrides: env YTU_ASR_BACKEND (one of the three) and YTU_ASR_MODEL.

Every engine runs in an isolated, uv-cached throw-away env, never in the skill's env:
    uv run --no-project --quiet --python 3.12 --with <pkg>==<ver> [--with av<19] \
        python -I scripts/ytu/asr_worker.py --backend B --model M [--lang L] AUDIO OUT_JSON
(timeout cfg.timeout("asr"); first run downloads the engine + model).

Audio (transient, deleted in ``finally``):
    yt-dlp -f "ba[ext=m4a]/ba/b" -x --audio-format mp3 --audio-quality 5
           --postprocessor-args "ffmpeg:-ac 1 -ar 16000" [--download-sections "*S-E"] -o tmp/audio.%(ext)s URL
Times from the worker are relative to the audio start; the window start is added so all
times are ABSOLUTE (section cut verified frame-accurate, see frames.py).
Transcript: source "asr-parakeet"|"asr-mlx"|"asr-fw", is_generated=True,
track.code = provenance["source"] = "asr:<engine>:<model>", kind "asr".
Cache: "transcript" parts {"video_id","source","track","model","pkg","window","lang","v":1}.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
from pathlib import Path

from .. import proc
from ..cache import Cache
from ..config import Config
from ..errors import BadInput, MissingDependency, ProcessFailed
from ..models import Availability, Segment, TrackRef, Transcript, VideoMeta, Window, Word, make_id
from ..urls import canonical_url
from .frames import _cookie_args, _tmpdir, _ytdlp

log = logging.getLogger("ytu.asr")

PARAKEET_LANGS = frozenset(
    "bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split())

REQUIREMENTS_DIR = Path(__file__).resolve().parent.parent / "asr_requirements"


def _pins(backend: str) -> list[str]:
    """Pip specs from requirements-<backend>.txt (one per line; blank lines and # comments skipped)."""
    lines = (REQUIREMENTS_DIR / f"requirements-{backend}.txt").read_text(encoding="utf-8").splitlines()
    return [ln.split("#", 1)[0].strip() for ln in lines if ln.split("#", 1)[0].strip()]


# backend -> (short source name, pip specs, default model); pkgs[0] is the engine itself
BACKENDS: dict[str, tuple[str, list[str], str]] = {
    "parakeet": ("asr-parakeet", _pins("parakeet"), "mlx-community/parakeet-tdt-0.6b-v3"),
    "mlx-whisper": ("asr-mlx", _pins("mlx-whisper"), "mlx-community/whisper-small-mlx"),
    "faster-whisper": ("asr-fw", _pins("faster-whisper"), "small"),
}
WORKER = Path(__file__).resolve().parent.parent / "asr_worker.py"
UV_REMEDIATION = "Install uv (https://docs.astral.sh/uv/) — ASR engines are fetched on demand into an isolated env."


def select_backend(system: str, machine: str, lang: str | None) -> str:
    """Pure. system = platform.system().lower(), machine = platform.machine().lower()."""
    base = (lang or "").split("-")[0].lower() or None
    if system == "darwin" and machine in ("arm64", "aarch64"):
        return "parakeet" if base is None or base in PARAKEET_LANGS else "mlx-whisper"
    return "faster-whisper"


def _base_lang(lang: str | None) -> str | None:
    return (lang or "").split("-")[0].lower() or None


class AsrProvider:
    name = "asr"

    def __init__(self, cfg: Config, cache: Cache):
        self.cfg, self.cache = cfg, cache

    def available(self) -> Availability:
        """ok iff uvx (or uv) and ffmpeg on PATH. Does not install anything."""
        if not proc.which("uv"):
            return Availability(False, "uv not found on PATH", UV_REMEDIATION)
        if not proc.which("ffmpeg"):
            return Availability(False, "ffmpeg not found on PATH",
                                "brew install ffmpeg (macOS) / sudo apt install ffmpeg (Debian/Ubuntu)")
        return Availability(True)

    def _engine(self, lang: str | None) -> tuple[str, str]:
        backend = os.environ.get("YTU_ASR_BACKEND") or select_backend(
            platform.system().lower(), platform.machine().lower(), lang)
        if backend not in BACKENDS:
            raise BadInput(f"unknown YTU_ASR_BACKEND {backend!r}", f"Use one of: {', '.join(BACKENDS)}.")
        return backend, os.environ.get("YTU_ASR_MODEL") or BACKENDS[backend][2]

    def _track(self, lang: str | None) -> TrackRef:
        backend, model = self._engine(lang)
        return TrackRef(source=BACKENDS[backend][0], code=f"asr:{backend}:{model}", lang=lang, kind="asr",
                        name=f"Local ASR ({backend}, {model})")

    def inventory(self, meta: VideoMeta) -> list[TrackRef]:
        return [self._track(_base_lang(meta.language))]

    def fetch(self, video_id: str, track: TrackRef | None, window: Window | None = None) -> Transcript:
        lang = _base_lang(track.lang if track else None)
        backend, model = self._engine(lang)
        source, pkgs, _ = BACKENDS[backend]
        code = f"asr:{backend}:{model}"
        win = [round(window.start, 3), round(window.end, 3)] if window else None
        parts = {"video_id": video_id, "source": source, "track": code, "model": model, "pkg": pkgs,
                 "window": win, "lang": lang, "v": 1}
        hit = self.cache.get("transcript", parts)
        if hit is not None:
            return Transcript.from_dict(hit)
        av = self.available()
        if not av.ok:
            raise MissingDependency(av.reason, av.remediation)

        tmp = _tmpdir(self.cfg)
        try:
            audio = self._download_audio(video_id, window, tmp)
            out = tmp / "asr.json"
            cmd = ["uv", "run", "--no-project", "--quiet", "--python", "3.12"]
            for p in pkgs:
                cmd += ["--with", p]
            cmd += ["python", "-I", str(WORKER), "--backend", backend, "--model", model,
                    "--lang", lang or "", str(audio), str(out)]
            log.info("running local ASR (%s, %s); first run downloads engine + model", backend, model)
            res = proc.run(cmd, timeout=self.cfg.timeout("asr"), check=False,
                           env={"HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1"})
            if res.returncode or not out.exists():
                raise ProcessFailed(f"local ASR ({backend}) failed",
                                    "Re-run with -v; try YTU_ASR_BACKEND=faster-whisper or a smaller YTU_ASR_MODEL.",
                                    returncode=res.returncode, stderr_tail=res.stderr[-1500:])
            data = json.loads(out.read_text(encoding="utf-8"))   # data only; never executed
        finally:
            shutil.rmtree(tmp, ignore_errors=True)   # audio is transient (ToS)

        offset = window.start if window else 0.0
        segments = worker_segments(data, offset)
        if not segments:
            log.warning("ASR returned no speech for %s%s", video_id, f" {win}" if win else "")
        provenance = {"tool": backend, "tool_version": pkgs[0], "model": model, "source": code,
                      "format": "asr-json", "window": win, "detected_language": data.get("language")}
        t = Transcript(video_id=video_id, segments=segments, source=source,
                       track=TrackRef(source=source, code=code, lang=lang or data.get("language"), kind="asr",
                                      name=f"Local ASR ({backend}, {model})"),
                       is_generated=True, language=lang or data.get("language"), provenance=provenance)
        self.cache.put("transcript", parts, t.to_dict())
        return t

    def _download_audio(self, video_id: str, window: Window | None, tmp: Path) -> Path:
        args = ["-q", "-f", "ba[ext=m4a]/ba/b", "-x", "--audio-format", "mp3", "--audio-quality", "5",
                "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000"]
        if window:
            args += ["--download-sections", f"*{window.start:.3f}-{window.end:.3f}"]
        args += [*_cookie_args(self.cfg), "-o", str(tmp / "audio.%(ext)s"), canonical_url(video_id)]
        _ytdlp(self.cfg, args, "section")
        files = sorted(tmp.glob("audio.mp3")) or [p for p in tmp.glob("audio.*") if not p.name.endswith(".part")]
        if not files:
            raise ProcessFailed("yt-dlp produced no audio file", "Re-run with -v; is ffmpeg installed?")
        return files[0]


def worker_segments(data: dict, offset: float = 0.0) -> list[Segment]:
    """Pure: asr_worker JSON -> Segments with absolute times (offset added), ids s0001.. in time order.
    Empty-text segments are dropped; nothing is invented."""
    raw = sorted((s for s in data.get("segments") or [] if str(s.get("text", "")).strip()),
                 key=lambda s: float(s["start"]))
    out = []
    for n, s in enumerate(raw, 1):
        words = [Word(w=str(w["w"]), start=round(float(w["start"]) + offset, 3), end=round(float(w["end"]) + offset, 3),
                      conf=w.get("conf")) for w in s.get("words") or [] if str(w.get("w", "")).strip()]
        out.append(Segment(id=make_id("s", n), start=round(float(s["start"]) + offset, 3),
                           end=round(float(s["end"]) + offset, 3), text=str(s["text"]).strip(), words=words or None))
    return out
