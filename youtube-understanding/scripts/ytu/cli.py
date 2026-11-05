"""CLI: argparse subcommands. JSON to stdout, logs to stderr, typed exit codes (errors.py).

Subcommands: doctor, inspect, transcript, analyze, ask, frames, package.
Handlers are thin: build Config + Cache, parse the URL, call the pipeline function, print JSON.
Success: {"ok": true, ...result}. Failure: {"ok": false, "error": {...}} and exit code.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .cache import Cache
from .config import STRATEGIES, Config
from .errors import BadInput, InternalError, YTUError
from .models import Window
from .proc import redact
from .urls import parse_ts, parse_url
from .util.log import setup_logging

log = logging.getLogger("ytu.cli")


def _ts(s: str) -> float:
    try:
        return parse_ts(s)
    except BadInput as e:
        raise argparse.ArgumentTypeError(e.message) from None


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _window_arg(s: str) -> tuple[float, float]:
    a, sep, b = s.partition("-")
    if not sep:
        raise argparse.ArgumentTypeError("use START-END, e.g. 12:00-15:30")
    return _ts(a), _ts(b)


def _ts_list(s: str) -> list[float]:
    return [_ts(x) for x in _csv(s)]


NOTES_HELP = ("host-agent notes JSON to validate and merge: {\"summary\": str, \"items\": [{\"claim\", \"kind\": "
              "speech|code|diagram|slide|ui|inference, \"t_start\": \"MM:SS\", \"quote\"?, \"section\"?, "
              "\"demonstrated\"?, \"frames\"?: [path]} | {\"contradicts\": \"E0012\", \"claim\": why}]}")


class _JsonArgParser(argparse.ArgumentParser):
    """Argument errors keep the stdout contract: one JSON object, exit 2 (usage stays on stderr)."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        err = BadInput(f"{self.prog}: {message}", f"Run `{self.prog} --help` for usage.")
        emit({"ok": False, "error": err.to_dict()})
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("common")
    g.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug (stderr)")
    g.add_argument("--pretty", action="store_true", help="indent the JSON output (default: compact)")
    g.add_argument("--strategy", choices=STRATEGIES, help="auto (default) | transcript | gemini | local")
    g.add_argument("--force-refresh", "--force", dest="force_refresh", action="store_true", default=None,
                   help="ignore caches (re-fetch / re-run)")
    g.add_argument("--cache-dir", type=Path)
    g.add_argument("--models", type=_csv, help="comma-separated Gemini model ladder")
    g.add_argument("--model", help="pin this Gemini model first in the ladder")
    g.add_argument("--transcript-providers", type=_csv, help="comma-separated, e.g. ytdlp,yta,asr")
    g.add_argument("--lang", help="preferred transcript language (e.g. en, de)")
    g.add_argument("--max-cost-usd", type=float)
    g.add_argument("--max-input-tokens", type=int)
    g.add_argument("--max-frames", type=int)
    g.add_argument("--cookies", type=Path, help="cookies.txt for yt-dlp (only after a bot-check error; ban risk)")
    g.add_argument("--dry-run", action="store_true", default=None, help="plan + cost estimate, no paid calls")

    win = argparse.ArgumentParser(add_help=False)
    w = win.add_argument_group("window")
    w.add_argument("--start", type=_ts, help="window start (SS, MM:SS, HH:MM:SS, 1h2m3s)")
    w.add_argument("--end", type=_ts, help="window end")
    w.add_argument("--window", type=_window_arg, help="START-END shorthand, e.g. 12:00-15:30")

    p = _JsonArgParser(prog="yt.py", description="YouTube video understanding (JSON on stdout).")
    p.add_argument("--version", action="version", version=f"youtube-understanding {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", parents=[common], help="check dependencies, key, cache, network")
    d.add_argument("--network", action="store_true", help="also probe youtube.com / googleapis.com")

    i = sub.add_parser("inspect", parents=[common], help="metadata, chapters, tracks, plan (no downloads)")
    i.add_argument("url")

    t = sub.add_parser("transcript", parents=[common, win], help="fetch transcript")
    t.add_argument("url")
    t.add_argument("--format", dest="fmt", choices=("json", "srt", "txt", "text"), default="json")
    t.add_argument("--max-chars", type=int, default=50_000)
    t.add_argument("--offset", type=int, default=0, help="resume at segment index (from next_offset)")

    a = sub.add_parser("analyze", parents=[common, win], help="understand (A) / tutorial (B) analysis")
    a.add_argument("url")
    a.add_argument("--mode", choices=("understand", "tutorial"), default="understand")
    a.add_argument("--resolution", choices=("low", "medium", "high"), help="force Gemini media resolution")
    a.add_argument("--question", help="focus the analysis on this question")

    q = sub.add_parser("ask", parents=[common, win], help="targeted question, optionally near --at (D)")
    q.add_argument("url")
    q.add_argument("question")
    q.add_argument("--at", type=_ts, help="timestamp (default: URL &t=)")
    q.add_argument("--before", type=float, default=60.0)
    q.add_argument("--after", type=float, default=90.0)

    f = sub.add_parser("frames", parents=[common, win], help="extract frames (paths for host-agent vision)")
    f.add_argument("url")
    f.add_argument("--at", type=_ts_list, action="append", help="timestamp(s), comma-separated; repeatable")
    f.add_argument("--every", type=float, default=10.0, help="interval seconds for --start/--end")
    f.add_argument("--frame-strategy", choices=("interval", "keyframes", "scene"), default="interval")
    f.add_argument("--cues", action="store_true", help="grab frames at transcript visual cues")
    f.add_argument("--height", type=int, default=720)

    k = sub.add_parser("package", parents=[common, win], help="write knowledge package (C)")
    k.add_argument("url")
    k.add_argument("--out", type=Path, help="output dir (default ./youtube-knowledge/<id>)")
    k.add_argument("--mode", choices=("understand", "tutorial"), default="understand")
    k.add_argument("--with-frames", action="store_true", help="grab frames for visual items / cues (ffmpeg)")
    k.add_argument("--merge-notes", type=Path, metavar="NOTES_JSON", help=NOTES_HELP)
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    over = {
        "strategy": args.strategy, "force_refresh": args.force_refresh, "cache_dir": args.cache_dir,
        "models": args.models, "transcript_providers": args.transcript_providers, "lang": args.lang,
        "max_cost_usd": args.max_cost_usd, "max_input_tokens": args.max_input_tokens,
        "max_frames": args.max_frames, "cookies": args.cookies, "dry_run": args.dry_run,
    }
    cfg = Config.load(over)
    if getattr(args, "model", None):
        cfg.models = [args.model] + [m for m in cfg.models if m != args.model]
    return cfg


def _window(args) -> Window | None:
    s, e = getattr(args, "start", None), getattr(args, "end", None)
    w = getattr(args, "window", None)
    if w is not None:
        if s is not None or e is not None:
            raise BadInput("use either --window or --start/--end, not both")
        s, e = w
    if s is None and e is None:
        return None
    if s is None or e is None:
        raise BadInput("--start and --end must be given together")
    try:
        return Window(s, e, "user")
    except ValueError as err:
        raise BadInput(str(err), "--end must be greater than --start.") from None


def _cue_times(cfg: Config, cache: Cache, ref, window: Window | None) -> list[float]:
    from .pipeline import strategy
    from .providers import ytdlp_meta

    meta = ytdlp_meta.fetch_metadata(cfg, cache, ref)
    strategy.check_video_ok(meta, need_gemini=False)
    tr = strategy.get_transcript(cfg, cache, meta, window=window)
    out: list[float] = []
    for t, _, _ in strategy.visual_cues(tr):
        t = min(t + 2.0, max(0.0, (meta.duration or t + 2.0) - 0.5))
        if window and not (window.start <= t <= window.end):
            continue
        if all(abs(t - o) >= 10.0 for o in out):
            out.append(t)
    if not out:
        raise BadInput("no visual cues found in the transcript", "Use --at TS or --start/--end --every N.")
    return out[: cfg.max_frames]


def dispatch(args: argparse.Namespace) -> dict:
    cfg = config_from_args(args)
    cache = Cache(cfg.cache_dir, force=cfg.force_refresh)
    cmd = args.command
    if cmd == "doctor":
        from .pipeline.doctor import run_doctor
        return run_doctor(cfg, network=args.network)
    ref = parse_url(args.url)
    if cmd == "inspect":
        from .pipeline.strategy import run_inspect
        return run_inspect(cfg, cache, ref)
    if cmd == "transcript":
        from .pipeline.strategy import run_transcript
        return run_transcript(cfg, cache, ref, lang=args.lang, window=_window(args), fmt=args.fmt,
                              max_chars=args.max_chars, offset=args.offset)
    if cmd == "analyze":
        from .pipeline.analyze import run_analyze
        return run_analyze(cfg, cache, ref, mode=args.mode, window=_window(args), question=args.question,
                           resolution=args.resolution)
    if cmd == "ask":
        from .pipeline.ask import run_ask
        at = args.at if args.at is not None else ref.start_s
        return run_ask(cfg, cache, ref, args.question, at=at, before=args.before, after=args.after,
                       window=_window(args))
    if cmd == "frames":
        from .providers.frames import run_frames
        w = _window(args)
        at = sorted({t for group in (args.at or []) for t in group})
        if args.cues:
            at = sorted(set(at) | set(_cue_times(cfg, cache, ref, w)))
            w = None if at else w
        if not at and w is None:
            raise BadInput("frames needs --at TS (repeatable), --start/--end (or --window), or --cues")
        if at and w is not None:
            raise BadInput("use --at/--cues or a window, not both")
        return run_frames(cfg, cache, ref, at=at or None, start=w.start if w else None,
                          end=w.end if w else None, strategy=args.frame_strategy, every_s=args.every,
                          height=args.height, max_frames=cfg.max_frames)
    if cmd == "package":
        from .pipeline.package import run_package
        return run_package(cfg, cache, ref, out_dir=args.out, mode=args.mode, with_frames=args.with_frames,
                           merge_notes_path=args.merge_notes, window=_window(args))
    raise BadInput(f"unknown command {cmd}")


def dumps(obj, pretty: bool = False) -> str:
    """The exact stdout encoding (compact unless --pretty); also used to size bounded outputs."""
    if pretty:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def emit(obj: dict, stream=None, pretty: bool = False) -> None:
    stream = stream or sys.stdout
    stream.write(dumps(obj, pretty) + "\n")
    stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # argparse exits 2 on bad args
    setup_logging(args.verbose)
    pretty = bool(getattr(args, "pretty", False))
    try:
        result = dispatch(args)
    except YTUError as e:
        log.error("%s: %s", type(e).__name__, e.message)
        emit({"ok": False, "error": e.to_dict()}, pretty=pretty)
        return e.exit_code
    except NotImplementedError as e:
        err = InternalError(f"not implemented: {args.command} {e}".strip(), "This feature is not built yet.")
        emit({"ok": False, "error": err.to_dict()}, pretty=pretty)
        return err.exit_code
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001
        log.debug("unhandled", exc_info=True)
        err = InternalError(redact(f"unexpected {type(e).__name__}: {e}"))
        emit({"ok": False, "error": err.to_dict()}, pretty=pretty)
        return err.exit_code
    emit({"ok": True, **result}, pretty=pretty)
    return 0
