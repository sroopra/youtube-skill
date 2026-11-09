"""Typed errors mapped to CLI exit codes.

Every error carries an actionable ``remediation`` string. The CLI prints
``{"ok": false, "error": err.to_dict()}`` on stdout and exits ``err.exit_code``.

Exit codes:
    1 internal / unexpected      (InternalError, ProcessFailed)
    2 bad arguments / input      (BadInput)
    3 missing dependency         (MissingDependency)
    4 network / blocked / timeout(NetworkError, Blocked, SubprocessTimeout)
    5 auth / missing API key     (AuthError, MissingKey)
    6 video unavailable/private/live (VideoUnavailable, LiveStream)
    7 over budget                (OverBudget)
"""

from __future__ import annotations

import re


class YTUError(Exception):
    exit_code = 1
    default_remediation = "Re-run with -v for details; report a bug if it persists."

    def __init__(self, message: str, remediation: str | None = None, **details):
        super().__init__(message)
        self.message = message
        self.remediation = remediation or self.default_remediation
        self.details = details

    def to_dict(self) -> dict:
        d = {
            "type": type(self).__name__,
            "message": self.message,
            "remediation": self.remediation,
            "exit_code": self.exit_code,
        }
        if self.details:
            d["details"] = self.details
        return d


class InternalError(YTUError):
    exit_code = 1


class ProcessFailed(YTUError):
    """A subprocess exited non-zero. ``details`` has cmd0, returncode, stderr_tail (redacted)."""

    exit_code = 1
    default_remediation = "Inspect stderr_tail in the error details; re-run with -v."


class BadInput(YTUError):
    exit_code = 2
    default_remediation = "Check the arguments; run with --help for usage."


class MissingDependency(YTUError):
    exit_code = 3
    default_remediation = "Install the missing tool (see `yt.py doctor`) and retry."


class NetworkError(YTUError):
    exit_code = 4
    default_remediation = (
        "Check network access to youtube.com / googleapis.com. In sandboxed agents allow "
        "the domains (Claude Code: sandbox.network.allowedDomains; Codex: "
        "[sandbox_workspace_write] network_access = true)."
    )


class Blocked(NetworkError):
    default_remediation = (
        "YouTube blocked or rate-limited this IP (bot check / 429). Wait and retry later; "
        "use a residential network; as a last resort pass --cookies FILE (account ban risk)."
    )


class SubprocessTimeout(NetworkError):
    default_remediation = "The operation timed out; retry, or raise the timeout (YTU_TIMEOUT_* / config.toml)."


class AuthError(YTUError):
    exit_code = 5
    default_remediation = "Check the API key / credentials."


class MissingKey(AuthError):
    default_remediation = (
        "Ask the user to add their key: one line GEMINI_API_KEY=... in "
        "~/.config/youtube-understanding/.env (typed in an editor, then chmod 600). "
        "Do not read, print or ask for the key yourself. "
        "Transcript-only commands work without a key (--strategy transcript)."
    )


class VideoUnavailable(YTUError):
    exit_code = 6
    default_remediation = "The video is private, removed, region/age/member-restricted or otherwise unavailable."


class LiveStream(VideoUnavailable):
    default_remediation = "Live streams are not supported; retry after the stream ends and the replay is processed."


class OverBudget(YTUError):
    exit_code = 7
    default_remediation = (
        "Raise --max-cost-usd / --max-input-tokens, narrow the window (--start/--end), "
        "use --strategy transcript, or run with --dry-run to see the estimate."
    )


# --- yt-dlp stderr classification -------------------------------------------------

_YTDLP_PATTERNS: list[tuple[re.Pattern, type[YTUError], str | None]] = [
    (re.compile(r"live stream recording is not available|This live event will begin", re.I), LiveStream, None),
    (re.compile(r"Sign in to confirm your age|age-restricted|inappropriate for some users", re.I), VideoUnavailable,
     "Age-restricted video: pass --cookies FILE exported from a logged-in browser (ban risk)."),
    (re.compile(r"members-only|Join this channel", re.I), VideoUnavailable,
     "Members-only video: requires --cookies from an account with membership."),
    (re.compile(r"Private video|Video unavailable|video is unavailable|This video has been removed|not available in your country|"
                r"This video is no longer available|account associated with this video has been terminated", re.I),
     VideoUnavailable, None),
    (re.compile(r"Sign in to confirm you.re not a bot|HTTP Error 429|Too Many Requests|HTTP Error 403", re.I),
     Blocked, None),
    (re.compile(r"getaddrinfo|Name or service not known|nodename nor servname|Temporary failure in name resolution|"
                r"Network is unreachable|Connection (refused|reset)|timed out|Unable to download (API page|webpage)", re.I),
     NetworkError, None),
    (re.compile(r"is not a valid URL|Incomplete YouTube ID|Unsupported URL", re.I), BadInput, None),
]


def classify_ytdlp_error(stderr: str, returncode: int | None = None) -> YTUError:
    """Map yt-dlp stderr to a typed error. ``stderr`` must already be redacted."""
    lines = [ln for ln in (stderr or "").strip().splitlines() if ln.strip()]
    err_lines = [ln for ln in lines if ln.startswith("ERROR")] or lines[-3:]
    tail = "\n".join(err_lines)[-800:]
    for pat, cls, rem in _YTDLP_PATTERNS:
        if pat.search(stderr or ""):
            return cls(f"yt-dlp: {tail or 'failed'}", rem, returncode=returncode)
    return ProcessFailed(f"yt-dlp failed (exit {returncode}): {tail}", returncode=returncode)
