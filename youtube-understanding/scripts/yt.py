#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yt-dlp[default,deno,curl-cffi]==2026.10.26",
#     "youtube-transcript-api==1.2.4",
#     "google-genai==2.34.0",
# ]
# ///
"""youtube-understanding entrypoint.

    uv run scripts/yt.py <doctor|inspect|transcript|analyze|ask|frames|package> [...] --help

JSON on stdout, diagnostics on stderr, exit codes: 0 ok, 1 internal, 2 bad args, 3 missing dep,
4 network/blocked, 5 auth/missing key, 6 video unavailable/live, 7 over budget.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ytu.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
