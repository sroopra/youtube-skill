# Examples

`rfscVS0vtbw/` holds the notes for minutes 52:39 to 58:27 of the freeCodeCamp course "Learn Python - Full Course for Beginners" (https://www.youtube.com/watch?v=rfscVS0vtbw).

The repo keeps only the files that the skill and the agent wrote: `knowledge.md`, `evidence.json`, `metadata.json` and `host-notes.json`. The video belongs to its creator. For that reason, the raw transcript (`transcript.srt`, `transcript.json`) and the screenshots (`screenshots/`) are not in the repo, and `.gitignore` excludes them. The screenshot links in `knowledge.md` therefore do not open on GitHub.

To create the missing files on your computer, run these commands in the repo folder. They take about 3 minutes. They need a Gemini key and cost about $0.09.

```bash
Y=youtube-understanding/scripts/yt.py
U="https://www.youtube.com/watch?v=rfscVS0vtbw"
OUT=examples/rfscVS0vtbw

# 1. Take the extra screenshot at 57:38 that one note points to.
mkdir -p "$OUT/screenshots"
uv run "$Y" frames "$U" --at 57:38 | jq -r '.frames[0].path' | xargs -I{} cp {} "$OUT/screenshots/t3458000.jpg"

# 2. Write the transcript and screenshots, and merge the notes. Expect "accepted": 12.
uv run "$Y" package "$U" --mode tutorial --window 52:39-58:27 --with-frames --out "$OUT" --merge-notes "$OUT/host-notes.json"
```
