"""Standalone ASR worker, run in an isolated, throw-away uv environment:

    uv run --no-project --quiet --python 3.12 --with <engine-pkg>==<ver> python -I asr_worker.py \
        --backend parakeet|mlx-whisper|faster-whisper --model M [--lang xx] AUDIO OUT_JSON

Does NOT import ``ytu`` (different interpreter/env). Reads only the local audio file and
writes OUT_JSON:
    {"engine", "model", "language", "segments": [{"start","end","text","words":[{"w","start","end","conf"}]}]}
Times are relative to the start of AUDIO; the caller adds any window offset.
Engine libraries print progress to stdout/stderr, so results go to a file, not stdout.
"""

from __future__ import annotations

import argparse
import json
import sys


def merge_tokens(tokens: list[dict]) -> list[dict]:
    """Parakeet sub-word tokens (" P", "yt", "h") -> words. A token starting with whitespace
    (or the first token) begins a new word; conf = min token confidence."""
    words: list[dict] = []
    for tok in tokens:
        text = tok.get("text", "")
        if not text.strip():
            continue
        if not words or text[:1].isspace():
            words.append({"w": text.strip(), "start": tok["start"], "end": tok["end"], "conf": tok.get("confidence")})
        else:
            w = words[-1]
            w["w"] += text.strip()
            w["end"] = tok["end"]
            c = tok.get("confidence")
            if c is not None:
                w["conf"] = c if w["conf"] is None else min(w["conf"], c)
    return words


def run_parakeet(audio: str, model: str, lang: str | None) -> dict:
    from parakeet_mlx import from_pretrained

    res = from_pretrained(model).transcribe(audio)
    segs = []
    for s in res.sentences:
        toks = [{"text": t.text, "start": float(t.start), "end": float(t.end),
                 "confidence": float(getattr(t, "confidence", 1.0))} for t in s.tokens]
        segs.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip(),
                     "words": merge_tokens(toks)})
    return {"language": lang, "segments": segs}


def _whisper_seg(s: dict) -> dict:
    words = [{"w": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"]),
              "conf": float(w.get("probability")) if w.get("probability") is not None else None}
             for w in (s.get("words") or []) if w.get("word", "").strip()]
    return {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip(), "words": words}


def run_mlx_whisper(audio: str, model: str, lang: str | None) -> dict:
    import mlx_whisper

    res = mlx_whisper.transcribe(audio, path_or_hf_repo=model, word_timestamps=True, language=lang,
                                 condition_on_previous_text=False, hallucination_silence_threshold=2.0)
    return {"language": res.get("language"), "segments": [_whisper_seg(s) for s in res.get("segments", [])]}


def run_faster_whisper(audio: str, model: str, lang: str | None) -> dict:
    from faster_whisper import WhisperModel

    m = WhisperModel(model, device="cpu", compute_type="int8")
    segments, info = m.transcribe(audio, language=lang, word_timestamps=True, vad_filter=True,
                                  condition_on_previous_text=False)
    out = []
    for s in segments:
        out.append(_whisper_seg({"start": s.start, "end": s.end, "text": s.text,
                                 "words": [{"word": w.word, "start": w.start, "end": w.end,
                                            "probability": w.probability} for w in (s.words or [])]}))
    return {"language": info.language, "segments": out}


ENGINES = {"parakeet": run_parakeet, "mlx-whisper": run_mlx_whisper, "faster-whisper": run_faster_whisper}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=sorted(ENGINES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", default="")
    ap.add_argument("audio")
    ap.add_argument("out_json")
    a = ap.parse_args(argv)
    res = ENGINES[a.backend](a.audio, a.model, a.lang or None)
    res.update(engine=a.backend, model=a.model)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
