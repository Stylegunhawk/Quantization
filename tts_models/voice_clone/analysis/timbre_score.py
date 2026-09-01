#!/usr/bin/env python
"""Score any WAV for how much it sounds like the reference speaker.

Raw speaker-embedding cosine is uninterpretable on its own, so both ends of the
scale are anchored in MATCHED conditions:

  ceiling  split-half of the reference itself -- same person, mic, room, bandwidth.
           Not 1.0, because nobody matches themselves exactly.
  floor    mean of four synthetic voices that are definitively not the speaker.

    pct = 100 * (cos - floor) / (ceiling - floor)      0% = generic TTS, 100% = you

Two calibration traps this exists to avoid, both of which inflate the number:
  * a "negative control" that is actually the same speaker (scored 0.9795 once);
  * comparing across bandwidths -- a 48kHz reference vs a 16kHz capture once
    produced "123% of the way to you". Split-half keeps both sides identical.

Usage:
    ../../.venv/bin/python timbre_score.py                    # baseline table
    ../../.venv/bin/python timbre_score.py path/to/new.wav ...
"""
import os
import statistics as st
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

HERE = Path(__file__).resolve().parent
EXP = HERE.parent                       # voice_clone/
TTS = EXP.parent                        # tts_models/ -- the floor voices live here

# Your own reference take. Override with VOICE_REF=/path/to/take.wav
# DO NOT change this once you have results: the ceiling is derived from the
# reference and the floor is measured against it, so every historical
# percentage moves with it. A new reference means rescoring everything.
REFERENCE = Path(os.environ.get("VOICE_REF", EXP / "recordings/voice_ref3/ref3_48k.wav"))
FLOOR_VOICES = [TTS / f"output_{n}.wav" for n in ("kokoro", "piper", "say", "inflect")]
ZERO_SHOT = [EXP / f"recordings/voice_ref3/s2_icl_{i}.wav" for i in (1, 2, 3)]

_model = None


def _embed_array(audio: np.ndarray) -> np.ndarray:
    """L2-normalised speaker embedding for a mono float32 waveform."""
    global _model
    if _model is None:
        _model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
    v = np.array(_model.extract_speaker_embedding(mx.array(audio)), copy=False).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


def embed(path: Path) -> np.ndarray:
    """Speaker embedding for a WAV file, downmixed to mono."""
    a, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    return _embed_array(a)


def calibrate() -> tuple[np.ndarray, float, float]:
    """Return (reference embedding, ceiling, floor) for the configured reference."""
    if not REFERENCE.exists():
        raise SystemExit(
            f"reference take not found: {REFERENCE}\n"
            "The author's recordings are not in the repo. Record ~30s of your own\n"
            "speech and point at it:  VOICE_REF=/path/to/your.wav python timbre_score.py"
        )
    a, _ = sf.read(str(REFERENCE), dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    half = len(a) // 2
    ceiling = float(_embed_array(a[:half]) @ _embed_array(a[half:]))
    ref = _embed_array(a)
    missing = [p for p in FLOOR_VOICES if not p.exists()]
    if missing:
        raise SystemExit(
            "missing floor voices: " + ", ".join(p.name for p in missing) +
            "\nregenerate them from tts_models/ with run_kokoro.py, run_piper.py, "
            "run_say.py, run_inflect.py (they are gitignored as generated output)."
        )
    floor = st.mean(float(ref @ embed(p)) for p in FLOOR_VOICES)
    return ref, ceiling, floor


def main(argv: list[str]) -> None:
    ref, ceiling, floor = calibrate()
    pct = lambda v: 100 * (v - floor) / (ceiling - floor)
    print(f"CEILING {ceiling:.4f}   FLOOR {floor:.4f}")

    for p in FLOOR_VOICES:
        v = float(ref @ embed(p))
        print(f"  floor  {p.name:28s} {v:.4f} = {pct(v):6.1f}%")

    present = [p for p in ZERO_SHOT if p.exists()]
    if present:
        vs = [float(ref @ embed(p)) for p in present]
        med = st.median(vs)
        spread = pct(max(vs)) - pct(min(vs))
        print(f"\n  zero-shot ICL (n={len(vs)}) median {med:.4f} = {pct(med):6.1f}%"
              f"  (spread {spread:.1f}pp)")

    for arg in argv:
        p = Path(arg)
        if not p.exists():
            print(f"  {arg}: not found")
            continue
        v = float(ref @ embed(p))
        print(f"  {p.name:34s} {v:.4f} = {pct(v):6.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
