#!/usr/bin/env python
"""Segment 48kHz takes into utterances + train_raw.jsonl for Qwen3-TTS SFT.

Parakeet supplies sentence timestamps; each sentence becomes one training clip.
Run from this directory:  ../../.venv/bin/python build_dataset.py
"""
import json
import os
from pathlib import Path

import soundfile as sf
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.utils import load_model

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

OUT = EXP / "dataset/finetune_data"

# Drop your raw takes here -- one file per recording script. Override with
# RECORDINGS_DIR=/path/to/dir. Deliberately NOT recursive: recordings/ also holds
# generated clones (s2_icl_*.wav), and those must never end up in training data.
SRC_DIR = Path(os.environ.get("RECORDINGS_DIR", EXP / "recordings/raw"))
SOURCES = sorted(p for p in SRC_DIR.glob("*") if p.suffix.lower() in (".wav", ".m4a"))

# The 10s ref_audio every row conditions on. This MUST be the same file
# timbre_score.py anchors to -- if the training ref changes between runs, the
# 9.5%-vs-next-run comparison the whole experiment turns on is uncontrolled.
# So it defaults to timbre_score.py's default, and only falls back to the first
# take on a fresh clone where that file does not exist. Override with VOICE_REF.
_SCORING_REF = EXP / "recordings/voice_ref3/ref3_48k.wav"
REF_TAKE = Path(os.environ["VOICE_REF"]) if os.environ.get("VOICE_REF") else (
    _SCORING_REF if _SCORING_REF.exists() else SOURCES[0] if SOURCES else None)

MIN_SEC, MAX_SEC, MIN_CHARS, PAD = 1.0, 20.0, 8, 0.12


def main() -> None:
    if not SOURCES:
        raise SystemExit(
            f"no .wav/.m4a in {SRC_DIR}\n"
            "The author's recordings are not in the repo. Record the passages in\n"
            "../scripts/*.txt -- one continuous file each -- and put them there."
        )
    (OUT / "wavs").mkdir(parents=True, exist_ok=True)
    model = load_model("mlx-community/parakeet-tdt-0.6b-v3")
    tmp = os.environ.get("TMPDIR", "/tmp") + "/w"

    rows, n, total = [], 0, 0.0
    for src in SOURCES:
        if not src.exists():
            print("skip (missing):", src)
            continue
        audio, sr = sf.read(str(src), dtype="float32")
        result = generate_transcription(model=model, audio=str(src), output_path=tmp, format="txt")
        for s in result.sentences:
            text, dur = s.text.strip(), s.end - s.start
            # fragments and monologues are both useless: too short to carry prosody,
            # too long to batch at 2 on a 15GB card.
            if dur < MIN_SEC or dur > MAX_SEC or len(text) < MIN_CHARS:
                continue
            i0 = max(0, int((s.start - PAD) * sr))
            i1 = min(len(audio), int((s.end + PAD) * sr))
            n += 1
            total += dur
            path = OUT / "wavs" / f"utt{n:04d}.wav"
            sf.write(str(path), audio[i0:i1], sr)
            rows.append({"audio": f"./wavs/{path.name}", "text": text, "ref_audio": "./ref.wav"})

    # One consistent ref_audio for every row -- the recipe conditions on it per batch,
    # so varying it would make the speaker embedding a moving target.
    audio, sr = sf.read(str(REF_TAKE), dtype="float32")
    sf.write(str(OUT / "ref.wav"), audio[: int(10 * sr)], sr)
    (OUT / "train_raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    print(f"{n} utterances, {total:.1f}s ({total / 60:.1f} min) -> {OUT}/train_raw.jsonl")
    if total < 600:
        print(f"WARNING: {total / 60:.1f} min. The 1.7-min run scored 9.5%, below a "
              f"generic TTS voice. Target 30-45 min before training again.")


if __name__ == "__main__":
    main()
