"""Downsample the finetune set to 24kHz mono - the tokenizer's native rate.

RECOVERED from the session transcript, verbatim as it was actually run to
produce dataset/finetune_24k/ (the set that trained the 9.5% checkpoint).
Kept here as the record; the resample belongs in build_dataset.py now.
"""
import soundfile as sf, numpy as np, json
from pathlib import Path
from scipy.signal import resample_poly
rs = lambda a, o, n: resample_poly(a, n, o)

D = Path("../tts_models/finetune_data")
OUT = Path("../tts_models/finetune_24k"); (OUT/"wavs").mkdir(parents=True, exist_ok=True)
tot_in = tot_out = 0
for src in sorted(D.glob("wavs/*.wav")) + [D/"ref.wav"]:
    a, sr = sf.read(src, dtype="float32", always_2d=False)
    if a.ndim > 1: a = a.mean(1)
    assert sr == 48000, sr
    b = rs(a, 48000, 24000).astype("float32")
    dst = (OUT/"wavs"/src.name) if src.parent.name == "wavs" else (OUT/src.name)
    sf.write(dst, b, 24000)
    tot_in += src.stat().st_size; tot_out += dst.stat().st_size
(OUT/"train_raw.jsonl").write_text((D/"train_raw.jsonl").read_text())
print(f"24kHz mono: {tot_in/1e6:.1f}MB -> {tot_out/1e6:.1f}MB")
print("files:", len(list((OUT/'wavs').glob('*.wav'))), "+ ref.wav")
