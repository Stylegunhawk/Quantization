"""Segment the 48kHz takes into utterances + train_raw.jsonl for Qwen3-TTS SFT."""
import json, os, soundfile as sf, numpy as np
from pathlib import Path
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.utils import load_model

OUT = Path("../tts_models/finetune_data"); (OUT/"wavs").mkdir(parents=True, exist_ok=True)
SRCS = ["../tts_models/voice_ref3/ref3_48k.wav",
        "../tts_models/voice_ref/ref_48k.wav",
        "../tts_models/voice_ref2/ref2_48k.wav"]
m = load_model("mlx-community/parakeet-tdt-0.6b-v3"); tmp = os.environ.get("TMPDIR","/tmp")+"/w"
rows, n, total = [], 0, 0.0
for src in SRCS:
    a, sr = sf.read(src, dtype="float32")
    r = generate_transcription(model=m, audio=src, output_path=tmp, format="txt")
    for s in r.sentences:
        txt = s.text.strip()
        dur = s.end - s.start
        if dur < 1.0 or dur > 20.0 or len(txt) < 8:      # skip fragments the model can't learn from
            continue
        pad = 0.12                                        # keep a little air either side
        i0, i1 = max(0,int((s.start-pad)*sr)), min(len(a), int((s.end+pad)*sr))
        n += 1; total += dur
        p = OUT/"wavs"/f"utt{n:04d}.wav"
        sf.write(p, a[i0:i1], sr)
        rows.append({"audio": f"./wavs/{p.name}", "text": txt, "ref_audio": "./ref.wav"})
# one consistent ref_audio for every row, as the recipe strongly recommends
a, sr = sf.read("../tts_models/voice_ref3/ref3_48k.wav", dtype="float32")
sf.write(OUT/"ref.wav", a[:int(10*sr)], sr)
(OUT/"train_raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(f"{n} utterances, {total:.1f}s ({total/60:.1f} min) -> {OUT}/train_raw.jsonl")
for r in rows[:4]: print("  ", r["text"][:80])
