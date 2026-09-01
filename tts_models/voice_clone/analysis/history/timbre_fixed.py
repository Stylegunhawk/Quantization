"""Timbre fidelity with a MATCHED-CONDITION ceiling.

The ceiling must be the same speaker, same mic, same bandwidth — otherwise the metric
scores recording conditions rather than voice. Split-half of the reference is that.
"""
import numpy as np, mlx.core as mx, soundfile as sf
from mlx_audio.tts.utils import load_model
m = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")

def emb_arr(a):
    v = np.array(m.extract_speaker_embedding(mx.array(a)), copy=False).reshape(-1)
    return v/(np.linalg.norm(v)+1e-9)
def embed(p):
    a, sr = sf.read(p, dtype="float32", always_2d=False)
    if a.ndim > 1: a = a.mean(1)
    return emb_arr(a)

a, sr = sf.read("voice_ref/ref_48k.wav", dtype="float32")
h = len(a)//2
first, second = emb_arr(a[:h]), emb_arr(a[h:])
ceiling = float(first @ second)
ref = emb_arr(a)

floor_vals = [float(ref @ embed(c)) for c in
              ["output_kokoro.wav","output_piper.wav","output_say.wav","output_inflect.wav"]]
floor = sum(floor_vals)/len(floor_vals)

print(f"  CEILING  same speaker/mic/bandwidth (ref first half vs second half) = {ceiling:.4f}")
print(f"  FLOOR    different speakers (4 TTS voices, approx - bandwidth differs) = {floor:.4f}")
print("  --- clones from the 48kHz reference ---")
for t in ["clone48_12s_icl","clone48_12sYOURS_icl","clone48_12s_xvec","clone48_12sYOURS_xvec"]:
    v = float(ref @ embed(f"voice_ref/{t}.wav"))
    print(f"  {t:20} {v:.4f}   {100*(v-floor)/(ceiling-floor):5.1f}% of the way to you")
