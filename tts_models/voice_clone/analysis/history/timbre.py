"""Objective timbre fidelity: cosine similarity of speaker embeddings, ref vs clone."""
import numpy as np, mlx.core as mx, soundfile as sf
from mlx_audio.tts.utils import load_model

m = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")

def embed(path):
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1: a = a.mean(1)
    e = m.extract_speaker_embedding(mx.array(a))
    v = np.array(e, copy=False).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)

ref = "../stt_models/captures/20260817-120355.wav"
r = embed(ref)
print(f"reference: {ref}  (dim {r.size})")
for tag in ["icl", "xvec"]:
    c = embed(f"clone_{tag}.wav")
    print(f"  {tag:5} cosine similarity to reference = {float(r @ c):.4f}")

# Negative controls: definitely-different speakers. Without these the cosine above is
# an uncalibrated number — the first control I tried (sample_for_tts.wav) scored 0.98,
# which showed it was the SAME speaker, not that the metric worked.
ctls = ["output_kokoro.wav","output_piper.wav","output_say.wav","output_inflect.wav","output_kokoro_mlx_8bit.wav"]
print("  --- negative controls (different speakers) ---")
for c in ctls:
    try:
        print(f"  {c:22} = {float(r @ embed(c)):.4f}")
    except Exception as e:
        print(f"  {c:22} skipped ({type(e).__name__})")
