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

a, sr = sf.read("voice_ref2/ref2_48k.wav", dtype="float32")
h = len(a)//2
ceiling = float(emb_arr(a[:h]) @ emb_arr(a[h:]))   # same speaker/mic/bandwidth
ref = emb_arr(a)
floor_v = [float(ref @ embed(c)) for c in
           ["output_kokoro.wav","output_piper.wav","output_say.wav","output_inflect.wav"]]
floor = sum(floor_v)/len(floor_v)
print(f"  CEILING (ref2 first half vs second half) = {ceiling:.4f}")
print(f"  FLOOR   (4 different TTS voices)         = {floor:.4f}")
print("  --- sample_1 clones ---")
for t in ["clone_s1_icl","clone_s1_xvec"]:
    v=float(ref @ embed(f"voice_ref2/{t}.wav"))
    print(f"  {t:14} {v:.4f}   {100*(v-floor)/(ceiling-floor):5.1f}% of the way to you")
print("  --- previous recording's clones, scored against THIS reference ---")
for t in ["clone48_12s_icl","clone48_12s_xvec"]:
    v=float(ref @ embed(f"voice_ref/{t}.wav"))
    print(f"  {t:14} {v:.4f}   {100*(v-floor)/(ceiling-floor):5.1f}%")
# cross-check: are the two recordings the same speaker to this metric?
print(f"\n  ref1 vs ref2 (same speaker, different session) = {float(ref @ embed('voice_ref/ref_48k.wav')):.4f}")
