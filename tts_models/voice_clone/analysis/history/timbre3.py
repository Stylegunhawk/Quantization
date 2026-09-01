import numpy as np, mlx.core as mx, soundfile as sf, statistics as st
from mlx_audio.tts.utils import load_model
m = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
def emb_arr(a):
    v = np.array(m.extract_speaker_embedding(mx.array(a)), copy=False).reshape(-1)
    return v/(np.linalg.norm(v)+1e-9)
def embed(p):
    a,sr = sf.read(p, dtype="float32", always_2d=False)
    if a.ndim>1: a=a.mean(1)
    return emb_arr(a)
a,sr = sf.read("voice_ref3/ref3_48k.wav", dtype="float32"); h=len(a)//2
ceiling = float(emb_arr(a[:h]) @ emb_arr(a[h:])); ref = emb_arr(a)
floor = st.mean(float(ref @ embed(c)) for c in
    ["output_kokoro.wav","output_piper.wav","output_say.wav","output_inflect.wav"])
pct = lambda v: 100*(v-floor)/(ceiling-floor)
print(f"  CEILING {ceiling:.4f}   FLOOR {floor:.4f}")
for mode in ["icl","xvec"]:
    vs=[float(ref @ embed(f"voice_ref3/s2_{mode}_{i}.wav")) for i in (1,2,3)]
    print(f"  {mode:5} " + "  ".join(f"{v:.4f}" for v in vs) +
          f"   median {st.median(vs):.4f} = {pct(st.median(vs)):5.1f}%  (spread {pct(max(vs))-pct(min(vs)):.1f}pp)")
print("  --- earlier references, scored against THIS reference ---")
for lbl,p in [("old conversational","voice_ref/clone48_12s_icl.wav"),
              ("phonetic script","voice_ref2/clone_s1_icl.wav")]:
    v=float(ref @ embed(p)); print(f"  {lbl:20} {v:.4f} = {pct(v):5.1f}%")
