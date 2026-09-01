import numpy as np, mlx.core as mx, soundfile as sf, statistics as st
from mlx_audio.tts.utils import load_model
S="../dataset"   # was a scratchpad path; the fine-tuned wav now lives under dataset/
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
floors = {c: float(ref @ embed(c)) for c in
    ["output_kokoro.wav","output_piper.wav","output_say.wav","output_inflect.wav"]}
floor = st.mean(floors.values())
pct = lambda v: 100*(v-floor)/(ceiling-floor)
print(f"CEILING {ceiling:.4f}  FLOOR {floor:.4f}")
for k,v in floors.items(): print(f"   floor voice {k:24s} {v:.4f} -> {pct(v):5.1f}%")
zs=[float(ref @ embed(f"voice_ref3/s2_icl_{i}.wav")) for i in (1,2,3)]
print(f"\nzero-shot ICL median {st.median(zs):.4f} = {pct(st.median(zs)):5.1f}%")
v=float(ref @ embed(f"{S}/ft/finetuned/ft_2_cpu.wav"))
print(f"FINE-TUNED ft_2_cpu  {v:.4f} = {pct(v):5.1f}%")
