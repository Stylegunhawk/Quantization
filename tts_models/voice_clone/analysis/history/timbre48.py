import numpy as np, mlx.core as mx, soundfile as sf
from mlx_audio.tts.utils import load_model
m = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
def embed(p):
    a, sr = sf.read(p, dtype="float32", always_2d=False)
    if a.ndim > 1: a = a.mean(1)
    v = np.array(m.extract_speaker_embedding(mx.array(a)), copy=False).reshape(-1)
    return v/(np.linalg.norm(v)+1e-9)
r = embed("voice_ref/ref_48k.wav")
print("reference: voice_ref/ref_48k.wav (48kHz Voice Memos)")
same = float(r @ embed("../stt_models/captures/20260817-120355.wav"))   # same speaker, other rec
ctl  = [float(r @ embed(c)) for c in ["output_kokoro.wav","output_piper.wav","output_say.wav","output_inflect.wav"]]
diff = sum(ctl)/len(ctl)
print(f"  same-speaker ceiling (old 16k capture) = {same:.4f}")
print(f"  different-speaker floor (4 TTS voices) = {diff:.4f}")
print("  --- clones ---")
for t in ["clone48_12s_icl","clone48_12s_xvec"]:
    v = float(r @ embed(f"voice_ref/{t}.wav"))
    print(f"  {t:20} {v:.4f}  ({100*(v-diff)/(same-diff):5.1f}% of the way to you)")
print("  --- previous run, 16kHz 7.2s reference, for comparison ---")
for t in ["clone_icl","clone_xvec"]:
    v = float(r @ embed(f"{t}.wav"))
    print(f"  {t:20} {v:.4f}  ({100*(v-diff)/(same-diff):5.1f}%)")
