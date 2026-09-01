"""Clone from the 48kHz Voice Memos reference. Both modes, memory watched."""
import sys, time, numpy as np, mlx.core as mx, soundfile as sf, psutil
from mlx_audio.tts.utils import load_model

REF  = sys.argv[1]
TEXT = open(sys.argv[2]).read().strip()
TAG  = sys.argv[3]
TARGET = "Parakeet transcribes twenty one seconds of audio in under half a second on this machine."

m = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
print(f"loaded | peak {mx.get_peak_memory()/1e6:.0f}MB | ref {sf.info(REF).duration:.1f}s")

for mode, kw in [("icl", dict(ref_audio=REF, ref_text=TEXT)), ("xvec", dict(ref_audio=REF))]:
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    try:
        res = list(m.generate(text=TARGET, **kw))
    except Exception as e:
        print(f"{mode:5} FAILED: {type(e).__name__}: {e}"); continue
    dt = time.perf_counter() - t0
    a = np.array(res[0].audio, copy=False); sr = getattr(res[0], "sample_rate", 24000)
    out = f"voice_ref2/clone_{TAG}_{mode}.wav"
    sf.write(out, a, sr)
    print(f"{mode:5} {dt:5.2f}s | {len(a)/sr:5.2f}s audio | mlx peak {mx.get_peak_memory()/1e6:.0f}MB "
          f"| free {psutil.virtual_memory().available/1e6:.0f}MB -> {out}")
