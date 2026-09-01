"""N samples per mode - generation is stochastic, single runs don't separate modes."""
import sys, time, numpy as np, mlx.core as mx, soundfile as sf
from mlx_audio.tts.utils import load_model
REF, TXT, TAG, N = sys.argv[1], open(sys.argv[2]).read().strip(), sys.argv[3], int(sys.argv[4])
TARGET = "Parakeet transcribes twenty one seconds of audio in under half a second on this machine."
m = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
print(f"loaded {mx.get_peak_memory()/1e6:.0f}MB | ref {sf.info(REF).duration:.1f}s")
for mode, kw in [("icl", dict(ref_audio=REF, ref_text=TXT)), ("xvec", dict(ref_audio=REF))]:
    for i in range(1, N+1):
        mx.reset_peak_memory(); t0=time.perf_counter()
        res = list(m.generate(text=TARGET, **kw))
        a = np.array(res[0].audio, copy=False); sr = getattr(res[0],"sample_rate",24000)
        out = f"voice_ref3/{TAG}_{mode}_{i}.wav"; sf.write(out, a, sr)
        print(f"  {mode:5} run{i} {time.perf_counter()-t0:5.2f}s {len(a)/sr:5.2f}s peak {mx.get_peak_memory()/1e6:.0f}MB -> {out}")
