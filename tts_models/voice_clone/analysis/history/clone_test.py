"""Zero-shot clone smoke test: both modes, same reference clip."""
import time, mlx.core as mx, soundfile as sf, numpy as np
from mlx_audio.tts.utils import load_model

REF_WAV  = "../stt_models/captures/20260817-120355.wav"
REF_TEXT = "My name is Siddhesh. Let's see if we can make this better."
TARGET   = "Parakeet transcribes twenty one seconds of audio in under half a second on this machine."

m = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
print(f"loaded | peak {mx.get_peak_memory()/1e6:.0f}MB")

for tag, kw in [("icl", dict(ref_audio=REF_WAV, ref_text=REF_TEXT)),
                ("xvec", dict(ref_audio=REF_WAV))]:
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    res = list(m.generate(text=TARGET, **kw))
    dt = time.perf_counter() - t0
    a = res[0].audio
    a = np.array(a, copy=False) if not isinstance(a, np.ndarray) else a
    sr = getattr(res[0], "sample_rate", 24000)
    out = f"clone_{tag}.wav"
    sf.write(out, a, sr)
    print(f"{tag:5} {dt:5.2f}s  {len(a)/sr:5.2f}s audio @{sr}Hz  peak {mx.get_peak_memory()/1e6:.0f}MB -> {out}")
