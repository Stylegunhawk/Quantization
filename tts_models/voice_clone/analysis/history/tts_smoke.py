"""Smoke test: does Qwen3-TTS load and clone on this machine at all?"""
import time, mlx.core as mx, soundfile as sf
from mlx_audio.tts.utils import load_model

REPO = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
t0 = time.perf_counter()
m = load_model(REPO)
print(f"loaded in {time.perf_counter()-t0:.1f}s | mlx peak {mx.get_peak_memory()/1e6:.0f}MB")
print("type:", type(m).__name__)
print("has generate:", hasattr(m, "generate"), "| speaker_encoder:", getattr(m, "speaker_encoder", None) is not None)
print("speech_tokenizer:", getattr(m, "speech_tokenizer", None) is not None,
      "| has_encoder:", getattr(getattr(m, "speech_tokenizer", None), "has_encoder", None))
