import gc
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

from metrics import measure, log_result

TEXT = "The quick brown fox jumps over the lazy dog, testing lightweight offline text to speech."
OUT_DIR = Path(__file__).parent
SAMPLE_RATE = 24000
VARIANTS = ["bf16", "8bit", "4bit"]


def synthesize(model) -> np.ndarray:
    chunks = [r.audio for r in model.generate(TEXT, voice="af_heart", lang_code="a")]
    return np.array(mx.concatenate(chunks))


def main() -> None:
    for variant in VARIANTS:
        repo = f"mlx-community/Kokoro-82M-{variant}"
        model = load_model(repo)
        synthesize(model)  # warm-up: first call pays a one-time Metal kernel-compile tax
        # MLX uses Apple Silicon's unified memory, not torch's MPS allocator, so GPU usage
        # isn't observable through track_gpu — gpu_mem_mb is n/a. mlx_peak_mb reads MLX's own
        # allocator instead and is the real footprint; ram_peak_mb only corroborates it.
        audio, run_metrics = measure(lambda: synthesize(model), "mps", track_gpu=False)
        sf.write(OUT_DIR / f"output_kokoro_mlx_{variant}.wav", audio, SAMPLE_RATE)
        log_result(OUT_DIR / "results.csv", f"kokoro-mlx-{variant}", len(TEXT), len(audio) / SAMPLE_RATE, run_metrics)
        print(variant, run_metrics)

        # Free before the next variant loads. Without this the previous variant's weights are
        # still resident when the next one is measured, so mlx_peak_mb accumulates instead of
        # reporting a per-variant footprint (observed: 1501 -> 1774 -> 2054 MB, monotonic).
        del model
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
