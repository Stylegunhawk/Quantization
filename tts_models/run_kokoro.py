from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

from device import pick_device
from metrics import measure, log_result

TEXT = "The quick brown fox jumps over the lazy dog, testing lightweight offline text to speech."
OUT_DIR = Path(__file__).parent
SAMPLE_RATE = 24000


def synthesize(pipeline: KPipeline) -> np.ndarray:
    chunks = [audio for _, _, audio in pipeline(TEXT, voice="af_heart")]
    return np.concatenate(chunks)


def main() -> None:
    device = pick_device()
    pipeline = KPipeline(lang_code="a", device=device)
    audio, run_metrics = measure(lambda: synthesize(pipeline), device)
    sf.write(OUT_DIR / "output_kokoro.wav", audio, SAMPLE_RATE)
    log_result(OUT_DIR / "results.csv", "kokoro-82m", len(TEXT), len(audio) / SAMPLE_RATE, run_metrics)
    print(run_metrics)


if __name__ == "__main__":
    main()
