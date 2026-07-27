import sys
from pathlib import Path

import soundfile as sf
from huggingface_hub import snapshot_download

from device import pick_device
from metrics import measure, log_result

TEXT = "The quick brown fox jumps over the lazy dog, testing lightweight offline text to speech."
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR / "Inflect-Micro-v2"


def load_model(device: str):
    if not MODEL_DIR.exists():
        snapshot_download("owensong/Inflect-Micro-v2", local_dir=MODEL_DIR)
    sys.path.insert(0, str(MODEL_DIR))
    from inference import InflectTTS

    return InflectTTS(str(MODEL_DIR), device=device)


def main() -> None:
    device = pick_device()
    tts = load_model(device)
    (sample_rate, audio), run_metrics = measure(lambda: tts.synthesize(TEXT), device)
    sf.write(OUT_DIR / "output_inflect.wav", audio, sample_rate)
    log_result(OUT_DIR / "results.csv", "inflect-micro-v2", len(TEXT), len(audio) / sample_rate, run_metrics)
    print(run_metrics)


if __name__ == "__main__":
    main()
