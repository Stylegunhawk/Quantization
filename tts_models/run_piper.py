import wave
from pathlib import Path

from piper import PiperVoice

from device import pick_device
from metrics import measure, log_result

TEXT = "Installing Inflect-Micro-v2's own requirements.txt pulls in vanilla phonemizer, which overwrites files that Kokoro's phonemizer-fork needs at the same import path — broke Kokoro until I uninstalled phonemizer and force-reinstalled phonemizer-fork. README now tells you to skip that line and lists the repair command."
OUT_DIR = Path(__file__).parent
VOICE_PATH = OUT_DIR / "en_US-lessac-medium.onnx"


def synthesize(voice: PiperVoice) -> None:
    with wave.open(str(OUT_DIR / "output_piper.wav"), "wb") as wav_file:
        voice.synthesize_wav(TEXT, wav_file)


def main() -> None:
    # Piper runs on onnxruntime (CPU); device is logged for comparison, not used for compute.
    device = pick_device()
    voice = PiperVoice.load(str(VOICE_PATH))
    _, run_metrics = measure(lambda: synthesize(voice), "cpu")
    with wave.open(str(OUT_DIR / "output_piper.wav")) as wav_file:
        audio_seconds = wav_file.getnframes() / wav_file.getframerate()
    log_result(OUT_DIR / "results.csv", "piper", len(TEXT), audio_seconds, run_metrics)
    print(run_metrics)


if __name__ == "__main__":
    main()
