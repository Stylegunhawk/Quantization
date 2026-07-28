import subprocess
from pathlib import Path

import soundfile as sf

from metrics import measure, log_result

TEXT = "The quick brown fox jumps over the lazy dog, testing lightweight offline text to speech."
OUT_DIR = Path(__file__).parent
OUT_PATH = OUT_DIR / "output_say.wav"
VOICE = "Samantha"


def synthesize() -> None:
    subprocess.run(
        ["say", "-v", VOICE, "-o", str(OUT_PATH), "--data-format=LEI16@22050", TEXT],
        check=True,
    )


def main() -> None:
    # `say` is a thin client: the synthesis happens in a shared, on-demand XPC service, so
    # this process's own CPU and RSS say nothing about the cost of the work. They are
    # reported as n/a rather than as misleading near-zero numbers. Latency and RTF are
    # still real — they measure wall-clock to produce the file, whoever did the work.
    #
    # Sampling the service's RSS instead was tried and dropped. It was never comparable to
    # Kokoro's model-resident memory (a shared, already-warm daemon with voice data
    # page-cached gives an RSS delta, not a footprint), and it wasn't even reliable: the
    # service idles out and `say -o file` does not dependably spawn it, so the lookup found
    # nothing at all on this machine. findings.md already labels the number non-comparable.
    _, run_metrics = measure(synthesize, "cpu", track_gpu=False, track_process=False)
    audio_seconds = sf.info(OUT_PATH).duration
    log_result(OUT_DIR / "results.csv", "macos-say", len(TEXT), audio_seconds, run_metrics)
    print(run_metrics)


if __name__ == "__main__":
    main()
