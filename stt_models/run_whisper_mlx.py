"""openai/whisper-small via mlx-audio, mirroring voicebox's MLXSTTBackend.

`base` was tested and dropped — on both a TTS-generated clip and a real
human-voice clip it hallucinated worse AND ran slower than `small`
(see docs/RESULTS.md). `small` is the baseline going forward.

    ./.venv/bin/python run_whisper_mlx.py                     # whisper-small, 3 runs
    RUNS=5 ./.venv/bin/python run_whisper_mlx.py              # 5 runs
    WHISPER_LANGUAGE=en ./.venv/bin/python run_whisper_mlx.py # pin language, tail-loop test
    ./.venv/bin/python run_whisper_mlx.py openai/whisper-base # any repo
"""

import gc
import os
import sys
from pathlib import Path

import mlx.core as mx
import soundfile as sf
from mlx_audio.stt import load

from metrics import measure, measure_load, log_result

AUDIO_PATH = "sample_for_tts.wav"
RESULTS_CSV = Path("results.csv")
MODELS = sys.argv[1:] or ["openai/whisper-small"]
RUNS = int(os.environ.get("RUNS", "3"))
LANGUAGE = os.environ.get("WHISPER_LANGUAGE") or None

duration = sf.info(AUDIO_PATH).duration
kwargs = {"language": LANGUAGE} if LANGUAGE else {}


def extract(result) -> str:
    text = result.text if hasattr(result, "text") else (
        result.get("text") if isinstance(result, dict) else str(result)
    )
    return text.strip()


for repo in MODELS:
    label = repo.split("/")[-1] + "-mlx" + (f"-lang-{LANGUAGE}" if LANGUAGE else "")
    print(f"\n=== {label} ===")

    model, load_metrics = measure_load(lambda: load(repo))
    print(f"load: {load_metrics['load_s']}s | mlx peak {load_metrics['mlx_load_peak_mb']} MB "
          f"| swap +{load_metrics['load_swap_delta_mb']} MB")

    # Warm-up, discarded: the first call pays a one-time Metal kernel-compile tax. Without
    # this, run 1 measures compilation and every later run measures inference — the exact
    # confound tts_models/docs/findings.md ("MLX cold-start caveat") already fixed for TTS
    # and this folder had not carried over.
    model.generate(AUDIO_PATH, **kwargs)

    for run in range(1, RUNS + 1):
        result, run_metrics = measure(lambda: model.generate(AUDIO_PATH, **kwargs))
        text = extract(result)
        print(f"run {run}: {run_metrics['latency_s']}s | RTF {run_metrics['latency_s'] / duration:.4f} "
              f"| mlx peak {run_metrics['mlx_peak_mb']} MB | rss spike {run_metrics['ram_spike_mb']} MB "
              f"| cpu {run_metrics['cpu_pct']}%/{run_metrics['cpu_peak_pct']}% peak "
              f"| swap +{run_metrics['swap_delta_mb']} MB | free min {run_metrics['sys_avail_min_mb']} MB")
        if run == 1:
            print(f"  transcript: {text!r}")
        log_result(
            RESULTS_CSV, model=label, text=text, audio_seconds=duration,
            run_metrics=run_metrics, run=run,
            load_s=load_metrics["load_s"], mlx_load_peak_mb=load_metrics["mlx_load_peak_mb"],
        )

    del model
    gc.collect()
    mx.clear_cache()
