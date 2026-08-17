"""Qwen3-ASR (0.6B / 1.7B, 4-bit MLX quant) via mlx-audio.

Pass one repo per process for a clean per-model memory reading — running both in
one process leaves the first model's weights resident while the second's baseline
is taken (that drift is visible in the pre-2026-08-12 results.csv rows).

    ./.venv/bin/python run_qwen3_asr_mlx.py mlx-community/Qwen3-ASR-0.6B-4bit
    RUNS=5 ./.venv/bin/python run_qwen3_asr_mlx.py mlx-community/Qwen3-ASR-1.7B-4bit
"""

import gc
import os
import sys
from pathlib import Path

import mlx.core as mx
import soundfile as sf
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.utils import load_model

from metrics import measure, measure_load, log_result

AUDIO_PATH = "sample_for_tts.wav"
RESULTS_CSV = Path("results.csv")
TRANSCRIPT_OUT = os.environ.get("TMPDIR", "/tmp") + "/qwen_asr_transcript"
MODELS = sys.argv[1:] or [
    "mlx-community/Qwen3-ASR-0.6B-4bit",
    "mlx-community/Qwen3-ASR-1.7B-4bit",
]
RUNS = int(os.environ.get("RUNS", "3"))

duration = sf.info(AUDIO_PATH).duration


def transcribe(model):
    return generate_transcription(
        model=model,
        audio=AUDIO_PATH,
        output_path=TRANSCRIPT_OUT,
        format="txt",
        verbose=False,
    )


def extract(result) -> str:
    if not hasattr(result, "text"):
        # generate_transcription writes to output_path and could return None; str(None)
        # would silently log the 4-char string "None" as if it were a transcript.
        raise RuntimeError(f"no .text on {type(result).__name__}: {result!r}")
    return result.text.strip()


for repo in MODELS:
    label = repo.split("/")[-1]
    print(f"\n=== {label} ===")

    model, load_metrics = measure_load(lambda: load_model(repo))
    print(f"load: {load_metrics['load_s']}s | mlx peak {load_metrics['mlx_load_peak_mb']} MB "
          f"| rss spike {load_metrics['load_ram_spike_mb']} MB "
          f"| swap +{load_metrics['load_swap_delta_mb']} MB "
          f"| free min {load_metrics['load_sys_avail_min_mb']} MB")

    # Warm-up, discarded — see the same comment in run_whisper_mlx.py. This is why the old
    # results.csv showed 1.7B "faster" than 0.6B: 0.6B ran first in-process and paid the
    # kernel-compile tax, 1.7B did not.
    transcribe(model)

    for run in range(1, RUNS + 1):
        result, run_metrics = measure(lambda: transcribe(model))
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
