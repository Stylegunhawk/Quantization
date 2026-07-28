import csv
import time
from pathlib import Path
from typing import Callable, TypeVar

import psutil
import torch

T = TypeVar("T")

FIELDNAMES = [
    "timestamp", "model", "device", "text_chars", "rtf",
    "latency_s", "cpu_pct",
    "ram_baseline_mb", "ram_peak_mb", "ram_spike_mb", "gpu_mem_mb",
]


def measure(fn: Callable[[], T], device: str, track_gpu: bool = True, track_process: bool = True) -> tuple[T, dict]:
    process = psutil.Process()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ram_baseline_mb = process.memory_info().rss / (1024 * 1024)

    # cpu_times() deltas across exactly the fn() call, NOT cpu_percent() sampling windows:
    # a cpu_percent(interval=0.1) before fn() covers 0.1s of idle *before* the work starts,
    # and one after it covers 0.1s of idle *after* the work has finished — neither window
    # contains any synthesis. Can exceed 100% when the backend uses multiple cores, which
    # is the useful signal.
    cpu_before = process.cpu_times()
    start = time.perf_counter()
    result = fn()
    latency_s = time.perf_counter() - start
    cpu_after = process.cpu_times()

    cpu_seconds = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    cpu_pct = cpu_seconds / latency_s * 100 if latency_s > 0 else 0.0
    ram_peak_mb = process.memory_info().rss / (1024 * 1024)

    # track_gpu=False means this backend's GPU/ANE usage isn't observable through torch's
    # allocators (e.g. MLX, or a system service like macOS `say`) — report n/a, not a false zero.
    gpu_mem_mb = 0.0 if track_gpu else None
    if track_gpu and device == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    elif track_gpu and device == "mps" and hasattr(torch, "mps"):
        gpu_mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)

    return result, {
        "device": device,
        "latency_s": round(latency_s, 4),
        # track_process=False means the work happens in another process (e.g. macOS `say`
        # delegates to a shared XPC service), so our own CPU and RSS measure nothing. Blank,
        # not 0 — a cpu_pct of 0.4 reads as "almost free", which is the opposite of known.
        "cpu_pct": round(cpu_pct, 1) if track_process else None,
        "ram_baseline_mb": round(ram_baseline_mb, 2) if track_process else None,
        "ram_peak_mb": round(ram_peak_mb, 2) if track_process else None,
        "ram_spike_mb": round(ram_peak_mb - ram_baseline_mb, 2) if track_process else None,
        "gpu_mem_mb": round(gpu_mem_mb, 2) if gpu_mem_mb is not None else None,
    }


def log_result(csv_path: Path, model: str, text_chars: int, audio_seconds: float, run_metrics: dict) -> None:
    is_new = not csv_path.exists()
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "text_chars": text_chars,
        "rtf": round(run_metrics["latency_s"] / audio_seconds, 4) if audio_seconds else 0,
        **run_metrics,
    }
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
