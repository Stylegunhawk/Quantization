import csv
import time
from pathlib import Path
from typing import Callable, TypeVar

import psutil
import torch

T = TypeVar("T")

FIELDNAMES = [
    "timestamp", "model", "device", "text_chars", "rtf",
    "latency_s", "cpu_baseline_pct", "cpu_peak_pct", "cpu_spike_pct",
    "ram_baseline_mb", "ram_peak_mb", "ram_spike_mb", "gpu_mem_mb",
]


def measure(fn: Callable[[], T], device: str) -> tuple[T, dict]:
    process = psutil.Process()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    cpu_baseline_pct = psutil.cpu_percent(interval=0.1)
    ram_baseline_mb = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()
    result = fn()
    latency_s = time.perf_counter() - start

    cpu_peak_pct = psutil.cpu_percent(interval=0.1)
    ram_peak_mb = process.memory_info().rss / (1024 * 1024)

    gpu_mem_mb = 0.0
    if device == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    elif device == "mps" and hasattr(torch, "mps"):
        gpu_mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)

    return result, {
        "device": device,
        "latency_s": round(latency_s, 4),
        "cpu_baseline_pct": cpu_baseline_pct,
        "cpu_peak_pct": cpu_peak_pct,
        "cpu_spike_pct": round(cpu_peak_pct - cpu_baseline_pct, 2),
        "ram_baseline_mb": round(ram_baseline_mb, 2),
        "ram_peak_mb": round(ram_peak_mb, 2),
        "ram_spike_mb": round(ram_peak_mb - ram_baseline_mb, 2),
        "gpu_mem_mb": round(gpu_mem_mb, 2),
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
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
