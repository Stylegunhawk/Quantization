"""Measurement harness for the TTS benchmark.

Unlike `stt_models/metrics.py` (which started as a copy of this file and then
diverged), this one keeps torch: Kokoro-PyTorch and Inflect are real torch/MPS
models here, and `gpu_mem_mb` is a genuine number for them. What was back-ported
from the STT folder is the memory instrumentation — a polling sampler instead of a
single post-hoc RSS read, plus MLX's own allocator counters. See
`docs/findings.md` bugs #9-#11.
"""

import csv
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

import psutil
import torch

try:  # only installed for the MLX variants; the torch back ends don't need it
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None

T = TypeVar("T")

# New columns are appended, never inserted, so older rows stay readable as trailing blanks.
FIELDNAMES = [
    "timestamp", "model", "device", "text_chars", "rtf",
    "latency_s", "cpu_pct",
    "ram_baseline_mb", "ram_peak_mb", "ram_spike_mb", "gpu_mem_mb",
    "cpu_peak_pct", "mlx_peak_mb", "swap_delta_mb", "sys_avail_min_mb",
]

MB = 1024 * 1024


class _Sampler(threading.Thread):
    """Polls RSS, swap, free RAM and per-interval CPU% while the measured call runs.

    A single RSS read after the call is a snapshot, not a peak — and on a
    memory-pressured Mac it can land *below* the baseline once the OS evicts pages,
    which is what produced the negative "spikes" in this log's older rows.
    """

    def __init__(self, process: psutil.Process, interval: float = 0.05) -> None:
        super().__init__(daemon=True)
        self.process = process
        self.interval = interval
        self._done = threading.Event()  # not `_stop`: that name shadows Thread._stop()
        self.peak_rss_mb = process.memory_info().rss / MB
        self.peak_swap_mb = psutil.swap_memory().used / MB
        self.min_avail_mb = psutil.virtual_memory().available / MB
        self.peak_cpu_pct = 0.0

    def run(self) -> None:
        prev_cpu, prev_t = self.process.cpu_times(), time.perf_counter()
        while not self._done.is_set():
            self._done.wait(self.interval)
            now = time.perf_counter()
            cpu = self.process.cpu_times()
            elapsed = now - prev_t
            if elapsed > 0:
                busy = (cpu.user - prev_cpu.user) + (cpu.system - prev_cpu.system)
                self.peak_cpu_pct = max(self.peak_cpu_pct, busy / elapsed * 100)
            prev_cpu, prev_t = cpu, now
            self.peak_rss_mb = max(self.peak_rss_mb, self.process.memory_info().rss / MB)
            self.peak_swap_mb = max(self.peak_swap_mb, psutil.swap_memory().used / MB)
            self.min_avail_mb = min(self.min_avail_mb, psutil.virtual_memory().available / MB)

    def stop(self) -> None:
        self._done.set()
        self.join(timeout=2)


def measure(fn: Callable[[], T], device: str, track_gpu: bool = True, track_process: bool = True) -> tuple[T, dict]:
    process = psutil.Process()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ram_baseline_mb = process.memory_info().rss / MB
    swap_baseline_mb = psutil.swap_memory().used / MB

    # reset_peak_memory() zeroes MLX's high-water mark; the next allocation re-sets it to
    # current *active* memory, so the value read back after fn() is resident weights +
    # activations. This is the MLX equivalent of gpu_mem_mb, which torch cannot see for
    # MLX models (docs/findings.md bug #7 reported those as n/a).
    if mx is not None:
        mx.reset_peak_memory()

    sampler = _Sampler(process)
    sampler.start()

    # cpu_times() deltas across exactly the fn() call, NOT cpu_percent() sampling windows:
    # a cpu_percent(interval=0.1) before fn() covers 0.1s of idle *before* the work starts,
    # and one after it covers 0.1s of idle *after* the work has finished — neither window
    # contains any synthesis. Can exceed 100% when the backend uses multiple cores, which
    # is the useful signal. Counts CPU only: GPU work does not appear here, so a low
    # cpu_pct is not evidence of a cheap run.
    cpu_before = process.cpu_times()
    start = time.perf_counter()
    result = fn()
    latency_s = time.perf_counter() - start
    cpu_after = process.cpu_times()
    sampler.stop()

    cpu_seconds = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    cpu_pct = cpu_seconds / latency_s * 100 if latency_s > 0 else 0.0

    # track_gpu=False means this backend's GPU/ANE usage isn't observable through torch's
    # allocators (e.g. MLX, or a system service like macOS `say`) — report n/a, not a false
    # zero. For MLX specifically, mlx_peak_mb below now carries the real number.
    gpu_mem_mb = 0.0 if track_gpu else None
    if track_gpu and device == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated() / MB
    elif track_gpu and device == "mps" and hasattr(torch, "mps"):
        gpu_mem_mb = torch.mps.current_allocated_memory() / MB

    return result, {
        "device": device,
        "latency_s": round(latency_s, 4),
        # track_process=False means the work happens in another process (e.g. macOS `say`
        # delegates to a shared XPC service), so our own CPU and RSS measure nothing. Blank,
        # not 0 — a cpu_pct of 0.4 reads as "almost free", which is the opposite of unknown.
        "cpu_pct": round(cpu_pct, 1) if track_process else None,
        "cpu_peak_pct": round(sampler.peak_cpu_pct, 1) if track_process else None,
        "ram_baseline_mb": round(ram_baseline_mb, 2) if track_process else None,
        "ram_peak_mb": round(sampler.peak_rss_mb, 2) if track_process else None,
        "ram_spike_mb": round(sampler.peak_rss_mb - ram_baseline_mb, 2) if track_process else None,
        "gpu_mem_mb": round(gpu_mem_mb, 2) if gpu_mem_mb is not None else None,
        # Blank when the work isn't in this process (bug #7's principle): a 0 there would be
        # "no MLX allocations here", which says nothing about what the other process did.
        # For an in-process CPU backend like Piper, 0 is a genuine confirmed zero.
        "mlx_peak_mb": round(mx.get_peak_memory() / MB, 2) if mx is not None and track_process else None,
        "swap_delta_mb": round(sampler.peak_swap_mb - swap_baseline_mb, 2),
        "sys_avail_min_mb": round(sampler.min_avail_mb, 2),
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


def demo() -> None:
    """Self-check: a known allocation and a known busy-wait must show up as a peak."""
    def work():
        blob = bytearray(64 * MB)  # torch-free allocation, so this runs without MLX too
        deadline = time.perf_counter() + 0.3
        while time.perf_counter() < deadline:
            pass
        return len(blob)

    result, m = measure(work, "cpu")
    assert result == 64 * MB
    assert 0.3 <= m["latency_s"] < 1.0, m
    assert m["ram_peak_mb"] >= m["ram_baseline_mb"]  # a polled peak can never go below baseline
    assert m["ram_spike_mb"] >= 60, m                # the 64MB bytearray is visible in RSS
    assert m["cpu_peak_pct"] > 50, m                 # the busy-wait pins a core
    assert m["gpu_mem_mb"] == 0.0, m                 # cpu device, torch reports a true zero
    print("metrics self-check passed:", m)


if __name__ == "__main__":
    demo()
