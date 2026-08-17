"""Measurement harness for the STT benchmark.

Diverged from `tts_models/metrics.py` (which it started as a copy of) on purpose:
this folder is MLX-only, so torch is gone entirely, and RAM is sampled by a polling
thread plus MLX's own allocator counters instead of a single post-hoc RSS read.
See `docs/RESULTS.md` for why the RSS-snapshot version produced negative spikes.
The memory instrumentation here was back-ported to `tts_models/metrics.py` (its
bugs #9-#12) on 2026-08-13; that copy keeps torch, since its models need it.
"""

import csv
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

import mlx.core as mx
import psutil

T = TypeVar("T")

FIELDNAMES = [
    "timestamp", "model", "run", "device", "text_chars", "rtf",
    "latency_s", "load_s", "cpu_pct", "cpu_peak_pct",
    "ram_baseline_mb", "ram_peak_mb", "ram_spike_mb",
    "mlx_peak_mb", "mlx_load_peak_mb",
    "swap_delta_mb", "sys_avail_min_mb", "text",
]

MB = 1024 * 1024


class _Sampler(threading.Thread):
    """Polls RSS, swap, free RAM and per-interval CPU% while the measured call runs.

    A single RSS read after the call is a snapshot, not a peak — and on a
    memory-pressured Mac it can land *below* the baseline once the OS evicts pages,
    which is how this benchmark ended up logging negative "spikes". Polling is the
    cheapest thing that actually sees a peak.
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


def measure(fn: Callable[[], T], device: str = "mlx", track_process: bool = True) -> tuple[T, dict]:
    process = psutil.Process()
    ram_baseline_mb = process.memory_info().rss / MB
    swap_baseline_mb = psutil.swap_memory().used / MB

    # reset_peak_memory() zeroes the high-water mark; the next allocation re-sets it to
    # current *active* memory, so the value read back after fn() is resident weights +
    # activations — the real MLX footprint, which RSS can't see reliably under paging.
    mx.reset_peak_memory()

    sampler = _Sampler(process)
    sampler.start()

    # cpu_times() deltas across exactly the fn() call, NOT cpu_percent() sampling windows:
    # a cpu_percent(interval=0.1) before fn() covers 0.1s of idle *before* the work starts,
    # and one after it covers 0.1s of idle *after* the work has finished — neither window
    # contains any inference. Can exceed 100% when the backend uses multiple cores, which
    # is the useful signal. Note this counts CPU only: MLX work on the GPU does not appear
    # here, so a low cpu_pct is not evidence of a cheap run.
    cpu_before = process.cpu_times()
    start = time.perf_counter()
    result = fn()
    latency_s = time.perf_counter() - start
    cpu_after = process.cpu_times()
    sampler.stop()

    cpu_seconds = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    cpu_pct = cpu_seconds / latency_s * 100 if latency_s > 0 else 0.0

    return result, {
        "device": device,
        "latency_s": round(latency_s, 4),
        # track_process=False means the work happens in another process, so our own CPU and
        # RSS measure nothing. Blank, not 0 — a cpu_pct of 0.4 reads as "almost free", which
        # is the opposite of unknown.
        "cpu_pct": round(cpu_pct, 1) if track_process else None,
        "cpu_peak_pct": round(sampler.peak_cpu_pct, 1) if track_process else None,
        "ram_baseline_mb": round(ram_baseline_mb, 2) if track_process else None,
        "ram_peak_mb": round(sampler.peak_rss_mb, 2) if track_process else None,
        "ram_spike_mb": round(sampler.peak_rss_mb - ram_baseline_mb, 2) if track_process else None,
        "mlx_peak_mb": round(mx.get_peak_memory() / MB, 2),
        "swap_delta_mb": round(sampler.peak_swap_mb - swap_baseline_mb, 2),
        "sys_avail_min_mb": round(sampler.min_avail_mb, 2),
    }


def measure_load(fn: Callable[[], T]) -> tuple[T, dict]:
    """Same instrumentation, for the model-load call — the cost `measure()` used to miss."""
    result, load_metrics = measure(fn)
    return result, {
        "load_s": load_metrics["latency_s"],
        "mlx_load_peak_mb": load_metrics["mlx_peak_mb"],
        "load_ram_spike_mb": load_metrics["ram_spike_mb"],
        "load_swap_delta_mb": load_metrics["swap_delta_mb"],
        "load_sys_avail_min_mb": load_metrics["sys_avail_min_mb"],
    }


def log_result(
    csv_path: Path,
    model: str,
    text: str,
    audio_seconds: float,
    run_metrics: dict,
    run: int = 1,
    load_s: float | None = None,
    mlx_load_peak_mb: float | None = None,
) -> None:
    is_new = not csv_path.exists()
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "run": run,
        # The transcript itself, not just its length: every accuracy claim in
        # docs/RESULTS.md was previously unrecoverable from this "raw log".
        "text": text,
        "text_chars": len(text),
        "rtf": round(run_metrics["latency_s"] / audio_seconds, 4) if audio_seconds else 0,
        "load_s": load_s,
        "mlx_load_peak_mb": mlx_load_peak_mb,
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
        big = mx.zeros((2048, 2048))  # 16MB fp32
        mx.eval(big)
        deadline = time.perf_counter() + 0.3
        while time.perf_counter() < deadline:
            pass
        return "ok"

    result, m = measure(work)
    assert result == "ok"
    assert 0.3 <= m["latency_s"] < 1.0, m
    assert m["mlx_peak_mb"] >= 16, m                 # the 16MB array is visible to MLX
    assert m["ram_peak_mb"] >= m["ram_baseline_mb"]  # a polled peak can never go below baseline
    assert m["cpu_peak_pct"] > 50, m                 # the busy-wait pins a core
    print("metrics self-check passed:", m)


if __name__ == "__main__":
    demo()
