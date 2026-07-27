# TTS Model Testing Folder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `quant/tts_models/` — runnable scripts to test Kokoro-82M and Piper TTS locally, logging latency/CPU/RAM/GPU-spike metrics to a CSV for comparison.

**Architecture:** Two shared helpers (`device.py` for device detection, `metrics.py` for resource-spike measurement + CSV logging) consumed by two independent run scripts (`run_kokoro.py`, `run_piper.py`), each producing a `.wav` and a row in `results.csv`.

**Tech Stack:** Python, `torch`, `kokoro`, `piper-tts`, `psutil`, `soundfile`.

## Global Constraints
- No custom error handling beyond what libraries raise (spec: "Error handling" section) — this is a manual test harness, not a service.
- Device priority: `cuda` → `mps` → `cpu` (spec: `device.py`).
- Metrics baseline is sampled immediately before synthesis (current system load), not idle-system baseline (spec: `metrics.py`).
- Git: never commit in this repo unless explicitly asked; when asked, single-line message, no trailers ([[feedback_git_commits]] memory).

---

### Task 1: Shared helpers — `device.py` and `metrics.py`

**Files:**
- Create: `quant/tts_models/device.py`
- Create: `quant/tts_models/metrics.py`
- Test: `quant/tts_models/test_smoke.py` (helper portion only, extended in Task 3)

**Interfaces:**
- Produces: `device.pick_device() -> str` (one of `"cuda"`, `"mps"`, `"cpu"`)
- Produces: `metrics.measure(fn: Callable[[], T], device: str) -> tuple[T, dict]` — dict keys: `device, latency_s, cpu_baseline_pct, cpu_peak_pct, cpu_spike_pct, ram_baseline_mb, ram_peak_mb, ram_spike_mb, gpu_mem_mb`
- Produces: `metrics.log_result(csv_path: Path, model: str, text_chars: int, audio_seconds: float, metrics: dict) -> None` — appends one row to `csv_path`, writing the header if the file is new

- [ ] **Step 1: Write `device.py`**

```python
import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

- [ ] **Step 2: Write `metrics.py`**

```python
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
```

- [ ] **Step 3: Sanity-check imports**

Run: `python -c "import sys; sys.path.insert(0, 'quant/tts_models'); import device, metrics; print(device.pick_device())"`
Expected: prints `cpu`, `mps`, or `cuda` with no traceback.

---

### Task 2: `run_kokoro.py`

**Files:**
- Create: `quant/tts_models/run_kokoro.py`

**Interfaces:**
- Consumes: `device.pick_device()`, `metrics.measure(fn, device)`, `metrics.log_result(csv_path, model, text_chars, audio_seconds, run_metrics)`
- Produces: `quant/tts_models/output_kokoro.wav`, one row in `quant/tts_models/results.csv` with `model="kokoro-82m"`

- [ ] **Step 1: Write `run_kokoro.py`**

```python
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

from device import pick_device
from metrics import measure, log_result

TEXT = "The quick brown fox jumps over the lazy dog, testing lightweight offline text to speech."
OUT_DIR = Path(__file__).parent
SAMPLE_RATE = 24000


def synthesize(device: str) -> np.ndarray:
    pipeline = KPipeline(lang_code="a", device=device)
    chunks = [audio for _, _, audio in pipeline(TEXT, voice="af_heart")]
    return np.concatenate(chunks)


def main() -> None:
    device = pick_device()
    audio, run_metrics = measure(lambda: synthesize(device), device)
    sf.write(OUT_DIR / "output_kokoro.wav", audio, SAMPLE_RATE)
    log_result(OUT_DIR / "results.csv", "kokoro-82m", len(TEXT), len(audio) / SAMPLE_RATE, run_metrics)
    print(run_metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd quant/tts_models && pip install -r requirements.txt && python run_kokoro.py`
Expected: prints a metrics dict, creates `output_kokoro.wav` and `results.csv`. If `KPipeline`'s signature differs from the installed `kokoro` version, adjust the constructor call to match `pip show kokoro`'s model card — this is a manual test script, not load-bearing production code.

---

### Task 3: `run_piper.py` and smoke test

**Files:**
- Create: `quant/tts_models/run_piper.py`
- Create: `quant/tts_models/test_smoke.py`

**Interfaces:**
- Consumes: same `device`/`metrics` helpers as Task 2
- Produces: `quant/tts_models/output_piper.wav`, one row in `results.csv` with `model="piper"`

- [ ] **Step 1: Write `run_piper.py`**

```python
import wave
from pathlib import Path

from piper import PiperVoice

from device import pick_device
from metrics import measure, log_result

TEXT = "The quick brown fox jumps over the lazy dog, testing lightweight offline text to speech."
OUT_DIR = Path(__file__).parent
VOICE_PATH = OUT_DIR / "en_US-lessac-medium.onnx"


def synthesize(voice: PiperVoice) -> None:
    with wave.open(str(OUT_DIR / "output_piper.wav"), "wb") as wav_file:
        voice.synthesize(TEXT, wav_file)


def main() -> None:
    # Piper is CPU-only (onnxruntime); device is logged for comparison, not used for compute.
    device = pick_device()
    voice = PiperVoice.load(str(VOICE_PATH))
    _, run_metrics = measure(lambda: synthesize(voice), "cpu")
    with wave.open(str(OUT_DIR / "output_piper.wav")) as wav_file:
        audio_seconds = wav_file.getnframes() / wav_file.getframerate()
    log_result(OUT_DIR / "results.csv", "piper", len(TEXT), audio_seconds, run_metrics)
    print(run_metrics)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `test_smoke.py`**

```python
import csv
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent


def _wav_is_valid(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def test_kokoro_runs_and_logs():
    subprocess.run([sys.executable, str(DIR / "run_kokoro.py")], check=True, cwd=DIR)
    assert _wav_is_valid(DIR / "output_kokoro.wav")
    rows = list(csv.DictReader((DIR / "results.csv").open()))
    assert any(r["model"] == "kokoro-82m" for r in rows)


def test_piper_runs_and_logs():
    subprocess.run([sys.executable, str(DIR / "run_piper.py")], check=True, cwd=DIR)
    assert _wav_is_valid(DIR / "output_piper.wav")
    rows = list(csv.DictReader((DIR / "results.csv").open()))
    assert any(r["model"] == "piper" for r in rows)


if __name__ == "__main__":
    test_kokoro_runs_and_logs()
    test_piper_runs_and_logs()
    print("smoke test passed")
```

- [ ] **Step 3: Run the smoke test**

Run: `cd quant/tts_models && python test_smoke.py`
Expected: prints `smoke test passed`; `output_kokoro.wav`, `output_piper.wav`, and `results.csv` (with both model rows) exist in the folder.

---

### Task 4: `requirements.txt` and `README.md`

**Files:**
- Create: `quant/tts_models/requirements.txt`
- Create: `quant/tts_models/README.md`

- [ ] **Step 1: Write `requirements.txt`**

```
torch
kokoro
piper-tts
soundfile
psutil
numpy
```

- [ ] **Step 2: Write `README.md`**

```markdown
# TTS Model Testing

Local sandbox to compare Kokoro-82M and Piper on latency, CPU/RAM/GPU spike.

## Setup (Mac or Windows)

    pip install -r requirements.txt

Piper needs a voice model downloaded once, next to this README:

    curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
    curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

## Run

    python run_kokoro.py
    python run_piper.py

Each run writes a `.wav` (listen and compare by ear) and appends one row to
`results.csv` — device used, latency, real-time-factor, and the CPU/RAM/GPU
spike caused by that run on top of whatever else was running on the machine
at the time.

## Smoke test

    python test_smoke.py
```

- [ ] **Step 3: Verify the whole folder end-to-end**

Run: `cd quant/tts_models && pip install -r requirements.txt && python test_smoke.py`
Expected: `smoke test passed`, `results.csv` has two rows (`kokoro-82m`, `piper`).

- [ ] **Step 4: Commit**

Per repo rule ([[feedback_git_commits]]): do NOT commit. Leave changes for the user to commit manually.
