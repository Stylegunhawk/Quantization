# TTS Model Testing Folder — Design

## Purpose
A local sandbox to try out lightweight, fully offline TTS models — Kokoro-82M
and Piper — and compare their latency and resource cost, as a precursor to a
possible future cross-platform "read selected text" desktop utility (separate,
not-yet-scoped project).

## Scope
In scope: two runnable scripts (Kokoro, Piper), a shared device-detection
helper, a shared resource-metrics helper, and a CSV log of runs.
Out of scope: the desktop app itself (global shortcut, accessibility text
capture, native OS web search) — a separate future spec.

## Location
`quant/tts_models/` inside this repo.

## Components

- **`requirements.txt`** — `kokoro`, `piper-tts`, `torch`, `soundfile`, `psutil`.
- **`device.py`** — `pick_device()` returns `"cuda"` → `"mps"` → `"cpu"`
  (first available), so `run_kokoro.py`/`run_piper.py` work unmodified on the
  Mac (MPS) and the Windows/CUDA box.
- **`metrics.py`** — `measure(fn)` context/wrapper that:
  - samples CPU% and process RAM (via `psutil`) immediately before calling `fn`
    (this is the "current load" baseline, not an idle-system baseline — the
    user wants to know the spike on top of whatever else is running)
  - samples peak CPU%/RAM during/after the call
  - if CUDA is active, reads `torch.cuda.max_memory_allocated()`; if MPS,
    reads `torch.mps.current_allocated_memory()` (best-effort — MPS exposes no
    utilization%, only memory)
  - returns a dict: `latency_s, cpu_baseline_pct, cpu_peak_pct, cpu_spike_pct,
    ram_baseline_mb, ram_peak_mb, ram_spike_mb, gpu_mem_mb, device`
- **`run_kokoro.py`** / **`run_piper.py`** — load the model, synthesize a fixed
  test sentence, save a `.wav`, wrap the synthesis call in `metrics.measure`,
  append one row (model name, text length, RTF, + all metrics fields above) to
  `results.csv` (created with header if missing).
- **`README.md`** — per-OS setup: pip install steps, and the one-time Piper
  voice `.onnx` download.
- **`test_smoke.py`** — runs both scripts on a short string; asserts each
  output `.wav` is non-empty/valid audio and that `results.csv` gained a row.

## Data flow
`run_*.py` → `device.pick_device()` → model loads on that device →
`metrics.measure(synthesize_fn)` runs synthesis while sampling →
`.wav` written + one row appended to `results.csv`.

## Error handling
None beyond what the libraries raise — this is a local manual test harness,
not a service. If Piper's voice file is missing, the library's own
`FileNotFoundError` is sufficient; no custom wrapping.

## Testing
`test_smoke.py` is the single self-check (per the checklist above) — not a
full test suite, just enough to catch "the script is broken."
