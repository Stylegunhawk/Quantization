# TTS Model Testing — Findings

Tested on: MacBook (Apple Silicon, MPS backend), macOS. Env managed with `uv`.
Test sentence: 88 chars ("The quick brown fox jumps over the lazy dog, testing
lightweight offline text to speech.")

## Models tested

| Model | Params | License | Voice | Notes |
|---|---|---|---|---|
| Kokoro-82M | 82M | Apache 2.0 | multi-voice | best quality, needs GPU/MPS to be fast |
| Piper | ~few M (onnx) | MIT | per-voice-file | CPU-only, fastest, smallest footprint |
| Inflect-Micro-v2 | 9.36M | Apache 2.0 | fixed, English only | newer/less proven, slower than Piper on CPU per its own benchmarks |

## Results (see `results.csv` for raw log)

Kokoro's first run includes a one-time weight download — the 18.7s row below
is cold-start, not steady-state. Kokoro's `gpu_mem_mb` was `0` in early runs
due to a measurement bug (see Bugs section, #4) — fixed, and its real MPS
footprint is now visible.

| model | device | latency (s) | RTF | ram spike (MB) | gpu mem (MB) |
|---|---|---|---|---|---|
| piper | cpu | 0.23–0.70 | 0.04–0.09 | ~97–314 | 0 |
| kokoro-82m (cold) | mps | 18.74 | 3.22 | -135 (see note) | n/a (pre-fix) |
| kokoro-82m (warm, pre-fix) | mps | 4.95–11.54 | 0.85–1.98 | 8.5 | n/a (pre-fix) |
| kokoro-82m (warm, post-fix) | mps | 2.07–4.02 | 0.36–0.65 | 11 (see note) | **582.9–819.3** |
| inflect-micro-v2 | mps | 1.07–4.02 | 0.16–0.75 | 55–150 | 41.7 |

Same 316-char input, all three, post-fix — the cleanest apples-to-apples
comparison so far:

| | piper | inflect-micro-v2 | kokoro-82m |
|---|---|---|---|
| latency | 0.70s | 3.21s | 4.02s |
| RTF | 0.039 | 0.162 | 0.649 |
| gpu mem | 0 (cpu-only) | 41.4 MB | 819.3 MB |

Piper is ~5.7x faster than Kokoro and stays CPU-only. Kokoro's real GPU
footprint (819 MB) is ~20x Inflect's, consistent with its 82M vs 9.36M
parameter count — that cost buys noticeably more natural prosody (pause/
intonation handling from punctuation), which Inflect's smaller model doesn't
attempt.

Negative RAM spikes on some Kokoro rows are an artifact of the baseline
sample landing right after a GC/model-load memory drop, not a real deficit —
treat single-run RAM/CPU spike numbers as noisy; `results.csv` has multiple
runs per model to average over.

## Recommendation

- **Piper** is the clear pick for "fastest, lightest, CPU-only" — no GPU
  needed, sub-second latency, MIT license.
- **Kokoro-82M** gives noticeably better voice quality at the cost of needing
  MPS/CUDA to stay fast; on CPU-only it would likely lose to Piper on speed.
- **Inflect-Micro-v2** doesn't beat either on this hardware — kept only as a
  smallest-footprint reference point (9.36M params, fixed voice).

## Bugs found while testing (fixed in this folder)

1. **Piper API rename** — `piper-tts` 1.6.0 renamed
   `PiperVoice.synthesize(text, wav_file)` to `.synthesize_wav(...)`.
   `run_piper.py` uses the new name.
2. **Kokoro's spaCy dependency blocked by network** — `misaki` (Kokoro's
   G2P) auto-downloads spaCy's `en_core_web_sm` from a GitHub release URL;
   `github.com` was unreachable on this network. Worked around by installing
   the Hugging Face mirror (`spacy/en_core_web_sm`) instead — see
   `README.md` Setup section.
3. **Inflect's `requirements.txt` breaks Kokoro** — installing plain
   `phonemizer` (pulled in by Inflect-Micro-v2's own requirements) overwrites
   files that Kokoro's `phonemizer-fork` needs at the same import path.
   README's setup steps skip that line; repair command included there too.
4. **Kokoro's `gpu_mem_mb` always read `0.0`** — `run_kokoro.py` built the
   `KPipeline` *inside* the function passed to `measure()`, so it (and its
   MPS tensors) went out of scope and got freed the instant that function
   returned, before `measure()` sampled `torch.mps.current_allocated_memory()`.
   Fixed by moving pipeline construction outside `measure()`, matching
   `run_piper.py`/`run_inflect.py` (model load outside, only inference
   timed) — Kokoro's `results.csv` rows before this fix are not comparable
   to rows after it.

Full setup/run instructions: `../README.md`.
