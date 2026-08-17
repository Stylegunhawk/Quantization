# TTS Model Testing — Findings

Tested on: MacBook (Apple Silicon, MPS backend), macOS. Env managed with `uv`.
Short sentence: 88 chars ("The quick brown fox jumps over the lazy dog, testing
lightweight offline text to speech."). Longer/natural sentence: 231 chars (a
short-story definition, more representative of real prose than the pangram).

## Models tested

| Model | Params | License | Voice | Notes |
|---|---|---|---|---|
| Kokoro-82M (PyTorch/MPS) | 82M | Apache 2.0 | multi-voice | best quality, needs GPU/MPS to be fast |
| Kokoro-82M (MLX) | 82M | Apache 2.0 | multi-voice | same weights, ported to Apple's MLX framework; bf16/8bit/4bit variants |
| Piper | ~few M (onnx) | MIT | per-voice-file | CPU-only, fastest, smallest footprint |
| Inflect-Micro-v2 | 9.36M | Apache 2.0 | fixed, English only | newer/less proven, slower than Piper on CPU per its own benchmarks |
| macOS `say` (native) | n/a (system) | Apple (built-in) | many system voices | AVSpeechSynthesizer via the `say` CLI — ships with the OS, no install |

## Results (see `results.csv` for raw log)

Kokoro's first run includes a one-time weight download — the 18.7s row is
cold-start, not steady-state. Kokoro's `gpu_mem_mb` was `0` in early runs due
to a measurement bug (Bugs #4) — fixed, real MPS footprint now visible.

**Data hygiene note (2026-07-27):** `metrics.py` had two more measurement
bugs fixed after these rows were logged (Bugs #6, #7 below) — CPU% columns
before this fix are system-wide noise, not per-process, and `gpu_mem_mb: 0`
in rows before the fix doesn't distinguish "confirmed zero" (Piper, genuinely
CPU-only) from "not observable through torch" (`say`, MLX) — both used to
read `0`. Rows logged after the fix report a blank `gpu_mem_mb` for the
latter case instead. Treat CPU% and `gpu_mem_mb=0` in older rows as noise/
ambiguous, not signal.

**Data hygiene note (2026-08-13):** the RAM columns changed meaning. Every
`ram_peak_mb` logged before this date is a single RSS read taken *after* the
call returned — a snapshot, not a peak (Bugs #9, #10) — which is why so many
`ram_spike_mb` values are negative. Rows from 2026-08-13 on are a polled peak
and can never fall below their baseline, so **RAM columns are not comparable
across that date**; the negative values are artifacts, not measurements. Four
columns were also appended (`cpu_peak_pct`, `mlx_peak_mb`, `swap_delta_mb`,
`sys_avail_min_mb`) and are blank for older rows. Latency and RTF columns are
unaffected throughout.

**88-char sentence:**

| model | device | latency (s) | RTF | ram spike (MB) | gpu mem (MB) |
|---|---|---|---|---|---|
| piper | cpu | 0.23–0.70 | 0.04–0.09 | ~97–314 | 0 |
| kokoro-82m, pytorch (cold) | mps | 18.74 | 3.22 | -135 (see note) | n/a (pre-fix) |
| kokoro-82m, pytorch (warm, pre-fix) | mps | 4.95–11.54 | 0.85–1.98 | 8.5 | n/a (pre-fix) |
| kokoro-82m, pytorch (warm, post-fix) | mps | 0.78–4.02 | 0.13–0.65 | 11–30 (see note) | **582.9–819.3** |
| kokoro-82m, mlx bf16/8bit/4bit (warm) | mps | 0.79–0.80 | 0.135–0.136 | noisy, no clean staircase by precision | n/a (see note) |
| inflect-micro-v2 | mps | 1.07–4.02 | 0.16–0.75 | 55–150 | 41.7 |
| macos-say (native) | cpu | 0.88–1.01 | 0.17–0.19 | not comparable (see note) | n/a |

**231-char sentence (all six, one shot) — the most apples-to-apples
comparison so far, and it changes the ranking:**

| model | latency (s) | RTF | ram peak/spike (MB) | gpu mem (MB) |
|---|---|---|---|---|
| piper | 0.73 | 0.048 | 313 (+46) | 0 |
| kokoro-mlx-8bit | 2.18 | 0.153 | 32 (-29) | n/a |
| kokoro-mlx-4bit | 2.35 | 0.156 | 36 (-22) | n/a |
| kokoro-mlx-bf16 | 2.32 | 0.163 | 33 (-1) | n/a |
| inflect-micro-v2 | 2.88 | 0.203 | 224 (-96) | 41.9 |
| macos-say (native) | 3.92 | **0.281** | not comparable (see note) | n/a |
| kokoro-82m, pytorch | 5.25 | 0.360 | 203 (-103) | 702.9 |

`say`'s RTF looked competitive (0.17–0.19) on the short pangram but is
**second-worst of six on the longer sentence** (0.281) — it doesn't scale as
well as Piper or MLX-Kokoro. Don't generalize from the 88-char number alone;
the recommendation below uses the 231-char ranking.

**MLX-Kokoro vs PyTorch-Kokoro**, same weights in principle, is the other
big change from the longer sentence: on 88 chars both back ends were ~0.79s
(no visible difference); on 231 chars MLX is **~2.2x faster** (2.2–2.4s vs
5.25s). ~~and doesn't carry Kokoro's 700+ MB GPU footprint the same way~~ —
**that memory claim was wrong** and is retracted: it inferred an advantage from
a blank column. MLX's own allocator reports **~1.46–1.50 GB** during synthesis,
*more* than PyTorch-Kokoro's 583–819MB `gpu_mem_mb`, so pick MLX for latency,
not for memory (Bug #11). Quantization
(bf16 vs 8bit vs 4bit) made no measurable latency difference in either test —
all three variants land within ~0.02–0.03s of each other. If precision
matters for quality, there's no speed reason to pick a smaller quantization;
that call is an audio-quality one (see `output_kokoro_mlx_*.wav` — a manual
listen is needed to judge whether 4bit is audibly worse).

Negative RAM spikes on some rows are an artifact of the baseline sample
landing right after a GC/model-load memory drop, not a real deficit — treat
single-run RAM/CPU spike numbers as noisy; `results.csv` has multiple runs
per model to average over.

**MLX cold-start caveat:** the very first run of all three MLX variants in
one process logged 10.2s/2.6s/3.0s (`results.csv` rows for the first
`kokoro-mlx-*` batch) — a one-time Metal kernel-compile tax, not steady
state. `run_kokoro_mlx.py` now does an in-process warm-up call before
measuring (fixed the *within-run* problem — bf16 went from 10.18s to 0.79s
in the very next run), but Metal's shader cache also persists **on disk
across processes**, so even a "cold" fresh-process run afterward benefits
from whatever an earlier run already compiled. There's no cheap way to test
truly-fresh-machine latency without invalidating that system cache — treat
the numbers above as steady-state-on-a-warmed-machine, not first-ever-run.

**`macos-say`'s RAM figure isn't a footprint, and isn't this Python
process** — `say` is a thin client; actual synthesis runs in a persistent,
shared per-user system daemon (`SpeechSynthesisServerXPC`). `run_say.py`
samples that process's RSS directly (Bugs #5) instead of the wrapper's. But
it's an RSS delta on an *already-warm, shared* daemon over a ~1s window,
with voice data likely mmap'd/page-cached from prior use — not a private-
process footprint the way Kokoro's GPU number is. Don't read "~2 MB vs
Kokoro's 819 MB" as "700x lighter" — it's not the same kind of measurement.
No GPU/ANE number is reported (`n/a`) because CoreSpeech doesn't go through
Metal/torch, and there's no public per-process query API for the Neural
Engine without `sudo powermetrics`.

## Recommendation

Based on the 231-char ranking (the more representative comparison):

- **Piper** is still the fastest and most portable — CPU-only, no GPU
  needed, sub-second latency at every length tested, MIT license, runs
  anywhere. Default choice unless native quality or Mac-only voice naturalness
  matters more than raw speed.
- **Kokoro-82M via MLX** is the way to get Kokoro's voice quality without
  PyTorch/MPS's cost: ~2.2x faster than the PyTorch backend on longer text,
  same weights, no crash risk from precision tricks (PyTorch's MPS backend
  hard-crashed under `.half()` and errored under `autocast` on this model —
  MLX doesn't hit that wall). If natural prosody matters and you're
  Mac-only, this beats plain PyTorch-Kokoro on latency and stability — but
  **not on memory**, where it is now measured ~0.7GB *heavier* (Bug #11).
- **macOS `say` (native)** is zero-install and cheapest on RAM for a short
  phrase, but its RTF degrades on longer text (second-worst of six at 231
  chars) — pick it for short, one-off utterances (notifications, prompts),
  not for narrating longer text where Piper or MLX-Kokoro win on speed.
- **Kokoro-82M (plain PyTorch/MPS)** — superseded by the MLX port above for
  this hardware; keep only if you specifically need the PyTorch ecosystem
  (fine-tuning, existing PyTorch pipeline integration).
- **Inflect-Micro-v2** doesn't beat any of the above on this hardware — kept
  only as a smallest-footprint reference point (9.36M params, fixed voice).

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
5. **`macos-say`'s RAM reading measured the wrong process** — `run_say.py`
   originally sampled `psutil.Process()` (the Python wrapper), but `say`
   dispatches synthesis to `SpeechSynthesisServerXPC`, a separate long-lived
   system daemon — so the reading was near-meaningless. `results.csv`'s
   `macos-say` row before this fix (`ram_peak_mb` ~187) isn't comparable to
   later rows.
   *Correction (2026-08-13):* this entry used to describe the fix as a `pid`
   argument on `measure()` that samples the daemon instead. That approach was
   tried and **dropped** — see the comment in `run_say.py:27-31`: the service
   idles out and `say -o file` doesn't dependably spawn it, so the lookup
   often found nothing, and even when it worked the number was an RSS delta on
   a warm shared daemon, not a footprint. `measure()` has no `pid` parameter.
   The actual fix is `track_process=False`, which reports CPU and RAM as blank
   (`n/a`) for this backend rather than as a wrong number.
6. **CPU% columns measured the whole system, not the process** —
   `metrics.py` called `psutil.cpu_percent(interval=0.1)` (module-level),
   which is system-wide CPU utilization, not `process.cpu_percent()`. This
   is why `cpu_spike_pct` went negative on so many rows across every model —
   it was picking up unrelated background load, not the workload being
   measured. Fixed to call `process.cpu_percent()` on the actual process
   being measured (primed with a throwaway call first, since `psutil`
   returns `0.0` on a process's first `cpu_percent()` call). CPU% columns in
   rows before this fix reflect system noise, not the model.
7. **`gpu_mem_mb: 0` conflated "confirmed zero" with "not observable"** —
   Piper's `0` (genuinely CPU-only, no GPU code path exists) and MLX's/
   `say`'s `0` (GPU/ANE may be in use, but neither goes through torch's
   allocator, so there was nothing to read) were indistinguishable in the
   column — anyone averaging it would conclude MLX uses no GPU memory, which
   isn't what was measured. `measure()` now takes a `track_gpu` flag; when
   `False`, `gpu_mem_mb` is written as blank (`n/a`), not `0`. `run_say.py`
   and `run_kokoro_mlx.py` both pass `track_gpu=False`.
8. **`results.csv` kept re-acquiring CRLF line endings** — the repo-level
   `.gitattributes` fix (`eol=lf`) normalizes the file in git, but Python's
   `csv` module defaults `lineterminator` to `\r\n` regardless of OS, so
   every `log_result()` append wrote CRLF straight back into the working-tree
   file — the "CRLF will be replaced by LF" warning would have recurred on
   every `git add` indefinitely. Fixed by passing `lineterminator="\n"`
   explicitly to `csv.DictWriter`; existing file content was normalized to
   LF in place to match.
9. **`ram_peak_mb` was never a peak** — `measure()` read RSS once *after* the
   call returned, so every "peak" in this log is a post-hoc snapshot.
   `metrics.py` now polls RSS, swap, free system RAM and per-interval CPU% on a
   50ms sampler thread for the duration of the call, and adds a `cpu_peak_pct`
   column. Immediate payoff: Piper's average CPU is 335% but its **peak is
   405%** — the averages were hiding how many cores it actually pins.
10. **"Negative RAM spikes are a GC artifact" was the wrong explanation** —
    the note above blamed the baseline sample landing after a GC/model-load
    memory drop. Two real causes, found while fixing the same symptom in
    `../stt_models`: (a) on this 8GB machine under sustained memory pressure
    the OS **evicts pages during synthesis**, so RSS genuinely falls below
    baseline; and (b) MLX allocates weights at **load** time, outside the
    measured window, so the inference window was never going to contain the
    allocation. A snapshot-minus-snapshot across the wrong window can only
    produce noise. Fixed by #9 (a polled peak cannot go below its baseline).
11. **MLX memory was reported as `n/a` when MLX exposes its own counters** —
    bug #7 correctly stopped writing a false `0` for backends torch can't see,
    but stopped there. `mx.get_peak_memory()` / `mx.reset_peak_memory()` report
    MLX's own allocator high-water mark. `metrics.py` now logs `mlx_peak_mb`,
    so MLX-Kokoro's footprint is a number instead of a blank: **~1.46-1.50 GB**
    during synthesis of the 88-char sentence. Note this is *larger* than
    PyTorch-Kokoro's 583-819MB `gpu_mem_mb`, so the earlier suggestion that
    MLX "doesn't carry Kokoro's 700+ MB GPU footprint the same way" was
    reading a measurement gap as an advantage. The two counters still aren't
    strictly comparable (unified memory vs a torch MPS allocation), but MLX is
    clearly not cheaper on memory — its win is latency.
12. **`run_kokoro_mlx.py` never freed a variant before loading the next** — all
    three ran in one process, so each variant's `mlx_peak_mb` included the
    previous ones' resident weights. Observed as a monotonic climb of
    1501 → 1774 → 2054 MB, which reads as "4bit uses the most memory" when the
    truth is the opposite of an ordering — it was just last. Fixed with
    `del model` / `gc.collect()` / `mx.clear_cache()` per iteration; the
    numbers then flatten to **1501 / 1461 / 1465 MB**, i.e. **quantization
    buys no runtime memory either**, matching the existing finding that it buys
    no latency. That run also showed the unfreed version pushing the machine
    into swap (`swap_delta_mb` 367, free RAM down to 612MB) where the freed
    version stays at zero swap.

Bugs #9-#12 were found in `../stt_models` first (see
`../stt_models/docs/RESULTS.md`, where the same measurement bugs invalidated
two of three headline findings) and back-ported here on 2026-08-13.

Additionally checked: Apple's Foundation Models framework (macOS 26+,
on-device LLM — text generation/tool-calling, not TTS) compiles and runs
fine from a plain `swiftc` command-line binary on this machine, no Xcode
project needed. `SystemLanguageModel.default.availability` currently
reports `unavailable(appleIntelligenceNotEnabled)` — the API works, Apple
Intelligence just isn't toggled on for this Mac/account
(System Settings → Apple Intelligence & Siri). Not otherwise relevant to
TTS comparison; noted here since it was tested in the same session.

Full setup/run instructions: `../README.md`.
