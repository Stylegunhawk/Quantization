# STT Model Testing — Findings

Tested on: MacBook M1, 8GB RAM (~2.7GB available, 1.27GB swap already in use
before the session started — this machine runs under sustained memory
pressure), Apple Silicon / MLX. Env managed with `uv`, Python 3.11.

Test clip: `sample_for_tts.wav` — a real human-voice recording (not
TTS-generated), 7.25s, converted from `.m4a` via macOS's native `afconvert`
(no extra deps needed; `miniaudio` can't decode AAC/M4A directly).

**All numbers below are from the 2026-08-12 re-run** with a warm-up call and
3 timed runs per model, each model in its own process. The earlier
single-run numbers are archived in `../results_pre_warmup_fix.csv` and are
**not comparable** — they were dominated by one-time Metal kernel
compilation (Bug #1). Two of the three headline claims in the previous
version of this document were wrong; see "Corrections" at the bottom.

## Models tested

| Model | Runtime | Size on disk | Notes |
|---|---|---|---|
| `openai/whisper-base` | mlx-audio | 144MB | **Dropped** — see below |
| `openai/whisper-small` | mlx-audio | 481MB | Whisper baseline |
| `mlx-community/Qwen3-ASR-0.6B-4bit` | mlx-audio | 708MB | **Fastest, and the recommendation** |
| `mlx-community/Qwen3-ASR-1.7B-4bit` | mlx-audio | 1.6GB | Largest tested; heaviest at runtime |

All run through `mlx_audio.stt` — `load()` for Whisper,
`load_model()`/`generate_transcription()` for Qwen3-ASR (mlx-audio's own
API surface, same library, different entry points per its docs).

## Results

Medians of 3 timed runs (whisper: 6, across two separate processes).
`results.csv` has every row, now including the full transcript text.

| model | RTF (median) | spread | latency (s) | load (s) | MLX peak, load | MLX peak, inference | CPU avg / peak | swap Δ | min free RAM |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-ASR-0.6B-4bit** | **0.0643** | 1.88% | 0.466 | 0.75 | 681MB | 1311MB | 44.0% / 78.8% | 0 | 1480MB |
| Qwen3-ASR-1.7B-4bit | 0.1496 | 1.21% | 1.085 | 1.11 | 1535MB | 2314MB | 28.8% / 110.7% | 0 | 927MB |
| whisper-small (`language="en"`) | 0.3153 | 0.73% | 2.286 | 2.95 | 992MB | 1170MB | 43.9% / 97.2% | 0 | 1853MB |
| whisper-small (auto-detect) | 0.3506 | 0.46% | 2.543 | 3.80 | 841MB | 1170MB | 40.6% / 97.1% | 0 | 1660MB |

### The harness is now precise enough to trust

Run-to-run spread is **0.46–1.88%**. whisper-small was run twice — first and
last in the session, in separate processes — and landed at 0.3506 vs 0.3502
median (full range 0.3497–0.3513, **±0.23%**). Machine drift over the whole
session is negligible, so any gap above ~2% below is a real difference, not
noise. The old harness showed a 1.54× swing on this same model/clip.

### Qwen3-ASR-0.6B is the fastest, by a wide margin

- **5.5× faster than whisper-small** (RTF 0.0643 vs 0.3506) and 2.3× faster
  than the 1.7B variant.
- **0.6B beats 1.7B on speed by 2.33×** (0.0643 vs 0.1496). Both spreads are
  under 2%, so this is unambiguous.
- 0.6B also loads fastest by far: **0.75s vs whisper-small's 3.80s**.

### No tail hallucination on either Qwen3-ASR size — and pinning the language does not fix Whisper's

whisper-small emits **49 periods** after the real speech ends (a repeated
`. . . .` loop). Qwen3-ASR emits 2–3, i.e. normal sentence punctuation only.
This reproduced on every run of every configuration.

Setting `language="en"` (the open question from the previous version of this
doc) **does not suppress the loop** — still exactly 49 periods, byte-identical
transcript. It does make Whisper **10.1% faster** (RTF 0.3153 vs 0.3506) by
skipping language detection, and cuts load time from 3.80s to 2.95s. Worth
setting for the speed, useless for the hallucination.

### On-disk size does not predict runtime footprint

The interesting memory result: **0.6B-4bit's inference peak (1311MB) is
*higher* than whisper-small's (1170MB)** despite a 708MB vs 481MB checkpoint,
and it is 630MB above its own load peak. Activations, not weights, dominate
the peak for the Qwen models. Ranking by checkpoint size would get this
backwards.

### Is the 1.7B checkpoint safe on this 8GB machine? Yes, measured — but it is the tightest

Concrete answer to the previous version's open question:

- MLX peak during inference: **2314MB** (1535MB just to load).
- Minimum free system RAM observed during its runs: **927MB** — the lowest of
  any model, and the only one under 1GB.
- **Swap did not grow during any measured window** (load or inference) for any
  model, including the 1.7B. Session-wide swap use went 1274MB → 1395MB
  (+121MB) over ~15 minutes, but no measured window coincided with the
  increase, so that is background activity, not this benchmark.

So it runs without thrashing, but it is 2.3× slower than the 0.6B while using
~1GB more peak memory and leaving under 1GB of headroom. There is no
measured reason to prefer it on this hardware.

### CPU% is a weak signal here, as expected

The 1.7B has the **lowest** average CPU (28.8%) and the **highest** peak
(110.7%) — it is the most GPU-bound of the set. MLX GPU work does not appear
in `cpu_times()`, so a low `cpu_pct` never means a cheap run. Read it as
"how much of the work spilled onto CPU", nothing more.

### Accuracy: one word error each, and Whisper's is new

| model | disputed word | tail |
|---|---|---|
| whisper-small | "solv**ed** across both sizes" | 49 periods |
| Qwen3-ASR-0.6B | "**Webb item**" for "verbatim" | clean |
| Qwen3-ASR-1.7B | "**Verb item**" for "verbatim" | clean |

Both Qwen sizes independently transcribe "that part is **solid** across both
sizes"; whisper-small says "**solved**". Two independent models agreeing
suggests "solid" is correct and **whisper-small has its own substitution
error** — which the previous version of this doc missed because it only
tracked the word "verbatim". Whisper still wins on that one word.

⚠️ **Not verified against a written ground truth.** There is no reference
transcript for this clip, so no WER is computed anywhere in this folder —
every accuracy statement above is a model-consensus argument, not a measured
error rate. Writing down the actual sentence that was recorded would make all
of it checkable; until then treat the accuracy column as weaker evidence than
the timing columns.

## Why `whisper-base` was dropped

Tested on two clips (a Kokoro-TTS-generated one, and this real-voice one).
On both, `base` was garbled with a wildly hallucinated tail (foreign words,
language tags, unrelated content) where `small` produced a near-correct
transcript. That accuracy gap is qualitative and large, so it survives the
timing corrections below — but note the *speed* half of the original argument
("base was also slower") rested on unwarmed numbers and should be treated as
unmeasured. Accuracy alone is sufficient reason to keep it dropped.

## Recommendation

**Qwen3-ASR-0.6B-4bit.** Fastest (5.5× whisper-small), fastest to load,
no tail hallucination, and 550MB less peak memory than the 1.7B. Its one
known weakness is the "verbatim" → "Webb item" substitution.

Use whisper-small only where that specific class of word accuracy matters
more than a 5.5× speed difference, and if so pass `language="en"` (free 10%)
and strip trailing punctuation runs in post-processing, because the model
will not stop emitting them.

Skip the 1.7B on this hardware.

## Bugs found while testing (fixed in this folder)

1. **Every timing number was measuring Metal kernel compilation** — neither
   run script did a warm-up call, so the first (and in the old harness,
   *only*) timed run per process paid a one-time kernel-compile tax. Effect
   was large and non-uniform: whisper-small 0.6072 → 0.3506 RTF, and
   Qwen-0.6B 0.2725 → 0.0643 (**4.2×** overstated). This is the same bug
   `tts_models/docs/findings.md` documents under "MLX cold-start caveat" and
   had **already fixed** in `run_kokoro_mlx.py:25` — this folder was copied
   from `tts_models` but never carried the fix across. Both `run_*.py` now
   do a discarded warm-up call before the timed loop.
   *Same caveat applies as in the TTS doc:* Metal's shader cache also
   persists on disk across processes, so these are steady-state-on-a-warmed-
   machine numbers, not first-ever-run.
2. **`ram_peak_mb` was not a peak** — `metrics.py` read RSS once *after* the
   call returned, so it was a post-hoc snapshot. `metrics.py` now polls RSS,
   swap and CPU on a 50ms sampler thread for the duration of the call.
3. **Negative `ram_spike_mb` on every row was misdiagnosed** — the previous
   version of this doc blamed model loading happening outside `measure()`.
   That cannot produce a negative number (it would produce ≈0). The real
   cause is two-fold: (a) on this memory-pressured machine the OS *evicts*
   pages during inference, so RSS genuinely falls; and (b) MLX allocates its
   weights during **load**, so the inference window was never going to show
   the allocation at all — measured RSS spike during Qwen inference is
   0.08–2.2MB, because the 758MB/1081MB spike happens in the load phase.
   The old harness was watching the wrong window. `tts_models/docs/findings.md`
   carries the same "negative RAM spikes are a GC artifact" hand-wave — that
   explanation is also incomplete for the MLX rows there.
4. **RAM was measured with the wrong instrument entirely** — RSS cannot see
   MLX's unified-memory allocations reliably under paging. MLX exposes its own
   allocator counters; `metrics.py` now records `mx.get_peak_memory()` around
   both load and inference (`mlx_load_peak_mb`, `mlx_peak_mb`). These are the
   memory numbers to read; the RSS columns are kept only as corroboration.
   `tts_models` reports `n/a` for MLX GPU memory (its Bug #7) — this API would
   fill that gap there too.
5. **`torch` was inflating the baseline by 170MB and putting a false column in
   the CSV** — measured: `import torch` costs 170.5MB RSS, which was 84% of
   whisper's 202MB "baseline". MLX never touches torch. `device.py` existed
   only to write `mps` into the CSV, which was not a fact about the run (MLX
   picks its own default device; torch's MPS availability is irrelevant).
   `device.py` and the `torch` requirement are deleted; the column now reads
   `mlx`.
6. **`results.csv` did not contain the transcripts** — the column was
   `text_chars` only, so every accuracy claim in this document was
   unrecoverable from the "raw log" it cited. There is now a `text` column.
7. **Both Qwen models shared one process, so neither had a clean memory
   baseline** — the first model's weights were still resident when the
   second's baseline was taken (visible as `ram_baseline_mb` 372 → 395MB in
   the archived CSV). `run_qwen3_asr_mlx.py` now takes the repo as an
   argument so each model gets its own process, and frees the model
   (`del` / `gc.collect()` / `mx.clear_cache()`) between iterations if you do
   pass several.
8. **`str(result)` could silently log the string `"None"` as a transcript** —
   `generate_transcription` writes to `output_path` and returning `None`
   would have been logged as a valid 4-character result. Now raises.

Found later, in `dictate_app.py` rather than the benchmark, but the same class of
measurement mistake:

9. **An unset `blocksize` on the input stream cost 80% of the audio and made
   the app look broken.** `sounddevice`'s default `blocksize=0` lets PortAudio
   choose; on this machine it chose **15 frames** — a callback every 0.94ms,
   ~1067 per second. Each one takes the GIL, which starved the process's main
   thread badly enough that (a) the `NSEvent` hotkey monitors stopped seeing
   keystrokes, so the stop press was swallowed, (b) `AppHelper.callAfter` never
   painted the floating panel, and (c) PortAudio dropped the input it couldn't
   hand over in time — a 3.4s recording arrived as **0.68s** of audio. All three
   symptoms read as "the GUI and the shortcut don't work", and none of them are
   GUI bugs. Setting `blocksize` to 100ms (1600 frames) fixed all three: 40
   callbacks instead of ~4200, and 4.00s captured from a 4s window.

   Worth noting for the streaming work later: at 15-frame blocks the per-callback
   Python overhead dominates completely, so any chunked/streaming pipeline needs
   an explicit block size before its first-token-latency number means anything.
10. **Qwen3-ASR does not return an empty string for silence — it confabulates, in
    a language it picks itself.** Two recordings of an empty room came back as
    fluent Devanagari (`काके सप्पी राज भिवर…`) and Chinese (`嗯，你们那个苹果。`),
    with no error and normal-looking RTF. With auto-paste on, that text goes
    straight into whatever document is focused. Two fixes, both in
    `dictate_app.py`: `language="en"` is now pinned (`generate_transcription`
    forwards it to `model.generate`), and a clip whose peak amplitude never
    exceeds `SILENCE_PEAK` is rejected before the model ever sees it.

    This also puts a caveat on the accuracy table above: a model that answers
    confidently when given nothing will do the same when given something it can't
    make out, so WER on clean speech is an optimistic bound on how it behaves on
    a real dictation clip that starts mid-breath.
11. **The panel sent the user to the Desktop instead of overlaying the fullscreen app
    they were in** — and then back again a moment later. Two wrong fixes preceded
    the right one, both from the same wrong premise: that the *window* needed
    different properties.

    - **Wrong fix 1: window level.** The theory was that `NSFloatingWindowLevel`
      (3) cannot be placed on a fullscreen app's Space, so macOS switches you to a
      Space where the window can live. `NSStatusWindowLevel` (25) plus
      `…Stationary` changed nothing, which also refutes the theory: `speak_app.py`
      floats at level 3 over fullscreen apps every day.
    - **Wrong fix 2: keeping focus.** The panel was a
      `NSWindowStyleMaskNonactivatingPanel` shown with `orderFront_`, deliberately
      never activating, because the synthetic ⌘V at the end of a recording goes to
      whatever holds focus. That is the actual cause. **An accessory app that
      orders a window front without activating does not get placed on your
      Space** — `CanJoinAllSpaces` and `FullScreenAuxiliary` notwithstanding.
      Activating is what puts it where you are, and an accessory app activating
      does not switch Spaces, which is exactly what `speak_app.py`'s comment on
      that line says.

    Fixed by matching `speak_app.py` exactly — floating level, two collection
    behaviors, `unhide_` + `activateIgnoringOtherApps_` +
    `makeKeyAndOrderFront_` — and solving the paste separately:
    `Recorder._restore_focus()` records the frontmost app *before* showing the
    panel, hides the app afterwards (`hide_`, not just `orderOut_`), then polls
    until that app is frontmost again before posting ⌘V. If it cannot confirm
    within 1s the transcript is left on the clipboard rather than typed somewhere
    unintended. Measured: focus returns in 0.06–0.11s, so polling costs nothing
    against the fixed sleep it replaces.

    A side benefit: the panel holds focus while recording, so `esc` now cancels a
    recording — previously impossible, and noted as such in the code.

    The lesson is about diagnosis, not AppKit: the first two attempts changed a
    property of the *window* when the difference between the working app and the
    broken one was a property of the *app*. Both apps' panels were configured
    almost identically; only the activation differed, and it was commented out on
    purpose.

12. **Spotlight indexed the bundle and still would not offer it.** Searching
    "dictate" listed the *Dictation* keyboard setting, `dictate_app.py`, its
    `.pyc`, an unrelated `DictateWindow/` folder, and even a clipboard entry
    containing the text "Dictate.app" — everything except the app. Yet the index
    holds it correctly: `kMDItemKind = "Application"`,
    `kMDItemDisplayName = "Dictate"`, `kMDItemUseCount = 4`, and
    `mdfind -name Dictate` returns it first. So this is not an indexing failure,
    and `lsregister -f` plus `mdimport` change nothing — both already succeed.
    What the app bundle lacked was a **standard app location**; the Spotlight UI
    appears to route app bundles to its Applications section and source that
    section from LaunchServices' app directories, so a bundle living in a repo
    folder falls out of both the app list and the file list.

    The obvious fix does not work: a plain symlink at
    `~/Applications/Dictate.app` is invisible to Spotlight —
    `mdfind -onlyin ~/Applications Dictate` returns nothing, because the index
    stores the symlink, not what it points at. What does work is a real
    directory whose *contents* are the symlink:

        mkdir -p ~/Applications/Dictate.app
        ln -sfn "$PWD/Dictate.app/Contents" ~/Applications/Dictate.app/Contents

    Now `mdfind -onlyin ~/Applications Dictate` returns the path and its kind is
    `Application`, with still exactly one copy of the code — the stub's `cd -P`
    resolves back to the repo, so editing `dictate_app.py` needs no reinstall.
    Verified by launching it: model loaded, Accessibility still granted (macOS
    attributes it to the physical path, which is unchanged), menu bar icon 39x30.
    The one cost is cosmetic: `codesign -v` reports "unsealed contents present in
    the bundle root", and `spctl -a` rejects the installed bundle — but it rejects
    the in-repo bundle identically, since both are ad-hoc signed and unnotarized,
    and neither is quarantined, so nothing is actually gated.

    Unconfirmed: that a standard location is *what* the Spotlight UI requires is
    inferred from the index evidence above, not observed in the UI.

13. **The hotkey fired *and* the browser acted on it.** ⌥⌘R is reading mode in
    both Brave and Chrome, so one press started a recording and reformatted the
    page. This is not a collision to be dodged by picking another combination —
    it is that a global `NSEvent` monitor can only **observe**. The keystroke is
    delivered to the frontmost app no matter what the handler does, so any
    combination some app has claimed would double-fire, and the app that claims
    it next year is unknowable today.

    Fixed with a `CGEventTap` on `kCGSessionEventTap`, which sits ahead of
    delivery: returning `None` from the callback drops the event, so the browser
    never sees ⌥⌘R. It also replaced *both* monitors rather than one — a session
    tap sees events bound for this app too, which is the only reason the local
    monitor existed. Same Accessibility grant; a refused tap returns `None` from
    `CGEventTapCreate`, which is now reported instead of leaving a dead hotkey.

    Two things the tap needs that a monitor did not:

    - macOS **disables a tap whose callback was too slow** and never re-enables
      it, so the hotkey would die mid-session with nothing in the log. The
      disable is delivered as an event, so `kCGEventTapDisabledByTimeout` is
      handled by calling `CGEventTapEnable` again.
    - The run-loop source must be kept alive alongside the tap. Dropping it
      unregisters the tap silently.

    ponytail cost, recorded in the code: a session-wide tap hands *every*
    keystroke to this process to own one combination. Carbon's
    `RegisterEventHotKey` consumes one key and sees no others, which is the right
    shape, but it needs ctypes structs and a Carbon event handler. Nothing here
    reads what keys mean, only their codes — but `DICTATE_DEBUG=1` prints those,
    so it is not something to leave on while typing a password.

    Verified end to end through the bundle with a synthetic ⌥⌘R:
    `keyCode=15 flags=0x180000 hit=True` (swallowed), panel on the active Space
    at level 3 with Dictate frontmost, 4.10s of audio transcribed in 0.56s, and
    the app's own ⌘V visible in the same trace as `keyCode=9 flags=0x100000
    hit=False` — passed through, which is also proof the focus handoff completed.

## Next steps

- Write down the clip's actual sentence and compute real WER — the only
  remaining unmeasured axis (see the accuracy warning above).
- Add Parakeet-tdt-0.6b-v3 (mlx-community) for a third comparison point;
  `README.md` already claims it is tested and it is not.
- ~~Consider back-porting fixes #1–#4 to `tts_models/metrics.py`.~~
  **Done 2026-08-13** (its bugs #9–#12). It kept torch, since Kokoro-PyTorch and
  Inflect genuinely need it, and its `results.csv` was migrated in place rather
  than archived. It paid off twice: MLX-Kokoro's footprint is now a number
  (~1.46–1.50GB) instead of `n/a`, which **retracted** that doc's claim that
  MLX avoids Kokoro's GPU memory cost — MLX is ~0.7GB *heavier* than the
  PyTorch backend there, not lighter. Its RAM columns are now split across
  2026-08-13 the same way this folder's are.
- Test on a longer clip. `tts_models` found that ranking changed between an
  88-char and a 231-char input; 7.25s of audio may be short enough that
  per-call fixed overhead still dominates.

## Corrections to the previous version of this document

| Previous claim | Status |
|---|---|
| "Qwen3-ASR is ~2.2-3x faster than whisper-small" | **Understated.** 0.6B is 5.5×. |
| "1.7B was faster than 0.6B in this run" | **Wrong, and reversed.** 0.6B is 2.33× faster. Cause was the warm-up confound, not noise. |
| "Negative ram_spike is because load happens before measure()" | **Wrong mechanism.** See Bug #3. |
| "Neither Qwen3-ASR size nailed 'verbatim'" | **Holds**, but whisper-small has its own error ("solved" vs "solid") that was missed. |
| "No tail hallucination on Qwen3-ASR" | **Holds.** Reproduced on all runs. |
