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

> **Superseded 2026-08-31 by Parakeet-TDT-0.6B-v3** — 2.04× faster than the
> 0.6B and the only model to get this clip fully right. See that section below.
> What follows was the recommendation among the four models tested on 08-12.

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

## Vocabulary biasing: measured, and only partly effective

Qwen3-ASR takes a `system_prompt`, and `mlx_audio`'s `generate_transcription`
forwards it (filtering kwargs against the model's `generate` signature, so it
is silently dropped for models like Whisper that have no such parameter).
Feeding it a list of proper nouns is genuine context biasing, not a prompt
trick, and it costs ~0.03s on a 15–21s clip.

Two of this app's own captures happened to contain the failure, which makes
this the first accuracy claim in this document with real ground truth — the
words are ones I know were said:

| clip | no biasing | biasing | after the text pass |
|---|---|---|---|
| 111735 | "streaming **Nimoton** model" | "streaming **NemoTone**" | **Nemotron** ✓ |
| 111735 | "already done **QN** three years" | "**QN**" / "**Qwen**" ⁽¹⁾ | **Qwen3** ✓ |
| 111314 | "such as **NemoTone** Live Translate" | **Nemotron** ✓ | Nemotron ✓ |

⁽¹⁾ The bare comma-separated list left "QN" alone; wrapping it in a sentence
("Vocabulary that appears in this audio: …") fixed that word **and broke
ordinary ones** — "What other features or what other models should I test
next" became "What other features are other models should test next". That is
the whole reason biasing is not trusted on its own: the prompt wording
changes words nobody asked it to change. The app sends the bare list.

So the app applies three layers (`apply_vocab`), and the two deterministic
ones are what actually close the gap:

- **Explicit aliases** — the only mechanism that can fix a mangling which is
  itself an English word. `gwen` for Qwen3 is the case: fuzzy matching scores
  it 0.67, and the dictionary guard below protects it regardless, because
  "Gwen" is in `/usr/share/dict/words`.
- **Fuzzy match, cutoff 0.75, skipping dictionary words.** Chosen from
  measured ratios, not guessed: real errors land at 0.77–0.94 (a personal name
  from my own `vocab.txt` at 0.77 — the lowest seen, and what sets the cutoff —
  then `nimoton`→Nemotron 0.80, `nemotone`→0.88, `parakeets`→Parakeet 0.94)
  and the words that must not be rewritten land at
  0.67 (`when`→Qwen3, `mix`→MLX, `voltron`→Nemotron). A 0.10 margin is thin,
  which is why the dictionary guard exists as well — `when`, `mix`, `item`
  and `solid` are all in `web2`, so they are never candidates.

End to end through `--file`, both captures now come out with all three names
right. `--check` asserts every pair above, including the must-not-change ones.

⚠️ The dictionary guard cuts both ways: any mangling that happens to be a real
word is invisible to the fuzzy pass forever, and needs an alias. That is the
deliberate trade — a false rewrite of "when" into "Qwen3" in the middle of a
sentence is much worse than a missed fix.

## Next steps

- Write down the clip's actual sentence and compute real WER — the only
  remaining unmeasured axis (see the accuracy warning above). `captures.jsonl`
  is now the better corpus for this than the single benchmark clip.
- ~~Add Parakeet-tdt-0.6b-v3 (mlx-community) for a third comparison point;
  `README.md` already claims it is tested and it is not.~~ **Done 2026-08-31** —
  see "Parakeet-TDT-0.6B-v3" below. It is the new batch recommendation, and
  needed no new harness code. `README.md` corrected.
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
- ~~Streaming.~~ **Done 2026-08-17** — see "Streaming: Nemotron" below.
  `mlx-community/parakeet-tdt-0.6b-v3` is now tested in batch (below) but
  cannot stream — no cache-aware state is exposed for it.
  `Voxtral-Mini-4B-Realtime-2602-4bit` (3.1GB) stays deferred: its weights
  alone exceed the 1.7B Qwen's entire 2314MB peak.
- `metrics.py` still cannot evaluate a streaming model — it times one blocking
  call, so it has no column for time-to-first-token. The numbers below were
  measured by `dictate_app.py --live-file`, not by the benchmark harness.

## Streaming: Nemotron-3.5-ASR-streaming-0.6B (8-bit)

`mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit`, 756MB on disk. A
cache-aware streaming FastConformer-**RNN-T** with language-ID conditioning.
This is now live mode in `dictate_app.py`, selected from the menu bar's Model
submenu.

### It is cheaper than the model it replaces

| configuration | MLX peak |
|---|---|
| Nemotron 8-bit, load | 797MB |
| Nemotron 8-bit, streaming inference | **970MB** |
| Qwen3-ASR-0.6B-4bit, inference (for comparison) | 1311MB |
| **both models resident**, Qwen inference | **2425MB** |

The last row is the one that decided the design. 2425MB is worse than the 1.7B
Qwen alone, which already left this machine under 1GB free — so the app holds
**one model at a time** and live mode *replaces* Qwen3 rather than joining it,
which makes it the *lighter* configuration. A second hotkey holding both models
was the obvious design and the measurement ruled it out.

Swapping was then measured rather than assumed, since a naive switch would load
the new model while the old one is still alive — the 2425MB case exactly. With
the old model dropped first (`del`, `gc.collect()`, `mx.clear_cache()`), MLX's
peak resets to **0MB** in 0.10s, so each swap peaks at one model's cost:

| model | mode | peak, loaded and warmed | after free |
|---|---|---|---|
| Qwen3-ASR 0.6B | batch | 1375MB | 0MB |
| Qwen3-ASR 1.7B | batch | 2426MB | 0MB |
| Nemotron 0.6B | live | **942MB** | 0MB |

Swaps take 1.5–2.5s including the warm-up. `--models --swap` reproduces the
table; the 1.7B is the tightest configuration the app can be put into, and the
streaming model is the cheapest.

### Less look-ahead is slower, not faster

The model ships four trained `att_context_size` settings. Sweeping them
offline on two clips (compute only — the audio is already on disk):

| `att_context_size` | look-ahead | chunk audio | TTFT | RTF | updates (21s clip) |
|---|---|---|---|---|---|
| `[56, 0]` | 0ms | 0.08s | 0.86s | **0.682** | 83 |
| `[56, 3]` | 240ms | 0.32s | 0.19s | 0.217 | 46 |
| `[56, 6]` | 480ms | 0.56s | 0.17s | 0.149 | 26 |
| `[56, 13]` | 1040ms | 1.12s | 0.12s | 0.104 | 16 |

Zero look-ahead is **6.6× more expensive** than the default and has the
*worst* TTFT. One-frame chunks pay the per-chunk overhead 14× as often, and
that dominates the latency it was supposed to save. Anyone tuning this by
intuition would pick `[56, 0]` and make the app slower.

Live latency is a different quantity from this table: it is
`chunk audio + compute`, because the future audio has to happen before it can
be encoded. So `[56, 13]` trails your voice by ~1.1s and `[56, 3]` by ~0.32s.

**`[56, 3]` was the app's setting on that reasoning, and it was wrong** — see
"Look-ahead is not a latency/accuracy trade" below. The app now uses the
model's default `[56, 13]`.

### Look-ahead is not a latency/accuracy trade — smaller is just worse

Choosing `[56, 3]` for responsiveness assumed the usual trade: less future
context, faster feedback, slightly worse text. Measured against each clip's own
non-streaming `generate()` output as the reference, on six clips:

| `att_context_size` | matches the reference | RTF |
|---|---|---|
| `[56, 3]` | **1 of 6** | 0.20 |
| `[56, 6]` | 2 of 6 | 0.15 |
| **`[56, 13]`** (model default) | **6 of 6** | **0.10** |

The losses are content words, not punctuation: `[56, 3]` dropped "okay" and
"like" outright, and wrote "**nemoton**" where `[56, 13]` writes "Nemotron" —
a mangling the vocabulary layer then has to undo, caused entirely by the
setting. `[56, 6]` dropped "a better" and downgraded "Nemotron Live Transcribe"
to "live transcribe".

So the low-latency settings are worse on accuracy *and* cost twice the compute.
The only thing `[56, 13]` gives up is how far the on-screen text trails while
you are still speaking (~1.1s vs ~0.3s), which for dictation is the cheap axis:
the words land either way, and a dropped word has to be fixed by hand.

Two lessons, both mine: the intuition that a smaller chunk must be more
responsive ignored that per-chunk overhead dominates at these sizes, and the
first sweep printed only `text[:110]`, which hid exactly the tail differences
that mattered.

### Measured through the app's own live path

`--live-file`, mic and keystrokes excluded, on two real captures:

| clip | audio | first text | updates | revisions | RTF |
|---|---|---|---|---|---|
| 111314 | 21.1s | 0.43s | 46 | **0** | 0.213 |
| 111735 | 15.0s | 0.51s | 34 | **1** | 0.215 |

**RNN-T output is almost purely append-only** — 1 revision in 80 updates. The
`diff_update` backspace path is therefore rarely taken, but it is not
theoretical: it fired once, and without it that update would have appended a
duplicate instead of correcting the text.

### Accuracy against Qwen3, on clips with known ground truth

| said | Qwen3-ASR-0.6B | Nemotron 8-bit |
|---|---|---|
| "Nemotron" | "NemoTone" / "Nimoton" ✗ | **"Nemotron"** ✓ |
| "Live Transcribe" | "Live Translate" ✗ | **"Live Transcribe"** ✓ |
| "Qwen3 ASR" | "QN three years ago" ✗ | "Queen three ASR" ✗ |
| "should I test next" | "should I test next" ✓ | "should attest next" ✗ |

Nemotron gets its own name right unprompted, which Qwen3 never does. Neither
gets "Qwen3", and Nemotron's "**Queen** three" is worse than it looks for the
vocabulary layer: "queen" is a real dictionary word, so the fuzzy pass is
barred from touching it, and the fix would need a *two-word* alias, which
`apply_vocab` does not support. That is the clearest case yet for n-gram
aliases.

⚠️ Punctuation and casing differ between the two engines, and no WER is
computed here either — this is still four hand-checked words, not a metric.

### Two bugs the first real live session exposed

Both were invisible to every test written before it, and both were found by
reading the app's own log rather than by running anything.

**14. The startup model was never freed, so a swap cost both models.** The log
line `Nemotron 0.6B — live (live, peak 1656MB)` was the tell: standalone
Nemotron peaks at 942MB, and the missing 714MB is Qwen3's weights. Cause was
not the swap logic, which correctly dropped the session's reference — it was
`main()` keeping the first model in a local variable. That frame lives for as
long as `runEventLoop()` runs, so the name kept the weights alive no matter
what the Engine did. One `del model` after handing it to the session fixes it;
measured after: 942MB, **0MB overhead vs standalone**, in both directions.

This is the failure the one-model-at-a-time design exists to prevent, and it
shipped anyway, because every memory measurement until then had been taken in
a script that *returned* rather than in a process that stays running.

**15. `cancel()` on an idle live session started recording.** It was written
as a bare `self.toggle()`, which is correct when a session is running and
exactly backwards when it is not. `Engine.switch` calls cancel before
swapping, so switching *to* Nemotron and then back silently opened a
microphone stream, and that worker thread held the model — leaking 799MB on
the way back and, worse, recording without the icon saying so. Fixed by
returning early when idle, and by giving both session types a `close()` that
also **joins the worker**: the generator inside it owns a reference to the
model being replaced, so the swap has to wait for it to exit.

### Live sessions are now in the corpus

Live mode originally wrote nothing to `captures/` — a deliberate simplification
that turned out to be wrong the first time the transcripts needed reviewing,
since the only copy of the text was in whatever field it had been typed into.
Live rows are now saved like batch rows (clips prefixed `live-`, rows carrying
`mode: "live"`), which also makes the two engines directly comparable on
identical audio.

One column does not transfer: a live row's `latency_s` is the wait *after* you
stop, because the rest of the work happened while you were still speaking. Do
not read it as a batch RTF.

### Correction: "matches the offline reference" is fidelity, not accuracy

The look-ahead table above compares each streaming setting against the same
model's non-streaming `generate()` output. That measures whether streaming is
faithful to the model's own offline decode — it does **not** measure whether
either is right, and I presented it as though it did.

A later clip shows the difference plainly. `[56, 13]` reproduces `generate()`
exactly, as the rule predicts, and both produce *worse* text than the
low-latency setting did:

| pass | text |
|---|---|
| `[56, 3]` | "Is the **ordinary** like **male spectrogram** audio encoder…" |
| `[56, 13]` and `generate()` | "Is the **Oden Goder** like **Malspectrogram** audio encoder…" |

"Oden Goder" is nonsense; "ordinary" is at least a word. So on this clip the
setting that loses words elsewhere is the one that read the sentence better.
With no written ground truth there is no way to score this, which is the same
gap this document has flagged from the start — and it is now blocking a real
decision (which `LIVE_ACS` to ship) rather than an academic one.

**Resolved, and it reverses the choice.** Asked which reading matched the
sentence, the speaker confirmed **"ordinary"** — so on the only comparison in
this document with ground truth, the low-latency setting read the sentence
correctly and the model's own offline decode did not. The other difference on
that clip ("male spectrogram" vs "Malspectrogram") the vocabulary now resolves
to the same term either way, which leaves that one word as the discriminator.

The app therefore ships `[56, 3]`, accepting worse fidelity, twice the compute
(RTF 0.20, still ~5× headroom) and the risk of a dropped trailing word, in
exchange for the one thing that was actually verified plus 0.32s of lag instead
of 1.12s.

Worth stating how thin that is: **one word, on one clip.** The six-clip table
is a stronger *measurement* and a weaker *argument*, because it scores the wrong
thing. Neither settles the question. What would: transcribing a clip whose
sentence was written down before it was spoken — the corpus this document has
wanted from the beginning, now blocking a shipped default rather than a
footnote.

### The vocabulary's real gap was multi-word, and the words are ordinary

Three mishearings resisted every layer: "Webb item" (verbatim), "Queen three"
(Qwen3), "male spectrogram" (mel spectrogram). They share a structure worth
naming — each is **built from words that are already English**, so:

- the fuzzy pass is barred from touching them by the dictionary guard, which is
  working exactly as intended ("male", "queen", "item" must never be rewritten);
- a per-word alias cannot see across the space.

Phrase aliases close it: `Mel spectrogram: male spectrogram`. Fuzzy now also
matches against aliases, because the variants multiply — the same sentence gave
"male spectrogram" at `[56, 3]` and "Malspectrogram" at `[56, 13]`, and
`malspectrogram`→`melspectrogram` scores 0.93, so one listed spelling covers
its neighbours.

Re-swept the whole corpus after the change: **5 rewrites across 251 distinct
words, every one correct, no false positives.** The guard still holds — "a male
voice" and "the spectrogram" are untouched, and only the full two-word phrase
fires.

### One private API is load-bearing

The live path is assembled from `StreamingLogMelSpectrogram` (incremental
centered mel, bounded memory), `ConformerStreamingState` (per-layer attention
and conv caches), and `model._decode_prompted_chunks(...)` — which is
**private**. The public `stream_generate()` only accepts an array that already
exists, which a live microphone by definition does not, so there is no
supported way to do this today. If a `mlx_audio` upgrade renames that method,
live mode breaks and batch mode does not.

## Parakeet-TDT-0.6B-v3 — the fastest and the most accurate, at 2× the memory

`mlx-community/parakeet-tdt-0.6b-v3`, 2.3GB on disk (bf16 — mlx-community
publishes no quantised variant). A TDT (token-and-duration Transducer)
FastConformer. Listed as "tested" in `README.md` since this folder was
created; it was not. It is now.

It needed **no new code**: `run_qwen3_asr_mlx.py` was already engine-agnostic
(`load_model` + `generate_transcription`, which filters kwargs against the
model's own `generate` signature), so the same script takes any repo. The
misleading filename is the only thing that hid that.

    ./.venv/bin/python run_qwen3_asr_mlx.py mlx-community/parakeet-tdt-0.6b-v3

### Results — all three models re-measured in one session, 2026-08-31

The archived 2026-08-12 rows were **not** reused. The machine was at 3.9GB
swap when this run started vs 1.27GB on 08-12, so every model was re-run.
Qwen3-0.6B came back at RTF 0.0638 against its archived 0.0643 (**0.8%**),
which is what licenses comparing across the two sessions at all.

| model | RTF (median) | latency (s) | load (s) | MLX peak, load | MLX peak, inference | min free RAM |
|---|---|---|---|---|---|---|
| **parakeet-tdt-0.6b-v3** | **0.0313** | 0.227 | 1.69 | 2431MB | **2617MB** | 1082MB |
| Qwen3-ASR-0.6B-4bit | 0.0638 | 0.463 | 0.90 | 681MB | 1311MB | 2071MB |
| nemotron-0.6b-8bit (batch) | 0.1045 | 0.758 | 0.70 | 760MB | 925MB | 2380MB |

- **Parakeet is 2.04× faster than Qwen3-0.6B** and 3.3× faster than Nemotron
  decoded in batch. That makes it 11× whisper-small on the same clip.
- It is also the **only model to load slowly** (1.69s vs 0.70–0.90s) and the
  only one whose load moved swap at all (**+199MB**). Inference moved none.
- **2617MB inference peak is the highest number in this document** — above the
  1.7B Qwen's 2314MB, which this folder already declined to ship for that
  reason. Whether that verdict transfers is argued below; it does not.

Nemotron's batch RTF is new here too. Every previous Nemotron number was
measured through `dictate_app.py --live-file`; this is the first time it ran
through `metrics.py`, and it confirms the streaming path (RTF 0.10 at
`[56, 13]`) costs essentially nothing over the model's own batch decode.

### It gets the benchmark clip completely right — the first model to do so

The 7.25s clip has been the source of this document's two long-running
accuracy disputes. Parakeet settles both:

| model | "verbatim" | "solid"/"solved" | tail |
|---|---|---|---|
| **parakeet** | **"verbatim"** ✓ | **"solid"** ✓ | clean |
| Qwen3-ASR-0.6B | "Webb item" ✗ | "solid" ✓ | clean |
| Qwen3-ASR-1.7B | "Verb item" ✗ | "solid" ✓ | clean |
| nemotron 8-bit | *dropped* — "why the" ✗ | "solid" ✓ | clean |
| whisper-small | "verbatim" ✓ | "solved" ✗ | 49 periods |

Nobody had gotten both. Whisper had "verbatim" and lost "solid"; the Qwen
sizes had "solid" and lost "verbatim"; Nemotron, measured here for the first
time on this clip, is the worst of the four — it does not mangle "verbatim",
it **deletes** it.

Same caveat as everywhere else in this document: still no written ground
truth, so this is a four-way consensus argument. It is a stronger one than
before, because the two words now have a 4–1 and 4–1 majority instead of 2–1.

### On the capture corpus, it wins on the words the vocabulary layer exists for

Three clips from `captures/` whose words are known (`docs` has cited all three
as ground truth). Raw model output, **vocabulary layer off**, so this is the
model and not the post-processing:

| said | parakeet | Qwen3-0.6B | Nemotron |
|---|---|---|---|
| "Nemotron Live Transcribe" | **"Nemotron Live Transcribe"** ✓ | "NemoTone Live Translate" ✗✗ | **"Nemotron Live Transcribe"** ✓ |
| "Nemotron" (111735) | "Nematron" ~ | "Nimoton" ✗ | **"nemotron"** ✓ |
| "Qwen3 ASR" | "Q3ASR" ~ | "QN three years ago" ✗ | "Queen three ASR" ✗ |
| "should I test next" | **"should I test next"** ✓ | ✓ | "should attest next" ✗ |
| "mel spectrogram" | "mail spectrogram" ~ | "Megaprogram" ✗ | "Mails Pector" ✗ |

`~` means the existing vocabulary layer already fixes it — verified, not
assumed: `Nematron`→Nemotron and `mail spectrogram`→Mel spectrogram both fire
off aliases and fuzzy rules already in `vocab.txt`. Only **`Q3ASR` was new**,
and it is the easiest kind of mangling this layer handles: not an English
word, so one alias covers it. `Qwen3:` gained `q3asr` in `vocab.txt` and
`vocab.example.txt`; `--check` passes.

So after post-processing Parakeet is **5 of 5** on these clips, Qwen3 is 1 of
5, and Nemotron is 3 of 5. Parakeet's one clear defect is a stutter it
invents — "what more **mode mode** models" where the others write "what more
models". A duplicated syllable is a visible error; "Live Translate" for "Live
Transcribe" is an invisible one, which is the worse failure for dictation.

### The 2617MB does not disqualify it, and the reason is the 1.7B's real problem

This folder rejected Qwen3-1.7B at a 2314MB peak, so a 2617MB model looks
like a straightforward re-run of that decision. It is not, because the 1.7B
was rejected for being **slower and heavier** — 2.33× the time of the 0.6B for
more memory, which is dominated on both axes and needs no memory argument at
all. Parakeet is 2.04× *faster* than the 0.6B, so it is the first model here
where memory is actually the price of something.

What was measured on this run: minimum free RAM 1082MB, **no swap growth
during inference**, and the +199MB of swap it did move happened during load.
It fits, on the machine as it stood, with a smaller margin than anything else
in this folder.

Two things make that margin thinner than the table suggests, and both are
about the app rather than the benchmark:

- The benchmark process holds one model and exits. `dictate_app.py` stays
  resident, and bug #14 is the record of what happens when a swap leaves two
  models alive — Nemotron measured 1656MB instead of 942MB because Qwen3's
  weights were still held by a `main()` local. Parakeet + anything is
  ~3.5GB. The one-model-at-a-time design is what makes it survivable, and
  that design is now load-bearing rather than merely tidy.
- 2617MB is a **bf16** number. Every other model in this table is quantised
  (4-bit or 8-bit). An 8-bit Parakeet would plausibly land near Nemotron's
  925MB with most of the speed, and none exists on the Hub today — which is
  the single highest-value thing anyone could do to this comparison.

### Recommendation, and what it costs

**Parakeet-TDT-0.6B-v3 replaces Qwen3-ASR-0.6B-4bit as the batch model**:
2.04× faster, the only model to get the benchmark clip fully right, and 5/5
on the capture corpus after post-processing where Qwen3 manages 1/5.

Keep Nemotron for live mode — Parakeet is a TDT transducer and `mlx_audio`
exposes no cache-aware streaming state for it, so the live path cannot use it
without the same private-API surgery documented above, against a model that
has no trained look-ahead settings to surgery *with*.

Costs, stated plainly: 2× the inference memory, 2× the load time, a
stutter-duplication failure mode the Qwen models do not have, and a bf16-only
checkpoint on a machine that has wanted quantised weights for every other
model in this folder.

### Shipped in the app, as a menu entry and not as the default

`Parakeet 0.6B` is now in `MODELS`, batch mode. Verified through the app's own
path rather than the benchmark's:

- `--models --swap`: **2719MB loaded-and-warmed, freeing back to 0MB**, same as
  every other entry. That number is 100MB above the benchmark's 2617MB because
  it includes the warm-up decode. Two Parakeets would be 5.4GB, so the
  one-model-at-a-time invariant is now doing more work than it ever has.
- `--file` on capture 111735: loaded in 2.9s, 15s of audio in **0.45s**, and
  the vocabulary layer resolved both names — "Nematron"→Nemotron and
  "Q3ASR"→Qwen3 — end to end, which is the first time this document's accuracy
  claim and its post-processing claim have been checked in the same run.
- `--live-file` unchanged at RTF 0.201, confirming the Nemotron path was not
  disturbed.

**The startup default stays Qwen3-ASR 0.6B.** Parakeet is the better model and
the wrong thing to load automatically: `DEFAULT_MODEL` is what gets loaded at
launch on a machine that boots with ~1.4GB free, and 2719MB of that is a
choice worth making deliberately rather than every morning. `model.txt`
remembers the pick, so it is a one-time click.

Two small things fixed while wiring it in, both the same shape — a hardcoded
model where a derived one belonged:

- `REPO` was a second hand-maintained copy of the default repo string, free to
  drift out of sync with `MODELS`. It is now `MODELS[DEFAULT_MODEL][0]`.
- `--file` loaded `REPO` rather than the saved choice, so it could only ever
  check the default — the model that needs checking least. It now loads what
  the menu bar has selected, which is how Parakeet was verified above.

### Open: the `[56, 3]` decision may rest on a misheard word

Not a Parakeet finding, but this run produced the evidence and it reverses a
shipped default, so it belongs here.

`LIVE_ACS = [56, 3]` ships because of exactly one word on clip
`live-20260817-143520`: `[56, 3]` read it as "**ordinary**" and `[56, 13]` as
"**Oden Goder**", and the speaker confirmed "ordinary". That single
confirmation outweighed a six-clip table favouring `[56, 13]`.

Both of the other models transcribe that position as **"audio encoder"**:

| model | opening phrase |
|---|---|
| parakeet | "Is the **audio encoder** like a mail spectrogram?" |
| Qwen3-0.6B | "Is the **audio encoder** like, Megaprogram?" |
| nemotron `[56, 13]` | "Is the **Oden Goder** like…" |
| nemotron `[56, 3]` | "Is the **ordinary** like…" |

"Oden Goder" and "ordinary" are both recognisable manglings of "audio
encoder", and the sentence continues "…what input does the audio encoder
take?", so the phrase is demonstrably in this clip. "Is the audio encoder
like a mel spectrogram" is also the only reading that parses as English;
"Is the ordinary like mel spectrogram audio encoder" does not.

If that is right, then **neither** streaming setting got the word, the
confirmation resolved a question the two settings were not actually disputing,
and the six-clip fidelity table is unopposed — which points back at `[56, 13]`,
the model default, at half the compute. The app has not been changed: only
the speaker knows what was said, and this is the same missing-ground-truth
gap that has blocked this question from the beginning.

## Corrections to the previous version of this document

| Previous claim | Status |
|---|---|
| "Qwen3-ASR is ~2.2-3x faster than whisper-small" | **Understated.** 0.6B is 5.5×. |
| "1.7B was faster than 0.6B in this run" | **Wrong, and reversed.** 0.6B is 2.33× faster. Cause was the warm-up confound, not noise. |
| "Negative ram_spike is because load happens before measure()" | **Wrong mechanism.** See Bug #3. |
| "Neither Qwen3-ASR size nailed 'verbatim'" | **Holds**, but whisper-small has its own error ("solved" vs "solid") that was missed. |
| "No tail hallucination on Qwen3-ASR" | **Holds.** Reproduced on all runs. |

Added 2026-08-31, on claims made since:

| Previous claim | Status |
|---|---|
| "Qwen3-ASR-0.6B is the fastest" | **Superseded.** Parakeet is 2.04× faster. |
| "Neither Qwen3 size nailed 'verbatim'" (still open) | **Closed.** Parakeet gets it, and "solid", on the same run. |
| `README.md`: "Parakeet is on the list but not yet tested" | **Fixed.** Tested; it is now the batch recommendation. |
| "`[56, 3]` ships because the speaker confirmed 'ordinary'" | **Questioned.** Two other models read that word as "audio encoder". Unresolved — see the open item above. |
