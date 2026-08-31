# STT Model Testing

Local sandbox to compare Whisper and Qwen3-ASR on latency, RTF, and CPU/RAM
peak — same pattern as `../tts_models`, adapted for speech-to-text. (Parakeet
is on the list but not yet tested — see `docs/RESULTS.md` next steps.)

Each project folder in this repo gets its **own** venv (see `.gitignore`'s
`*/.venv` rule) — this is not shared with `tts_models/.venv`.

- **The benchmark** — `run_*.py` transcribe the same clip with each engine,
  warm up, then append one row per timed run to `results.csv`.
  Findings: `docs/RESULTS.md`.
- **Test clip** — `sample_for_tts.wav`, a real human-voice recording
  (converted from `sample_for_tts.m4a` via macOS's built-in `afconvert` —
  `miniaudio`, mlx-audio's STT decoder dependency, doesn't read AAC/M4A).
- **`results_pre_warmup_fix.csv`** — archived rows from before the warm-up
  and memory-measurement fixes. Not comparable to `results.csv`; kept only
  so the corrections in `docs/RESULTS.md` can be checked against them.

MLX-only: this folder has no `torch` dependency, and memory is read from MLX's
own allocator (`mx.get_peak_memory()`), not RSS. `metrics.py` has therefore
diverged from `tts_models/metrics.py`; `docs/RESULTS.md` explains why.

## Setup (Apple Silicon)

    uv venv
    uv pip install -r requirements.txt

## Run

    ./.venv/bin/python metrics.py                 # self-check the harness first

    ./.venv/bin/python run_whisper_mlx.py         # openai/whisper-small
    ./.venv/bin/python run_qwen3_asr_mlx.py mlx-community/Qwen3-ASR-0.6B-4bit
    ./.venv/bin/python run_qwen3_asr_mlx.py mlx-community/Qwen3-ASR-1.7B-4bit

One model per process keeps its memory baseline clean. Knobs:

    RUNS=5 ...                        # timed runs per model (default 3)
    WHISPER_LANGUAGE=en ...           # skip language detection, ~10% faster

`whisper-base` was tried and dropped — see `docs/RESULTS.md` for why.

## Dictate app (push-to-talk)

`dictate_app.py` — the STT counterpart to `../tts_models/speak_app.py`. Resident
menu-bar app holding one switchable model in memory (menu bar → Model); **⌥⌘R** starts recording, **⌥⌘R**
again stops it, **esc** throws the recording away, and the transcript is pasted into
whatever app you were in. A floating panel shows elapsed time and a live input meter
while you speak — the menu-bar icon can say "armed" but not "hearing you", and those
differ.

The panel takes focus while you speak, exactly as Kokoro Speak's does: an accessory
app that orders a window front *without* activating is not placed on your Space, and
macOS bounces you to the Desktop and back instead (`docs/RESULTS.md` bug #11). Focus
is handed back before the ⌘V, and if that can't be confirmed within a second the
transcript is left on the clipboard rather than typed into the wrong window.

Push-to-talk, not streaming: no text appears until you stop. At the 0.6B's
measured RTF of 0.064 a 10s utterance transcribes in ~0.6s, so the wait after you
stop is short enough that partial results aren't worth the complexity. Streaming
models (`nemotron-3.5-asr-streaming`, `Voxtral-Realtime`) are the upgrade path if
long-form dictation ever needs it.

### Switching models (menu bar → Model)

A submenu with a tick next to the loaded one. The choice is saved to `model.txt` (gitignored,
like `vocab.txt` — it is which model *you* picked) and restored next launch; the default is Qwen3-ASR 0.6B, the fastest measured and the one with no tail
hallucination.

| menu entry | mode | single-model peak |
|---|---|---|
| Qwen3-ASR 0.6B | batch — pastes when you stop | 1375MB |
| Qwen3-ASR 1.7B | batch | **2426MB** — tightest that still runs here |
| Nemotron 0.6B — live | live — types as you speak | **942MB**, the lightest |

**Exactly one model is ever loaded.** Two resident peak at 2425MB, worse than the 1.7B alone,
on a machine that already had under 1GB free. Switching frees the old model *before* loading
the new one, which returns all of it (MLX's peak drops to 0), so a swap never costs more than
the heavier of the two. A swap takes 1.5–2.5s, runs off the run loop so the menu doesn't
freeze, and stops any recording in progress — continuing to record against a freed model
would crash. It also *waits* for the live worker to exit, because that worker holds the model
being replaced.

List them, and prove the free actually happens, without the GUI:

    ./.venv/bin/python dictate_app.py --models           # list, mark the saved choice
    ./.venv/bin/python dictate_app.py --models --swap    # load each in turn, print peaks

Adding a model is one line in `MODELS`, as long as `mlx_audio` can load it and its mode
("batch" or "live") matches how it decodes.

### Live mode (Nemotron, types as you speak)

Pick **Nemotron 0.6B — live** from the Model menu. Same ⌥⌘R to start and stop, but the text is
**typed into whatever field has focus while you talk** instead of pasted in one go at the end.
The model is `nemotron-3.5-asr-streaming-0.6b-8bit`, a cache-aware streaming
FastConformer-RNNT — and at 942MB it is the *cheapest* of the three, not the most expensive.

Measured on this repo's own captures (`--live-file`): first text at **0.43–0.51s**, RTF
**0.21** — about 5× real-time headroom.

`LIVE_ACS` is the look-ahead, and choosing it was less obvious than it looked. The model's
default `[56, 13]` reproduces the non-streaming `generate()` on 6 of 6 clips where `[56, 3]`
differs on 5, and is half the compute — but that measures *fidelity to the model's own offline
decode, not accuracy*. On the one clip whose wording was confirmed, `generate()` was itself
wrong: `[56, 13]` heard "Oden Goder" where `[56, 3]` heard the correct "ordinary".

So the app ships `[56, 3]`: ~0.3s of on-screen lag instead of ~1.1s, twice the compute, and a
trailing word can go missing ("okay" did, on one clip). `[56, 0]` is slower than both. One
confirmed word is thin evidence — the sweep and the correction are in `docs/RESULTS.md`, and
settling it properly needs a clip whose sentence is written down first.

Two behaviours that differ from batch mode, both deliberate:

- **No panel.** The panel activates the app to place itself on your Space, and activating
  steals the focus live typing needs to keep. The menu-bar icon is the only indicator.
- **Rows are marked `mode: "live"`** in `captures.jsonl`, and their clips are prefixed
  `live-`. The `latency_s` of a live row is the wait *after* you stop, not the time to
  transcribe the clip — most of that work already happened while you spoke — so its `rtf` is
  not comparable to a batch row's. Same audio, same log, different meaning.

Test it without a mic or any permission — prints each update instead of typing it, and marks
revisions with `->`:

    ./.venv/bin/python -u dictate_app.py --live-file captures/<clip>.wav

The vocabulary applies here too, with one adjustment: only words followed by a space are
fixed. The last word of a live transcript is still being decoded, and rewriting a fragment is
how "desh" would become "Siddesh" mid-word. The final word is fixed when the session ends.

### Vocabulary (names the model keeps getting wrong)

Menu bar → **Vocabulary…** opens a window; type the terms you use, one per line. It saves
as you type and is re-read before every transcription, so a word added there applies to the
next thing you say — no restart. Edits are written atomically, so an interrupted save can't
truncate the list.

The list lives in `vocab.txt`, which is **gitignored**: which words you mispronounce, and
which names matter to you, is personal. `vocab.example.txt` is the tracked starting point —
the window seeds itself from it the first time you open it, and `--check` asserts against it
so a fresh clone passes and your own edits can never break the self-check.

    Qwen3: gwen, qn, quen
    Nemotron: nimoton, nemotone
    Kokoro

Three things happen to the words on that list, weakest guarantee last:

1. They are passed as Qwen3-ASR's `system_prompt`, which biases decoding itself. Costs
   ~0.03s. Measured: it turned "NemoTone" into "Nemotron" on one capture and left
   "Nimoton" alone on the next — real, but not something to rely on.
2. Anything after a colon is replaced exactly. This is the only layer that can fix a
   mangling which is itself an English word — "gwen" for Qwen3 — because layer 3 refuses
   to touch those on purpose.
3. Remaining unknown words are fuzzy-matched at a 0.75 cutoff, skipping anything in
   `/usr/share/dict/words`. Measured on this app's own captures: `nemotone`→Nemotron 0.88,
   `nimoton`→0.80, `sudesh`→Siddesh 0.77 — while the dangerous neighbours ("when"→Qwen3,
   "mix"→MLX) sit at 0.67, well clear of the cutoff and protected by the dictionary anyway.

An alias may contain spaces, and that turns out to be the important case: every mishearing
that survived the other layers was **made of ordinary English words** — "male spectrogram" for
mel spectrogram, "Queen three" for Qwen3, "Webb item" for verbatim. The dictionary guard
protects those words by design, and a per-word alias can't see across the gap, so a phrase
alias is the only thing that reaches them.

Fuzzy matching also runs against the aliases, not just the terms. The same sentence produced
"male spectrogram" at one look-ahead setting and "Malspectrogram" at another, and each variant
is a character or two from a form already listed (`malspectrogram`→`melspectrogram` scores
0.93) — so listing one spelling covers its neighbours. Measured on the whole capture corpus:
**5 rewrites across 251 distinct words, all correct, no false positives.**

`--check` asserts all of the above against real transcripts from `captures/`, including the
words that must *not* change ("a male voice", "the spectrogram", "females spectrogram").

### Installing it as a real app

`Dictate.app` next to this file is a launchable bundle holding the model in memory
the way Kokoro Speak does. Same construction as `../tts_models/Kokoro Speak.app`:
just an `Info.plist` and a shell stub that runs `dictate_app.py` from this venv, so
there is **no build step** and no frozen copy of the code — edit the script and
relaunch.

    open -a "$PWD/Dictate.app"          # run it straight from the repo
    tail -f /tmp/dictate.log            # the only place startup failures are visible

To make Spotlight offer it, install it into a standard app location. Two lines, and
still one copy of the code:

    mkdir -p ~/Applications/Dictate.app
    ln -sfn "$PWD/Dictate.app/Contents" ~/Applications/Dictate.app/Contents

A real directory whose `Contents` is the symlink, **not** a symlink to the bundle:
Spotlight indexes a symlink as a symlink, so `ln -s …app ~/Applications/` looks
right in Finder and stays invisible to search. Check with
`mdfind -onlyin ~/Applications Dictate` — it should print the path. The bundle
itself only needs to be indexed as an Application (`mdls -name kMDItemKind`); being
indexed is not enough on its own, the location is what makes Spotlight offer it
(`docs/RESULTS.md` bug #12).

Auto-start at login: System Settings → General → Login Items, add `Dictate.app`.

Two details that are load-bearing, both learned on the TTS side:

- The stub launches python as a **child, never `exec`**. With `exec`, the process is
  the one LaunchServices registered as `Dictate.app` while its `NSBundle` is
  `Python.app`; a process whose registration and bundle disagree gets **no menu bar
  slot** — the status item exists, answers every call, reports a zero-height window,
  and is invisible forever. The app now logs its icon size at startup
  (`menu bar icon ready (39x30)`) so that failure is one line in the log.
- Launched from the bundle, macOS attributes permissions to **Dictate**, not to the
  venv's python — a separate TCC entry from the one a terminal launch uses. The
  startup log prints whichever one currently needs the grant.

### Running it from a terminal

    uv pip install -r requirements_app.txt

    ./.venv/bin/python dictate_app.py --check                    # meter logic, no model
    ./.venv/bin/python dictate_app.py --file sample_for_tts.wav  # self-check, no mic
    ./.venv/bin/python dictate_app.py --live-file <clip>.wav     # live pipeline, no mic/keys
    ./.venv/bin/python dictate_app.py --once 5                   # mic test, 5s
    ./.venv/bin/python dictate_app.py --keys                     # hotkey diagnosis
    ./.venv/bin/python -u dictate_app.py                         # the app
    ./.venv/bin/python dictate_app.py --models                   # list/verify models

**Why ⌥⌘R and not the obvious ⌥⌘D:** ⌥⌘D is macOS's own "Turn Dock Hiding On/Off"
symbolic hotkey. The system consumes those before any app sees them, so a handler
bound to it can never fire — the only visible effect is the Dock flickering. No code
change fixes that; the key has to be one *macOS* doesn't already own. `--keys` prints
every keystroke that reaches the app, which separates that failure ("everything prints
except my combination") from a missing permission ("nothing prints at all").

Ordinary apps claiming the same combination is a different problem with a real fix.
⌥⌘R is reading mode in Brave and Chrome, and a global `NSEvent` monitor can only
observe — the browser got the keystroke too, so one press recorded *and* reformatted
the page. The hotkey is a `CGEventTap` instead, which sits ahead of delivery and drops
the event, so this app owns ⌥⌘R everywhere (`docs/RESULTS.md` bug #13). The trade-off
is that every keystroke passes through this process to catch one; `DICTATE_DEBUG=1`
prints keycodes, so don't leave it on while typing a password.

Needs two macOS permissions, and fails quietly in different ways without them:

- **Accessibility** (System Settings → Privacy & Security → Accessibility) — add
  **Dictate** if you launch the bundle, or the interpreter path if you run it from a
  terminal. These are two separate entries and granting one does nothing for the
  other; likewise Kokoro Speak being trusted has never helped here, since it runs on
  a different interpreter (Homebrew python3.14 vs uv cpython-3.11). The startup log
  prints exactly which one needs it and whether it is already granted. Without it the
  hotkey installs but never fires and the synthetic ⌘V does nothing.
- **Microphone** — prompted on first record. Denied looks identical to a working app
  except every recording is digital silence. Two things catch it now: the panel's
  meter stays empty and says so after a second, and a clip whose peak never exceeds
  `SILENCE_PEAK` is never sent to the model at all — asking Qwen3-ASR what silence
  says gets you a fluent answer in a language it picked itself.

The same location constraint as Kokoro Speak applies: don't keep the repo in
`~/Documents`, `~/Desktop` or `~/Downloads`, or macOS's sandbox blocks the venv
when launched outside a terminal (see `../tts_models/docs/kokoro_speak_app_guide.md`).

Every utterance is saved to `captures/` with its transcript in `captures.jsonl`
(both gitignored — it's your voice). That is deliberate: you know what you said, so
normal use accumulates the labelled set needed for the real WER number
`docs/RESULTS.md` currently lacks.
