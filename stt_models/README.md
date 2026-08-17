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
menu-bar app holding Qwen3-ASR-0.6B in memory; **⌥⌘R** starts recording, **⌥⌘R**
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
    ./.venv/bin/python dictate_app.py --once 5                   # mic test, 5s
    ./.venv/bin/python dictate_app.py --keys                     # hotkey diagnosis
    ./.venv/bin/python -u dictate_app.py                         # the app

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
