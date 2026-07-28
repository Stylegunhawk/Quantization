# Streaming TTS — Resident Model, Audio Straight to Speakers

Tested 2026-07-27 on Apple M1, macOS 26.5.2, Python 3.14.0, `mlx` 0.32.0,
`mlx-audio` 0.4.6, `sounddevice` 0.5.5. Model:
`mlx-community/Kokoro-82M-bf16`. Implementation: `../speak_mlx.py`.

## What this tests, and why it's a different question

`findings.md` benchmarks **file synthesis**: how long to produce a complete
`.wav`. Every number there — latency, RTF — is time-to-finish-a-file.

For a read-aloud hotkey that is the wrong metric. What you feel when you press
the key is **time to first audio (TTFA)**. After the first sample reaches the
speakers, generation only has to stay ahead of playback (RTF < 1), and all six
models in `findings.md` clear that comfortably. So an RTF of 0.048 vs 0.163 is
invisible in daily use, while a TTFA of 1s vs 8s is the entire product.

This doc measures TTFA, with the model held in memory and PCM written directly
to the output device — no intermediate file.

## Setup

Two costs the file benchmarks never surfaced:

- **Model load: 5.6–6.2s** (weights already cached locally), plus a warm-up
  `generate()` to burn the one-time Metal shader-compile tax. Paid once at
  startup, so it must happen in a resident process. A hotkey that spawns
  Python per press pays ~6s every time and the inference speed is irrelevant.
- **Resident cost while idle: 79 MB RSS, 0.5% CPU.** This is a real
  private-process footprint, unlike the `macos-say` RSS delta discussed in
  `findings.md`.

Trigger path is a Unix socket (`/tmp/kokoro-speak.sock`), so the hotkey side is
a shell one-liner with no Python startup: `pbpaste | nc -U /tmp/kokoro-speak.sock`.

## Result: TTFA is flat in selection size

Final measurements, four utterances of increasing length:

| chars | chunks | TTFA | total (gen+play) | audio |
|---|---|---|---|---|
| 6 | 1 | 0.43s | 2.26s | 1.4s |
| 88 | 1 | 1.01s | 6.97s | 5.8s |
| 221 | 3 | 0.89s | 16.82s | 15.8s |
| 410 | 4 | 0.95s | 27.41s | 26.3s |

`total` ≈ TTFA + `audio`, which is the point: playback is gapless and overlaps
generation, so wall-clock is dominated by how long the speech itself lasts, not
by synthesis.

## The three findings that got it there

### 1. `mlx_audio`'s `generate()` does not usefully stream

It is a generator, but it does most of its work before yielding the first
chunk. Naively iterating it and writing each chunk gave a TTFA that scaled with
**total** text length:

| chars | TTFA (naive iteration) |
|---|---|
| 6 | 0.913s |
| 88 | 1.381s |
| 221 | 2.280s |
| 410 | **7.976s** |

Worse, pre-splitting into short sentences and handing the whole string to one
`generate()` call did not help — 162 chars as four short sentences measured
**3.898s**, worse than 221 chars as a single sentence. So the chunk boundaries
inside `generate()` are not where the yielding happens.

The fix is to split the text yourself and make **one `generate()` call per
sentence**, playing each result as it completes. This is the single most
important fact for anyone building interactive TTS on MLX-Kokoro; the
per-variant numbers in `findings.md` give no hint of it.

### 2. Without lookahead, every sentence boundary is an audible gap

`sounddevice`'s `stream.write()` blocks until the buffer drains. Generate →
write → generate means the next sentence's ~1s of synthesis happens *after*
the previous one has finished playing, so you hear a gap between every
sentence.

Fixed with one sentence of lookahead: a producer thread generating into a
`queue.Queue(maxsize=1)` while the main thread writes. Generation runs ~6x
faster than realtime (5.8s of audio in ~1.0s), so one slot of lookahead is
enough to stay ahead indefinitely. MLX generation from a non-main thread works
fine here.

### 3. Only the first chunk is on the critical path

Every chunk after the first is generated while earlier audio plays, so its
length is free. The first chunk's length is pure latency. Splitting the first
chunk tighter than the rest (`FIRST_CHUNK_CHARS = 90` vs
`MAX_CHUNK_CHARS = 180`) took the 221-char case from **2.33s → 0.89s**, at the
cost of one slightly early pause. Later chunks keep full-sentence prosody.

The remaining ~1.0s floor is the 90-char first chunk. `FIRST_CHUNK_CHARS ≈ 50`
would get roughly 0.6s in exchange for one more early pause.

A comma fallback handles run-on sentences: a 400-char comma-spliced sentence
has no `.` to split on and would otherwise cost the full ~8s TTFA on its own.
Text with neither sentence terminators nor commas passes through unsplit — an
accepted limit, not a bug.

## Quantization: still nothing to gain

`findings.md` already showed bf16/8bit/4bit within noise on latency. That holds
here — nothing about streaming changes it, because the bottleneck is per-call
fixed overhead (G2P/phonemization on CPU), not matmul throughput. bf16 is the
default: no quantization loss, and 79 MB resident is already negligible. Switch
to 8bit only if you want to halve that.

## The app: `../speak_app.py`

`speak_mlx.py` is the measured reference above — start it, send text, it speaks
and cannot be interrupted. `speak_app.py` is the daily-use version: same core
(it imports `split_sentences` rather than copying it) plus playback controls.

Ships as `../Kokoro Speak.app` — Info.plist plus a short shell stub that runs
`speak_app.py` from the venv. No py2app, no frozen copy of the code to keep in
sync, and it's double-clickable and draggable into Login Items.
`LSUIElement` makes it an accessory app: no Dock icon, no menu bar, and
activating it does not switch Spaces.

### A status item created before the run loop never gets a slot

The menu-bar icon existed and was simply invisible. `NSStatusItem` was built in
`Ui.__init__`, which runs before `AppHelper.runEventLoop()`, so there was no live
connection to the menu-bar server to negotiate a position with — the item was
parked offscreen forever.

The first fix was `AppHelper.callAfter` from `__init__`, on the reasoning that later
`callAfter`s would run FIFO behind it. That worked once and then failed on the next
launch with `'Ui' object has no attribute 'status_item'`: a block enqueued **before
the run loop exists** is not reliably delivered, and `icon()` evaluates
`self.status_item` on the worker thread *before* deferring anything. Now created
from `applicationDidFinishLaunching:` on an app delegate — the documented point at
which the app is running — with `status_item = None` until then and both readers
tolerating it, because playback can begin before the menu bar does.

`setDelegate_` does not retain, so the delegate is held in a module global. A local
would be collected and the callback would never fire.

Diagnosing this without eyes on the screen is worth remembering: query the app's
own accessibility tree from another process.

```python
_, extras = AXUIElementCopyAttributeValue(AXUIElementCreateApplication(pid), "AXExtrasMenuBar", None)
_, kids = AXUIElementCopyAttributeValue(extras, "AXChildren", None)   # AXTitle, AXPosition
```

Before the fix: `AXTitle '🔈'` at `x:-1, y:888` on a 1440x900 screen. After:
`x:1006, y:-27`. The reference point that makes those numbers readable is
`ControlCenter`'s own extras — its **visible** Battery/Clock/Wi-Fi items all report
`y:-26` (the menu bar is simply hidden behind a fullscreen app), while its
**unplaced** items sit at `x:0, y:900` — the same parked-offscreen signature.

Never assert on an absolute position when checking this. macOS assigns the slot and
moves it whenever another app adds or drops an extra, and the whole bar shifts to a
negative `y` while it is hidden behind a fullscreen app. Derive the expected row from
the other running apps' extras instead; a literal `0 < y < 30` failed on runs where
the icon was correctly placed.

### The launcher stub must not `exec`

The same invisible icon came back, but only when launched from Spotlight or Finder —
never under launchd, and never running `speak_app.py` from a terminal. Same
signature as above (`x:-1, y:888`), but the delegate had definitely run: the item
existed with its title set.

The stub ended in `exec python speak_app.py`. `exec` means the python process **is**
the process LaunchServices registered as `Kokoro Speak.app` — but python's
`NSBundle.mainBundle()` is its own `Python.app`. A process whose LaunchServices
registration and main bundle disagree is given no menu-bar slot at all: the status
item is created, reports a zero-height button window, and stays invisible. launchd
never goes through LaunchServices, which is exactly why that path always worked.

Running python as a plain child fixes it — unregistered, it claims its own slot like
any terminal launch. Measured on a stub bundle, `PLACED` meaning a real slot:

| stub | result |
|---|---|
| `exec python probe.py` | not placed |
| `unset __CFBundleIdentifier; exec python probe.py` | not placed |
| `python probe.py` (child) | **placed** |
| `unset __CFBundleIdentifier; python probe.py` | **placed** |

So the bundle identifier in the environment is irrelevant; `exec` is the whole bug.
The stub now backgrounds python and `wait`s on it, which costs two things worth
keeping: a `trap ... TERM INT` to forward signals, or `launchctl unload` and logging
out leave a 700 MB model resident with nothing owning it; and `wait "$child"` as the
last command, so the script exits with python's status and the plist's
`KeepAlive{SuccessfulExit:false}` still sees the `0` that "already running" reports
instead of respawning in a loop.

### A speed change is heard 2 sentences later, and that is buffer depth

`nudge` updates `self.speed` immediately; the delay is entirely how much audio is
already generated. `speed` is an input to Kokoro's duration predictor, not a playback
rate, so it cannot affect samples that exist — and resampling them instead would shift
pitch. Mid-sentence is therefore unavailable at any price short of regenerating the
sentence and restarting it.

The producer originally generated a sentence and *then* blocked on `pending.put`,
which kept **three** in flight: the one playing, the one queued, and one finished and
waiting for a slot. Measured with a model that records the speed each sentence was
generated at: change the speed while `S2` played and `S5` was the first sentence to
use it. Waiting for the slot *before* generating removes the third, and the same
measurement gives `S4` — 2 sentences.

Going to 1 was rejected. It needs the queued audio discarded and regenerated, so
every press buys dead air for a full generate (~0.7s) at the moment the current
sentence ends. Wrong trade for a nudge control.

The cost of the change is that the producer now has one sentence of playback to
generate the next, where it had two. A short sentence followed by a long one can
overrun that and empty the queue; the consumer fills the gap with silence, so it is a
brief pause rather than static. `pending`'s `maxsize` is the knob: raising it buys the
slack back at one more sentence of speed lag.

### The repo cannot live in `~/Documents`

This one cost an hour, so: an app launched by **Finder or launchd** gets its own
privacy (TCC) identity with no access to `~/Documents`, `~/Desktop`, or
`~/Downloads`. Python couldn't read its own venv config and died before running
a line of our code:

```
PermissionError: [Errno 1] Operation not permitted: '.../.venv/pyvenv.cfg'
```

Running `speak_app.py` from a terminal works throughout, because it inherits the
terminal's already-granted access — so the bug looks like "the app is broken but
the script is fine". No permission prompt appears, and ad-hoc code-signing the
bundle doesn't produce one either: the process doing the reading is the `exec`'d
interpreter, not the signed bundle. The repo now lives at `~/quant`, outside all
three protected folders, which needs no permissions at all. Granting the venv's
`python` Full Disk Access is the alternative if the repo must stay put.

The stub redirects to `/tmp/kokoro-speak.log` when no terminal is attached
(`[ -t 1 ] || exec >> …`), because a Finder-launched failure is otherwise
completely silent.

### The first utterance pays 4x, and the warm-up text was a red herring

The first utterance of each process measured **3.465s TTFA** against 0.7–0.9s for
every one after it, despite a warm-up `generate()` at startup. The warm-up text
was `"warm up"` — 7 characters. The theory was that MLX specialises kernels per
tensor shape, so a 90-character first chunk still compiled most of what it needed
on the critical path. Warming up on a sentence of roughly `FIRST_CHUNK_CHARS`
appeared to fix it: first utterance 0.845s, second 1.115s — flat.

That theory was wrong; the real cause is in the next section. The longer warm-up
text was kept anyway — it costs one sentence of startup time, and startup is
already dominated by the model load.

Two wrong turns worth recording. Activating the app before generating (on the
theory that background QoS was pinning it to efficiency cores) looked like it
helped once and did not survive a repeat; so did setting
`pthread_set_qos_class_self_np` on the worker thread, which was written, measured,
and deleted. Warming up on `"x" * 90` instead of real words hangs the process for
minutes — the phonemiser expands it to ninety separate letter names.

The panel is still shown before generating rather than at first audio, but for the
honest reason: it is immediate feedback that the key registered.

**The evidence against per-shape.** A later re-measure caught the first utterance after a
cold launch at **3.698s** (38 chars) with every one after it at 0.6–0.9s,
*including lengths never seen before* (41c, 60c) — which the kernel-specialisation
story does not explain. What was ruled out by measurement: the model is warm after
the startup warm-up (0.919s for the same 38-char text on the main thread, 0.782s on
a fresh worker thread), and the first `sd.OutputStream` open costs **1.093s against
0.684s**, worth only ~0.4s.

### The real cause is memory eviction, and it recurs after every idle period

The symptom is not limited to cold launch. A resident daemon left alone for
20–40 minutes pays it again — observed in normal use as **4.920s** followed by
**0.783s** for a comparable length. Measured while idle:

```
RSS 27 MB   |   phys_footprint 706 MB   |   phys_footprint_peak 5248 MB
```

706 MB of footprint with 27 MB resident means almost nothing is in physical RAM.
macOS's memory compressor squeezes out pages nothing has touched recently; the
model isn't unloaded, it's compressed, and the next keypress faults it back in
before generating a sample. RSS climbed 27 → 48 → 75 MB as it was exercised.

The log line now carries `idle Ns`, which confirmed it on the same text within one
session — the only variable being how long it sat:

```
359 chars | ttfa 5.439s | idle 815s
359 chars | ttfa 2.024s | idle  43s
```

Left unfixed deliberately. The fixes are all worse than the problem: a keepalive
timer generating throwaway audio every few minutes to keep pages hot burns battery
permanently to save 4s occasionally, and `mlock`-ing 700 MB denies the rest of the
system memory it should be allowed to reclaim.

### Static at the start of an utterance is an underrun, not a codec problem

Reported as crackle on the first words, worst after the app had been idle. The cause
was the shape of the playback loop: `with open_stream() as stream` **starts** the
stream, and the next thing the loop does is block waiting for the first sentence to
generate. So the device clock ran with nothing being written for the whole generation
window, and the hardware replayed stale buffer content. It scaled exactly the way the
report described, because that window is TTFA: ~0.6s warm, ~4s after the model's pages
have been compressed out.

Measured on Bluetooth earbuds, which is where the artifacts actually show up:

```
constructor (opens device, active=False)         3164 ms   <- the expensive half
start()                                            64 ms
started with no data for 2s -> write_available   8192      <- fully drained, no exception
stream.latency                                  0.6162s (14790 frames)
```

Underflow raises nothing — sounddevice ignores it — so none of this was ever going to
appear in a log.

**The obvious fix trades one artifact for another.** Deferring `start()` until the first
chunk was ready did remove the underrun, and the crackle came back on a narrower
window: the first two words. Starting a Bluetooth stream takes a moment before the link
carries clean audio, and deferring the start had moved that ramp onto the first
syllables.

Both have to be off the critical path, so the stream now starts early — during
generation, where the ramp costs nothing — and the consumer keeps it fed with `CUSHION`
blocks of silence while it waits. The cushion is deliberately thin: the buffer is 0.6s
deep, and anything queued ahead of the real audio delays speech by exactly that much.
Two blocks is 200ms of worst-case delay, and it's a named constant because it is tuned
by ear on the worst-case device, not derived.

TTFA after the change: 0.703s and 0.792s, unmoved.

Note for the stubs: the fake model had to start producing *audible* samples rather than
zeros, because silence is now a legitimate thing the player writes and the check has to
tell the two apart.

### Two ways the player wedged forever, both looking like "no sound"

Both presented identically and neither logged anything: the panel appeared, the menu
bar icon sat there, every later utterance was accepted over the socket and silently
discarded. `_serve` is a single thread, so anything that blocks it permanently
swallows the entire queue behind it. Diagnosis was `sample <pid>`: `serve_socket` in
`__accept` and `_serve` parked on a lock, with `AudioIOProc` still alive — a stream
held open by a thread that would never return.

**1. A failed generate skipped the queue sentinel.** `produce()` had no exception
handling, so anything raised inside `model.generate` bypassed its `pending.put(None)`
while the consumer sat in `pending.get()` with no timeout, waiting for a sentinel that
was never coming. Fixed at both ends: the sentinel now goes out from a `finally`, and
the consumer polls with a timeout, giving up when the producer is no longer alive.
The exception is printed, where it used to vanish.

**2. Barge-in did not resume a paused player.** `cancel()` sets `playing` to release
the writer before it checks the stop flag; `submit()` did not. Pause with space, then
press the hotkey, and the writer stayed in `playing.wait()` forever — permanently
dead until you happened to press space again. One line.

The tell for #2 in the log is `played 0.0s` with a large `total`: the utterance was
timed and never delivered a single block.

Both have regression checks, and both were confirmed to fail against the original code
rather than assumed to cover it — `test_producer_failure_does_not_wedge_the_player`
and `test_barge_in_while_paused`. A latent formatting crash surfaced while writing
them: `ttfa` stays `None` when nothing plays, and formatting it raised, turning a
handled failure into a second confusing error. It prints `n/a` now.

### A resident daemon outlives its audio device list

PortAudio enumerates devices **once**, when `sounddevice` is imported. In a process
meant to run for weeks, anything that changes the device set afterwards — plugging
in headphones, waking a display, Bluetooth connecting — invalidates that snapshot.
A fresh process opened the identical 24000/mono stream fine, which is what proves
staleness rather than a device that genuinely can't do it.

This has **two** failure modes, and only the loud one is obvious:

```
||PaMacCore (AUHAL)|| Error on line 1332: err='-10851'  (kAudioUnitErr_InvalidPropertyValue)
error: Error opening OutputStream: Internal PortAudio error [PaErrorCode -9986]
```

The dangerous one is silent. With Bluetooth earbuds connected after startup, the
open **succeeded**, every sample was written, and the log read perfectly healthy —
`played 10.6s` — while nothing was audible, because it was playing to the device
that had been default at import. A first fix that only re-enumerated on error was
therefore no fix at all.

`open_stream()` now re-enumerates unconditionally before every open. The reason is
purely that it measured **1-2ms**, not the ~0.2s assumed: too cheap to justify
detecting staleness when it can just be eliminated. Cheap enough that it doesn't
register against a 0.6s TTFA.

The check for it asserts on re-enumeration *and* on audio still playing — a stub
that plays happily to a stale device is exactly the bug, so testing only "sound came
out" would pass while broken.

### One instance only

`speak_app.py` unlinked and rebound the socket at startup, so a second launch —
an accidental double-click, exactly what happened here — silently stole every
trigger from the first instance and left two models resident. Guarded with an
`fcntl.flock` on `/tmp/kokoro-speak.lock`, which the kernel releases however the
process dies, so unlike a pidfile it can't go stale. The duplicate exits **0**,
not 1, so the LaunchAgent's `KeepAlive` (set to `SuccessfulExit: false`) doesn't
respawn it every 10 seconds forever.

- **Global hotkey ⌥⌘S**, owned by the app itself via
  `NSEvent.addGlobalMonitorForEventsMatchingMask_handler_` — no Shortcuts.app, no
  Services wiring. It reads the current selection, so there is no ⌘C either. Needs
  Accessibility permission, the same grant `copy_selection()` already needed, so one
  prompt covers both; the app prints whether the grant is live at startup, because a
  global monitor that was never granted is installed successfully and then silently
  never fires. It waits 200ms before posting its synthetic ⌘C: the hotkey's own ⌘ and
  ⌥ are still physically down at that point and would corrupt it.
  - A **local** monitor is installed alongside the global one. A global monitor only
    sees events delivered to *other* apps, so while our own panel holds focus — or in
    the moment after `esc` before focus has finished returning — the hotkey was dead.
  - `copy_selection()` falls back to the existing clipboard when the synthetic ⌘C
    doesn't change `changeCount`. Apps that auto-copy on selection (terminals
    especially) make the ⌘C a no-op, and the hotkey did nothing there. The cost is
    that pressing it with nothing selected reads whatever was on the clipboard.
- **Menu-bar item.** 🔈 idle, 🔊 speaking, ⏸ paused, with a Quit item. With no Dock
  icon and no window between utterances this is the only permanent sign the app is
  running — launching it from Spotlight looks like nothing happened otherwise, and
  with the single-instance guard in place it genuinely does nothing.
- **Floating control panel.** A borderless `NSPanel` at `NSFloatingWindowLevel`
  with `CanJoinAllSpaces | FullScreenAuxiliary`. Shows play/pause state, rate,
  and sentence progress. No tkinter in this Python build, so the GUI is Cocoa
  via `pyobjc`.
- **Keys:** `space` pause/resume · `←` `→` rate ±0.1 (0.5–2.0×) · `esc` stop.
  Two non-obvious requirements, both of which silently break the panel rather
  than erroring:
  - `NSWindow.canBecomeKeyWindow` returns NO for a **borderless** window, and a
    window that can't become key never receives `keyDown_`. `Panel` overrides it
    to YES; without that the keys look implemented and do nothing, while the
    keystrokes go to the app underneath.
  - `NSPanel` defaults `hidesOnDeactivate` to **YES**, so the panel vanished on
    every app or Space switch. Set to NO.
- Keys are ordinary key events in a focused panel, so they need **no**
  Accessibility or Input Monitoring permission. The cost is that showing the
  panel activates this app, taking focus from whatever you were reading until
  `esc`. Keeping focus in the other app would mean a global event tap
  (permission + swallowing space/esc system-wide) or modifier hotkeys like
  `⌥space` instead of bare ones.
- **Pause granularity 0.1s**, because playback writes the audio in
  `SAMPLE_RATE // 10`-sample blocks and checks the pause/stop flags between
  them. TTFA is unaffected: 146 chars measured 0.863s, in line with the table
  above.
- **Rate changes apply from the next sentence**, not mid-sentence — Kokoro's
  `speed` is a generation argument, so applying it sooner means regenerating
  audio already produced.
- **Barge-in works**: a new request cancels the current utterance
  (`stream.abort()` drops what is already buffered) rather than queueing
  behind it.
- **Auto-start:** `../com.kokoro.speak.plist` → `~/Library/LaunchAgents`,
  `launchctl load` it. `KeepAlive` restarts it if it dies; log at
  `/tmp/kokoro-speak.log`. Login Items would also work but wouldn't restart it.
- **Settings** are the `CONFIG` block at the top of the file. A preferences
  window for six constants isn't worth the code.

State machine check: `../test_speak_app.py` drives pause/resume/barge-in and the
rate clamp against a stub model and a stub output stream — no GUI, no speakers.

## Limits

- **Apple Silicon only.** MLX does not run on Windows or NVIDIA. The
  `mlx-community/Kokoro-82M-*` repos are MLX-format weights, unusable on CUDA.
  A Windows build is a separate implementation on `kokoro-onnx`; the
  architecture above (resident process, sentence chunking, one-slot lookahead,
  short first chunk) transfers unchanged, only the model backend swaps.
- **Selection.** Three ways to get text in, in increasing order of coverage and
  cost:
  1. `pbpaste | nc -U …` — needs a ⌘C by hand. No permissions, works everywhere.
  2. A Services-menu Quick Action passing the selection on stdin (wiring below).
     No permission, no ⌘C, but only works in apps that publish their selection
     to Services.
  3. Empty payload (`nc -U … < /dev/null`) — `copy_selection()` posts a
     synthetic ⌘C with `CGEventPost`, reads `NSPasteboard`, and puts the old
     clipboard back. Works in every app, needs Accessibility permission. Because
     the bundle stub `exec`s the venv interpreter, the process that needs the
     grant is that Python binary, not `Kokoro Speak.app` — the permission entry
     is named accordingly, which is ugly but harmless. Fixing the attribution
     means a real frozen bundle (py2app/PyInstaller).
- **Not logged to `results.csv`.** This is an app, not a benchmark run; TTFA
  is printed per utterance instead. The CSV schema in `metrics.py` has no TTFA
  column and file-synthesis rows aren't comparable to these anyway.

## Reproducing

```sh
open "Kokoro Speak.app"                      # the app; or, to see its output:
.venv/bin/python -u speak_app.py             # same thing with logs on the terminal
.venv/bin/python -u speak_mlx.py             # reference daemon from the measurements above

pbpaste | nc -U /tmp/kokoro-speak.sock       # speak the clipboard
nc -U /tmp/kokoro-speak.sock < /dev/null     # speak the current selection
```

Run only one daemon — they bind the same socket.

Use `-u`: the daemon's stdout is block-buffered when piped, so without it the
`loaded`/TTFA lines sit invisible in the pipe buffer while the process looks
hung.

Hotkey, clipboard version: Shortcuts.app → new shortcut → *Run Shell Script*
with the `pbpaste` line, input none → assign a key.

Hotkey, live-selection via synthetic ⌘C: *Run Shell Script* with
`/usr/bin/nc -U /tmp/kokoro-speak.sock < /dev/null`, input none. Then grant
Accessibility to the venv's `python` binary (System Settings → Privacy &
Security → Accessibility → **+** → ⌘⇧G →
`…/tts_models/.venv/bin/`). Without the grant `CGEventPost` fails silently and
the log says `nothing selected`.

Hotkey, live-selection via Services (no permission): same shortcut, but tick
*Use as Quick Action → Services Menu*, add *Receive text input from Quick
Actions*, and set the Run Shell Script action to Input: Shortcut Input, Pass
input: **to stdin**, script `/usr/bin/nc -U /tmp/kokoro-speak.sock`. Assign the
key in System Settings → Keyboard → Keyboard Shortcuts → Services.

Autostart: drag `Kokoro Speak.app` into Login Items, or use
`com.kokoro.speak.plist` → `~/Library/LaunchAgents` → `launchctl load` if you
want restart-on-crash and a log.
