"""Push-to-talk dictation: Qwen3-ASR-0.6B held in memory, mic straight to the clipboard.

⌥⌘R starts recording, ⌥⌘R again stops it, esc throws it away — the clip is transcribed in
one pass and pasted into whatever app you were in. A floating panel shows a live meter.
Not streaming: no text appears until you stop. At the 0.6B's measured RTF of 0.064
(docs/RESULTS.md) a 10s utterance transcribes in ~0.6s, so the wait after you stop is
short enough that partial results aren't worth the complexity.

Start:      .venv/bin/python -u dictate_app.py
Self-check: .venv/bin/python dictate_app.py --check
File check: .venv/bin/python dictate_app.py --file sample_for_tts.wav
Mic test:   .venv/bin/python dictate_app.py --once 5
Key debug:  .venv/bin/python dictate_app.py --keys

Every utterance is written to captures/ with its transcript appended to captures.jsonl —
you know what you said, so daily use accumulates the labelled set needed for a real WER
number, which docs/RESULTS.md currently lacks. Everything tunable is in the CONFIG block.
"""

import fcntl
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from AppKit import (
    NSApplication,
    NSApplicationActivateIgnoringOtherApps,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSColor,
    NSFloatingWindowLevel,
    NSFont,
    NSMakeRect,
    NSMenu,
    NSMenuItem,
    NSPanel,
    NSPasteboard,
    NSPasteboardTypeString,
    NSScreen,
    NSStatusBar,
    NSTextField,
    NSVariableStatusItemLength,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless,
    NSWorkspace,
)
from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
from Foundation import NSObject
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.utils import load_model
from PyObjCTools import AppHelper
from Quartz import (
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopGetCurrent,
    CGEventCreateKeyboardEvent,
    CGEventGetFlags,
    CGEventGetIntegerValueField,
    CGEventPost,
    CGEventSetFlags,
    CGEventTapCreate,
    CGEventTapEnable,
    kCFRunLoopCommonModes,
    kCGEventFlagMaskAlternate,
    kCGEventFlagMaskCommand,
    kCGEventFlagMaskControl,
    kCGEventFlagMaskShift,
    kCGEventKeyDown,
    kCGEventTapDisabledByTimeout,
    kCGEventTapDisabledByUserInput,
    kCGEventTapOptionDefault,
    kCGHeadInsertEventTap,
    kCGHIDEventTap,
    kCGKeyboardEventKeycode,
    kCGSessionEventTap,
)

# ---- CONFIG ---------------------------------------------------------------
REPO = "mlx-community/Qwen3-ASR-0.6B-4bit"   # 5.5x faster than whisper-small, no tail loop
LOCK_PATH = Path("/tmp/dictate.lock")
HERE = Path(__file__).parent
CAPTURE_DIR = HERE / "captures"
CAPTURE_LOG = HERE / "captures.jsonl"
WARMUP_AUDIO = HERE / "sample_for_tts.wav"
SAMPLE_RATE = 16000          # what Qwen3-ASR and Whisper both expect; avoids a resample
# Pin the language. Qwen3-ASR is multilingual and auto-detects, and given near-silence it
# does not return nothing — it confabulates in whatever language it guessed. Two recordings
# of an empty room produced fluent Devanagari and Chinese, which auto-paste would have put
# straight into whatever document was open. Set to None if you want auto-detection back.
LANGUAGE = "en"
MAX_SECONDS = 120            # auto-stop, so a forgotten hotkey can't record all afternoon
MIN_SECONDS = 0.3            # shorter than this is a double-tap, not speech
AUTO_PASTE = True            # False leaves the text on the clipboard for you to paste
ICON_IDLE, ICON_RECORDING, ICON_BUSY = "🎙", "🔴", "⏳"
PANEL_WIDTH, PANEL_HEIGHT = 430, 46
PANEL_BOTTOM_MARGIN = 90     # px above the bottom of the screen
HOLD_SECONDS = 1.6           # how long the finished transcript stays on screen
SILENCE_PEAK = 0.001         # below this the mic is producing digital silence, not quiet room
# Global hotkey: ⌥⌘R ("record"). keyCodes are physical keys, so 15 is 'r' on any layout.
#
# NOT ⌥⌘D, the obvious mnemonic: that is macOS's own "Turn Dock Hiding On/Off" symbolic
# hotkey. The system consumes those before anything here sees them, so the handler never
# fires and the only visible effect is the Dock flickering. Nothing in the code can override
# that — the key has to be one *macOS* doesn't own. Ordinary apps are a different matter:
# Brave and Chrome both use ⌥⌘R for reading mode, and the event tap below takes it from them.
# Use `--keys` to check a replacement actually arrives before trusting it.
HOTKEY_NAME = "⌥⌘R"
HOTKEY_KEYCODE = 15
HOTKEY_FLAGS = kCGEventFlagMaskCommand | kCGEventFlagMaskAlternate
# Compared against every modifier, not just ours, so ⌥⇧⌘R doesn't also fire.
MODIFIER_MASK = (
    kCGEventFlagMaskCommand | kCGEventFlagMaskAlternate
    | kCGEventFlagMaskShift | kCGEventFlagMaskControl
)
# ---------------------------------------------------------------------------

KEY_V, KEY_ESC = 9, 53
METER_WIDTH = 16
# Frames per audio callback. MUST be set explicitly. Left at sounddevice's default of 0,
# PortAudio picks, and on this machine it picked 15 frames — a callback every 0.94ms, ~1067
# per second. Each one takes the GIL, which starved the main thread badly enough that the
# hotkey handler (NSEvent monitors then, an event tap now) stopped seeing keystrokes and the
# panel's callAfter never painted, while PortAudio dropped ~80% of the input it could not hand
# over in time. One 100ms block
# instead of a hundred 1ms ones fixes all three at once, and matches the meter's refresh rate.
BLOCK_FRAMES = SAMPLE_RATE // 10
DEBUG = bool(os.environ.get("DICTATE_DEBUG"))   # verbose key + panel tracing


def single_instance() -> object:
    """Refuse to start if another copy holds the lock; return the lock so it outlives main.

    Two resident copies would both answer the hotkey and both keep a model in memory —
    on an 8GB machine that is the difference between working and swapping. An flock is
    released by the kernel however the process dies, so unlike a pidfile it can't go stale.
    """
    lock = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("dictate is already running")
        raise SystemExit(0)   # not a failure, so a KeepAlive agent won't respawn us
    return lock


def open_input_stream(on_block) -> sd.InputStream:
    """Open the *current* default mic, not the one that was default at import.

    PortAudio enumerates devices once, when sounddevice is imported. A daemon resident for
    days outlives that snapshot: connect AirPods afterwards and it keeps reading the stale
    default. The open succeeds and every callback fires, so the failure looks like "the
    model got worse" rather than "we recorded the wrong microphone". Same lesson as
    ../tts_models/speak_app.py's open_stream(), same 1-2ms fix.
    """
    sd._terminate()
    sd._initialize()
    return sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_FRAMES,
        callback=lambda indata, frames, t, status: on_block(indata.copy()),
    )


def transcribe(model, wav_path: Path) -> str:
    """One clip in, text out. Same call path as run_qwen3_asr_mlx.py, so the benchmark
    numbers in docs/RESULTS.md describe this app's latency directly."""
    result = generate_transcription(
        model=model, audio=str(wav_path), language=LANGUAGE,
        output_path=str(CAPTURE_DIR / "last_transcript"), format="txt", verbose=False,
    )
    if not hasattr(result, "text"):
        # generate_transcription also writes output_path, so it can plausibly return None;
        # str(None) would paste the word "None" into your document.
        raise RuntimeError(f"no .text on {type(result).__name__}: {result!r}")
    return result.text.strip()


def paste(text: str, auto: bool = AUTO_PASTE) -> None:
    """Put the text on the clipboard and press ⌘V into the frontmost app.

    auto=False is the safety valve: the caller passes it when it could not confirm that the
    app you were dictating into is frontmost again, in which case pressing ⌘V would type
    your sentence into whatever *is* — this app, or worse, something unrelated.

    ponytail: the transcript is left on the clipboard afterwards rather than restoring the
    previous contents. Restoring needs a delay long enough for the other app to have read
    the board, which is unknowable, and getting it wrong pastes the wrong thing — for
    dictation, keeping what you just said is the more useful outcome anyway.
    """
    board = NSPasteboard.generalPasteboard()
    board.clearContents()
    board.setString_forType_(text, NSPasteboardTypeString)
    if not auto:
        print("left on the clipboard — press ⌘V to paste it")
        return
    for keydown in (True, False):
        event = CGEventCreateKeyboardEvent(None, KEY_V, keydown)
        CGEventSetFlags(event, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, event)


def log_capture(wav_path: Path, text: str, seconds: float, latency: float) -> None:
    """Append the labelled pair. This is the WER corpus docs/RESULTS.md says is missing:
    you know what you actually said, so every dictation is a free ground-truth sample."""
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": REPO, "wav": wav_path.name, "text": text,
        "audio_seconds": round(seconds, 3), "latency_s": round(latency, 3),
        "rtf": round(latency / seconds, 4) if seconds else None,
    }
    with CAPTURE_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")


class MenuTarget(NSObject):
    def quit_(self, sender) -> None:
        NSApplication.sharedApplication().terminate_(None)


def meter(rms: float, width: int = METER_WIDTH) -> str:
    """RMS level to a bar. Gain set from the measured numbers: ordinary speech (~0.05 RMS)
    fills about half, and a quiet room (~0.003, measured by `--once`) shows nothing. Telling
    those two apart at a glance is the entire job."""
    filled = min(width, int(rms / 0.12 * width + 0.5))
    return "█" * filled + "·" * (width - filled)


class Panel(NSPanel):
    def canBecomeKeyWindow(self) -> bool:
        # NSWindow returns NO for borderless windows, and a window that can't become key
        # never receives keyDown_ — without this override esc is silently dead.
        return True


class KeyView(NSView):
    """Swallows keystrokes while the panel has focus, and makes esc cancel.

    Unhandled keys are dropped rather than passed to super, which would beep. While the
    panel is up you are dictating, not typing, so eating them is the useful behaviour.
    """

    def acceptsFirstResponder(self) -> bool:
        return True

    def keyDown_(self, event) -> None:
        if event.keyCode() == KEY_ESC:
            # Off the UI thread: cancel waits for focus to return, which must not block
            # the run loop that has to repaint the panel away.
            threading.Thread(target=RECORDER.cancel, daemon=True).start()


class Ui:
    """Menu-bar icon plus a borderless floating panel — the Panel/Ui pair from
    ../tts_models/speak_app.py, deliberately kept identical in the parts that decide *where*
    the panel appears.

    The activation is the load-bearing part, and not for the reason it looks like. An
    accessory app activating does not switch Spaces, so activating floats the panel over
    whatever you are in, fullscreen included. Ordering the window front *without* activating
    does not stay put: macOS sends you to the Desktop and back. Two attempts to keep focus
    on the app you were typing in — a non-activating panel, then a higher window level —
    both had that symptom, and only matching speak_app.py fixed it.

    The price is that the panel holds focus while you speak, so the synthetic ⌘V cannot be
    posted until focus has gone back. Recorder._restore_focus does that, and refuses to
    paste if it can't confirm it.

    All methods are safe to call from any thread.
    """

    def __init__(self) -> None:
        rect = NSMakeRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT)
        self.panel = Panel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        # Floating + join-all-spaces is what makes it visible over other apps, including
        # fullscreen ones. Without FullScreenAuxiliary it hides behind them.
        self.panel.setLevel_(NSFloatingWindowLevel)
        self.panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        # NSPanel defaults hidesOnDeactivate to YES, which hid the panel the moment focus
        # moved even though the recording was still running.
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        # Clicks fall through to the app underneath — nothing here is clickable, and an
        # invisible mouse target parked over someone's editor is its own bug.
        self.panel.setIgnoresMouseEvents_(True)

        self.view = view = KeyView.alloc().initWithFrame_(rect)
        view.setWantsLayer_(True)
        view.layer().setCornerRadius_(12.0)
        view.layer().setBackgroundColor_(NSColor.colorWithWhite_alpha_(0.10, 0.94).CGColor())
        self.panel.setContentView_(view)

        self.label = NSTextField.labelWithString_("")
        self.label.setFrame_(NSMakeRect(16, 13, PANEL_WIDTH - 32, 20))
        # Monospaced digits so the elapsed counter doesn't jitter the meter sideways.
        self.label.setFont_(NSFont.monospacedDigitSystemFontOfSize_weight_(13, 0))
        self.label.setTextColor_(NSColor.whiteColor())
        view.addSubview_(self.label)

        # Position is set per show(), not here — see the reason there.
        self.status_item = None   # filled in once the app is running, see AppDelegate

    def add_status_item(self) -> None:
        self.menu_target = MenuTarget.alloc().init()   # must outlive this method
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength
        )
        self.status_item.button().setTitle_(ICON_IDLE)
        menu = NSMenu.alloc().init()
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Quit Dictate", "quit:", "q")
        item.setTarget_(self.menu_target)
        menu.addItem_(item)
        self.status_item.setMenu_(menu)

        # A status item denied a menu bar slot still exists and answers every call — it just
        # reports a zero-height window and is invisible forever. That is what happens when the
        # process's LaunchServices registration and its NSBundle disagree, which is why
        # Dictate.app/Contents/MacOS/dictate launches python as a child rather than exec'ing
        # it. Log the size so the next occurrence is one line in /tmp/dictate.log rather than
        # an afternoon of "the app starts but there's no icon".
        #
        # callLater, not an immediate read: the slot is assigned during layout, one run-loop
        # turn after setMenu_. Measured synchronously it is 39x0 even when perfectly healthy,
        # which is a false alarm that costs exactly as much time as the real bug.
        AppHelper.callLater(0.5, self._report_slot)

    def _report_slot(self) -> None:
        frame = self.status_item.button().window().frame()
        if frame.size.height < 1:
            print(f"WARNING: menu bar item got no slot ({frame.size.width:.0f}x"
                  f"{frame.size.height:.0f}) — the icon will be invisible")
        else:
            print(f"menu bar icon ready ({frame.size.width:.0f}x{frame.size.height:.0f})")

    def icon(self, glyph: str) -> None:
        if self.status_item is None:
            return
        AppHelper.callAfter(self.status_item.button().setTitle_, glyph)

    def show(self, text: str) -> None:
        def run() -> None:
            self.label.setStringValue_(text)
            # Placed on every show rather than once at startup: mainScreen() is the display
            # holding keyboard focus, so on a second monitor the panel appears where you are
            # actually typing. Origin is added in because a secondary display's coordinates
            # do not start at zero.
            screen = NSScreen.mainScreen().frame()
            self.panel.setFrameOrigin_((
                screen.origin.x + (screen.size.width - PANEL_WIDTH) / 2,
                screen.origin.y + PANEL_BOTTOM_MARGIN,
            ))
            app = NSApplication.sharedApplication()
            app.unhide_(None)   # hide() below hides the whole app, not just the panel
            # The three lines that decide the panel lands *here* and not on the Desktop —
            # see the class docstring. Focus comes back before anything is pasted.
            app.activateIgnoringOtherApps_(True)
            self.panel.makeKeyAndOrderFront_(None)
            self.panel.makeFirstResponder_(self.view)
            # onActiveSpace=False means macOS could not place the panel where you are, which
            # is the state that gets you sent to the Desktop. Logged unconditionally in that
            # case, because it is invisible from the inside otherwise.
            on_space = self.panel.isOnActiveSpace()
            if DEBUG or not on_space:
                front = NSWorkspace.sharedWorkspace().frontmostApplication()
                print(f"  panel: onActiveSpace={on_space} level={self.panel.level()} "
                      f"frontmost={front.localizedName() if front else None}", flush=True)

        AppHelper.callAfter(run)

    def status(self, text: str) -> None:
        AppHelper.callAfter(self.label.setStringValue_, text)

    def hide(self) -> None:
        def run() -> None:
            self.panel.orderOut_(None)
            # hide_, not just orderOut_: ordering the window out leaves this app active, so
            # the app you were typing in stays unfocused and the ⌘V has nowhere to land.
            # Safe to call twice — _restore_focus hides, and _finish's finally hides again.
            NSApplication.sharedApplication().hide_(None)

        AppHelper.callAfter(run)


class AppDelegate(NSObject):
    """Creates the status item at the one moment that reliably works — a status item built
    before the app is running gets no menu bar slot. Same reasoning as speak_app.py."""

    def applicationDidFinishLaunching_(self, notification) -> None:
        self.ui.add_status_item()


class Recorder:
    """Toggled by the hotkey: one recording at a time, transcribed off the UI thread."""

    def __init__(self, model, ui: Ui) -> None:
        self.model, self.ui = model, ui
        self.lock = threading.Lock()
        self.stream = None
        self.blocks: list[np.ndarray] = []
        self.session = 0   # so an old auto-stop timer can't cut a later recording short
        self.started = 0.0
        self.level = 0.0   # written by the audio callback, read by the meter thread
        self.peak = 0.0
        self.target = None   # the app to paste back into, captured before we take focus

    def toggle(self) -> None:
        with self.lock:
            if self.stream is None:
                self._start()
                return
            blocks, self.blocks = self.blocks, []
            stream, self.stream = self.stream, None
        stream.stop()
        stream.close()
        self._finish(blocks)

    def _on_block(self, block: np.ndarray) -> None:
        """Audio callback. Keep it to arithmetic: this runs on PortAudio's realtime thread,
        where anything slow shows up as dropped input rather than as an error."""
        self.blocks.append(block)
        self.level = float(np.sqrt(np.mean(np.square(block))))
        self.peak = max(self.peak, float(np.abs(block).max()))

    def _start(self) -> None:
        self.blocks = []
        self.session += 1
        self.level = self.peak = 0.0
        self.started = time.perf_counter()
        # Whose window the transcript belongs to, recorded *before* show() activates us and
        # makes the answer "Dictate". Asking again at paste time is too late.
        self.target = NSWorkspace.sharedWorkspace().frontmostApplication()
        # Paint before opening the device, not after: re-enumerating PortAudio and starting
        # the stream costs ~0.2s, and a panel that appears a fifth of a second after the
        # keypress reads as a hotkey that didn't work.
        self.ui.icon(ICON_RECORDING)
        self.ui.show(f"🔴 recording…   {HOTKEY_NAME} to stop")
        self.stream = open_input_stream(self._on_block)
        self.stream.start()
        print("recording…")
        threading.Thread(target=self._tick, args=(self.session,), daemon=True).start()
        # Auto-stop guard. A timer rather than a check inside the callback, so it fires even
        # if the mic goes silent or the device disappears mid-recording. Daemon, or a pending
        # timer keeps the process alive for MAX_SECONDS after you quit.
        timer = threading.Timer(MAX_SECONDS, self._auto_stop, args=(self.session,))
        timer.daemon = True
        timer.start()

    def _tick(self, session: int) -> None:
        """Repaint the meter at 10Hz until this recording ends.

        The menu-bar icon can say "armed" but not "hearing you", and those differ: a denied
        Microphone permission records flawless digital silence, so the app looks healthy right
        up to the empty transcript. SILENCE_PEAK separates that from a merely quiet room.
        """
        while True:
            with self.lock:
                if self.stream is None or self.session != session:
                    return
            elapsed = time.perf_counter() - self.started
            if elapsed > 1.0 and self.peak < SILENCE_PEAK:
                self.ui.status(f"🔴 {elapsed:5.1f}s   no signal — check Microphone permission")
            else:
                self.ui.status(f"🔴 {elapsed:5.1f}s   {meter(self.level)}   {HOTKEY_NAME} to stop")
            time.sleep(0.1)

    def _auto_stop(self, session: int) -> None:
        # Only stop the recording this timer was created for. Without the session check, a
        # timer left over from an earlier clip fires mid-way through a later one and stops
        # it early — the recording just ends on its own with no indication why.
        with self.lock:
            if self.stream is None or self.session != session:
                return
        print(f"auto-stopped at {MAX_SECONDS}s")
        self.toggle()

    def cancel(self) -> None:
        """Throw away the recording in progress — esc, from the panel's KeyView.

        Only reachable while the panel has focus, which is exactly while recording. Bumping
        the session is what stops the meter thread and orphans the auto-stop timer.
        """
        with self.lock:
            if self.stream is None:
                return
            self.blocks = []
            stream, self.stream = self.stream, None
            self.session += 1
        stream.stop()
        stream.close()
        print("cancelled")
        self.ui.icon(ICON_IDLE)
        self._restore_focus()

    def _restore_focus(self) -> bool:
        """Take the panel down and wait until the app that was frontmost when recording
        started is frontmost again. False means don't paste.

        Polled rather than slept: activation is asynchronous, so a fixed delay is either too
        short — ⌘V into the wrong window — or slower than it needs to be on every single
        dictation. The activate call goes through the main thread because AppKit wants it
        there; frontmostApplication is only read.
        """
        self.ui.hide()
        target = self.target
        if target is None or target.processIdentifier() == os.getpid():
            return False
        deadline = time.time() + 1.0
        while time.time() < deadline:
            front = NSWorkspace.sharedWorkspace().frontmostApplication()
            if front is not None and front.processIdentifier() == target.processIdentifier():
                return True
            AppHelper.callAfter(
                target.activateWithOptions_, NSApplicationActivateIgnoringOtherApps
            )
            time.sleep(0.05)
        print(f"focus did not return to {target.localizedName()}")
        return False

    def _flash(self, text: str) -> None:
        """Hold a message on the panel long enough to read it; the caller's finally clause
        takes the panel down afterwards. Runs on the hotkey's own worker thread, never the UI
        thread, so sleeping here doesn't freeze anything."""
        self.ui.status(text if len(text) <= 64 else text[:63] + "…")
        time.sleep(HOLD_SECONDS)

    def _finish(self, blocks: list[np.ndarray]) -> None:
        self.ui.icon(ICON_BUSY)
        try:
            if not blocks:
                print("nothing recorded — is Microphone permission granted?")
                self._flash("nothing recorded — check Microphone permission")
                return
            audio = np.concatenate(blocks).reshape(-1)
            seconds = len(audio) / SAMPLE_RATE
            if seconds < MIN_SECONDS:
                print(f"too short ({seconds:.2f}s), ignored")
                self._flash(f"too short ({seconds:.2f}s), ignored")
                return

            # Don't ask the model what silence says. It answers — fluently, and sometimes in
            # another language — and AUTO_PASTE would put that in your document. Measured off
            # the clip rather than self.peak so a recording started immediately after this one
            # can't overwrite the value mid-check.
            peak = float(np.abs(audio).max())
            if peak < SILENCE_PEAK:
                print(f"silent clip (peak {peak:.5f}), not transcribed")
                self._flash("nothing heard — check Microphone permission")
                return

            CAPTURE_DIR.mkdir(exist_ok=True)
            wav_path = CAPTURE_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.wav"
            sf.write(wav_path, audio, SAMPLE_RATE)

            self.ui.status(f"⏳ transcribing {seconds:.1f}s…")
            t0 = time.perf_counter()
            text = transcribe(self.model, wav_path)
            latency = time.perf_counter() - t0
            print(f"{seconds:.2f}s audio → {latency:.2f}s (RTF {latency / seconds:.3f}): {text!r}")

            if not text:
                print("empty transcript, nothing pasted")
                self._flash("empty transcript — nothing heard")
                return
            log_capture(wav_path, text, seconds, latency)
            # No panel flash of the transcript on the way out: it would mean either delaying
            # the paste by HOLD_SECONDS or re-activating this app immediately after posting
            # ⌘V, and a keystroke the target app has not processed yet does not survive that.
            # The text appearing in your document is the feedback.
            paste(text, auto=AUTO_PASTE and self._restore_focus())
        finally:
            self.ui.icon(ICON_IDLE)
            self.ui.hide()


def install_tap(on_key) -> object:
    """Install a keyDown event tap on the run loop; return it, or None if it was refused.

    on_key(keycode, flags) -> bool, where True means swallow the keystroke.

    A tap rather than the NSEvent monitors this app used to have, because a monitor can only
    *observe*: the keystroke is delivered to the frontmost app regardless, so ⌥⌘R started a
    recording *and* toggled Brave's and Chrome's reading mode. Only a tap sits ahead of
    delivery and can drop an event, which is the one way to take a combination another app
    has already claimed. It also replaces both monitors rather than one: a session tap sees
    events bound for this app too, which is the gap the local monitor existed to cover.

    ponytail: a session-wide tap for a single combination is heavier than it needs to be —
    every keystroke you type is handed to this process, and Carbon's RegisterEventHotKey
    would consume the one key without seeing any of the others. That needs ctypes structs and
    a Carbon event handler, so it is the upgrade path, not the first version. Nothing here
    reads what the keys mean, only their codes, and DICTATE_DEBUG=1 prints those — so don't
    leave it on while typing a password.
    """
    def callback(proxy, event_type, event, refcon):
        if event_type in (kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput):
            # macOS disables a tap whose callback was too slow, and never re-enables it on
            # its own: the hotkey would be dead for the rest of the session with nothing in
            # the log. The disable arrives as an event here, so it can be undone immediately.
            print("event tap was disabled — re-enabling", flush=True)
            CGEventTapEnable(tap, True)
            return event
        keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        flags = int(CGEventGetFlags(event)) & MODIFIER_MASK
        swallow = on_key(int(keycode), flags)
        return None if swallow else event

    tap = CGEventTapCreate(
        kCGSessionEventTap, kCGHeadInsertEventTap, kCGEventTapOptionDefault,
        1 << kCGEventKeyDown, callback, None,
    )
    if tap is None:
        return None
    source = CFMachPortCreateRunLoopSource(None, tap, 0)
    CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes)
    CGEventTapEnable(tap, True)
    # Both must outlive this function: the source is what keeps the tap on the run loop.
    return (tap, source)


def install_hotkey(recorder: Recorder) -> object:
    """Own the hotkey directly, so no Shortcuts.app or Services wiring is needed.

    Needs Accessibility permission — the same grant paste() needs to post its ⌘V, so one
    prompt covers both. Without it CGEventTapCreate simply returns None, so the failure is
    reported rather than silent: a dead hotkey with no explanation is the worst outcome here.

    What needs the grant depends on how this was launched, and the two are separate entries:
    started from Dictate.app the responsible process is the bundle, so System Settings wants
    "Dictate"; started from a terminal it is the interpreter's real path. Granting one does
    nothing for the other — which is also why ../tts_models/speak_app.py being trusted has
    never helped here.
    """
    trusted = AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})

    def fire() -> None:
        # The hotkey's own ⌘ and ⌥ are still physically down and would corrupt the
        # synthetic ⌘V at the end of a recording. Wait for the release rather than race it.
        time.sleep(0.2)
        recorder.toggle()

    def on_key(keycode: int, flags: int) -> bool:
        hit = keycode == HOTKEY_KEYCODE and flags == HOTKEY_FLAGS
        if DEBUG:
            print(f"  keyCode={keycode} flags=0x{flags:x} hit={hit}", flush=True)
        if hit:
            threading.Thread(target=fire, daemon=True).start()
        return hit   # swallowed, so the frontmost app never sees ⌥⌘R

    tap = install_tap(on_key)
    print(f"hotkey {HOTKEY_NAME} installed "
          f"(accessibility {'granted' if trusted else 'NOT granted — hotkey is dead'})"
          if tap else "hotkey NOT installed — the event tap was refused")
    bundle = os.environ.get("DICTATE_BUNDLE")
    needs = bundle if bundle else str(Path(sys.executable).resolve())
    print(f"  grant Accessibility to: {needs}")
    return tap


def load_and_warm():
    t0 = time.perf_counter()
    model = load_model(REPO)
    # Burn the one-time Metal kernel-compile tax before accepting a hotkey press, or the
    # first dictation of the session pays ~4x the latency of every later one — the exact
    # confound that invalidated the first benchmark run (docs/RESULTS.md bug #1).
    if WARMUP_AUDIO.exists():
        CAPTURE_DIR.mkdir(exist_ok=True)
        transcribe(model, WARMUP_AUDIO)
    print(f"loaded {REPO} in {time.perf_counter() - t0:.1f}s")
    return model


def cli_file(path: str) -> None:
    """Self-check: transcribe a file, no mic, no Accessibility, no paste."""
    model = load_and_warm()
    t0 = time.perf_counter()
    text = transcribe(model, Path(path))
    assert text, "empty transcript"
    print(f"{path} → {time.perf_counter() - t0:.2f}s: {text!r}")


def cli_once(seconds: float) -> None:
    """Record from the mic for N seconds, transcribe, print. Tests the audio path."""
    model = load_and_warm()
    blocks: list[np.ndarray] = []
    stream = open_input_stream(blocks.append)
    with stream:
        print(f"recording {seconds}s — speak now…")
        time.sleep(seconds)
    audio = np.concatenate(blocks).reshape(-1)
    CAPTURE_DIR.mkdir(exist_ok=True)
    wav_path = CAPTURE_DIR / f"once-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    sf.write(wav_path, audio, SAMPLE_RATE)
    t0 = time.perf_counter()
    text = transcribe(model, wav_path)
    print(f"{len(audio) / SAMPLE_RATE:.2f}s audio → {time.perf_counter() - t0:.2f}s: {text!r}")


def cli_check() -> None:
    """No mic, no model, no permissions — just the meter arithmetic."""
    assert meter(0.0) == "·" * METER_WIDTH
    assert meter(1.0) == "█" * METER_WIDTH
    quiet, loud = meter(0.003), meter(0.05)   # measured room noise vs ordinary speech
    assert quiet.count("█") == 0, quiet       # a quiet room must read as nothing
    assert 4 <= loud.count("█") <= 12, loud   # speech must be unmistakably mid-scale

    # The silence gate: a dead mic must be rejected, a quiet room must still go through.
    dead = np.zeros(SAMPLE_RATE, dtype="float32")
    room = (np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 0.003).astype("float32")
    assert float(np.abs(dead).max()) < SILENCE_PEAK
    assert float(np.abs(room).max()) > SILENCE_PEAK, float(np.abs(room).max())
    print(f"checks passed: quiet={quiet!r} speech={loud!r} silence-gate ok")


def cli_keys() -> None:
    """Print keyCode and modifiers for every key you press. Nothing is swallowed here.

    This distinguishes the two ways a hotkey dies, which look identical from the outside:
    Accessibility not granted (nothing prints at all) versus macOS owning the combination
    already and consuming it first (everything prints except that one). ⌥⌘D is the second
    kind — the Dock-hiding shortcut — which is why this app uses ⌥⌘R. An *app* owning the
    combination is a third case that looks fine here and still misbehaves, because both you
    and the app act on it; that is what the tap swallowing the key fixes.
    """
    global _KEYS
    trusted = AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
    print(f"accessibility {'granted' if trusted else 'NOT granted — nothing will print'}")
    print(f"target: keyCode={HOTKEY_KEYCODE} flags=0x{int(HOTKEY_FLAGS):x} ({HOTKEY_NAME})")
    print("press keys anywhere, ^C to stop")

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    def on_key(keycode: int, flags: int) -> bool:
        hit = " <-- MATCH" if keycode == HOTKEY_KEYCODE and flags == HOTKEY_FLAGS else ""
        print(f"keyCode={keycode:3d} flags=0x{flags:x}{hit}", flush=True)
        return False   # pass everything through: this is a probe, not the hotkey

    _KEYS = install_tap(on_key)
    if _KEYS is None:
        print("the event tap was refused — grant Accessibility and try again")
        return
    AppHelper.runEventLoop()


def main() -> None:
    global _LOCK, _DELEGATE, _HOTKEY, RECORDER
    _LOCK = single_instance()   # must outlive main(): closing the file drops the lock
    model = load_and_warm()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon
    ui = Ui()
    _DELEGATE = AppDelegate.alloc().init()   # setDelegate_ doesn't retain
    _DELEGATE.ui = ui
    app.setDelegate_(_DELEGATE)
    # Module-global because KeyView is instantiated by AppKit, which gives no way to hand it
    # a reference — same reason speak_app.py has a global PLAYER.
    RECORDER = Recorder(model, ui)
    _HOTKEY = install_hotkey(RECORDER)   # monitors die if garbage collected
    AppHelper.runEventLoop()


if __name__ == "__main__":
    if "--check" in sys.argv:
        cli_check()
    elif "--keys" in sys.argv:
        cli_keys()
    elif "--file" in sys.argv:
        cli_file(sys.argv[sys.argv.index("--file") + 1])
    elif "--once" in sys.argv:
        cli_once(float(sys.argv[sys.argv.index("--once") + 1]))
    else:
        main()
