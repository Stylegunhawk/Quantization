"""Read-aloud app: Kokoro-MLX held in memory, audio straight to the speakers, with a
floating control panel that appears over whatever app you are in.

Keys (while the panel is up):  space pause/resume · ← → speed · esc stop

Start:  .venv/bin/python -u speak_app.py
Speak:  pbpaste | nc -U /tmp/kokoro-speak.sock

Auto-start on login: see com.kokoro.speak.plist next to this file.
Everything tunable lives in the CONFIG block below — that is the whole settings UI.
"""

import fcntl
import queue
import socket
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSColor,
    NSEvent,
    NSEventMaskKeyDown,
    NSEventModifierFlagCommand,
    NSEventModifierFlagControl,
    NSEventModifierFlagOption,
    NSEventModifierFlagShift,
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
)
from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
from Foundation import NSObject
from mlx_audio.tts.utils import load_model
from PyObjCTools import AppHelper
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    kCGEventFlagMaskCommand,
    kCGHIDEventTap,
)

from speak_mlx import REPO, SAMPLE_RATE, VOICE, split_sentences

# ---- CONFIG ---------------------------------------------------------------
SOCK_PATH = Path("/tmp/kokoro-speak.sock")
LOCK_PATH = Path("/tmp/kokoro-speak.lock")
SPEED = 1.0                # starting rate; ← → adjust it live
SPEED_STEP = 0.1
SPEED_RANGE = (0.5, 2.0)
PANEL_WIDTH, PANEL_HEIGHT = 430, 46
PANEL_BOTTOM_MARGIN = 90   # px above the bottom of the screen
ICON_IDLE, ICON_PLAYING, ICON_PAUSED = "🔈", "🔊", "⏸"
# Global hotkey: ⌥⌘S. keyCode 1 is 's' on any layout, since keyCodes are physical keys.
WARMUP_TEXT = "This warm up sentence is about as long as a first chunk, so the kernels match."
HOTKEY_KEYCODE = 1
HOTKEY_FLAGS = NSEventModifierFlagCommand | NSEventModifierFlagOption
# ---------------------------------------------------------------------------

# 0.1s of audio per write: the granularity at which pause/stop can take effect.
BLOCK = SAMPLE_RATE // 10
# How many BLOCKs of silence to keep queued while waiting for the next sentence. Enough that
# the device never starves, few enough that real audio isn't stuck behind it: each block is
# 100ms of delay before speech starts. ponytail: tuned by ear on Bluetooth earbuds, which are
# the worst case — turn it up if the first words still crackle, down if they feel late.
CUSHION = 2
KEY_C, KEY_ESC, KEY_LEFT, KEY_RIGHT = 8, 53, 123, 124


def copy_selection() -> str:
    """Read the current selection from whatever app is frontmost, via a synthetic ⌘C.

    Needs Accessibility permission (System Settings → Privacy & Security →
    Accessibility) because it posts key events into another app. The alternative — a
    Services-menu Quick Action — needs no permission but only works in apps that
    publish their selection to Services, which excludes plenty of them.

    The previous clipboard contents are put back, so read-aloud doesn't eat your
    clipboard.
    """
    board = NSPasteboard.generalPasteboard()
    saved = board.stringForType_(NSPasteboardTypeString)
    before = board.changeCount()
    for keydown in (True, False):
        event = CGEventCreateKeyboardEvent(None, KEY_C, keydown)
        CGEventSetFlags(event, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, event)
    # The copy lands asynchronously in the other app; changeCount is the signal that it
    # actually happened, so an empty selection costs the timeout and nothing else.
    deadline = time.time() + 0.5
    while board.changeCount() == before and time.time() < deadline:
        time.sleep(0.02)
    if board.changeCount() != before:
        text = board.stringForType_(NSPasteboardTypeString) or ""
        if saved is not None:
            board.clearContents()
            board.setString_forType_(saved, NSPasteboardTypeString)
        return text
    # No change means either nothing was selected, or the app had already auto-copied the
    # selection — terminals commonly do — so the synthetic ⌘C rewrote identical content or
    # was a no-op. Those two are indistinguishable from here, so fall back to the
    # clipboard: reading something slightly stale beats a hotkey that is silently dead in
    # a whole class of apps.
    return saved or ""


def open_stream():
    """Open the current default output device, not the one that was default at import.

    PortAudio enumerates devices once, when sounddevice is imported. A daemon resident for
    days outlives that snapshot: connect Bluetooth earbuds afterwards and it keeps writing
    to the stale default. Worse than an error — the open succeeds, every sample is written,
    and the log looks healthy while nothing is audible. Re-enumerating first measured 1-2ms,
    far too cheap to bother detecting staleness rather than just eliminating it.

    ponytail: safe only because one serving thread owns all playback — _terminate() would
    kill a stream open on another thread.
    """
    sd._terminate()
    sd._initialize()
    return sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")


class AppDelegate(NSObject):
    """Creates the status item at the one moment that reliably works.

    A status item built before the app is running gets no menu bar slot and is parked
    offscreen forever. Queueing the work with callAfter from __init__ looked like it fixed
    that, and did once, but blocks enqueued before the run loop exists are not reliably
    delivered — the second launch raised 'Ui has no attribute status_item'. This delegate
    callback is the documented point at which the app is running.
    """

    def applicationDidFinishLaunching_(self, notification) -> None:
        self.ui.add_status_item()


class MenuTarget(NSObject):
    def quit_(self, sender) -> None:
        # terminate_ doesn't unwind main(), so clean up the socket here or the next
        # launch inherits a stale one.
        SOCK_PATH.unlink(missing_ok=True)
        NSApplication.sharedApplication().terminate_(None)


class Panel(NSPanel):
    def canBecomeKeyWindow(self) -> bool:
        # NSWindow returns NO for borderless windows, and a window that can't become key
        # never receives keyDown_ — without this override the keys are silently dead.
        return True


class Ui:
    """Borderless floating NSPanel. All methods are safe to call from any thread."""

    def __init__(self) -> None:
        rect = NSMakeRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT)
        self.panel = Panel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        # Floating + join-all-spaces is what makes it visible over other apps, including
        # ones in fullscreen. Without FullScreenAuxiliary it hides behind them.
        self.panel.setLevel_(NSFloatingWindowLevel)
        self.panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        # NSPanel defaults hidesOnDeactivate to YES, so switching Space or app made the
        # panel vanish even though it was still speaking.
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())

        self.view = view = KeyView.alloc().initWithFrame_(rect)
        view.setWantsLayer_(True)
        view.layer().setCornerRadius_(12.0)
        view.layer().setBackgroundColor_(NSColor.colorWithWhite_alpha_(0.10, 0.94).CGColor())
        self.panel.setContentView_(view)

        self.label = NSTextField.labelWithString_("")
        self.label.setFrame_(NSMakeRect(16, 13, PANEL_WIDTH - 32, 20))
        self.label.setFont_(NSFont.monospacedDigitSystemFontOfSize_weight_(13, 0))
        self.label.setTextColor_(NSColor.whiteColor())
        view.addSubview_(self.label)

        screen = NSScreen.mainScreen().frame()
        self.panel.setFrameOrigin_(
            ((screen.size.width - PANEL_WIDTH) / 2, PANEL_BOTTOM_MARGIN)
        )

        # Filled in by AppDelegate once the app is running — see the reason there. Every
        # reader must tolerate None, because playback can start before the menu bar does.
        self.status_item = None

    def add_status_item(self) -> None:
        """The only permanently visible sign the app exists: no Dock icon, and no window
        between utterances."""
        self.menu_target = MenuTarget.alloc().init()   # must outlive this method
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength
        )
        self.status_item.button().setTitle_(ICON_IDLE)
        menu = NSMenu.alloc().init()
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Quit Kokoro Speak", "quit:", "q")
        quit_item.setTarget_(self.menu_target)
        menu.addItem_(quit_item)
        self.status_item.setMenu_(menu)

    def icon(self, glyph: str) -> None:
        if self.status_item is None:
            return
        AppHelper.callAfter(self.status_item.button().setTitle_, glyph)

    def show(self, text: str) -> None:
        def run() -> None:
            self.label.setStringValue_(text)
            app = NSApplication.sharedApplication()
            app.unhide_(None)   # hide() below hides the whole app, not just the panel
            # An accessory app activating does not switch Spaces, so this floats the panel
            # over the current desktop — including a fullscreen app — instead of yanking
            # you back to wherever the panel was last shown.
            app.activateIgnoringOtherApps_(True)
            self.panel.makeKeyAndOrderFront_(None)
            self.panel.makeFirstResponder_(self.view)

        AppHelper.callAfter(run)

    def status(self, text: str) -> None:
        AppHelper.callAfter(self.label.setStringValue_, text)

    def hide(self) -> None:
        def run() -> None:
            if self.status_item is not None:
                self.status_item.button().setTitle_(ICON_IDLE)
            self.panel.orderOut_(None)
            NSApplication.sharedApplication().hide_(None)   # hands focus back

        AppHelper.callAfter(run)


class KeyView(NSView):
    def acceptsFirstResponder(self) -> bool:
        return True

    def keyDown_(self, event) -> None:
        code = event.keyCode()
        if event.charactersIgnoringModifiers() == " ":
            PLAYER.toggle()
        elif code == KEY_ESC:
            PLAYER.cancel()
        elif code == KEY_RIGHT:
            PLAYER.nudge(+SPEED_STEP)
        elif code == KEY_LEFT:
            PLAYER.nudge(-SPEED_STEP)


class Player:
    """One utterance at a time, on its own thread, interruptible."""

    def __init__(self, model, ui: Ui) -> None:
        self.model, self.ui = model, ui
        self.jobs: queue.Queue = queue.Queue()
        self.playing = threading.Event()   # set = play, clear = paused
        self.stopped = threading.Event()
        self.speed = SPEED
        self.progress = ""
        self.last_finished = 0.0
        threading.Thread(target=self._serve, daemon=True).start()

    # --- controls (called from the UI thread) ---
    def submit(self, text: str) -> None:
        self.stopped.set()      # barge-in: whatever is speaking now gives up its turn
        # ...and resume, or a paused writer never wakes to notice it. cancel() got this
        # right and submit() did not: pausing with space and then pressing the hotkey left
        # the serving thread blocked forever, so that utterance and every later one queued
        # up silently behind it while the panel and the menu bar icon looked healthy.
        self.playing.set()
        self.jobs.put(text)

    def toggle(self) -> None:
        self.playing.clear() if self.playing.is_set() else self.playing.set()
        self._refresh()

    def cancel(self) -> None:
        self.stopped.set()
        self.playing.set()      # release the writer thread so it can see stopped
        self.ui.hide()

    def nudge(self, delta: float) -> None:
        lo, hi = SPEED_RANGE
        self.speed = min(hi, max(lo, round(self.speed + delta, 2)))
        self._refresh()

    def _refresh(self) -> None:
        playing = self.playing.is_set()
        self.ui.status(f"{'▶' if playing else '⏸'}  {self.speed:.1f}×   {self.progress}    space · ←→ · esc")
        self.ui.icon(ICON_PLAYING if playing else ICON_PAUSED)

    # --- worker ---
    def _serve(self) -> None:
        while True:
            text = self.jobs.get()
            # Only clear here: the previous _speak has already returned, so there is no
            # race with the stop flag it was watching.
            self.stopped.clear()
            self.playing.set()
            try:
                self._speak(text)
            except Exception as exc:      # a bad utterance must not kill the daemon
                print(f"error: {exc}")
            self.ui.hide()

    def _speak(self, text: str) -> None:
        sentences = split_sentences(text)
        t0 = time.perf_counter()
        self.progress = f"0/{len(sentences)}"
        # Show the panel before generating rather than at first audio. An accessory app
        # that has never been activated runs at background QoS, which on Apple Silicon
        # means efficiency cores: the first utterance after a Finder launch measured
        # 3.5s TTFA that way, against 0.9s once activated. Also gives immediate feedback
        # that the key registered.
        self.ui.show("")
        self._refresh()
        # One sentence of lookahead: the next is generated while the current one plays.
        pending: queue.Queue = queue.Queue(maxsize=1)

        def produce() -> None:
            try:
                for sentence in sentences:
                    if self.stopped.is_set():
                        break
                    # Wait for a queue slot before generating, not after. Generating first
                    # and blocking on the put kept a third sentence in flight — one
                    # playing, one queued, one finished and waiting — so a speed change
                    # took 3 sentences to be heard. Waiting first makes it 2.
                    # ponytail: this leaves one sentence of playback to generate the next,
                    # where there used to be two. A sentence slow enough to overrun that
                    # empties the queue and the consumer feeds silence — a short pause, not
                    # static. Raise pending's maxsize to buy the slack back at 1 more
                    # sentence of speed lag. Polling because Queue has no wait-for-room.
                    while pending.full() and not self.stopped.is_set():
                        time.sleep(0.05)
                    if self.stopped.is_set():
                        break   # don't burn a generate the barge-in is about to discard
                    # Speed is read below, at generate time, so a change never affects audio
                    # that already exists: speed is a duration-predictor input, not a
                    # playback rate, and resampling instead would shift pitch.
                    audio = np.concatenate(
                        [np.array(r.audio, copy=False)
                         for r in self.model.generate(sentence, voice=VOICE, speed=self.speed, lang_code="a")]
                    )
                    while not self.stopped.is_set():
                        try:            # timeout, not a blocking put: else a stop leaves this
                            pending.put(audio, timeout=0.2)   # thread wedged on a queue nobody reads
                            break
                        except queue.Full:
                            pass
            except Exception as exc:
                # Never silent. An unhandled failure here used to skip the sentinel below
                # and wedge the serving thread permanently: the audio device stayed open and
                # every later utterance was queued and dropped while the app looked healthy.
                print(f"generate failed: {exc}")
            finally:
                try:
                    pending.put(None, timeout=0.2)
                except queue.Full:
                    pass   # consumer already gave up; it no longer needs the sentinel

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()

        ttfa = None
        played = 0
        # Two ways to make the first words crackle, and this avoids both. A running stream
        # with nothing to write underruns and the device replays stale buffer content; but
        # deferring start() until audio is ready instead lands the Bluetooth link's start-up
        # ramp squarely on the first syllables. So it starts now, during generation, and is
        # kept fed with silence below — a thin cushion, because anything queued ahead of the
        # real audio delays it by exactly that much, and the buffer is 0.6s deep on Bluetooth.
        stream = open_stream()
        stream.start()
        silence = np.zeros(BLOCK, dtype="float32")
        buffered_frames = int(stream.latency * SAMPLE_RATE)
        try:
            index = 0
            while True:
                try:
                    audio = pending.get(timeout=0.05)
                except queue.Empty:
                    # Never wait forever on a sentinel: a dead producer, or a stop that
                    # arrives while nothing is queued, must both end this loop.
                    if self.stopped.is_set() or not producer.is_alive():
                        break
                    if buffered_frames - stream.write_available < CUSHION * BLOCK:
                        stream.write(silence)   # top up: an empty running stream is static
                    continue
                if audio is None:
                    break
                index += 1
                self.progress = f"{index}/{len(sentences)}"
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                self._refresh()
                for off in range(0, len(audio), BLOCK):
                    self.playing.wait()
                    if self.stopped.is_set():
                        break
                    stream.write(audio[off:off + BLOCK])
                    played += min(BLOCK, len(audio) - off)
                if self.stopped.is_set():
                    stream.abort()   # drop what is already buffered, don't drain it
                    break
        finally:
            stream.stop()    # drains the tail; abort() above is the barge-in path
            stream.close()
        # idle: seconds since the last utterance ended. Without it a slow ttfa is
        # unattributable — a long gap means the model's pages were compressed out of RAM
        # and the first request paid to fault them back in.
        idle = time.perf_counter() - self.last_finished if self.last_finished else 0.0
        # ttfa stays None when nothing was ever played — a failed generate, or a stop that
        # landed before the first block. Formatting None raises, which used to turn a
        # handled failure into a second, confusing error.
        print(
            f"{len(text):5d} chars | {len(sentences):2d} chunks | "
            f"ttfa {f'{ttfa:.3f}s' if ttfa is not None else '   n/a'} | "
            f"total {time.perf_counter() - t0:.2f}s | played {played / SAMPLE_RATE:.1f}s | "
            f"idle {idle:.0f}s"
        )
        self.last_finished = time.perf_counter()


def serve_socket() -> None:
    SOCK_PATH.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(SOCK_PATH))
    server.listen(1)
    print(f"listening on {SOCK_PATH} — read the selection with:  nc -U {SOCK_PATH} < /dev/null")
    while True:
        conn, _ = server.accept()
        with conn:
            data = b""
            while chunk := conn.recv(4096):
                data += chunk
        # Empty payload means "read whatever is selected right now". Sending text still
        # works, so the pbpaste trigger is unchanged.
        text = data.decode("utf-8", "replace").strip() or copy_selection()
        if text:
            PLAYER.submit(text)
        else:
            print("nothing selected")


def install_hotkey() -> list:
    """Own the ⌥⌘S hotkey directly, so no Shortcuts.app or Services wiring is needed.

    A global NSEvent monitor needs Accessibility permission — the same grant
    copy_selection() needs to post its ⌘C — so one prompt covers both. Without it the
    monitor is installed but never fires, hence the explicit trusted check: a silent
    dead hotkey is the worst possible failure here.
    """
    trusted = AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
    print(f"hotkey ⌥⌘S installed (accessibility {'granted' if trusted else 'NOT granted — hotkey is dead'})")

    # Compare against every modifier, not just ours, so ⌥⇧⌘S doesn't also fire.
    all_modifiers = (
        NSEventModifierFlagCommand | NSEventModifierFlagOption
        | NSEventModifierFlagShift | NSEventModifierFlagControl
    )

    def pressed(event) -> bool:
        return event.keyCode() == HOTKEY_KEYCODE and (event.modifierFlags() & all_modifiers) == HOTKEY_FLAGS

    def on_global(event) -> None:
        if pressed(event):
            threading.Thread(target=speak_selection, daemon=True).start()

    def on_local(event):
        if not pressed(event):
            return event
        threading.Thread(target=speak_selection, daemon=True).start()
        return None   # swallow it, so the panel doesn't also see the keystroke

    # A global monitor only sees events delivered to *other* apps. Whenever we are the
    # active app — panel up, or the moment after esc before focus has finished returning —
    # it is blind, and the hotkey looked permanently dead until the next app switch. The
    # local monitor covers exactly that window.
    return [
        NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskKeyDown, on_global),
        NSEvent.addLocalMonitorForEventsMatchingMask_handler_(NSEventMaskKeyDown, on_local),
    ]


def speak_selection() -> None:
    # The hotkey's own ⌘ and ⌥ are still physically down at this point and would corrupt
    # the synthetic ⌘C. Wait for the release rather than racing it.
    time.sleep(0.2)
    if text := copy_selection():
        PLAYER.submit(text)
    else:
        print("nothing selected")


def single_instance() -> "object":
    """Refuse to start if another copy is already running; return the held lock.

    Double-clicking the app twice used to start a second daemon that unlinked the
    first one's socket and stole every trigger, leaving two models resident and one
    orphaned process. An flock is released by the kernel however the process dies, so
    unlike a pidfile it can't go stale.
    """
    lock = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Exit 0, not 1: already-running is not a failure, and the LaunchAgent's
        # KeepAlive={SuccessfulExit:false} would otherwise respawn this every 10s.
        print("kokoro-speak is already running")
        raise SystemExit(0)
    return lock


def main() -> None:
    global PLAYER, _LOCK
    _LOCK = single_instance()   # must outlive main(): closing the file drops the lock
    t0 = time.perf_counter()
    model = load_model(REPO)
    # Burn the one-time Metal kernel-compile tax before accepting requests.
    # The text is FIRST_CHUNK_CHARS-sized, but not for the reason first assumed: kernel
    # specialisation per tensor shape turned out not to be what makes the first utterance
    # slow (that is page eviction — see docs/streaming.md). Kept because it costs one
    # sentence against a model load that dwarfs it.
    # Real words, not padding: "x" * 90 phonemises to ninety separate letter names and
    # takes minutes.
    for _ in model.generate(WARMUP_TEXT, voice=VOICE, lang_code="a"):
        pass
    print(f"loaded {REPO} in {time.perf_counter() - t0:.1f}s")

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon
    ui = Ui()
    global _DELEGATE   # setDelegate_ doesn't retain, so a local would be collected
    _DELEGATE = AppDelegate.alloc().init()
    _DELEGATE.ui = ui
    app.setDelegate_(_DELEGATE)
    PLAYER = Player(model, ui)
    global _HOTKEY
    _HOTKEY = install_hotkey()   # the monitors stop firing if these are garbage collected
    threading.Thread(target=serve_socket, daemon=True).start()
    try:
        AppHelper.runEventLoop()
    finally:
        SOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
