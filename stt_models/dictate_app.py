"""Push-to-talk dictation: one STT model held in memory, mic straight to the clipboard.

⌥⌘R starts recording, ⌥⌘R again stops it, esc throws it away — the clip is transcribed in
one pass and pasted into whatever app you were in. A floating panel shows a live meter.

Menu bar → Model switches engines, and exactly one is ever loaded (two resident peak at
2425MB on an 8GB machine — see MODELS). Qwen3-ASR transcribes when you stop; Nemotron
streams and types as you speak. Menu bar → Vocabulary… fixes names the model mishears.

Start:      .venv/bin/python -u dictate_app.py
Self-check: .venv/bin/python dictate_app.py --check
File check: .venv/bin/python dictate_app.py --file sample_for_tts.wav
Live check: .venv/bin/python dictate_app.py --live-file captures/<clip>.wav
Models:     .venv/bin/python dictate_app.py --models [--swap]
Mic test:   .venv/bin/python dictate_app.py --once 5
Key debug:  .venv/bin/python dictate_app.py --keys

Every utterance is written to captures/ with its transcript appended to captures.jsonl —
you know what you said, so daily use accumulates the labelled set needed for a real WER
number, which docs/RESULTS.md currently lacks. Everything tunable is in the CONFIG block.
"""

import difflib
import fcntl
import gc
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
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
    NSScrollView,
    NSStatusBar,
    NSTextField,
    NSTextView,
    NSVariableStatusItemLength,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
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
    CGEventKeyboardSetUnicodeString,
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
VOCAB_FILE = HERE / "vocab.txt"      # edited from the menu bar: Vocabulary…
VOCAB_CUTOFF = 0.75                  # fuzzy threshold; see apply_vocab for the measurements
DICT_FILE = Path("/usr/share/dict/words")
_DICT = None                         # the word list, loaded once on first use
VOCAB_EXAMPLE = HERE / "vocab.example.txt"   # tracked seed list; vocab.txt is not
SAMPLE_RATE = 16000          # what Qwen3-ASR and Whisper both expect; avoids a resample
# Pin the language. Qwen3-ASR is multilingual and auto-detects, and given near-silence it
# does not return nothing — it confabulates in whatever language it guessed. Two recordings
# of an empty room produced fluent Devanagari and Chinese, which auto-paste would have put
# straight into whatever document was open. Set to None if you want auto-detection back.
LANGUAGE = "en"
MAX_SECONDS = 600            # auto-stop, so a forgotten hotkey can't record all afternoon
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

KEY_V, KEY_ESC, KEY_DELETE = 9, 53, 51
# ---- models ----------------------------------------------------------------------------
# Menu bar → Model. Exactly one is ever loaded: measured, two resident peak at 2425MB, worse
# than the 1.7B alone (2314MB) which already left this 8GB machine under 1GB free. Switching
# frees the old model first, which returns all of it (peak drops to 0), so a swap never costs
# more than the heavier of the two — see docs/RESULTS.md.
#
# "batch" transcribes once when you stop; "live" types as you speak. Adding a model is one
# line here, as long as mlx_audio can load it and the mode matches how it decodes.
MODELS = {
    "Qwen3-ASR 0.6B": ("mlx-community/Qwen3-ASR-0.6B-4bit", "batch"),
    "Qwen3-ASR 1.7B": ("mlx-community/Qwen3-ASR-1.7B-4bit", "batch"),
    "Nemotron 0.6B — live": ("mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit", "live"),
}
DEFAULT_MODEL = "Qwen3-ASR 0.6B"    # fastest measured, and the one with no tail hallucination
MODEL_FILE = HERE / "model.txt"     # remembers the choice; same idea as vocab.txt
LIVE_REPO = MODELS["Nemotron 0.6B — live"][0]
LIVE_LANGUAGE = "en-US"      # a prompt_dictionary key, not a plain code — "en" also works
# [left, right] look-ahead, one of the four the model was trained with. Settled by measurement
# and then by the one comparison with confirmed ground truth, which reversed the measurement:
#
# right=13 (the model's default) reproduces the non-streaming generate() exactly on 6 of 6
# clips, where right=3 differs on 5. That is *fidelity to the model's own offline decode*, not
# accuracy — and on the one clip whose wording was confirmed, generate() itself was wrong:
# right=13 heard "Oden Goder" where right=3 heard the correct "ordinary". The other difference
# on that clip ("male spectrogram" vs "Malspectrogram") the vocabulary now resolves either way.
#
# So right=3: worse fidelity, better on the only ground-truthed word, and 0.32s of on-screen
# lag instead of 1.12s. It costs twice the compute (RTF 0.20 vs 0.10, still ~5x headroom) and
# can drop a trailing word — "okay" went missing on one clip. right=0 is slower than both
# (RTF 0.68) because one-frame chunks pay the per-chunk overhead 14x as often.
#
# One confirmed word is thin evidence. Transcribe a clip whose sentence is written down before
# treating this as settled — docs/RESULTS.md has the full sweep and the correction.
LIVE_ACS = [56, 3]
LIVE_TYPE_CHUNK = 40         # unicode chars per synthetic key event
# ---------------------------------------------------------------------------------------
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

# Filled in by main(). Declared here so importing this module (a --check run, a test) cannot
# raise NameError from a callback that fires before main has assigned them.
ENGINE = None        # owns the loaded model; the hotkey's target
RECORDER = None      # what esc cancels — the Engine, which forwards to the live session
VOCAB_EDITOR = None
_LOCK = _DELEGATE = _HOTKEY = None


def load_choice() -> str:
    """The model chosen last time, or the default if the file is missing or stale."""
    try:
        name = MODEL_FILE.read_text().strip()
    except OSError:
        return DEFAULT_MODEL
    return name if name in MODELS else DEFAULT_MODEL


def save_choice(name: str) -> None:
    """Same atomic write as the vocabulary: a half-written name would fall back to default."""
    tmp = MODEL_FILE.with_suffix(".txt.tmp")
    tmp.write_text(name + "\n")
    os.replace(tmp, MODEL_FILE)


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


def load_vocab(path: Path = VOCAB_FILE) -> tuple[list[str], dict[str, str]]:
    """Read vocab.txt into (terms, alias -> term). Read on every transcription, not cached,
    so editing the list takes effect on the next thing you say.

    A line is either a term (`Nemotron`) or a term with the manglings you have actually seen
    (`Qwen3: gwen, qn`). An alias may contain spaces (`Mel spectrogram: male spectrogram`),
    which is the only way to fix a mishearing split across two words. `#` comments and blank
    lines are ignored.
    """
    terms: list[str] = []
    aliases: dict[str, str] = {}
    if not path.exists():
        return terms, aliases
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        term, _, rest = line.partition(":")
        term = term.strip()
        if not term:
            continue
        terms.append(term)
        for alias in rest.split(","):
            alias = alias.strip().lower()
            if alias:
                aliases[alias] = term
    return terms, aliases


def _dictionary() -> set[str]:
    """macOS's own word list, used to refuse to rewrite words that are already real English.
    Without it fuzzy matching turns "when" into "Qwen3" and "mix" into "MLX" — both measured
    at ratio 0.67, which is under the 0.75 cutoff but not by a comfortable margin."""
    global _DICT
    if _DICT is None:
        try:
            _DICT = {w.strip().lower() for w in DICT_FILE.read_text(errors="ignore").splitlines()}
        except OSError:
            _DICT = set()   # no word list -> fuzzy stays off, see apply_vocab
    return _DICT


def apply_vocab(text: str, terms: list[str], aliases: dict[str, str]) -> str:
    """Fix vocabulary the model misheard. Three passes, in order of confidence.

    1. Multi-word aliases ("male spectrogram"), replaced as a phrase. The only pass that can
       fix a mishearing split across tokens — and every real example so far has also been
       built from ordinary English words ("male", "queen", "item"), which passes 2 and 3
       cannot touch: an alias is matched per word, and the dictionary guard protects words
       that are already English.
    2. Single-word aliases — likewise the only way to fix a mangling that is itself a real
       word ("gwen" for Qwen3), since pass 3 deliberately leaves those alone.
    3. Fuzzy, cutoff 0.75, skipping anything in the system dictionary. Measured on this
       app's own captures: nemotone->Nemotron 0.88, nimoton->Nemotron 0.80,
       sudesh->Siddesh 0.77, while the dangerous neighbours sit at 0.67.
    """
    phrases = {alias: term for alias, term in aliases.items() if " " in alias}
    if phrases:
        # Longest alias first, so one containing a shorter one still wins. \s+ between words
        # because a transcript can split them over a double space or a newline.
        pattern = "|".join(
            r"\b" + r"\s+".join(re.escape(w) for w in alias.split()) + r"\b"
            for alias in sorted(phrases, key=len, reverse=True)
        )
        text = re.sub(
            pattern,
            lambda m: phrases[" ".join(m.group(0).lower().split())],
            text, flags=re.IGNORECASE,
        )

    # Multi-word terms are kept out of the word passes: they can never equal a single token,
    # and letting difflib compare against them only invents near-misses.
    word_terms = {t.lower(): t for t in terms if " " not in t}
    word_aliases = {a: t for a, t in aliases.items() if " " not in a}
    # Fuzzy matches against the aliases too, not only the terms. A mangling tends to reappear
    # in variants — the same sentence produced "male spectrogram" at one look-ahead setting
    # and "Malspectrogram" at another — and each variant is within a character or two of a
    # mangling already listed (malspectrogram->melspectrogram scores 0.93). Listing one form
    # therefore covers its neighbours, instead of needing a line per spelling.
    fuzzy = {**word_terms, **word_aliases}
    fuzzy_ok = bool(_dictionary()) if fuzzy else False

    def fix(match: re.Match) -> str:
        word = match.group(0)
        key = word.lower()
        if key in word_aliases:
            return word_aliases[key]
        if key in word_terms:
            return word_terms[key]         # right word, wrong capitalisation
        if not fuzzy_ok or len(key) < 4 or key in _dictionary():
            return word
        hit = difflib.get_close_matches(key, list(fuzzy), n=1, cutoff=VOCAB_CUTOFF)
        return fuzzy[hit[0]] if hit else word

    return re.sub(r"[A-Za-z0-9'\-]+", fix, text)


def transcribe(model, wav_path: Path) -> str:
    """One clip in, text out. Same call path as run_qwen3_asr_mlx.py, so the benchmark
    numbers in docs/RESULTS.md describe this app's latency directly.

    The vocabulary is applied twice over: as Qwen3-ASR's `system_prompt`, which biases
    decoding itself, and as a text fix afterwards for what that missed. Measured on
    captures/20260817-111314.wav, biasing alone turned "NemoTone" into "Nemotron" for
    +0.01s; on the next clip it did not, which is why the second pass exists.
    """
    terms, aliases = load_vocab()
    result = generate_transcription(
        model=model, audio=str(wav_path), language=LANGUAGE,
        output_path=str(CAPTURE_DIR / "last_transcript"), format="txt", verbose=False,
        # Dropped by generate_transcription for models whose generate() has no such
        # parameter (whisper), so this stays valid if REPO changes.
        system_prompt=", ".join(terms) if terms else None,
    )
    if not hasattr(result, "text"):
        # generate_transcription also writes output_path, so it can plausibly return None;
        # str(None) would paste the word "None" into your document.
        raise RuntimeError(f"no .text on {type(result).__name__}: {result!r}")
    return apply_vocab(result.text.strip(), terms, aliases)


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


def type_text(text: str) -> None:
    """Type a string into whatever field has focus, without touching the clipboard.

    A keyboard event carrying a unicode string ignores its keyCode, so this types characters
    the physical layout may not even have. Split into LIVE_TYPE_CHUNK pieces because a single
    event with a very long string is dropped by some apps.
    """
    for i in range(0, len(text), LIVE_TYPE_CHUNK):
        piece = text[i:i + LIVE_TYPE_CHUNK]
        for keydown in (True, False):
            event = CGEventCreateKeyboardEvent(None, 0, keydown)
            CGEventKeyboardSetUnicodeString(event, len(piece), piece)
            CGEventPost(kCGHIDEventTap, event)


def press_backspace(count: int) -> None:
    """Delete `count` characters left of the cursor."""
    for _ in range(count):
        for keydown in (True, False):
            CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, KEY_DELETE, keydown))


def diff_update(typed: str, new: str) -> tuple[int, str]:
    """(backspaces, text to type) to turn what is on screen into `new`.

    A streaming model yields cumulative text and *revises* it — "what more" can become "what
    models" two chunks later — so the typed field has to be edited, not appended to. Deleting
    back to the common prefix is the smallest edit that is always correct, and costs nothing
    in the common case where the new text merely extends the old (0 backspaces).
    """
    keep = len(os.path.commonprefix([typed, new]))
    return len(typed) - keep, new[keep:]


def vocab_stable(text: str, terms: list[str], aliases: dict[str, str]) -> str:
    """Apply the vocabulary to every word except the last one.

    The last word of a live transcript is still being formed, and fuzzy-matching a fragment
    is how "desh" becomes "Siddesh" — visible in this repo's own captures corpus. Words with
    a space after them are settled, so those are safe to fix; the tail is left raw until the
    next update, or until the session ends and the whole string is fixed.
    """
    cut = text.rfind(" ") + 1
    return apply_vocab(text[:cut], terms, aliases) + text[cut:] if cut else text


def log_capture(wav_path: Path, text: str, seconds: float, latency: float,
                repo: str = REPO, mode: str = "batch") -> None:
    """Append the labelled pair. This is the WER corpus docs/RESULTS.md says is missing:
    you know what you actually said, so every dictation is a free ground-truth sample.

    `mode` matters when reading the numbers: a batch latency is the whole transcription, a
    live one is only the wait after you stop, so their rtf columns are not comparable.
    """
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": repo, "mode": mode, "wav": wav_path.name, "text": text,
        "audio_seconds": round(seconds, 3), "latency_s": round(latency, 3),
        "rtf": round(latency / seconds, 4) if seconds else None,
    }
    with CAPTURE_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")


class MenuTarget(NSObject):
    def quit_(self, sender) -> None:
        NSApplication.sharedApplication().terminate_(None)

    def vocab_(self, sender) -> None:
        VOCAB_EDITOR.show()

    def model_(self, sender) -> None:
        # On a thread: the swap takes seconds, and the menu is on the run loop.
        threading.Thread(target=ENGINE.switch, args=(str(sender.title()),), daemon=True).start()


class VocabEditor(NSObject):
    """The Vocabulary… window: type the words you want recognised, one per line.

    Saves on every keystroke, so there is no Save button and no unsaved state to lose —
    and load_vocab() re-reads the file per transcription, so a word typed here is in
    effect for the very next thing you say, with no restart.

    ponytail: a plain text view, not a table of term/alias columns. The file format is two
    fields on a line; a table would be more code for the same edit.
    """

    # Class attributes rather than an init override: subclassing NSObject means super().init()
    # needs objc.super, and there is nothing here worth importing it for.
    window = None
    text = None

    def _build(self) -> None:
        self.window = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 420, 320),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable,
            NSBackingStoreBuffered, False,
        )
        self.window.setTitle_("Dictate Vocabulary")
        self.window.setReleasedWhenClosed_(False)   # reopened from the menu, so keep it alive
        self.window.center()

        hint = NSTextField.labelWithString_(
            "One term per line. Add manglings you've seen after a colon —\n"
            "two-word ones work too:   Mel spectrogram: male spectrogram"
        )
        hint.setFrame_(NSMakeRect(14, 262, 392, 44))
        hint.setFont_(NSFont.systemFontOfSize_(11))
        hint.setTextColor_(NSColor.secondaryLabelColor())
        self.window.contentView().addSubview_(hint)

        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(14, 14, 392, 240))
        scroll.setHasVerticalScroller_(True)
        scroll.setBorderType_(2)   # NSBezelBorder
        self.text = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 392, 240))
        self.text.setFont_(NSFont.monospacedSystemFontOfSize_weight_(12, 0))
        self.text.setAutomaticQuoteSubstitutionEnabled_(False)
        # Autocorrect in a list of words the spell checker has never seen would fight you.
        self.text.setAutomaticSpellingCorrectionEnabled_(False)
        self.text.setDelegate_(self)
        scroll.setDocumentView_(self.text)
        self.window.contentView().addSubview_(scroll)

    def show(self) -> None:
        if self.window is None:
            self._build()
        seed = VOCAB_FILE if VOCAB_FILE.exists() else VOCAB_EXAMPLE
        existing = seed.read_text() if seed.exists() else ""
        self.text.setString_(existing)
        NSApplication.sharedApplication().unhide_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    def textDidChange_(self, notification) -> None:
        """Save on every keystroke — no Save button, nothing to forget.

        Written to a sibling temp file and renamed, never in place: os.replace is atomic, so
        a crash or a quit mid-save leaves the previous list intact instead of a file truncated
        halfway through a word. This is a list you typed by hand; losing it to a half-write
        would be worse than any amount of saved code.
        """
        tmp = VOCAB_FILE.with_suffix(".txt.tmp")
        tmp.write_text(str(self.text.string()))
        os.replace(tmp, VOCAB_FILE)


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

        # Model picker. A submenu of checkmarks rather than a text field: the choices are a
        # fixed list, and a typo in a typed repo name would only surface as a failed download.
        models = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Model", "", "")
        submenu = NSMenu.alloc().init()
        self.model_items = {}
        for name in MODELS:
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(name, "model:", "")
            item.setTarget_(self.menu_target)
            submenu.addItem_(item)
            self.model_items[name] = item
        models.setSubmenu_(submenu)
        menu.addItem_(models)
        menu.addItem_(NSMenuItem.separatorItem())

        for title, action, key in (("Vocabulary…", "vocab:", ""), ("Quit Dictate", "quit:", "q")):
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
            item.setTarget_(self.menu_target)
            menu.addItem_(item)
        self.status_item.setMenu_(menu)

    def mark_model(self, name: str) -> None:
        """Tick the loaded model. Run on the main thread — Engine.switch calls it via callAfter,
        because it finishes on a worker and AppKit is not thread-safe."""
        for title, item in getattr(self, "model_items", {}).items():
            item.setState_(1 if title == name else 0)   # NSControlStateValueOn / Off

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
        # Tick whichever model main() loaded, so the menu agrees with reality from the start.
        if ENGINE is not None:
            self.ui.mark_model(ENGINE.name)


class Recorder:
    """Toggled by the hotkey: one recording at a time, transcribed off the UI thread."""

    def __init__(self, model, ui: Ui, repo: str = REPO) -> None:
        self.model, self.ui, self.repo = model, ui, repo
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

    def close(self) -> None:
        """Stop and release. Batch transcription happens on a worker too, but it holds only the
        clip once started, so cancelling before a swap is enough."""
        self.cancel()

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
            log_capture(wav_path, text, seconds, latency, self.repo)
            # No panel flash of the transcript on the way out: it would mean either delaying
            # the paste by HOLD_SECONDS or re-activating this app immediately after posting
            # ⌘V, and a keystroke the target app has not processed yet does not survive that.
            # The text appearing in your document is the feedback.
            paste(text, auto=AUTO_PASTE and self._restore_focus())
        finally:
            self.ui.icon(ICON_IDLE)
            self.ui.hide()


class LiveSession:
    """Live mode: Nemotron streams, and the text is typed into the focused field as you talk.

    Same hotkey and the same one-at-a-time contract as Recorder, so install_hotkey takes
    either. Deliberately different in one way: **no panel**. The panel activates this app to
    get itself onto your Space (docs/RESULTS.md bug #11), and activating steals the focus that
    live typing needs to keep. The menu bar icon is the only indicator here.

    The clip and its transcript are saved like batch mode's, so the two engines can be
    compared on identical audio — the row carries mode:"live" because its latency means
    something different (see _run).
    """

    def __init__(self, model, ui: Ui, repo: str = LIVE_REPO) -> None:
        self.model, self.ui, self.repo = model, ui, repo
        self.lock = threading.Lock()
        self.stream = None
        self.audio: queue.Queue = queue.Queue()
        self.running = False
        self.typed = ""
        self.blocks: list[np.ndarray] = []   # kept for captures/, not for decoding
        self.stopped = 0.0
        self.worker = None

    def toggle(self) -> None:
        with self.lock:
            if self.stream is None:
                self._start()
                return
            self.running = False          # the worker drains, finalises, then types the fix
            self.stopped = time.perf_counter()
            stream, self.stream = self.stream, None
        stream.stop()
        stream.close()

    def cancel(self) -> None:
        """esc, and what Engine.switch calls before swapping. Must be a no-op when idle: this
        used to be a bare toggle(), which *started* a recording on an idle session — and that
        worker then held the model, so the next swap could not free it (799MB leaked)."""
        if self.stream is None:
            return
        self.toggle()

    def close(self) -> None:
        """Stop and release the model. The worker owns a generator holding self.model, so the
        swap has to wait for it to exit or the weights stay resident behind the new model."""
        self.cancel()
        worker = self.worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)
            if worker.is_alive():
                print("live worker did not exit — its model may stay resident")
        self.worker = None

    def _start(self) -> None:
        self.typed = ""
        self.running = True
        self.audio = queue.Queue()
        self.blocks = []
        self.stopped = 0.0
        self.ui.icon(ICON_RECORDING)
        self.stream = open_input_stream(self._on_block)
        self.stream.start()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        print("live: speak — text appears as you go")

    def _on_block(self, block: np.ndarray) -> None:
        """PortAudio's realtime thread: hand the block to the decoder and keep a copy. Both are
        cheap; anything slow here shows up as dropped input rather than as an error."""
        mono = block[:, 0].copy()
        self.audio.put(mono)
        self.blocks.append(mono)

    def _chunks(self):
        """Mic blocks -> encoder chunks, keeping every cache alive across the whole session.

        This is the model's own O(n) streaming path: the mel frontend emits only frames whose
        STFT window is complete, and the conformer state carries the attention/conv caches, so
        nothing is recomputed as the utterance grows.
        """
        # Imported here, not at module scope: these are private-ish submodules, and a batch-mode
        # launch should neither pay for them nor break if a mlx_audio update moves them.
        import mlx.core as mx
        from mlx_audio.stt.models.nemotron_asr.audio import StreamingLogMelSpectrogram
        from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState

        mel = StreamingLogMelSpectrogram(self.model.preprocessor_config)
        state = ConformerStreamingState(self.model.encoder, att_context_size=LIVE_ACS)
        while self.running:
            try:
                block = self.audio.get(timeout=0.2)
            except queue.Empty:
                continue
            frames = mel.push(mx.array(block))
            if frames.shape[1]:
                for encoded in state.push(frames):
                    yield self.model.apply_prompt(encoded, LIVE_LANGUAGE)
        # Flush: the tail of the audio is still inside the frontend's lookahead window.
        frames = mel.push(mx.array(np.zeros(0, dtype="float32")), final=True)
        for encoded in state.push(frames, final=True):
            yield self.model.apply_prompt(encoded, LIVE_LANGUAGE)

    def _run(self) -> None:
        terms, aliases = load_vocab()
        text = ""
        try:
            # _decode_prompted_chunks keeps the RNN-T decoder state across chunks and yields
            # the cumulative transcript. ponytail: it is private (leading underscore). The
            # public stream_generate() only accepts a finished array, which cannot be a live
            # mic; pin mlx_audio if an upgrade renames this.
            for result in self.model._decode_prompted_chunks(self._chunks()):
                text = vocab_stable(result.text, terms, aliases)
                self._emit(text)
        except Exception as exc:                       # a dead worker must not look like silence
            print(f"live transcription failed: {exc!r}")
        finally:
            if text:
                self._emit(apply_vocab(text, terms, aliases))   # fix the last word too
            # Print the transcript, not just its length: the field it was typed into is not a
            # record you can review later, and a live session used to leave nothing else.
            tail = time.perf_counter() - self.stopped if self.stopped else 0.0
            print(f"live: done in {tail:.2f}s after stop — {len(self.typed)} chars: "
                  f"{self.typed!r}")
            self._save(tail)
            self.ui.icon(ICON_IDLE)

    def _save(self, tail: float) -> None:
        """Write the clip and its transcript, so live rows sit beside batch rows on the same
        footing. `tail` is the wait *after* you stop, not the time to transcribe the whole
        clip: live mode has already done most of the work by then, which is the point of it.
        """
        blocks, self.blocks = self.blocks, []
        if not blocks or not self.typed:
            return
        audio = np.concatenate(blocks)
        seconds = len(audio) / SAMPLE_RATE
        CAPTURE_DIR.mkdir(exist_ok=True)
        wav_path = CAPTURE_DIR / f"live-{time.strftime('%Y%m%d-%H%M%S')}.wav"
        sf.write(wav_path, audio, SAMPLE_RATE)
        log_capture(wav_path, self.typed, seconds, tail, self.repo, mode="live")

    def _emit(self, text: str) -> None:
        back, addition = diff_update(self.typed, text)
        if not back and not addition:
            return
        press_backspace(back)
        type_text(addition)
        self.typed = text


class Engine:
    """Owns the one loaded model and the session driving it.

    The hotkey and esc target *this* object rather than the session, so switching models never
    reinstalls the event tap — identity stays stable while the model underneath is replaced.
    """

    def __init__(self, ui: Ui) -> None:
        self.ui = ui
        self.lock = threading.Lock()
        self.name = None
        self.session = None
        self.loading = False

    def switch(self, name: str) -> None:
        """Free the current model, then load `name`. Call from a worker thread: loading takes
        1.7–3.7s and this must not run on the run loop, or the menu freezes mid-swap."""
        if name not in MODELS:
            print(f"unknown model {name!r}")
            return
        with self.lock:
            if self.loading or name == self.name:
                return
            self.loading = True
        try:
            if self.session is not None:
                # close(), not cancel(): this also waits for a live worker to exit, and that
                # worker holds the model being replaced.
                self.session.close()
            repo, mode = MODELS[name]
            self.ui.icon(ICON_BUSY)
            self.ui.status(f"⏳ loading {name}…")
            # Drop the old model *before* loading the new one. Measured: this returns all of
            # it, so the swap peaks at the heavier model rather than at their sum.
            self.session = None
            model, self.name = None, None
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            model = load_and_warm(repo, mode)
            self.session = (LiveSession if mode == "live" else Recorder)(model, self.ui, repo)
            self.name = name
            save_choice(name)
            print(f"model: {name} ({mode}, peak {mx.get_peak_memory()/1e6:.0f}MB)")
        except Exception as exc:
            # A failed switch must not leave a dead hotkey with no explanation.
            print(f"could not load {name}: {exc!r}")
            self.ui.status(f"could not load {name}")
        finally:
            self.loading = False
            self.ui.icon(ICON_IDLE)
            AppHelper.callAfter(self.ui.mark_model, self.name)

    def toggle(self) -> None:
        if self.session is None:
            print("no model loaded" + (" yet — still loading" if self.loading else ""))
            return
        self.session.toggle()

    def cancel(self) -> None:
        if self.session is not None:
            self.session.cancel()


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


def install_hotkey(recorder) -> object:
    """Own the hotkey directly, so no Shortcuts.app or Services wiring is needed.

    `recorder` is a Recorder or a LiveSession — both toggle() on the hotkey and cancel() on
    esc, which is all this needs to know about either.

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


def load_and_warm(repo: str = REPO, mode: str = "batch"):
    """Load one model and burn its one-time Metal kernel-compile tax."""
    t0 = time.perf_counter()
    model = load_model(repo)
    # Without this the first dictation after a load pays ~4x the latency of every later one —
    # the exact confound that invalidated the first benchmark run (docs/RESULTS.md bug #1).
    if WARMUP_AUDIO.exists():
        CAPTURE_DIR.mkdir(exist_ok=True)
        if mode == "live":
            # Warm the streaming path specifically: it compiles different kernels from
            # generate(), so warming the wrong one leaves the first utterance slow.
            for _ in model.stream_generate(str(WARMUP_AUDIO), language=LIVE_LANGUAGE,
                                           att_context_size=LIVE_ACS):
                pass
        else:
            transcribe(model, WARMUP_AUDIO)
    print(f"loaded {repo} ({mode}) in {time.perf_counter() - t0:.1f}s")
    return model


def cli_models() -> None:
    """List the models, mark the saved choice, and prove a swap frees the old one.

    No mic, no permissions, no GUI — this is the check that the picker's real risk (two models
    resident at once) does not happen.
    """
    choice = load_choice()
    for name, (repo, mode) in MODELS.items():
        print(f"{'*' if name == choice else ' '} {name:24} {mode:5} {repo}")
    print(f"\nsaved choice: {choice}  ({MODEL_FILE if MODEL_FILE.exists() else 'default, no file'})")

    if "--swap" not in sys.argv:
        print("pass --swap to load each model in turn and print the peak memory")
        return
    for name, (repo, mode) in MODELS.items():
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        model = load_and_warm(repo, mode)
        peak = mx.get_peak_memory() / 1e6
        del model
        gc.collect()
        mx.clear_cache()
        freed = mx.get_peak_memory() / 1e6
        mx.reset_peak_memory()
        print(f"  {name:24} peak {peak:7.0f}MB   after free, peak resets to "
              f"{mx.get_peak_memory()/1e6:.0f}MB (was {freed:.0f})")
    print("one model at a time: each peak above is a single model's, never a sum")


def cli_live(path: str) -> None:
    """Stream a file through the live pipeline and print each update instead of typing it.

    The whole live path except the mic and the keystrokes, so it needs no permissions and
    cannot type into a window by accident. `->` marks a revision (the model changed its mind
    about text already on screen), which is the case diff_update exists for.
    """
    import mlx.core as mx
    from mlx_audio.stt.models.nemotron_asr.audio import StreamingLogMelSpectrogram
    from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState

    model = load_model(LIVE_REPO)
    audio, sr = sf.read(path, dtype="float32")
    assert sr == SAMPLE_RATE, f"{path} is {sr}Hz, expected {SAMPLE_RATE}"
    terms, aliases = load_vocab()

    mel = StreamingLogMelSpectrogram(model.preprocessor_config)
    state = ConformerStreamingState(model.encoder, att_context_size=LIVE_ACS)

    def chunks():
        for i in range(0, len(audio), BLOCK_FRAMES):
            frames = mel.push(mx.array(audio[i:i + BLOCK_FRAMES]))
            if frames.shape[1]:
                for encoded in state.push(frames):
                    yield model.apply_prompt(encoded, LIVE_LANGUAGE)
        frames = mel.push(mx.array(np.zeros(0, dtype="float32")), final=True)
        for encoded in state.push(frames, final=True):
            yield model.apply_prompt(encoded, LIVE_LANGUAGE)

    t0 = time.perf_counter()
    typed, updates, revisions, first = "", 0, 0, None
    for result in model._decode_prompted_chunks(chunks()):
        text = vocab_stable(result.text, terms, aliases)
        back, addition = diff_update(typed, text)
        if not back and not addition:
            continue
        if first is None:
            first = time.perf_counter() - t0
        updates += 1
        revisions += 1 if back else 0
        print(f"{time.perf_counter() - t0:6.2f}s  {'->' if back else '+ '} {addition!r}"
              + (f"  (after {back} backspaces)" if back else ""))
        typed = text
    final = apply_vocab(typed, terms, aliases)
    seconds, compute = len(audio) / sr, time.perf_counter() - t0
    print(f"\n{seconds:.1f}s audio | first text at {first:.2f}s | {updates} updates, "
          f"{revisions} revisions | compute {compute:.2f}s (RTF {compute / seconds:.3f})")
    print(f"final: {final!r}")
    assert final, "empty transcript"


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

    # Vocabulary. The pairs below are real transcripts from captures/, not invented.
    terms, aliases = load_vocab(VOCAB_EXAMPLE)
    assert terms, f"no terms parsed from {VOCAB_EXAMPLE}"
    assert aliases.get("gwen") == "Qwen3", aliases
    fix = lambda s: apply_vocab(s, terms, aliases)                          # noqa: E731
    assert fix("streaming Nimoton model") == "streaming Nemotron model"     # fuzzy, 0.80
    assert fix("such as NemoTone Live") == "such as Nemotron Live"          # fuzzy, 0.88
    assert fix("gwen three point five") == "Qwen3 three point five"         # alias only
    # Words the fuzzy pass must leave alone: real English inside the cutoff's blast radius.
    for safe in ("when I mix the item", "it is solid", "a corridor"):
        assert fix(safe) == safe, fix(safe)
    assert fix("QWEN3 and mlx") == "Qwen3 and MLX", fix("QWEN3 and mlx")    # capitalisation

    # Multi-word aliases. Every one of these is a real mishearing from captures/, and none is
    # reachable by the word passes: "male", "queen" and "item" are all English.
    assert fix("like male spectrogram audio") == "like Mel spectrogram audio"
    assert fix("already done Queen three ASR") == "already done Qwen3 ASR"
    assert fix("Webb item across both") == "verbatim across both"
    assert fix("such as Nemo Tone Live") == "such as Nemotron Live"
    assert fix("MALE  SPECTROGRAM") == "Mel spectrogram"          # case and spacing
    assert fix("male\nspectrogram") == "Mel spectrogram"          # split over a line
    # A phrase must not fire on a substring of a longer word, or on only half of itself.
    assert fix("a male voice") == "a male voice"
    assert fix("the spectrogram") == "the spectrogram"
    assert fix("females spectrogram") == "females spectrogram"
    # Fuzzy reaches the aliases too, which covers a variant of a known mangling without a line
    # per spelling. Both forms the same sentence produced must land on the same term.
    assert fix("like Malspectrogram audio") == "like Mel spectrogram audio"
    assert fix("like male spectrogram audio") == "like Mel spectrogram audio"

    # Live typing. diff_update is what keeps the field correct when the model revises text it
    # already emitted, which every streaming update potentially does.
    assert diff_update("", "Hello") == (0, "Hello")                  # first text
    assert diff_update("What more", "What more models") == (0, " models")   # pure append
    assert diff_update("What more f", "What more models") == (1, "models")  # revision
    assert diff_update("abc", "abc") == (0, ""), "no-op must type nothing"
    assert diff_update("abc", "x") == (3, "x")                       # nothing in common
    # The tail word is left raw so a half-decoded fragment is never rewritten: the same word
    # must be left alone while it is still the last thing typed, and fixed once a space proves
    # it finished. This matters more since fuzzy started matching aliases too — "desh" now
    # reaches Siddesh through the "sudesh" alias (0.80), so a half-typed name would be
    # rewritten mid-word without this.
    assert vocab_stable("streaming nimoton", terms, aliases) == "streaming nimoton"
    assert vocab_stable("streaming nimoton ", terms, aliases) == "streaming Nemotron "
    assert vocab_stable("nimoton is", terms, aliases) == "Nemotron is"

    # Model picker. Every entry must be loadable and have a mode the app knows how to drive,
    # and an unknown or half-written saved choice must fall back rather than fail to start.
    assert DEFAULT_MODEL in MODELS
    for name, (repo, mode) in MODELS.items():
        assert mode in ("batch", "live"), (name, mode)
        assert "/" in repo, (name, repo)
    saved = MODEL_FILE.read_text() if MODEL_FILE.exists() else None
    try:
        MODEL_FILE.write_text("Qwen3-ASR 1.7B\n")
        assert load_choice() == "Qwen3-ASR 1.7B"
        MODEL_FILE.write_text("a model that was renamed\n")
        assert load_choice() == DEFAULT_MODEL, "a stale choice must fall back"
    finally:
        MODEL_FILE.write_text(saved) if saved is not None else MODEL_FILE.unlink(missing_ok=True)
    print(f"checks passed: quiet={quiet!r} speech={loud!r} silence-gate ok "
          f"| vocab {len(terms)} terms, {len(aliases)} aliases | diff+live ok")


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
    global _LOCK, _DELEGATE, _HOTKEY, RECORDER, VOCAB_EDITOR, ENGINE
    _LOCK = single_instance()   # must outlive main(): closing the file drops the lock
    choice = load_choice()
    repo, mode = MODELS[choice]
    model = load_and_warm(repo, mode)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon
    ui = Ui()
    _DELEGATE = AppDelegate.alloc().init()   # setDelegate_ doesn't retain
    _DELEGATE.ui = ui
    app.setDelegate_(_DELEGATE)
    # Module-globals because AppKit instantiates KeyView and MenuTarget itself, giving no way
    # to hand either a reference — same reason speak_app.py has a global PLAYER.
    ENGINE = Engine(ui)
    ENGINE.name = choice
    ENGINE.session = (LiveSession if mode == "live" else Recorder)(model, ui, repo)
    # Hand the model over and forget it here. A local reference would live in this frame for
    # as long as runEventLoop() runs, so the first model loaded could never be freed: the
    # session dropped it on a switch and this name kept it alive anyway. Measured cost of
    # getting this wrong — Nemotron reported a 1656MB peak instead of its own 942MB, because
    # Qwen3's ~714MB of weights were still resident behind the swap.
    del model
    RECORDER = ENGINE       # what esc cancels; the Engine forwards to whichever session
    VOCAB_EDITOR = VocabEditor.alloc().init()
    print(f"model: {choice} ({mode})")
    # The hotkey holds the Engine, not the session, so switching models never reinstalls it.
    _HOTKEY = install_hotkey(ENGINE)   # monitors die if garbage collected
    AppHelper.runEventLoop()


if __name__ == "__main__":
    if "--check" in sys.argv:
        cli_check()
    elif "--keys" in sys.argv:
        cli_keys()
    elif "--file" in sys.argv:
        cli_file(sys.argv[sys.argv.index("--file") + 1])
    elif "--live-file" in sys.argv:
        cli_live(sys.argv[sys.argv.index("--live-file") + 1])
    elif "--models" in sys.argv:
        cli_models()
    elif "--once" in sys.argv:
        cli_once(float(sys.argv[sys.argv.index("--once") + 1]))
    else:
        main()
