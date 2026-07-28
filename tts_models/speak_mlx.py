"""Resident Kokoro-MLX speaker: model loaded once, audio streamed straight to the
speakers as it is generated. No .wav files, no per-request model load.

Start:  .venv/bin/python speak_mlx.py
Speak:  pbpaste | nc -U /tmp/kokoro-speak.sock

Prints time-to-first-audio (TTFA) per utterance — the only latency that matters
here, since playback overlaps generation from the first chunk onward.
"""

import os
import queue
import re
import socket
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from mlx_audio.tts.utils import load_model

SOCK_PATH = Path("/tmp/kokoro-speak.sock")
SAMPLE_RATE = 24000
VOICE = "af_heart"
MAX_CHUNK_CHARS = 180
FIRST_CHUNK_CHARS = 90
# bf16 and 8bit measured within noise on this machine (see docs/findings.md);
# bf16 avoids quantization loss for free. Flip to "8bit" to halve resident memory.
REPO = "mlx-community/Kokoro-82M-bf16"


def split_sentences(text: str) -> list[str]:
    """Break text into units small enough that generating the first one is fast.

    TTFA is set entirely by the first unit, so this is what makes latency
    independent of how much text was selected.
    """
    out = []
    for part in re.split(r"(?<=[.!?])\s+", text.strip()):
        # ponytail: comma fallback for run-on sentences — a 400-char comma-spliced
        # sentence has no '.' to split on and would cost ~8s of TTFA on its own.
        while len(part) > MAX_CHUNK_CHARS and ", " in part[:MAX_CHUNK_CHARS]:
            cut = part[:MAX_CHUNK_CHARS].rindex(", ") + 1
            out.append(part[:cut])
            part = part[cut:].lstrip()
        out.append(part)
    out = [p for p in out if p]
    # Only the first chunk is on the critical path — every later one is generated while
    # earlier audio plays. Split it tighter than the rest to buy TTFA, at the cost of one
    # slightly early pause. Prosody of later chunks is left alone.
    if out and len(out[0]) > FIRST_CHUNK_CHARS and ", " in out[0][:FIRST_CHUNK_CHARS]:
        head = out[0]
        cut = head[:FIRST_CHUNK_CHARS].rindex(", ") + 1
        out[:1] = [head[:cut], head[cut:].lstrip()]
    return out


def speak(model, text: str) -> None:
    sentences = split_sentences(text)
    t0 = time.perf_counter()
    # One sentence of lookahead: the next one is generated while the current one
    # plays, so playback is gapless without generating everything up front.
    pending: queue.Queue = queue.Queue(maxsize=1)

    def generate_all() -> None:
        for sentence in sentences:
            chunks = [np.array(r.audio, copy=False) for r in model.generate(sentence, voice=VOICE, lang_code="a")]
            pending.put(np.concatenate(chunks))
        pending.put(None)

    threading.Thread(target=generate_all, daemon=True).start()

    ttfa = None
    samples = 0
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while (audio := pending.get()) is not None:
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            samples += len(audio)
            stream.write(audio)
    print(
        f"{len(text):5d} chars | {len(sentences):2d} chunks | ttfa {ttfa:.3f}s | "
        f"total {time.perf_counter() - t0:.2f}s | audio {samples / SAMPLE_RATE:.1f}s"
    )


def main() -> None:
    t0 = time.perf_counter()
    model = load_model(REPO)
    # First generate() pays a one-time Metal shader-compile tax; burn it before we
    # accept requests so the first real utterance isn't 10x slower than the rest.
    for _ in model.generate("warm up", voice=VOICE, lang_code="a"):
        pass
    print(f"loaded {REPO} in {time.perf_counter() - t0:.1f}s")

    SOCK_PATH.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(SOCK_PATH))
    server.listen(1)
    print(f"listening on {SOCK_PATH} — send text with:  pbpaste | nc -U {SOCK_PATH}")

    try:
        while True:
            conn, _ = server.accept()
            with conn:
                data = b""
                while chunk := conn.recv(4096):
                    data += chunk
            text = data.decode("utf-8", "replace").strip()
            # ponytail: serial — a new request waits for the current one to finish
            # speaking. Add a worker thread + sd.stop() if you want barge-in.
            if text:
                speak(model, text)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        SOCK_PATH.unlink(missing_ok=True)
        os.write(1, b"\nstopped\n")


if __name__ == "__main__":
    main()
