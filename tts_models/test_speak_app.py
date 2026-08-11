"""Check the pause/stop/barge-in state machine without a model, a speaker, or a GUI.

Stubs are swapped in as module attributes, so speak_app itself needs no seams for testing.

These drive real threads, so nothing here sleeps for a fixed interval waiting for work to
happen: wait_until/wait_stable poll for the condition instead. A fixed sleep is either
slower than it needs to be or fails on a loaded machine, and tuning one to sit between
those is how a suite becomes flaky. The one deliberate sleep left is in FakeStream.write,
standing in for realtime playback so that pausing is observable at all.
"""

import socket
import threading
import time

import numpy as np

import speak_app

TIMEOUT = 20.0        # generous: a slow machine should be slow, not failing


def wait_until(condition, what: str, timeout: float = TIMEOUT) -> None:
    """Poll until condition() is true, or fail saying what was being waited for."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def wait_stable(value, what: str, settle: float = 0.3, timeout: float = TIMEOUT):
    """Wait until value() stops changing, and return it.

    Used before asserting that something does *not* change: a block already being written
    when pause was pressed would otherwise fail the check on timing alone. Reaching a
    stable value is itself the evidence that writing stopped — 0.3s is 15 block-writes.
    """
    deadline = time.monotonic() + timeout
    previous = object()
    while time.monotonic() < deadline:
        current = value()
        if current == previous:
            return current
        previous = current
        time.sleep(settle)
    raise AssertionError(f"{what}: value never stopped changing within {timeout}s")


class FakeStream:
    latency = 0.6            # a Bluetooth-sized buffer, where the artifacts actually show up
    write_available = 14790  # always room, so the cushion logic is exercised

    def __init__(self, sink: list) -> None:
        self.sink = sink
        self.active = False
        self.peaks: list[float] = []   # per-block loudness, to tell silence from speech
        self.silent_blocks = 0

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False

    def write(self, block) -> None:
        peak = float(np.abs(block).max()) if len(block) else 0.0
        self.peaks.append(peak)
        if peak > 0.0:
            self.sink.append(len(block))   # real audio only; silence is the idle cushion
        else:
            self.silent_blocks += 1
        time.sleep(0.02)   # stand in for realtime playback so pausing is observable


    def abort(self) -> None:
        self.active = False


class FakeSd:
    def __init__(self) -> None:
        self.written: list = []
        self.reinitialised = False
        self.stream: FakeStream | None = None

    def OutputStream(self, **kwargs) -> FakeStream:
        self.stream = FakeStream(self.written)
        return self.stream

    def _terminate(self) -> None:
        pass

    def _initialize(self) -> None:
        self.reinitialised = True


class StaleSd(FakeSd):
    """Refuses to open until the device list has been re-enumerated — a resident daemon
    after Bluetooth earbuds connect."""

    def OutputStream(self, **kwargs) -> FakeStream:
        if not self.reinitialised:
            raise RuntimeError("Internal PortAudio error [PaErrorCode -9986]")
        return FakeStream(self.written)


class FakeModel:
    def generate(self, text, voice=None, speed=1.0, lang_code="a"):
        class Chunk:
            # Audible, not zeros: the player writes silence as an idle cushion, so the stubs
            # have to be able to tell the two apart.
            audio = np.full(speak_app.SAMPLE_RATE, 0.5, dtype="float32")   # 1s per sentence
        yield Chunk()


class ExplodingModel:
    """Fails the first generate, then works — a resident daemon must survive one bad
    utterance."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, text, voice=None, speed=1.0, lang_code="a"):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("phonemiser blew up")
        return FakeModel().generate(text)


class SlowModel:
    """Generation takes a while, as it does after the model's pages have been compressed
    out of RAM."""

    def generate(self, text, voice=None, speed=1.0, lang_code="a"):
        time.sleep(0.5)
        return FakeModel().generate(text)


class RecordingModel:
    """Records the speed each sentence was generated at, to measure how long a speed change
    takes to be heard. Generation is cheap relative to playback here, as it is in reality
    (~0.7s to generate against ~3s to play), so the lookahead is as deep as it ever gets."""

    def __init__(self) -> None:
        self.speeds: list[float] = []

    def generate(self, text, voice=None, speed=1.0, lang_code="a"):
        time.sleep(0.05)
        self.speeds.append(speed)
        return FakeModel().generate(text)


class NullUi:
    """Counts hide() calls, which is how a test knows an utterance is finished: _serve
    hides the panel once per job, whether it played, was cut short, or failed."""

    def __init__(self) -> None:
        self.hides = 0

    def show(self, text: str) -> None: pass
    def status(self, text: str) -> None: pass
    def icon(self, glyph: str) -> None: pass

    def hide(self) -> None:
        self.hides += 1


BLOCKS_PER_SENTENCE = speak_app.SAMPLE_RATE // speak_app.BLOCK


def test_pause_resume_and_barge_in():
    speak_app.sd = fake = FakeSd()
    player = speak_app.Player(FakeModel(), NullUi())

    sentences = 8
    player.submit(" ".join(f"Sentence number {i}." for i in range(sentences)))
    wait_until(lambda: fake.written, "playback to start")

    player.toggle()          # pause
    paused_at = wait_stable(lambda: len(fake.written), "playback after pause")
    assert paused_at < sentences * BLOCKS_PER_SENTENCE, \
        "utterance finished before the pause could be observed — test is not exercising pause"

    player.toggle()          # resume
    wait_until(lambda: len(fake.written) > paused_at, "playback to resume")

    player.submit("Replacement.")
    total = wait_stable(lambda: len(fake.written), "playback after barge-in")
    assert total < sentences * BLOCKS_PER_SENTENCE, f"barge-in did not cut the first utterance ({total} blocks)"


def test_speed_clamps():
    player = speak_app.Player(FakeModel(), NullUi())
    for _ in range(40):
        player.nudge(+speak_app.SPEED_STEP)
    assert player.speed == speak_app.SPEED_RANGE[1]
    for _ in range(40):
        player.nudge(-speak_app.SPEED_STEP)
    assert player.speed == speak_app.SPEED_RANGE[0]


def test_refreshes_device_list_before_playing():
    """The failure this guards against is silent: a stale device list still opens and still
    accepts every sample, it just plays to a device nobody is listening to."""
    speak_app.sd = fake = StaleSd()
    player = speak_app.Player(FakeModel(), NullUi())
    player.submit("Earbuds connected while it was idle.")
    wait_until(lambda: fake.written, "audio after the device list was refreshed")
    assert fake.reinitialised, "opened the speaker without re-enumerating devices first"


def test_barge_in_while_paused():
    """Pausing with space and then triggering the hotkey used to wedge the player forever:
    the writer sat in playing.wait() and never saw the stop flag."""
    speak_app.sd = fake = FakeSd()
    player = speak_app.Player(FakeModel(), NullUi())

    player.submit(" ".join(f"Sentence number {i}." for i in range(8)))
    wait_until(lambda: fake.written, "playback to start")
    player.toggle()                       # pause
    paused_at = wait_stable(lambda: len(fake.written), "playback after pause")

    player.submit("This must be spoken even though playback was paused.")
    wait_until(lambda: len(fake.written) > paused_at, "a paused player to speak a new utterance")


def test_device_is_fed_silence_while_generating():
    """Two artifacts, one fix. A running stream with nothing to write underruns and the device
    replays stale buffer content; starting it only once audio is ready instead puts the
    Bluetooth link's start-up ramp on the first syllables. So it starts early and is kept fed
    with silence until real audio arrives."""
    speak_app.sd = fake = FakeSd()
    player = speak_app.Player(SlowModel(), NullUi())

    player.submit("Generation is slow for this one.")
    # SlowModel takes 0.5s, so this lands mid-generation however loaded the machine is.
    wait_until(lambda: fake.stream is not None and fake.stream.silent_blocks > 0,
               "the waiting device to be fed silence instead of left starved")
    assert fake.stream.active, "device should be open and running while generating"
    assert not fake.written, "played audio before any had been generated"

    wait_until(lambda: fake.written, "the generated audio to play")


def test_speed_change_is_heard_within_two_sentences():
    """← → can only affect audio that has not been generated yet, so the question is how many
    sentences are in flight. Generating before waiting for a queue slot kept three — playing,
    queued, and one finished and blocked on the put — and the change took 3 sentences to be
    heard. Waiting for the slot first leaves two."""
    speak_app.sd = fake = FakeSd()
    model = RecordingModel()
    ui = NullUi()
    player = speak_app.Player(model, ui)

    sentences = 9
    player.submit(" ".join(f"Sentence number {i}." for i in range(sentences)))
    # Midway through the third sentence: far enough in that the lookahead is fully primed.
    wait_until(lambda: len(fake.written) >= 2 * BLOCKS_PER_SENTENCE + BLOCKS_PER_SENTENCE // 2,
               "the third sentence to start playing")
    playing = len(fake.written) // BLOCKS_PER_SENTENCE   # 0-based index being written now
    player.nudge(+speak_app.SPEED_STEP)
    wait_until(lambda: ui.hides > 0, "the utterance to finish")

    at_new_speed = [i for i, s in enumerate(model.speeds) if s != speak_app.SPEED]
    assert at_new_speed, "the new speed was never used at all"
    assert model.speeds[0] == speak_app.SPEED, "changed the speed of audio generated before the press"
    lag = at_new_speed[0] - playing
    assert lag <= 2, f"speed change took {lag} sentences to be heard, expected at most 2"


def test_producer_failure_does_not_wedge_the_player():
    """The bug this guards: a failed generate skipped the queue sentinel, the consumer
    blocked on it forever, and every later utterance was accepted and silently dropped
    while the app still showed its panel and menu bar icon."""
    speak_app.sd = fake = FakeSd()
    ui = NullUi()
    player = speak_app.Player(ExplodingModel(), ui)

    player.submit("This utterance fails to generate.")
    wait_until(lambda: ui.hides == 1, "the failed utterance to be given up on")
    assert not fake.written, "the failed utterance should have produced no audio"

    player.submit("This one must still be spoken.")
    wait_until(lambda: fake.written, "serving thread wedged — a bad utterance killed the daemon")


def test_mcp_speak_stop_status():
    """The MCP tools are thin wrappers over PLAYER — this checks the wiring (server starts,
    tool calls reach PLAYER), not playback logic, which the tests above already cover."""
    import asyncio

    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    speak_app.sd = fake = FakeSd()
    speak_app.PLAYER = speak_app.Player(FakeModel(), NullUi())

    # A real Kokoro Speak instance may already be running (e.g. via the launchd agent)
    # and listening on the real MCP_PORT. Binding an ephemeral port instead of reusing
    # the module default avoids the test silently driving that live instance — which
    # would speak out loud for real, never touch `fake`, and fail on an unrelated
    # timeout with a misleading message.
    with socket.socket() as s:              # ponytail: tiny rebind race, beats colliding
        s.bind(("127.0.0.1", 0))            # with the app that's probably on 8765
        speak_app.MCP_PORT = s.getsockname()[1]

    threading.Thread(target=speak_app.serve_mcp, daemon=True).start()

    def port_open() -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", speak_app.MCP_PORT)) == 0

    wait_until(port_open, "mcp server to start listening")

    url = f"http://127.0.0.1:{speak_app.MCP_PORT}/mcp"

    async def call(name: str, arguments: dict):
        async with streamable_http_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)

    status_before = asyncio.run(call("status", {}))
    assert not status_before.is_error, f"status tool failed: {status_before}"

    asyncio.run(call("speak", {"text": "MCP test utterance."}))
    wait_until(lambda: fake.written, "audio after speak() was called over MCP")

    asyncio.run(call("stop", {}))
    assert speak_app.PLAYER.stopped.is_set(), "PLAYER.cancel() was not triggered by the stop tool"


if __name__ == "__main__":
    test_pause_resume_and_barge_in()
    test_speed_clamps()
    test_refreshes_device_list_before_playing()
    test_barge_in_while_paused()
    test_device_is_fed_silence_while_generating()
    test_speed_change_is_heard_within_two_sentences()
    test_producer_failure_does_not_wedge_the_player()
    test_mcp_speak_stop_status()
    print("speak_app checks passed")
