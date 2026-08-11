# Kokoro Speak MCP Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose Kokoro Speak's read-aloud capability as an MCP tool (`speak`/`stop`/`status`) that Claude Code can call directly, alongside the app's existing hotkey and Unix-socket triggers.

**Architecture:** A third daemon thread (`serve_mcp()`) is added to `speak_app.py`'s `main()`, running a `FastMCP` server (`streamable-http` transport) on `127.0.0.1:8765`. Its three tools are thin wrappers that call straight into the existing module-level `PLAYER` object — no new playback logic. A project-scoped `.mcp.json` registers the server with Claude Code.

**Tech Stack:** Python, `mcp` (official Python SDK, `FastMCP`), existing `threading`/`sounddevice`/`mlx-audio` stack already in `speak_app.py`.

## Global Constraints

- **Never run `git commit` in this repo** — per `quant/CLAUDE.md`, the user commits manually. No task below includes a commit step. `git add` is used to stage each task's changes so progress is checkpointed and reviewable/revertable (`git diff --staged`, `git restore --staged`) before the user commits.
- `MCP_PORT = 8765` (arbitrary unregistered port, matches the design doc).
- No new test framework — new tests are plain functions in the existing `test_speak_app.py`, following its established style (module-level fakes, `wait_until`/`wait_stable` polling helpers, called from `if __name__ == "__main__":`), not pytest fixtures.
- No skill file — per the approved design, the MCP tools' own descriptions carry the "how/when to use this" information.
- Design reference: `docs/superpowers/specs/2026-08-10-kokoro-speak-mcp-tool-design.md`.

---

### Task 1: Embedded MCP server (`speak`/`stop`/`status`)

**Files:**
- Modify: `tts_models/speak_app.py`
- Modify: `tts_models/requirements_app.txt`
- Test: `tts_models/test_speak_app.py`

**Interfaces:**
- Consumes: existing `speak_app.Player` (`submit(text: str)`, `cancel()`, `playing: threading.Event`, `stopped: threading.Event`, `progress: str`, `speed: float`), existing `speak_app.PLAYER` module global set in `main()`.
- Produces: `speak_app.MCP_PORT` (int constant), `speak_app.serve_mcp()` (blocking function, thread target — no args, no return).

- [ ] **Step 1: Add the `mcp` dependency and install it**

In `tts_models/requirements_app.txt`, add one line:

```
mcp
```

Run: `cd tts_models && uv pip install -r requirements_app.txt`

This goes first, ahead of the test, so the RED step in Step 3 fails because `serve_mcp`
doesn't exist yet — not because the `mcp` package itself is missing (a real background
instance of this app, `com.kokoro.speak`, may already be running via launchd; installing
the dependency doesn't touch it).

- [ ] **Step 2: Write the failing test**

Add to `tts_models/test_speak_app.py`. This needs `threading` and `socket` imported at the top of the file (add `import threading` and `import socket` next to the existing `import time`):

```python
def test_mcp_speak_stop_status():
    """The MCP tools are thin wrappers over PLAYER — this checks the wiring (server starts,
    tool calls reach PLAYER), not playback logic, which the tests above already cover."""
    import asyncio

    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

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
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)

    status_before = asyncio.run(call("status", {}))
    assert not status_before.isError, f"status tool failed: {status_before}"

    asyncio.run(call("speak", {"text": "MCP test utterance."}))
    wait_until(lambda: fake.written, "audio after speak() was called over MCP")

    asyncio.run(call("stop", {}))
    assert speak_app.PLAYER.stopped.is_set(), "PLAYER.cancel() was not triggered by the stop tool"
```

Add the call to the bottom of the file, in the `if __name__ == "__main__":` block, after the existing calls:

```python
    test_mcp_speak_stop_status()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd tts_models && .venv/bin/python -u test_speak_app.py`
Expected: `AttributeError: module 'speak_app' has no attribute 'serve_mcp'` — the `mcp` package is already installed (Step 1), so this fails on the missing implementation, not the missing dependency.

- [ ] **Step 4: Implement `MCP_PORT` and `serve_mcp()`**

In `tts_models/speak_app.py`, add the import next to the other third-party imports (after `import sounddevice as sd`, before the `from AppKit import (` block):

```python
from mcp.server.fastmcp import FastMCP
```

In the CONFIG block, add next to `LOCK_PATH`:

```python
MCP_PORT = 8765            # arbitrary unregistered port for the embedded MCP server
```

Add the new function near `serve_socket()` (same section of the file):

```python
def serve_mcp() -> None:
    server = FastMCP("kokoro-speak", host="127.0.0.1", port=MCP_PORT)

    @server.tool()
    def speak(text: str) -> str:
        """Speak the given text aloud through the Kokoro Speak macOS app.
        Requires the Kokoro Speak app to be running — if this tool is
        unreachable, tell the user to open Kokoro Speak via Spotlight."""
        PLAYER.submit(text)
        return f"queued {len(text)} chars"

    @server.tool()
    def stop() -> str:
        """Stop whatever Kokoro Speak is currently reading aloud."""
        PLAYER.cancel()
        return "stopped"

    @server.tool()
    def status() -> dict:
        """Report whether Kokoro Speak is currently playing, and at what
        progress and speed."""
        return {
            "playing": PLAYER.playing.is_set(),
            "progress": PLAYER.progress,
            "speed": PLAYER.speed,
        }

    server.run(transport="streamable-http")
```

In `main()`, start the thread next to the existing socket-server thread:

```python
    threading.Thread(target=serve_socket, daemon=True).start()
    threading.Thread(target=serve_mcp, daemon=True).start()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd tts_models && .venv/bin/python -u test_speak_app.py`
Expected: `speak_app checks passed` (no assertion errors, no exceptions).

- [ ] **Step 6: Stage the changes**

```bash
git add tts_models/speak_app.py tts_models/requirements_app.txt tts_models/test_speak_app.py
```

---

### Task 2: Register the MCP server with Claude Code (project scope)

**Files:**
- Create: `.mcp.json` (repo root)

**Interfaces:**
- Consumes: the running `speak_app.py`'s MCP server at `http://127.0.0.1:8765/mcp` (from Task 1).
- Produces: nothing consumed by later tasks — this is the last task.

- [ ] **Step 1: Register the server at project scope**

```bash
claude mcp add kokoro-speak http://127.0.0.1:8765/mcp --transport http --scope project
```

If the installed Claude Code version uses different flag names, this command errors with
its own usage message — no separate `--help` check needed first.

- [ ] **Step 2: Verify Claude Code can see the server**

Run: `claude mcp list`
Expected: `kokoro-speak` listed as connected. The app should already be running (e.g. via the
`com.kokoro.speak` launchd agent, if installed per `docs/kokoro_speak_app_guide.md` Step 7);
if it's not running, start it first with `cd tts_models && .venv/bin/python -u speak_app.py`.

- [ ] **Step 3: Stage the change**

```bash
git add .mcp.json
```

---

## Self-Review

- **Spec coverage:** `serve_mcp()` + three tools (Task 1) ✓, `.mcp.json` project-scoped config (Task 2) ✓, `mcp` dependency added (Task 1, Step 1) ✓, in-process test extending `test_speak_app.py` (Task 1, Steps 2–5) ✓, no skill file (per design, none added) ✓, no commits anywhere (Global Constraints + every task ends on `git add`, not `git commit`) ✓.
- **Placeholder scan:** no TBD/TODO.
- **Type consistency:** `PLAYER.submit(text: str)`, `PLAYER.cancel()`, `PLAYER.playing`/`PLAYER.stopped` (`threading.Event`), `PLAYER.progress` (`str`), `PLAYER.speed` (`float`) — all match the existing `Player` class in `speak_app.py:275-316` and are used identically in Task 1's tools and test.
- **Post-review fixes applied (external review via /remote-control, verified against this machine before applying):**
  - Task 1's test now binds an ephemeral port and overrides `speak_app.MCP_PORT` before starting `serve_mcp()`, instead of reusing the real `8765` — confirmed the `com.kokoro.speak` launchd agent is genuinely running on this machine (pid check via `launchctl list`), so the original test would have silently driven that live instance once Task 1 shipped.
  - Task 1 reordered: `mcp` dependency install is now Step 1 (was Step 3), so the RED step (Step 3) fails on the missing `serve_mcp` implementation, not a missing import — confirmed `mcp` is not currently installed in `.venv`.
  - Task 2 collapsed from 5 steps to 3 (dropped the `--help` pre-check and the manual start/stop dance) — a reasonable simplification, kept independent of scope.
  - Two other review suggestions (switch `.mcp.json` to user-scope; cut the `status()` tool) were **not applied** — both reverse explicit choices made earlier in this conversation, and the user re-confirmed both when asked directly.
