# Kokoro Speak MCP Tool — Design

## Purpose
Expose the Kokoro Speak app's read-aloud capability (`tts_models/speak_app.py`)
as an MCP tool, so Claude Code can trigger speech directly instead of only via
the global hotkey or the Unix-socket CLI trigger.

## Scope
In scope: an embedded MCP server inside `speak_app.py`, three tools
(`speak`, `stop`, `status`), and a project-scoped `.mcp.json` registering it.
Out of scope: a Claude Code skill file (the tools' own MCP descriptions are
sufficient — no separate meta-documentation layer needed for three
self-explanatory tools); any change to the existing hotkey or socket trigger,
which remain as-is; Windows/cross-platform packaging (separate, not-yet-scoped
idea — see memory `kokoro-speak-mcp-tool-idea`... this doc supersedes that
memory's placeholder once implemented).

## Location
`quant/tts_models/speak_app.py` (extended in place) and a new
`quant/.mcp.json`.

## Components

- **`serve_mcp()`** (new function in `speak_app.py`) — builds a `FastMCP`
  instance (official `mcp` Python SDK), registers the three tools below, and
  runs it with the `streamable-http` transport bound to
  `127.0.0.1:MCP_PORT`. `MCP_PORT = 8765` is a new constant in the CONFIG
  block next to `SOCK_PATH`/`LOCK_PATH` (arbitrary unregistered port; change
  it if it conflicts with something on your machine). Started as a daemon
  thread from `main()`,
  alongside the existing `threading.Thread(target=serve_socket, ...)` call —
  same lifecycle, no new process, no new lifecycle management.
- **Tools**, each a thin wrapper over the existing module-level `PLAYER`
  object — no new playback logic:
  - `speak(text: str) -> str` — calls `PLAYER.submit(text)`, returns a short
    confirmation (e.g. `"queued {len(text)} chars"`). Tool description
    includes: *"Requires the Kokoro Speak app to be running — if this tool is
    unreachable, tell the user to open Kokoro Speak via Spotlight."* Unlike
    the socket trigger, empty text is not a "read the current selection"
    shortcut — Claude always passes explicit text, so that magic doesn't
    apply here.
  - `stop() -> str` — calls `PLAYER.cancel()`, returns `"stopped"`.
  - `status() -> dict` — returns
    `{"playing": PLAYER.playing.is_set(), "progress": PLAYER.progress,
    "speed": PLAYER.speed}`, read directly off existing `Player` state.
- **`quant/.mcp.json`** — project-scoped MCP config, HTTP transport, pointing
  at `http://127.0.0.1:MCP_PORT/mcp`. Exact key names need a doc-check against
  Claude Code's current `.mcp.json` schema for HTTP-transport servers at
  implementation time — not assumed here.
- **`requirements_app.txt`** — add `mcp` (official Python SDK).

## Data flow
Claude Code → HTTP call to `127.0.0.1:MCP_PORT` → `FastMCP` tool handler →
same `PLAYER.submit()` / `PLAYER.cancel()` / `PLAYER` state reads the socket
trigger already uses → same `Player`/`Ui` playback and on-screen panel. MCP is
a second front door onto machinery that already exists; nothing about
synthesis or playback changes.

## Error handling
- App not running: the HTTP connection is refused, surfaced to Claude as a
  tool-call error. No retry logic — the `speak` tool's own description primes
  Claude to read that specific failure as "app isn't running" and tell the
  user to launch it, rather than retrying blindly.
- Port already bound by something else: not specially handled. `single_
  instance()`'s existing flock already prevents two copies of this app running
  at once; an unrelated process squatting `MCP_PORT` is a rare edge case that
  can fail loud at startup, consistent with how model-load failures aren't
  wrapped elsewhere in `main()`.

## Testing
Extend the existing `test_speak_app.py` (which already covers the
pause/stop/barge-in state machine) with one in-process check: start
`serve_mcp()`, make a real `speak` / `status` / `stop` call against it over
the loopback HTTP server, and assert `PLAYER` state changed accordingly. No
new test framework, fixtures, or per-tool suites — one check that fails if the
MCP wiring breaks.
