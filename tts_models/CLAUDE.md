# tts_models — read before moving anything

Two unrelated things share this folder:

* **Kokoro Speak — a working utility.** Menu-bar app, hotkey, launchd agent, MCP
  server. People use it. Treat the layout below as load-bearing.
* **`voice_clone/` — an experiment.** Cloning my own voice with Qwen3-TTS.
  Self-contained; nothing outside it depends on anything inside it.

## Do not move these — the app resolves them by relative path

`Kokoro Speak.app/Contents/MacOS/kokoro-speak` computes:

```sh
HERE="$(cd "$(cd -P "$(dirname "$0")" && pwd)/../../.." && pwd)"
"$HERE/.venv/bin/python" -u "$HERE/speak_app.py" &
```

Three levels up from `Contents/MacOS/` is **this directory**. So all of these
must stay at the root, as siblings:

| | |
|---|---|
| `Kokoro Speak.app/` | the bundle Spotlight has registered with LaunchServices |
| `speak_app.py` | what the launcher runs |
| `.venv/` | the interpreter the launcher runs it with |

`com.kokoro.speak.plist` hardcodes `__REPO__/Kokoro Speak.app/Contents/MacOS/kokoro-speak`
with `WorkingDirectory __REPO__` (substituted at install time). Moving the bundle
breaks the launchd agent, and re-registering with Spotlight is not automatic.

The launcher starts python as a **child, deliberately not `exec`** — there is a
long comment in the stub explaining why, and it cost an hour to find. Do not
"simplify" it to an exec.

## `output_*.wav` are gitignored but not disposable

They are TTS benchmark outputs, regenerable via `run_kokoro.py`, `run_piper.py`,
`run_say.py`, `run_inflect.py`. Four of them are **also the floor voices** for
`voice_clone/analysis/timbre_score.py`. Deleting them does not lose data, but it
does break the experiment's metric until they are regenerated — and regenerating
them changes the floor slightly, which shifts every historical percentage.

## The venv is half-broken

66 of 69 console scripts in `.venv/bin/` have a shebang pointing at
`~/Documents/quant/tts_models/.venv/bin/python3` — a stale path
from before the repo moved. **`.venv/bin/python` itself works** (it is a symlink
to the Homebrew interpreter), so `.venv/bin/python -m <module>` is fine and
`.venv/bin/<script>` is not. Use `~/.local/bin/hf` for the HF CLI. Not yet fixed.

## Experiment invariants

* Score anything new with `voice_clone/analysis/timbre_score.py` before claiming
  an improvement. The bar is **55.1%** (zero-shot). A full fine-tune on 1.7
  minutes scored 9.5% — below a generic Kokoro voice.
* Do not retrain on the 1.7-minute dataset. More epochs cannot add information
  that is not in 29 clips. The constraint is data volume; see
  `voice_clone/README.md`.
* `voice_clone/reference/` holds two cloned third-party repos, kept for
  comparison. They are not dependencies — nothing imports from them.

## Git

Repo-wide rule from `../CLAUDE.md` applies here and is absolute: **never run
`git commit`.** The user commits manually.
