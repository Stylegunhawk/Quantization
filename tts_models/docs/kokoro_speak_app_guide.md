# Setup Guide: Kokoro Speak on macOS

This guide explains how to configure and run the **Kokoro Speak** reader app on Apple Silicon macOS.

> [!IMPORTANT]
> **Apple Silicon (M1/M2/M3/M4) is required.** Kokoro Speak utilizes Apple's MLX framework for speech synthesis, which only runs on Apple Silicon chips. Intel-based Macs are not supported.

---

## Step 1: Install Dependencies

Kokoro Speak runs in a Python virtual environment. We recommend using **`uv`** for fast package installation (installable via `brew install uv`), but standard Python tools also work.

Navigate to the `tts_models` directory:
```bash
cd tts_models
```

Set up the environment and install only the application dependencies using [requirements_app.txt](../requirements_app.txt):
```bash
# Using uv (Recommended)
uv venv
uv pip install -r requirements_app.txt

# Or using standard python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_app.txt
```

---

## Step 2: Model Download (Automatic)

**No manual model download is required by default.** 

When you run the application for the first time, `mlx-audio` will automatically fetch the Kokoro weights model (`mlx-community/Kokoro-82M-bf16`) from Hugging Face and cache it under `~/.cache/huggingface/hub/`.

If you prefer to pre-download the model explicitly (e.g., to verify the download progress or cache it before going offline), run:
```bash
# Using Hugging Face CLI
.venv/bin/huggingface-cli download mlx-community/Kokoro-82M-bf16

# Or using Python directly
.venv/bin/python -c "from mlx_audio.tts.utils import load_model; load_model('mlx-community/Kokoro-82M-bf16')"
```

---

## Step 3: Location Constraints

> [!WARNING]
> **Do not keep the repository in `~/Documents`, `~/Desktop`, or `~/Downloads`.**  
> When launched from Finder or `launchd`, macOS’s privacy sandbox (TCC) blocks apps from accessing these directories. Python will fail to read its own virtual environment configuration and crash silently. Place the repo somewhere safe, like `~/quant/` or `~/Developer/`.

---

## Step 4: Grant Accessibility Permissions

To read the selected text in other applications and listen to the global keyboard shortcut (`⌥⌘S`), you must grant **Accessibility** permissions to the Python binary running the application:

1. Open **System Settings** → **Privacy & Security** → **Accessibility**.
2. Click the **+** (Add) button.
3. When the file dialog opens, press `⌘ + ⇧ + G` (Go to Folder).
4. Paste the path to the Python binary inside your virtual environment (e.g., `/Users/<your-username>/quant/tts_models/.venv/bin/python`) and select it.

---

## Step 5: Run the Application

You have two choices for running the app:

### Option A: Running from the Terminal (Best for troubleshooting/logs)
Run the script directly with unbuffered output so you can see live performance logs and TTS timings:
```bash
.venv/bin/python -u speak_app.py
```

### Option B: Double-click from Finder (Runs in background)
Double-click [Kokoro Speak.app](../Kokoro%20Speak.app) in Finder. It runs as an accessory application in the background (no Dock icon, but a 🔈 icon will appear in the macOS menu bar).  
*Note: Any crashes or logs during background launch are directed to `/tmp/kokoro-speak.log`.*

---

## Step 6: How to Use It

Once the app is running:
* **Reading Text:** Highlight text in any application and press **`⌥⌘S`** (Option + Command + S).
* **Player Controls:** While speaking, a floating overlay panel will show the sentence progress. You can use:
  * `space` — Pause / Resume
  * `←` / `→` — Decrease / Increase playback speed
  * `esc` — Stop speaking
* **Quit:** Click the 🔈 menu-bar icon and choose **Quit**.

### CLI Triggers (Optional)
You can trigger speech manually over a Unix socket:
```bash
# Speak the current clipboard contents
pbpaste | nc -U /tmp/kokoro-speak.sock

# Speak the current active screen selection (Accessibility permission required)
nc -U /tmp/kokoro-speak.sock < /dev/null
```

---

## Step 7: Configure Autostart at Login (Optional)

If you want the application to automatically run when you log into your Mac:

### Option A: System Login Items (Easiest)
Go to **System Settings** → **General** → **Login Items** and drag [Kokoro Speak.app](../Kokoro%20Speak.app) into the list.

### Option B: launchd Agent (Auto-restarts if it crashes)
To install the plist daemon template (which logs to `/tmp/kokoro-speak.log`):
```bash
# Substitute template variables with current absolute paths and install
sed "s|__REPO__|$PWD|g" com.kokoro.speak.plist > ~/Library/LaunchAgents/com.kokoro.speak.plist

# Load and start the background agent
launchctl load ~/Library/LaunchAgents/com.kokoro.speak.plist
```

To stop the daemon later:
```bash
launchctl unload ~/Library/LaunchAgents/com.kokoro.speak.plist
```

---

## References

* For detailed architecture notes on latency, TTFA (Time To First Audio), and streaming performance optimizations, check [docs/streaming.md](streaming.md).
* For benchmark comparisons with Piper, macOS `say`, and Inflect-Micro-v2, read the [README.md](../README.md).