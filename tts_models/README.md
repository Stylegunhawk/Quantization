# TTS Model Testing

Local sandbox to compare Kokoro-82M, Piper, and Inflect-Micro-v2 on latency,
CPU/RAM/GPU spike.

## Setup (Mac or Windows, using `uv`)

    uv venv
    uv pip install -r requirements.txt

Piper needs a voice model downloaded once, next to this README:

    curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
    curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

Kokoro needs spaCy's `en_core_web_sm` model, which it normally fetches from
a GitHub release on first run. If that host is blocked on your network
(fails with a `spacy-models` download timeout), grab the mirrored copy from
Hugging Face instead and install it standalone (`--no-deps` avoids a spaCy
version downgrade — rename the file to match the version printed by the
`--no-deps` install error if HF's filename doesn't already match):

    uv run hf download spacy/en_core_web_sm en_core_web_sm-any-py3-none-any.whl --local-dir /tmp/spacy_model
    mv /tmp/spacy_model/en_core_web_sm-any-py3-none-any.whl /tmp/spacy_model/en_core_web_sm-3.7.1-py3-none-any.whl
    uv pip install --no-deps /tmp/spacy_model/en_core_web_sm-3.7.1-py3-none-any.whl

Inflect-Micro-v2 downloads itself automatically on first run of
`run_inflect.py` (into `Inflect-Micro-v2/`, via `huggingface_hub`). If the
import fails on `scipy`, install its own extra deps — but skip its
`phonemizer` line, which clobbers files that Kokoro's `phonemizer-fork`
needs at the same import path:

    uv pip install scipy soundfile espeakng-loader num2words Unidecode

If you already ran the full `Inflect-Micro-v2/requirements.txt` and Kokoro
now fails with `EspeakWrapper has no attribute 'set_data_path'`, repair it:

    uv pip uninstall phonemizer
    uv pip install --force-reinstall --no-deps phonemizer-fork

## Run

    uv run run_kokoro.py
    uv run run_piper.py
    uv run run_inflect.py

Each run writes a `.wav` (listen and compare by ear) and appends one row to
`results.csv` — device used, latency, real-time-factor, and the CPU/RAM/GPU
spike caused by that run on top of whatever else was running on the machine
at the time.

## Smoke test

    uv run test_smoke.py

## Notes

- `kokoro`'s `KPipeline` constructor may differ slightly across package
  versions — if `run_kokoro.py` errors on import/construction, check
  `pip show kokoro` and the model card on Hugging Face for the current
  signature and adjust the one call in `synthesize()`.
- `piper-tts` 1.6.0 renamed `PiperVoice.synthesize(text, wav_file)` to
  `.synthesize_wav(text, wav_file)` — `run_piper.py` already uses the new
  name; if a future version renames it again, `voice.synthesize(text)`
  alone now returns an iterable of audio chunks instead of writing a wav.
- Piper is CPU-only (runs on onnxruntime); the `device` column for its rows
  in `results.csv` reflects what `pick_device()` detected on the machine,
  not what Piper actually used.
- Inflect-Micro-v2 (9.36M params, 37.5MB, fixed voice, English only, Apache
  2.0) is much newer and less proven than Kokoro/Piper — its own published
  benchmarks show it slower on CPU than Piper. Included for comparison
  against your "smallest possible" goal, not as a replacement recommendation.
  `Inflect-Nano-v2` (~4M params) is an even smaller sibling if you want to
  try that instead — same API, just change the repo id and `MODEL_DIR` in
  `run_inflect.py`.
