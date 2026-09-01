# Voice cloning experiment — Qwen3-TTS

Cloning **my own voice** with Qwen3-TTS. This is an *experiment*, not a utility.
Nothing in here is wired into the Kokoro Speak app or the MCP server — see
`../CLAUDE.md` for what must not move.

**Status: zero-shot works at 55%. Fine-tuning failed at 9.5% because the dataset
was 1.7 minutes. Waiting on ~40 minutes of new recordings.**

---

## The two mechanisms

| | Zero-shot clone | Fine-tune |
|---|---|---|
| What changes | nothing — weights are frozen | the weights themselves |
| Input at generation | reference clip, every single time | just `speaker=SPEAKER` |
| Model | `Qwen3-TTS-12Hz-0.6B-Base` | Base → a CustomVoice-style checkpoint |
| Result | **55.1%** | **9.5%** |

`-Base` is the voice-clone model (it ships a `speaker_encoder`, 8.9M params).
`-CustomVoice` is the named-speaker model — it is Base *minus* the encoder,
carrying 9 preset Qwen voices at ids 2861–3066. Fine-tuning starts from Base
(Qwen's docs designate it as the FT base) and produces a CustomVoice-style
checkpoint with your `SPEAKER` name registered at id 3000.

## The measurement

`analysis/timbre_score.py`. Raw cosine between speaker embeddings means nothing
on its own, so both ends are anchored in **matched conditions**:

```
pct = 100 * (cos - floor) / (ceiling - floor)
```

* **ceiling 0.9956** — split-half of the reference against itself. Same person,
  mic, room, bandwidth. This is the highest score physically achievable.
* **floor 0.8432** — mean of Kokoro, Piper, macOS `say`, Inflect.

0% = generic TTS, 100% = indistinguishable from me.

Two calibration errors that inflated the number before they were caught:

1. The first negative control was `sample_for_tts.wav`, which scored **0.9795 —
   higher than either clone**, because it was my own voice. A negative control
   has to be a different speaker.
2. Comparing a 48kHz reference against a 16kHz capture produced *"123.2% of the
   way to you"*. Bandwidth mismatch breaks the scale; split-half fixes it.

Four floor voices rather than one, because any single voice may sit unusually
close or far. The spread is real: Kokoro alone is 12.7%, Piper is −6.4%.

## Results

| | cosine | scale |
|---|---|---|
| ceiling (me vs me) | 0.9956 | 100% |
| **zero-shot ICL** | 0.9272 | **55.1%** (spread 1.0pp over 3 runs) |
| zero-shot x-vector | 0.9028 | 39.1% (spread 12.3pp) |
| Kokoro | 0.8626 | 12.7% |
| **full fine-tune, 1.7 min** | 0.8577 | **9.5%** |
| floor | 0.8432 | 0% |

### Zero-shot: 55% is a method ceiling

Three references, escalating in quality, all scored on the same scale:

| Reference | ICL |
|---|---|
| `voice_ref/` — conversational, 16kHz | ~39% |
| `voice_ref2/` — phonetically varied, *performed* | ~39% |
| `voice_ref3/` — purpose-written conversational, 48kHz | **55.1%** |

Then it stopped moving. Bandwidth raised 16k→48k, transcript verified
word-accurate, script rewritten conversational, duration extended — the last two
bought nothing. A single reference clip carries a bounded amount of speaker
information, and reference-side levers are exhausted.

**ICL beats x-vector by 16 points** and is far more stable (1.0pp vs 12.3pp
spread). The branch is literally `use_icl = ref_audio is not None and ref_text is
not None`. One x-vector run produced 12.08s of audio for a 5.5s sentence.

Note that *performed* delivery scored no better than an ordinary take. Read the
recording scripts conversationally, not theatrically.

### Fine-tune: failed on data volume, not method

29 utterances / 1.7 min, lr 2e-5, 3 epochs, T4. Loss 11.59 → 4.07. Output was
noisy and unintelligible, scoring **below Kokoro**.

The loss curve is the tell. `zero0303/qwen3-tts-ljspeech-finetuned` on the Hub
used **200 samples** over 3 epochs and went 20.4 → 10.7. Ours fell *further* on
*less* data — the signature of memorising 29 clips rather than learning a voice.

Every comparable run on the Hub also uses a far lower LR:

| Model | Data | LR | Epochs |
|---|---|---|---|
| `zero0303/…ljspeech-finetuned` | 200 samples | 5e-6 | 3 |
| `duarteocarmo/…0.6B_e10_l1e6` | 200 samples | 1e-6 | 10 |
| `kgptalkie/…baraq-obama` | single speaker | 2e-6 | 5 |
| `g-group-ai-lab/gwen-tts-0.6B` | ~1000 hours | — | — |
| **ours** | **29 / 1.7 min** | **2e-5** | **3** |

**More epochs cannot fix this.** 29 clips contain a fixed amount of information
about how I sound; extra passes fit those 29 harder without adding a single new
phoneme or intonation pattern.

## Layout

```
voice_clone/
  notebooks/    qwen3tts_finetune_colab.ipynb   full FT, 13GB peak, run on T4
                qwen3tts_lora_colab.ipynb       LoRA trial, ~5GB peak, UNTESTED
  scripts/      script_{1,2,3}_*.txt            ~40 min of recording material
  recordings/   voice_ref/   take 1, conversational 16kHz    -> ~39%
                voice_ref2/  take 2, phonetic + performed    -> ~39%
                voice_ref3/  take 3, conversational 48kHz    -> 55.1%  (the reference)
  dataset/      finetune_data/  29 utts at 48kHz + train_raw.jsonl
                finetune_24k/   same, resampled to the tokenizer's native 24kHz
  analysis/     timbre_score.py    the metric — run this on anything new
                build_dataset.py   Parakeet-segmented utterances -> train_raw.jsonl
                history/           exploratory scripts, kept as a record
  reference/    Qwen3-TTS-EasyFinetuning/  full wrapper: WebUI, Docker, avg embeddings
                Qwen3-TTS-finetune/        thin wrapper, 1.7B-only
                upstream_sft_12hz.py       pristine copy, for diffing the patches
```

Training data lives in a **private** HF dataset repo, set as `DATASET_REPO` in
cell 1 of the notebook (Colab cannot use the browser upload widget from a
terminal-connected client).

## The five upstream patches

`sft_12hz.py` is written for the **1.7B on an A100**. Cell 3 of the notebook
resets it from git and re-applies all five, so it is always patched from pristine:

| | |
|---|---|
| **a** | `flash_attention_2` → `sdpa`. FA2 needs Ampere+; T4 is Turing. |
| **b** | Drop `log_with="tensorboard"` — needs a `logging_dir`, and `init_trackers` is never called. Dead config that only crashes. |
| **c** | `mixed_precision="bf16"` → `fp16`, weights → **fp32**. T4 has no bf16 hardware, and `accelerate` wraps the optimizer in a `GradScaler` which throws on fp16 gradients — AMP needs fp32 masters. |
| **d** | **Upstream bug.** `text_embedding` outputs `text_hidden_size`=2048 in both sizes, but talker `hidden_size` is 2048 on the 1.7B and **1024** on the 0.6B. A `text_projection` MLP bridges it and every inference path uses it; `sft_12hz.py:89` does not. Invisible on the 1.7B, a shape error on the 0.6B. Confirmed independently — `Qwen3-TTS-EasyFinetuning/src/sft_12hz.py:706` has the identical fix. |
| **e** | 8-bit Adam. See below. |

`shutil.copytree(MODEL_PATH, ...)` also needs `--init_model_path` to be a real
local directory, not a hub id — otherwise it raises `FileNotFoundError` *after*
the epoch has trained. Cell 3b snapshot-downloads it first.

## Memory: "0.6B" is marketing

Read straight from the safetensors header:

| | 0.6B-Base | 0.6B-CustomVoice |
|---|---|---|
| `talker.model` | 754.8M | 754.8M |
| `talker.code_predictor` | 141.6M | 141.6M |
| `talker.text_projection` | 6.3M | 6.3M |
| `talker.codec_head` | 3.1M | 3.1M |
| `speaker_encoder` | **8.9M** | absent |
| **total** | **914.6M** | 905.8M |

Full fp32 AdamW = weights 3.7 + grads 3.7 + states 7.3 = **14.6GB**, against
14.56GB usable on a T4. It OOMs inside `optimizer.step()`, where Adam lazily
allocates `exp_avg`/`exp_avg_sq`. **Batch size is irrelevant** — optimizer state
does not depend on it. 8-bit Adam drops states to 1.8GB; observed peak ~13GB.

## Using your own voice

My recordings and the dataset built from them are deliberately not in this repo
(`.gitignore` covers `recordings/` and `dataset/`). To run it on yourself:

1. Record the passages in `scripts/*.txt` — one continuous file per script,
   same room and mic throughout. Drop them in `recordings/raw/`.
2. `VOICE_REF=recordings/raw/<your first take>.wav python analysis/build_dataset.py`
3. Set `DATASET_REPO` and `SPEAKER` in cell 1 of either notebook, push
   `dataset/finetune_data/` to your own private HF dataset repo, and run it.
4. Score the output: `VOICE_REF=... python analysis/timbre_score.py out.wav`

Pick the reference take once and never change it — the ceiling and floor are both
measured against it, so every percentage in this README moves if it does.

## Reproducing

```bash
cd analysis
../../.venv/bin/python timbre_score.py                 # baseline table
../../.venv/bin/python timbre_score.py /path/to/new.wav
```

The four floor voices (`../../output_*.wav`) are **gitignored generated output**.
If missing, regenerate from `tts_models/`: `run_kokoro.py`, `run_piper.py`,
`run_say.py`, `run_inflect.py`.

## Next

1. Record `scripts/script_{1,2,3}_*.txt` — same room, mic and distance as
   `voice_ref3`. ~40 minutes total. **This is the binding constraint.**
2. `analysis/build_dataset.py` to re-segment. Expect 400-600 utterances.
3. Run the **LoRA** notebook first: ~18M trainable params vs 905.8M means far
   less capacity to memorise, which is exactly what went wrong at 1.7 min. Also
   5GB instead of 13GB, and megabyte checkpoints.
4. Drop LR to 5e-6 for full FT (LoRA wants 1e-4 — adapters start at zero).
5. Score against **55.1%**. Anything below that is worse than doing nothing.

### Open threads

* All six GPU generation runs failed silently; only CPU/fp32 completed. The
  notebook now prints stderr and return codes — `RC -9` would mean the system
  OOM-killed the process (12.7GB system RAM, 3.7GB fp32 checkpoints), anything
  else is a real Python error. Not yet diagnosed.
* `EasyFinetuning/src/embed_speaker.py:231` **averages** the speaker embedding
  across all clips; we use a single 10s `ref.wav`. Free improvement, untried.
* `sruckh/Qwen3-TTS-finetune` changes `hidden_states[codec_mask[:, :-1]]` to
  `[:, 1:]` in the sub-talker loss. Tracing `dataset.py`, `[:, 1:]` looks
  correct — `hidden[p]` predicts position `p+1`, so code at `q` needs
  `hidden[q-1]`. But EasyFinetuning keeps `[:, :-1]`, so the forks disagree and
  this is unverified. Both shapes match, so it never crashes; it would just
  train the sub-talker on leaked input.
* Community **LoRA/QLoRA** adapters for this base exist even though upstream
  ships no PEFT path: `aguken-ai/…-hi-LoRA-Finetuned-BNB-NF4` (4-bit),
  `loubna1101/Qwen3-TTS-Darija-LoRa` (single-speaker custom voice — closest
  match to this use case).
