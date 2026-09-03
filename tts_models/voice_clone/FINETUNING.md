# Fine-tuning Qwen3-TTS — pipeline notes

Companion to `README.md`. That file covers the *experiment* (what we're cloning,
how it's scored, why zero-shot beat the first fine-tune). This file covers the
*pipeline*: what upstream actually does, every patch we apply to it and why, and
— the part worth your time — the mistakes made getting here and what each cost.

Everything below was verified against the pinned clone at
`reference/Qwen3-TTS` (commit `022e286`), not recalled. **Line numbers are
pinned to that commit.** `reference/upstream_sft_12hz.py` is a byte-identical
copy; the notebook's simulator asserts they match, so an upstream push cannot
silently invalidate a citation here.

Status as of **2026-09-03**: **a verified off-by-one in upstream's label
alignment invalidates all three runs so far** — see §3(i). The 16-epoch run
completed cleanly (335 updates, loss 12.30 → 2.32, `bad_grads: 0`, all four
checkpoints verified as trained, host RSS flat at 8.57GB across all four saves)
and still produced unusable audio: 1 of 10 generations emitted `codec_eos`. The
cause is that upstream trains a shifted objective, which is learnable and wrong.
Patch (i) fixes both call sites; applied and statically verified, **not yet
validated by a run.**

Two evaluation defects were also fixed: the default test sentence was *verbatim*
the first line of `ref.wav`'s own transcript, and the zero-shot baseline needs
`ref_text` (55.1% is the ICL number; x-vector is 39.1%) — a matched
`ref_icl.wav`/`ref_icl.txt` pair now lives in the HF dataset.

---

## 1. Run it

Notebook: `notebooks/qwen3tts_lora_colab.ipynb`, on a Colab T4. One pass, top
to bottom; its header carries the cell-by-cell order and the expected output of
each gate.

If the notebook changes mid-session, replace the affected cells rather than
restarting — a live runtime keeps its clone, patched file, base model and
tokenised dataset, and cell 4 re-patches from pristine so it is always safe to
re-run. (A separate `_PATCH.ipynb` was used for this and then deleted: a
hot-swap file that lags the notebook it patches is worse than no file.)

Order: GPU check → install/clone → patch → base download → dataset pull →
tokenise → verify → train → generate → listen → push.

Set `DATASET_REPO` and `SPEAKER` in cell 1. **`DATASET_REPO` is a placeholder on
purpose — this repo is public and the dataset repo id is not.** Cell 1 asserts
you changed it.

---

## 1b. What is upstream's and what is ours

`reference/Qwen3-TTS/finetuning/` is the entire official fine-tuning offering —
four files, ~18KB:

| File | What it is | How we use it |
|---|---|---|
| `README.md` | a 4-step workflow: JSONL → codes → SFT → inference | followed in outline, diverged from in specifics (below) |
| `prepare_data.py` | extracts `audio_codes` with the 12Hz tokenizer | **run verbatim**, upstream flags, never edited |
| `dataset.py` | `TTSDataset` + `collate_fn` — builds the text/codec channels, the EOS label, the speaker slot | **never touched.** Imported by `sft_12hz.py`. §2 quotes it because it defines the label layout, not because we change it |
| `sft_12hz.py` | the training loop | **patched in place**, 9 patches (§3), from pristine each run |

Written by us, with no upstream counterpart:

* **the notebook** — orchestration, the eight verification gates (§4), resource probing
* **`gen.py`** — replaces the README's 6-line inference snippet, which cannot run
  here (below)
* **`analysis/build_dataset.py`** — produces `train_raw.jsonl` in upstream's
  documented format from raw recordings plus script text
* **`analysis/timbre_score.py`** — the metric. Upstream ships none

Upstream has **no PEFT path at all**: `sft_12hz.py` is full fine-tuning only, and
the README never mentions LoRA. Patches (f) and (f2) inject it.

### Where we deliberately diverge from the README

Not stylistic — each of these fails outright as written, on this hardware or with
this checkpoint:

| README says | We do | Why |
|---|---|---|
| `--init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base` (a **hub id**) | a local `snapshot_download` path | `sft_12hz.py:128` does `copytree(MODEL_PATH, ...)`, which needs a real directory. A hub id raises `FileNotFoundError` **after** the first epoch has already trained. The README's own command cannot complete. |
| `dtype=torch.bfloat16` | fp32 | T4 is sm_75; bf16 is emulated and crawls (§3c) |
| `attn_implementation="flash_attention_2"` | `sdpa` | flash-attn needs sm_80+ (§3a) |
| `--lr 2e-6` (step 3) / `2e-5` (shell script) | `1e-4` | those are full-FT rates, and the README contradicts itself between its two examples. LoRA adapters start at zero and must travel |
| `--batch_size 32` (step 3) / `2` (shell script) | `2` | 32 will not fit in 15GB at any precision |
| `--num_epochs 10` / `3` | `16` | with `gradient_accumulation_steps=4` hardcoded, 3 epochs is ~63 weight updates (§2) |
| inference from `output/checkpoint-epoch-2` | newest complete checkpoint | hardcoding it 404s when a run saved fewer epochs — `from_pretrained` treats the missing directory as a hub repo id. We inherited this bug from the README and hit it |
| inference with default sampling | `--greedy` sets **both** sampling flags | the README's snippet samples, which on T4 fp16 asserts (§5) |

The README is written for an Ampere+ GPU and the 1.7B model. On a T4 with the
0.6B, following it literally fails at the `text_projection` shape error (§3d)
before any of the above matters.

### What upstream gets right and we follow exactly

* the JSONL schema — `audio`, `text`, `ref_audio` — which is why GATE 4 asserts
  the key is `audio` (§4)
* "Strongly recommended: use the same `ref_audio` for all samples." We use one
  shared 10s `ref.wav` for all 167 clips
* the three-stage shape: raw JSONL → `prepare_data.py` → `sft_12hz.py`
* single-speaker only. The README says multi-speaker is future work, and the
  speaker is registered at a single id (3000) accordingly

---

## 2. Upstream facts you need before touching anything

### `step` is not a weight update

`sft_12hz.py:44` sets `gradient_accumulation_steps=4`, `:71` wraps the loop body
in `with accelerator.accumulate(model)`, and `:117` guards the clip with
`if accelerator.sync_gradients`. So `optimizer.step()` is a no-op 3 times in 4.

**167 clips at batch 2 = 84 dataloader steps per epoch, but only 21 weight
updates.** Upstream's progress print uses the dataloader counter. Reading a loss
curve against it overstates the run by 4×. Our patched log prints `updates:`
alongside `Step` for exactly this reason.

There is no LR scheduler and no warmup — constant `--lr`. (Checked: nothing
matches scheduler/warmup, and `:62` prepares only model/optimizer/dataloader.)

### The stop token is taught, and it's 2150

`dataset.py` `collate_fn` writes, as the final `codec_0` label of every clip:

```python
codec_0_labels[i, 8+text_ids_len-1+codec_ids_len] = config.talker_config.codec_eos_token_id
```

Training does `labels=codec_0_labels[:, 1:]` against
`inputs_embeds=input_embeddings[:, :-1, :]`, so the **last codec frame is
supervised to predict EOS**, and that position is inside the attention mask.

The base `config.json` gives `codec_eos_token_id: 2150` (not the 4198 default in
`configuration_qwen3_tts.py:393` — the checkpoint overrides it), which matches
the `eos_token_id:2150` generation stops on. Training and inference agree.

EOS is **one label in ~94** per clip, so it's ~1% of the gradient signal. A
model that runs to `max_new_tokens` early in training has not yet relearned it;
that is not evidence of a broken pipeline.

Other ids: `codec_pad_id 2148`, `codec_bos_id 2149`, `codec_nothink_id 2155`.

### The speaker embedding never comes from the table during training

`sft_12hz.py` overwrites the slot directly:

```python
input_codec_embedding[:, 6, :] = speaker_embedding      # from speaker_encoder(ref_mels)
```

and `dataset.py` sets `codec_embedding_mask[i, 6] = False` so the table lookup
there is zeroed first. At save time, `:156` writes that vector into
`codec_embedding.weight[3000]`. At inference,
`modeling_qwen3_tts.py:2095` does `talker.get_input_embeddings()(spk_id)` →
`codec_embedding[3000]` → placed at the same position (`:2171`).

**Consequence that will mislead you:** row 3000 has norm ~9.9 while real codec
rows average 0.49 (sd 0.043) — a z-score of ~220. That is *correct*. Speaker
embeddings and codec token embeddings are different kinds of vector with no
reason to share a scale, and the model was pretrained to receive one at that
position. Base row 3000 is ~0.015 because it's an unused slot.

The speaker encoder has no dropout or batchnorm, so `model.train()` cannot
perturb the captured embedding.

### The two vocabularies never touch

`talker.vocab_size = 3072`, `code_predictor.vocab_size = 2048`
(`configuration_qwen3_tts.py:373` and `:189`). The split is explicit:

| Site | Goes to | Size |
|---|---|---|
| `modeling_qwen3_tts.py:1670` | talker's own embeddings — `codec_0` | 3072 |
| `modeling_qwen3_tts.py:1684` | code predictor's embeddings — groups 1..N | 2048 |

Same split at `:1623`/`:1625` and `:1986`/`:1988`. The predictor's `lm_head` is
`Linear(hidden, 2048)` (`:1168`), so its own tokens are structurally in range,
and speaker id 3000 lands in the talker's 3072 table.

**There is no path for a `codec_0` in the 2048–3071 band to index a 2048-row
table.** We spent two sessions on that hypothesis. It is false.

### `do_sample=False` is not greedy

`do_sample` (talker) and `subtalker_dosample` (code predictor) are independent.
`qwen3_tts_model.py:325` hard-defaults `subtalker_dosample=True` and `:346`
picks it separately, so passing only `do_sample=False` leaves the code predictor
sampling at top_k=50, temperature=0.9 — and its sampled ids feed the embedding
lookup at `:1684`.

Any "greedy" result obtained without setting both flags rules nothing out.

### Base snapshot is 2.34 GiB

`model.safetensors` 1.8 GB + `speech_tokenizer/model.safetensors` 682.3 MB. The
**3.7 GB figure is the fp32 checkpoint we write** (905.8M non-LoRA params × 4),
not the bf16 model we read. Do not use one as a threshold for the other.

---

## 3. The patches

All applied to `sft_12hz.py` in one cell, from pristine (`git checkout` first —
re-running the cell is idempotent and safe).

| | What | Why |
|---|---|---|
| **a** | `flash_attention_2` → `sdpa` | flash-attn needs sm_80+; T4 is sm_75 |
| **b** | drop `log_with="tensorboard"` | needs a `logging_dir`; `init_trackers` is never called and nothing `.log()`s |
| **c** | `mixed_precision="no"`, fp32 weights | see below |
| **d** | add `text_projection` | **upstream bug** — see below |
| **e** | `AdamW` → `AdamW8bit` | fp32 Adam states are 7.3 GB against 14.56 usable |
| **f** | wrap talker in LoRA | trains 2.56% of weights (23,789,568 / 929,578,240) |
| **f2** | resolve unwrapped talker for `.model` hops | see below |
| **g** | merged save, key renaming, `.clone()` | see below |
| **g2** | atomic checkpoint via `.tmp` + rename | see below |
| **g2b** | don't copytree the base weights | save was writing 1.70 GiB for nothing |
| **g2c** | `del state_dict` + `gc.collect()` | **this is what killed two runs** |
| **g3** | save every 4th epoch + last | 16 × 4.4 GB = 70 GB against ~55 GB free |
| **h** | log grad_norm, updates, bad_grads | upstream computes the norm and discards it |
| **i** | fix label alignment, both losses | **upstream bug — the objective was shifted** |

### (c) Why pure fp32 and not fp16

T4 is Turing: `is_bf16_supported()` returns **True but only via emulation**,
which crawls — so check `get_device_capability()[0] < 8`, never that flag.

Weights must be fp32 regardless: `accelerator.prepare` wraps the optimizer in a
GradScaler, and fp16 weights + a scaler raises *"Attempting to unscale FP16
gradients"*. So fp16 buys only autocast matmuls — measured at ~5–7 minutes on a
~30-minute run. What it costs is the whole GradScaler failure class: on overflow
accelerate **skips `optimizer.step()` and halves the scale, silently**, while
the step counter keeps counting.

fp32 also matches inference, which must be fp32 on pre-Ampere anyway (§5).

**Do not instrument `GradScaler.get_scale()` as a health metric.** It starts at
65536, halves on overflow, and doubles only after `growth_interval=2000`
*consecutive* good steps. A 336-update run never reaches that, so the scale can
only fall or stay flat by construction, and the early drops (the scaler
calibrating — designed behaviour) look identical to the pathology. `grad_norm`
plus a non-finite counter is the honest signal.

### (d) The upstream bug

`text_embedding` is `nn.Embedding(vocab, text_hidden_size=2048)` in both model
sizes, but talker `hidden_size` is 2048 on the 1.7B and **1024 on the 0.6B**.
The model has a `text_projection` MLP (`modeling_qwen3_tts.py:1575`) and every
inference path uses it (`:1978`, `:2079`, `:2124`, …). `sft_12hz.py:89` does
not. On the 1.7B that's invisible (2048→2048); on the 0.6B it's a shape error.

Project first, then mask — the MLP has `bias=True`, so masking last is required
to keep padded positions at zero.

### (f2) PEFT shadows `.model`

`get_peft_model` returns a `PeftModel` whose own `.model` **is** the LoraModel
wrapper, so `model.talker.model.text_embedding` stops reaching the talker's
inner model and raises `AttributeError` at the first training step. One-hop
attribute forwarding rescues `.text_projection` and `.code_predictor`; it cannot
rescue a `.model` hop. Resolve the unwrapped talker once via `get_base_model()`
and use it for embedding lookups — but leave `model.talker(...)` wrapped, since
that is the call LoRA must intercept for anything to train.

### (g) Three ways to save a checkpoint identical to the base model

All three produce a directory that loads clean, generates clean, reports the
right speaker, and contains base weights. **This is the most dangerous failure
mode in the pipeline** because nothing errors.

1. **PEFT key names.** `lora/layer.py:129` does `self.base_layer = base_layer`,
   registering it as a submodule, so every targeted projection saves as
   `q_proj.base_layer.weight`. Stripping only the `talker.base_model.model.`
   prefix leaves that infix; the base model never finds `q_proj.weight` and
   `from_pretrained` leaves those layers at init. **Two** renames are needed.
2. **`.to('cpu')` aliasing.** `Tensor.to('cpu')` returns `self` when already on
   CPU, so the saved dict holds *references* to the merged weights and the
   following `unmerge_adapter()` rewinds the very values being written.
   `.clone()` is load-bearing. Latent on CUDA (where `.to('cpu')` copies), fires
   the moment training runs CPU-only.
3. **The copytree window.** `copytree` brings the base `model.safetensors`,
   `save_file` overwrites it ~30 s later. Die in between and the directory is
   *full size* and wrong. A size heuristic cannot see this. (g2) fixes it.

Guard with in-training asserts on key names, and verify **values** afterwards —
see GATE 6 in §4.

### (g2) Atomic checkpoints

Build in `checkpoint-epoch-N.tmp`, `os.rename` to the final name after
`save_file` returns. Rename within a filesystem is atomic, so a checkpoint
directory either doesn't exist or is complete.

`rmtree(final_dir)` must come **before** `copytree`, not after `save_file` —
`rename` onto a non-empty directory is `ENOTEMPTY`, so deleting last just moves
the window. Step 6 clears `output/` up front so there's nothing to delete.

This retires the old `>= 0.9 * max(size)` filter, which failed in both
directions: it passed a full-size directory holding base weights, and with
exactly one checkpoint `max` was that checkpoint itself, so a truncated
directory cleared its own threshold.

### (g2c) The host-RAM bug that killed two runs

`state_dict` is a plain local in `train()` and nothing frees it, so it survives
from one save to the next — **3.37 GiB** of fp32 CPU tensors. At the next save,
`state_dict = {...}` builds the replacement *in full before rebinding the name*,
so both exist simultaneously:

```
9.02 GiB (after first save) + 3.37 GiB (building second) = 12.39 GiB
                                        Colab cap        = 12.70 GiB
```

**The signature is exact: both runs died at their second save, never a first.**
Run 1 wrote `checkpoint-epoch-0` then died in epoch-1's copytree; run 2 wrote
`checkpoint-epoch-3` then died in epoch-7's. It appeared as a bare `^C` with no
traceback, which reads like a user interrupt and is not one.

This is a cost of choosing fp32 (c): upstream's bf16 state dict is 1.69 GiB, so
even two copies fit. **The fp32 decision was priced in GPU memory and never in
host RAM.**

Fix: `del state_dict, weight` + `gc.collect()` after the rename. Each save also
prints `peak host RSS so far` so you can watch it flatten rather than climb.

### (i) The labels are off by one, in two places

**This is the bug that made three runs produce unusable audio.** It is
upstream's, and it is silent: the shifted objective is perfectly learnable, so
the loss curve looks healthy all the way down.

**Root cause, both times.** `self.loss_function` is HuggingFace's. transformers
resolves it from the class name via `LOSS_MAPPING`, and both
`Qwen3TTSTalkerForConditionalGeneration` (`ForConditionalGeneration` matches) and
`Qwen3TTSTalkerCodePredictorModelForConditionalGeneration` land on
**`ForCausalLMLoss`, which shifts internally**:

```python
if shift_labels is None:
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()
```

Verified in transformers **4.57.3** — the version `qwen-tts` pins — and unchanged
in 5.12.1. Both call sites pass labels that are *already* aligned, so the shift
happens twice.

**(i.1) Talker / `codec_0`.** `sft_12hz.py` feeds `inputs_embeds[:, :-1]` with
`labels=codec_0_labels[:, 1:]` — the manual-shift idiom, correct for a raw
`cross_entropy` call and wrong when the loss shifts too. Simulated on
`L = [0..7]`:

```
upstream : logits[j] -> L[j+2]      two positions ahead
correct  : logits[j] -> L[j+1]
```

Every frame is predicted from the wrong context. And `codec_eos` is supervised
on the **second-to-last** codec frame instead of the last — verified numerically
at realistic dimensions (T=121, eos at 120): logits index 118 under upstream,
119 after the fix. The model learns to stop from a context that never occurs at
inference, which is why 335 updates with a converged loss still never emitted
it.

Fix: pass **full-length** inputs and **unshifted** labels, letting the internal
shift be the only shift. Full length rather than `labels[:, :-1]` because eos
sits at the final position of the longest item in a batch, and truncating drops
its supervision. `hidden_states[codec_mask[:, :-1]]` becomes
`hidden_states[codec_mask]` to match — row counts are identical either way
(verified: 186 = 2 × 93).

**(i.2) Sub-talker / groups 1–15.** Deeper — inside the model file, not the
script. `forward_finetune` builds `logits[j] = lm_head[j](hidden[j+1])`, and
`hidden[j+1]` has seen `codec_0..codec_j`, so `logits[j]` is *already* the
prediction for group `j+1` == `labels[j]`. `loss_function` then shifts it onto
`labels[j+1]`. Not reachable by argument, so the patch computes this loss in
`sft_12hz.py` and discards the one `forward_sub_talker_finetune` returns.

**What this invalidates.** Every checkpoint from the three runs to date. The loss
curves were real measurements of the wrong objective. Nothing else in this
document is affected — the RAM fix, the atomic save, the key renaming, the fp32
decision and every gate stand.

---

## 4. Verification gates

Placed inline at the point each artifact is produced, so they can't be skipped.

| Gate | Cell | Catches |
|---|---|---|
| 2 | install/clone | wrong cwd, incomplete clone, pip that left dist-info but no working import |
| 3 | patch | **patched file doesn't compile**; counters outside the sync block |
| 3b | base download | partial snapshot — copytrees fine, then won't load |
| 4 | dataset pull | **not 24 kHz**, missing wavs, missing `ref.wav`, wrong jsonl key |
| 5b | tokenise | row count mismatch, empty codes |
| 6 | after training | **checkpoint numerically identical to base** |
| 6b | after training | loss vs *updates*, non-finite grad count |
| 7 | generation | silence, truncation, runaway, clipping |

**GATE 3** matters more than it looks: every assert in the patch cell checks a
string was *replaced*, none check the result is valid Python. A patch landing at
the wrong indentation passes all of them and dies at step 6, ~10 minutes and a
2.34 GiB download later.

**GATE 6** is the one that covers all three §3(g) failure modes. It opens each
checkpoint and compares `q_proj` tensors against the base — `q_proj` because
LoRA targeted it, so a merged save *must* have moved it. `codec_embedding` would
be a false pass: row 3000 is written unconditionally even by a no-op run. Call
`.float()` on both sides — the checkpoint is fp32 and the base bf16, and
`Tensor.equal` across dtypes is version-dependent; where it returns False for a
mismatch rather than promoting, `not equal` is always True and the gate silently
passes everything.

**GATE 4**'s jsonl key is `audio` — `build_dataset.py:284` writes it and
`prepare_data.py:46` reads `line['audio']`. Asserting the name checks tokenizer
compatibility rather than guessing a schema.

---

## 5. Generation

**fp16 generation is broken on T4 and the failure is loud but misleading.**

```
TensorCompare.cu:109: _assert_async_cuda_kernel:
Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed
```

fp16 logits go non-finite → softmax produces `inf`/`nan` → `multinomial`
validates and asserts. Torch **does** validate; execution never reaches an
embedding lookup, which is why no vocabulary-range theory can explain it.

Greedy survives fp16 (argmax never builds a probability tensor) but produces
garbage from the same NaN logits — and never emits EOS, so it runs to
`max_new_tokens`. That asymmetry is the confirmation: sampling is what turns a
numerical problem into a crash.

**Always pass `--fp32` on pre-Ampere.** fp32 inference is only ~5.1 GB.

Run generation in a **subprocess**: a device-side assert tears down the CUDA
context for the whole process, so every later CUDA call in that kernel fails,
including a fresh `from_pretrained` on a different checkpoint. In a subprocess
the assert kills only the child.

Use `CUDA_LAUNCH_BLOCKING=1` for diagnostics only — without it the assert
surfaces at an unrelated later op with no kernel name. It serialises every
launch; never leave it on for training. And **grep stderr for `Assertion`**
rather than tailing it: an assert prints one line per offending block/thread,
thousands of them, so a `stderr[-800:]` truncates away the one line that names
the fault.

`--max-new 256` (21.3 s at 12 Hz) caps the cost of a checkpoint that never
stops. Detect runaway as `duration > 0.9 * cap`, **not** equality: the decoder
emitted 982 frames against a 1024-token cap, so an exact test calls a runaway
"stopped".

---

## 6. Measured numbers

Colab T4 (15360 MiB, sm_75), torch 2.11.0+cu128, peft 0.20.0, 167 clips /
20.0 min / 24 kHz.

| | |
|---|---|
| trainable params | 23,789,568 of 929,578,240 (2.5592%) |
| non-LoRA params | 905,788,672 |
| tokenised length | ~93 codes per clip |
| dataloader steps | 84 / epoch |
| **weight updates** | **21 / epoch** |
| speed (fp32) | ~0.73 s/step, ~61 s/epoch |
| peak GPU | **9.0 GB** fp32 (8.8 GB under fp16 — barely different) |
| host `state_dict` | 3.37 GiB fp32 (1.69 GiB bf16) |
| checkpoint on disk | 4.1 GB |
| save duration | 2m44s before (g2b) |
| loss | 11.33 → 2.78 over 167 updates, still falling |

Loss context: random over a 2048 codec vocab is ln(2048) = 7.62. **4.54 is
perplexity ~94 — a model that has learned something and cannot synthesise
speech.** 2.78 is ~16. Judge checkpoints against this, not against zero.

---

## 7. Mistakes made, and what each cost

The most useful section. Most of these looked right at the time.

**Reasoning that was locally sound but empirically wrong**

- **Removed the overlap dedup in `build_dataset.py`** because the file's own
  comment said duplicate words were harmless. Result: 127 clips instead of 166.
  difflib aligns globally, so a duplicated span lets it match half a passage
  against each copy, splitting runs beyond the gap threshold. *Two independent
  fixes had been conflated into one change.* Both measurements are now in the
  file's comment.
- **Predicted the CUDA assert would fire at the embedding lookup** (`Indexing.cu`,
  out-of-range index) because "CUDA's multinomial doesn't validate". It does, via
  `_assert_async_cuda_kernel`. Cause was right (fp16), mechanism wrong.
- **Predicted fp32 would fix the runaway** and generation would stop at 5–6 s.
  It didn't — all three surviving runs hit `max_new_tokens`. Precision and
  stopping are independent problems.
- **Told the user to watch `fp16_scale` as a health metric.** It could only fall
  or stay flat in a 336-update run; it would have looked alarming and been
  correct simultaneously.
- **Used a z-score to judge speaker embedding row 3000.** The criterion was
  meaningless — speaker and codec embeddings have no reason to share a scale.
  Established what to expect *after* running the check, not before.

**Estimates that were wrong**

- **Wall clock, twice.** First counted optimizer steps when wall clock is set by
  forward steps (4× under). Then estimated 60–95 min for the fp32 run; actual is
  ~30. Derived from a rate measured under a different precision.
- **GPU memory, twice.** Said ~5 GB, actual 8.8. Then said 11–13 GB for fp32,
  actual 9.0.
- **Never priced host RAM at all** when choosing fp32 — which is what killed
  two runs (§3 g2c).

**Self-inflicted by a fix**

- **The atomic-save patch broke the cells that read its output.**
  `checkpoint-epoch-7.tmp` matches `checkpoint-epoch-*` and `int("7.tmp")`
  raises. Filter on `.isdigit()`.
- **Three of six verification gates would have hard-stopped a healthy run**: the
  base-size threshold used the checkpoint's size (3.0 GiB) instead of the
  snapshot's (2.34), the dataset gate guessed `audio_path`/`wav` when the key is
  `audio`, and the import gate imported `peft` one cell before it was installed.
  *Each failed on correct input.* The lesson: testing that a check runs is not
  testing that it passes — validate thresholds against real artifacts.
- **A `_good` filter that only checked duration** would have passed six seconds
  of digital silence to the scorer, which returns a percentage for silence
  rather than refusing it.

**Environmental**

- `torchao` 0.10.0 ships in Colab; peft's `dispatch_torchao` calls
  `is_torchao_available()`, which **raises** below 0.16.0 rather than returning
  False (`import_utils.py:147`). Uninstall it (the function short-circuits on
  `find_spec` being None); do not upgrade, which drags a torch upgrade onto a
  working CUDA stack. Uninstall in the *install* cell, before anything imports
  peft.
- `pkill`/cleanup after a training command runs only if training returns, so it
  tells you nothing about a killed run.

---

## 8. Open

- Does 336 updates recover the stop token? EOS is ~1% of the gradient signal.
- Does timbre score peak before epoch 15? Score `_fp32_greedy` files — greedy is
  deterministic, so differences across epochs are weights, not sampling luck.
  The bar is **55.1%** (zero-shot); the 1.7-minute full fine-tune scored 9.5%.
- Whether 20 minutes is enough. Target was 30–45.

Not worth revisiting: the `codec_0 >= 2048` theory (§2), and fp16 for either
training or inference on this hardware (§3c, §5).
