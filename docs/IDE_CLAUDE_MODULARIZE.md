# IDE Claude Prompt — Repo Cleanup + Full Modularization (KitPri v4)

> Paste everything below the line into IDE Claude. Run **PHASE 0 first and stop** — do not let it start refactoring until you've approved the deletion list.

---

## PROJECT CONTEXT

**KitPri v4** — Samsung PRISM Worklet 25ST31BMS, B.M.S. College of Engineering.
Binary audio classifier: **"Cooking" vs "Not Cooking"**, deployed to Samsung SmartThings edge devices (~60 MB RAM budget).

Current pipeline: **AST teacher → knowledge distillation → MobileNetV2 student → INT8 quantization.**

The code currently works but is **rigid**: model choices, preprocessing constants, and file paths are hardcoded and duplicated across scripts. I want it modularized so that swapping a model, changing a hyperparameter, or altering preprocessing is a **one-line config change**, not a hunt-and-replace across five files.

---

## DATA ACCESS MODEL — READ THIS BEFORE ANYTHING ELSE

**The audio dataset is NOT stored locally and must never be committed.** It lives on Kaggle and is regenerated or downloaded on demand.

This means:

1. **No dataset audio exists in this repo, and none should be added.** If you find `.wav`/`.mp3`/`.flac` files anywhere, flag them as Bucket E immediately.
2. **All dataset paths come from config**, with the default pointing at the Kaggle location and an override for local runs:
   ```yaml
   data:
     root: ${oc.env:KITPRI_DATA_ROOT,/kaggle/input/datasets/ayushalia/kitpri-v4}
     audio_subdir: audio_32k
     metadata_dir: metadata
   ```
   Code must never hardcode `/kaggle/...`. Reading from config is fine; embedding the literal in `src/` is not.
3. **Training and dataset generation run on Kaggle, not locally.** Do not attempt to execute them here, and do not report them as "failing" — they are expected to be un-runnable without the data. Verify they are *syntactically valid and importable* instead.
4. **Metadata CSVs may be present locally** even when audio is not. Handle the case where a CSV row references a file that doesn't exist: fail with a clear message naming the missing path, never silently skip rows (silent skipping previously caused a 1,228-file data loss to go unnoticed).

### Test fixtures — the one exception

Keep **3–5 short sample clips** in `tests/fixtures/` (~2 MB total). These are *not* the dataset; they exist solely so the preprocessing round-trip test can run. Choose clips whose probabilities appear in `results/kitpri_v4_distilled_mobilenet/test_predictions.csv` so expected values are known.

If no such clips are available, say so explicitly and mark the round-trip test as skipped with a clear reason — **do not fabricate expected values or write a test that trivially passes.**

---

# PHASE 0 — REPO AUDIT (DO THIS FIRST, THEN STOP)

**Before writing or refactoring anything, produce a complete inventory of the repository.**

Walk every file in the repo and classify it into exactly one of these buckets:

### Bucket A — Directly used
Imported, executed, or loaded at runtime by the inference path or the training path. State *which* script uses it.

### Bucket B — Indirectly used
Not executed, but required for the project to be understood, reproduced, or submitted — README, architecture diagram, result JSONs/CSVs cited in the report, licence files, reports.

### Bucket C — Unused / dead / removable
Nothing references it, directly or indirectly. For each file, state **why** you concluded it's unused.

### Bucket D — Duplicates
Same content or same purpose existing in more than one place. Identify which copy is canonical and which are redundant.

### Bucket E — Must NEVER be committed
Secrets, tokens, `.env`, dataset audio, checkpoints over 100 MB, `__pycache__`, `.ipynb_checkpoints`, virtualenvs, `.DS_Store`.

**Output format — a single table:**

| File | Size | Bucket | Used by / Reason | Action |
|---|---|---|---|---|

Then produce a summary:
- Total files, total repo size
- Count and combined size per bucket
- **An explicit list of files you recommend deleting**, each with a one-line justification

### Specific things to check for

1. **Filename collisions.** Two different models are both named `best_model.pt`:
   - AST teacher: ~329 MiB, state dict keys begin `audio_spectrogram_transformer.*`
   - MobileNetV2 student: ~8.5 MB, keys begin `conv_stem.*`
   Identify every copy and which is which by inspecting the keys — do not guess from the path.

2. **Redundant model formats.** There may be `.pt`, `.npy`, and TorchScript versions of the same weights. Determine which are actually loaded by code and which are leftovers.

3. **Files exceeding GitHub's 100 MB limit** (`find . -type f -size +100M`).

4. **Virtualenvs.** A `venv/` (or `env/`, `.venv/`) containing torch/torchaudio/torchvision is typically 2-4 GB and contains `.so` files over 100 MB. Confirm it is gitignored AND not already tracked:
   ```bash
   git ls-files | grep -E "site-packages|pyvenv.cfg" | head
   ```
   If tracked, report it prominently - it needs `git rm -r --cached` and possibly history rewriting before any push.

5. **Stray dataset audio.** The dataset is remote (see Data Access Model). Any `.wav`/`.mp3`/`.flac` in the repo other than `tests/fixtures/` is an error - flag it.

6. **Stale outputs** — old confusion matrices, superseded training logs, results from discarded experiment versions.

7. **Secrets** — grep the working tree *and git history* for Telegram bot tokens, API keys, `.env` files.

### RULES FOR PHASE 0

- **Do not delete anything yet.** Produce the list and stop.
- **When unsure, classify as "uncertain" rather than "unused."** A wrongly deleted artifact may be unrecoverable; a wrongly kept file costs nothing.
- **Never recommend deleting anything under `results/`** without explicitly flagging that these are the evidence base for reported metrics.
- If you find a committed secret, say so **immediately and prominently** — it must be rotated, not merely deleted.

**STOP after Phase 0 and wait for my approval before proceeding.**

---

# PHASE 1 — TARGET ARCHITECTURE

Once I approve the deletions, restructure the repo to this shape:

```
kitpri/
├── configs/
│   ├── base.yaml                # shared defaults
│   ├── audio/
│   │   └── mel_32k_10s.yaml     # the current preprocessing profile
│   ├── models/
│   │   ├── ast_teacher.yaml
│   │   ├── mobilenetv2_student.yaml
│   │   └── efficientnet_b0.yaml # for future comparison runs
│   └── experiments/
│       ├── train_teacher.yaml
│       ├── distill.yaml
│       └── quantize.yaml
├── src/kitpri/
│   ├── __init__.py
│   ├── config.py                # loads + validates YAML, resolves inheritance
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── io.py                # load/resample/mono/pad/truncate
│   │   └── features.py          # mel-spectrogram + normalization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py          # name -> builder, the swap point
│   │   ├── ast.py
│   │   ├── mobilenet.py
│   │   └── base.py              # shared interface all models satisfy
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── synthesis.py         # v4 synthetic mixing
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── distill.py
│   │   └── callbacks.py         # early stopping, checkpointing
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── threshold.py         # sweep + selection
│   ├── deploy/
│   │   ├── __init__.py
│   │   ├── quantize.py
│   │   └── export.py            # TorchScript / ONNX
│   └── inference/
│       ├── __init__.py
│       └── predictor.py         # single reusable class
├── scripts/                     # thin CLI wrappers, no logic
│   ├── create_dataset.py
│   ├── train.py
│   ├── distill.py
│   ├── quantize.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
├── results/
├── docs/
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Core design requirements

### 1. Single source of truth for preprocessing

Preprocessing constants currently appear in multiple scripts. They must be defined **once**, in config, and consumed everywhere — training, evaluation, quantization calibration, inference, and the Telegram bot.

**The current verified values (do not change them):**

```yaml
audio:
  sample_rate: 32000
  clip_duration: 10.0
  n_mels: 128
  n_fft: 1024
  hop_length: 512
  db_top_db: 80
  normalization: per_clip     # (x - x.mean()) / (x.std() + 1e-6)
  channels: 3                 # via .repeat(3, 1, 1)
  resize: null                # NO resize — natural 128 x 626 shape
```

**Critical:** there is deliberately **no resize to 224×224** and **no fixed ImageNet normalization**. Normalization is per-clip using that clip's own statistics. Adding a `Resize` or fixed `Normalize` silently degrades accuracy without raising an error. Add a comment saying exactly this in `features.py`.

Add a runtime assertion that the feature tensor shape matches what the loaded model expects, so a mismatch fails loudly instead of silently producing garbage.

### 2. Model registry — the swap point

```python
# src/kitpri/models/registry.py
MODEL_REGISTRY = {}

def register_model(name):
    def deco(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return deco

def build_model(cfg):
    if cfg.name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{cfg.name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[cfg.name](cfg)
```

Adding a new architecture must require **only**: a new `@register_model("name")` function plus a YAML file. No edits to training, evaluation, or inference code.

Every model must expose the same interface: input `(B, 3, n_mels, T)`, output a **single logit** of shape `(B, 1)`.

### 3. Unified predictor

One `Predictor` class used by `scripts/predict.py`, the evaluation loop, **and** the Telegram bot — so the bot can never drift from training preprocessing:

```python
class Predictor:
    def __init__(self, model_path, config_path=None, threshold=None, device="cpu"): ...
    def predict_file(self, path) -> dict:   # {probability, label, prediction}
    def predict_batch(self, paths) -> list[dict]: ...
```

It must transparently handle **both** TorchScript INT8 models (`torch.jit.load`) and plain state-dict checkpoints, auto-detecting which it was given.

### 4. Label convention — centralize it

**Verified:** `1` = cooking, `0` = noncooking. `sigmoid(logit)` = P(cooking).
(Confirmed against `test_predictions.csv`: rows prefixed `c_` have `true_label=1`.)

Define this **once** as a constant with a comment. There is a known bug where the Telegram bot classifies everything as "Cooking" — a suspected label-convention flip. Centralizing this makes that class of bug impossible to reintroduce.

### 5. Reproducibility

- A `seed` field in config, applied to `random`, `numpy`, and `torch`.
- `create_dataset.py` **must** be seeded — otherwise a reviewer regenerating the dataset gets different clips than the ones the reported metrics came from.
- Every run writes a `run_config.yaml` snapshot next to its outputs, recording the exact config used.

### 6. Resume-from-checkpoint

Kaggle sessions time out. Every training entry point needs `--resume` support that restores model, optimizer, scheduler, and epoch state.

### 7. No hardcoded paths

All paths come from config or CLI args with sensible defaults. Zero `/kaggle/...` or `/Users/...` strings anywhere in `src/`.

---

# PHASE 2 — MIGRATION RULES

1. **Behaviour must not change.** This is a refactor. After it, `scripts/predict.py` must produce the **same probabilities** as the current `inference/predict.py` on the same input. Verify this explicitly and show the numbers.

2. **Migrate incrementally**, verifying at each step. Do not rewrite everything then test at the end.

3. **Do not invent numbers.** Every metric in docs or configs must trace to a file in `results/`. If you can't find a value, say so.

4. **Preserve the verified constants exactly.** If you find code that disagrees with the values above, **stop and report the conflict** — do not silently reconcile.

5. **Keep the README's Known Limitations section.** It honestly documents that recall is below production bar and that the INT8 threshold was never independently re-tuned. Do not soften or remove it.

---

# PHASE 3 — TESTS

**Most tests must run without the dataset.** Generate audio synthetically in the test itself (e.g. `numpy.random.randn` written to a temp WAV via `soundfile`) rather than depending on real clips.

### Tests that need no audio files at all

- **Feature shape:** a synthetic 10 s clip at 32 kHz produces exactly `(1, 3, 128, 626)`.
- **Preprocessing invariance:** a synthetic 16 kHz input and a synthetic 3 s input both produce the correct final shape (exercises the resample and pad paths).
- **Registry:** every registered model builds and accepts `(1, 3, 128, 626)`, returning `(1, 1)`.
- **Threshold logic:** probability exactly at the threshold classifies as cooking (`>=`, not `>`).
- **Config loading:** each YAML in `configs/` loads and validates; the audio profile matches the verified constants.
- **Missing-file handling:** a metadata CSV row pointing at a non-existent audio file raises a clear error naming the path (does not silently skip).

### The one test that needs real audio

- **Round-trip:** `Predictor` on a fixture clip reproduces the probability recorded in `test_predictions.csv` (FP32 model) within tolerance.

This is **the highest-value test in the suite** — it is the only one that proves preprocessing matches training rather than merely producing correctly-shaped tensors. Everything else can pass while predictions are silently wrong.

Use the fixtures in `tests/fixtures/`. If none exist, mark it skipped with an explicit reason and state prominently in your summary that preprocessing correctness is **unverified**. Never fabricate expected values.

---

# KNOWN ISSUE TO CARRY FORWARD (document, don't silently fix)

The **0.44 threshold was tuned on the FP32 student's validation probabilities**, then applied to the INT8 model. Quantization can shift confidence calibration, so this is not strictly valid.

Additionally, the INT8 model's output is quantized to a coarse grid — roughly **0.216 per logit step**, about **5 percentage points of probability** near the decision boundary. The INT8 model cannot express any probability between ~0.446 and ~0.500. Thresholds 0.44 and 0.50 therefore do give different predictions, but finer tuning (0.44 vs 0.45) is meaningless on this model.

Make the threshold config-driven and **per-model**, so INT8 can carry its own value once someone re-sweeps it properly.

---

# DELIVERABLE ORDER

1. **Phase 0 audit table + deletion list → STOP for my approval.**
2. Skeleton: configs + `src/` package structure, no logic moved yet.
3. Migrate `audio/` and verify feature parity against current output.
4. Migrate `models/` + registry, verify each model loads.
5. Migrate `inference/`, verify identical probabilities to the current script.
6. Migrate `training/`, `eval/`, `deploy/`.
7. Tests.
8. Update README and Dockerfile for the new layout.

Confirm you understand, then begin **Phase 0 only**.
