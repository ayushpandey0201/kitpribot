# Handoff Prompt for IDE Claude — KitPri v4 Samsung PRISM Submission

> Paste everything below the line into IDE Claude.

---

## CONTEXT

I'm working on **KitPri v4**, a Samsung PRISM project (Worklet ID 25ST31BMS) at B.M.S. College of Engineering. It's a binary audio classifier — **"Cooking" vs "Not Cooking"** — for deployment on Samsung SmartThings edge devices (~60 MB RAM budget).

I need to submit an **End Review milestone** to the Samsung PRISM Web Portal and push a repository to the Samsung enterprise GitHub. A repo scaffold has already been built and tested. Your job is to complete the remaining pieces.

---

## WHAT IS ALREADY DONE AND VERIFIED

**Do not re-derive or "correct" these. They were verified by loading the actual model files and running inference end-to-end.**

### Preprocessing pipeline (CRITICAL — must match exactly everywhere)

```python
SAMPLE_RATE  = 32000
CLIP_DURATION = 10.0          # seconds
N_MELS       = 128
N_FFT        = 1024
HOP_LENGTH   = 512
NUM_SAMPLES  = 320000         # 32000 * 10.0

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

# wav -> mono -> resample to 32k -> pad/truncate to 320000 samples
mel = mel_transform(wav_32k)
mel_db = db_transform(mel)
mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)   # PER-CLIP normalize
mel_db = mel_db.repeat(3, 1, 1)                             # mono -> 3 channel
x = mel_db.unsqueeze(0)                                     # (1, 3, 128, 626)
```

**Two things that trip people up — do NOT "fix" these:**
1. There is **NO resize to 224x224**. The spectrogram keeps its natural `128 x 626` shape.
2. There is **NO fixed ImageNet normalization**. Normalization is **per-clip**, using that clip's own mean/std.

Adding a `Resize` or a fixed `Normalize` will silently degrade accuracy without raising an error.

### Models (all verified by loading)

| Model | Architecture | Params | Size | Notes |
|---|---|---|---|---|
| Teacher | HuggingFace `ASTForAudioClassification` | 86.2 M | ~329 MiB | Exceeds GitHub's 100 MB file limit — do NOT commit |
| Student FP32 | `timm.create_model('mobilenetv2_100', num_classes=1)` | 2,225,153 | 8.49 MB | Loads `strict=True` |
| Student INT8 | Static PTQ, TorchScript | 2,225,153 | 2.80 MB | `torch.jit.load()`, self-contained, no timm needed |

- Model output is a **single logit**. Apply `torch.sigmoid()` → `P(cooking)`.
- **Label convention: `1` = cooking, `0` = noncooking.** Verified against `test_predictions.csv` (rows prefixed `c_` have `true_label=1`).
- Teacher checkpoint dict has keys: `model_state`, `epoch`, `val_f1`. No config is stored inside.

### Verified metrics (test set = 450 clips, 225/225 balanced)

| Stage | Test F1 | Test Acc | Precision | Recall |
|---|---|---|---|---|
| AST teacher | 0.8129 | 0.8200 | 0.8462 | 0.7822 |
| Student FP32 @ thr 0.50 | 0.7226 | 0.7356 | 0.7598 | 0.6889 |
| Student FP32 @ thr 0.44 | 0.7318 | 0.7378 | 0.7488 | 0.7156 |
| Student INT8 @ thr 0.50 | 0.6910 | 0.7178 | 0.7634 | 0.6311 |

Chosen threshold: **0.44** (swept on validation set only; test set never used for selection).

### Repo scaffold already built and smoke-tested

```
├── README.md                                   # DONE
├── Dockerfile                                  # DONE
├── requirements.txt                            # DONE
├── .gitignore                                  # DONE
├── inference/
│   ├── predict.py                              # DONE — tested, works
│   ├── student_mobilenet_int8_scripted.pt      # DONE
│   └── student_mobilenet_fp32.pt               # DONE
├── training/
│   ├── dataset_creation.py                     # STUB — needs real code
│   ├── train_ast.py                            # STUB — needs real code
│   ├── distill_mobilenet.py                    # STUB — needs real code
│   └── quantize.py                             # STUB — needs real code
├── results/                                    # DONE (all JSON/CSV/PNG artifacts)
└── docs/
    ├── architecture_diagram.png                # DONE
    ├── KitPri_v4_Report.pdf                    # DONE
    ├── End_Review.pptx                         # MISSING
    └── reports/                                # MISSING
```

`predict.py` was tested against the real INT8 model and confirmed working, including: 16 kHz input auto-resampling, short-clip auto-padding, batch mode, and JSON output.

---

## WHAT I NEED YOU TO DO

### TASK 1 — Populate the training scripts (HIGHEST PRIORITY)

The four files in `training/` are empty stubs. I have the original code in Kaggle notebooks. For each one:

1. Convert the notebook code into a clean, runnable `.py` script.
2. **Remove all Kaggle-specific paths** (`/kaggle/input/...`, `/kaggle/working/...`). Replace with `argparse` arguments having sensible defaults.
3. Remove notebook-only artifacts: `!pip install` lines, `display()` calls, cell magics.
4. Add a module docstring explaining what the script does and how to run it.
5. **Set and expose a random seed** in `dataset_creation.py`. The synthetic mixing must be reproducible — otherwise a reviewer regenerating the dataset gets different clips than the ones my reported metrics came from.
6. Verify the preprocessing constants in these scripts match the block above **exactly**. If they don't, STOP and tell me — do not silently reconcile them.

### TASK 2 — Verify the repo end to end

```bash
pip install -r requirements.txt
python inference/predict.py --audio <some_real_test_clip>.wav
python inference/predict.py --audio_dir results/ --json
docker build -t kitpri:v4 .
docker run --rm -v "$(pwd)/clips:/data" kitpri:v4 --audio /data/sample.wav
```

Confirm the Docker build actually completes and the container runs. Report any failure verbatim rather than patching around it.

### TASK 3 — Sanity-check predict.py against known ground truth

This is the single most valuable correctness check available:

1. Take ~20 clips from the test set whose true labels are known (from `results/kitpri_v4_distilled_mobilenet/test_predictions.csv`).
2. Run `predict.py` on them.
3. Compare `predict.py`'s output probabilities against the `pred_prob` column in `test_predictions.csv`.

**They should match closely for the FP32 model.** If they don't, the preprocessing in `predict.py` diverges from training and must be fixed before submission. Report the actual numbers.

Note: `test_predictions.csv` contains FP32 student probabilities, so compare using `--model fp32`.

### TASK 4 — Fill in README placeholders

- Section 4.1: replace `<ADD YOUR KAGGLE DATASET LINK HERE>` with the real Kaggle dataset URL.
- Confirm the ESC-50 attribution and CC BY-NC 3.0 license note are present and accurate.

### TASK 5 — Pre-push safety audit

Before anything goes to the Samsung enterprise repo, check:

- [ ] **No secrets committed** — grep the whole repo and git history for the Telegram bot token, any API keys, `.env` files. If a token was ever committed, say so immediately: it must be regenerated, not just deleted.
- [ ] **No file exceeds 100 MB** (`find . -type f -size +100M`). The AST teacher (~329 MiB) must NOT be committed — Git LFS or external hosting only.
- [ ] **No dataset audio committed** (`find . -name "*.wav" | head`). `.gitignore` should already cover this — verify it works.
- [ ] `git status` is clean and only intended files are staged.

### TASK 6 — Telegram bot decision

There is an **unresolved bug**: the bot classifies all audio as "Cooking", suspected to be a label-convention flip.

Note the bot uses a **different architecture** from this pipeline — it runs DeiT-Small (`deit_small_patch16_224`, ~22 M params), NOT the AST teacher or the MobileNetV2 student. The AST teacher checkpoint will **not** load into the bot's current model class despite the class being named `ASTModel`.

Two options — recommend one and explain why:
- **(a)** Include the bot in a clearly-marked folder with a `KNOWN_ISSUES.md` documenting the bug and the diagnosis.
- **(b)** Exclude it from this milestone.

If including it: verify the label convention by checking that `label==1` rows in the training CSVs correspond to files under `audio_32k/cooking/`.

---

## GROUND RULES

1. **Do not invent numbers.** Every metric must come from a file in `results/`. If you can't find a number, say so — do not estimate.
2. **Do not change the preprocessing constants.** They're verified. If code you find disagrees with them, report the conflict rather than silently reconciling it.
3. **Verify before claiming.** If you say something works, you must have actually run it. Paste the real output.
4. **Flag file-name collisions.** Both runs use the filename `best_model.pt` for different models (AST teacher 329 MiB vs MobileNetV2 student 8.5 MB). This has already caused confusion. Use unambiguous names in the repo.
5. **Preserve the Known Limitations section in the README.** It honestly reports that recall is below production bar and that the INT8 threshold was not independently re-tuned. Do not soften or delete it.

---

## OUTSTANDING TECHNICAL ISSUE (document, don't necessarily fix today)

The **0.44 threshold was tuned on the FP32 student's validation probabilities**, then applied to the INT8 model. Quantization can shift confidence calibration, so this is not strictly valid.

Additionally, the INT8 model's output is quantized to a coarse grid — roughly **0.216 per logit step**, which is about **5 percentage points of probability** near the decision boundary. Concretely, the INT8 model cannot express any probability between ~0.446 and ~0.500. So thresholds 0.44 and 0.50 do produce different predictions, but any finer tuning (0.44 vs 0.45) is meaningless on this model.

**Proper fix (if time permits):** re-sweep the threshold using the INT8 model's own validation probabilities and report that number separately. Otherwise leave it documented as future work in the README.
