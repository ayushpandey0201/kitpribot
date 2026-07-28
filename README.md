# KitPri v4 — Cooking Sound Detection for SmartThings Edge Devices

**Samsung PRISM · Worklet ID 25ST31BMS · B.M.S. College of Engineering**

Binary audio classification (**Cooking** vs. **Not Cooking**) on short audio clips,
compressed to run on a Samsung SmartThings edge device within a ~60 MB RAM budget.

---

## 1. Quick Start (fastest path for reviewers)

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
python inference/predict.py --audio path/to/your_clip.wav
```

Expected output:

```
Model: INT8   Threshold: 0.44

FILE                                      P(cooking)  PREDICTION
--------------------------------------------------------------------
your_clip.wav                                 0.8123  Cooking
```

That is the whole demo — no training, no dataset download, no config editing.

### Telegram bot demo

A live end-to-end demo (send a voice message, get 🍳/🔇 back) lives in
[`telegram_bot/`](telegram_bot/README.md) — same `kitpri` Predictor, same
model, same threshold.

### Docker (no local Python setup needed)

```bash
docker build -t kitpri:v4 .
docker run --rm -v "$(pwd)/clips:/data" kitpri:v4 --audio /data/sample.wav
```

---

## 2. Inference Script Options

```bash
# single file
python inference/predict.py --audio sample.wav

# batch over a folder
python inference/predict.py --audio_dir ./clips/

# use the larger, more accurate FP32 model (requires timm)
python inference/predict.py --audio sample.wav --model fp32

# override the decision threshold
python inference/predict.py --audio sample.wav --threshold 0.50

# machine-readable output
python inference/predict.py --audio sample.wav --json
```

| Flag | Default | Description |
|---|---|---|
| `--audio` | — | Path to one audio file |
| `--audio_dir` | — | Directory of audio files (batch mode) |
| `--model` | `int8` | `int8` (2.9 MB, TorchScript) or `fp32` (8.5 MB, needs timm) |
| `--threshold` | `0.44` | Decision threshold, tuned on the validation set |
| `--json` | off | Emit JSON instead of a text table |

Input audio may be any sample rate, length, or channel count — the script
resamples to 32 kHz, downmixes to mono, and pads/truncates to 10.0 seconds.

> **Platform note (INT8):** the INT8 model is quantized with the fbgemm (x86)
> engine. On ARM hosts (e.g. Apple Silicon) `predict.py` refuses to run it and
> tells you to use `--model fp32` or the Docker container (which pins
> `--platform linux/amd64`). This prevents silently wrong predictions.

---

## 3. Results

| Stage | Model | Params | Size | Test F1 | Test Acc | CPU latency |
|---|---|---|---|---|---|---|
| Teacher | AST (HF `ASTForAudioClassification`) | 86.2 M | ~329 MiB | **0.8129** | 0.8200 | — |
| Student (FP32) | MobileNetV2 (`timm`) | 2.23 M | 8.49 MB | 0.7226 @ thr 0.50 | 0.7356 | 57.3 ms/clip |
| Student (FP32, tuned) | MobileNetV2 | 2.23 M | 8.49 MB | **0.7318** @ thr 0.44 | 0.7378 | 57.3 ms/clip |
| Student (INT8) | MobileNetV2 static PTQ | 2.23 M | **2.80 MB** | 0.6910 @ thr 0.50 | 0.7178 | **25.5 ms/clip** |

Compression: **329 MiB → 8.49 MB (38×) via distillation → 2.80 MB (118× total) via INT8 quantization.**
INT8 inference is **2.25× faster** than FP32 on CPU.

Test set: 450 clips, perfectly balanced (225 cooking / 225 non-cooking).

### AST teacher — test confusion matrix

|  | pred: non-cooking | pred: cooking |
|---|---|---|
| **true: non-cooking** | 193 | 32 |
| **true: cooking** | 49 | 176 |

### Distilled student (FP32) — test confusion matrix @ threshold 0.50

|  | pred: non-cooking | pred: cooking |
|---|---|---|
| **true: non-cooking** | 176 | 49 |
| **true: cooking** | 70 | 155 |

---

## 4. Method

### 4.1 Dataset (v4)

The earlier v2 dataset used clean, single-source clips and the resulting model
missed quiet sounds such as electrical hums. v4 was rebuilt so that **every clip
is a synthetic mixture of 2–4 layered sounds at varying relative volumes**,
which is substantially more realistic and substantially harder.

| | |
|---|---|
| Total clips | 7,200 |
| Train / Val / Test | 6,300 / 450 / 450 |
| Clip length | 10.0 s |
| Cooking sources | 10 organised subtypes |
| Non-cooking sources | 1,160-clip pool derived from ESC-50 |

**Dataset:** available on Kaggle — https://www.kaggle.com/datasets/ayushalia/kitpri-v4-dataset

The dataset itself is **not** committed to this repository. Use
`training/dataset_creation.py` to regenerate it, or download it from the link above.

### 4.2 Preprocessing

Identical at train and inference time:

| Parameter | Value |
|---|---|
| Sample rate | 32,000 Hz |
| Clip duration | 10.0 s (pad or truncate) |
| Channels | downmixed to mono |
| Mel bins (`n_mels`) | 128 |
| `n_fft` / `hop_length` | 1024 / 512 |
| dB conversion | `AmplitudeToDB(top_db=80)` |
| Normalization | **per-clip** `(x - x.mean()) / (x.std() + 1e-6)` |
| Channel expansion | `.repeat(3, 1, 1)` |
| Final tensor | `(1, 3, 128, 626)` |

> **Important for anyone porting this pipeline:** there is **no resize to 224×224**
> and **no fixed ImageNet normalization**. The spectrogram keeps its natural
> `128 × 626` shape and is normalized using its own statistics. Adding a `Resize`
> or a fixed `Normalize` will silently degrade accuracy without raising an error.

### 4.3 Model selection

EfficientNet-B0 (which performed well on the easier v2 dataset) was tried six
ways — varied freezing strategies, learning rates, dropout, augmentation, and
mixup — and every run landed in the same narrow band. That consistency indicated
an architecture limit rather than a tuning problem.

AST, designed for audio and using attention rather than image-style convolution
filters, beat all six EfficientNet runs within a single epoch, and became the teacher.

### 4.4 Distillation

| Hyperparameter | Value |
|---|---|
| Teacher | `kitpri_v4_ast_diagnostic` (test F1 0.8129) |
| Student | MobileNetV2 (`mobilenetv2_100`, timm), 2,225,153 params |
| Temperature | 3.0 |
| Alpha | 0.4 |
| Best epoch | 8 of 14 (early-stopped) |

### 4.5 Threshold tuning

Swept on the **validation set only** — the test set was never used for
threshold selection. Chosen threshold **0.44** (val F1 0.7747), which raises
test F1 from 0.7226 to 0.7318 versus the 0.50 default.

A lower-false-alarm alternative exists at threshold 0.87 (val FPR 0.098) but
collapses recall to 0.49 and is not recommended.

### 4.6 INT8 quantization

Static post-training quantization with a 500-clip calibration pass over real
training data. An earlier attempt using **dynamic** quantization broke the model
outright; dynamic PTQ quantizes weights only and leaves activation ranges
mismatched, which is inappropriate for CNN-style architectures.

---

## 5. Known Limitations

Reported openly rather than omitted:

1. **Overlapping-audio clips are the dominant error source.** `clip_type C`
   comprises **17% of all clips** and mixes a cooking sound with a competing
   non-cooking sound at an SNR sampled from **−5 to +20 dB** — meaning the
   cooking sound can be up to 5 dB *quieter* than the competing sound.
   Accuracy on these drops to roughly **39–50%**, versus **75–78%** on clean
   non-overlapping clips. This difficulty is by design — it reflects the
   real-world messiness the model is intended to handle.

2. **Unstructured negatives.** The "not cooking" class is a single large,
   unorganised pool of 1,160 ESC-50 sounds, making it a harder category to learn
   than "cooking", which is cleanly split into 10 subtypes.

3. **The teacher is data-limited, not model-limited.** Validation F1 peaks at
   epoch 1 and declines thereafter while training F1 climbs past 0.98. A larger
   model is unlikely to help; more diverse data and stronger augmentation would.

4. **The INT8 decision threshold was not independently re-tuned.** The 0.44
   threshold was selected using the FP32 student's validation probabilities.
   Quantization can shift a model's confidence calibration, and the INT8 model's
   output is quantized to a coarse grid (roughly 0.216 per logit step, i.e. about
   5 percentage points of probability near the decision boundary). Re-sweeping
   the threshold on INT8 validation probabilities is **outstanding future work**.

5. **Recall is below production bar.** At threshold 0.44 the FP32 student misses
   about 28% of cooking events; the INT8 model misses about 37% at threshold 0.50.
   This milestone demonstrates a working end-to-end pipeline, not a
   production-ready detector.

6. **The INT8 artifact is x86-only.** It was packed with the fbgemm engine;
   under ARM's qnnpack engine it loads but produces collapsed, meaningless
   probabilities (verified empirically). `predict.py` therefore refuses to run
   it on non-x86 hosts. Deploying to an ARM edge target requires re-quantizing
   with the qnnpack engine (or QAT) — future work.

---

## 6. Repository Structure

```
.
├── README.md
├── Dockerfile
├── requirements.txt
├── .gitignore
├── inference/
│   ├── predict.py                              # demo / inference entry point
│   ├── student_mobilenet_int8_scripted.pt      # 2.9 MB TorchScript (default)
│   └── student_mobilenet_fp32.pt               # 8.5 MB FP32 student
├── training/
│   ├── dataset_creation.py                     # regenerates the v4 dataset
│   ├── train_ast.py                            # teacher training
│   ├── distill_mobilenet.py                    # distillation
│   └── quantize.py                             # static INT8 PTQ
├── results/
│   ├── kitpri_v4_ast_diagnostic/
│   └── kitpri_v4_distilled_mobilenet/
└── docs/
    ├── architecture_diagram.png
    ├── End_Review.pptx
    └── reports/                                # weekly / monthly connect reports
```

> The AST teacher checkpoint (~329 MiB) is **not** committed — it exceeds
> GitHub's 100 MB per-file limit. Use Git LFS or host it externally and link here.

---

## 7. References

| Component | Source |
|---|---|
| AST (Audio Spectrogram Transformer) | HuggingFace `ASTForAudioClassification` — https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer |
| MobileNetV2 | `timm` — https://github.com/huggingface/pytorch-image-models |
| ESC-50 (non-cooking source audio) | https://github.com/karolpiczak/ESC-50 — **CC BY-NC 3.0 (non-commercial)** |

> **License note:** the non-cooking clips derive from ESC-50, which is licensed
> CC BY-NC 3.0. Any downstream commercial use requires sourcing replacement
> negative audio under appropriate licensing.

---

## 8. Reproducing Training

```bash
# 1. Regenerate the dataset (or download it from the Kaggle link in section 4.1)
#    Requires the raw_sources soundbank; seed 1337 reproduces the published build.
pip install 'kitpri[datagen]'
python training/dataset_creation.py --soundbank /path/to/raw_sources --out kitpri_v4_build

# 2. Train the AST teacher
python training/train_ast.py

# 3. Distil into MobileNetV2
python training/distill_mobilenet.py

# 4. Quantize to INT8
python training/quantize.py
```
