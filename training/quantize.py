"""
KitPri v4 — Static INT8 quantization (with calibration) for the distilled
MobileNetV2 student.

Per project history: dynamic INT8 quantization collapsed student F1 to near-
zero on v2 — diagnosed as wrong method for CNNs (dynamic only quantizes
weights, causing activation range mismatch at inference). Static quantization
with a calibration pass on real data is the correct approach for CNN
inference and is what this script does.

Steps:
1. Load the FP32 distilled MobileNetV2 checkpoint
2. Prepare the model for static quantization (FX graph mode — no manual
   fusion lists needed, unlike the older eager-mode API)
3. Run a calibration pass over a subset of TRAIN data (observes real
   activation ranges — this is the step that was missing before)
4. Convert to true INT8
5. Evaluate FP32 vs INT8 side by side on val + test to confirm no collapse
6. Export INT8 model + report file size

Usage:
    modal run training/quantize.py
"""

import modal

app = modal.App("kitpri-v4-quantize")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "timm",
        "pandas",
        "numpy",
        "scikit-learn",
        "soundfile",
        "tqdm",
    )
)

data_vol = modal.Volume.from_name("kitpri-v4-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("kitpri-checkpoints", create_if_missing=True)

DATA_ROOT = "/vol/data"
CKPT_ROOT = "/vol/checkpoints"
STUDENT_RUN = "kitpri_v4_distilled_mobilenet"


@app.function(
    image=image,
    cpu=4.0,  # static quantization + calibration run on CPU (torch quantization is CPU-only)
    volumes={DATA_ROOT: data_vol, CKPT_ROOT: ckpt_vol},
    timeout=60 * 30,
)
def quantize():
    import os
    import json
    import copy
    import time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import soundfile as sf
    import torchaudio
    import timm
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    from tqdm import tqdm

    SAMPLE_RATE = 32000
    CLIP_DURATION = 10.0
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512
    NUM_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

    DATA_DIR = os.path.join(DATA_ROOT, "kitpri_v4")
    STUDENT_DIR = os.path.join(CKPT_ROOT, STUDENT_RUN)

    device = torch.device("cpu")
    print("Device:", device, "(static quantization requires CPU)")

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    class EvalDataset(Dataset):
        def __init__(self, csv_path, root_dir, limit=None):
            self.df = pd.read_csv(csv_path)
            if limit:
                self.df = self.df.sample(n=min(limit, len(self.df)), random_state=1337).reset_index(drop=True)
            self.root_dir = root_dir

        def __len__(self):
            return len(self.df)

        def _load_wave(self, path):
            wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav_np.T)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            n = wav.shape[1]
            if n < NUM_SAMPLES:
                wav = torch.nn.functional.pad(wav, (0, NUM_SAMPLES - n))
            elif n > NUM_SAMPLES:
                wav = wav[:, :NUM_SAMPLES]
            return wav

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            path = os.path.join(self.root_dir, row["rel_path"])
            wav = self._load_wave(path)
            mel = mel_transform(wav)
            mel_db = db_transform(mel)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            mel_db = mel_db.repeat(3, 1, 1)
            label = float(row["label"])
            return mel_db, label, row["file_id"]

    train_csv = os.path.join(DATA_DIR, "metadata", "train.csv")
    val_csv = os.path.join(DATA_DIR, "metadata", "val.csv")
    test_csv = os.path.join(DATA_DIR, "metadata", "test.csv")

    calib_ds = EvalDataset(train_csv, DATA_DIR, limit=500)
    val_ds = EvalDataset(val_csv, DATA_DIR)
    test_ds = EvalDataset(test_csv, DATA_DIR)

    calib_loader = DataLoader(calib_ds, batch_size=16, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    base_model = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=1, in_chans=3)
    ckpt = torch.load(os.path.join(STUDENT_DIR, "best_model.pt"), map_location=device)
    base_model.load_state_dict(ckpt["model_state"])
    base_model.eval()
    print(f"Loaded FP32 student from {STUDENT_RUN}, epoch={ckpt.get('epoch')}, val_f1={ckpt.get('val_f1'):.4f}")

    fp32_size = sum(p.numel() * p.element_size() for p in base_model.parameters()) / (1024 ** 2)
    print(f"FP32 model size (params only): {fp32_size:.2f} MB")

    def evaluate(model, loader, label=""):
        all_probs, all_labels = [], []
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            for mel, lbl, _ in tqdm(loader, leave=False, desc=label):
                logits = model(mel).squeeze(1)
                probs = torch.sigmoid(logits).numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(lbl.numpy().tolist())
        dt = time.time() - t0
        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        f1 = f1_score(all_labels, preds, zero_division=0)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        acc = accuracy_score(all_labels, preds)
        avg_ms_per_clip = (dt / len(loader.dataset)) * 1000
        return dict(f1=f1, precision=prec, recall=rec, accuracy=acc,
                    inference_time_sec=dt, avg_ms_per_clip=avg_ms_per_clip)

    print("\n===== FP32 baseline evaluation =====")
    fp32_val = evaluate(base_model, val_loader, "FP32 val")
    fp32_test = evaluate(base_model, test_loader, "FP32 test")
    print(f"FP32 val:  F1={fp32_val['f1']:.4f}  ({fp32_val['avg_ms_per_clip']:.2f} ms/clip)")
    print(f"FP32 test: F1={fp32_test['f1']:.4f}  ({fp32_test['avg_ms_per_clip']:.2f} ms/clip)")

    model_to_quantize = copy.deepcopy(base_model)
    model_to_quantize.eval()

    from torch.ao.quantization import quantize_fx
    from torch.ao.quantization import QConfigMapping, get_default_qconfig

    qconfig_mapping = QConfigMapping().set_global(
        get_default_qconfig("fbgemm")
    )

    n_mel_frames = NUM_SAMPLES // HOP_LENGTH + 1
    example_inputs = (torch.randn(1, 3, N_MELS, n_mel_frames),)

    print("\nPreparing model for static quantization (FX graph mode)...")
    prepared_model = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)

    print(f"Running calibration pass on {len(calib_ds)} training clips...")
    prepared_model.eval()
    with torch.no_grad():
        for mel, _, _ in tqdm(calib_loader, desc="Calibration"):
            prepared_model(mel)

    print("Converting to INT8...")
    quantized_model = quantize_fx.convert_fx(prepared_model)

    print("\n===== INT8 quantized evaluation =====")
    int8_val = evaluate(quantized_model, val_loader, "INT8 val")
    int8_test = evaluate(quantized_model, test_loader, "INT8 test")
    print(f"INT8 val:  F1={int8_val['f1']:.4f}  ({int8_val['avg_ms_per_clip']:.2f} ms/clip)")
    print(f"INT8 test: F1={int8_test['f1']:.4f}  ({int8_test['avg_ms_per_clip']:.2f} ms/clip)")

    quant_path = os.path.join(STUDENT_DIR, "student_mobilenet_int8.pt")
    torch.save(quantized_model.state_dict(), quant_path)
    int8_file_size = os.path.getsize(quant_path) / (1024 ** 2)
    print(f"\nINT8 model file size: {int8_file_size:.2f} MB (target budget: 60 MB)")

    try:
        scripted = torch.jit.script(quantized_model)
        script_path = os.path.join(STUDENT_DIR, "student_mobilenet_int8_scripted.pt")
        scripted.save(script_path)
        print(f"Saved TorchScript version to student_mobilenet_int8_scripted.pt")
    except Exception as e:
        print(f"TorchScript export failed (non-fatal, state_dict still saved): {e}")

    f1_drop = fp32_test["f1"] - int8_test["f1"]
    speedup = fp32_test["avg_ms_per_clip"] / int8_test["avg_ms_per_clip"] if int8_test["avg_ms_per_clip"] > 0 else float("nan")

    summary = dict(
        fp32_val=fp32_val, fp32_test=fp32_test,
        int8_val=int8_val, int8_test=int8_test,
        fp32_size_mb=fp32_size,
        int8_size_mb=int8_file_size,
        size_reduction_pct=(1 - int8_file_size / fp32_size) * 100 if fp32_size > 0 else None,
        f1_drop_test=f1_drop,
        cpu_speedup=speedup,
        calibration_samples=len(calib_ds),
    )
    with open(os.path.join(STUDENT_DIR, "quantization_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== QUANTIZATION SUMMARY =====")
    print(json.dumps(summary, indent=2))

    if f1_drop > 0.05:
        print(f"\nWARNING: F1 dropped by {f1_drop:.4f} after quantization — this is a meaningful "
              f"accuracy loss, worth reviewing before deployment.")
    else:
        print(f"\nF1 drop after quantization: {f1_drop:.4f} — within acceptable range, "
              f"static quantization with calibration succeeded (no collapse like the dynamic-quant attempt).")

    ckpt_vol.commit()
    print(f"\nSaved to volume kitpri-checkpoints:/{STUDENT_RUN}/")
    print("  - student_mobilenet_int8.pt (state dict)")
    print("  - student_mobilenet_int8_scripted.pt (TorchScript, if successful)")
    print("  - quantization_report.json")


@app.local_entrypoint()
def main():
    quantize.remote()
