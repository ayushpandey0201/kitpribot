"""
KitPri v4 — Knowledge distillation: AST (teacher) -> MobileNetV2 (student).

Teacher: kitpri_v4_ast_diagnostic/best_model.pt (test_f1=0.8129, val_f1=0.8233)
Student: MobileNetV2, trained on a mix of hard labels + teacher soft labels.

Distillation loss = alpha * BCE(student_logits, hard_label)
                   + (1-alpha) * BCE(student_logits, soft_label_from_teacher)
where soft_label_from_teacher = sigmoid(teacher_logits / T)

This matches the same recipe used successfully in v2 (T=3.0, alpha=0.4).

Usage:
    modal run training/distill_mobilenet.py
"""

import modal

app = modal.App("kitpri-v4-distill")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "timm",
        "transformers",
        "peft",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "soundfile",
        "tqdm",
    )
)

data_vol = modal.Volume.from_name("kitpri-v4-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("kitpri-checkpoints", create_if_missing=True)

DATA_ROOT = "/vol/data"
CKPT_ROOT = "/vol/checkpoints"
RUN_NAME = "kitpri_v4_distilled_mobilenet"
TEACHER_RUN = "kitpri_v4_ast_diagnostic"

# distillation hyperparams — same recipe as v2 (worked well there)
TEMPERATURE = 3.0
ALPHA = 0.4  # weight on hard-label loss; (1-alpha) on soft-label loss


@app.function(
    image=image,
    gpu="A10G",
    volumes={DATA_ROOT: data_vol, CKPT_ROOT: ckpt_vol},
    timeout=60 * 60 * 6,
)
def distill():
    import os
    import json
    import time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import soundfile as sf
    import torchaudio
    import timm
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, accuracy_score,
        confusion_matrix, classification_report, roc_auc_score,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm

    SAMPLE_RATE = 32000
    CLIP_DURATION = 10.0
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512
    NUM_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)
    AST_SR = 16000

    DATA_DIR = os.path.join(DATA_ROOT, "kitpri_v4")
    TEACHER_DIR = os.path.join(CKPT_ROOT, TEACHER_RUN)
    OUT_DIR = os.path.join(CKPT_ROOT, RUN_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    SEED = 1337
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---------------- load teacher (AST, frozen, eval mode) ----------------
    AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
    teacher = ASTForAudioClassification.from_pretrained(
        AST_MODEL_NAME, num_labels=1, ignore_mismatched_sizes=True,
        problem_type="regression",
    )
    teacher_ckpt = torch.load(os.path.join(TEACHER_DIR, "best_model.pt"), map_location=device)
    teacher.load_state_dict(teacher_ckpt["model_state"])
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Loaded teacher from {TEACHER_RUN}, checkpoint epoch={teacher_ckpt.get('epoch')}, val_f1={teacher_ckpt.get('val_f1'):.4f}")

    ast_feature_extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL_NAME)

    # ---------------- dataset: produces BOTH student input (mel-as-image)
    # AND teacher input (raw 16kHz waveform for AST feature extractor) ----------------
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=27)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=80)

    class DistillDataset(Dataset):
        def __init__(self, csv_path, root_dir, train_mode=False):
            self.df = pd.read_csv(csv_path)
            self.root_dir = root_dir
            self.train_mode = train_mode

        def __len__(self):
            return len(self.df)

        def _load_wave_native(self, path):
            wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav_np.T)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav, sr

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            path = os.path.join(self.root_dir, row["rel_path"])
            wav, sr = self._load_wave_native(path)

            # --- student path: 32kHz mel spectrogram, EfficientNet/MobileNet style ---
            wav_32k = wav
            if sr != SAMPLE_RATE:
                wav_32k = torchaudio.functional.resample(wav_32k, sr, SAMPLE_RATE)
            n = wav_32k.shape[1]
            if n < NUM_SAMPLES:
                wav_32k = torch.nn.functional.pad(wav_32k, (0, NUM_SAMPLES - n))
            elif n > NUM_SAMPLES:
                wav_32k = wav_32k[:, :NUM_SAMPLES]

            mel = mel_transform(wav_32k)
            mel_db = db_transform(mel)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            if self.train_mode:
                mel_db = freq_mask(mel_db)
                mel_db = time_mask(mel_db)
            mel_db = mel_db.repeat(3, 1, 1)

            # --- teacher path: 16kHz raw waveform, AST feature extractor handles the rest ---
            wav_16k = wav
            if sr != AST_SR:
                wav_16k = torchaudio.functional.resample(wav_16k, sr, AST_SR)
            target_n = int(AST_SR * CLIP_DURATION)
            n16 = wav_16k.shape[1]
            if n16 < target_n:
                wav_16k = torch.nn.functional.pad(wav_16k, (0, target_n - n16))
            elif n16 > target_n:
                wav_16k = wav_16k[:, :target_n]
            wav_16k_np = wav_16k.squeeze(0).numpy()

            label = float(row["label"])
            return mel_db, wav_16k_np, label, row["file_id"]

    def collate_fn(batch):
        mels, wavs16k, labels, ids = zip(*batch)
        mels = torch.stack(mels)
        labels = torch.tensor(labels, dtype=torch.float32)
        ast_inputs = ast_feature_extractor(list(wavs16k), sampling_rate=AST_SR, return_tensors="pt")
        return mels, ast_inputs["input_values"], labels, ids

    train_csv = os.path.join(DATA_DIR, "metadata", "train.csv")
    val_csv = os.path.join(DATA_DIR, "metadata", "val.csv")
    test_csv = os.path.join(DATA_DIR, "metadata", "test.csv")

    train_ds = DistillDataset(train_csv, DATA_DIR, train_mode=True)
    val_ds = DistillDataset(val_csv, DATA_DIR, train_mode=False)
    test_ds = DistillDataset(test_csv, DATA_DIR, train_mode=False)

    BATCH_SIZE = 16  # smaller batch since teacher (AST) forward pass runs alongside
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ---------------- student: MobileNetV2 ----------------
    student = timm.create_model("mobilenetv2_100", pretrained=True, num_classes=1, in_chans=3, drop_rate=0.2)
    student = student.to(device)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student (MobileNetV2) params: {student_params:,}")

    hard_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)

    EPOCHS = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    PATIENCE = 6
    best_val_f1 = -1
    epochs_no_improve = 0
    best_epoch = -1
    history = []

    def distill_loss(student_logits, teacher_logits, hard_labels):
        hard_loss = hard_criterion(student_logits, hard_labels)
        with torch.no_grad():
            teacher_soft = torch.sigmoid(teacher_logits / TEMPERATURE)
        soft_loss = F.binary_cross_entropy_with_logits(
            student_logits / TEMPERATURE, teacher_soft
        ) * (TEMPERATURE ** 2)
        return ALPHA * hard_loss + (1 - ALPHA) * soft_loss

    def run_epoch(loader, train_mode):
        if train_mode:
            student.train()
        else:
            student.eval()

        total_loss = 0.0
        all_probs, all_labels = [], []

        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            for mel, ast_input, label, _ in tqdm(loader, leave=False):
                mel = mel.to(device)
                ast_input = ast_input.to(device)
                label = label.to(device).float()

                with torch.no_grad():
                    teacher_out = teacher(input_values=ast_input)
                    teacher_logits = teacher_out.logits.squeeze(-1)

                student_logits = student(mel).squeeze(1)
                loss = distill_loss(student_logits, teacher_logits, label)

                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                    optimizer.step()

                total_loss += loss.item() * mel.size(0)
                probs = torch.sigmoid(student_logits).detach().cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(label.detach().cpu().numpy().tolist())

        avg_loss = total_loss / len(loader.dataset)
        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float("nan")
        return avg_loss, acc, f1, prec, rec, auc

    print(f"\n===== DISTILLATION START (T={TEMPERATURE}, alpha={ALPHA}) =====\n")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1, train_prec, train_rec, train_auc = run_epoch(train_loader, train_mode=True)
        val_loss, val_acc, val_f1, val_prec, val_rec, val_auc = run_epoch(val_loader, train_mode=False)
        scheduler.step()
        dt = time.time() - t0
        gap = train_f1 - val_f1

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_f1={train_f1:.4f} val_f1={val_f1:.4f} (gap={gap:+.4f}) | "
            f"val_prec={val_prec:.4f} val_rec={val_rec:.4f} val_auc={val_auc:.4f} | "
            f"{dt:.1f}s"
        )

        history.append(dict(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            train_acc=train_acc, val_acc=val_acc, train_f1=train_f1, val_f1=val_f1,
            train_precision=train_prec, val_precision=val_prec,
            train_recall=train_rec, val_recall=val_rec,
            train_auc=train_auc, val_auc=val_auc,
            train_val_f1_gap=gap, lr=optimizer.param_groups[0]["lr"], time_sec=dt,
        ))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({"model_state": student.state_dict(), "epoch": epoch, "val_f1": val_f1},
                       os.path.join(OUT_DIR, "best_model.pt"))
            print(f"  -> new best val_f1={val_f1:.4f}, checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "training_log.csv"), index=False)
        ckpt_vol.commit()

    print(f"\nBest epoch: {best_epoch} | Best val_f1: {best_val_f1:.4f}\n")

    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
    axes[0].set_title("Distillation Loss vs Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(hist_df["epoch"], hist_df["train_f1"], label="train_f1")
    axes[1].plot(hist_df["epoch"], hist_df["val_f1"], label="val_f1")
    axes[1].axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    axes[1].set_title("Student F1 vs Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(hist_df["epoch"], hist_df["train_val_f1_gap"], color="red")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Train-Val F1 Gap"); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close()

    print("\n===== TEST SET EVALUATION (best student checkpoint) =====\n")
    ckpt = torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location=device)
    student.load_state_dict(ckpt["model_state"])
    student.eval()

    all_probs, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for mel, ast_input, label, fid in tqdm(test_loader):
            mel = mel.to(device)
            logits = student(mel).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(label.numpy().tolist())
            all_ids.extend(list(fid))

    all_preds = [1 if p >= 0.5 else 0 for p in all_probs]
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, zero_division=0)
    test_rec = recall_score(all_labels, all_preds, zero_division=0)
    test_auc = roc_auc_score(all_labels, all_probs)

    report = classification_report(all_labels, all_preds, target_names=["noncooking", "cooking"], digits=4)
    print(report)
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Teacher: {TEACHER_RUN} (test_f1=0.8129)\n")
        f.write(f"Distillation: T={TEMPERATURE}, alpha={ALPHA}\n")
        f.write(f"Best epoch: {best_epoch}\nBest val_f1: {best_val_f1:.4f}\n\n")
        f.write(f"TEST accuracy:  {test_acc:.4f}\nTEST f1:        {test_f1:.4f}\n")
        f.write(f"TEST precision: {test_prec:.4f}\nTEST recall:    {test_rec:.4f}\nTEST auc:       {test_auc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=["true_noncooking", "true_cooking"], columns=["pred_noncooking", "pred_cooking"])
    cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["noncooking", "cooking"], yticklabels=["noncooking", "cooking"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion Matrix — Distilled MobileNetV2 Test Set (F1={test_f1:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    pred_df = pd.DataFrame({
        "file_id": all_ids, "true_label": all_labels, "pred_label": all_preds,
        "pred_prob": all_probs, "correct": [int(t == p) for t, p in zip(all_labels, all_preds)],
    })
    pred_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    summary = dict(
        run_name=RUN_NAME, teacher_run=TEACHER_RUN, temperature=TEMPERATURE, alpha=ALPHA,
        best_epoch=best_epoch, best_val_f1=best_val_f1,
        total_epochs_run=len(history), early_stopped=len(history) < EPOCHS,
        test_accuracy=test_acc, test_f1=test_f1, test_precision=test_prec,
        test_recall=test_rec, test_auc=test_auc,
        train_val_f1_gap_at_best_epoch=history[best_epoch - 1]["train_val_f1_gap"],
        student_params=student_params,
        num_train=len(train_ds), num_val=len(val_ds), num_test=len(test_ds),
    )
    with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))

    ckpt_vol.commit()
    print(f"\nAll artifacts saved to volume kitpri-checkpoints:/{RUN_NAME}/")


@app.local_entrypoint()
def main():
    distill.remote()
