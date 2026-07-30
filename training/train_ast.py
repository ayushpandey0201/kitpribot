"""
KitPri v4 — AST (Audio Spectrogram Transformer) diagnostic run.

Purpose: EfficientNet-B0 has plateaued at val_f1 ~0.72-0.74 across three
regularization configs (overfit / underfit / balanced) on the v4 dataset.
This run checks whether AST hits a similar ceiling (= data problem, stop
tuning architecture) or does meaningfully better (= EfficientNet-B0
specific issue on this data).

Uses HuggingFace transformers ASTForAudioClassification + ASTFeatureExtractor,
same as the original True AST comparison noted in project history.

PROVENANCE NOTE (repo): the committed teacher (results/kitpri_v4_ast_diagnostic/,
a plain full-model checkpoint loaded directly by distill_mobilenet.py) came from
an earlier full fine-tune iteration of this script (lr 1e-4). The file preserved
here is its final evolution, which switched to LoRA adapters as a follow-up
experiment (RUN_NAME kitpri_v4_ast_lora).

Usage:
    modal run train_ast_v4.py
"""

import modal

app = modal.App("kitpri-v4-ast-diagnostic")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchaudio",
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
RUN_NAME = "kitpri_v4_ast_lora"


@app.function(
    image=image,
    gpu="A10G",
    volumes={DATA_ROOT: data_vol, CKPT_ROOT: ckpt_vol},
    timeout=60 * 60 * 6,
)
def train():
    import os
    import json
    import time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import soundfile as sf
    import torchaudio
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm

    SAMPLE_RATE = 32000
    CLIP_DURATION = 10.0
    NUM_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)
    AST_SR = 16000  # AST pretrained model expects 16kHz

    DATA_DIR = os.path.join(DATA_ROOT, "kitpri_v4")

    OUT_DIR = os.path.join(CKPT_ROOT, RUN_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    SEED = 1337
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL_NAME)

    class KitPriASTDataset(Dataset):
        def __init__(self, csv_path, root_dir):
            self.df = pd.read_csv(csv_path)
            self.root_dir = root_dir

        def __len__(self):
            return len(self.df)

        def _load_wave(self, path):
            wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav_np.T)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != AST_SR:
                wav = torchaudio.functional.resample(wav, sr, AST_SR)
            n = wav.shape[1]
            target_n = int(AST_SR * CLIP_DURATION)
            if n < target_n:
                wav = torch.nn.functional.pad(wav, (0, target_n - n))
            elif n > target_n:
                wav = wav[:, :target_n]
            return wav.squeeze(0).numpy()

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            path = os.path.join(self.root_dir, row["rel_path"])
            wav = self._load_wave(path)
            label = float(row["label"])
            return wav, label, row["file_id"]

    def collate_fn(batch):
        wavs, labels, ids = zip(*batch)
        inputs = feature_extractor(
            list(wavs), sampling_rate=AST_SR, return_tensors="pt"
        )
        labels = torch.tensor(labels, dtype=torch.float32)
        return inputs["input_values"], labels, ids

    train_csv = os.path.join(DATA_DIR, "metadata", "train.csv")
    val_csv = os.path.join(DATA_DIR, "metadata", "val.csv")
    test_csv = os.path.join(DATA_DIR, "metadata", "test.csv")

    train_ds = KitPriASTDataset(train_csv, DATA_DIR)
    val_ds = KitPriASTDataset(val_csv, DATA_DIR)
    test_ds = KitPriASTDataset(test_csv, DATA_DIR)

    BATCH_SIZE = 16  # AST is heavier than EfficientNet-B0, smaller batch
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = ASTForAudioClassification.from_pretrained(
        AST_MODEL_NAME,
        num_labels=1,
        ignore_mismatched_sizes=True,
        problem_type="regression",  # single logit, we apply BCEWithLogits ourselves
    )
    model = model.to(device)

    from peft import LoraConfig, get_peft_model

    # confirm actual attention projection names before targeting them —
    # a guessed pattern silently failed earlier in this same script (freeze
    # logic missed ".encoder.layer." because this transformers version uses
    # ".layers." and "q_proj"/"k_proj"/"v_proj"), so verify instead of assuming
    all_names = [n for n, _ in model.named_parameters()]
    sample_attn_names = [n for n in all_names if "attention" in n.lower()][:8]
    print("Sample attention param names:")
    for n in sample_attn_names:
        print(" ", n)

    # detect whether this checkpoint uses q_proj/k_proj/v_proj or query/key/value
    if any("q_proj" in n for n in all_names):
        lora_targets = ["q_proj", "k_proj", "v_proj"]
    else:
        lora_targets = ["query", "key", "value"]
    print(f"Using LoRA target_modules: {lora_targets}")

    # LoRA instead of manual layer freezing — naive freezing (even at 49%
    # trainable) still overfit by epoch 2 in the previous run. LoRA is the
    # standard approach for fine-tuning transformers on small datasets:
    # freeze the entire backbone, inject small rank-decomposition adapters
    # into the attention projections, only train those + the classifier head.
    # This trains <1% of params vs. AST's 86M — much stronger regularization
    # than block-freezing, while every layer still gets to adapt a little.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=lora_targets,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    if trainable / total > 0.1:
        print("WARNING: LoRA trainable % higher than expected — check target_modules above")


    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-3
    )

    EPOCHS = 25  # LoRA converges a bit slower with fewer trainable params
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    PATIENCE = 6
    best_val_f1 = -1
    epochs_no_improve = 0
    best_epoch = -1
    history = []

    def run_epoch(loader, train_mode):
        if train_mode:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        all_probs, all_labels = [], []
        n_seen = 0

        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            for input_values, label, _ in tqdm(loader, leave=False):
                input_values = input_values.to(device)
                label = label.to(device).float()

                out = model(input_values=input_values)
                logits = out.logits.squeeze(-1)
                loss = criterion(logits, label)

                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                bs = input_values.size(0)
                total_loss += loss.item() * bs
                n_seen += bs
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(label.detach().cpu().numpy().tolist())

        avg_loss = total_loss / n_seen
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

    print("\n===== AST TRAINING START =====\n")
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
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_f1": val_f1},
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
    axes[0].set_title("Loss vs Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(hist_df["epoch"], hist_df["train_f1"], label="train_f1")
    axes[1].plot(hist_df["epoch"], hist_df["val_f1"], label="val_f1")
    axes[1].axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    axes[1].set_title("F1 vs Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(hist_df["epoch"], hist_df["train_val_f1_gap"], color="red")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Train-Val F1 Gap"); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close()

    print("\n===== TEST SET EVALUATION (best checkpoint) =====\n")
    ckpt = torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_probs, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for input_values, label, fid in tqdm(test_loader):
            input_values = input_values.to(device)
            out = model(input_values=input_values)
            logits = out.logits.squeeze(-1)
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
    plt.title(f"Confusion Matrix — AST Test Set (F1={test_f1:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    pred_df = pd.DataFrame({
        "file_id": all_ids, "true_label": all_labels, "pred_label": all_preds,
        "pred_prob": all_probs, "correct": [int(t == p) for t, p in zip(all_labels, all_preds)],
    })
    pred_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    summary = dict(
        run_name=RUN_NAME, best_epoch=best_epoch, best_val_f1=best_val_f1,
        total_epochs_run=len(history), early_stopped=len(history) < EPOCHS,
        test_accuracy=test_acc, test_f1=test_f1, test_precision=test_prec,
        test_recall=test_rec, test_auc=test_auc,
        train_val_f1_gap_at_best_epoch=history[best_epoch - 1]["train_val_f1_gap"],
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
    train.remote()
