"""
Generic config-driven trainer for KitPri models.

Execution venue is Kaggle (dataset is remote) — locally this module is only
verified importable. Supports --resume: restores model, optimizer, scheduler,
and epoch (Kaggle sessions time out).

NOTE: this is forward-looking infrastructure. The exact v4 notebook training
loop must be checked against this when training/ is ported; hyperparameters
live in configs/experiments/, reference outcomes in results/.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kitpri.config import save_config
from kitpri.data import KitPriDataset
from kitpri.eval.metrics import binary_metrics
from kitpri.seeding import set_seed
from kitpri.training.callbacks import CheckpointSaver, EarlyStopping


class Trainer:
    def __init__(self, cfg, model, run_name: str, loss_fn=None):
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or torch.nn.BCEWithLogitsLoss()

        t = cfg.training
        self.epochs = int(t.epochs)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(t.lr),
            weight_decay=float(t.get("weight_decay", 0.0)))
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
            if t.get("scheduler") == "cosine" else None)
        self.stopper = EarlyStopping(int(t.get("early_stopping_patience", 5)))
        self.out_dir = Path(cfg.get("output_dir", "runs")) / run_name
        self.saver = CheckpointSaver(self.out_dir, run_name)
        self.start_epoch = 1

    def resume(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler and ckpt.get("scheduler_state"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.start_epoch = int(ckpt["epoch"]) + 1
        print(f"[trainer] Resumed from {path} at epoch {self.start_epoch}")

    def _epoch(self, loader, train: bool):
        self.model.train(train)
        probs, labels, total_loss = [], [], 0.0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.loss_fn(logit, y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                probs += torch.sigmoid(logit).flatten().tolist()
                labels += y.flatten().tolist()
        m = binary_metrics(probs, labels, threshold=0.5)
        m["loss"] = total_loss / len(loader.dataset)
        return m

    def fit(self) -> Path:
        set_seed(int(self.cfg.get("seed", 42)))
        save_config(self.cfg, self.out_dir / "run_config.yaml")

        bs = int(self.cfg.training.batch_size)
        train_loader = DataLoader(KitPriDataset(self.cfg, "train"), batch_size=bs, shuffle=True)
        val_loader = DataLoader(KitPriDataset(self.cfg, "val"), batch_size=bs)

        for epoch in range(self.start_epoch, self.epochs + 1):
            tr = self._epoch(train_loader, train=True)
            va = self._epoch(val_loader, train=False)
            if self.scheduler:
                self.scheduler.step()
            print(f"epoch {epoch}: train_f1={tr['f1']:.4f} val_f1={va['f1']:.4f} "
                  f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f}")
            self.saver.save_best(self.model, epoch, va["f1"])
            self.saver.save_resume(self.model, self.optimizer, self.scheduler, epoch)
            if self.stopper.step(va["f1"]):
                print(f"[trainer] Early stopping at epoch {epoch} "
                      f"(best val_f1={self.saver.best_val:.4f})")
                break
        return self.out_dir
