"""Training callbacks: early stopping and checkpointing (verified ckpt format)."""

from __future__ import annotations

from pathlib import Path

import torch


class EarlyStopping:
    """Stop when the monitored metric hasn't improved for `patience` epochs."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("-inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        if value > self.best:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class CheckpointSaver:
    """
    Writes checkpoints in the VERIFIED v4 format:
        {"model_state": ..., "epoch": int, "val_f1": float}
    plus a full resume checkpoint (optimizer/scheduler state) alongside.
    """

    def __init__(self, out_dir: str, name: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.best_val = float("-inf")

    def save_best(self, model, epoch: int, val_f1: float) -> bool:
        if val_f1 <= self.best_val:
            return False
        self.best_val = val_f1
        torch.save(
            {"model_state": model.state_dict(), "epoch": epoch, "val_f1": val_f1},
            self.out_dir / f"{self.name}_best.pt",
        )
        return True

    def save_resume(self, model, optimizer, scheduler, epoch: int) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
            },
            self.out_dir / f"{self.name}_resume.pt",
        )
