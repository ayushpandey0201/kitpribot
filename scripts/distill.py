#!/usr/bin/env python3
"""
Distil the AST teacher into the MobileNetV2 student (runs on Kaggle).

    python scripts/distill.py --config configs/experiments/distill.yaml [--resume PATH]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kitpri.config import Config, load_config, validate_audio_config  # noqa: E402
from kitpri.models import build_model                                  # noqa: E402
from kitpri.training import Trainer                                    # noqa: E402
from kitpri.training.distill import DistillationLoss                   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/distill.yaml")
    ap.add_argument("--run-name", default="distilled_student")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_audio_config(cfg)

    # Teacher: architecture auto-identified from checkpoint keys by Predictor
    # conventions; here we build AST explicitly and load the referenced ckpt.
    teacher = build_model(Config({"name": "ast", "num_classes": 1}))
    t_ckpt = torch.load(cfg.distillation.teacher_checkpoint,
                        map_location="cpu", weights_only=False)
    teacher.load_state_dict(t_ckpt.get("model_state", t_ckpt), strict=True)

    student = build_model(cfg.model)
    loss = DistillationLoss(teacher, cfg.distillation.temperature, cfg.distillation.alpha)

    class _HookedLoader:
        """Feeds each batch to the distillation loss (teacher forward) before yielding."""

        def __init__(self, loader, dloss):
            self.loader, self.dloss = loader, dloss
            self.dataset = loader.dataset

        def __iter__(self):
            teacher_device = next(self.dloss.teacher.parameters()).device
            for x, y in self.loader:
                self.dloss.set_inputs(x.to(teacher_device))
                yield x, y

        def __len__(self):
            return len(self.loader)

    class DistillTrainer(Trainer):
        def _epoch(self, loader, train: bool):
            return super()._epoch(_HookedLoader(loader, loss), train)

    trainer = DistillTrainer(cfg, student, run_name=args.run_name, loss_fn=loss)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
