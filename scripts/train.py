#!/usr/bin/env python3
"""
Train the AST teacher (runs on Kaggle — dataset is remote).

    python scripts/train.py --config configs/experiments/train_teacher.yaml [--resume PATH]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kitpri.config import load_config, validate_audio_config  # noqa: E402
from kitpri.models import build_model                          # noqa: E402
from kitpri.training import Trainer                            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/train_teacher.yaml")
    ap.add_argument("--run-name", default="ast_teacher")
    ap.add_argument("--resume", default=None, help="path to *_resume.pt checkpoint")
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_audio_config(cfg)
    trainer = Trainer(cfg, build_model(cfg.model), run_name=args.run_name)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
