#!/usr/bin/env python3
"""
Evaluate a model on a metadata split (default: test) using the unified
Predictor — identical preprocessing to deployment by construction.

    python scripts/evaluate.py --model inference/student_mobilenet_fp32.pt \
        --config configs/experiments/distill.yaml --split test
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kitpri.config import load_config          # noqa: E402
from kitpri.data.dataset import load_metadata  # noqa: E402
from kitpri.eval import binary_metrics         # noqa: E402
from kitpri.inference import Predictor         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True, help="experiment config (data + audio + model)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    predictor = Predictor(args.model, config_path=args.config, threshold=args.threshold)
    items = load_metadata(cfg.data, args.split)

    probs, labels = [], []
    for it in items:
        if not it["path"].exists():
            sys.exit(f"ERROR: metadata references missing audio: {it['path']}")
        probs.append(predictor.predict_file(str(it["path"]))["probability"])
        labels.append(it["label"])

    m = binary_metrics(probs, labels, threshold=predictor.threshold)
    print(json.dumps({"split": args.split, "n": len(items),
                      "threshold": predictor.threshold, **m}, indent=2))


if __name__ == "__main__":
    main()
