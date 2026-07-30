#!/usr/bin/env python3
"""
Regenerate the v4 synthetic dataset (requires the raw_sources soundbank).

    python scripts/create_dataset.py --config configs/base.yaml --seed 42

STATUS: delegates to kitpri.data.synthesis, which is an honest stub — the
WORKING generator is training/dataset_creation.py (the original script that
built the published dataset; use that one). Seeding is mandatory so
regeneration is byte-identical to the clips behind the reported metrics.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kitpri.config import load_config      # noqa: E402
from kitpri.data.synthesis import synthesize_dataset  # noqa: E402
from kitpri.seeding import set_seed        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=None, help="overrides config seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed if args.seed is not None else int(cfg.get("seed", 42)))
    synthesize_dataset(cfg)


if __name__ == "__main__":
    main()
