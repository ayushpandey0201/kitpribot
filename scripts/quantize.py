#!/usr/bin/env python3
"""
Quantize the FP32 student to INT8 (static PTQ) with a config-driven backend,
then export TorchScript. Backend MUST match the deployment target's CPU:
qnnpack for the ARM SmartThings target, fbgemm/x86 for dev machines.

    KITPRI_DATA_ROOT=~/Downloads/kitpri_v3_data \
    python scripts/quantize.py --config configs/experiments/quantize.yaml \
        --student inference/student_mobilenet_fp32.pt --out runs/int8
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kitpri.audio import expected_feature_shape                        # noqa: E402
from kitpri.config import load_config, validate_audio_config, save_config  # noqa: E402
from kitpri.data import KitPriDataset                                  # noqa: E402
from kitpri.deploy import export_torchscript, quantize_static_ptq      # noqa: E402
from kitpri.models import build_model                                  # noqa: E402
from kitpri.seeding import set_seed                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiments/quantize.yaml")
    ap.add_argument("--student", required=True, help="FP32 student checkpoint")
    ap.add_argument("--calib-split", default="val")
    ap.add_argument("--out", default="runs/int8")
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_audio_config(cfg)
    set_seed(int(cfg.get("seed", 42)))

    model = build_model(cfg.model)
    ckpt = torch.load(args.student, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)

    ds = KitPriDataset(cfg, args.calib_split)
    loader = torch.utils.data.DataLoader(ds, batch_size=16)
    calib = (x for x, _ in loader)

    int8 = quantize_static_ptq(model, calib, cfg.quantization)

    example = torch.zeros(1, *expected_feature_shape(cfg.audio))
    backend = cfg.quantization.backend
    out = os.path.join(args.out, f"student_mobilenet_int8_{backend}_scripted.pt")
    export_torchscript(int8, example, out)
    save_config(cfg, os.path.join(args.out, "run_config.yaml"))


if __name__ == "__main__":
    main()
