#!/usr/bin/env python3
"""
KitPri v4 — Knowledge distillation AST -> MobileNetV2 (STUB).

STATUS: The original implementation lives in the v4 Kaggle notebook and is NOT
present on this machine. This stub records the verified configuration the real
script must match. Port the notebook code; do not re-derive.

Verified facts the real implementation must satisfy
---------------------------------------------------
- Student: `timm.create_model('mobilenetv2_100', num_classes=1)` — 2,225,153
  params. Loads with strict=True from checkpoint key `model_state`.
- Distillation: temperature T=3.0, alpha=0.4, teacher = kitpri_v4_ast_diagnostic.
- Preprocessing identical to train_ast.py / inference/predict.py (32 kHz, 10 s,
  128 mels, n_fft 1024, hop 512, per-clip norm, no resize).
- Reference result: best epoch 8/14, test F1 0.7226 @ thr 0.50,
  0.7318 @ thr 0.44 (see results/kitpri_v4_distilled_mobilenet/).

Usage (target interface):
    python distill_mobilenet.py --data ./kitpri_v4_data \
        --teacher ./runs/ast/ast_teacher.pt --out ./runs/student \
        --temperature 3.0 --alpha 0.4
"""

raise SystemExit(
    "STUB: distill_mobilenet.py is not yet ported from the v4 Kaggle notebook. "
    "See the module docstring for the verified configuration."
)
