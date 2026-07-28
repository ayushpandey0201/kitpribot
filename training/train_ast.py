#!/usr/bin/env python3
"""
KitPri v4 — AST teacher training (STUB).

STATUS: The original implementation lives in the v4 Kaggle notebook and is NOT
present on this machine. This stub records the verified configuration the real
script must match. Port the notebook code; do not re-derive.

Verified facts the real implementation must satisfy
---------------------------------------------------
- Model: HuggingFace `ASTForAudioClassification` (86.2 M params), single output
  logit; sigmoid(logit) = P(cooking).
- Preprocessing (MUST match inference/predict.py exactly):
    SAMPLE_RATE=32000, CLIP_DURATION=10.0, N_MELS=128, N_FFT=1024,
    HOP_LENGTH=512, AmplitudeToDB(top_db=80), PER-CLIP mean/std normalization,
    3-channel repeat, natural 128x626 shape (NO resize, NO ImageNet normalize).
- Checkpoint format: dict with keys `model_state`, `epoch`, `val_f1`.
- Reference result: best epoch 1 (early stopped @ 6), test F1 0.8129,
  acc 0.8200, AUC 0.8976 (see results/kitpri_v4_ast_diagnostic/).

Usage (target interface):
    python train_ast.py --data ./kitpri_v4_data --out ./runs/ast --epochs 20
"""

raise SystemExit(
    "STUB: train_ast.py is not yet ported from the v4 Kaggle notebook. "
    "See the module docstring for the verified configuration."
)
