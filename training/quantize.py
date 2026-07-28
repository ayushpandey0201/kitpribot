#!/usr/bin/env python3
"""
KitPri v4 — INT8 static post-training quantization of the student (STUB).

STATUS: The original implementation lives in the v4 Kaggle notebook and is NOT
present on this machine. This stub records the verified configuration the real
script must match. Port the notebook code; do not re-derive.

Verified facts the real implementation must satisfy
---------------------------------------------------
- Method: static PTQ with 500 calibration clips, exported as TorchScript
  (`student_mobilenet_int8_scripted.pt`, 2.80 MB, torch.jit.load-able with no
  timm dependency).
- Reference result: test F1 0.6910 @ thr 0.50, 2.25x CPU speedup vs FP32
  (25.5 ms vs 57.3 ms per clip) — see results/.../quantization_report.json.
- KNOWN ISSUE to preserve in docs: the 0.44 threshold was tuned on FP32
  validation probabilities, not re-swept on INT8 outputs. INT8 logits sit on a
  coarse grid (~0.216/step, ~5 pp near the boundary; no expressible probability
  between ~0.446 and ~0.500), so fine-grained threshold tuning is meaningless
  on this model. Future work: re-sweep threshold on INT8 validation outputs.

Usage (target interface):
    python quantize.py --student ./runs/student/student_fp32.pt \
        --calib ./kitpri_v4_data/val --out ./runs/student_int8
"""

raise SystemExit(
    "STUB: quantize.py is not yet ported from the v4 Kaggle notebook. "
    "See the module docstring for the verified configuration."
)
