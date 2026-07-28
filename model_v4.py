"""
KitPri v4 model loading — MobileNetV2 FP32 student (distilled from AST teacher).

Replaces the earlier v6 DeiT-Small model, which showed weak class separation on
realistic mixed audio (diagnosed 2026-07-28: ~60% accuracy on v4 test clips,
probabilities compressed into 0.27-0.85 → most real-world audio read as
"Cooking" at the old 0.72 threshold). The v4 student's probabilities were
verified to match training-time test_predictions.csv within 0.0015.

Checkpoint: kitpri_v4_submission/inference/student_mobilenet_fp32.pt
Format:     dict with keys `model_state`, `epoch`, `val_f1`
Output:     single logit; sigmoid(logit) = P(cooking); 1 = cooking.
"""

import torch
import timm


def load_model(ckpt_path: str, device: torch.device):
    model = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=1)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    print(f"[model] Loaded KitPri v4 MobileNetV2 student from {ckpt_path} → {device}")
    return model
