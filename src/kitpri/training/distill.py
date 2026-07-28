"""
Knowledge distillation loss (teacher -> student), single-logit binary variant.

Verified v4 hyperparameters (run_summary.json): temperature=3.0, alpha=0.4.

ASSUMPTION — flagged, not silently decided: alpha is taken to weight the
HARD-label term (common convention):
    loss = alpha * BCE(student_logit, hard_label)
         + (1 - alpha) * T^2 * BCE(student_logit / T, sigmoid(teacher_logit / T))
The original Kaggle notebook is the authority; when training/ is ported,
verify the convention matches and correct here if it differs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DistillationLoss(torch.nn.Module):
    def __init__(self, teacher: torch.nn.Module, temperature: float, alpha: float):
        super().__init__()
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.T = float(temperature)
        self.alpha = float(alpha)
        self._x = None  # set per batch via set_inputs

    def set_inputs(self, x: torch.Tensor) -> None:
        self._x = x

    def forward(self, student_logit: torch.Tensor, hard_label: torch.Tensor) -> torch.Tensor:
        if self._x is None:
            raise RuntimeError("DistillationLoss.set_inputs(x) must be called each batch")
        with torch.no_grad():
            teacher_logit = self.teacher(self._x)
        soft_target = torch.sigmoid(teacher_logit / self.T)
        soft = F.binary_cross_entropy_with_logits(student_logit / self.T, soft_target)
        hard = F.binary_cross_entropy_with_logits(student_logit, hard_label)
        return self.alpha * hard + (1.0 - self.alpha) * (self.T ** 2) * soft
