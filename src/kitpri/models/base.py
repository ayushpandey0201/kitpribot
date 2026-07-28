"""
Shared model interface.

Every registered model MUST satisfy:
  * input:  (B, channels, n_mels, T) float tensor — e.g. (B, 3, 128, 626)
  * output: (B, 1) single logit; sigmoid(logit) = P(cooking)
    (label convention centralized in kitpri.labels — 1 = cooking)

Builders receive the model config (cfg.model) and return an nn.Module.
"""

import torch


def check_model_interface(model: torch.nn.Module, feature_shape: tuple[int, int, int]) -> None:
    """Runtime assertion: model accepts the configured feature shape and emits (B, 1)."""
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, *feature_shape))
    if tuple(out.shape) != (1, 1):
        raise RuntimeError(
            f"Model output shape {tuple(out.shape)} != (1, 1). Every KitPri "
            "model must emit a single logit per clip."
        )
