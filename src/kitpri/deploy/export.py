"""Model export: TorchScript (deployment format used by the INT8 student)."""

from __future__ import annotations

from pathlib import Path

import torch


def export_torchscript(model: torch.nn.Module, example_input: torch.Tensor,
                       out_path: str) -> Path:
    """Script (fallback: trace) the model and save a self-contained archive."""
    model = model.cpu().eval()
    try:
        scripted = torch.jit.script(model)
    except Exception:
        scripted = torch.jit.trace(model, example_input)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out))
    print(f"[export] TorchScript saved: {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return out
