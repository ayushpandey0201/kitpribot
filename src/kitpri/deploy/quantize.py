"""
Static post-training INT8 quantization with a CONFIG-DRIVEN backend.

Why backend matters: the original v4 artifact was packed with fbgemm (x86).
Verified empirically: under ARM's qnnpack engine it loads but every output
collapses to ~0.01 — silently wrong. The SmartThings target is ARM, so the
deployable artifact must be built with backend=qnnpack. The engine MUST be
set BEFORE prepare/convert — that is the whole bug class this module fixes.
"""

from __future__ import annotations

import torch
from torch.ao.quantization import QuantWrapper, get_default_qconfig, prepare, convert


def quantize_static_ptq(model: torch.nn.Module, calibration_batches, quant_cfg):
    """
    model               -- FP32 nn.Module (eval mode enforced here)
    calibration_batches -- iterable of feature tensors (B, C, n_mels, T)
    quant_cfg           -- cfg.quantization (backend, calibration_samples)
    Returns the converted INT8 nn.Module (CPU).
    """
    backend = str(quant_cfg.backend)
    supported = torch.backends.quantized.supported_engines
    if backend not in supported:
        raise RuntimeError(
            f"Quantization backend '{backend}' not available on this host "
            f"(supported: {supported}). Build the artifact on a host that "
            "supports the TARGET's engine — do not silently fall back."
        )
    # CRITICAL: engine before prepare/convert.
    torch.backends.quantized.engine = backend

    model = model.cpu().eval()
    wrapped = QuantWrapper(model)
    wrapped.qconfig = get_default_qconfig(backend)
    prepare(wrapped, inplace=True)

    seen = 0
    limit = int(quant_cfg.get("calibration_samples", 500))
    with torch.no_grad():
        for batch in calibration_batches:
            wrapped(batch)
            seen += batch.shape[0]
            if seen >= limit:
                break
    if seen == 0:
        raise ValueError("No calibration data provided — static PTQ requires it")

    convert(wrapped, inplace=True)
    print(f"[quantize] backend={backend} calibrated on {seen} clips")
    return wrapped
