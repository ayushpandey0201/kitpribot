"""
Unified Predictor — the ONE inference path used by scripts/predict.py, the
evaluation loop, and the Telegram bot, so deployment can never drift from
training preprocessing.

Handles both model formats transparently:
  * TorchScript archives (INT8 student)  -> torch.jit.load
  * state-dict checkpoints (FP32 student / teacher) -> registry build + load
Format and architecture are AUTO-DETECTED from the file, never guessed from
the filename.

INT8 platform guard: the shipped INT8 artifact is fbgemm-packed (x86). Under
ARM's qnnpack engine it loads but produces collapsed, meaningless outputs
(verified empirically). On hosts without an x86 quantized engine we refuse
with a clear error instead of silently returning garbage.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import torch

from kitpri.audio import load_waveform, waveform_to_features, expected_feature_shape
from kitpri.config import Config, load_config, validate_audio_config
from kitpri.labels import LABEL_NAMES, label_from_probability
from kitpri.models import build_model
from kitpri.models.base import check_model_interface

_DEFAULT_AUDIO = Config({
    # Mirrors configs/audio/mel_32k_10s.yaml — used when no config_path is
    # given (e.g. the Telegram bot). Kept in one dict, verified values.
    "sample_rate": 32000, "clip_duration": 10.0, "n_mels": 128,
    "n_fft": 1024, "hop_length": 512, "db_top_db": 80,
    "normalization": "per_clip", "channels": 3, "resize": None,
})

# Verified checkpoint-key fingerprints -> registry names.
_ARCH_FINGERPRINTS = [
    ("audio_spectrogram_transformer.", "ast"),
    ("conv_stem.", "mobilenetv2_100"),
]


def _is_torchscript(path: str) -> bool:
    """TorchScript archives contain a code/ directory; state-dict zips do not."""
    try:
        with zipfile.ZipFile(path) as z:
            return any(n.split("/", 1)[-1].startswith("code/") for n in z.namelist())
    except zipfile.BadZipFile:
        return False


def _configure_quantized_engine(path: str) -> None:
    supported = torch.backends.quantized.supported_engines
    for eng in ("fbgemm", "x86"):
        if eng in supported:
            torch.backends.quantized.engine = eng
            return
    raise RuntimeError(
        f"{path} is an INT8 TorchScript model quantized for x86 (fbgemm) and "
        f"produces incorrect results on this CPU (supported engines: {supported}). "
        "Use the FP32 model here, or run inside the amd64 Docker container, or "
        "requantize with backend=qnnpack (configs/experiments/quantize.yaml)."
    )


class Predictor:
    def __init__(self, model_path: str, config_path: str | None = None,
                 threshold: float | None = None, device: str = "cpu"):
        self.device = torch.device(device)

        if config_path:
            cfg = load_config(config_path)
            validate_audio_config(cfg)
            self.audio_cfg = cfg.audio
            cfg_threshold = cfg.get("model", {}).get("threshold")
        else:
            self.audio_cfg = _DEFAULT_AUDIO
            cfg_threshold = None
        # Priority: explicit arg > model config > verified default (0.44,
        # from results/.../student_threshold_config.json).
        self.threshold = threshold if threshold is not None else (
            float(cfg_threshold) if cfg_threshold is not None else 0.44)

        model_path = str(Path(model_path).expanduser())
        if _is_torchscript(model_path):
            _configure_quantized_engine(model_path)
            self.model = torch.jit.load(model_path, map_location="cpu")
            # Quantized TorchScript is CPU-only; ignore requested device.
            self.device = torch.device("cpu")
        else:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state", ckpt.get("state_dict", ckpt)) \
                if isinstance(ckpt, dict) else ckpt
            arch = next((name for prefix, name in _ARCH_FINGERPRINTS
                         if any(k.startswith(prefix) for k in state)), None)
            if arch is None:
                raise ValueError(
                    f"Cannot identify architecture of {model_path} from its "
                    f"checkpoint keys (first keys: {list(state)[:3]}). "
                    f"Known fingerprints: {_ARCH_FINGERPRINTS}"
                )
            self.model = build_model(Config({"name": arch, "num_classes": 1,
                                             "pretrained": False}))
            self.model.load_state_dict(state, strict=True)
            self.model.to(self.device)
            check_model_interface(
                self.model.to("cpu"), expected_feature_shape(self.audio_cfg))
            self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_file(self, path: str) -> dict:
        wav = load_waveform(path, self.audio_cfg)
        feats = waveform_to_features(wav, self.audio_cfg).unsqueeze(0).to(self.device)
        prob = torch.sigmoid(self.model(feats)).item()
        label = label_from_probability(prob, self.threshold)
        return {
            "file": Path(path).name,
            "probability": round(prob, 6),
            "threshold": self.threshold,
            "label": label,
            "prediction": LABEL_NAMES[label],
        }

    def predict_batch(self, paths: list[str]) -> list[dict]:
        return [self.predict_file(p) for p in paths]
