"""
YAML config loading with includes, env-var resolution, and validation.

Features:
  * `include:` — list of paths relative to the including file, deep-merged in
    order; keys in the including file win.
  * `${oc.env:VAR}` / `${oc.env:VAR,default}` — resolved from the environment
    (OmegaConf-compatible syntax, no OmegaConf dependency).
  * Attribute access: cfg.audio.sample_rate.

No path may be hardcoded in src/ — anything environment-specific belongs in
configs/ or the environment.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_ENV_PATTERN = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")


class Config(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"Config has no key {key!r}. Available: {list(self)}") from e

    def __setattr__(self, key, value):
        self[key] = value


def _to_config(obj):
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def _resolve_env(obj):
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v) for v in obj]
    if isinstance(obj, str):
        def sub(m):
            var, default = m.group(1), m.group(2)
            val = os.environ.get(var)
            if val is not None:
                return val
            if default is not None:
                return default
            raise KeyError(f"Environment variable {var} is not set and no default given")
        return _ENV_PATTERN.sub(sub, obj)
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> Config:
    """Load a YAML config, resolving includes (relative to the file) and env vars."""
    path = Path(path).expanduser().resolve()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    merged: dict = {}
    for inc in raw.pop("include", []):
        inc_cfg = load_config(path.parent / inc)
        merged = _deep_merge(merged, inc_cfg)
    merged = _deep_merge(merged, raw)
    return _to_config(_resolve_env(merged))


def validate_audio_config(cfg: Config) -> None:
    """Fail loudly if the audio profile is missing required fields."""
    required = ["sample_rate", "clip_duration", "n_mels", "n_fft", "hop_length",
                "db_top_db", "normalization", "channels"]
    audio = cfg.get("audio")
    if audio is None:
        raise ValueError("Config has no 'audio' section — include an audio profile "
                         "(e.g. configs/audio/mel_32k_10s.yaml)")
    missing = [k for k in required if k not in audio]
    if missing:
        raise ValueError(f"Audio config missing required keys: {missing}")
    if audio.get("resize") is not None:
        raise ValueError(
            "audio.resize is set — KitPri v4 deliberately uses the natural "
            "spectrogram shape. A resize will silently degrade accuracy."
        )


def save_config(cfg: Config, path: str | Path) -> None:
    """Write a run_config.yaml snapshot next to run outputs (reproducibility)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items()}, f, sort_keys=False)
