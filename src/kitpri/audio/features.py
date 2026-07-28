"""
Feature extraction: waveform -> normalized 3-channel mel spectrogram.

CRITICAL — verified pipeline, do not "improve":
  * There is deliberately NO resize to 224x224. The spectrogram keeps its
    natural (n_mels x T) shape — 128 x 626 for the 10 s / 32 kHz profile.
  * There is deliberately NO fixed ImageNet normalization. Normalization is
    PER-CLIP: (x - x.mean()) / (x.std() + 1e-6), using that clip's own stats.
Adding a Resize or a fixed Normalize will silently degrade accuracy without
raising an error. predict-parity against training probabilities was verified
to 0.0015 with exactly this pipeline.
"""

from __future__ import annotations

import torch
import torchaudio

_transform_cache: dict = {}


def _transforms(audio_cfg):
    key = (audio_cfg.sample_rate, audio_cfg.n_fft, audio_cfg.hop_length,
           audio_cfg.n_mels, audio_cfg.db_top_db)
    if key not in _transform_cache:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(audio_cfg.sample_rate),
            n_fft=int(audio_cfg.n_fft),
            hop_length=int(audio_cfg.hop_length),
            n_mels=int(audio_cfg.n_mels),
        )
        db = torchaudio.transforms.AmplitudeToDB(top_db=float(audio_cfg.db_top_db))
        _transform_cache[key] = (mel, db)
    return _transform_cache[key]


def expected_feature_shape(audio_cfg) -> tuple[int, int, int]:
    """(channels, n_mels, time_frames) the model must accept."""
    num_samples = int(audio_cfg.sample_rate * float(audio_cfg.clip_duration))
    time_frames = num_samples // int(audio_cfg.hop_length) + 1
    return (int(audio_cfg.channels), int(audio_cfg.n_mels), time_frames)


def waveform_to_features(wav: torch.Tensor, audio_cfg) -> torch.Tensor:
    """
    Waveform [1, num_samples] -> feature tensor [channels, n_mels, T].
    Fails loudly if the produced shape differs from expected_feature_shape.
    """
    if audio_cfg.normalization != "per_clip":
        raise ValueError(
            f"Unsupported normalization {audio_cfg.normalization!r} — the "
            "verified pipeline uses per_clip. Refusing to guess."
        )
    mel_t, db_t = _transforms(audio_cfg)
    mel_db = db_t(mel_t(wav))
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)   # per-clip norm
    feats = mel_db.repeat(int(audio_cfg.channels), 1, 1)

    expected = expected_feature_shape(audio_cfg)
    if tuple(feats.shape) != expected:
        raise RuntimeError(
            f"Feature shape {tuple(feats.shape)} != expected {expected}. "
            "Input waveform length or audio config is inconsistent — refusing "
            "to feed the model a silently-wrong tensor."
        )
    return feats
