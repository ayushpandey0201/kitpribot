"""
Audio I/O: file -> mono waveform at the configured sample rate and length.

Formats soundfile can't read natively (ogg/opus voice messages, m4a, mp3) are
converted through ffmpeg. All parameters come from the audio config profile —
nothing is hardcoded here.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import soundfile as sf
import torch
import torchaudio

_FFMPEG_FORMATS = {".ogg", ".oga", ".opus", ".m4a", ".webm", ".mp3", ".aac"}


def _read_any(path: str, target_sr: int) -> tuple[torch.Tensor, int]:
    """Read audio to float32 [channels, frames]; ffmpeg fallback when needed."""
    ext = os.path.splitext(path)[1].lower()
    tmp = None
    try:
        if ext in _FFMPEG_FORMATS:
            t = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            t.close()
            tmp = t.name
            proc = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                 "-ar", str(target_sr), "-ac", "1", tmp],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.strip()}")
            path = tmp
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def load_waveform(path: str, audio_cfg) -> torch.Tensor:
    """
    File -> mono waveform [1, num_samples] at audio_cfg.sample_rate,
    padded with silence or truncated to audio_cfg.clip_duration seconds.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    target_sr = int(audio_cfg.sample_rate)
    num_samples = int(target_sr * float(audio_cfg.clip_duration))

    wav, sr = _read_any(path, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    n = wav.shape[1]
    if n < num_samples:
        wav = torch.nn.functional.pad(wav, (0, num_samples - n))
    elif n > num_samples:
        wav = wav[:, :num_samples]
    return wav
