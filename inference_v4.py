"""
KitPri v4 audio preprocessing + inference for the Telegram bot.

MUST match v4 training exactly (verified against test_predictions.csv):
  32 kHz mono, 10.0 s pad/truncate, MelSpectrogram(n_fft=1024, hop=512,
  n_mels=128), AmplitudeToDB(top_db=80), PER-CLIP mean/std normalization,
  3-channel repeat, natural 128x626 shape.

Deliberately NO resize to 224x224 and NO ImageNet normalization — adding
either silently degrades accuracy.

Telegram voice messages arrive as .ogg — converted to wav via ffmpeg first.
"""

import os
import subprocess
import tempfile

import torch
import torchaudio
import soundfile as sf

SAMPLE_RATE = 32_000
CLIP_DURATION = 10.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)  # 320_000

DEFAULT_THRESHOLD = 0.44  # tuned on v4 validation set (student_threshold_config.json)

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
)
_db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)


def _load_wave(path: str) -> tuple[torch.Tensor, int]:
    """Read audio to mono float32 tensor [1, T]; ffmpeg-convert formats soundfile can't read."""
    ext = os.path.splitext(path)[1].lower()
    tmp_wav = None
    try:
        if ext in {".ogg", ".m4a", ".opus", ".webm", ".mp3"}:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            tmp_wav = tmp.name
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", tmp_wav],
                capture_output=True, check=True,
            )
            path = tmp_wav
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav, sr
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


def preprocess(path: str) -> torch.Tensor:
    """Audio file → model input tensor (1, 3, 128, 626)."""
    wav, sr = _load_wave(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    n = wav.shape[1]
    if n < NUM_SAMPLES:
        wav = torch.nn.functional.pad(wav, (0, NUM_SAMPLES - n))
    elif n > NUM_SAMPLES:
        wav = wav[:, :NUM_SAMPLES]

    mel_db = _db_transform(_mel_transform(wav))
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)  # per-clip norm
    mel_db = mel_db.repeat(3, 1, 1)
    return mel_db.unsqueeze(0)


@torch.no_grad()
def classify(path: str, model, device, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """File path → {"label": str, "prob": float}. prob = P(cooking), 1 = cooking."""
    x = preprocess(path).to(device)
    prob = torch.sigmoid(model(x)).item()
    label = "Cooking" if prob >= threshold else "Not Cooking"
    return {"label": label, "prob": prob}
