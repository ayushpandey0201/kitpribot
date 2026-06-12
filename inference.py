"""
Audio preprocessing + TTA inference — matches v6 Kaggle notebook exactly.
"""

import numpy as np
import subprocess
import tempfile
import os
import torch
import torchaudio
import torchaudio.transforms as T
from torchvision import transforms
import soundfile as sf

# ── Constants (must match v6 training) ────────────────────────────────────────
SAMPLE_RATE    = 32_000
CLIP_DURATION  = 5
N_MELS         = 128
N_FFT          = 1024
HOP_LENGTH     = 320
F_MIN          = 50.0
F_MAX          = 14_000.0
IMG_SIZE       = 224
CLIP_SAMPLES   = SAMPLE_RATE * CLIP_DURATION  # 160_000

# ── Mel transform ─────────────────────────────────────────────────────────────
_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
    n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=2.0,
)
_amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)
_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_resize = transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True)


def load_and_resample(path: str) -> torch.Tensor:
    """Load any audio file, convert to mono 32kHz, pad/trim to CLIP_SAMPLES."""
    ext = os.path.splitext(path)[1].lower()
    tmp_wav = None

    try:
        # Convert ogg/m4a/opus (voice messages) to wav via ffmpeg
        if ext in {".ogg", ".m4a", ".opus", ".webm"}:
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_wav.close()
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", "32000", "-ac", "1", tmp_wav.name],
                capture_output=True, check=True,
            )
            load_path = tmp_wav.name
        else:
            load_path = path

        data, sr = sf.read(load_path, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()  # [channels, frames]

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != SAMPLE_RATE:
            waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

        # Pad or trim
        n = waveform.shape[1]
        if n >= CLIP_SAMPLES:
            waveform = waveform[:, :CLIP_SAMPLES]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, CLIP_SAMPLES - n))

        return waveform  # [1, 160000]

    finally:
        if tmp_wav and os.path.exists(tmp_wav.name):
            os.unlink(tmp_wav.name)


def waveform_to_mel_tensor(waveform: torch.Tensor) -> torch.Tensor:
    """Waveform [1, T] → normalised 3-channel mel image [3, 224, 224]."""
    mel = _mel_transform(waveform)
    mel = _amplitude_to_db(mel)
    mel_min, mel_max = mel.min(), mel.max()
    if (mel_max - mel_min) > 1e-6:
        mel = (mel - mel_min) / (mel_max - mel_min)
    else:
        mel = torch.zeros_like(mel)
    mel = mel.repeat(3, 1, 1)
    mel = _resize(mel)
    mel = _normalize(mel)
    return mel


def predict_with_tta(waveform, model, device, n_tta=5):
    """TTA inference → float probability [0,1] where 1=cooking."""
    shift_step = CLIP_SAMPLES // n_tta
    probs = []
    with torch.no_grad():
        for i in range(n_tta):
            shifted = torch.roll(waveform, shifts=i * shift_step, dims=1)
            mel = waveform_to_mel_tensor(shifted).unsqueeze(0).to(device)
            logit = model(mel)
            probs.append(torch.sigmoid(logit).item())
    return float(np.mean(probs))


def classify(path: str, model, device, threshold: float = 0.72) -> dict:
    """File path → {"label": str, "prob": float}."""
    waveform = load_and_resample(path)
    prob = predict_with_tta(waveform, model, device)
    label = "Cooking" if prob >= threshold else "Not Cooking"
    return {"label": label, "prob": prob}