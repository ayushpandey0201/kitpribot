#!/usr/bin/env python3
"""
KitPri v4 - Cooking Sound Detection : Inference / Demo Script
=============================================================
Classifies an audio clip as "Cooking" or "Not Cooking".

Two model backends are supported:
  1. INT8 TorchScript student (default) - 2.9 MB, self-contained, CPU-only.
     Requires only torch + torchaudio + soundfile.
  2. FP32 student  - 8.5 MB, slightly higher accuracy. Additionally needs `timm`.

Usage
-----
    python predict.py --audio sample.wav
    python predict.py --audio sample.wav --model int8
    python predict.py --audio sample.wav --model fp32 --threshold 0.44
    python predict.py --audio_dir ./clips/            # batch mode
    python predict.py --audio sample.wav --json       # machine-readable output

Label convention
----------------
The model has a SINGLE output logit. sigmoid(logit) = P(cooking).
    probability >= threshold  ->  "Cooking"      (label 1)
    probability <  threshold  ->  "Not Cooking"  (label 0)
This matches the training CSVs, where label==1 corresponds to audio_32k/cooking/.
"""

import argparse
import json
import os
import sys
import glob

import torch
import torchaudio
import soundfile as sf

# ----------------------------------------------------------------------------
# Preprocessing constants.
# These MUST match the values used during distillation training. Changing any
# of them will silently degrade accuracy rather than raise an error.
# ----------------------------------------------------------------------------
SAMPLE_RATE = 32000
CLIP_DURATION = 10.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

# Decision threshold tuned on the validation set (see student_threshold_config.json).
DEFAULT_THRESHOLD = 0.44

HERE = os.path.dirname(os.path.abspath(__file__))
INT8_PATH = os.path.join(HERE, "student_mobilenet_int8_scripted.pt")
FP32_PATH = os.path.join(HERE, "student_mobilenet_fp32.pt")

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)
_db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)


# ----------------------------------------------------------------------------
# Audio -> model input
# ----------------------------------------------------------------------------
def load_wave(path):
    """Read an audio file to a mono float32 tensor of shape (1, num_samples)."""
    wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav_np.T)
    if wav.shape[0] > 1:                       # stereo (or more) -> mono
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def preprocess(path):
    """
    Convert an audio file into the model input tensor.

    Returns a tensor of shape (1, 3, 128, 626) for a 10 s clip:
      1   = batch
      3   = mel-spectrogram replicated across 3 channels (CNN expects 3)
      128 = mel bins
      626 = time frames  (= NUM_SAMPLES / HOP_LENGTH + 1)

    NOTE: there is deliberately NO resize to 224x224 and NO ImageNet
    normalization. The spectrogram keeps its natural shape and is normalized
    per-clip using its own mean/std. This differs from typical vision
    preprocessing - do not "helpfully" add a Resize or Normalize here.
    """
    wav, sr = load_wave(path)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    n = wav.shape[1]
    if n < NUM_SAMPLES:                        # pad short clips with silence
        wav = torch.nn.functional.pad(wav, (0, NUM_SAMPLES - n))
    elif n > NUM_SAMPLES:                      # truncate long clips
        wav = wav[:, :NUM_SAMPLES]

    mel = _mel_transform(wav)
    mel_db = _db_transform(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)   # per-clip norm
    mel_db = mel_db.repeat(3, 1, 1)                             # mono -> 3ch
    return mel_db.unsqueeze(0)                                  # add batch dim


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def load_model(kind="int8"):
    """Load either the INT8 TorchScript model or the FP32 student."""
    if kind == "int8":
        if not os.path.exists(INT8_PATH):
            sys.exit(f"ERROR: model file not found: {INT8_PATH}")
        # The INT8 artifact was packed with the fbgemm (x86) engine. It loads
        # under qnnpack (ARM) but produces meaningless outputs (verified: all
        # probabilities collapse to ~0.01 regardless of input). Only allow
        # x86 engines; on ARM hosts, direct users to the FP32 model.
        supported = torch.backends.quantized.supported_engines
        for eng in ("fbgemm", "x86"):
            if eng in supported:
                torch.backends.quantized.engine = eng
                break
        else:
            sys.exit(
                "ERROR: the INT8 model is quantized for x86 (fbgemm) and gives\n"
                f"incorrect results on this CPU (supported engines: {supported}).\n"
                "Use --model fp32 here, or run the INT8 model in the provided\n"
                "Docker container with --platform linux/amd64."
            )
        model = torch.jit.load(INT8_PATH, map_location="cpu")
        model.eval()
        return model

    if kind == "fp32":
        try:
            import timm
        except ImportError:
            sys.exit("ERROR: --model fp32 requires timm. Run: pip install timm")
        if not os.path.exists(FP32_PATH):
            sys.exit(f"ERROR: model file not found: {FP32_PATH}")

        model = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=1)
        ckpt = torch.load(FP32_PATH, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    sys.exit(f"ERROR: unknown model kind '{kind}' (expected 'int8' or 'fp32')")


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_one(model, path, threshold=DEFAULT_THRESHOLD):
    x = preprocess(path)
    logit = model(x)
    prob = torch.sigmoid(logit).item()
    return {
        "file": os.path.basename(path),
        "probability_cooking": round(prob, 4),
        "threshold": threshold,
        "prediction": "Cooking" if prob >= threshold else "Not Cooking",
        "label": 1 if prob >= threshold else 0,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Classify audio clips as Cooking / Not Cooking (KitPri v4)."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", help="path to a single audio file (.wav/.flac/.ogg)")
    src.add_argument("--audio_dir", help="directory of audio files (batch mode)")
    ap.add_argument("--model", default="int8", choices=["int8", "fp32"],
                    help="which model to run (default: int8)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"decision threshold (default: {DEFAULT_THRESHOLD})")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    model = load_model(args.model)

    if args.audio:
        if not os.path.exists(args.audio):
            sys.exit(f"ERROR: file not found: {args.audio}")
        files = [args.audio]
    else:
        exts = ("*.wav", "*.WAV", "*.flac", "*.ogg", "*.mp3")
        files = sorted(f for e in exts for f in glob.glob(os.path.join(args.audio_dir, e)))
        if not files:
            sys.exit(f"ERROR: no audio files found in {args.audio_dir}")

    results = [predict_one(model, f, args.threshold) for f in files]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nModel: {args.model.upper()}   Threshold: {args.threshold}\n")
        print(f"{'FILE':<40} {'P(cooking)':>11}  PREDICTION")
        print("-" * 68)
        for r in results:
            print(f"{r['file']:<40} {r['probability_cooking']:>11.4f}  {r['prediction']}")
        print()


if __name__ == "__main__":
    main()
