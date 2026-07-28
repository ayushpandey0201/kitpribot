#!/usr/bin/env python3
"""
Classify audio clips as Cooking / Not Cooking (thin CLI over kitpri.Predictor).

    python scripts/predict.py --audio clip.wav --model inference/student_mobilenet_fp32.pt
    python scripts/predict.py --audio_dir clips/ --json
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kitpri.inference import Predictor  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "inference", "student_mobilenet_int8_scripted.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", help="single audio file")
    src.add_argument("--audio_dir", help="directory of audio files")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model file: TorchScript INT8 or state-dict checkpoint (auto-detected)")
    ap.add_argument("--config", default=None, help="optional experiment/audio config YAML")
    ap.add_argument("--threshold", type=float, default=None,
                    help="decision threshold (default: config value or 0.44)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    predictor = Predictor(args.model, config_path=args.config, threshold=args.threshold)

    if args.audio:
        files = [args.audio]
    else:
        exts = ("*.wav", "*.WAV", "*.flac", "*.ogg", "*.mp3")
        files = sorted(f for e in exts for f in glob.glob(os.path.join(args.audio_dir, e)))
        if not files:
            sys.exit(f"ERROR: no audio files found in {args.audio_dir}")

    results = predictor.predict_batch(files)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nModel: {os.path.basename(args.model)}   Threshold: {predictor.threshold}\n")
        print(f"{'FILE':<40} {'P(cooking)':>11}  PREDICTION")
        print("-" * 68)
        for r in results:
            print(f"{r['file']:<40} {r['probability']:>11.4f}  {r['prediction']}")
        print()


if __name__ == "__main__":
    main()
