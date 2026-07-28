# KitPri v4 — Telegram Bot Demo

Live demo of the KitPri v4 classifier: send an audio file or voice message,
get back 🍳 **Cooking** or 🔇 **Not Cooking**.

Runs the **FP32 MobileNetV2 student** (threshold 0.44) through the unified
`kitpri.inference.Predictor` — the exact same preprocessing and model code as
`inference/predict.py`, so the bot cannot drift from the evaluated pipeline.

## Setup (from the repo root)

```bash
python3 -m venv venv && source venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pip install -e .            # installs the kitpri package
brew install ffmpeg          # voice messages arrive as .ogg
```

## Run

```bash
export TELEGRAM_BOT_TOKEN=...    # from @BotFather — NEVER pass via --token (visible in `ps`)
python telegram_bot/bot.py
```

Expected log:

```
Using device: mps
Model loaded and ready (threshold=0.44).
Bot is polling…
```

Long polling — no ngrok, webhooks, or port forwarding needed.

## Options

| Flag | Default | Description |
|---|---|---|
| `--ckpt` | `inference/student_mobilenet_fp32.pt` | model checkpoint |
| `--config` | `configs/experiments/distill.yaml` | audio profile + per-model threshold |
| `--threshold` | from config (0.44) | override decision threshold |

## Notes

- The token is also logged inside Telegram API URLs by `httpx` — bot logs are
  gitignored (`*.log`); never commit them.
- On Apple Silicon the model runs on MPS automatically; falls back to CPU.
- The retired v6 DeiT flow (`best_ckpt.pt`, threshold 0.72) is gone — it had
  weak class separation on realistic audio and was replaced by the verified
  v4 student (6/6 probability parity vs training).
