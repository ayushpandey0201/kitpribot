# Kitchen Audio Telegram Bot — Mac M1 Setup

## Prerequisites

- Python 3.10+
- `ffmpeg` installed (for ogg/m4a decoding)
- `best_ckpt.pt` in this folder

---

## 1. Install ffmpeg

```bash
brew install ffmpeg
```

---

## 2. Create a virtual environment

```bash
cd telegram_bot
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ For Mac M1, PyTorch MPS is used automatically if available.
> Make sure you have PyTorch ≥ 2.1 with MPS support.

---

## 4. Create your Telegram Bot

1. Open Telegram → search **@BotFather**
2. Send `/newbot`
3. Give it a name (e.g. `KitchenAudioBot`) and username (e.g. `my_kitchen_audio_bot`)
4. Copy the **HTTP API token** (looks like `7123456789:AAF...`)

---

## 5. Run the bot

```bash
python bot.py --token YOUR_BOT_TOKEN --ckpt best_ckpt.pt
```

You should see:
```
Using device: mps
[model] Loaded checkpoint from best_ckpt.pt → mps
Model loaded and ready.
Bot is polling…
```

---

## 6. Test it

- Open Telegram, find your bot
- Send `/start`
- Upload a `.wav` / `.mp3` file, or record a **voice message**
- Bot replies: 🍳 **Cooking** or 🔇 **Not Cooking**

---

## File structure

```
telegram_bot/
├── bot.py           ← main bot entry point
├── model.py         ← ASTModel definition (matches v6 Kaggle training)
├── inference.py     ← mel spectrogram + TTA inference pipeline
├── requirements.txt
├── best_ckpt.pt     ← place your downloaded checkpoint here
└── README.md
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `MPS not available` | Update to PyTorch ≥ 2.1; falls back to CPU automatically |
| `Error loading state_dict` | Ensure `best_ckpt.pt` is from v6 DeiT-Small training |
| `ffmpeg not found` | Run `brew install ffmpeg` |
| `ogg decode error` | Voice messages are `.ogg`; ffmpeg handles this |

---

## Notes

- **No ngrok needed** — the bot uses **long polling**, not webhooks.  
  It works on any network, no port forwarding required.
- The model runs on **MPS** (Apple Silicon GPU) — first inference may take ~2s to warm up, subsequent ones ~0.3–0.5s.
- TTA runs 5 passes per clip for more stable predictions.
