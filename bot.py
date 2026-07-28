"""
Telegram bot for kitchen audio classification.
Accepts: audio file uploads (.wav / .mp3 / .ogg / .m4a) + voice messages.
Outputs: 🍳 Cooking  or  🔇 Not Cooking

Usage:
    python bot.py --token YOUR_BOT_TOKEN --ckpt best_ckpt.pt
"""

import argparse
import asyncio
import logging
import os
import tempfile
from pathlib import Path

import torch
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from kitpri.inference import Predictor


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Globals (set at startup) ──────────────────────────────────────────────────
PREDICTOR: Predictor | None = None


# ── Handlers ─────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 *Kitchen Audio Classifier*\n\n"
        "Send me an audio file or a voice message and I'll tell you whether it sounds like cooking.\n\n"
        "Supported formats: `.wav` `.mp3` `.ogg` `.m4a` `.flac`",
        parse_mode="Markdown",
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Just send an audio file or record a voice message.\n"
        "I'll reply with 🍳 *Cooking* or 🔇 *Not Cooking*.",
        parse_mode="Markdown",
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle both Document audio uploads and Voice messages."""
    message = update.message

    # Determine which type of audio was sent
    if message.voice:
        tg_file = await message.voice.get_file()
        suffix = ".ogg"
    elif message.audio:
        tg_file = await message.audio.get_file()
        suffix = Path(message.audio.file_name or "audio.mp3").suffix or ".mp3"
    elif message.document:
        fname = message.document.file_name or ""
        ext = Path(fname).suffix.lower()
        if ext not in {".wav", ".mp3", ".ogg", ".m4a", ".flac"}:
            await message.reply_text("⚠️ Please send a `.wav`, `.mp3`, `.ogg`, `.m4a`, or `.flac` file.")
            return
        tg_file = await message.document.get_file()
        suffix = ext
    else:
        await message.reply_text("⚠️ I can only process audio files or voice messages.")
        return

    # Show "typing…" while processing
    await context.bot.send_chat_action(chat_id=message.chat_id, action="typing")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        await tg_file.download_to_drive(tmp_path)
        logger.info(f"Downloaded audio to {tmp_path} ({os.path.getsize(tmp_path)} bytes)")

        result = PREDICTOR.predict_file(tmp_path)

        if result["prediction"] == "Cooking":
            reply = "🍳 *Cooking*"
        else:
            reply = "🔇 *Not Cooking*"

        await message.reply_text(reply, parse_mode="Markdown")
        logger.info(f"Classified: {result['prediction']} (p={result['probability']:.3f})")

    except Exception as e:
        logger.exception("Inference error")
        await message.reply_text(f"❌ Error during classification: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def fallback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Please send an audio file or voice message. Type /help for instructions."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global PREDICTOR

    parser = argparse.ArgumentParser(description="Kitchen Audio Telegram Bot")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"),
                        help="Telegram Bot Token (or set TELEGRAM_BOT_TOKEN env var)")
    parser.add_argument("--ckpt", default="kitpri_v4_submission/inference/student_mobilenet_fp32.pt",
                        help="Path to the KitPri v4 FP32 student checkpoint")
    parser.add_argument("--config", default="kitpri_v4_submission/configs/experiments/distill.yaml",
                        help="kitpri experiment config (audio profile + per-model threshold)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the config threshold (default: from config, 0.44)")
    args = parser.parse_args()

    if not args.token:
        parser.error("provide --token or set TELEGRAM_BOT_TOKEN")

    # Unified kitpri Predictor — the SAME code path as scripts/predict.py and
    # evaluation, so bot preprocessing can never drift from training.
    device = get_device()
    logger.info(f"Using device: {device}")
    PREDICTOR = Predictor(args.ckpt, config_path=args.config,
                          threshold=args.threshold, device=str(device))
    logger.info(f"Model loaded and ready (threshold={PREDICTOR.threshold}).")

    # Build application
    app = Application.builder().token(args.token).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.VOICE, handle_audio))
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.Document.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, fallback))

    logger.info("Bot is polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
