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

from model import load_model
from inference import classify


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
MODEL = None
DEVICE = None
THRESHOLD = 0.72


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

        result = classify(tmp_path, MODEL, DEVICE, threshold=THRESHOLD)
        label = result["label"]

        if label == "Cooking":
            reply = "🍳 *Cooking*"
        else:
            reply = "🔇 *Not Cooking*"

        await message.reply_text(reply, parse_mode="Markdown")
        logger.info(f"Classified: {label} (p={result['prob']:.3f})")

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
    global MODEL, DEVICE, THRESHOLD

    parser = argparse.ArgumentParser(description="Kitchen Audio Telegram Bot")
    parser.add_argument("--token", required=True, help="Telegram Bot Token from @BotFather")
    parser.add_argument("--ckpt", default="best_ckpt.pt", help="Path to best_ckpt.pt")
    parser.add_argument("--threshold", type=float, default=0.72, help="Probability threshold for Cooking class (default: 0.72)")
    args = parser.parse_args()

    THRESHOLD = args.threshold

    # Load model once at startup
    DEVICE = get_device()
    logger.info(f"Using device: {DEVICE}")
    MODEL = load_model(args.ckpt, DEVICE)
    logger.info("Model loaded and ready.")

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
