#!/usr/bin/env bash
# KitPri Telegram bot launcher — zero-setup path for new users:
#   1. put TELEGRAM_BOT_TOKEN=... in a .env file at the repo root (gitignored)
#   2. run:  ./start.sh
# First run bootstraps EVERYTHING automatically (venv, CPU torch, all deps,
# the kitpri package). Subsequent runs skip straight to starting the bot.
#
# Usage:
#   ./start.sh          start (bootstraps first if needed)
#   ./start.sh setup    bootstrap only (venv + deps), don't start
#   ./start.sh stop     stop the bot
#   ./start.sh status   is it running?
#   ./start.sh log      follow the live log
#
# Custom model (see telegram_bot/README.md § "Swapping in a new model"):
#   optionally set in .env —
#     KITPRI_BOT_CKPT=inference/my_new_model.pt
#     KITPRI_BOT_THRESHOLD=0.52
set -euo pipefail
cd "$(dirname "$0")"      # repo root (script lives there)

PY=venv/bin/python

bootstrap() {
  # Fast path: everything already importable -> nothing to do.
  if [[ -x "$PY" ]] && "$PY" -c "import kitpri, torch, telegram" 2>/dev/null; then
    return 0
  fi
  echo "── first-time setup ─────────────────────────────────────────"
  if [[ ! -x "$PY" ]]; then
    echo "creating venv/ ..."
    python3 -m venv venv
  fi
  # Always `python -m pip` (never venv/bin/pip — shebangs break if the repo moves)
  echo "installing dependencies (CPU-only torch — a few minutes on first run) ..."
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt
  echo "installing the kitpri package ..."
  "$PY" -m pip install --quiet -e . --no-deps
  "$PY" -c "import kitpri, torch, telegram" \
    || { echo "setup FAILED — try: $PY -m pip install -r requirements.txt"; exit 1; }
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "NOTE: ffmpeg not found — Telegram VOICE notes (.ogg) won't decode."
    echo "      install it:  brew install ffmpeg   (macOS)"
    echo "                   sudo apt-get install -y ffmpeg   (Ubuntu/Debian)"
  fi
  echo "── setup complete ───────────────────────────────────────────"
}

case "${1:-start}" in
  setup)
    bootstrap
    exit 0 ;;
  stop)
    pkill -f "telegram_bot/bot.py" 2>/dev/null && echo "bot stopped" || echo "bot was not running"
    exit 0 ;;
  status)
    pgrep -f "telegram_bot/bot.py" >/dev/null \
      && echo "bot RUNNING (pid $(pgrep -f 'telegram_bot/bot.py' | tr '\n' ' '))" \
      || echo "bot STOPPED"
    exit 0 ;;
  log)
    # Redact any token that older log lines may contain
    exec tail -f bot.log | sed -E 's#bot[0-9]+:[A-Za-z0-9_-]+#bot<TOKEN-REDACTED>#g' ;;
  start) ;;
  *) echo "usage: ./start.sh [setup|stop|status|log]"; exit 2 ;;
esac

bootstrap

# Token: .env first, else inherited environment. Never passed as a CLI arg.
if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${TELEGRAM_BOT_TOKEN:?not set — create .env at repo root containing: TELEGRAM_BOT_TOKEN=<token from @BotFather>}"

# Optional model override from .env — read by bot.py itself, echoed here for visibility
[[ -n "${KITPRI_BOT_CKPT:-}" ]] && echo "using custom checkpoint: $KITPRI_BOT_CKPT"
[[ -n "${KITPRI_BOT_THRESHOLD:-}" ]] && echo "using custom threshold: $KITPRI_BOT_THRESHOLD"

# Only one polling instance may exist per token (Telegram getUpdates conflict).
# PTB's graceful shutdown can hang on SIGTERM, so wait and escalate to SIGKILL.
if pkill -f "telegram_bot/bot.py" 2>/dev/null; then
  for _ in $(seq 1 5); do
    pgrep -f "telegram_bot/bot.py" >/dev/null || break
    sleep 1
  done
  pkill -9 -f "telegram_bot/bot.py" 2>/dev/null || true
  sleep 1
fi

nohup "$PY" telegram_bot/bot.py > bot.log 2>&1 &
disown

# Wait for model load, then report
for _ in $(seq 1 20); do
  grep -q "Model loaded" bot.log 2>/dev/null && break
  sleep 1
done

if pgrep -f "telegram_bot/bot.py" >/dev/null; then
  grep -E "Using device|Model loaded" bot.log | sed 's/^/  /'
  echo "bot RUNNING (pid $(pgrep -f 'telegram_bot/bot.py' | tr '\n' ' ')) — follow with: ./start.sh log"
else
  echo "bot FAILED to start — last log lines:"
  tail -5 bot.log
  exit 1
fi
