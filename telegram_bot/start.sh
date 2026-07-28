#!/usr/bin/env bash
# KitPri Telegram bot launcher — easiest path: put TELEGRAM_BOT_TOKEN=... in
# a .env file at the repo root (gitignored), then run:  telegram_bot/start.sh
#
# Usage:
#   telegram_bot/start.sh          start (restarts if already running)
#   telegram_bot/start.sh stop     stop the bot
#   telegram_bot/start.sh status   is it running?
#   telegram_bot/start.sh log      follow the live log
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, regardless of where invoked from

case "${1:-start}" in
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
  *) echo "usage: telegram_bot/start.sh [stop|status|log]"; exit 2 ;;
esac

# Token: .env first, else inherited environment. Never passed as a CLI arg.
if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${TELEGRAM_BOT_TOKEN:?not set — create .env at repo root containing: TELEGRAM_BOT_TOKEN=<token from @BotFather>}"

PY=venv/bin/python
[[ -x "$PY" ]] || PY=python3

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
  echo "bot RUNNING (pid $(pgrep -f 'telegram_bot/bot.py' | tr '\n' ' ')) — follow with: telegram_bot/start.sh log"
else
  echo "bot FAILED to start — last log lines:"
  tail -5 bot.log
  exit 1
fi
