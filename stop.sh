#!/usr/bin/env bash
# Stop ALL KitPri bot processes on THIS machine — LOCAL ONLY.
#
# The 24/7 production bot runs on the Oracle Cloud VM, which is a different
# computer: nothing this script does can ever reach it. (To manage that one
# deliberately: ssh kitpri-vm 'sudo systemctl stop kitpri-bot')
set -euo pipefail

if pgrep -f "telegram_bot/bot.py" >/dev/null; then
  pkill -f "telegram_bot/bot.py" || true
  # PTB can hang on SIGTERM — verify, then escalate to SIGKILL
  for _ in $(seq 1 5); do
    pgrep -f "telegram_bot/bot.py" >/dev/null || break
    sleep 1
  done
  pkill -9 -f "telegram_bot/bot.py" 2>/dev/null || true
  echo "all LOCAL bot processes stopped"
else
  echo "no LOCAL bot processes were running"
fi
echo "(the cloud bot on the Oracle VM is untouched — this script can't reach it)"
