#!/usr/bin/env bash
# KitPri launcher — zero-setup path for new users: just run  ./start.sh
# First run bootstraps EVERYTHING (venv, CPU torch, all deps, the kitpri
# package). No token yet? It asks what you want — run your own local test
# bot (guided @BotFather walkthrough) or model-development-only — and
# prints the exact steps either way.
#
# Usage:
#   ./start.sh          start a LOCAL bot on this machine (bootstraps + guides first if needed)
#   ./start.sh setup    bootstrap only (venv + deps), don't start
#   ./start.sh stop     stop the LOCAL bot (same as ./stop.sh; never touches the cloud)
#   ./start.sh status   is a LOCAL bot running?
#   ./start.sh log      follow the LOCAL bot's live log
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
    exec ./stop.sh ;;
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

# ── No token? Interactive first-run: choose your path ────────────────────────
if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
  if [[ ! -t 0 ]]; then   # non-interactive shell (CI, ssh -c, …): keep old behavior
    echo "TELEGRAM_BOT_TOKEN not set — create .env at the repo root containing:"
    echo "  TELEGRAM_BOT_TOKEN=<token from @BotFather>"
    exit 1
  fi

  cat <<'MENU'

No bot token found. What do you want to do?

  1) Run MY OWN Telegram bot on this laptop
     A private sandbox: message YOUR bot an audio clip, get 🍳/🔇 back —
     useful for testing code/model changes end-to-end before shipping them.
     Needs a free token from Telegram's @BotFather (~2 min, guided below).
     Note: the official @kitpribot already runs 24/7 in the cloud, and a
     token can only be polled by ONE bot at a time — so local testing
     always uses a separate token/bot of your own.

  2) Model development only — no bot, no token
     Environment is already set up (that just happened). You get the
     commands for running inference locally and for shipping a new model
     to the 24/7 cloud bot.

MENU
  read -rp "Choose [1/2]: " choice
  case "${choice:-}" in
    1)
      cat <<'STEPS'

── Get a bot token (everything happens inside the Telegram app) ──────────────

  1. In Telegram's search bar type:   @BotFather    → open it (blue checkmark)
  2. Send it the message:             /newbot
  3. It asks for a display name  →    My KitPri Test Bot        (anything)
  4. It asks for a username      →    classifier_kitpri_test_bot     (must end in 'bot',
                                                                 must be unused)
  5. BotFather replies "Done! …" containing a token that looks like:

        1234567890:AAExampleExampleExampleExample1234

     Copy that whole line (tap it — Telegram copies on tap).

STEPS
      read -rsp "Paste your token here (input is hidden, saved to .env): " tok; echo
      if [[ ! "$tok" =~ ^[0-9]+:[A-Za-z0-9_-]{30,}$ ]]; then
        echo "✗ that doesn't look like a bot token (expected digits:letters, e.g. 1234567890:AAE...)"
        echo "  nothing saved — run ./start.sh again to retry"
        exit 1
      fi
      printf 'TELEGRAM_BOT_TOKEN=%s\n' "$tok" >> .env   # append keeps any KITPRI_BOT_* lines
      chmod 600 .env
      export TELEGRAM_BOT_TOKEN="$tok"
      echo "✓ saved to .env (private, gitignored). Starting your bot…"
      echo "  when it's up: open YOUR bot in Telegram (step 4's username), press START,"
      echo "  and send it a voice note or audio file."
      echo
      ;;
    2)
      cat <<'DEV'

── Model development quickstart (no bot needed) ──────────────────────────────

  # classify a clip with the shipped model right now:
  venv/bin/python inference/predict.py --audio path/to/clip.wav --model fp32

  # ship a NEW model to the 24/7 cloud bot (full guide:
  # telegram_bot/README.md § "Swapping in a new model"):
  1. put your checkpoint at            inference/my_new_model.pt
  2. re-sweep its threshold on the VALIDATION set (0.44 belongs to the old model)
  3. echo '!inference/my_new_model.pt' >> .gitignore     # *.pt ignored by default
  4. git add -A && git commit -m 'model: vNext' && git push origin main
  5. on the VM:  ssh kitpri-vm
       nano kitpri/.env      # add: KITPRI_BOT_CKPT=inference/my_new_model.pt
                             #      KITPRI_BOT_THRESHOLD=<your swept value, e.g. 0.52>
       cd kitpri && git pull && sudo systemctl restart kitpri-bot
       journalctl -u kitpri-bot -n 5    # 'Checkpoint:' line confirms the new model

  Environment is ready — happy training. (Run ./start.sh again anytime to
  set up a local test bot.)

DEV
      exit 0 ;;
    *)
      echo "no valid choice — aborting (run ./start.sh again)"; exit 1 ;;
  esac
fi

# Optional model override from .env — read by bot.py itself, echoed here for visibility
[[ -n "${KITPRI_BOT_CKPT:-}" ]] && echo "using custom checkpoint: $KITPRI_BOT_CKPT"
[[ -n "${KITPRI_BOT_THRESHOLD:-}" ]] && echo "using custom threshold: $KITPRI_BOT_THRESHOLD"

# ── Production-identity guard ────────────────────────────────────────────────
# The 24/7 cloud VM owns @kitpribot. Telegram gives a token to the NEWEST
# poller, so starting that identity here would silently hijack production.
# Ask Telegram whose token this is and refuse the production identity.
PROD_BOT_USERNAME="kitpribot"
tok_user=$(curl -sm 5 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe" 2>/dev/null \
           | sed -n 's/.*"username":"\([^"]*\)".*/\1/p') || true
if [[ "${tok_user:-}" == "$PROD_BOT_USERNAME" && "${KITPRI_FORCE_PROD:-}" != "1" ]]; then
  cat <<'GUARD'

✗ REFUSING to start: the token in .env belongs to @kitpribot — the PRODUCTION
  bot that runs 24/7 on the Oracle VM. Starting it here would steal polling
  from the cloud (Telegram always favors the newest poller) and users' messages
  would land on this laptop instead.

  For local testing, use your OWN test bot:
    1. remove the TELEGRAM_BOT_TOKEN line from .env
    2. run ./start.sh — it walks you through creating one with @BotFather

  (Deliberately taking over production from this machine requires:
     KITPRI_FORCE_PROD=1 ./start.sh
   — stop the cloud service first: ssh kitpri-vm 'sudo systemctl stop kitpri-bot')

GUARD
  exit 1
fi
[[ -n "${tok_user:-}" ]] && echo "token identity: @${tok_user} (not production — ok)"

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
  echo "LOCAL bot RUNNING on this machine (pid $(pgrep -f 'telegram_bot/bot.py' | tr '\n' ' ')) — follow with: ./start.sh log"
  # Same-token clash guard: if another instance (usually the 24/7 CLOUD bot)
  # already polls this token, Telegram rejects one of them — detect & stand down.
  sleep 5
  if grep -q "Conflict" bot.log; then
    echo ""
    echo "⚠ CONFLICT: another bot instance is ALREADY polling this token —"
    echo "  almost certainly the 24/7 cloud bot on the Oracle VM."
    echo "  Two pollers on one token fight and both drop messages, so this"
    echo "  LOCAL bot has been stopped again. The cloud bot is unaffected."
    echo ""
    echo "  To test locally, use your OWN test bot (separate token):"
    echo "    1. remove the token line from .env"
    echo "    2. run ./start.sh — it walks you through @BotFather"
    ./stop.sh >/dev/null
    exit 1
  fi
else
  echo "bot FAILED to start — last log lines:"
  tail -5 bot.log
  exit 1
fi
