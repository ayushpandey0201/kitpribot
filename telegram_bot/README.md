# KitPri v4 — Telegram Bot Demo

Live demo of the KitPri v4 classifier: send an audio file or voice message,
get back 🍳 **Cooking** or 🔇 **Not Cooking**.

Runs the **FP32 MobileNetV2 student** (threshold 0.44) through the unified
`kitpri.inference.Predictor` — the exact same preprocessing and model code as
`inference/predict.py`, so the bot cannot drift from the evaluated pipeline.

> **The bot is live 24/7 in the cloud** — it runs on an Oracle Always-Free VM
> under systemd, so `@kitpribot` answers even when every laptop is closed.
> Setup below is only needed to run your *own* instance; hosting details are
> in [Cloud hosting](#cloud-hosting--how-the-live-bot-runs) and [DEPLOY.md](DEPLOY.md).

## Setup (from the repo root)

**There is no manual setup.** The launcher bootstraps everything on first run
— creates `venv/`, installs CPU-only torch + all dependencies, installs the
`kitpri` package — then starts the bot:

```bash
echo 'TELEGRAM_BOT_TOKEN=<token from @BotFather>' > .env   # one-time, gitignored
./start.sh                                       # bootstraps + starts
```

Only external nicety: `ffmpeg` for voice notes (`brew install ffmpeg` /
`sudo apt-get install -y ffmpeg`) — the launcher warns if it's missing.
To bootstrap without starting (e.g. on a fresh server): `./start.sh setup`.

## Run — the easy way

```bash
./start.sh
```

That's the whole flow. On a machine with no token configured, the launcher
asks what you want:

1. **Run my own local test bot** — it walks you through @BotFather inside the
   terminal (send `/newbot`, pick a name like `My KitPri Test Bot`, a username
   like `ayush_kitpri_test_bot`, paste the token when prompted — input is
   hidden and saved to a private `.env`). Your local bot needs its *own*
   token: the official one is in use by the 24/7 cloud instance, and Telegram
   allows one poller per token.
2. **Model development only** — no bot, no token; it prints the local
   inference command and the exact steps to ship a new model to the cloud bot.

Already have a token? Put it in `.env` yourself and the menu never appears:

```
TELEGRAM_BOT_TOKEN=<token from @BotFather>
```

Day-to-day commands (all of these act on the **local machine only** — the
cloud bot lives on a different computer and is managed via
`ssh kitpri-vm 'sudo systemctl …'`):

```bash
./start.sh            # start a LOCAL bot (auto-restarts a running instance)
./stop.sh             # stop ALL local bot processes — never touches the cloud
./start.sh setup     # first-time bootstrap only (venv + deps), no start
./start.sh status     # is a LOCAL bot running?
./start.sh log        # follow the LOCAL bot's live log
```

If you start a local bot with the **same token** the cloud bot uses, the
launcher detects the Telegram `Conflict` within seconds, stops the local
instance automatically, and tells you how to get your own test token.

## Run — manual

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

| Flag          | Default                               | Description                         |
| ------------- | ------------------------------------- | ----------------------------------- |
| `--ckpt`      | `inference/student_mobilenet_fp32.pt` | model checkpoint                    |
| `--config`    | `configs/experiments/distill.yaml`    | audio profile + per-model threshold |
| `--threshold` | from config (0.44)                    | override decision threshold         |

## Swapping in a new model

Trained a better classifier? The bot picks it up in one of three ways — no
code changes needed. The `Predictor` auto-detects the format: TorchScript
archives, training checkpoints (`{"model_state": ...}` exactly as
`training/*.py` save them), or bare state dicts all load as-is.

**Option A — config-driven (recommended).** Add two lines to the repo-root
`.env` and restart; nothing else changes:

```bash
KITPRI_BOT_CKPT=inference/my_new_model.pt      # path relative to repo root
KITPRI_BOT_THRESHOLD=0.52                      # from YOUR validation sweep
```
```bash
./start.sh          # restarts with the new model — log line confirms it
```

**Option B — drop-in replace.** Overwrite
[`inference/student_mobilenet_fp32.pt`](../inference/student_mobilenet_fp32.pt)
with your new checkpoint and restart. Also update `threshold:` in
[`configs/models/mobilenetv2_student.yaml`](../configs/models/mobilenetv2_student.yaml)
— that's where the default 0.44 lives.

**Option C — one-off manual run.** `python telegram_bot/bot.py --ckpt path/to/model.pt --threshold 0.52`

**Rules that keep you honest:**
1. **Re-sweep the threshold on the validation set** for every new model — 0.44
   belongs to the current FP32 student, not to your new one
   (`results/.../student_threshold_sweep.csv` shows the method).
2. **Same architecture?** Nothing to do — MobileNetV2 and AST checkpoints are
   fingerprinted automatically. **New architecture?** Register it in
   [`src/kitpri/models/registry.py`](../src/kitpri/models/registry.py) and add its
   key fingerprint in [`src/kitpri/inference/predictor.py`](../src/kitpri/inference/predictor.py).
3. **Shipping to the cloud bot:** `.gitignore` blocks `*.pt` by default — add a
   negation (`!inference/my_new_model.pt`), commit, push, then
   `ssh kitpri-vm 'cd kitpri && git pull && sudo systemctl restart kitpri-bot'`.
   (Or skip git: `scp my_new_model.pt kitpri-vm:kitpri/inference/` + restart.)
   If you used Option A, remember the VM has its **own** `.env` — update it there too.

## Cloud hosting — how the live bot runs

The production instance runs on an **Oracle Cloud Always-Free VM** at $0/month
(deployed 30 Jul 2026). Because the bot uses **long polling** (outbound HTTPS
only), the VM needs no public URL, webhook, TLS certificate, or open inbound
port.

| Component | Detail |
| --- | --- |
| VM | VM.Standard.E2.1.Micro — 1 OCPU x86, 1 GB RAM — `ap-hyderabad-1` |
| OS / runtime | Ubuntu 24.04 · Python 3.12 venv · CPU-only torch |
| Memory | +2 GB swapfile (1 GB RAM alone is too tight for torch + model) |
| Model | FP32 MobileNetV2 student, threshold 0.44, CPU (≈7 s load, ≈670 MB steady) |
| Supervision | systemd [`kitpri-bot.service`](kitpri-bot.service): `Restart=always`, enabled at boot, `TimeoutStopSec=15` (PTB can hang on SIGTERM) |
| Secrets | `.env` (mode 600) delivered via `scp`, injected with systemd `EnvironmentFile` — never committed, never in argv |

Day-2 operations (from any machine with the deploy key):

```bash
ssh kitpri-vm systemctl status kitpri-bot     # health
ssh kitpri-vm journalctl -u kitpri-bot -f     # live logs
ssh kitpri-vm 'cd kitpri && git pull && sudo systemctl restart kitpri-bot'   # deploy update
```

Full from-scratch instructions (VM creation → swap → install → systemd →
troubleshooting): **[DEPLOY.md](DEPLOY.md)**.

> ⚠️ **One poller per token.** Telegram rejects concurrent `getUpdates`
> consumers (HTTP 409). While the cloud bot is running, do not start a local
> instance with the same token — stop one side first
> (`./start.sh stop` locally, or `sudo systemctl stop kitpri-bot`
> on the VM).

## Notes

- The token is also logged inside Telegram API URLs by `httpx` — bot logs are
  gitignored (`*.log`); never commit them.
- On Apple Silicon the model runs on MPS automatically; falls back to CPU.
- The retired v6 DeiT flow (`best_ckpt.pt`, threshold 0.72) is gone — it had
  weak class separation on realistic audio and was replaced by the verified
  v4 student (6/6 probability parity vs training).
