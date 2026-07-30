# Deploying the KitPri bot to Oracle Cloud (Always Free)

The bot uses **long polling** — it only makes *outbound* HTTPS calls to Telegram.
No public URL, no open inbound ports, no security-list changes needed. Any
always-on Linux VM works; Oracle's Always Free tier costs $0.

> **One poller per token.** Telegram allows only one `getUpdates` consumer.
> Before (or right after) the cloud bot starts, stop the laptop one:
> `./start.sh stop`. Two pollers = `Conflict: terminated by other
> getUpdates request` errors.

## 1. Create the VM

1. Sign up at <https://www.oracle.com/cloud/free/> (card required for identity
   verification only — Always Free resources never charge).
2. Console → **Compute → Instances → Create instance**.
3. Image: **Ubuntu 24.04**. Shape: **VM.Standard.A1.Flex** (Ampere ARM —
   Always Free allows up to 4 OCPU / 24 GB total). **1 OCPU / 6 GB** is plenty.
   - ARM is fine: the bot uses the **FP32** student, which runs anywhere.
     (Only the INT8 TorchScript model is x86/fbgemm-bound — the bot never loads it.)
   - If you hit *"Out of capacity"* for A1, retry later, try another
     availability domain, or fall back to **VM.Standard.E2.1.Micro** (x86,
     1 GB RAM — works, but add swap; see Troubleshooting).
4. Add your SSH **public** key (`cat ~/.ssh/id_ed25519.pub`), create, and copy
   the instance's **public IP**.

## 2. Install the bot

```bash
ssh ubuntu@<PUBLIC_IP>

sudo apt-get update
sudo apt-get install -y python3-venv libsndfile1 ffmpeg git

git clone https://github.com/ayushpandey0201/kitpribot.git kitpri
cd kitpri

# one command does venv + CPU torch + all deps + kitpri package:
./start.sh setup
```

<details>
<summary>Manual equivalent of <code>start.sh setup</code></summary>

```bash
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt
venv/bin/python -m pip install -e . --no-deps
```

</details>

## 3. Token + smoke test

```bash
printf 'TELEGRAM_BOT_TOKEN=<token from @BotFather>\n' > .env
chmod 600 .env

# foreground test — Ctrl-C once you see "Bot is polling…"
set -a; source .env; set +a
venv/bin/python telegram_bot/bot.py
```

Send the bot a voice note from your phone while it runs — you should get
🍳 / 🔇 back.

## 4. Run forever with systemd

```bash
sudo cp telegram_bot/kitpri-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kitpri-bot

systemctl status kitpri-bot          # should be "active (running)"
journalctl -u kitpri-bot -f          # live logs
```

`Restart=always` + `enable` means the bot survives crashes **and** VM reboots.

## 5. Updating the bot

```bash
cd ~/kitpri
git pull
sudo systemctl restart kitpri-bot
```

Same flow ships a **new model**: commit the new `.pt` (add a `.gitignore`
negation like `!inference/my_new_model.pt` — checkpoints are ignored by
default), push, `git pull` + restart here. If the model is selected via
`KITPRI_BOT_CKPT` in `.env`, remember the VM has its own `.env` to update.
Details: [README.md § Swapping in a new model](README.md#swapping-in-a-new-model).

## Troubleshooting

- **`Conflict: terminated by other getUpdates request`** — another instance is
  polling with the same token (usually the laptop). `./start.sh stop`
  locally, then `sudo systemctl restart kitpri-bot`.
- **OOM / killed on E2.1.Micro (1 GB)** — add swap:
  ```bash
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  ```
- **Different clone path or user** — edit `User=`, `WorkingDirectory=`,
  `EnvironmentFile=`, `ExecStart=` in
  `/etc/systemd/system/kitpri-bot.service`, then
  `sudo systemctl daemon-reload && sudo systemctl restart kitpri-bot`.
- **Logs contain the token?** No — `bot.py` silences httpx request-URL logging,
  and journald output stays on the VM regardless.
