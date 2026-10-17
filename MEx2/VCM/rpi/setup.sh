#!/usr/bin/env bash
# ============================================================================
#  VCM — ONE-COMMAND SETUP
#
#  After  git clone <repo>  (this folder), run:
#
#      bash setup.sh
#
#  It does everything needed to reproduce the device on ANY Raspberry Pi:
#    1. system packages (PortAudio, mpv, SPI, venv, ffmpeg)
#    2. python virtualenv (.venv) + pip install -r requirements.txt
#    3. Wi-Fi (SSID/password) switch — the ONLY configuration the device needs
#    4. spoken responses: pre-recorded MP3s (shipped in audio/responses/)
#    5. sanity checks (model files, microphone)
#    6. asks "Activate the VCM program now? [Y/n]" -> hands over to run.sh
#
#  No API keys, no accounts, no smart-home credentials: weather is Open-Meteo
#  (keyless) and the "lights" are the ReSpeaker HAT's own LEDs.
#
#  Afterwards, start it any time with:   bash run.sh
#  (Non-interactive contexts skip the prompts automatically.)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"
CONFIG="runtime/config.json"
EXAMPLE="runtime/config.example.json"

say()  { printf '\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }

INTERACTIVE=0; [ -t 0 ] && INTERACTIVE=1
lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }
ask() { # ask "prompt" "default" -> sets REPLY
  REPLY="$2"
  [ "$INTERACTIVE" = "1" ] || return 0
  read -r -p "$1 [$2]: " REPLY </dev/tty || REPLY="$2"
}

# ----------------------------------------------------------------------------
say "VCM setup — $(pwd)"
[ -f "$PY" ] && ok "virtualenv already present (reusing)"
[ -f "$CONFIG" ] && ok "config.json already present"

# --- 1. system packages ------------------------------------------------------
NEED=0
command -v mpv >/dev/null 2>&1 || NEED=1
python3 -c "import ctypes.util as c; c.find_library('portaudio')" >/dev/null 2>&1 || NEED=1
dpkg -s portaudio19-dev >/dev/null 2>&1 || NEED=1
if [ "$NEED" = "1" ]; then
  say "installing system packages (sudo needed, ~2-5 min)"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y portaudio19-dev mpv python3-venv python3-pip \
                            build-essential libffi-dev python3-spidev \
                            alsa-utils ffmpeg git
  else
    warn "apt-get not found (not a Debian Pi?) — install PortAudio + mpv manually"
  fi
else
  ok "system packages already present"
fi

# --- 2. virtualenv + python deps ---------------------------------------------
if [ ! -x "$PY" ]; then
  say "creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi
say "installing python dependencies (requirements.txt)"
"$PY" -m pip install --upgrade pip --quiet
"$PY" -m pip install -r requirements.txt --quiet
# apa102-pi WITHOUT its deps: adafruit-circuitpython-bitbangio -> Adafruit-Blinka
# -> lgpio has no Pi wheel for py3.13 (source build needs swig) and would abort
# the whole install. leds.py only uses apa102_pi's spidev driver, so the deps
# are dead weight.
"$PY" -m pip install --no-deps apa102-pi --quiet
ok "python dependencies installed"

# --- 3. Wi-Fi — the only configuration ---------------------------------------
if [ ! -f "$CONFIG" ]; then
  cp "$EXAMPLE" "$CONFIG"
  ok "created $CONFIG from example"
fi
if [ "$INTERACTIVE" = "1" ]; then
  ask "  switch Wi-Fi now? (y/N)" "n"
  if [ "$(lower "$REPLY")" = "y" ]; then
    ask "  SSID" ""
    SS="$REPLY"; ask "  password" ""; PW="$REPLY"
    if [ -n "$SS" ]; then
      if [ -n "$PW" ]; then sudo nmcli device wifi connect "$SS" password "$PW" \
      else sudo nmcli device wifi connect "$SS"; fi || warn "wifi switch failed — set it later with: sudo nmcli device wifi connect <ssid> password <pw>"
      hostname -I || true
    fi
  fi
else
  warn "non-interactive: Wi-Fi prompt skipped — connect with: sudo nmcli device wifi connect <ssid> password <pw>"
fi

# --- 4. spoken responses (pre-recorded MP3s, shipped in audio/responses/) ----
# v6: responses are pre-generated MP3 files (piper "Jarvis" voice, synthesized
# offline by rpi_inf/gen_responses.py on a laptop). Nothing to download here;
# piper itself is no longer a runtime dependency (kept in requirements.txt as
# a comment for regeneration).
N_RESP=$(ls audio/responses/*.mp3 2>/dev/null | wc -l)
if [ "$N_RESP" -ge 39 ]; then
  ok "spoken responses: $N_RESP MP3s in audio/responses/"
else
  warn "spoken responses: only $N_RESP MP3s in audio/responses/ (expected 39) - some commands will answer with beeps only"
fi

# --- 5. sanity checks ---------------------------------------------------------
say "sanity checks"
for f in model/crnn.onnx model/crnn_meta.json \
         model/jarvis_cnn.onnx model/jarvis_cnn.meta.json; do
  [ -f "$f" ] && ok "model: $f" || warn "MISSING model file: $f"
done
for f in music/music_1.mp3 audio/timer/TIMER_10s.mp3 audio/timer/TIMER_30s.mp3 \
         audio/timer/TIMER_1m.mp3; do
  [ -f "$f" ] && ok "audio: $f" || warn "MISSING audio file: $f"
done
"$PY" runtime/vcm.py --list-mics 2>/dev/null | head -5 || warn "no mic list (normal off-Pi); on the Pi check the ReSpeaker HAT"

# --- 6. hand over -------------------------------------------------------------
cat <<'EOF'

Setup complete.
  - start later:             bash run.sh
  - benchmarks (on the dev machine):  ../rpi_inf/perf.py bench / wake
  - config (city/voice):     runtime/config.json   (re-run setup.sh to change)
EOF
ask "Activate the VCM program now? (Y/n)" "y"
if [ "$(lower "$REPLY")" != "n" ]; then
  exec bash run.sh "$@"
else
  echo "OK — start it any time with:  bash run.sh"
fi