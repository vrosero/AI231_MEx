#!/usr/bin/env bash
# run.sh — start the VCM voice assistant (after setup.sh).
#
#   bash run.sh              # live hands-free loop
#   bash run.sh --no-tts     # disable spoken responses (beeps stay on)
#
# Idempotent: creates the venv/config if missing (delegating to setup.sh),
# then execs runtime/vcm.py. No API keys, no accounts, no placeholders —
# the only config is Wi-Fi (set once with setup.sh or nmcli).
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "== no venv — running setup.sh first =="
  bash setup.sh
fi

if [ ! -f runtime/config.json ]; then
  cp runtime/config.example.json runtime/config.json
  echo "== created runtime/config.json (defaults are fine; edit only to change city/voice)"
fi

echo "== starting VCM (venv python) =="
exec "$PY" runtime/vcm.py "$@"