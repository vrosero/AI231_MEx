#!/usr/bin/env bash
# ============================================================================
#  setup_env.sh — create VCM's own Python environments (Mac side), once.
#
#    .venv_train   CRNN prepare/train/eval, ONNX, report figures/PDF/decks
#    .venv_ww      wake-CNN trainer (torch 2.8 — produced the deployed ONNX)
#    .venv_inf     rpi_inf: perf.py bench/wake, check_contract.py
#
#  The Pi keeps its own runtime venv, created ON the Pi by rpi/setup.sh
#  (never here — sounddevice/spidev/RPi.GPIO do not belong on a Mac).
#
#  Idempotent: existing venvs are left alone. Re-run freely.
#  Usage:  bash setup_env.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Same base interpreters the proven envs were built from:
TRAIN_BASE="${TRAIN_BASE:-/opt/anaconda3/envs/onit/bin/python3}"   # 3.12 (mlenv's base)
WW_BASE="${WW_BASE:-/usr/bin/python3}"                             # 3.9  (wake env's base)

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }

# --- .venv_train -------------------------------------------------------------
if [ ! -x .venv_train/bin/python ]; then
  step "creating .venv_train ($("$TRAIN_BASE" --version 2>&1))"
  "$TRAIN_BASE" -m venv .venv_train
  .venv_train/bin/python -m pip install --upgrade pip --quiet
  .venv_train/bin/python -m pip install -r requirements-train.txt --quiet
  ok ".venv_train ready"
else
  ok ".venv_train exists — skipped"
fi

# --- .venv_inf ---------------------------------------------------------------
if [ ! -x .venv_inf/bin/python ]; then
  step "creating .venv_inf ($("$TRAIN_BASE" --version 2>&1))"
  "$TRAIN_BASE" -m venv .venv_inf
  .venv_inf/bin/python -m pip install --upgrade pip --quiet
  .venv_inf/bin/python -m pip install -r requirements-inf.txt --quiet
  ok ".venv_inf ready"
else
  ok ".venv_inf exists — skipped"
fi

# --- .venv_ww ----------------------------------------------------------------
if [ ! -x .venv_ww/bin/python ]; then
  step "creating .venv_ww ($("$WW_BASE" --version 2>&1), torch 2.8)"
  "$WW_BASE" -m venv .venv_ww
  .venv_ww/bin/python -m pip install --upgrade pip --quiet
  .venv_ww/bin/python -m pip install -r requirements-ww.txt --quiet
  ok ".venv_ww ready"
else
  ok ".venv_ww exists — skipped"
fi

# --- verify imports ----------------------------------------------------------
step "verifying imports"
.venv_train/bin/python -c "import torch, numpy, soundfile, huggingface_hub, pyarrow, onnx, onnxruntime, matplotlib, reportlab, pptx, PIL; print('  .venv_train: torch', torch.__version__, '| ort', __import__('onnxruntime').__version__)"
.venv_inf/bin/python -c "import torch, numpy, soundfile, huggingface_hub, pyarrow, onnx, onnxruntime; print('  .venv_inf:   torch', torch.__version__, '| ort', __import__('onnxruntime').__version__)"
.venv_ww/bin/python -c "import torch, numpy, soundfile, onnx, onnxruntime; assert torch.__version__.startswith('2.8'), torch.__version__; print('  .venv_ww:    torch', torch.__version__, '| ort', __import__('onnxruntime').__version__)"

step "all three venvs ready — next: bash run_all.sh"