#!/usr/bin/env bash
# ============================================================================
#  run_all.sh — ONE-COMMAND REPRODUCIBLE PIPELINE
#
#  Redoes the whole lifecycle from a clean checkout:
#    1. prepare    — download HF shards (cached) + build log-mel feature cache
#    2. train      — CRNN, 60 epochs, seed 42, MPS (Apple silicon) or CPU
#    3. post_train — eval test+holdout -> single-file ONNX -> deploy to
#                    rpi/model -> regenerate report figures -> print numbers
#    4. wake       — train the "Hello Jarvis" CNN (train_cnn.py) and
#                    deploy jarvis_cnn.onnx + meta to rpi/model
#    5. bench      — rpi_inf/perf.py bench + wake (accuracy, FAR, latency, RTF)
#    6. docs       — rebuild VCM_Report.pdf, VCM_Slides.pptx, vcm_deck.pptx
#
#  Usage:
#      bash run_all.sh              # full pipeline (~3 h on MPS)
#      bash run_all.sh --skip-train # reuse the existing checkpoint (steps 3-6)
#      bash run_all.sh --skip-wake  # keep the deployed wake CNN
#      bash run_all.sh --skip-docs  # no PDF/PPTX rebuild
#
#  Requirements: run `bash setup_env.sh` once — it creates the three project
#  venvs this script uses (.venv_train, .venv_ww, .venv_inf). Paths can be
#  overridden:  TRAIN_PY=... WW_PY=... INF_PY=... bash run_all.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

TRAIN_PY="${TRAIN_PY:-$PWD/.venv_train/bin/python}"
WW_PY="${WW_PY:-$PWD/.venv_ww/bin/python}"
INF_PY="${INF_PY:-$PWD/.venv_inf/bin/python}"

SKIP_TRAIN=0; SKIP_WAKE=0; SKIP_DOCS=0
for a in "$@"; do
  case "$a" in
    --skip-train) SKIP_TRAIN=1 ;;
    --skip-wake)  SKIP_WAKE=1 ;;
    --skip-docs)  SKIP_DOCS=1 ;;
    *) echo "unknown arg: $a"; exit 2 ;;
  esac
done

# preflight — fail fast with the fix, instead of halfway through
for py in "$TRAIN_PY" "$INF_PY"; do
  if [ ! -x "$py" ]; then
    echo "missing $py" >&2
    echo "  -> run: bash setup_env.sh   (or override with TRAIN_PY=... INF_PY=...)" >&2
    exit 1
  fi
done
if [ "$SKIP_WAKE" -eq 0 ] && [ ! -x "$WW_PY" ]; then
  echo "missing $WW_PY" >&2
  echo "  -> run: bash setup_env.sh   (or --skip-wake to keep the deployed wake CNN)" >&2
  exit 1
fi

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

# --- 1. prepare -------------------------------------------------------------
step "1/6 prepare (feature cache; skips shards that already exist)"
(cd model && "$TRAIN_PY" vcm_model.py prepare --splits train,test,holdout \
    --sup --tag sup)

# --- 2. train ---------------------------------------------------------------
if [ "$SKIP_TRAIN" -eq 0 ]; then
  step "2/6 train CRNN (60 epochs, seed 42)"
  (cd model && ulimit -n 4096 && "$TRAIN_PY" vcm_model.py train \
      --arch crnn --epochs 60 --sup --cache-tag sup --tag crnn \
      2>&1 | tee run_outputs/logs/train_crnn_full.log)
else
  step "2/6 train SKIPPED (reusing model/ckpt/crnn.pt)"
fi

# --- 3. post_train ----------------------------------------------------------
step "3/6 post_train (eval -> ONNX -> deploy -> figures)"
(cd model && "$TRAIN_PY" post_train.py)

# --- 4. wake ----------------------------------------------------------------
if [ "$SKIP_WAKE" -eq 0 ]; then
  step "4/6 wake CNN train (30 epochs, seed 42) + deploy"
  (cd model/wakeword/cnn && "$WW_PY" train_cnn.py \
      --dataset ../../../dataset/Jarvis \
      --out ../cnn_out --epochs 30 --batch 64 --lr 1e-3 --seed 42)
  cp model/wakeword/cnn_out/jarvis_cnn.onnx rpi/model/jarvis_cnn.onnx
  cp model/wakeword/cnn_out/jarvis_cnn.meta.json rpi/model/jarvis_cnn.meta.json
  echo "deployed jarvis_cnn.onnx + meta -> rpi/model/"
else
  step "4/6 wake SKIPPED (keeping deployed rpi/model/jarvis_cnn.onnx)"
fi

# --- 5. bench ---------------------------------------------------------------
step "5/6 benchmarks (rpi_inf/perf.py)"
(cd rpi_inf && "$INF_PY" perf.py bench)
(cd rpi_inf && "$INF_PY" perf.py wake)

# --- 6. docs ----------------------------------------------------------------
if [ "$SKIP_DOCS" -eq 0 ]; then
  step "6/6 rebuild report PDF + slide decks"
  (cd report && "$TRAIN_PY" make_figures.py 2>/dev/null || \
     (cd figures && "$TRAIN_PY" make_figures.py))
  (cd report && "$TRAIN_PY" make_pdf.py)
  (cd report && "$TRAIN_PY" make_pptx.py)
  (cd report && "$TRAIN_PY" make_deck.py)
else
  step "6/6 docs SKIPPED"
fi

step "DONE — numbers above; paste into report/deck as needed"