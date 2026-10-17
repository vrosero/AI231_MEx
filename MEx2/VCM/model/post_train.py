#!/usr/bin/env python3
"""Post-training pipeline for the CRNN (run AFTER train completes).

Steps:
  1. eval on test + holdout (writes run_outputs/eval/eval_{split}_crnn.json)
  2. convert the checkpoint to a single-file ONNX + meta
  3. deploy to rpi/model/ (crnn.onnx + _meta.json)
  4. regenerate report figures (fig_results reads the eval JSONs)
  5. print the numbers to paste into the report/deck
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Run with the project training venv (run_all.sh does): subprocesses then use
# the same interpreter — no hardcoded machine path.
PY = sys.executable
MODEL = os.path.join(HERE, "vcm_model.py")
CKPT = os.path.join(HERE, "ckpt", "crnn.pt")
ONNX_DIR = os.path.join(HERE, "run_outputs", "onnx")
RPI_MODEL = os.path.join(HERE, "..", "rpi", "model")
FIGDIR = os.path.join(HERE, "..", "report", "figures")


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        sys.exit(f"step failed: {cmd}")


def main():
    # 1. eval test + holdout (cache-tag sup = merged splits)
    for split in ("test", "holdout"):
        run([PY, MODEL, "eval", "--ckpt", CKPT, "--tag", "crnn",
             "--split", split, "--cache-tag", "sup"], cwd=HERE)

    # 2. single-file ONNX + meta
    out = os.path.join(ONNX_DIR, "crnn.onnx")
    run([PY, MODEL, "export", "--ckpt", CKPT, "--out", out], cwd=HERE)

    # 3. deploy to rpi/model
    os.makedirs(RPI_MODEL, exist_ok=True)
    for f in ("crnn.onnx", "crnn_meta.json"):
        shutil.copy2(os.path.join(ONNX_DIR, f), os.path.join(RPI_MODEL, f))
        print("deployed", f, "->", RPI_MODEL, flush=True)

    # 4. figures (fig_results auto-reads the eval JSONs)
    run([PY, os.path.join(FIGDIR, "make_figures.py")], cwd=FIGDIR)

    # 5. print numbers for the docs
    for split in ("test", "holdout"):
        p = os.path.join(HERE, "run_outputs", "eval", f"eval_{split}_crnn.json")
        d = json.load(open(p))
        print(f"\n== {split} ==")
        for k in ("intent_accuracy", "intent_macro_f1", "fine_accuracy",
                  "fine_macro_f1", "far_oos", "oos_recall"):
            if k in d:
                print(f"  {k}: {d[k]}")


if __name__ == "__main__":
    main()