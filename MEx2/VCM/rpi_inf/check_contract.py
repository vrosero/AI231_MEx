#!/usr/bin/env python3
"""check_contract.py — verify the deployed inference stack still matches training.

Three checks (all must pass before a demo):

1. FEATURE PARITY   rpi/runtime/features.logmel_from_array vs
                    model/vcm_model.wav_to_logmel_np on real waveforms
                    (the runtime must compute byte-identical log-mels).
2. ONNX PARITY      torch checkpoint vs deployed rpi/model/crnn.onnx
                    (per-element max diff on random + real inputs).
3. WAKE PARITY      torch WakeCNN (vendored wake_cnn_base) vs deployed
                    rpi/model/jarvis_cnn.onnx.

Usage:
    python3 check_contract.py            # run all three
    python3 check_contract.py --quick    # fewer samples
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RPI = os.path.join(ROOT, "rpi")
MODEL = os.path.join(ROOT, "model")

sys.path.insert(0, os.path.join(RPI, "runtime"))
sys.path.insert(0, MODEL)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_features(n=24):
    """training front-end vs runtime front-end on the same waveforms."""
    import vcm_model as V
    import features as F

    rows = V.load_split_rows("test", verbose=False, sup=True)
    rng = np.random.default_rng(42)
    picks = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    worst = 0.0
    bad = 0
    for i in picks:
        r = rows[int(i)]
        a = V.decode_audio(r["audio"])
        if a is None:
            continue
        x, sr = a
        f_train = V.wav_to_logmel_np(x, sr)
        f_rt = F.logmel_from_array(x, sr)
        if f_train is None or f_rt is None:
            bad += 1
            continue
        worst = max(worst, float(np.abs(f_train - f_rt).max()))
    print("1) FEATURE PARITY: worst |train-runtime| = %.3e  (%d clips, %d undecodable)"
          % (worst, len(picks), bad))
    return worst, bad == 0


def check_onnx(n=8):
    """torch ckpt vs deployed ONNX on random + real cached mels."""
    import torch
    import onnxruntime as ort
    import vcm_model as V

    ck = torch.load(os.path.join(MODEL, "ckpt", "crnn.pt"),
                    map_location="cpu", weights_only=False)
    model = V.build_model(ck["arch"], ck["n_fine"], ck["n_intent"],
                          base=ck["base"], dropout=ck.get("dropout", 0.35))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    onnx_p = os.path.join(RPI, "model", "crnn.onnx")
    sess = ort.InferenceSession(onnx_p, providers=["CPUExecutionProvider"])

    worst = 0.0
    for b in (1, 8):
        x = np.random.randn(b, 1, 40, 96).astype(np.float32)
        with torch.no_grad():
            tf_, ti_ = model(torch.from_numpy(x))
        of, oi = sess.run(None, {"logmel": x})
        worst = max(worst,
                    float(np.abs(tf_.numpy() - of).max()),
                    float(np.abs(ti_.numpy() - oi).max()))

    # real mels from the cache
    mels = np.load(os.path.join(MODEL, "cache", "testsup.npy"), mmap_mode="r")
    idx = np.linspace(0, mels.shape[0] - 1, min(n, mels.shape[0])).astype(int)
    x = np.asarray(mels[idx], dtype=np.float32)
    with torch.no_grad():
        tf_, ti_ = model(torch.from_numpy(x))
    of, oi = sess.run(None, {"logmel": x})
    worst = max(worst,
                float(np.abs(tf_.numpy() - of).max()),
                float(np.abs(ti_.numpy() - oi).max()))
    print("2) ONNX PARITY (CRNN): worst torch-vs-ORT diff = %.3e  (B=1/8 random + %d real mels)"
          % (worst, len(idx)))
    return worst


def check_wake(n=12):
    """vendored torch WakeCNN vs deployed jarvis_cnn.onnx."""
    import torch
    import onnxruntime as ort

    base = load_module("wcb", os.path.join(MODEL, "wakeword", "cnn", "wake_cnn_base.py"))
    ck = torch.load(os.path.join(MODEL, "wakeword", "cnn_out", "model.pt"),
                    map_location="cpu", weights_only=False)
    model = base.WakeCNN(n_mels=ck["n_mels"], t_frames=ck["t_frames"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    wrapped = torch.nn.Sequential(model, torch.nn.Sigmoid()).eval()

    onnx_p = os.path.join(RPI, "model", "jarvis_cnn.onnx")
    sess = ort.InferenceSession(onnx_p, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n):
        m = rng.normal(-3.0, 3.0, size=(1, 40, 98)).astype(np.float32)
        with torch.no_grad():
            tp = float(wrapped(torch.from_numpy(m).unsqueeze(0)))
        op = float(sess.run(["prob"], {"mel": m[None]})[0][0])
        worst = max(worst, abs(tp - op))
    print("3) WAKE PARITY: worst torch-vs-ORT diff = %.3e  (%d random mels)" % (worst, n))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    f_worst, f_ok = check_features(n=8 if args.quick else 24)
    o_worst = check_onnx(n=4 if args.quick else 12)
    w_worst = check_wake(n=4 if args.quick else 12)

    ok = f_ok and f_worst < 1e-4 and o_worst < 1e-4 and w_worst < 1e-4
    print("\nCONTRACT %s" % ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())