#!/usr/bin/env python3
"""perf.py - record & measure VCM model inference performance (rpi_inf).

This is the model-inference measurement home (rpi_inf/): benchmarks, eval
parity checks and the in-program session recorder. It lives SEPARATELY from
rpi/ so the deployed runtime folder stays executable-only.

Two halves in one file:

A) In-program recorder (used by runtime/vcm.py while the assistant runs):
       import perf
       perf.session_start(cfg, ...)          # at program start
       perf.add("classify", ms)              # per-stage timing
       perf.event("wake")                    # counted events
       perf.command(fine=..., ok=...)        # one finished command
       perf.session_end()                    # on exit: saves JSON + PRINTS
                                             # the end-of-program report
   Output: perf_results/session_<ts>.json + a console table (uptime, wakes,
   commands, ok-rate, per-stage latency mean/p50/p95/max).

B) Standalone benchmarks - the deck fields, reproducible on any Pi
   ("Pi 4 latency reproduced by the posted script"):
       python3 perf.py bench   # VCM: accuracy, false-accept rate, latency p95,
                               #      RTF, onnxruntime thread sweep
       python3 perf.py wake    # wake word: keyword accuracy + false-accept rate
       python3 perf.py live    # re-print the latest session report
       python3 perf.py report  # aggregate everything in perf_results/

No torch / flask / sounddevice - numpy + onnxruntime + stdlib only, so it runs
anywhere the runtime runs. Accuracy parts additionally use the HF parquet
labels when huggingface_hub + pyarrow are installed (they are on the dev
machine; on the Pi install them with:
    pip install huggingface_hub pyarrow soundfile).

Deck field mapping ("Validation on the Raspberry Pi 4"):
    Keyword acc        -> perf.py wake   (positive recall)
    Intent acc         -> perf.py bench  (intent head on the test split)
    False-accept rate  -> perf.py wake (negative/noise) + bench (OOS @ MIN_CONF)
    Latency p95 / RTF  -> perf.py bench  (100 warm-up + 100 timed runs, p95 / 1.6 s window)
    onnxruntime threads-> perf.py bench  (intra_op 1/2/4 sweep)
"""
import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RPI = os.path.join(HERE, "..", "rpi")                 # the deployed runtime
if os.path.isdir(os.path.join(RPI, "runtime")):
    sys.path.insert(0, os.path.join(RPI, "runtime"))  # features.py, wakegate.py
    MODEL_DIR = os.path.join(RPI, "model")            # rpi/model: deployed onnx
else:
    # Pi layout: runtime/model live beside rpi_inf (no rpi/ level)
    RPI = os.path.dirname(HERE)
    sys.path.insert(0, os.path.join(RPI, "runtime"))  # features.py, wakegate.py
    MODEL_DIR = os.path.join(RPI, "model")            # deployed onnx + meta

RESULTS_DIR = os.path.join(HERE, "perf_results")
CACHE_DIR = os.path.join(HERE, "..", "model", "cache")
WAKE_DATA_DIR = (os.path.join(HERE, "..", "dataset", "Jarvis")
                 if os.path.isdir(os.path.join(HERE, "..", "dataset", "Jarvis"))
                 else os.path.join(RPI, "..", "dataset", "Jarvis"))

# The live runtime rejects a command below this fine-confidence (vcm.py
# MIN_CONF). The bench FAR is reported at this operating point.
MIN_CONF = 0.25
WINDOW_S = 96 * 200 / 12000.0        # 96 log-mel frames x hop 200 @ 12 kHz


def _now():
    return time.strftime("%Y%m%d-%H%M%S")


def _pctl(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q)) if len(a) else None


# ---------------------------------------------------------------------------
# A) In-program session recorder
# ---------------------------------------------------------------------------
class _Acc:
    """Running stats for one timed stage (constant memory)."""

    __slots__ = ("n", "total", "mx", "vals")

    def __init__(self, keep=400):
        self.n = 0
        self.total = 0.0
        self.mx = 0.0
        self.vals = []          # last `keep` values, for p50/p95

    def add(self, ms):
        self.n += 1
        self.total += ms
        if ms > self.mx:
            self.mx = ms
        self.vals.append(ms)
        if len(self.vals) > self.vals_max:
            self.vals.pop(0)

    vals_max = 400

    def summary(self):
        if not self.n:
            return None
        return {"n": self.n, "mean_ms": self.total / self.n,
                "p50_ms": _pctl(self.vals, 50), "p95_ms": _pctl(self.vals, 95),
                "max_ms": self.mx}


_enabled = True
_s = None            # active session dict
_t0 = None           # session wall-clock start


def session_start(cfg=None, **meta):
    """Begin a recording session (call once when the program starts)."""
    global _s, _t0
    if not _enabled:
        return
    _s = {"type": "session", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
          "meta": meta or {}, "stages": {}, "events": {}, "commands": [],
          "outcomes": {}}
    if isinstance(cfg, dict):
        # Keep only non-secret keys in the report.
        _s["meta"]["config"] = {k: v for k, v in cfg.items()
                                if not any(s in k for s in
                                           ("password", "api_key", "email"))}
    _t0 = time.time()


def add(stage, ms, **fields):
    """Record one timed sample of `stage` (milliseconds). No-op if disabled."""
    if not (_enabled and _s is not None):
        return
    acc = _s["stages"].get(stage)
    if acc is None:
        acc = _s["stages"][stage] = _Acc()
    acc.add(float(ms))
    if fields:
        _s.setdefault("stage_fields", {}).setdefault(stage, []).append(fields)


def event(name, **fields):
    """Count an event (e.g. a wake-word hit). No-op if disabled."""
    if not (_enabled and _s is not None):
        return
    _s["events"][name] = _s["events"].get(name, 0) + 1
    if fields:
        _s.setdefault("event_fields", {}).setdefault(name, []).append(fields)


def command(**kw):
    """One finished command attempt (recognized or not)."""
    if not (_enabled and _s is not None):
        return
    _s["commands"].append(kw)
    if len(_s["commands"]) > 2000:
        _s["commands"].pop(0)
    key = "%s|%s" % (kw.get("recognized"), kw.get("ok"))
    _s["outcomes"][key] = _s["outcomes"].get(key, 0) + 1


def session_end(print_report=True):
    """Finalize the session: save JSON and print the end-of-program report."""
    global _s
    if not (_enabled and _s is not None):
        return None
    _s["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _s["uptime_s"] = round(time.time() - (_t0 or time.time()), 1)
    # Freeze live _Acc objects into plain dicts (json-serializable).
    _s["stages"] = {k: (v.summary() if isinstance(v, _Acc) else v)
                    for k, v in _s["stages"].items()}
    _s["summary"] = _session_summary(_s)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "session_%s.json" % _now())
    with open(path, "w") as f:
        json.dump(_s, f, indent=1, default=float)
    if print_report:
        print_session_report(_s, path)
    _s = None
    return path


def _session_summary(s):
    cmds = s["commands"]
    n_cmd = len(cmds)
    n_ok = sum(1 for c in cmds if c.get("ok"))
    n_rec = sum(1 for c in cmds if c.get("recognized"))
    return {"uptime_s": s["uptime_s"], "wakes": s["events"].get("wake", 0),
            "commands": n_cmd, "recognized": n_rec, "ok": n_ok,
            "ok_rate": round(n_ok / n_cmd, 3) if n_cmd else None,
            "recognised_rate": round(n_rec / n_cmd, 3) if n_cmd else None}


def print_session_report(s, path=None):
    """Human-readable end-of-program report (also printed by vcm.py)."""
    summ = s.get("summary") or _session_summary(s)
    line = "=" * 66
    print("\n" + line)
    print("VCM session report   %s -> %s   (uptime %.0f s)"
          % (s.get("started"), s.get("ended"), summ["uptime_s"]))
    print(line)
    print("wake-word hits      : %d" % summ["wakes"])
    print("command attempts    : %d" % summ["commands"])
    print("recognized (conf>=%.2f): %d (%s)"
          % (MIN_CONF, summ["recognized"],
             _fmt_pct(summ["recognised_rate"])))
    print("actions ok          : %d (%s)"
          % (summ["ok"], _fmt_pct(summ["ok_rate"])))
    stages = s.get("stages") or {}
    if stages:
        print("\n%-22s %7s %9s %9s %9s %9s"
              % ("stage", "n", "mean", "p50", "p95", "max") + "  (ms)")
        for name in ("wake_gate_frame", "trim", "features", "classify",
                     "action", "respond", "command_total"):
            acc = stages.get(name)
            if not acc:
                continue
            d = acc.summary() if isinstance(acc, _Acc) else acc
            if d:
                print("%-22s %7d %9.2f %9.2f %9.2f %9.2f"
                      % (name, d["n"], d["mean_ms"], d["p50_ms"],
                         d["p95_ms"], d["max_ms"]))
    cmds = s.get("commands") or []
    if cmds:
        per = {}
        for c in cmds:
            lbl = c.get("fine") or "(no speech)"
            per.setdefault(lbl, [0, 0])
            per[lbl][0] += 1
            per[lbl][1] += 1 if c.get("ok") else 0
        print("\nper-label (attempts / ok):")
        for lbl in sorted(per):
            print("  %-32s %4d / %4d" % (lbl, per[lbl][0], per[lbl][1]))
    if path:
        print("\nsaved: %s" % os.path.relpath(path, HERE))
    print("deck fields: latency p95/RTF from `python3 perf.py bench`,")
    print("            keyword acc + FAR from `python3 perf.py wake`.")
    print(line + "\n")


def _fmt_pct(x):
    return "-" if x is None else "%.1f%%" % (100.0 * x)


# ---------------------------------------------------------------------------
# B1) bench - VCM accuracy / FAR / latency / RTF / threads
# ---------------------------------------------------------------------------

def _load_mels():
    """Test-split log-mels from the training cache (exactly what the Pi
    runtime computes). Returns (mels[N,40,96], n) or (None, 0)."""
    p = os.path.join(CACHE_DIR, "testsup.npy")   # supplemental-merged test cache
    if not os.path.isfile(p):
        print("!! %s not found - run model/vcm_model.py prepare first" % p)
        return None, 0
    mels = np.load(p, allow_pickle=True)
    if mels.dtype == object:
        mels = np.stack([np.asarray(m, dtype=np.float32) for m in mels])
    mels = np.asarray(mels, dtype=np.float32)
    if mels.ndim == 4 and mels.shape[1] == 1:
        mels = mels[:, 0]
    return mels, mels.shape[0]


def _load_test_labels(n):
    """True fine/intent labels for the cached test mels, in cache order.
    Returns (fine[N], intent[N], oos[N]) or (None, None, None)."""
    try:
        sys.path.insert(0, os.path.join(HERE, "..", "model"))  # vcm_model.py
        import vcm_model as V
        rows = V.load_split_rows("test", verbose=False, sup=True)
    except Exception as e:
        print("!! labels unavailable (%s) - latency-only bench" % e)
        print("   (pip install huggingface_hub pyarrow soundfile enables it)")
        return None, None, None
    if len(rows) != n:
        print("!! %d rows vs %d cached mels - labels misaligned, "
              "latency-only bench" % (len(rows), n))
        return None, None, None
    fine, intent, oos = [], [], []
    for r in rows:
        is_oos = bool(r.get("out_of_scope")) or \
            str(r.get("command", "")).upper() == "OUT_OF_SCOPE"
        oos.append(is_oos)
        if is_oos:
            fine.append(None)
            intent.append(None)
        else:
            fl = V.fine_label_of(r.get("command"), r.get("slot_value"))
            fine.append(fl)
            intent.append(r.get("command"))
    return fine, intent, oos


def _f1_from_confusion(cm):
    """Macro-F1 from a confusion matrix (no sklearn)."""
    f1s = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def bench(args):
    import onnxruntime as ort
    from features import VCMClassifier

    onnx = os.path.join(MODEL_DIR, "crnn.onnx")
    meta_p = os.path.join(MODEL_DIR, "crnn_meta.json")
    clf = VCMClassifier(onnx, meta_p)
    meta = clf.meta
    mels, n = _load_mels()
    if mels is None:
        return 1
    if args.n and args.n < n:
        mels = mels[:args.n]
        n = args.n
    print("[bench] model: %s (%.1f MB, %s, params %.2f M)"
          % (os.path.basename(onnx), os.path.getsize(onnx) / 1e6,
             meta.get("arch"), meta.get("params_M", 0)))
    print("[bench] test mels: %d  (window %.2f s)" % (n, WINDOW_S))

    res = {"type": "bench", "when": _now(),
           "model": {"file": os.path.basename(onnx),
                     "size_MB": round(os.path.getsize(onnx) / 1e6, 3),
                     "arch": meta.get("arch"),
                     "params_M": meta.get("params_M"),
                     "best_epoch": meta.get("best_epoch"),
                     "val_intent_acc": meta.get("best_val_intent_acc")},
           "env": _env(), "n_test": n}

    # ---- latency: warm-up then timed single runs --------------------------
    # Prof's method (efficiency slides 81-82 + class): WARMUP warm-up
    # runs, then RUNS timed runs (defaults 100 + 100).
    runs = max(10, args.runs)
    warmup = max(10, args.warmup)
    idx = np.linspace(0, n - 1, min(runs, n)).astype(int)
    for _ in range(warmup):
        clf.predict(mels[idx[0]])               # warm-up (JIT, caches)
    samples = np.take(mels, idx, axis=0)    # one [40,96] row per timed run
    lat = []
    for m in samples:
        t0 = time.perf_counter()
        clf.predict(m)
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.asarray(lat)
    res["latency_ms"] = {"n": int(lat.size), "warmup": int(warmup),
                         "mean": float(lat.mean()),
                         "p50": _pctl(lat, 50), "p95": _pctl(lat, 95),
                         "max": float(lat.max())}
    res["rtf"] = res["latency_ms"]["p95"] / (WINDOW_S * 1000.0)
    print("[bench] latency ms: mean %.2f  p50 %.2f  p95 %.2f  max %.2f "
          "-> RTF %.4f (p95 / %.2f s window)"
          % (res["latency_ms"]["mean"], res["latency_ms"]["p50"],
             res["latency_ms"]["p95"], res["latency_ms"]["max"],
             res["rtf"], WINDOW_S))

    # ---- onnxruntime thread sweep (deck: "onnxruntime . thr") ------------
    threads = [int(t) for t in str(args.threads).split(",") if t.strip()]
    res["threads"] = {}
    x = mels[idx[:max(5, min(20, idx.size))]]
    in_name = clf._in_name
    for t in threads:
        so = ort.SessionOptions()
        so.intra_op_num_threads = t
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(onnx, so,
                                    providers=["CPUExecutionProvider"])
        for j in range(2):                          # warm-up
            sess.run(None, {in_name: x[0][None, None]})
        ts = []
        for j in range(x.shape[0]):
            t0 = time.perf_counter()
            sess.run(None, {in_name: x[j][None, None]})
            ts.append((time.perf_counter() - t0) * 1000.0)
        res["threads"][t] = {"p50_ms": _pctl(ts, 50), "p95_ms": _pctl(ts, 95)}
        print("[bench] intra_op=%d  p50 %.2f ms  p95 %.2f ms"
              % (t, res["threads"][t]["p50_ms"], res["threads"][t]["p95_ms"]))

    # ---- accuracy + false-accept rate ------------------------------------
    fine_true, intent_true, oos = _load_test_labels(n)
    if fine_true is not None:
        probs, iprobs = [], []
        B = 256
        for i0 in range(0, n, B):
            xb = mels[i0:i0 + B][:, None, :, :]
            fl, il = clf._sess.run(None, {clf._in_name: xb})
            probs.append(_softmax(fl))
            iprobs.append(_softmax(il))
        probs = np.concatenate(probs)              # [N, 32]
        iprobs = np.concatenate(iprobs)            # [N, 19]
        fine_pred = np.argmax(probs, axis=1)
        intent_pred = np.argmax(iprobs, axis=1)
        conf = probs.max(axis=1)

        in_scope = np.asarray([not o for o in oos])
        known = [i for i, fl in enumerate(fine_true)
                 if fl in clf.fine_labels]
        fine_acc = float(np.mean([fine_pred[i] == clf.fine_labels.index(
            fine_true[i]) for i in known])) if known else None
        i_known = [i for i, it in enumerate(intent_true)
                   if it in clf.intent_labels]
        intent_acc = float(np.mean([intent_pred[i] == clf.intent_labels.index(
            intent_true[i]) for i in i_known])) if i_known else None

        # confusion (fine head, in-scope rows)
        labels = clf.fine_labels
        li = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for i in known:
            cm[li[fine_true[i]], fine_pred[i]] += 1
        macro_f1 = _f1_from_confusion(cm)

        # FAR on OUT_OF_SCOPE at the runtime operating point + sweep
        oos_idx = np.where(~in_scope)[0]
        far = {}
        for thr in (0.10, 0.25, 0.35, 0.50, 0.75, 0.90):
            far[str(thr)] = float(np.mean(conf[oos_idx] >= thr)) \
                if oos_idx.size else None
        # in-scope accepted accuracy at MIN_CONF (acc/FAR trade-off pair)
        acc_at = float(np.mean((fine_pred[in_scope] ==
                                [li.get(fine_true[i], -1)
                                 for i in np.where(in_scope)[0]])
                               & (conf[in_scope] >= MIN_CONF))) \
            if in_scope.any() else None

        res["accuracy"] = {
            "fine_acc": fine_acc, "intent_acc": intent_acc,
            "macro_f1_fine": macro_f1,
            "n_in_scope": int(in_scope.sum()), "n_oos": int(oos_idx.size),
            "accepted_acc_at_min_conf": acc_at,
            "far_oos_by_threshold": far,
            "far_oos_at_min_conf": far.get(str(MIN_CONF)),
        }
        print("[bench] fine acc %.4f  intent acc %.4f  macro-F1 %.4f "
              "(n_in_scope %d)"
              % (fine_acc, intent_acc, macro_f1, in_scope.sum()))
        if oos_idx.size:
            print("[bench] FAR on OUT_OF_SCOPE @ conf>=%.2f : %.4f "
                  "(sweep %s)" % (MIN_CONF, far.get(str(MIN_CONF)),
                                  ", ".join("%s:%.3f" % (k, v)
                                            for k, v in far.items())))

        # confusion CSV next to the json
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_p = os.path.join(RESULTS_DIR, "bench_confusion_test.csv")
        with open(csv_p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + labels)
            for i, l in enumerate(labels):
                w.writerow([l] + cm[i].tolist())
        res["confusion_csv"] = os.path.relpath(csv_p, HERE)
        print("[bench] confusion -> %s" % res["confusion_csv"])

    _save(res, "bench_%s.json" % _now())
    _print_deck_table(res)
    return 0


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _env():
    try:
        import onnxruntime as ort
        ort_v = ort.__version__
    except Exception:
        ort_v = "?"
    mem = None
    try:
        with open("/proc/meminfo") as f:            # Pi / Linux
            for ln in f:
                if ln.startswith("MemTotal"):
                    mem = ln.split()[1] + " kB"
                    break
    except OSError:
        pass
    return {"platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "machine": platform.machine(),
            "mem": mem, "onnxruntime": ort_v}


def _print_deck_table(res):
    acc = res.get("accuracy") or {}
    print("\n" + "=" * 60)
    print("deck: Validation on the Raspberry Pi 4")
    print("=" * 60)
    print("Model        : %s  %.2f MB  params %.2f M"
          % (res["model"]["arch"], res["model"]["size_MB"],
             res["model"]["params_M"] or 0))
    print("Keyword acc  : %s   (perf.py wake for the wake word)"
          % _fmt_pct(acc.get("fine_acc")))
    print("Intent acc   : %s" % _fmt_pct(acc.get("intent_acc")))
    print("False-accept : %s  (OOS @ conf>=%.2f)"
          % (_fmt_pct(acc.get("far_oos_at_min_conf")), MIN_CONF))
    print("Latency p95  : %.2f ms   RTF %.4f"
          % (res["latency_ms"]["p95"], res["rtf"]))
    thr = res.get("threads") or {}
    best = min(thr.items(), key=lambda kv: kv[1]["p95_ms"]) if thr else None
    print("ORT threads  : %s" % (best and "intra_op=%s (p95 %.2f ms)"
                                   % (best[0], best[1]["p95_ms"])))
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# B2) wake - keyword accuracy + false-accept rate for "Hello Jarvis"
# ---------------------------------------------------------------------------

def wake(args):
    import onnxruntime as ort
    from features import _load_wav_mono, _resample_linear
    import wakegate as WG

    model_p = os.path.join(MODEL_DIR, WG.CNN_MODEL)
    if not os.path.isfile(model_p):
        print("!! %s not found" % model_p)
        return 1
    sess = ort.InferenceSession(model_p,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    thr = float(args.threshold)

    sets = {"positive": None, "negative": None, "noise": None}
    base = os.path.join(WAKE_DATA_DIR)
    for k in sets:
        d = os.path.join(base, k)
        if not os.path.isdir(d):
            print("!! missing %s" % d)
            return 1
        wavs = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))
        if args.limit:
            wavs = wavs[:args.limit]
        sets[k] = [os.path.join(d, w) for w in wavs]

    scores = {"positive": [], "negative": [], "noise": []}
    lms = []                            # precomputed logmels, for latency
    for k, paths in sets.items():
        t0 = time.time()
        for p in paths:
            try:
                x, sr = _load_wav_sr(p)   # (x, sr) mono float32
                x16 = _resample_linear(x, sr, WG.CNN_RATE) if sr != WG.CNN_RATE else x
                lm = WG._cnn_logmel(x16.astype(np.float32))
                lms.append(lm)
                prob = sess.run(["prob"], {"mel": lm[None, None]})[0]
                scores[k].append(float(np.asarray(prob).ravel()[0]))
            except Exception as e:
                print("   skip %s (%s)" % (os.path.basename(p), e))
        print("[wake] %s: %d clips scored in %.1fs"
              % (k, len(scores[k]), time.time() - t0))

    pos = np.asarray(scores["positive"])
    neg = np.asarray(scores["negative"] + scores["noise"])
    kw_acc = float(np.mean(pos >= thr)) if pos.size else None
    far = float(np.mean(neg >= thr)) if neg.size else None
    # rank-based AUC (Mann-Whitney)
    auc = None
    if pos.size and neg.size:
        allv = np.concatenate([pos, neg])
        ranks = np.empty_like(allv)
        order = np.argsort(allv, kind="mergesort")
        ranks[order] = np.arange(1, allv.size + 1)
        auc = float((ranks[:pos.size].sum() -
                     pos.size * (pos.size + 1) / 2.0)
                    / (pos.size * neg.size))

    res = {"type": "wake", "when": _now(),
           "model": os.path.basename(model_p), "threshold": thr,
           "n": {k: len(v) for k, v in scores.items()},
           "keyword_acc": kw_acc, "far": far, "auc": auc,
           "score_stats": {k: {"mean": float(np.mean(v)) if len(v) else None,
                               "p95": _pctl(v, 95) if len(v) else None,
                               "max": float(np.max(v)) if len(v) else None}
                           for k, v in scores.items()},
           "sweep": {}}
    for t in (0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60):
        res["sweep"][str(t)] = {
            "keyword_acc": float(np.mean(pos >= t)) if pos.size else None,
            "far": float(np.mean(neg >= t)) if neg.size else None}
    print("[wake] keyword acc %s   FAR %s   AUC %s  (thr %.2f)"
          % (_fmt(kw_acc), _fmt(far), _fmt(auc), thr))
    print("[wake] sweep: " + ", ".join(
        "%s: acc %s / far %s" % (t, _fmt(d["keyword_acc"]),
                                 _fmt(d["far"]))
        for t, d in res["sweep"].items()))
    # ---- latency: prof's method (efficiency slides 81-82 + class) ---------
    # WARMUP warm-up runs on one clip, then RUNS timed runs, each timing
    # only the wake CNN's sess.run forward pass (no feature extraction).
    if lms:
        runs = max(10, int(getattr(args, "runs", 100)))
        warmup = max(10, int(getattr(args, "warmup", 100)))
        idx = np.linspace(0, len(lms) - 1, min(runs, len(lms))).astype(int)
        for _ in range(warmup):
            sess.run(["prob"], {"mel": lms[idx[0]][None, None]})
        lat = []
        for lm in np.take(np.asarray(lms), idx, axis=0):
            t0 = time.perf_counter()
            sess.run(["prob"], {"mel": lm[None, None]})
            lat.append((time.perf_counter() - t0) * 1000.0)
        lat = np.asarray(lat)
        wlat = {"n": int(lat.size), "warmup": int(warmup),
                "mean": float(lat.mean()), "p50": _pctl(lat, 50),
                "p95": _pctl(lat, 95), "max": float(lat.max())}
        res["wake_latency_ms"] = wlat
        res["wake_rtf"] = wlat["p95"] / (WG.CNN_T_FRAMES * WG.CNN_HOP
                                         / float(WG.CNN_RATE) * 1000.0)
        print(" wake latency ms: mean %.2f  p50 %.2f  p95 %.2f  max %.2f"
              " -> RTF %.4f (%d frames x hop %d @ %d Hz)"
              % (wlat["mean"], wlat["p50"], wlat["p95"], wlat["max"],
                 res["wake_rtf"], WG.CNN_T_FRAMES, WG.CNN_HOP, WG.CNN_RATE))
    _save(res, "wake_%s.json" % _now())
    return 0


def _fmt(x):
    return "-" if x is None else "%.4f" % x


def _load_wav_sr(path):
    """(x, sr) mono float32 via stdlib wave (same decoder as features.py)."""
    from features import _load_wav_mono_raw
    return _load_wav_mono_raw(path)


# ---------------------------------------------------------------------------
# B3) live / report
# ---------------------------------------------------------------------------

def live(args):
    """Re-print the latest session report."""
    path = _latest("session_*.json")
    if not path:
        print("no session_*.json in %s yet - run the program first"
              % os.path.relpath(RESULTS_DIR, HERE))
        return 1
    with open(path) as f:
        s = json.load(f)
    print_session_report(s, path)
    return 0


def report(args):
    """Aggregate all results in perf_results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(RESULTS_DIR, fn)) as f:
                d = json.load(f)
        except Exception:
            continue
        rows.append((fn, d))
    if not rows:
        print("nothing in %s yet" % os.path.relpath(RESULTS_DIR, HERE))
        return 1
    print("%-34s %-8s %s" % ("file", "type", "headline"))
    print("-" * 100)
    for fn, d in rows:
        t = d.get("type", "?")
        if t == "bench":
            head = ("fine %.3f intent %.3f FAR %.3f p95 %.1f ms RTF %.4f"
                    % (d.get("accuracy", {}).get("fine_acc") or -1,
                       d.get("accuracy", {}).get("intent_acc") or -1,
                       d.get("accuracy", {}).get("far_oos_at_min_conf") or -1,
                       d.get("latency_ms", {}).get("p95") or -1,
                       d.get("rtf") or -1))
        elif t == "wake":
            head = "kw acc %.3f FAR %.3f AUC %.3f (thr %.2f)" % (
                d.get("keyword_acc") or -1, d.get("far") or -1,
                d.get("auc") or -1, d.get("threshold", -1))
        else:
            s = d.get("summary", {})
            head = ("uptime %.0fs wakes %d cmds %d ok %s"
                    % (s.get("uptime_s") or 0, s.get("wakes") or 0,
                       s.get("commands") or 0, _fmt_pct(s.get("ok_rate"))))
        print("%-34s %-8s %s" % (fn, t, head))
    return 0


def _latest(pattern):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cands = [os.path.join(RESULTS_DIR, f) for f in
             sorted(os.listdir(RESULTS_DIR)) if f.startswith(pattern[:8])]
    cands = [c for c in cands if c.endswith(".json")]
    return cands[-1] if cands else None


def _save(res, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    p = os.path.join(RESULTS_DIR, name)
    with open(p, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("[saved] %s" % os.path.relpath(p, HERE))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="VCM performance recorder + benchmarks (deck metrics)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="VCM: accuracy, FAR, latency p95, RTF, threads")
    b.add_argument("--n", type=int, default=0, help="limit test clips (0=all)")
    b.add_argument("--runs", type=int, default=100, help="timed latency runs (prof: 100)")
    b.add_argument("--warmup", type=int, default=100, help="warm-up runs (prof: 100)")
    b.add_argument("--threads", default="1,2,4", help="intra_op sweep list")
    b.set_defaults(func=bench)

    w = sub.add_parser("wake", help="wake word: keyword acc + false-accept rate")
    w.add_argument("--threshold", type=float, default=0.5)
    w.add_argument("--limit", type=int, default=0, help="clips per set (0=all)")
    w.add_argument("--runs", type=int, default=100, help="timed latency runs (prof: 100)")
    w.add_argument("--warmup", type=int, default=100, help="warm-up runs (prof: 100)")
    w.set_defaults(func=wake)

    lv = sub.add_parser("live", help="re-print the latest session report")
    lv.set_defaults(func=live)

    rp = sub.add_parser("report", help="aggregate perf_results/*.json")
    rp.set_defaults(func=report)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()