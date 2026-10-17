#!/usr/bin/env python3
"""
Jarvis wake-word CNN training script.

Trained on the final dataset: positive 1200, positive_real 122, negative
3000, negative_real 694, noise 800. The room-negative test slice is the
full kept set of negative_real/ (block-interleaved). Same 694 negative_real
windows remain, block-interleaved (every 8th window -> test, every 8th+3
-> val, rest train) so overlapping windows from the same event never
straddle train/test.

Dataset (dataset/Jarvis):
  positive/       1200 wav  synthetic TTS "Jarvis" phrasings (6 x 200)
  positive_real_1/  62 wav  REAL recordings (hello/hey/jarvis/jarvis_please/wake_up)
  positive_real_2/  60 wav  REAL re-takes + 10 okay_jarvis (read via positive_real*)
  negative/       3000 wav  10 command words x 300 (no wake word)
  noise/           800 wav  MUSAN / ESC-50 environmental noise
  negative_real/   694 wav  REAL room windows (kept set; quarantine deleted)

Split design (source-disjoint, stratified by phrasing / word):
  positives (synthetic):  80% train / 10% val / 10% test   (per-clip groups)
  positives (real):       ~half train / ~20% val / ~30% test, stratified per
                          phrasing — real clips are scarce, so the test slice
                          is deliberately real-heavy (this is the metric that
                          matters: recall on real speech).
  negatives + noise:      80% train / 10% val / 10% test, stratified by word.
  Noise TRAIN slice doubles as the augmentation mixing source.

Model: same WakeCNN (249,281 params) and log-mel front-end as v2
  (16 kHz, 1.0 s, 40 mels, 98 frames) so the exported ONNX stays a
  drop-in replacement for rpi/model/jarvis_cnn.onnx.

Outputs (out_dir):
  model.pt, split.json, history.json, report.json, jarvis_cnn.onnx, meta.json
"""
import argparse, json, random, re, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---- reuse the exact front-end + model so ONNX stays compatible ------------
# (vendored into wake_cnn_base.py from the tree that produced the deployed
#  ONNX — no imports across repos)
import importlib.util
_BASE = Path(__file__).resolve().parent / "wake_cnn_base.py"
_spec = importlib.util.spec_from_file_location("wcb", str(_BASE))
wcb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wcb)

SR, N, N_MELS, N_FFT, HOP, T_FRAMES = wcb.SR, wcb.N, wcb.N_MELS, wcb.N_FFT, wcb.HOP, wcb.T_FRAMES
logmel, WIN_FN, MEL_BANK = wcb.logmel, wcb.WIN_FN, wcb.MEL_BANK
WakeCNN, load_wav, load_wav_fixed = wcb.WakeCNN, wcb.load_wav, wcb.load_wav_fixed
_resample = wcb._resample

SEED = 42
PHRASE_RE = re.compile(r"^(hello_jarvis|hey_jarvis|jarvis_please|jarvis_wake_up|jarvis)_\d+$")
WORD_RE = re.compile(r"^([a-z]+)_\d+\.wav$", re.I)


def phrase_of(path: str) -> str:
    """Phrasing label for positives — synthetic (Hello_Jarvis_0.wav) or real
    (hello_jarvis_1.wav). Case-insensitive, handles both naming styles."""
    stem = Path(path).stem.lower()
    for ph in ("hello_jarvis", "hey_jarvis", "jarvis_please", "jarvis_wake_up"):
        if stem.startswith(ph + "_") or stem == ph:
            return ph
    if stem.startswith("okay_jarvis"):
        return "okay_jarvis"
    if stem.startswith("jarvis"):
        return "jarvis"
    return "other"


def word_of(path: str) -> str:
    """Command word for negatives (down_1000.wav -> 'down')."""
    m = WORD_RE.match(Path(path).name)
    return m.group(1).lower() if m else "other"


def stratified_split(items, strat_fn, fracs=(0.8, 0.1, 0.1), seed=SEED):
    """Stratified train/val/test split at clip level, stratified by strat_fn.
    Every clip in this dataset is an independent recording/render (its own
    'speaker'), so clip-level splitting is group-disjoint by construction.
    Guarantees >=1 clip per slice whenever a stratum has >=3 clips.
    Returns (train, val, test)."""
    rng = random.Random(seed)
    buckets = {}
    for it in items:
        buckets.setdefault(strat_fn(it[0]), []).append(it)
    tr, va, te = [], [], []
    for key in sorted(buckets):
        grp = buckets[key][:]
        rng.shuffle(grp)
        n = len(grp)
        n_te = int(round(n * fracs[2]))
        n_va = int(round(n * fracs[1]))
        n_tr = n - n_va - n_te
        if n_te < 1 and n >= 3:          # guarantee >=1 test clip per stratum
            n_te, n_tr = 1, n_tr - 1
        if n_va < 1 and n >= 2:
            n_va, n_tr = 1, n_tr - 1
        tr += grp[:n_tr]
        va += grp[n_tr:n_tr + n_va]
        te += grp[n_tr + n_va:]
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te


def split_real_stratified(items, seed=SEED):
    """Real positives: per phrasing, ~50% train / ~20% val / ~30% test with
    at least 1 clip in each slice whenever the phrasing has >=3 clips.
    Real clips are scarce, so the test slice is deliberately real-heavy —
    recall on real speech is the metric that matters."""
    return stratified_split(items, strat_fn=phrase_of,
                            fracs=(0.5, 0.2, 0.3), seed=seed)


class WWDataset(Dataset):
    """Same augmentation as v2 (train only): polarity flip, gain jitter,
    speed perturbation, noise mixing at random SNR from the noise train slice."""

    def __init__(self, items, augment=True, noise_files=None):
        super().__init__()
        self.items = items
        self.augment = augment
        self.noise_files = noise_files or []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        if self.augment:
            x = load_wav(path)
            if random.random() < 0.5:
                x = -x
            x = x * random.uniform(0.7, 1.3)
            if random.random() < 0.3:                    # speed perturb
                r = random.uniform(0.9, 1.1)
                x = _resample(x, int(SR / r))
                if len(x) >= N:
                    st = random.randint(0, len(x) - N)
                    x = x[st:st + N]
                else:
                    x = np.pad(x, (0, N - len(x)))
            if self.noise_files and random.random() < 0.7:
                nx, nsr = _sf_read(random.choice(self.noise_files))
                if len(nx) < N:
                    nx = np.pad(nx, (0, N - len(nx)))
                else:
                    st = random.randint(0, len(nx) - N)
                    nx = nx[st:st + N]
                sig_rms = float(np.sqrt(np.mean(np.square(x)))) + 1e-8
                noi_rms = float(np.sqrt(np.mean(np.square(nx)))) + 1e-8
                snr = random.uniform(5.0, 20.0)
                nx = nx * (sig_rms / noi_rms) * (10.0 ** (-snr / 20.0))
                x = x + nx
                peak = np.abs(x).max()
                if peak > 1.0:
                    x = x / peak
        else:
            x = load_wav_fixed(path)
        return x, label


def _sf_read(path):
    """Load wav -> (float32 mono in [-1,1] at SR, sr). Used for noise mixing."""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = _resample(x, sr)
    return x.astype(np.float32), sr


def collate(batch):
    xs, ys = zip(*batch)
    return np.stack(xs), np.array(ys, np.int64)


def evaluate(model, items, device, batch=64):
    """Deterministic eval: center-crop features, returns probs + truths."""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            xs = np.stack([load_wav_fixed(p) for p, _ in chunk])
            mels = torch.stack([
                torch.from_numpy(logmel(xs[j], WIN_FN, N_FFT, HOP, N_MELS, SR))
                for j in range(len(chunk))
            ]).to(device).permute(0, 2, 1).unsqueeze(1)
            p = torch.sigmoid(model(mels)).cpu().numpy()
            preds.append(p)
            truths.append(np.array([l for _, l in chunk]))
    return np.concatenate(preds), np.concatenate(truths)


def metrics_at(preds, truths, thr):
    tp = int(((preds >= thr) & (truths == 1)).sum())
    fp = int(((preds >= thr) & (truths == 0)).sum())
    fn = int(((preds < thr) & (truths == 1)).sum())
    tn = int(((preds < thr) & (truths == 0)).sum())
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    fpr = fp / max(1, fp + tn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(thr=thr, tp=tp, fp=fp, fn=fn, tn=tn,
                recall=round(rec, 4), precision=round(prec, 4),
                fpr=round(fpr, 4), f1=round(f1, 4))


def auc_rank(preds, truths):
    order = np.argsort(preds, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(preds) + 1)
    # average ranks for ties
    sp = np.asarray(preds)[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    n_pos = int(truths.sum()); n_neg = len(truths) - n_pos
    return float((ranks[truths == 1].sum() - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--export-onnx", default=None,
                    help="path for exported ONNX (default: out_dir/jarvis_cnn.onnx)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    ds_root = Path(args.dataset)
    out_dir = Path(args.out) if args.out else ds_root.parent / "cnn_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")

    # ---------------- collect items ----------------
    pos_syn = [(str(f), 1) for f in sorted((ds_root / "positive").glob("*.wav"))]
    # real recordings live in positive_real_1/ (first batch) and
    # positive_real_2/ (re-takes + okay_jarvis); read every positive_real* dir
    pos_real = []
    for d in sorted(ds_root.glob("positive_real*")):
        pos_real.extend((str(f), 1) for f in sorted(d.glob("*.wav")))
    negs = [(str(f), 0) for f in sorted((ds_root / "negative").glob("*.wav"))]
    noise = [(str(f), 0) for f in sorted((ds_root / "noise").glob("*.wav"))]
    print(f"pos_syn={len(pos_syn)} pos_real={len(pos_real)} neg={len(negs)} noise={len(noise)}")

    # ---------------- splits ----------------
    syn_tr, syn_va, syn_te = stratified_split(pos_syn, strat_fn=phrase_of)
    real_tr, real_va, real_te = split_real_stratified(pos_real)
    # negatives stratified by word, noise by file (each noise clip its own group)
    neg_tr, neg_va, neg_te = stratified_split(negs, strat_fn=word_of)
    noi_tr, noi_va, noi_te = stratified_split(noise, strat_fn=lambda p: "noise")

    # ---- real-room negatives, block-interleaved split -------------------------
    # windows overlap (0.5 s hop), so a random split would put near-duplicate
    # windows of the same sound event on both sides of the split. Interleave
    # by window index instead: contiguous blocks stay together.
    room = [(str(f), 0) for f in sorted((ds_root / "negative_real").glob("*.wav"))]
    room = sorted(room, key=lambda it: it[0])          # room_NNNN_score.wav order
    room_tr, room_va, room_te = [], [], []
    for i, it in enumerate(room):
        if i % 8 == 7:
            room_te.append(it)
        elif i % 8 == 3:
            room_va.append(it)
        else:
            room_tr.append(it)

    train_items = syn_tr + real_tr + neg_tr + noi_tr + room_tr
    val_items = syn_va + real_va + neg_va + noi_va + room_va
    test_items = syn_te + real_te + neg_te + noi_te + room_te
    noise_train_files = [p for p, _ in noi_tr]

    split_json = dict(
        seed=args.seed,
        counts=dict(
            train=dict(syn_pos=len(syn_tr), real_pos=len(real_tr), neg=len(neg_tr), noise=len(noi_tr),
                       room_neg=len(room_tr), total=len(train_items)),
            val=dict(syn_pos=len(syn_va), real_pos=len(real_va), neg=len(neg_va), noise=len(noi_va),
                     room_neg=len(room_va), total=len(val_items)),
            test=dict(syn_pos=len(syn_te), real_pos=len(real_te), neg=len(neg_te), noise=len(noi_te),
                      room_neg=len(room_te), total=len(test_items))),
        files=dict(
            train=[p for p, _ in train_items], val=[p for p, _ in val_items],
            test=[p for p, _ in test_items],
            test_real_pos=[p for p, _ in real_te], val_real_pos=[p for p, _ in real_va],
            test_syn_pos=[p for p, _ in syn_te], test_neg=[p for p, _ in neg_te],
            test_noise=[p for p, _ in noi_te]))
    (out_dir / "split.json").write_text(json.dumps(split_json, indent=2))
    print(f"split: train={len(train_items)} val={len(val_items)} test={len(test_items)} "
          f"(test real_pos={len(real_te)})")

    tr = DataLoader(WWDataset(train_items, augment=True, noise_files=noise_train_files),
                    batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=0)
    va = DataLoader(WWDataset(val_items, augment=False), batch_size=args.batch,
                    shuffle=False, collate_fn=collate, num_workers=0)

    model = WakeCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos_w = torch.tensor([args.pos_weight], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}")

    history, best_auc, best_ep = [], 0.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss, n = 0.0, 0
        for xb_np, yb_np in tr:
            xb = torch.from_numpy(xb_np).to(device)
            yb = torch.from_numpy(yb_np).to(device).float()
            opt.zero_grad()
            # per-sample logmel (v2 logmel takes a 1-D waveform)
            mels = torch.stack([
                torch.from_numpy(logmel(xb[j].detach().cpu().numpy(), WIN_FN,
                                        N_FFT, HOP, N_MELS, SR))
                for j in range(xb.shape[0])
            ]).to(device).permute(0, 2, 1).unsqueeze(1)   # (B,1,M,T)
            logits = model(mels).squeeze(-1)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(yb); n += len(yb)
        sched.step()

        vp, vt = evaluate(model, val_items, device)
        vacc = float(np.mean((vp >= 0.5) == vt))
        vauc = auc_rank(vp, vt)
        hist = dict(epoch=epoch, train_loss=round(tot_loss / max(1, n), 4),
                    val_acc=round(vacc, 4), val_auc=round(vauc, 4),
                    sec=round(time.time() - t0, 1))
        history.append(hist)
        print(f"[{epoch:02d}] loss={hist['train_loss']:.4f} val_acc={vacc:.3f} "
              f"val_auc={vauc:.4f} ({hist['sec']}s)")
        if vauc > best_auc:
            best_auc, best_ep = vauc, epoch
            torch.save({"state_dict": model.state_dict(), "n_mels": N_MELS,
                        "t_frames": T_FRAMES, "sr": SR, "clip_s": 1.0,
                        "n_fft": N_FFT, "hop": HOP, "win": N_FFT,
                        "best_epoch": epoch, "best_auc": auc_rank(vp, vt),
                        "val_acc": vacc, "seed": args.seed,
                        "split": "syn80/10/10 + real50/20/30 stratified"},
                       out_dir / "model.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # ---------------- test with best model ----------------
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device,
                                     weights_only=False)["state_dict"])
    model.eval()

    tp_, tt_ = evaluate(model, test_items, device)
    real_te_items = real_te
    rp_, rt_ = evaluate(model, real_te_items, device)
    syn_te_items = syn_te
    sp_, st_ = evaluate(model, syn_te_items, device)
    neg_te_items = neg_te + noi_te
    np_, nt_ = evaluate(model, neg_te_items, device)
    # real-room negatives are the headline metric
    roomneg_items = room_te
    rn_, rnt_ = evaluate(model, roomneg_items, device)
    allneg_items = neg_te + noi_te + room_te
    an_, ant_ = evaluate(model, allneg_items, device)

    report = dict(
        device=device, params=n_params, best_epoch=best_ep, best_val_auc=round(best_auc, 4),
        split_counts=split_json["counts"],
        val_auc=round(best_auc, 4),
        test_all=dict(n=len(tp_), auc=round(auc_rank(tp_, tt_), 4),
                      **{f"@{t}": metrics_at(tp_, tt_, t) for t in (0.5, 0.35, 0.15)}),
        test_real_pos=dict(n=len(rp_), mean=round(float(rp_.mean()), 4),
                           min=round(float(rp_.min()), 4), max=round(float(rp_.max()), 4),
                           **{f"recall@{t}": round(float((rp_ >= t).mean()), 4) for t in (0.5, 0.35, 0.15)}),
        test_syn_pos=dict(n=len(sp_), mean=round(float(sp_.mean()), 4),
                          **{f"recall@{t}": round(float((sp_ >= t).mean()), 4) for t in (0.5, 0.35, 0.15)}),
        test_neg_noise=dict(n=len(np_), mean=round(float(np_.mean()), 4),
                            max=round(float(np_.max()), 4),
                            **{f"fpr@{t}": round(float((np_ >= t).mean()), 4) for t in (0.5, 0.35, 0.15)}),
        test_neg_real_room=dict(n=len(rn_), mean=round(float(rn_.mean()), 4),
                                max=round(float(rn_.max()), 4),
                                **{f"fpr@{t}": round(float((rn_ >= t).mean()), 4) for t in (0.5, 0.35, 0.15)}),
        test_neg_all=dict(n=len(an_), mean=round(float(an_.mean()), 4),
                          max=round(float(an_.max()), 4),
                          **{f"fpr@{t}": round(float((an_ >= t).mean()), 4) for t in (0.5, 0.35, 0.15)}),
        real_by_phrase={},
    )
    # per-phrasing real recall (fill per-phrase keys; a single shared dict
    # would carry only the last phrase)
    for thr in (0.5, 0.35, 0.15):
        per = {}
        for ph in sorted(set(phrase_of(p) for p, _ in real_te_items)):
            idx = [i for i, (p, _) in enumerate(real_te_items) if phrase_of(p) == ph]
            per[ph] = round(float((rp_[idx] >= thr).mean()), 4)
        report["real_by_phrase"][f"recall@{thr}"] = per

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # ---------------- export ONNX ----------------
    onnx_path = Path(args.export_onnx) if args.export_onnx else out_dir / "jarvis_cnn.onnx"
    try:
        _export_onnx(model, onnx_path, best_ep, best_auc)
    except Exception as e:
        print(f"[!] ONNX export failed: {e}")

    print("\n=== TEST ===")
    print(f" all      n={report['test_all']['n']} auc={report['test_all']['auc']}")
    for t in (0.5, 0.35, 0.15):
        m = report["test_all"][f"@{t}"]
        print(f"  @{t}: P={m['precision']:.3f} R={m['recall']:.3f} FPR={m['fpr']:.3f} F1={m['f1']:.3f}")
    print(f" real_pos n={report['test_real_pos']['n']} mean={report['test_real_pos']['mean']} "
          f"min={report['test_real_pos']['min']} recall@0.15={report['test_real_pos']['recall@0.15']}")
    print(f" syn_pos  n={report['test_syn_pos']['n']} mean={report['test_syn_pos']['mean']} "
          f"recall@0.15={report['test_syn_pos']['recall@0.15']}")
    print(f" neg+noise n={report['test_neg_noise']['n']} max={report['test_neg_noise']['max']} "
          f"fpr@0.15={report['test_neg_noise']['fpr@0.15']}")
    print(f" neg_room n={report['test_neg_real_room']['n']} max={report['test_neg_real_room']['max']} "
          f"fpr@0.15={report['test_neg_real_room']['fpr@0.15']}")
    print(f" neg_ALL n={report['test_neg_all']['n']} max={report['test_neg_all']['max']} "
          f"fpr@0.15={report['test_neg_all']['fpr@0.15']}")
    print(f" real_by_phrase@0.15: {report['real_by_phrase']['recall@0.15']}")
    print(f"\noutputs in {out_dir}")


def _export_onnx(model, onnx_path, best_ep, best_auc):
    import onnxruntime as ort
    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.net = m
        def forward(self, x):
            return torch.sigmoid(self.net(x))
    # Export must happen on CPU: the dummy input is CPU and an MPS-resident
    # model makes torch.onnx.export fail with a device mismatch
    # (slow_conv2d_forward_mps). Move a copy to CPU first.
    model.to("cpu")
    wrapped = Wrap(model).eval()
    x = torch.randn(1, 1, N_MELS, T_FRAMES)
    torch.onnx.export(wrapped, x, str(onnx_path), input_names=["mel"],
                      output_names=["prob"],
                      dynamic_axes={"mel": {0: "batch"}, "prob": {0: "batch"}},
                      opset_version=13)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        m = rng.normal(-3.0, 3.0, size=(1, N_MELS, T_FRAMES)).astype(np.float32)
        with torch.no_grad():
            tp = float(wrapped(torch.from_numpy(m).unsqueeze(0)))
        op = float(sess.run(["prob"], {"mel": m})[0][0])
        worst = max(worst, abs(tp - op))
    meta = dict(source="train_cnn.py", best_epoch=best_ep, best_auc=round(best_auc, 4),
                n_mels=N_MELS, t_frames=T_FRAMES, sr=SR, clip_s=1.0,
                n_fft=N_FFT, hop=HOP, win=N_FFT,
                input="mel (B,1,40,98) float32 log-mel — 40 mel bins x 98 frames (10 ms hop), 16 kHz",
                output="prob (B,) sigmoid probability",
                runtime="onnxruntime CPUExecutionProvider",
                preprocessing="resample 16k -> logmel(n_fft=512, hop=160, 40 mels, center-padded) -> crop/pad to 98 frames -> transpose to (40,98)")
    Path(str(onnx_path)).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"onnx saved {onnx_path} (parity worst diff {worst:.2e})")


if __name__ == "__main__":
    main()