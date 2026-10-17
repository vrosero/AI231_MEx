#!/usr/bin/env python3
"""VCM — Voice Command Model: ONE script for the whole model lifecycle.

The VCM is the deployed CRNN (PiCommandCRNN, 0.367 M params): a
depthwise-separable CNN front-end + bidirectional GRU + dual head
(37 fine labels / 20 intents). Trains solely on the HF dataset
`airimonda/ai231-me2-voice-commands` (no local WAVs).

Subcommands
  prepare   download the HF parquet shards + build the log-mel feature cache
  train     train the CRNN, dual head fine+intent
  eval      evaluate a checkpoint on the test or holdout split + latency bench
  export    export a checkpoint to ONNX (+ meta.json) for the Raspberry Pi
  plot      plot training curves from a train_*.jsonl log

Feature contract (MUST stay identical to rpi/runtime/features.py — verified):
    12 kHz mono -> STFT (n_fft=400, hop=200) -> power -> 40-band mel (50-6000 Hz)
    -> log -> crop/pad to [40, 96] (pad -11.5) -> per-clip z-score.
HF clips are 16 kHz and are resampled to 12 kHz in the loader (same linear
interpolation the Pi runtime uses), so the deployed ONNX contract is unchanged:
    input  logmel        [batch, 1, 40, 96] float32
    output fine_logits   [batch, n_fine]   (37)
    output intent_logits [batch, n_intent]  (20 = 19 commands + OUT_OF_SCOPE)

Label scheme (built from the HF columns `command` + `slot_value`, same naming
convention the Pi dispatcher `rpi/runtime/actions.py` routes on):
    ALARM '6:00 AM'        -> ALARM_6_00AM        (the HF dataset slots are
                                                   6/8 AM, 9 PM)
    BRIGHTNESS '20 percent'-> BRIGHTNESS_20
    COLOR 'Red'            -> COLOR_RED
    CREATE_REMINDER 'Drink water' -> CREATE_REMINDER_DRINK_WATER
    TEMPERATURE '18 degrees' -> TEMPERATURE_18
    TIMER '10 seconds'     -> TIMER_10s ; '1 minute' -> TIMER_1m
    no slot                -> command as-is (PLAY_MUSIC, LIGHT_ON, ...)
    OUT_OF_SCOPE           -> OUT_OF_SCOPE (its own fine label AND intent)

Usage:
  python vcm_model.py prepare [--splits train,val,test,holdout] [--with-numerals N]
  python vcm_model.py train --arch crnn --epochs 60 --sup --cache-tag sup --tag crnn
  python vcm_model.py eval  --ckpt ckpt/crnn.pt --tag crnn [--split test]
  python vcm_model.py export --ckpt ckpt/crnn.pt --out run_outputs/onnx/crnn.onnx
  python vcm_model.py plot --log run_outputs/logs/train_crnn.jsonl
"""
import argparse
import collections
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Configuration (single source of truth for the model)
# ---------------------------------------------------------------------------
HF_DATASET = "airimonda/ai231-me2-voice-commands"

SAMPLE_RATE = 12000        # deployed Pi runtime rate (features.py)
N_MELS = 40
N_FFT = 400                # ~33 ms @12k
HOP = 200                  # ~16.7 ms @12k -> 60 frames/s
MAX_FRAMES = 96            # ~1.6 s window
PAD_VALUE = -11.5
F_MIN, F_MAX = 50.0, 6000.0
ENERGY_NORM = True         # RMS-rescale waveform before STFT (no-op after the
REF_RMS = 0.1              # per-clip z-score)

CACHE_DIR = os.path.join(HERE, "cache")
CKPT_DIR = os.path.join(HERE, "ckpt")
OUT_DIR = os.path.join(HERE, "run_outputs")

# slot_value -> fine-label suffix
_SLOT_MAP = {
    ("ALARM", "6:00 AM"): "ALARM_6_00AM",
    ("ALARM", "8:00 AM"): "ALARM_8_00AM",
    ("ALARM", "9:00 PM"): "ALARM_9_00PM",
    ("BRIGHTNESS", "20 percent"): "BRIGHTNESS_20",
    ("BRIGHTNESS", "60 percent"): "BRIGHTNESS_60",
    ("BRIGHTNESS", "100 percent"): "BRIGHTNESS_100",
    ("COLOR", "Red"): "COLOR_RED",
    ("COLOR", "Blue"): "COLOR_BLUE",
    ("COLOR", "Green"): "COLOR_GREEN",
    ("CREATE_REMINDER", "Drink water"): "CREATE_REMINDER_DRINK_WATER",
    ("CREATE_REMINDER", "Exercise"): "CREATE_REMINDER_EXERCISE",
    ("CREATE_REMINDER", "Study"): "CREATE_REMINDER_STUDY",
    ("TEMPERATURE", "18 degrees"): "TEMPERATURE_18",
    ("TEMPERATURE", "22 degrees"): "TEMPERATURE_22",
    ("TEMPERATURE", "26 degrees"): "TEMPERATURE_26",
    ("TIMER", "10 seconds"): "TIMER_10s",
    ("TIMER", "30 seconds"): "TIMER_30s",
    ("TIMER", "1 minute"): "TIMER_1m",
}


def fine_label_of(command, slot_value):
    """(command, slot_value) -> fine label. Generic fallback for unseen slots."""
    key = (command, (slot_value or "").strip())
    if key in _SLOT_MAP:
        return _SLOT_MAP[key]
    if slot_value:
        suf = "".join(ch if ch.isalnum() else "_" for ch in slot_value).strip("_")
        return f"{command}_{suf}".upper()
    return command


# ---------------------------------------------------------------------------
# HF dataset loading (huggingface_hub + pyarrow + soundfile; no `datasets`)
# ---------------------------------------------------------------------------
_PARQUET_FILES = {
    "train": [f"data/train-{i:05d}-of-00002.parquet" for i in range(2)],
    "test": ["data/test-00000-of-00001.parquet"],
    "holdout": ["data/holdout-00000-of-00001.parquet"],
    "numerals": [f"data/numerals-{i:05d}-of-00006.parquet" for i in range(6)],
    # supplemental file sets: merged by --sup, numerals stay excluded
    "sup_synth": ["supplemental_synth/train-00000-of-00001.parquet"],
    "sup_fil": [f"supplemental_fil/train-{i:05d}-of-00003.parquet" for i in range(3)],
    "neg_train": ["synthetic_negatives/train-00000-of-00001.parquet"],
    "neg_test": ["synthetic_negatives/test-00000-of-00001.parquet"],
}


def _download_split_files(split):
    from huggingface_hub import hf_hub_download
    paths = []
    for fn in _PARQUET_FILES[split]:
        paths.append(hf_hub_download(HF_DATASET, fn, repo_type="dataset"))
    return paths


_COLS = ["audio", "file", "transcript", "command", "variation", "slot_value",
         "out_of_scope", "bucket", "speaker_id", "source", "is_synthetic",
         "duration_s"]


def _read_rows(path, cols=None):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=cols or _COLS)
    return t.to_pylist()


def load_split_rows(split, with_numerals=0, verbose=True, sup=False):
    """Download (cached) the split's parquet shards and return a list of row
    dicts: {file, transcript, command, variation, slot_value, out_of_scope,
    bucket, speaker_id, source, is_synthetic, duration_s, audio_bytes}.

    numerals: if >0, subsample that many OUT_OF_SCOPE SpeechCommands_v2 clips
    into the train rows (extra OOS hard negatives; the split is 66,390 clips).

    sup: also merge the repo's supplemental file sets into the split —
      supplemental_synth routed by its voice_split column (train/test/holdout),
      synthetic_negatives train->train / test->test, supplemental_fil->train.
      Label space is unchanged (verified: every (command, slot_value) pair maps
      onto the same 37 fine labels via fine_label_of/_SLOT_MAP)."""
    import pyarrow.parquet as pq

    rows = []
    for p in _download_split_files(split):
        rows.extend(_read_rows(p))
    if verbose:
        print(f"[data] {split}: {len(rows)} rows from "
              f"{len(_PARQUET_FILES[split])} shard(s)", flush=True)
    if split == "train" and with_numerals > 0:
        num = []
        for p in _download_split_files("numerals"):
            num.extend(_read_rows(p))
        step = max(1, len(num) // max(1, with_numerals))
        num = num[::step][:with_numerals]
        rows.extend(num)
        if verbose:
            print(f"[data] + numerals subsample: {len(num)} OUT_OF_SCOPE clips",
                  flush=True)
    if sup:
        n0 = len(rows)
        # supplemental_synth: route by the voice_split column
        synth = _read_rows(_download_split_files("sup_synth")[0],
                           cols=_COLS + ["voice_split"])
        for r in synth:
            if r.pop("voice_split", None) == split:
                rows.append(r)
        if split == "train":
            for p in _download_split_files("sup_fil"):
                rows.extend(_read_rows(p))
            for p in _download_split_files("neg_train"):
                rows.extend(_read_rows(p))
        elif split == "test":
            for p in _download_split_files("neg_test"):
                rows.extend(_read_rows(p))
        if verbose:
            print(f"[data] + supplementals: {len(rows) - n0} rows -> "
                  f"{split} total {len(rows)}", flush=True)
    return rows


def decode_audio(audio_struct):
    """HF audio struct {bytes, path} -> (float32 mono [-1,1], sr). None on error."""
    try:
        import soundfile as sf
        buf = audio_struct.get("bytes") if isinstance(audio_struct, dict) else None
        if not buf:
            return None
        x, sr = sf.read(io.BytesIO(buf), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return x.astype(np.float32), int(sr)
    except Exception:
        return None


def resample_linear(x, sr, target_sr):
    """Linear-interpolation resample — identical to rpi/runtime/features.py."""
    if sr == target_sr:
        return x
    dur = len(x) / sr
    new_n = max(1, int(round(dur * target_sr)))
    old_pos = np.arange(len(x))
    new_pos = np.linspace(0, len(x) - 1, new_n)
    return np.interp(new_pos, old_pos, x).astype(np.float32)


# ---------------------------------------------------------------------------
# Features: numpy STFT + mel (identical math to the Pi runtime features.py)
# ---------------------------------------------------------------------------
def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


_MEL_FILTERS = {}


def _get_mel_filters(sr, n_fft, n_mels, f_min, f_max):
    key = (sr, n_fft, n_mels, f_min, f_max)
    if key in _MEL_FILTERS:
        return _MEL_FILTERS[key]
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    m = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    m_hz = _mel_to_hz(m)
    bins = (m_hz * (n_fft // 2 + 1) / (sr / 2)).astype(int)
    f = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c > l:
            f[i - 1, l:c] = (freqs[l:c] - m_hz[i - 1]) / (m_hz[i] - m_hz[i - 1])
        if r > c:
            f[i - 1, c:r] = (m_hz[i + 1] - freqs[c:r]) / (m_hz[i + 1] - m_hz[i])
    f = np.clip(f, 0, 1)
    norm = f.max(axis=1, keepdims=True)
    norm[norm == 0] = 1
    f = f / norm
    _MEL_FILTERS[key] = f.astype(np.float32)
    return _MEL_FILTERS[key]


def wav_to_logmel_np(x, sr):
    """float32 mono waveform (any sr) -> [N_MELS, MAX_FRAMES] log-mel, z-normed.
    Same pipeline as the Pi runtime features.logmel_from_array."""
    try:
        x = resample_linear(np.asarray(x, dtype=np.float32), sr, SAMPLE_RATE)
        if ENERGY_NORM:
            rms = float(np.sqrt(np.mean(x ** 2)))
            if rms > 1e-6:
                x = x * (REF_RMS / rms)
        if len(x) < HOP:
            x = np.pad(x, (0, HOP - len(x)))
        n_frames = 1 + (len(x) - N_FFT) // HOP
        if n_frames <= 0:
            return None
        idx = np.arange(N_FFT)[None, :] + (np.arange(n_frames) * HOP)[:, None]
        frames = x[idx] * np.hanning(N_FFT).astype(np.float32)
        spec = np.fft.rfft(frames.astype(np.float64), axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float64)
        mel = _get_mel_filters(SAMPLE_RATE, N_FFT, N_MELS, F_MIN, F_MAX).astype(np.float64)
        logmel = np.log(power @ mel.T + 1e-5).astype(np.float32)   # [T, n_mels]
        logmel = np.nan_to_num(logmel, nan=PAD_VALUE, posinf=0.0, neginf=PAD_VALUE)
        logmel = logmel.T                                          # [n_mels, T]
        T = logmel.shape[1]
        if T >= MAX_FRAMES:
            logmel = logmel[:, :MAX_FRAMES]
        else:
            logmel = np.pad(logmel, ((0, 0), (0, MAX_FRAMES - T)),
                            constant_values=PAD_VALUE)
        mu = logmel.mean()
        sd = logmel.std()
        if sd < 1e-5:
            sd = 1e-5
        return ((logmel - mu) / sd).astype(np.float32)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Label maps + PyTorch Dataset over the HF rows
# ---------------------------------------------------------------------------
def build_label_maps(rows):
    """fine = command+slot label; intent = command (the actionable group).
    OUT_OF_SCOPE is its own fine label and its own intent."""
    fine = sorted({fine_label_of(r["command"], r.get("slot_value")) for r in rows})
    intents = sorted({r["command"] for r in rows})
    fine2idx = {l: i for i, l in enumerate(fine)}
    intent2idx = {l: i for i, l in enumerate(intents)}
    fine2intent = {fine_label_of(r["command"], r.get("slot_value")): r["command"]
                   for r in rows}
    fine2transcript = {}
    for r in rows:
        fine2transcript.setdefault(fine_label_of(r["command"], r.get("slot_value")),
                                   r["transcript"] or "")
    return fine, fine2idx, intent2idx, fine2intent, fine2transcript


def _cache_path(split, tag=""):
    return os.path.join(CACHE_DIR, f"{split}{tag}.npy")


def build_feature_cache(splits=("train", "val", "test", "holdout"), tag="", sup=False):
    """Decode every clip once into a [N, 1, 40, 96] float32 shard per split.
    One pass over the audio; training then reads the .npy shards, not the
    parquet.

    sup: build the cache over the MERGED rows (supplemental_synth by
    voice_split, supplemental_fil->train, synthetic_negatives->train/test) so
    the shard stays aligned with load_split_rows(split, sup=True) order."""
    import pyarrow.parquet as pq  # noqa: F401  (rows already hold decoded bytes)
    os.makedirs(CACHE_DIR, exist_ok=True)
    for split in splits:
        out = _cache_path(split, tag)
        if os.path.exists(out):
            print(f"[cache] {split}: {out} exists, skip", flush=True)
            continue
        rows = load_split_rows(split, sup=sup)
        arr = np.zeros((len(rows), 1, N_MELS, MAX_FRAMES), dtype=np.float32)
        bad = 0
        t0 = time.time()
        for i, r in enumerate(rows):
            a = decode_audio(r["audio"])
            if a is None:
                bad += 1
                continue
            f = wav_to_logmel_np(a[0], a[1])
            if f is None:
                bad += 1
                continue
            arr[i, 0] = f
        np.save(out, arr)
        dt = time.time() - t0
        print(f"[cache] {split}: {len(rows)} clips -> {out} "
              f"({os.path.getsize(out) // 1024 // 1024} MB) unreadable={bad} "
              f"in {dt:.0f}s ({len(rows) / max(dt, 1e-6):.0f} clips/s)", flush=True)


class CommandDataset:
    """Rows + cached features -> (x, fine_idx, intent_idx, src_flag, row).

    src_flag: 1.0 = synthetic, 0.0 = human (from the `is_synthetic` column) —
    carried in the collate batch for provenance/debugging."""

    def __init__(self, rows, feats, fine2idx, intent2idx, fine2intent):
        self.rows = rows
        self.feats = feats
        self.fine2idx = fine2idx
        self.intent2idx = intent2idx
        self.fine2intent = fine2intent

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.feats is not None:
            f = self.feats[i]
            x = torch.from_numpy(np.asarray(f)).clone()
            if x.ndim == 2:
                x = x.unsqueeze(0)
        else:
            a = decode_audio(r["audio"])
            f = wav_to_logmel_np(a[0], a[1]) if a is not None else None
            x = torch.from_numpy(f).unsqueeze(0) if f is not None \
                else torch.zeros(1, N_MELS, MAX_FRAMES)
        fl = fine_label_of(r["command"], r.get("slot_value"))
        fine = self.fine2idx[fl]
        intent = self.intent2idx[self.fine2intent[fl]]
        src = torch.tensor(float(r.get("is_synthetic", 0) or 0))
        return x, fine, intent, src, r


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])          # [B,1,40,96]
    fines = torch.tensor([b[1] for b in batch])      # [B]
    intents = torch.tensor([b[2] for b in batch])    # [B]
    srcs = torch.stack([b[3] for b in batch])        # [B]
    rows = [b[4] for b in batch]
    return xs, fines, intents, srcs, rows


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


class _SeqHead(nn.Module):
    """Head over a pooled sequence embedding (2-D input): trunk -> fine + intent."""
    def __init__(self, feat_dim, n_fine, n_intent, dropout=0.35):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(dropout),
        )
        self.fc_fine = nn.Linear(256, n_fine)
        self.fc_intent = nn.Linear(256, n_intent)

    def forward(self, z):
        z = self.classifier(z)
        return self.fc_fine(z), self.fc_intent(z)


class PiCommandCRNN(nn.Module):
    """CRNN (primary): depthwise-separable CNN front-end -> BiGRU -> dual head.

    The CNN compresses frequency (40 -> 5 mel bands) while keeping 12 time
    frames; the bidirectional GRU models their temporal order, which a purely
    pooled CNN discards. Contract:
        input  [B, 1, 40, 96] log-mel
        output (fine_logits, intent_logits)
    """

    def __init__(self, n_fine, n_intent, n_mels=N_MELS, base=48, dropout=0.35,
                 rnn_hidden=128):
        super().__init__()

        def dwsep(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, padding=1, groups=cin),
                nn.Conv2d(cin, cout, 1),
                nn.BatchNorm2d(cout), nn.ReLU(),
            )

        self.features = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
            dwsep(base, base),
            nn.MaxPool2d(2),                    # 40x96 -> 20x48
            dwsep(base, base * 2),
            dwsep(base * 2, base * 2),
            nn.MaxPool2d(2),                    # -> 10x24
            dwsep(base * 2, base * 4),
            nn.MaxPool2d(2),                    # -> 5x12
            nn.Dropout2d(dropout),
        )
        feat_dim = base * 4
        self.rnn = nn.GRU(feat_dim, rnn_hidden, batch_first=True,
                          bidirectional=True)
        self.head = _SeqHead(2 * rnn_hidden, n_fine, n_intent, dropout)
        # Analytic GRU cost hook (added by compute_flops): 2 directions x
        # 3 gates x (input*hidden + hidden*hidden) x T frames, MACs -> x2.
        t_frames = MAX_FRAMES // 8              # 96 -> 12 after three /2 pools
        macs = 2 * 3 * (feat_dim * rnn_hidden + rnn_hidden * rnn_hidden) * t_frames
        self.rnn_flops = 2 * macs

    def forward(self, x):
        z = self.features(x)                    # [B, C, 5, 12]
        z = z.mean(dim=2)                       # squeeze frequency -> [B, C, T]
        z = z.transpose(1, 2)                   # [B, T, C] (batch_first GRU)
        out, _ = self.rnn(z)                    # [B, T, 2H]
        z = out.mean(dim=1)                     # temporal mean pool -> [B, 2H]
        return self.head(z)


ARCHS = {"crnn": PiCommandCRNN}


def build_model(name, n_fine, n_intent, base=48, dropout=0.35):
    return ARCHS[name](n_fine, n_intent, base=base, dropout=dropout)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def _conv_flops(cin, cout, kh, kw, oh, ow, groups=1):
    return cin // groups * cout * kh * kw * oh * ow


def compute_flops(model, n_mels=N_MELS, max_frames=MAX_FRAMES):
    """Analytic forward-pass FLOPs (MACs*2), closed-form per layer."""
    feat = model.features
    cin = 1
    h, w = n_mels, max_frames
    flops = 0
    for mod in feat.modules():
        if isinstance(mod, nn.Conv2d):
            flops += 2 * _conv_flops(cin, mod.out_channels, mod.kernel_size[0],
                                     mod.kernel_size[1], h, w, mod.groups)
            cin = mod.out_channels
        elif isinstance(mod, nn.MaxPool2d):
            ks = mod.kernel_size if isinstance(mod.kernel_size, (tuple, list)) \
                else [mod.kernel_size, mod.kernel_size]
            h //= ks[0]
            w //= ks[1]
    if hasattr(model, "rnn_flops"):
        flops += model.rnn_flops
    cl = model.head.classifier[0]
    flops += 2 * (cl.in_features * cl.out_features)
    flops += 2 * (256 * model.head.fc_fine.out_features)
    flops += 2 * (256 * model.head.fc_intent.out_features)
    return flops


# ---------------------------------------------------------------------------
# Metrics (accuracy, macro/weighted F1, per-intent recall, WER)
# ---------------------------------------------------------------------------
def _confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _f1_from_cm(cm, average="macro"):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(tp), where=(prec + rec) > 0)
    if average == "macro":
        return float(f1.mean())
    support = cm.sum(axis=1).astype(float)
    total = support.sum()
    if total == 0:
        return 0.0
    return float((f1 * support).sum() / total)


def wer(ref, hyp):
    r = ref.lower().split()
    h = hyp.lower().split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return d[len(r)][len(h)] / max(len(r), 1)


def evaluate_predictions(y_true_fine, y_pred_fine, y_true_intent, y_pred_intent,
                         fine_labels, intent_labels, fine2transcript,
                         transcripts_true):
    out = {}
    n_fine = len(fine_labels)
    n_intent = len(intent_labels)

    out["fine_accuracy"] = float(np.mean([t == p for t, p in
                                          zip(y_true_fine, y_pred_fine)]))
    cm_fine = _confusion(y_true_fine, y_pred_fine, n_fine)
    out["fine_macro_f1"] = _f1_from_cm(cm_fine, "macro")
    out["fine_weighted_f1"] = _f1_from_cm(cm_fine, "weighted")

    out["intent_accuracy"] = float(np.mean([t == p for t, p in
                                            zip(y_true_intent, y_pred_intent)]))
    cm_intent = _confusion(y_true_intent, y_pred_intent, n_intent)
    out["intent_macro_f1"] = _f1_from_cm(cm_intent, "macro")

    per = {}
    for idx in range(n_intent):
        mask = [i for i, t in enumerate(y_true_intent) if t == idx]
        if not mask:
            continue
        correct = sum(1 for i in mask if y_pred_intent[i] == y_true_intent[i])
        per[intent_labels[idx]] = round(correct / len(mask), 4)
    out["per_intent_recall"] = per

    perf = {}
    for idx in range(n_fine):
        mask = [i for i, t in enumerate(y_true_fine) if t == idx]
        if not mask:
            continue
        correct = sum(1 for i in mask if y_pred_fine[i] == y_true_fine[i])
        perf[fine_labels[idx]] = round(correct / len(mask), 4)
    out["per_fine_recall"] = perf

    wers = [wer(gt, fine2transcript.get(fine_labels[yp], ""))
            for gt, yp in zip(transcripts_true, y_pred_fine)]
    out["wer_mean"] = float(np.mean(wers))
    out["wer_median"] = float(np.median(wers))

    out["confusion_intent"] = cm_intent.tolist()
    out["intent_order"] = intent_labels
    return out


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def collect_predictions(model, ds, device, batch=128):
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0,
                    collate_fn=collate_fn)
    pf, pi, tf, ti = [], [], [], []
    for x, fine, intent, src, _ in dl:
        x = x.to(device)
        lf, li = model(x)
        pf.append(lf.argmax(1).cpu().tolist())
        pi.append(li.argmax(1).cpu().tolist())
        tf.append(fine.tolist())
        ti.append(intent.tolist())
    import itertools
    return (list(itertools.chain(*pf)), list(itertools.chain(*pi)),
            list(itertools.chain(*tf)), list(itertools.chain(*ti)))


def benchmark_torch(model, device, n=400):
    """Per-inference latency (batch=1), warm cache."""
    model.eval()
    dummy = torch.randn(1, 1, N_MELS, MAX_FRAMES, device=device)
    for _ in range(30):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps" and hasattr(torch.backends.mps, "synchronize"):
            torch.backends.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    return dict(mean_ms=float(ts.mean()), p50_ms=float(np.percentile(ts, 50)),
                p95_ms=float(np.percentile(ts, 95)), p99_ms=float(np.percentile(ts, 99)))


# ---------------------------------------------------------------------------
# Subcommand: prepare
# ---------------------------------------------------------------------------
def cmd_prepare(args):
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    build_feature_cache(splits, tag=args.tag, sup=bool(args.sup))
    # dataset card facts for the report
    rows = load_split_rows(splits[0], verbose=False, sup=bool(args.sup)) if splits else []
    if rows:
        durs = [r.get("duration_s") or 0 for r in rows]
        print(f"[stats] {splits[0]}: {len(rows)} clips, "
              f"{sum(durs) / 3600:.2f} h, {len({r['speaker_id'] for r in rows})} speakers, "
              f"{len({r['command'] for r in rows})} commands")


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------
def cmd_train(args):
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    tag = args.tag

    rows = load_split_rows("train", with_numerals=args.numerals, verbose=False,
                           sup=args.sup)
    tr_pos, va_pos, val_spk = _split_positions(rows, val_frac=args.val_frac,
                                               seed=args.seed)
    tr_rows = [rows[i] for i in tr_pos]
    va_rows = [rows[i] for i in va_pos]
    print(f"[split] train={len(tr_rows)}  val={len(va_rows)}  "
          f"val_speakers={len(val_spk)} (speaker-disjoint)", flush=True)

    fine, fine2idx, intent2idx, fine2intent, fine2transcript = \
        build_label_maps(rows)
    n_fine, n_intent = len(fine), len(intent2idx)
    print(f"[labels] n_fine={n_fine} n_intent={n_intent}", flush=True)

    ft = np.load(_cache_path("train", tag=args.cache_tag), mmap_mode="r")
    # The feature cache shard is aligned to load_split_rows("train") order;
    # the val split is carved out of train as POSITIONS into that same order,
    # so both datasets index the same shard.
    assert len(rows) == ft.shape[0], (len(rows), ft.shape[0])

    tr_ds = CommandDataset(tr_rows, ft[tr_pos], fine2idx, intent2idx, fine2intent)
    va_ds = CommandDataset(va_rows, ft[va_pos], fine2idx, intent2idx, fine2intent)
    tl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate_fn)
    vl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    collate_fn=collate_fn)

    model = build_model(args.arch, n_fine, n_intent, base=args.base,
                        dropout=args.dropout).to(device)
    npar = count_params(model)
    nflop = compute_flops(model)
    print(f"[{tag}] arch={args.arch} params={npar / 1e6:.3f}M "
          f"flops={nflop / 1e6:.2f}M device={device} epochs={args.epochs} "
          f"batch={args.batch} lr={args.lr}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    cef = nn.CrossEntropyLoss(label_smoothing=args.ls)
    cei = nn.CrossEntropyLoss(label_smoothing=args.ls)

    def evaluate_val():
        model.eval()
        cf, ci, tf, ti = [], [], [], []
        with torch.no_grad():
            for x, fine_l, intent_l, src, _ in vl:
                x = x.to(device)
                lf, li = model(x)
                cf.append(lf.argmax(1).cpu().tolist())
                ci.append(li.argmax(1).cpu().tolist())
                tf.append(fine_l.tolist())
                ti.append(intent_l.tolist())
        import itertools
        cf = list(itertools.chain(*cf))
        ci = list(itertools.chain(*ci))
        tf = list(itertools.chain(*tf))
        ti = list(itertools.chain(*ti))
        iacc = sum(a == b for a, b in zip(ci, ti)) / len(ti)
        facc = sum(a == b for a, b in zip(cf, tf)) / len(tf)
        return iacc, facc

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    log_dir = os.path.join(OUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    lf = open(os.path.join(log_dir, f"train_{tag}.jsonl"), "w")
    best, best_ep, t_start = -1.0, 0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        trl, nb = 0.0, 0
        t0 = time.time()
        for x, fine_l, intent_l, src, _ in tl:
            x = x.to(device)
            fine_l = fine_l.to(device)
            intent_l = intent_l.to(device)
            opt.zero_grad()
            lff, lif = model(x)
            loss = cef(lff, fine_l) + 0.5 * cei(lif, intent_l)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            trl += loss.item() * x.size(0)
            nb += x.size(0)
        sched.step()
        trl /= nb
        acc, facc = evaluate_val()
        dt = time.time() - t0
        rec = dict(epoch=ep, train_loss=round(trl, 5),
                   val_intent_acc=round(acc, 5), val_fine_acc=round(facc, 5),
                   lr=sched.get_last_lr()[0], sec=round(dt, 1),
                   elapsed_min=round((time.time() - t_start) / 60, 1))
        lf.write(json.dumps(rec) + "\n")
        lf.flush()
        print(f"[{tag}] ep{ep:03d} loss={trl:.4f} val_intent={acc:.4f} "
              f"val_fine={facc:.4f} lr={sched.get_last_lr()[0]:.2e} ({dt:.0f}s)",
              flush=True)
        if acc > best:
            best = acc
            best_ep = ep
            torch.save({"state_dict": model.state_dict(), "arch": args.arch,
                        "fine": fine, "intent2idx": intent2idx,
                        "fine2intent": fine2intent,
                        "fine2transcript": fine2transcript,
                        "base": args.base, "dropout": args.dropout,
                        "n_fine": n_fine, "n_intent": n_intent,
                        "best_val_intent_acc": best, "best_epoch": best_ep,
                        "params": npar, "flops": nflop, "seed": args.seed,
                        "epochs": args.epochs, "dataset": HF_DATASET,
                        "feature": dict(sample_rate=SAMPLE_RATE, n_mels=N_MELS,
                                        n_fft=N_FFT, hop=HOP,
                                        max_frames=MAX_FRAMES)},
                       os.path.join(CKPT_DIR, f"best_{tag}.pt"))
            print(f"[{tag}]   ^ new best val_intent={best:.4f} @ep{ep} saved",
                  flush=True)
    lf.close()
    print(f"[{tag}] DONE best_val_intent={best:.4f}@ep{best_ep} "
          f"total={(time.time() - t_start) / 60:.1f}min", flush=True)


def _split_positions(rows, val_frac=0.05, seed=42):
    """Speaker-disjoint val split as POSITIONS into the original rows list
    (cache shards are aligned to that order)."""
    import random
    rng = random.Random(seed)
    by_spk = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_spk[r["speaker_id"]].append(i)
    spks = sorted(by_spk)
    rng.shuffle(spks)
    n_val = max(1, int(len(spks) * val_frac))
    val_idx = [i for s in spks[:n_val] for i in by_spk[s]]
    tr_idx = [i for s in spks[n_val:] for i in by_spk[s]]
    rng.shuffle(tr_idx)
    rng.shuffle(val_idx)
    return tr_idx, val_idx, set(spks[:n_val])


# ---------------------------------------------------------------------------
# Subcommand: eval  (test / holdout + latency benchmark + confusion CSV)
# ---------------------------------------------------------------------------
def cmd_eval(args):
    device = pick_device()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], ck["n_fine"], ck["n_intent"],
                        base=ck["base"], dropout=ck.get("dropout", 0.35)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    fine = ck["fine"]
    intent_labels = sorted(ck["intent2idx"].keys())
    fine2transcript = ck["fine2transcript"]
    print(f"[loaded] {args.ckpt}  arch={ck['arch']}  "
          f"params={count_params(model) / 1e6:.3f}M  device={device}  "
          f"best_val_intent_acc={ck.get('best_val_intent_acc')}", flush=True)

    split = args.split
    rows = load_split_rows(split, verbose=False, sup=bool(args.cache_tag))
    feats = np.load(_cache_path(split, tag=args.cache_tag), mmap_mode="r")
    assert feats.shape[0] == len(rows), (feats.shape[0], len(rows))
    # Use the CHECKPOINT's label maps (derived from the training rows), never
    # maps rebuilt from the eval split — the eval split's sorted label list can
    # be a subset (indices would shift and silently misalign the comparison).
    fine2idx = {l: i for i, l in enumerate(fine)}
    intent2idx = ck["intent2idx"]
    fine2intent = ck["fine2intent"]
    kept_idx, skipped = [], 0
    for i, r in enumerate(rows):
        fl = fine_label_of(r["command"], r.get("slot_value"))
        if fl in fine2idx and fine2intent.get(fl) in intent2idx:
            kept_idx.append(i)
        else:
            skipped += 1
    if skipped:
        print(f"[eval] note: {skipped}/{len(rows)} rows skipped "
              f"(labels unseen in training)", flush=True)
    rows_k = [rows[i] for i in kept_idx]
    feats_k = feats[kept_idx] if skipped else feats
    ds = CommandDataset(rows_k, feats_k, fine2idx, intent2idx, fine2intent)
    pf, pi, tf, ti = collect_predictions(model, ds, device)
    transcripts_true = [r["transcript"] or "" for r in ds.rows]

    res = evaluate_predictions(tf, pf, ti, pi, fine, intent_labels,
                               fine2transcript, transcripts_true)
    res["n_test"] = len(tf)
    res["arch"] = ck["arch"]
    res["params_M"] = round(count_params(model) / 1e6, 4)
    res["flops_M"] = round(compute_flops(model) / 1e6, 3)
    res["best_val_intent_acc"] = ck.get("best_val_intent_acc")
    res["best_epoch"] = ck.get("best_epoch")
    res["split"] = split
    res["dataset"] = HF_DATASET
    res["timing_" + device] = benchmark_torch(model, device)

    # FAR proxy on the holdout's OUT_OF_SCOPE clips (slide-deck field):
    # fraction of true OUT_OF_SCOPE clips predicted as an in-scope command.
    if split == "holdout" or "OUT_OF_SCOPE" in intent_labels:
        oos_idx = intent_labels.index("OUT_OF_SCOPE")
        mask = [i for i, t in enumerate(ti) if t == oos_idx]
        if mask:
            fa = sum(1 for i in mask if pi[i] != oos_idx)
            res["far_oos"] = round(fa / len(mask), 4)
            res["far_oos_n"] = len(mask)

    eval_dir = os.path.join(OUT_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    jpath = os.path.join(eval_dir, f"eval_{split}_{args.tag}.json")
    with open(jpath, "w") as f:
        json.dump(res, f, indent=2)
    import csv
    conf_dir = os.path.join(OUT_DIR, "confusion")
    os.makedirs(conf_dir, exist_ok=True)
    with open(os.path.join(conf_dir, f"confusion_{split}_{args.tag}.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + intent_labels)
        for i, lab in enumerate(intent_labels):
            w.writerow([lab] + res["confusion_intent"][i])
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("confusion_intent", "per_fine_recall")},
                     indent=2), flush=True)
    print("\n[per-intent recall]")
    for k, v in sorted(res["per_intent_recall"].items(), key=lambda x: x[1]):
        print(f"  {k:22s} {v:6.3f}  {'#' * int(v * 40)}", flush=True)
    print(f"\n[saved] {jpath}", flush=True)


# ---------------------------------------------------------------------------
# Subcommand: export (ONNX + meta.json, same contract as rpi/runtime)
# ---------------------------------------------------------------------------
class _Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        fine, intent = self.m(x)
        return fine, intent


def cmd_export(args):
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], ck["n_fine"], ck["n_intent"],
                        base=ck["base"], dropout=ck.get("dropout", 0.35))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    wrap = _Wrap(model)
    # batch-8 dummy: the dynamo exporter bakes the dummy batch into the GRU's
    # weight reshape, so a batch-1 dummy yields a graph that fails for any
    # other batch. A batch-8 dummy + dynamic_axes gives a batch-safe graph
    # (verified B=1 and B=8 parity 1.2e-06).
    dummy = torch.randn(8, 1, N_MELS, MAX_FRAMES)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(wrap, (dummy,), args.out,
                      input_names=["logmel"],
                      output_names=["fine_logits", "intent_logits"],
                      dynamic_axes={"logmel": {0: "batch"},
                                    "fine_logits": {0: "batch"},
                                    "intent_logits": {0: "batch"}},
                      opset_version=13)
    # torch.onnx.export leaves the module in TRAINING mode (torch 2.8) —
    # dropout re-activates and any later forward is wrong. Re-eval before the
    # parity check below.
    model.eval()
    wrap.eval()
    # torch 2.8 dynamo exporter may spill large weights (GRU) to an external
    # <out>.data file; inline them so the deployed artefact is a single file.
    data_p = args.out + ".data"
    if os.path.exists(data_p):
        import onnx
        m = onnx.load(args.out, load_external_data=True)
        for init in m.graph.initializer:
            if init.external_data:
                del init.external_data[:]
                init.data_location = 0
        onnx.save(m, args.out)
        os.remove(data_p)
        print(" inlined external weights -> single file", flush=True)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[export] {args.out}  ({size_mb:.2f} MB)  arch={ck['arch']}  "
          f"params={count_params(model) / 1e6:.3f}M", flush=True)

    meta = {
        "arch": ck["arch"], "params_M": round(count_params(model) / 1e6, 4),
        "n_fine": ck["n_fine"], "n_intent": ck["n_intent"],
        "fine_labels": ck["fine"],
        "intent_labels": sorted(ck["intent2idx"].keys()),
        "fine2intent": ck["fine2intent"],
        "fine2transcript": ck["fine2transcript"],
        "best_val_intent_acc": ck.get("best_val_intent_acc"),
        "best_epoch": ck.get("best_epoch"),
        "dataset": HF_DATASET,
        "feature": {"sample_rate": SAMPLE_RATE, "n_mels": N_MELS,
                    "n_fft": N_FFT, "hop": HOP, "max_frames": MAX_FRAMES,
                    "target_sr": SAMPLE_RATE, "pad_value": PAD_VALUE,
                    "mel_range": [F_MIN, F_MAX],
                    "normalize": "z-score per clip",
                    "input_shape": [1, N_MELS, MAX_FRAMES]},
        "onnx": {"file": os.path.basename(args.out),
                 "size_MB": round(size_mb, 3), "inputs": ["logmel"],
                 "outputs": ["fine_logits", "intent_logits"], "opset": 13},
    }
    mp = os.path.splitext(args.out)[0] + "_meta.json"
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[meta] {mp}", flush=True)

    # onnxruntime CPU benchmark + parity check (torch vs onnxruntime)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        x = np.random.randn(1, 1, N_MELS, MAX_FRAMES).astype(np.float32)
        for _ in range(30):
            sess.run(None, {"logmel": x})
        ts = []
        for _ in range(400):
            t0 = time.perf_counter()
            sess.run(None, {"logmel": x})
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1000
        bench = dict(mean_ms=float(ts.mean()), p50_ms=float(np.percentile(ts, 50)),
                     p95_ms=float(np.percentile(ts, 95)),
                     p99_ms=float(np.percentile(ts, 99)))
        meta["onnx"]["ort_cpu_bench"] = bench
        with open(mp, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[bench] onnxruntime CPU: {bench}", flush=True)
        # parity (batch of 8; per-element max diff)
        x = np.random.randn(8, 1, N_MELS, MAX_FRAMES).astype(np.float32)
        with torch.no_grad():
            tf_, ti_ = wrap(torch.from_numpy(x))
        of, oi = sess.run(None, {"logmel": x})
        worst = max(float(np.abs(tf_.numpy() - of).max()),
                    float(np.abs(ti_.numpy() - oi).max()))
        print(f"[parity] torch vs onnxruntime worst diff {worst:.2e}", flush=True)
    except Exception as e:
        print(f"[bench] onnxruntime not available ({e}); skipping CPU bench",
              flush=True)


# ---------------------------------------------------------------------------
# Subcommand: plot (training curves from train_<tag>.jsonl)
# ---------------------------------------------------------------------------
def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(l) for l in open(args.log) if l.strip()]
    eps = [r["epoch"] for r in recs]
    vi = [r["val_intent_acc"] for r in recs]
    vf = [r["val_fine_acc"] for r in recs]
    tl = [r["train_loss"] for r in recs]
    best_ep = max(recs, key=lambda r: r["val_intent_acc"])["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(eps, vi, color="#1f77b4", lw=2, marker="o", ms=3,
             label="Val intent acc")
    ax1.plot(eps, vf, color="#ff7f0e", lw=2, marker="s", ms=3,
             label="Val fine acc")
    ax1.axvline(best_ep, color="gray", ls="--", lw=1, alpha=0.7)
    ax1.annotate(f"best @ ep{best_ep}\n({max(vi):.3f})", xy=(best_ep, max(vi)),
                 xytext=(best_ep + 2, max(vi) - 0.06), fontsize=8, color="gray")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Validation accuracy per epoch")
    ax1.set_ylim(0, max(max(vi), max(vf)) * 1.15)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.plot(eps, tl, color="#2ca02c", lw=2, marker="o", ms=3, label="Train loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (fine + 0.5·intent)")
    ax2.set_title("Training loss per epoch")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = args.out or os.path.join(OUT_DIR, f"curves_{os.path.basename(args.log).replace('.jsonl', '')}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[plot] saved {out}", flush=True)
    print(f"[plot] best val_intent={max(vi):.4f}@ep{best_ep}  "
          f"final val_intent={vi[-1]:.4f}  final train_loss={tl[-1]:.4f}",
          flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="download HF shards + build feature cache")
    p.add_argument("--splits", default="train,test,holdout")
    p.add_argument("--tag", default="")
    p.add_argument("--sup", action="store_true",
                   help="merge supplemental_synth (by voice_split), "
                        "supplemental_fil (train only) and synthetic_negatives "
                        "(train/test) into the splits before caching")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="train a model")
    p.add_argument("--arch", default="crnn", choices=["crnn"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--ls", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tag", default="model")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--numerals", type=int, default=0,
                   help="subsample N SpeechCommands_v2 clips as extra "
                        "OUT_OF_SCOPE train negatives")
    p.add_argument("--sup", action="store_true",
                   help="merge supplemental_synth (by voice_split), "
                        "supplemental_fil (train only) and synthetic_negatives "
                        "(train) into the train rows; requires the cache built "
                        "with prepare --sup --tag <tag>")
    p.add_argument("--cache-tag", default="",
                   help="feature-cache tag to read (e.g. 'sup' for the "
                        "supplemental cache built by prepare --sup --tag sup)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="evaluate a checkpoint on test/holdout")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", default="model")
    p.add_argument("--split", default="test", choices=["test", "holdout", "train"])
    p.add_argument("--cache-tag", default="",
                   help="feature-cache tag to read (must match the split's "
                        "row order the checkpoint was trained with)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("export", help="export a checkpoint to ONNX + meta")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("plot", help="plot training curves")
    p.add_argument("--log", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_plot)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
