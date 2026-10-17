# Jarvis wake-word CNN — training notes

**Script:** `model/wakeword/cnn/train_cnn.py` · **Seed:** 42

## Data (5,816 clips)

| Source | Clips | Split (train/val/test) |
|---|---:|---|
| `positive/` synthetic TTS (6 phrasings × 200) | 1200 | 960 / 120 / 120 |
| `positive_real_1/` + `positive_real_2/` real recordings | 122 | 62 / 24 / 36 |
| `negative/` command words (10 × 300) | 3000 | 2400 / 300 / 300 |
| `noise/` MUSAN/ESC-50 | 800 | 640 / 80 / 80 |
| `negative_real/` real-room windows (kept set) | 694 | 521 / 87 / 86 |
| **Total** | **5816** | **4583 / 611 / 622** |

Room windows stay block-interleaved (every 8th → test, every 8th+3 → val) so
overlapping windows from the same sound event never straddle the split.

## Results (best epoch 20, val AUC 1.0 — first hit at epoch 11; the
## neighbouring jarvis_cnn.meta.json previously said 23, written by an older
## train_cnn.py — corrected Oct 16 to match model.pt / report.json)

| Test slice | n | @0.50 |
|---|---:|---|
| All | 622 | AUC 0.9975 · P .9935 / R .9808 / FPR .0021 / F1 .9871 |
| Real positives | 36 | recall .9167 (mean 0.9012, min 0.0003) |
| Synthetic positives | 120 | recall 1.000 |
| Neg + noise (synthetic) | 380 | FPR .0026 (max 0.7365) |
| **Real-room negatives** | **86** | **FPR 0.000 (max 0.0649)** |

Real-positive recall by phrasing @0.50: hello_jarvis 1.00 · hey_jarvis 0.78 ·
jarvis 0.83 · jarvis_please 1.00 · jarvis_wake_up 1.00 · okay_jarvis 1.00.

At the deployed threshold 0.99 (final value, Oct 16) the room-negative FPR is 0.

## Artifacts

- `model/wakeword/cnn_out/` (canonical copy): `model.pt`, `split.json`,
  `history.json`, `report.json`, `jarvis_cnn.onnx`, `jarvis_cnn.meta.json`.
- Deployed copy: `rpi/model/jarvis_cnn.onnx` + `jarvis_cnn.meta.json`
  (byte-identical to the canonical copy).
- ONNX: opset 13, ir_version 7, **single file** (997,871 B), exported with
  the legacy TorchScript exporter (`dynamo=False`), which keeps the input a
  clean 4-D `(B,1,40,98)` tensor. (The newer dynamo exporter emits opset 18 +
  an external-data sidecar and a rank-3 input bug; it is not used.)
- Input contract: `(B,1,40,98)` log-mel, sigmoid prob.

## Real-room negatives

The `negative_real/` windows are the headline metric: they are recordings of
the actual deployment rooms, so the room-negative FPR (max score 0.0649 @0.50,
0.0 FPR) is the number that predicts real-world false wakes. The
block-interleaved split keeps overlapping windows of the same sound event on
one side of the split.

## Per-phrasing recall loop

The `real_by_phrase` report loop indexes `per` per phrase explicitly; a single
shared dict would carry only the last phrase's mean.