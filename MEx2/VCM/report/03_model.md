# 03 · Model — CRNN architecture and training

!(figures/fig_crnn.png)

## 1. Primary model — `PiCommandCRNN` (0.367 M parameters)

A **CRNN**: a depthwise-separable CNN front-end compresses the log-mel
spectrogram, a bidirectional GRU models its temporal order, and a dual head
emits fine + intent logits.

| Block | Output |
|---|---|
| Input | 1 × 40 × 96 log-mel |
| Conv 3×3 × 48 + BN + ReLU | 48 × 40 × 96 |
| DSConv (dw 3×3 + pw 1×1, 48 ch) | 48 × 40 × 96 |
| MaxPool 2×2 | 48 × 20 × 48 |
| DSConv ×2 (96 ch) | 96 × 20 × 48 |
| MaxPool 2×2 | 96 × 10 × 24 |
| DSConv (192 ch) | 192 × 10 × 24 |
| MaxPool 2×2 | 192 × 5 × 12 |
| Dropout2d | 192 × 5 × 12 |
| squeeze frequency → BiGRU(192 → 128, 2 dirs) | 256 × 12 |
| temporal mean pool | 256 |
| Head: Linear 256→256 + ReLU + Dropout | 256 |
| **Head A** — fine | **37 logits** |
| **Head B** — intent | **20 logits** |

* **Why the RNN:** the CNN compresses frequency (40 → 5 mel bands) while keeping
  12 time frames; the bidirectional GRU models their *temporal order*, which a
  purely pooled CNN discards — commands are sequences ("set a timer for ten
  seconds"), not bags of frames.
* **Loss:** `CE(fine) + 0.5 · CE(intent)` — the fine head is the primary signal;
  the intent head regularises toward the action grouping and is the
  checkpoint-selection metric.
* **Cost:** 62.78 M FLOPs (analytic, GRU cost computed in closed form:
  2 directions × 3 gates × (192·128 + 128²) × 12 frames) — §05.

## 2. Training recipe

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW, weight decay 1e-4 |
| Learning rate | 1e-3, cosine annealing over 60 epochs |
| Batch | 128 |
| Label smoothing | 0.05 (both heads) |
| Dropout | 0.35 (+ Dropout2d in the trunk) |
| Gradient clip | 5.0 (global norm) |
| Epochs | 60 |
| Seed | 42 |
| Data | merged splits (`--sup`): 28,849 train / 987 val (speaker-disjoint, 33 speakers) |
| Hardware | **Apple M-series MPS** (local Mac) — no cluster needed |

Checkpoint selection is on **validation intent accuracy**; the exported artefact
is ONNX (opset 13) with a `_meta.json` carrying the feature config, label maps
and `fine2intent`/`fine2transcript` tables so the Pi runtime needs no other
state. The export step also runs an onnxruntime CPU benchmark and a
torch-vs-ORT parity check (per-element max diff) before the artefact is
considered valid.

## 3. Reproducibility

```bash
# on the Mac (feature cache prebuilt with prepare --sup --cache-tag sup)
python model/vcm_model.py train --arch crnn --epochs 60 --sup --cache-tag sup --tag crnn
python model/vcm_model.py eval  --ckpt ckpt/crnn.pt --tag crnn
python model/vcm_model.py export --ckpt ckpt/crnn.pt --out run_outputs/onnx/crnn.onnx
```

**Next:** [04_results.md](04_results.md)