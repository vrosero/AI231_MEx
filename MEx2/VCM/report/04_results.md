# 04 · Results — the deployed CRNN

!(figures/fig_results.png)

The deployed voice-command model is the CRNN of §03, evaluated on the merged
speaker-disjoint splits of §02 (test n = 6,305 · holdout n = 463). No other
model was trained or compared — the CRNN is the one model this project built,
deployed, and measured.

## 1. Test split (n = 6,305, speaker-disjoint)

| Metric | Value |
|---|---:|
| Intent accuracy | **91.6 %** |
| Intent macro-F1 | 90.3 % |
| Fine accuracy | 87.4 % |
| FAR_oos (min_conf 0.25) | 0.261 |
| OUT_OF_SCOPE recall | 0.739 |

Per-intent test recall: strongest **ALARM 0.970 · TEMPERATURE 0.969 ·
TIMER 0.966**; weakest **OUT_OF_SCOPE 0.739 · VOLUME_UP 0.785 · LIGHT_OFF
0.840**.

## 2. Holdout (n = 463, unseen speakers)

**88.6 %** intent (macro-F1 85.3, fine 81.4) — the honest generalisation
number, measured on clips the model never saw in any form.

## 3. Latency

MPS inference: **4.2 ms mean / 4.4 ms p95** (batch 1, measured in eval);
onnxruntime CPU measured on both machines with the prof's protocol (100 warm-up +
100 timed runs; both re-run Oct 16 on the retrained models):

| Model | Mac (ARM) p95 | Raspberry Pi 4 p95 | RTF (Pi) |
|---|---:|---:|---:|
| CRNN (command) | **0.57 ms** | **9.36 ms** | 0.0058 |
| Wake CNN | **0.57 ms** | **4.85 ms** | 0.0049 |

Full JSONs (with `warmup: 100` for audit) are in `rpi_inf/perf_results/`
on both machines: the Oct 16 `bench_20261016-213215` and `wake_20261016-213457` result files.

## 4. The honest caveat — real-voice out-of-scope

The 16 real-voice OOS holdout clips remain hard (FAR_oos 0.938 on that tiny
subset — the model still mis-accepts real non-command speech; synthetic-negative
rejection is what improved). Real-voice OOS clips are scarce in train (270);
the synthetic negatives taught the model the *kinds* it was shown, not the
domain of real non-command speech. This is a data problem, not a modelling one,
and stays the top item for the next data round; the runtime's `min_conf 0.25`
gate remains the safety net.

**Next:** [05_efficiency.md](05_efficiency.md)