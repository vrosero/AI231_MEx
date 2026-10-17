# 05 · Efficiency — why this model is "tiny"

!(figures/fig_efficiency.png)

## 1. Model complexity

| Model | Params | FLOPs (forward, batch 1) | ONNX size |
|---|---:|---:|---:|
| **CRNN (deployed VCM)** | **0.367 M** | **62.78 M** | ~1.6 MB |
| Wake CNN ("Hello Jarvis") | 0.249 M | ~18 M | 1.0 MB |

The CRNN's CNN trunk pools 40×96 → 5×12 before the BiGRU sees the sequence, so
temporal modelling costs only 12 GRU steps over 192-dim frames. Total resident
model footprint on the Pi (command model + wake gate) is ~2.6 MB of weights.

## 2. Latency (measured, prof's protocol: 100 warm-up + 100 timed runs)

| Model | Mac (ARM) mean / p95 | Raspberry Pi 4 mean / p95 | RTF (Pi, p95) |
|---|---|---|---:|
| **CRNN (deployed VCM)** | **0.56 / 0.57 ms** | **9.26 / 9.36 ms** | 0.0058 |
| Wake CNN ("Hello Jarvis") | **0.54 / 0.57 ms** | **4.73 / 4.85 ms** | 0.0049 |

Both models were timed on both machines with the same protocol —
100 warm-up runs, then 100 timed forward passes (`sess.run` only, high-resolution
`perf_counter`); the result JSONs carry `"warmup": 100` for audit, and live in
`rpi_inf/perf_results/` on both machines (both re-run Oct 16 on the retrained models).
A thread sweep (1/2/4 ORT threads, p95): Mac 0.80 / 0.61 / 0.51 ms,
Pi 12.60 / 9.35 / 7.89 ms — the deployed runtime uses 2 threads.

The runtime loop runs on a **1 s audio frame budget**; the measured Pi 4
latency uses <1 % of it. The rest of the budget is the numpy log-mel front-end
(**3.3 ms** measured on the Pi for a 1 s @ 12 kHz window) and the wake gate
(4.85 ms p95 per scored frame, only while voiced).

## 3. Runtime memory

| Component | Footprint |
|---|---|
| onnxruntime session (CRNN) | ~20 MB RSS |
| numpy feature buffer (1 s @ 12 kHz) | ~50 KB |
| MP3 playback (mpv/ffplay, one file at a time) | ~15 MB while speaking |
| **Whole assistant (idle)** | **< 120 MB RSS** — comfortable in the Pi 4's 4 GB |

## 4. The always-on cost is near zero

The wake gate is event-driven: an RMS pre-gate discards silence at C-level cost, and
the wake CNN only scores while the voiced latch is open (0.6 s past the last voiced
frame). Idle CPU on the Pi sits at ~0 %, so the device can run 24/7 from a normal
phone-charger power supply.

## 5. Deployment artefacts

```
rpi/model/crnn.onnx      CRNN graph + weights (+ _meta.json: feature config, 37 fine / 20 intent maps)
rpi/model/jarvis_cnn.onnx        1.0 MB wake gate + meta (same folder as the CRNN)
rpi/audio/responses/*.mp3        40 pre-recorded spoken confirmations (shipped)
rpi/tts/en_GB-alan-medium.onnx   63 MB Piper voice — OFFLINE regeneration only
                                 (rpi_inf/gen_responses.py, gitignored), never needed on the Pi
```

`onnxruntime` (CPUExecutionProvider) is the only ML dependency on the Pi — no
PyTorch, no librosa, no cloud SDK in the model path.

**Next:** [06_wakeword.md](06_wakeword.md)