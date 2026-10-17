# rpi_inf — model-inference measurement

Everything about **measuring the models** lives here; the deployable runtime
stays in [`../rpi/`](../rpi/) (executable-only: `runtime/`, `model/`,
`setup.sh`, `run.sh`).

| File | What it does |
|---|---|
| `perf.py bench` | VCM accuracy (fine/intent), OOS false-accept @ `min_conf`, latency p50/p95, RTF, onnxruntime thread sweep — on the deployed `rpi/model/crnn.onnx` + the test-split feature cache |
| `perf.py wake` | wake-word keyword accuracy, false-accept rate, AUC + threshold sweep — on the deployed `rpi/model/jarvis_cnn.onnx` + `dataset/Jarvis/` |
| `perf.py live` | re-print the latest `session_*.json` (recorded by the runtime via `import perf`) |
| `perf.py report` | aggregate every JSON in `perf_results/` |
| `gen_responses.py` | OFFLINE: regenerate the 40 spoken-response MP3s in `rpi/audio/responses/` with piper (`en_GB-alan-medium`); imports the dispatcher so MP3s always match `actions.py` wording. Run on any machine with `pip install piper-tts` + ffmpeg; the Pi never needs it |

## Run

```bash
python3 perf.py bench          # needs: model/cache/testsup.npy (prepare) + rpi/model/*
python3 perf.py wake           # needs: rpi/model/jarvis_cnn.onnx + dataset/Jarvis/
python3 perf.py report
```

The runtime records its own live sessions through this module
(`rpi/runtime/vcm.py` does `sys.path` → `../rpi_inf`, `import perf`), so
`session_*.json` files land here too.

## Environments

| Where | Venv | Notes |
|---|---|---|
| Mac | `../.venv_inf` | runs `perf.py bench/wake/report`, `check_contract.py`; created by `../setup_env.sh` (pinned: `../requirements-inf.txt`) |
| Pi | `../rpi/.venv` | `perf.py` is torch-free by design — it runs inside the runtime venv, no separate install needed |

`gen_responses.py` additionally needs `piper-tts` + ffmpeg on the Mac
(`pip install piper-tts` inside `.venv_inf`); the Pi never needs it.

## Results so far

`perf_results/` holds the committed bench/wake JSONs. Latest — **Oct 16,
retrained deployed weights, both machines done** (the Pi 4 spot-check that was
open after the Oct 14 run has since been completed):

| File | Machine | Headline |
|---|---|---|
| `bench_20261016-192030.json` | Mac arm64 | fine 88.1% / intent 92.5%, accepted-acc @ `min_conf` 87.8%, CRNN p95 **0.57 ms**, RTF 0.00036 |
| `bench_20261016-213215.json` | **Pi 4** (aarch64) | CRNN mean **9.26 ms** / p95 **9.36 ms**, RTF 0.0058 (100 warm-up + 100 timed) |
| `wake_20261016-192043.json` | Mac arm64 | kw acc 1.00, FAR 0.0013, AUC 1.00 on the `dataset/Jarvis/` eval |
| `wake_20261016-213457.json` | **Pi 4** | kw acc 1.00, FAR 0.0013, AUC 1.00; wake forward mean 4.73 ms / p95 4.85 ms |

Note the two AUCs measure different slices: the numbers above are `perf.py
wake` on the full `dataset/Jarvis/` eval set; the held-out 622-window test-slice
metrics (AUC 0.9975, real-room-neg max 0.065) live in
`../model/wakeword/cnn_out/report.json` and are the ones quoted in the report.
The deployed wake gate is `wake_threshold = 0.99` in `rpi/runtime/config.json`.
