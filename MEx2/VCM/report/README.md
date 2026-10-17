# VCM — Tiny Voice Command Model for Smart Devices

**Deliverable:** VCM — real-world voice-command demo on Raspberry Pi
**Device:** Raspberry Pi 4 (4 GB) + ReSpeaker 2-Mics HAT + USB speaker *(only two devices)*
**Wake word:** "Hello Jarvis" · **Command model:** CRNN, 0.367 M parameters · **Fully on-device, no cloud, no LLM, no ASR**

> **Read me first.** This README is the hub of the report. Each numbered section is a
> standalone markdown file next to this README (with its own figures), so nothing is
> too long to read in one sitting. A compiled **PDF** and a slide deck **PPTX** of the
> same content are in this folder too.

---

## What this project is

A tiny, standalone voice-command system that runs entirely on a Raspberry Pi:

1. It listens continuously for the wake word **"Hello Jarvis"** (a 249 k-parameter CNN gate).
2. On wake, the ReSpeaker LEDs **blink cyan** (listening for a command).
3. A **0.367 M-parameter CRNN** (depthwise-separable CNN trunk → BiGRU) classifies the command into one of
   **20 intents / 37 fine labels** — no transcription, no ASR, no LLM.
4. The dispatcher executes a **real action**: turns the **ReSpeaker's own LEDs** on/off/
   colour/brightness (they *are* the lights — no smart bulb), plays/pauses/skips music
   from the mp3 playlist, plays the **per-slot timer chime MP3**, answers weather
   (keyless Open-Meteo, default Quezon City) and time (the Pi's own clock), and answers
   alarms, temperature, reminders, calls and messages with **pre-recorded spoken MP3s**
   (the "Jarvis" voice, synthesized offline).
5. The LEDs turn **steady green** while executing, and go **off** when done
   (or when the command is not recognized).

Everything the professor asked for — tiny, real-time on RPi, standalone, no cloud models,
no LLM, pure VCM doing commands 1–10 — is implemented on hardware.

## The 8 build items

| # | Item | Where | Status |
|---|------|-------|--------|
| 1 | Separate wake-word model, Kaggle `jarvis-wake-word-dataset`, "Hello Jarvis" | `rpi/model/jarvis_cnn.onnx`, `runtime/wakegate.py` | ✅ 249 k-param CNN, held-out AUC 0.9975, live gate 9/10 fired · 0 false wakes (Oct 16, retrained) |
| 2 | ReSpeaker LEDs: blink on wake, steady on execute, off when done/unrecognized | `runtime/leds.py` | ✅ APA102, 3-state model — and now also the lights |
| 3 | CRNN voice-command model | `rpi/model/crnn.onnx` | ✅ 0.367 M params, 37 fine / 20 intent — trained 60 eps on MPS, deployed as single-file ONNX |
| 4 | Music tasks from mp3 files | `rpi/music/`, `runtime/music.py` | ✅ play/pause/stop/next/volume ±10 % |
| 5 | Weather via Open-Meteo (keyless), default Quezon City | `runtime/weather.py` | ✅ single keyless provider |
| 6 | Time from the RPi clock | `runtime/actions.py` | ✅ `datetime.now()`, spoken |
| 7 | Light tasks = ReSpeaker LEDs (no Tapo, no Wi-Fi config) | `runtime/leds.py` via `actions.py` | ✅ on/off/brightness ×3/colour ×4 |
| 8 | TIMER/ALARM/TEMPERATURE/REMINDERS/CALL/MESSAGE without a UI | `runtime/actions.py` + `runtime/tts.py` | ✅ timer = per-slot chime MP3; the rest = pre-recorded spoken MP3 answers |

## Sections

| File | Contents |
|------|----------|
| [`01_overview.md`](01_overview.md) | Problem, constraints, architecture, pipeline figure |
| [`02_dataset.md`](02_dataset.md) | HF dataset + supplementals: 36,604 clips, splits, sources, class balance |
| [`03_model.md`](03_model.md) | CRNN architecture, training recipe (MPS) |
| [`04_results.md`](04_results.md) | Test/holdout eval, per-intent recall, OOS false-accept |
| [`05_efficiency.md`](05_efficiency.md) | Params, FLOPs, ONNX size, latency — why it is "tiny" |
| [`06_wakeword.md`](06_wakeword.md) | "Hello Jarvis" wake CNN: dataset, the from-scratch model, live verification |
| [`07_runtime_demo.md`](07_runtime_demo.md) | The 8 demo items, LED language, command→action map |
| [`08_rpi_setup.md`](08_rpi_setup.md) | **How to set up on the RPi — detailed, step by step** |
| [`09_verification.md`](09_verification.md) | What was tested on hardware, and the honest limits |

## Quick demo (after setup — see section 08)

```bash
cd ~/vcm
bash run.sh                 # live hands-free loop: mic → wake → classify → act → answer
bash run.sh --no-tts        # beeps only (disable spoken responses)
```

## Headline numbers

| | Value |
|---|---|
| Command model | CRNN, **0.367 M** params, **62.78 M** FLOPs, ~1.6 MB ONNX |
| Dataset | HF `ai231-me2-voice-commands` + supplementals: **36,604 clips / 21.0 h**, speaker-disjoint |
| Test split | 6,305 clips (incl. 250 synthetic out-of-scope negatives) + 463-clip holdout |
| CRNN accuracy | **91.6 %** intent / 87.4 % fine (test) · 88.6 % intent (holdout) · OOS recall 0.739 |
| Wake-word model | **249,281** params, held-out AUC 0.9975 (real-pos recall 0.917 @ 0.5, n=36; room-neg max 0.065) |
| Live streamed wake test (Pi) | **9/10** spoken attempts fired · **0 false fires** *(Oct 16, retrained weights, `evidence/logs_live_20261016.log`)* · original weights: 10/10 · 0/200 noise frames *(Oct 15)* |
| Real-time budget | **9.36 ms p95 CRNN / 4.85 ms p95 wake — measured on Pi 4** (prof's 100+100 protocol, Oct 16) — far under the 1 s frame budget |
| Spoken answers | pre-recorded MP3s (`en_GB-alan-medium` voice, synthesized offline) |

## Repository map

```
VCM/
├── model/                   ← training code (vcm_model.py, wakeword/) + cache + ckpt + run_outputs
├── rpi_inf/                 ← model-inference measurement (perf.py, perf_results/, gen_responses.py)
├── report/                  ← this report (01–09 md, figures, PDF/PPTX builders)
└── rpi/                     ← the deployable demo (copy this to the Pi)
    ├── runtime/             vcm.py (main) · wakegate · features · leds · music
    │                        timer · tts · weather · actions · mic · audio · config.json
    ├── model/               crnn.onnx + meta (37 fine / 20 intent)
    │                        + jarvis_cnn.onnx + meta (the "Hello Jarvis" gate)
    ├── music/               music mp3s · audio/timer/  per-slot timer chimes
    └── setup.sh · run.sh · requirements.txt · README.md
```

*Report generated 2026-10-14. Figures in [`report/figures/`](report/figures/).*