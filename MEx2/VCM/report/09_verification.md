# 09 · Verification — what was tested on hardware, and the honest limits

## 1. Hardware verification (Raspberry Pi 4, ReSpeaker 2-Mics HAT, USB speaker)

Verified on the Pi across the build rounds; the deployed stack is the CRNN
command model + the jarvis wake CNN, with the ReSpeaker LEDs as the lights and
no web UI.

| # | Component | Test | Result |
|---|---|---|---|
| 1 | Mic | `mic.py` capture on the ReSpeaker (`seeed2micvoicec`, hw:3,0) | ✅ 16 kHz capture, decimated to 12 kHz (hardware rejects 12 kHz directly) |
| 2 | Wake gate | 10 real "Hello Jarvis" clips streamed through `gate.process()`; retrained weights re-tested live at the Pi (9 spoken attempts + quiet-room time) | ✅ 10/10 fired (scores 0.994–0.999) · 200 noise frames → 0 false fires *(Oct 15, original weights)* · retrained weights live Oct 16: **9/10 fired, 0 false fires** (`evidence/logs_live_20261016.log`) |
| 3 | LEDs | cycle test: listening blink → executing steady → off | ✅ APA102 (GPIO bitbang variant), 3-state model |
| 4 | Command model | CRNN loads + runs on aarch64 ONNX Runtime | ✅ features byte-match training spec |
| 5 | Music | play/stop/next with ffplay fallback | ✅ audible playback |
| 6 | TTS spoken responses | pre-recorded MP3s (`en_GB-alan-medium` voice, `audio/responses/`) on ALARM/TEMPERATURE/REMINDERS/CALL/MESSAGE | ✅ spoken answers end-to-end (no runtime TTS) |
| 7 | Timer | per-slot chime MP3 (`TIMER_10s/30s/1m.mp3`) via the music backend | ✅ chime plays at slot end |
| 8 | Weather | Open-Meteo keyless query (Quezon City) | ✅ real conditions spoken |
| 9 | Lights = ReSpeaker LEDs | `LIGHT_ON/OFF`, `BRIGHTNESS_*`, `COLOR_*` → `leds.py` | ✅ dispatcher wired; visual check on the HAT |
| 10 | Full loop | `vcm.py` live: mic → wake → classify → dispatch → LEDs → spoken answer | ✅ ran end-to-end |
| 11 | **CRNN artefact** | single-file ONNX converted + parity-checked (torch vs ORT max diff 2.0e-06), deployed to `rpi/model/` | ✅ converted + deployed; runs live on aarch64 ORT — Pi live loop classifies commands with it (Oct 16 log, `evidence/logs_live_20261016.log`) |

## 2. Known limits (stated plainly)

1. **Real-voice out-of-scope is the honest weak point.** The model still
   mis-accepts real non-command speech (§04); the runtime's `min_conf 0.25`
   gate fails safe, and the next data round targets real OOS clips.
2. **The CRNN is 0.367 M parameters** — tiny and real-time (62.78 M FLOPs,
   **9.3 ms mean / 9.4 ms p95 measured on the Pi 4**, prof's 100+100 protocol);
   the BiGRU was chosen deliberately for temporal modelling.
3. **Wake model is trained mostly on TTS data** (the wake training adds 122 real
   clips + 694 real-room negative windows; live scores 0.994–0.999, room-neg
   max 0.065 vs threshold 0.99). Live-mic spoken wake tests on the Pi are
   still recommended before the demo; the deployed `wake_threshold` is 0.99
   (raised from 0.5 after live-room tuning, final value set Oct 16).
4. **Spoken answers play an MP3 (~0.2 s)** — synthesis is offline
   (`rpi_inf/gen_responses.py`); the runtime only plays files. Music, lights
   and timer commands answer by *acting*, not speaking, so they are unaffected.
5. **mpv** enables pause/volume; without it, the ffplay fallback plays but cannot
   pause.
6. **Simulated actions are spoken only** — ALARM, TEMPERATURE, CREATE_REMINDER,
   CALL, MESSAGE have no physical actuator (by design; the brief's UI was
   removed). Reminders persist only in memory.

## 3. Future work (in priority order)

| Priority | Item | Why |
|---|---|---|
| 1 | ~~Verify the CRNN ONNX on the Pi~~ **done Oct 16** | aarch64 ORT latency measured: CRNN p95 9.36 ms, wake p95 4.85 ms (prof's 100+100 protocol, retrained models) |
| 2 | ~~Live-mic wake + command threshold tuning in the actual demo room~~ **done Oct 15** | `wake_threshold` raised 0.5 → 0.75 after live-room tuning; 247 live wakes across 11 sessions; raised again to 0.99 on Oct 16 (final deployed value) |
| 3 | Record human clips for the weakest intents flagged by the confusion matrix | closes remaining domain gaps |
| 4 | Quantise the CRNN to int8 | ~4× smaller again; current Pi p95 is already 9.4 ms, so this is optional |

## 4. Deliverable checklist

| Requirement (from the brief) | Status |
|---|---|
| Build a dataset to train VCMs (collective) | ✅ HF `ai231-me2-voice-commands` + supplementals (36,604 clips / 21.0 h), documented in §02 |
| Build and train a VCM (individual) | ✅ CRNN 0.367 M (the deployed model), §03–04 |
| Design a benchmark for validating VCMs (collective) | ✅ speaker-disjoint splits, intent + fine + macro-F1 + per-intent recall + OOS false-accept, §02/§04 |
| Validate the performance of your VCM (individual) | ✅ test/holdout eval + confusion analysis, §04 |
| Build a real-world demo on RPi4/5 (individual) | ✅ full loop verified on the Pi; CRNN artefact deployed, §07/§09 |
| VCM must be tiny, real-time on RPi4/5 | ✅ 0.367 M params, 62.78 M FLOPs, **9.36 ms p95 measured on Pi 4**, §05 |
| Standalone — no cloud models | ✅ all inference on-device; network only for the weather *action* (keyless Open-Meteo), §01 |
| No LLM, no ASR — pure VCM doing commands 1–10 | ✅ 37-label classifier + dispatcher + spoken responses, §01/§07 |

---

*End of report. Back to the [README](README.md).*