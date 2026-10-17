# 01 · Overview — problem, constraints, architecture

!(figures/fig_pipeline.png)

## 1. The problem

ASR (automatic speech recognition) models are not desirable for on-device computing
because of their footprint. The goal of VCM is to build a **tiny Voice Command Model
(VCM)** that understands the most common commands humans issue to smart devices —
ranked from real-world Alexa/Google Home logs (259,164 commands) and consumer surveys:

1. Play music
2. Ask a question (weather, time)
3. Control lights (on/off)
4. Dim / colour lights
5. Set a timer
6. Set an alarm
7. Adjust thermostat temperature
8. Media control (pause/stop/next/volume)
9. Reminders and lists
10. Calls and messaging

## 2. Hard constraints (from the brief)

| Constraint | How this project meets it |
|---|---|
| **Standalone / no cloud models** | Wake CNN + CRNN both run locally via `onnxruntime`; the only network call is the *action* side (keyless Open-Meteo weather) — never a model call |
| **No LLM** | A fixed 37-label classifier; no text generation anywhere (the spoken answers are template sentences, not generated text) |
| **No full ASR** | Audio → log-mel spectrogram → intent/fine label directly; no audio→text stage |
| **Real-time on RPi 4/5** | 62.78 M FLOPs per inference — **9.3 ms mean / 9.4 ms p95 measured on a Pi 4** (onnxruntime, Oct 16 bench) |
| **Genuinely small** | 0.367 M params (~1.6 MB ONNX) + 0.249 M-param wake gate |
| **Minimal hardware** | The only devices are the ReSpeaker 2-Mics HAT (mic + LEDs) and a USB speaker — no bulb, no hub, no API keys |

## 3. Architecture — four stages, no transcription

```
mic ──▶ [0] WAKE GATE      RMS pre-gate + "Hello Jarvis" CNN (always-on, tiny)
     ──▶ [1] FEATURES      1 s window → log-mel 40×96, z-scored (numpy)
     ──▶ [2] CLASSIFIER    CRNN dual head: 37 fine labels + 20 intents
     ──▶ [3] DISPATCHER    plain Python → real action (LEDs / music / timer / TTS)
```

* **Stage 0 — wake gate.** An RMS energy pre-gate keeps idle CPU near zero; when
  energy rises, a rolling 1 s buffer is scored by the wake CNN. Only a genuine
  "Hello Jarvis" opens the command window (and blinks the LEDs).
* **Stage 1 — features.** Plain numpy log-mel (40 mel bands, 96 frames, 12 kHz),
  byte-identical to the training front-end — no librosa dependency at runtime.
* **Stage 2 — classifier.** One CRNN (depthwise-separable CNN trunk → BiGRU)
  with two heads: the
  *fine* head carries the slot value (e.g. `TIMER_10s`, `COLOR_BLUE`,
  `ALARM_8_00AM`), the *intent* head carries the action group (e.g. `TIMER`,
  `COLOR`, `ALARM`). The dispatcher routes on the fine label; the intent head is
  the task-success metric.
* **Stage 3 — dispatcher.** A plain-Python `dispatch(fine_label, ctx)` maps each
  of the 37 labels to a real action and returns `{ok, action, detail}`; every
  command except music/lights/timer also answers with a pre-recorded spoken
  MP3 (the "Jarvis" voice, synthesized offline). The ReSpeaker LEDs **are**
  the lights (no smart bulb, no web UI).

## 4. Design choices that make it tiny

* **Depthwise-separable convolutions** (Chollet 2017) in the trunk keep the
  whole forward pass at 62.78 M FLOPs — comfortably inside the 1 s frame budget.
* **CNN-then-RNN factoring**: the trunk pools 40×96 → 5×12 *before* the BiGRU, so
  temporal modelling costs 12 GRU steps over 192-dim frames — not 96 steps over
  40-dim raw audio.
* **Dual-head output** shares one trunk between slot and intent — one forward pass,
  two decisions.
* **Closed-set slots** instead of transcription: durations, times, colours and
  brightness levels are enumerated classes, not free text.
* **ONNX Runtime** on the Pi (CPUExecutionProvider) — no PyTorch at runtime.

## 5. What "success" means here

Evaluation is reported in section 04 on the merged speaker-disjoint test split
(6,305 clips, incl. 250 purpose-built out-of-scope negatives) and the untouched
holdout (463 clips): intent/fine accuracy, macro-F1, per-intent recall, and the
out-of-scope false-accept rate — the honest numbers for a real device, not an
in-domain ceiling.

The demo in section 07 runs the full loop on the Pi; the setup is in section 08.

**Next:** [02_dataset.md](02_dataset.md)