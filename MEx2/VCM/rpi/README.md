# VCM — Raspberry Pi voice assistant ("simple Alexa")

A hands-free voice command assistant for a Raspberry Pi, built around the
**VCM CRNN** command model (37 fine / 20 intent classes, 0.367 M parameters)
trained on the `airimonda/ai231-me2-voice-commands` dataset (+ supplementals).

Say **"Hello Jarvis"**, then a command. The ReSpeaker LEDs blink while it
listens, stay solid while it runs the action, and turn off when done (or
immediately if it didn't recognize the command).

```
ReSpeaker 2-Mics HAT (12 kHz)
  -> wake gate      energy pre-gate  +  jarvis_cnn.onnx "Hello Jarvis" CNN
  -> LEDs BLINK     (listening)
  -> capture ≤1.6 s command window (closes 0.4 s after you stop talking)
  -> log-mel [40,96]-> CRNN ONNX     (crnn.onnx)
  -> recognized?    LEDs SOLID -> run action -> LEDs off
  -> not recognized LEDs off
```

## Hardware (only two devices)

| Device | Role |
|---|---|
| **ReSpeaker 2-Mics HAT** | microphone + the 3 APA102 status LEDs (which are also the "lights") |
| **USB speaker / 3.5 mm speaker** | all audio out: music, timer chimes, spoken responses, beeps |

No smart bulb, no hub, no API keys. The Pi needs only Wi-Fi + SSH.

## Setup — two commands

```bash
bash setup.sh     # installs packages + venv; checks the response MP3s; asks Wi-Fi once
bash run.sh       # starts the assistant
```

That's it. `setup.sh` needs sudo once (apt packages). Wi-Fi is the only
configuration; everything else has working defaults (`runtime/config.json`).

`rpi/.venv` is the **Pi runtime** environment only. Training and measurement
use the project's own venvs on the Mac (`.venv_train`, `.venv_ww`, `.venv_inf`
— see the root `README.md`); they are never created here.

## The commands — all 37 fine labels

Every command answers with a **spoken response (pre-recorded MP3, "Jarvis"
voice)** except music, the timer chime, **TIME and WEATHER** (they log their
answer instead of speaking it), all of which act for real and stay silent:

| Intent | Fine labels | What happens |
|---|---|---|
| LIGHT_ON / LIGHT_OFF | `LIGHT_ON`, `LIGHT_OFF` | ReSpeaker LEDs on / off |
| BRIGHTNESS | `BRIGHTNESS_20`, `BRIGHTNESS_60`, `BRIGHTNESS_100` | LED brightness 20/60/100 % |
| COLOR | `COLOR_RED`, `COLOR_BLUE`, `COLOR_GREEN` | LED colour |
| PLAY_MUSIC / PAUSE / STOP / NEXT | same | 5-track mp3 playlist (`mpv`, ffplay fallback) |
| VOLUME_UP / VOLUME_DOWN | same | playlist volume ±10 % |
| TIMER | `TIMER_10s`, `TIMER_30s`, `TIMER_1m` | plays that slot's chime from `audio/timer/` |
| WEATHER | `WEATHER` | fetches real weather (Open-Meteo, keyless; default Quezon City) — answer in the log, no TTS |
| TIME | `TIME` | reads the Pi's clock — answer in the log, no TTS |
| ALARM | `ALARM_6_00AM`, `ALARM_8_00AM`, `ALARM_9_00PM` | spoken confirmation (simulated) |
| TEMPERATURE | `TEMPERATURE_18`, `TEMPERATURE_22`, `TEMPERATURE_26` | spoken confirmation (simulated) |
| CREATE_REMINDER | `..._DRINK_WATER`, `..._EXERCISE`, `..._STUDY` | spoken confirmation; kept in memory |
| LIST_REMINDERS | `LIST_REMINDERS` | speaks the reminder list |
| CALL / MESSAGE | `CALL`, `MESSAGE` | spoken confirmation (simulated) |
| (anything else) | `OUT_OF_SCOPE` | LEDs off, error beep — safely ignored |

## Folder layout

```
rpi/
├── run.sh               # ONE-LINE START: bash run.sh
├── setup.sh             # one-time install (packages + venv + Wi-Fi)
├── requirements.txt
├── model/               # ALL models: crnn.onnx (VCM) + jarvis_cnn.onnx
│                        #   (+ _meta.json each) — command model + wake gate
├── runtime/             # vcm.py main + features/mic/wakegate/actions/
│                        # music/timer/tts/leds/weather/audio
├── music/               # music_1..5.mp3
├── audio/timer/         # TIMER_10s.mp3 / TIMER_30s.mp3 / TIMER_1m.mp3
├── audio/responses/     # 40 pre-recorded spoken confirmations (shipped)
└── tts/                 # piper voice, ONLY for regenerating responses (gitignored)

../rpi_inf/               # model-inference measurement (NOT deployed to the Pi)
├── perf.py               # benchmarks: bench / wake / live / report + session recorder
└── perf_results/         # bench/wake/session JSONs + confusion CSV
```

## Testing without speaking

Classify one wav and run its action, then exit:

```bash
.venv/bin/python runtime/vcm.py --wav some_command.wav
```

List microphones (check the ReSpeaker is seen):

```bash
.venv/bin/python runtime/vcm.py --list-mics
```

## Benchmarks (deck numbers)

Benchmarks live in `../rpi_inf/` (the model-inference folder — not deployed
to the Pi):

```bash
cd ../rpi_inf
python3 perf.py bench    # VCM: accuracy, false-accept rate, latency p95, RTF,
                         #      onnxruntime thread sweep (needs test cache)
python3 perf.py wake     # wake word: keyword acc, false-accept rate, AUC
python3 perf.py live     # re-print the latest session report
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `APA102 unavailable` in the log | LEDs need SPI: add `dtparam=spi=on` to `/boot/firmware/config.txt` and reboot. The voice loop still runs without them. |
| No sound from the speaker | `aplay -l` to find it; make it the default sink: `wpctl set-default <id>` (or `pactl set-default-sink`). |
| Mic not found | `python3 runtime/vcm.py --list-mics`; the HAT shows as `seeed2micvoicec`. |
| Wake word never fires | check `wake_threshold` (0.99) and `wake_rms_threshold` (0.012) in `runtime/config.json`; speak ~30 cm from the mics. |
| Weather fails | it uses Open-Meteo (no key) — needs internet. City/lat/lon in `runtime/config.json`. |