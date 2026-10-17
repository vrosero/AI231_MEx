# 08 · How to set up on the Raspberry Pi — detailed guide

!(figures/fig_setup.png)

This guide assumes a **Raspberry Pi 4 (4 GB)**, a **ReSpeaker 2-Mics HAT**, and a
**USB speaker**. No monitor or keyboard is needed — everything is SSH. No smart
bulb, no hub, no API keys: the only devices are the ReSpeaker (mic + LEDs) and
the speaker, and the only configuration is Wi-Fi. Every command below is
copy-pasteable.

---

## Part A — one-time install (~30 min)

### Step 1 · Flash the SD card

On your laptop, with **Raspberry Pi Imager**:

1. Choose **Raspberry Pi OS (64-bit)** — Lite is fine, Desktop is fine too.
2. Press **Ctrl+Shift+X** (the cog icon) and set:
   * Hostname: `megumi-rpi`
   * Enable SSH: **yes**, with password authentication
   * Username / password: `megumi_rpi` / *your password*
   * Wi-Fi: your **home** network name + password *(this is the default network)*
   * Locale/timezone: your own
3. Write the image, insert the SD card into the Pi, attach the **ReSpeaker HAT on
   the 40-pin header**, plug in the USB speaker, attach power.
4. **Wait up to 5 minutes** for first boot.

### Step 2 · Find the Pi and SSH in

From your laptop (same Wi-Fi):

```bash
ssh megumi_rpi@megumi-rpi.local        # or use the IP your router shows
```

### Step 3 · Copy the project over

```bash
# from the laptop
scp -r VCM/rpi megumi_rpi@megumi-rpi.local:~/vcm
```

The folder already contains everything: the CRNN ONNX model, the wake-word CNN,
the runtime code, the music/timer audio, the pre-recorded spoken responses, and
the setup scripts. No voice model is downloaded — spoken answers are MP3s.

### Step 4 · Enable SPI (for the LEDs)

```bash
sudo raspi-config nonint do_spi 0      # enable SPI
# or: add dtparam=spi=on to /boot/firmware/config.txt and reboot
```

Without SPI the LEDs no-op but the voice loop still runs — the assistant is
usable even if this step is skipped.

### Step 5 · One-command setup

```bash
cd ~/vcm
bash setup.sh
```

`setup.sh` does everything:

1. system packages (PortAudio, mpv, SPI, venv, ffmpeg),
2. a Python virtualenv + `pip install -r requirements.txt`
   (`numpy, onnxruntime, sounddevice, apa102`),
3. Wi-Fi (SSID/password) switch — **the only configuration the device needs**,
4. spoken responses: pre-recorded MP3s shipped in `audio/responses/`
   (regenerate offline with `rpi_inf/gen_responses.py` — piper is not a
   runtime dependency),
5. sanity checks (model files present, microphone visible),
6. asks *"Activate the VCM program now? [Y/n]"* → hands over to `run.sh`.

## Part B — running it

```bash
bash run.sh              # live hands-free loop
bash run.sh --no-tts     # beeps only (disable spoken responses)
```

`run.sh` is idempotent: it creates the venv/config if missing, then execs
`runtime/vcm.py`. One line, no arguments, no placeholders.

## Configuration (`runtime/config.json`)

| Key | Default | Meaning |
|---|---|---|
| `wake_threshold` | 0.99 | wake CNN confidence gate |
| `wake_rms_threshold` | 0.012 | energy pre-gate |
| `min_conf` | 0.25 | command confidence gate |
| `tts_voice` | `en_GB-alan-medium` | the "Jarvis" voice |
| `weather_city` / `weather_lat` / `weather_lon` | Quezon City / 14.6357 / 121.0325 | weather location |
| `led_num` / `led_blink_hz` | 3 / 2.5 | ReSpeaker LED count and blink rate |
| `music_start_volume` | 60 | playlist start volume |

Everything has a working default — edit only to change city or voice.

## Testing without speaking

```bash
.venv/bin/python runtime/vcm.py --wav some_command.wav   # classify one wav + run its action
.venv/bin/python runtime/vcm.py --list-mics              # the HAT shows as seeed2micvoicec
```

## Benchmarks

Benchmarks live in `rpi_inf/` (the model-inference folder, not deployed):

```bash
cd ../rpi_inf
python3 perf.py bench    # VCM: accuracy, false-accept rate, latency p95, RTF
python3 perf.py wake     # wake word: keyword acc, false-accept rate, AUC
python3 perf.py live     # re-print the latest session report
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `APA102 unavailable` in the log | LEDs need SPI: `sudo raspi-config nonint do_spi 0` and reboot. The voice loop still runs without them. |
| No sound from the speaker | `aplay -l` to find it; make it the default sink: `wpctl set-default <id>` (or `pactl set-default-sink`). |
| Mic not found | `python3 runtime/vcm.py --list-mics`; the HAT shows as `seeed2micvoicec`. |
| Wake word never fires | check `wake_threshold` (0.99) and `wake_rms_threshold` (0.012) in `runtime/config.json`; speak ~30 cm from the mics. |
| Weather fails | it uses Open-Meteo (no key) — needs internet. City/lat/lon in `runtime/config.json`. |

**Next:** [09_verification.md](09_verification.md)