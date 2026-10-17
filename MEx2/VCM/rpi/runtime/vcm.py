#!/usr/bin/env python3
"""VCM - a voice assistant on the Raspberry Pi.

Pipeline (end to end):
    ReSpeaker 2-Mics HAT mic (12 kHz)
      -> wake gate  (energy pre-gate + CNN "Hello Jarvis")
      -> spoken "Yes?" cue (MP3, no LED)           <- the "yes?" prompt
      -> capture command window (closes 0.8 s after you stop talking,
         max 6 s; nothing voiced for 3 s -> give up)
      -> trim to speech onset -> log-mel [40,96]  (features.py, = training)
      -> VCM ONNX         (crnn.onnx, 37 fine / 20 intent incl OOS)
      -> recognized?      -> LEDs steady (EXECUTING) -> run action
                             -> feedback flash, then lights restored
      -> not recognized?  -> ORANGE flash, then lights restored
      (only an explicit LIGHT_OFF - or shutdown - leaves the LEDs dark)
    Actions (v3 - the ONLY devices are the ReSpeaker HAT and a USB speaker):
      lights  -> the ReSpeaker's own 3 APA102 LEDs (colour/brightness states)
      music   -> mp3 playlist (mpv / ffplay)
      timer   -> plays the pre-recorded chime for that slot (audio/timer/)
      weather -> Open-Meteo (keyless), answer logged (no TTS)
      time    -> the Pi's own clock, answer logged (no TTS)
      alarm / temperature / reminders / call / message
              -> simulated, answered with a pre-recorded spoken confirmation
                 (MP3, rpi_inf/gen_responses.py - no runtime TTS)
    Feedback: every command except music, the timer chime, TIME and WEATHER
             answers with a pre-recorded MP3 spoken response ("Jarvis"
             voice, audio/responses/). The wake cue is the same voice
             saying "Yes?" (MP3). Beeps: ack/error only (audio.py
             TonePlayer). The LEDs are NOT feedback - they are the
             light command's device.

Usage:
    python3 vcm.py                 # live hands-free loop
    python3 vcm.py --wav cmd.wav   # classify one wav, run its action, exit
    python3 vcm.py --list-mics     # show input devices and exit

Config: config.json (see config.example.json).
"""
import argparse, asyncio, json, logging, os, signal, sys, threading, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# perf.py lives in the sibling rpi_inf/ tree (VCM/rpi + VCM/rpi_inf).
# Try the documented layout first (two levels up from runtime/), then the
# flattened one (rpi_inf nested beside runtime/), so either deploy works.
for _cand in (
    os.path.join(os.path.dirname(os.path.dirname(HERE)), "rpi_inf"),
    os.path.join(os.path.dirname(HERE), "rpi_inf"),
):
    if os.path.isdir(_cand):
        sys.path.insert(0, _cand)
        break

import features
import mic
import perf
import wakegate
import actions
from audio import TonePlayer
from leds import ReSpeakerLEDs, C_ORANGE
from music import MusicPlayer
from timer import TimerPlayer
from tts import TTS

MODEL_DIR = os.path.join(HERE, "..", "model")
ONNX = os.path.join(MODEL_DIR, "crnn.onnx")
META = os.path.join(MODEL_DIR, "crnn_meta.json")
WAKE_DIR = os.path.join(HERE, "..", "model")   # wake CNN lives with the VCM model
CONFIG = os.path.join(HERE, "config.json")

CMD_SECONDS = 1.6          # fallback fixed window (only when audio=None)
CONFIRM_DELAY = 1.2        # pause before resuming listening
MIN_CONF = 0.25            # below this fine-confidence => "not recognized"
FRAME_MS = 30              # mic frame size (mic.stream_frames frame_ms)
CMD_MAX_S = 1.6            # hard cap on the command window
CMD_END_SILENCE = 0.4      # window closes this long after the last voiced frame
CMD_NO_SPEECH_S = 1.0      # give up if nothing voiced for this long after cue
CUE_PHRASE = "Yes?"         # spoken wake cue (MP3 in audio/responses/)
CUE_TAIL_FRAMES = 7         # live frames dropped after the drain (~0.21 s reverb)
DRAIN_GAP_S = 0.05          # frame-arrival gap that means "mic buffer drained"
SELFDEAF_TAIL_S = 0.5       # keep ignoring the mic this long after own speech ends
SPEECH_MIN_RUN = 5         # >=150 ms of voiced frames counts as speech (onset)

class _VCMFormatter(logging.Formatter):
    """Conversational main-flow lines print bare - timestamp + message only
    (the run.sh transcript format); every other logger keeps the
    'name LEVEL message' diagnostic prefix. Warnings and errors always keep
    the full prefix, whichever logger they come from.

    Two ways a line goes bare:
      - the logger is in BARE (vcm, wake - the main flow itself), or
      - the record carries extra={"vcm_bare": True} (command feedback emitted
        by other modules: music playback/volume, timer chime - these are
        part of the conversation, not diagnostics)."""
    BARE = frozenset(("vcm", "wake"))
    BARE_FLAG = "vcm_bare"

    def format(self, record):
        if record.levelno == logging.INFO and (
                record.name in self.BARE
                or getattr(record, self.BARE_FLAG, False)):
            return "%s %s" % (self.formatTime(record, self.datefmt),
                              record.getMessage())
        return super().format(record)


_root = logging.getLogger()
_root.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(_VCMFormatter("%(asctime)s %(name)s %(levelname)s %(message)s",
                              datefmt="%H:%M:%S"))
_root.addHandler(_h)
log = logging.getLogger("vcm")


# --- vcm-benchmark log (class holdout scoring) ------------------------------
# The class benchmark (github.com/airimonda/vcm-benchmark) scores the Pi by
# reading one JSON line per recognized command from a log file, ideally the
# newest .log in ~/vcm_benchmark/. Format (all the parser needs):
#   {"intent": <19 intents or OUT_OF_SCOPE>, "slot": <free text>,
#    "infer_ms": <feature+model ms>, "audio_ms": <speech window ms>}
# Joint fine labels (TIMER_10s, ALARM_9_00PM, ...) are split by the benchmark
# into intent + slot automatically; OUT_OF_SCOPE lines are read as rejections.
BENCH_DIR = os.path.expanduser("~/vcm_benchmark")


def _bench_log(intent, infer_ms, audio_ms):
    try:
        os.makedirs(BENCH_DIR, exist_ok=True)
        path = os.path.join(BENCH_DIR, time.strftime("%Y%m%d_%H%M%S") + ".log")
        with open(path, "a") as f:
            f.write(json.dumps({"intent": intent,
                                "slot": "",
                                "infer_ms": round(float(infer_ms), 2),
                                "audio_ms": round(float(audio_ms), 1)}) + "\n")
    except Exception as e:
        log.debug("bench log failed: %s", e)


def load_config():
    try:
        with open(CONFIG) as f:
            return json.load(f)
    except Exception as e:
        log.warning("config.json unreadable (%s) - defaults", e)
    return {}


def _trim_to_speech(audio, thr=0.012):
    """Trim raw command audio to the last utterance's onset.

    Splits the window into 30 ms frames, marks frames with RMS >= thr as
    voiced, groups maximal voiced runs (>= SPEECH_MIN_RUN frames, so short
    clicks/beep tails don't count), merges runs separated by < 0.5 s (mid-
    phrase pauses), and returns audio from the start of the utterance that
    contains the LAST qualifying run. Utterances shorter than 0.3 s (e.g. the
    wake-word tail spilling past the cue) are ignored. Returns (None, 0.0)
    when no speech is found.
    """
    frame = int(mic.SAMPLE_RATE * FRAME_MS / 1000)      # 360 samples
    n = 0 if audio is None else audio.size // frame
    if n == 0:
        return None, 0.0
    rms = np.sqrt(np.mean(np.square(
        audio[:n * frame].reshape(n, frame).astype(np.float64)), axis=1))
    voiced = rms >= thr
    runs = []                                           # maximal voiced runs
    i = 0
    while i < n:
        if voiced[i]:
            j = i
            while j + 1 < n and voiced[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    runs = [r for r in runs if r[1] - r[0] + 1 >= SPEECH_MIN_RUN]
    if not runs:
        return None, 0.0
    merge = max(1, int(0.5 / (FRAME_MS / 1000.0)))      # ~0.5 s pause tolerance
    utts = []                                           # [start, end] frames
    for a, b in runs:
        if utts and a - utts[-1][1] <= merge:
            utts[-1][1] = b
        else:
            utts.append([a, b])
    a, b = utts[-1]                                     # last utterance
    if (b - a + 1) * FRAME_MS < 300:                    # < 0.3 s -> not speech
        return None, 0.0
    return audio[a * frame:], (b - a + 1) * FRAME_MS / 1000.0


class App:
    def __init__(self, cfg, dry_run=False, perf_enabled=True):
        self.cfg = cfg
        self.dry_run = dry_run
        self.loop = asyncio.new_event_loop()

        # models + actors
        self.clf = features.VCMClassifier(ONNX, META)
        self.leds = ReSpeakerLEDs(
            enabled=cfg.get("leds_enabled", True),
            num_leds=cfg.get("led_num", 3),
            blink_hz=cfg.get("led_blink_hz", 2.5))
        self.music = MusicPlayer(start_volume=cfg.get("music_start_volume", 60),
                                 audio_device=cfg.get("music_audio_device"))
        self.timer = TimerPlayer()
        # Cue/ack beeps only - spoken commands answer with pre-recorded MP3s
        # (tts.py), music answers with its real effect. (Regression guard:
        # this line was accidentally commented out in the MP3 rewrite, which
        # crashed every wake with AttributeError: no attribute 'audio'.)
        self.audio = TonePlayer(enabled=cfg.get("beep", True))
        # Spoken responses - pre-generated MP3s in audio/responses/ (the
        # "Jarvis" voice, synthesized offline by rpi_inf/gen_responses.py).
        # No piper at runtime: no model load, no synth latency.
        self.tts = TTS(
            enabled=cfg.get("tts_enabled", True),
            volume=float(cfg.get("tts_volume", 1.0)))

        # v3: no Tapo bulb, no Flask UI. The ReSpeaker LEDs ARE the lights.
        self.ctx = {"music": self.music, "timer": self.timer,
                    "leds": self.leds, "cfg": cfg, "loop": self.loop,
                    "tts": self.tts, "dry_run": bool(dry_run)}
        self.mic_dev = None
        self._stop = False
        # Self-deaf guard: deadline until which live frames are dropped
        # because the assistant's own spoken response is in the mic.
        self._selfdeaf_until = 0.0
        # Performance recorder (perf_results/session_*.json + end-of-program
        # report). Disable with --no-perf.
        self.perf_enabled = bool(perf_enabled)
        if self.perf_enabled:
            perf.session_start(cfg, dry_run=dry_run)

    # ------------------------------------------------------------------ setup
    def start(self):
        # The asyncio loop runs in its own background thread (kept for the
        # dispatch contract; the v3 actions are all synchronous).
        self._loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True, name="asyncio-main")
        self._loop_thread.start()
        self.mic_dev = mic.select_device(self.cfg.get("mic_prefer",
                                                      ["respeaker", "seeed"]))
        log.info("VCM ready. Wake Jarvis up with \"Hello Jarvis\", "
                 "\"Hey Jarvis\", \"Jarvis\", \"Okay Jarvis\", "
                 "\"Jarvis, please\", or \"Jarvis, wake up\", then say your "
                 "command.")

    # ------------------------------------------------------------------ loop
    def run_live(self):
        gate = wakegate.make_gate(self.cfg, WAKE_DIR)
        self.leds.off()
        # The ReSpeaker's capture device is exclusive: opening a second stream
        # (mic.capture_seconds) while the wake stream is open fails with
        # -9985 "Device unavailable". So the command window is collected from
        # the SAME stream that feeds the wake gate.
        #
        # Siri-style flow: on wake -> spoken "Yes?" cue (no LED blink - the
        # LEDs are the light command's device), THEN capture. The cue echo
        # is dropped by drain detection (see _cue), not a blind frame count.
        # The window closes on its own (endpointing): CMD_END_SILENCE of quiet
        # after the last voiced frame, capped at CMD_MAX_S; if nothing voiced
        # is heard within CMD_NO_SPEECH_S of the cue, it closes too.
        max_need = int(round(self.cfg.get("cmd_max_seconds", CMD_MAX_S)
                             * mic.SAMPLE_RATE))
        end_sil = int(self.cfg.get("cmd_end_silence", CMD_END_SILENCE)
                      / (FRAME_MS / 1000.0))
        no_speech = int(self.cfg.get("cmd_no_speech_timeout", CMD_NO_SPEECH_S)
                        / (FRAME_MS / 1000.0))
        speech_thr = float(self.cfg.get("wake_rms_threshold", 0.012))
        win, voiced, quiet = [], 0, 0
        collecting = False
        draining = False          # dropping the buffered cue echo
        tail_left = 0             # live frames still to drop (reverb tail)
        last_t = 0.0
        try:
            for frame in mic.stream_frames(device=self.mic_dev):
                if self._stop:
                    break
                now = time.perf_counter()
                # Drop buffered stale audio (the cue echo, or the ack beep +
                # response start that piled up while _handle_command slept)
                # until frames arrive at real-time pace again - a gap between
                # frames means the mic buffer ran dry. Runs in BOTH branches:
                # after a command the backlog must not reach the wake gate
                # (self-wake fix).
                if draining:
                    if now - last_t >= DRAIN_GAP_S:
                        draining = False   # buffer drained - live audio follows
                    else:
                        last_t = now
                        continue           # stale audio still in the mic buffer
                last_t = now
                if collecting:
                    if tail_left > 0:      # drop the reverb tail, then listen
                        tail_left -= 1
                        continue
                    win.append(frame)
                    if float(np.sqrt(np.mean(np.square(frame)))) >= speech_thr:
                        voiced += 1
                        quiet = 0
                    else:
                        quiet += 1
                    total = sum(f.size for f in win)
                    timed_out = (voiced == 0 and len(win) >= no_speech)
                    finished = ((voiced > 0 and quiet >= end_sil)
                                or total >= max_need)
                    if timed_out or finished:
                        audio = (np.concatenate(win) if win
                                 else np.zeros(0, dtype=np.float32))
                        win, collecting, voiced, quiet = [], False, 0, 0
                        self._handle_command(audio=audio)
                        # _handle_command slept ~1.7 s (done hold + confirm
                        # delay): the mic buffer now holds the ack beep and
                        # the first seconds of the spoken response - drop
                        # them, then the self-deaf guard below covers the rest
                        draining = True
                    continue
                # Self-deaf: while the assistant's own spoken response
                # is playing (and for SELFDEAF_TAIL_S after it ends), drop
                # live frames so the wake CNN never scores them - this was
                # the "Jarvis wakes himself up on reminder replies" bug.
                # Music is NOT covered: the gate must stay live to hear
                # "stop the music" over a playing playlist.
                if self.tts.playing():
                    self._selfdeaf_until = now + SELFDEAF_TAIL_S
                    continue
                if now < self._selfdeaf_until:
                    continue
                if gate.process(frame):
                    # spoken "Yes?" cue - no LED blink: the LEDs are the
                    # light command's device, not a feedback channel
                    self._cue()              # blocking MP3 (or beep fallback)
                    last_t = time.perf_counter()
                    draining = True          # drop the cue as the mic heard it
                    tail_left = CUE_TAIL_FRAMES
                    log.info("Jarvis is listening. Speak now...")
                    perf.event("wake", engine=type(gate).__name__,
                              score=round(getattr(gate, "_confirmed_score",
                                       getattr(gate, "last_score", 0.0)), 4))
                    collecting = True        # window starts after the cue
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _handle_command(self, audio=None):
        # 1) capture the command window - no LED feedback: the LEDs are
        #    the light command's device , not a status channel
        if audio is None:
            log.info("VCM capturing command...")
            try:
                audio = mic.capture_seconds(CMD_SECONDS, device=self.mic_dev)
            except Exception as e:
                log.error("capture failed: %s", e)
                # flash ORANGE and restore - a failed capture must
                # not switch the lights off; only LIGHT_OFF does that.
                self.leds.done(color=C_ORANGE)
                return

        # 1b) trim to the last utterance - drops leading silence (and any
        # cue-beep tail) so the classifier sees speech from its onset, the
        # same distribution it was trained on. No speech at all -> ignore.
        t0 = time.perf_counter()
        audio, speech_s = _trim_to_speech(
            audio, thr=float(self.cfg.get("wake_rms_threshold", 0.012)))
        perf.add("trim", (time.perf_counter() - t0) * 1000.0,
                 speech_s=round(speech_s or 0, 2))
        if audio is None:
            log.info("VCM no speech captured - ignoring")
            self.leds.done(color=C_ORANGE)
            _bench_log("OUT_OF_SCOPE", 0.0, 0.0)
            perf.command(fine=None, conf=None, recognized=False, ok=False,
                         detail="no speech", speech_s=0.0)
            if self.cfg.get("beep", True):
                self._beep(ok=False)
            time.sleep(CONFIRM_DELAY)
            return
        log.info("Command window: %.2fs of speech", speech_s)

        # 2) classify
        t0 = time.perf_counter()
        pred = self.clf.predict_array(audio, mic.SAMPLE_RATE)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        perf.add("classify", infer_ms)
        fine = pred["fine_label"]
        conf = pred["fine_conf"]
        audio_ms = speech_s * 1000.0
        log.info("VCM predicted %s", fine)
        intent = pred.get("intent_label", fine)
        intent_conf = pred.get("intent_conf", 0.0)

        # 3) recognized?
        if conf < MIN_CONF:
            # the benchmark reads this log: a low-confidence rejection is the
            # runtime saying "out of scope" (counts as a reject, not a fire)
            _bench_log("OUT_OF_SCOPE", infer_ms, audio_ms)
            log.info("VCM not recognized - ignoring")
            log.debug("not recognized (conf %.3f < %.2f)", conf, MIN_CONF)
            self.leds.done(color=C_ORANGE)
            perf.command(fine=fine, conf=round(conf, 4), recognized=False,
                         ok=False, detail="low conf",
                         intent=intent, intent_conf=round(intent_conf, 4))
            if self.cfg.get("beep", True):
                self._beep(ok=False)
            time.sleep(CONFIRM_DELAY)
            return

        # 4) EXECUTING - steady LED while the action runs
        self.leds.executing()
        _bench_log(fine, infer_ms, audio_ms)
        t0 = time.time()
        res = actions.dispatch(fine, self.ctx)
        dt = time.time() - t0
        perf.add("action", dt * 1000.0, fine=fine, action=res.get("action"))
        log.info("VCM action: %s -> %s (%.2fs)",
                 res.get("action"), res.get("detail"), dt)
        perf.command(fine=fine, conf=round(conf, 4), recognized=True,
                     ok=bool(res.get("ok")), action=res.get("action"),
                     action_ms=round(dt * 1000.0, 1),
                     intent=intent, intent_conf=round(intent_conf, 4))

        # 5) DONE - brief steady, then the light state is restored.
        # Light commands (LIGHT_ON/OFF, COLOR_*, BRIGHTNESS_*) are skipped:
        # for them the LEDs ARE the device, so done() would immediately
        # undo what the action just set (this was the "lights on but LED
        # stays dark" bug - on() fired, then done() wiped it 450 ms later).
        # A rejected command (explicit OUT_OF_SCOPE or an unmapped label)
        # flashes ORANGE, not green - green is reserved for "command
        # recognized and executed". The ORANGE flash is
        # feedback only - if the lights were on they come back on after
        # the flash; only an explicit LIGHT_OFF leaves them dark.
        if res.get("action") not in ("light_on", "light_off",
                                     "color", "brightness"):
            self.leds.done(color=None if res.get("ok") else C_ORANGE)
        if self.cfg.get("beep", True):
            self._beep(ok=res.get("ok", False))
        time.sleep(CONFIRM_DELAY)

    def _cue(self):
        """Spoken 'Yes?' cue (pre-recorded MP3, the Jarvis voice) - plays
        BEFORE the command window opens, so the classifier never sees it.
        Playback blocks the loop; the frames that piled up in the mic
        buffer meanwhile are the cue as the mic heard it, and run_live
        drops them by watching for the buffer to drain (frames arriving
        at real-time pace again) instead of counting frames - a blind
        count overshoots whenever PortAudio overflows during the stall
        and would eat the first words of the command. Falls back to the
        1320 Hz beep when the MP3 is missing or TTS is disabled."""
        played = self.tts.speak(CUE_PHRASE, wait=True)
        if not played:
            self.audio.tone(freq=1320, ms=90)

    def _beep(self, ok=True):
        self.audio.tone(freq=880 if ok else 440, ms=120)

    def shutdown(self):
        log.info("VCM shutting down...")
        self.leds.close()
        self.music.close()
        try:
            self.tts.close()
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        # End-of-program performance report (deck: latency/acc/FAR fields).
        if self.perf_enabled:
            try:
                perf.session_end()
            except Exception as e:
                log.warning("perf report failed: %s", e)

    # ------------------------------------------------------------- one-shot
    def run_wav(self, path):
        t0 = time.perf_counter()
        pred = self.clf.predict_wav(path)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        import wave as _wave
        with _wave.open(path, "rb") as w:
            audio_ms = w.getnframes() / w.getframerate() * 1000.0
        fine = pred["fine_label"]; conf = pred["fine_conf"]
        _bench_log(fine, infer_ms, audio_ms)
        top5 = [(lbl, round(pr, 4)) for lbl, pr in
                sorted(zip(self.clf.fine_labels, pred["fine_probs"]),
                       key=lambda x: -x[1])[:5]]
        print(json.dumps({"fine": fine, "conf": round(conf, 4),
                          "intent": pred["intent_label"],
                          "intent_conf": round(pred["intent_conf"], 4),
                          "top5_fine": top5}, indent=2))
        if conf < MIN_CONF:
            print("-> not recognized"); return
        self._loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True, name="asyncio-main")
        self._loop_thread.start()
        res = actions.dispatch(fine, self.ctx)
        print("-> action:", json.dumps(res, indent=2))
        self.shutdown()


def main():
    ap = argparse.ArgumentParser(description="VCM voice assistant")
    ap.add_argument("--wav", help="classify one wav file and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="loop without playing music/timer audio (testing)")
    ap.add_argument("--no-tts", action="store_true",
                    help="disable spoken responses (beeps stay on)")
    ap.add_argument("--list-mics", action="store_true")
    ap.add_argument("--no-perf", action="store_true",
                    help="disable the performance recorder (perf_results/)")
    args = ap.parse_args()

    if args.list_mics:
        for i, name, sr in mic.list_inputs():
            print("#%d  %s  (%d Hz)" % (i, name, sr))
        return

    cfg = load_config()
    app = App(cfg, dry_run=args.dry_run, perf_enabled=not args.no_perf)
    if args.no_tts:
        app.tts.enabled = False
    if args.wav:
        app.run_wav(args.wav)
        return
    app.start()
    app.run_live()


if __name__ == "__main__":
    main()