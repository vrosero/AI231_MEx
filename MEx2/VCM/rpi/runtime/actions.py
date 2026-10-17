"""Action dispatcher for VCM.

Maps a predicted FINE label (37 classes incl OUT_OF_SCOPE) to a concrete
action. This is the bridge between the VCM and the world. v3 hardware is
MINIMAL: the only devices are the ReSpeaker 2-Mics HAT (mic + 3 APA102 LEDs)
and a USB speaker.

  LIGHT_ON / LIGHT_OFF              -> ReSpeaker LEDs on/off   (leds.py)
  COLOR_RED/BLUE/GREEN              -> ReSpeaker LEDs colour    (leds.py)
  BRIGHTNESS_20/60/100              -> ReSpeaker LED brightness (leds.py)
  PLAY_MUSIC / PAUSE / STOP / NEXT /
  VOLUME_UP / VOLUME_DOWN           -> mp3 playlist             (music.py)
  TIMER_10s / TIMER_30s / TIMER_1m  -> per-slot chime MP3       (timer.py)
  WEATHER                           -> Open-Meteo (keyless, logged)     (weather.py)
  TIME                              -> RPi system clock (logged)
  ALARM_* / TEMPERATURE_* / CREATE_REMINDER_* / LIST_REMINDERS /
  CALL / MESSAGE                    -> simulated, spoken only (pre-recorded MP3)

Every command except music, the timer chime, TIME and WEATHER answers with a
spoken sentence - a PRE-RECORDED MP3 in ../audio/responses/ (the "Jarvis"
voice, synthesized offline by rpi_inf/gen_responses.py; runtime/tts.py just
plays the file, no piper at runtime). TIME and WEATHER do NOT speak (user
request 2026-10-14): their answer goes to the log
(detail) only. ALARM, TEMPERATURE, CREATE_REMINDER, LIST_REMINDERS, CALL and
MESSAGE speak a simulated confirmation - there is no dashboard anymore (v3
removed the web UI). Reminders are kept in a small in-memory list so
LIST_REMINDERS has something to read. Lights, music, the timer chime, TIME
and WEATHER act for real and stay silent.

dispatch(fine_label, ctx) returns a dict {ok, action, detail, spoken?}
describing what happened (logged + perf). ctx is a dict or object exposing:
music, timer, leds, cfg, loop, tts.
"""
import datetime
import logging

log = logging.getLogger("actions")

# LED colours (R, G, B) for the ReSpeaker's 3 APA102 LEDs
LED_COLORS = {
    "COLOR_RED":   (200, 30, 30),
    "COLOR_BLUE":  (30, 60, 200),
    "COLOR_GREEN": (30, 180, 60),
    "COLOR_YELLOW": (200, 180, 30),
    "COLOR_WHITE": (220, 220, 220),
}

# LED brightness levels (0-31 global brightness scale of the APA102)
LED_BRIGHTNESS = {"BRIGHTNESS_20": 6, "BRIGHTNESS_60": 19, "BRIGHTNESS_100": 31}

ALARM_TIME = {"ALARM_6_00AM": "06:00", "ALARM_8_00AM": "08:00",
              "ALARM_9_00PM": "21:00"}
TEMP_C = {"TEMPERATURE_18": 18, "TEMPERATURE_22": 22, "TEMPERATURE_26": 26}
REMINDER_TEXT = {
    "CREATE_REMINDER_DRINK_WATER": "drink water",
    "CREATE_REMINDER_EXERCISE": "exercise",
    "CREATE_REMINDER_STUDY": "study",
}

# in-memory reminder store (v3: no dashboard, no reminders.json)
_REMINDERS = []

# v6 (pre-recorded MP3 responses): LIST_REMINDERS can hold any number of
# reminders, but the response MP3s are pre-generated files - the spoken
# number is capped at LIST_SPOKEN_CAP (the MP3 set is enumerated up to it
# by gen_responses.py). The TRUE count always goes to the log via "detail".
LIST_SPOKEN_CAP = 9


class Ctx:
    """Uniform accessor over a dict-or-object context."""
    def __init__(self, ctx):
        self._d = ctx
        self._is_dict = isinstance(ctx, dict)

    def __getattr__(self, name):
        if self._is_dict:
            return self._d.get(name)
        return getattr(self._d, name)


def dispatch(fine_label, ctx):
    """fine_label: one of the 37 fine labels. Returns {ok, action, detail}."""
    res = _dispatch_inner(fine_label, ctx)
    # commands that return "spoken" answer through the pre-recorded response
    # MP3s (runtime/tts.py). Lights, music and the timer chime return no
    # "spoken" - their real effect IS the answer.
    try:
        c = Ctx(ctx)
        tts = c.tts
        if res.get("spoken") and tts is not None:
            tts.say(res["spoken"])
    except Exception:
        pass  # feedback must never break the action
    return res


def _dispatch_inner(fine_label, ctx):
    """fine_label: one of the 37 fine labels. Returns {ok, action, detail}."""
    c = Ctx(ctx)
    music = c.music
    timer = c.timer
    leds = c.leds

    # ---------------- LIGHTS = the ReSpeaker's own LEDs (v3) ---------------
    if fine_label == "LIGHT_ON":
        if leds:
            leds.on()
        return {"ok": True, "action": "light_on", "detail": "LEDs ON"}
    if fine_label == "LIGHT_OFF":
        if leds:
            leds.off()
        return {"ok": True, "action": "light_off", "detail": "LEDs OFF"}
    if fine_label in LED_BRIGHTNESS:
        b = LED_BRIGHTNESS[fine_label]
        if leds:
            leds.brightness(b)
        return {"ok": True, "action": "brightness",
                "detail": "LED brightness %d%%" % round(100 * b / 31)}
    if fine_label in LED_COLORS:
        rgb = LED_COLORS[fine_label]
        if leds:
            leds.color(rgb)
        name = fine_label.replace("COLOR_", "").lower()
        return {"ok": True, "action": "color", "detail": "LED color %s" % name}

    # ---------------- MUSIC (mp3) ----------------------------------------
    if fine_label == "PLAY_MUSIC":
        if c.dry_run:
            return {"ok": True, "action": "play_music", "detail": "playing"}
        ok = music.play()
        return {"ok": ok, "action": "play_music",
                "detail": "playing" if ok else "no tracks / mpv not ready"}
    if fine_label == "PAUSE":
        if c.dry_run:
            return {"ok": True, "action": "pause", "detail": "paused"}
        ok = music.pause()
        return {"ok": ok, "action": "pause", "detail": "paused"}
    if fine_label == "STOP":
        if c.dry_run:
            return {"ok": True, "action": "stop", "detail": "stopped"}
        ok = music.stop()
        return {"ok": ok, "action": "stop", "detail": "stopped"}
    if fine_label == "NEXT":
        if c.dry_run:
            return {"ok": True, "action": "next", "detail": "next track"}
        ok = music.next()
        return {"ok": ok, "action": "next",
                "detail": "track %d" % (music.index + 1) if ok else "no tracks"}
    if fine_label == "VOLUME_UP":
        if not c.dry_run:
            music.volume_up()
        return {"ok": True, "action": "volume_up",
                "detail": "volume %d%%" % music.volume}
    if fine_label == "VOLUME_DOWN":
        if not c.dry_run:
            music.volume_down()
        return {"ok": True, "action": "volume_down",
                "detail": "volume %d%%" % music.volume}

    # ---------------- TIMER: play the per-slot chime MP3 (v3) -------------
    if fine_label in ("TIMER_10s", "TIMER_30s", "TIMER_1m"):
        slot = {"TIMER_10s": "10s", "TIMER_30s": "30s", "TIMER_1m": "1m"}[fine_label]
        if c.dry_run:
            return {"ok": True, "action": "timer", "detail": slot}
        ok, path = timer.play(fine_label) if timer else (False, None)
        return {"ok": ok, "action": "timer",
                "detail": (slot if path else "chime missing")}

    # ---------------- WEATHER (real conditions, logged, not spoken) -------
    if fine_label == "WEATHER":
        import weather
        w = weather.get_weather(c.cfg)
        if w.get("ok"):
            detail = "%s: %.0f°C, %s, humidity %s%%" % (
                w["city"], w["temp_c"], w["condition"], w["humidity"])
        else:
            detail = w.get("error", "weather unavailable")
        # no "spoken": the user asked for no TTS on weather - the answer
        # goes to the log (detail) only
        return {"ok": w.get("ok", False), "action": "weather",
                "detail": detail}

    # ---------------- TIME (real clock, logged, not spoken) ---------------
    if fine_label == "TIME":
        now = datetime.datetime.now()
        now_s = now.strftime("%H:%M:%S")
        # no "spoken": the user asked for no TTS on time - the answer
        # goes to the log (detail) only
        return {"ok": True, "action": "time", "detail": "it's %s" % now_s}

    # ---------------- ALARM / TEMPERATURE (simulated, spoken) -------------
    if fine_label in ALARM_TIME:
        hhmm = ALARM_TIME[fine_label]
        h, m = map(int, hhmm.split(":"))
        h12 = h % 12 or 12
        ampm = "AM" if h < 12 else "PM"
        # spoken: drop ":00" on the hour ("6 AM", not "6:00 AM"); keep
        # minutes for off-the-hour times ("6:30 AM")
        if m:
            spoken = "Alarm set for %d:%02d %s." % (h12, m, ampm)
        else:
            spoken = "Alarm set for %d %s." % (h12, ampm)
        return {"ok": True, "action": "alarm",
                "detail": "alarm %s" % hhmm,
                "spoken": spoken}
    if fine_label in TEMP_C:
        deg = TEMP_C[fine_label]
        return {"ok": True, "action": "temperature",
                "detail": "%.0f°C" % deg,
                "spoken": "Setting the temperature to %d degrees." % deg}

    # ------------- REMINDERS (store) / CALL / MESSAGE (simulated) ---------
    if fine_label in REMINDER_TEXT:
        text = REMINDER_TEXT[fine_label]
        _REMINDERS.append(text)
        return {"ok": True, "action": "create_reminder",
                "detail": "reminder: %s" % text,
                "spoken": "Reminder created: %s." % text}
    if fine_label == "LIST_REMINDERS":
        n = len(_REMINDERS)
        n_spoken = min(n, LIST_SPOKEN_CAP)
        if n_spoken == 0:
            spoken = "You have no reminders."
        elif n_spoken == 1:
            spoken = "You have 1 reminder: %s." % _REMINDERS[0]
        else:
            spoken = ("You have %d reminders. The first is %s."
                      % (n_spoken, _REMINDERS[0]))
        return {"ok": True, "action": "list_reminders",
                "detail": "%d reminder(s)" % n, "spoken": spoken}
    if fine_label == "CALL":
        return {"ok": True, "action": "call",
                "detail": "call", "spoken": "Calling."}
    if fine_label == "MESSAGE":
        return {"ok": True, "action": "message",
                "detail": "message", "spoken": "Message sent."}

    # ---------------- OUT_OF_SCOPE (v2 model has an explicit OOS head) ----
    if fine_label == "OUT_OF_SCOPE":
        return {"ok": False, "action": "none", "detail": "out of scope"}

    # ---------------- UNKNOWN --------------------------------------------
    return {"ok": False, "action": "none", "detail": "unmapped label"}