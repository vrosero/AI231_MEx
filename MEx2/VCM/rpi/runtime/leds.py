"""ReSpeaker 2-Mics Pi HAT - 3 APA102 RGB status LEDs.

The HAT's three RGB LEDs are driven over SPI by the `apa102` library (the
standard Seeed wiring). Each LED is addressed 0..2.

State model (per the design brief):
    LISTENING  - fast blink (all 3 LEDs, soft cyan)  -> "I heard 'Hello Jarvis',
                 I'm listening for your command"
    EXECUTING  - steady ON (all 3 LEDs, green)       -> "command recognized,
                 running it now"
    DONE       - brief steady (green) then off       -> "finished"
    REJECTED   - brief steady (ORANGE) then off      -> "out of scope / not
                 recognized" (was green, indistinguishable from success)
    OFF        - all off                             -> "idle"

NOTE: vcm.py no longer uses listening() - the wake cue is a spoken
"Yes?" MP3 (tts.py) and the LEDs are reserved for the light commands.
listening()/executing()/done() remain here for bench tools.

v3: the LEDs are ALSO the "lights" of the smart device - LIGHT_ON/OFF,
COLOR_* and BRIGHTNESS_* commands set them directly (on()/color()/
brightness()), so no separate smart bulb is needed.

Only an explicit LIGHT_OFF (or shutdown) leaves the LEDs dark.
executing()/done() feedback flashes (green/orange) restore the previous
light state afterwards via _restore, so a missed command can never
switch the lights off.

If the HAT isn't wired up / `apa102` isn't installed / running as non-root,
every method degrades to a no-op so the voice pipeline still runs. Set
"leds_enabled": false in config.json to force-disable.
"""
import logging
import threading
import time

log = logging.getLogger("leds")

# (R, G, B) 0-255
C_CYAN   = (0, 120, 160)
C_GREEN  = (0, 200, 60)
C_ORANGE = (255, 130, 0)    # not-recognized / out-of-scope flash
C_WHITE  = (220, 220, 220)
C_OFF    = (0, 0, 0)

NUM_LEDS = 3


class ReSpeakerLEDs:
    def __init__(self, enabled=True, num_leds=NUM_LEDS,
                 listen_color=C_CYAN, exec_color=C_GREEN,
                 blink_hz=2.5):
        self.enabled = bool(enabled)
        self.num_leds = num_leds
        self.listen_color = listen_color
        self.exec_color = exec_color
        self.blink_period = 1.0 / max(0.2, blink_hz)
        self._apa = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._state = "off"
        # the LEDs are the light device, so transient feedback
        # (EXECUTING green, REJECTED orange) must not leave them dark.
        # _restore remembers the light state to return to after a flash;
        # None means the lights were off, so feedback ends dark as before.
        self._restore = None
        if self.enabled:
            self._init_hardware()

    # ---- hardware ---------------------------------------------------------
    def _init_hardware(self):
        """Drive the HAT's 3 APA102s over SPI0 (BCM10=data, BCM11=clock).

        Cross-reference with mic_hat/interfaces/pixels.py (verified working on
        this Pi): it vendors tinue's APA102_Pi library, which defaults to
        hardware SPI on bus 0 - the pins the HAT's LEDs are wired to. The
        Pimoroni `apa102` 0.0.3 package instead defaults to GPIO bitbang on
        pins 14/15, which this HAT does not use, so constructing it with
        defaults writes to the wrong pins (silently). Preferred order:
          1. apa102_pi.driver.apa102.APA102 (tinue APA102_Pi 2.x, installed)
          2. apa102.core.APA102             (tinue APA102_Pi 0.0.x)
          3. apa102 (Pimoroni 0.0.3) forced onto SPI via gpio_data=10/gpio_clock=11
          4. apa102 (Pimoroni 0.0.3) GPIO bitbang 14/15 (last resort)
        """
        try:
            try:
                from apa102_pi.driver.apa102 import APA102  # tinue 2.x
                self._apa = APA102(num_led=self.num_leds, global_brightness=4)
                self._apa_seeed = True
            except ImportError:
                try:
                    from apa102.core import APA102  # tinue 0.0.x
                    self._apa = APA102(num_led=self.num_leds)
                    self._apa_seeed = True
                except ImportError:
                    import apa102 as _apa_mod
                    import inspect
                    sig = inspect.signature(_apa_mod.APA102.__init__)
                    if "num_led" in sig.parameters:
                        self._apa = _apa_mod.APA102(num_led=self.num_leds,
                                                    global_brightness=31)
                        self._apa_seeed = True
                    else:
                        # Pimoroni 0.0.3: force its SPI branch (BCM10/11);
                        # defaults would bitbang GPIO14/15 -> invisible.
                        self._apa = _apa_mod.APA102(count=self.num_leds,
                                                    gpio_data=10, gpio_clock=11,
                                                    brightness=1.0)
                        self._apa_seeed = True
                # Sanity: the object must actually expose a pixel API.
                if not ((hasattr(self._apa, "set_pixel") and hasattr(self._apa, "show"))
                        or (hasattr(self._apa, "set_rgb") and hasattr(self._apa, "write"))):
                    raise RuntimeError(
                        "APA102 object %r lacks set_pixel/show and set_rgb/write"
                        % type(self._apa).__name__)
            for i in range(self.num_leds):
                self._set_one(i, C_OFF)
            self._write()
            log.info("APA102 LEDs ready (%d) via %s", self.num_leds,
                     type(self._apa).__module__)
        except Exception as e:
            log.warning("APA102 unavailable (%s) - LEDs disabled, voice still runs", e)
            self._apa = None
            self.enabled = False

    def _set_one(self, i, rgb):
        if getattr(self, "_apa_seeed", False):
            self._apa.set_pixel(i, *rgb)
        else:
            self._apa.set_rgb(i, *rgb)

    def _write(self):
        if getattr(self, "_apa_seeed", False):
            self._apa.show()
        else:
            self._apa.write()

    def _set_all(self, rgb):
        if not self._apa:
            return
        with self._lock:
            try:
                for i in range(self.num_leds):
                    self._set_one(i, rgb)
                self._write()
            except Exception as e:
                log.debug("led write failed: %s", e)

    # ---- states -----------------------------------------------------------
    def _stop_blink(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.3)
        self._thread = None

    def _blink(self, rgb, period):
        self._stop_blink()
        self._stop.clear()

        def _run():
            on = False
            while not self._stop.is_set():
                on = not on
                self._set_all(rgb if on else C_OFF)
                time.sleep(period / 2.0)
            self._set_all(C_OFF)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def listening(self):
        """Fast blink - we heard the wake word, capturing the command."""
        self._state = "listening"
        self._blink(self.listen_color, self.blink_period)

    def executing(self):
        """Steady on - command recognized, action in progress."""
        self._state = "executing"
        self._stop_blink()
        self._set_all(self.exec_color)

    def done(self, hold_ms=450, color=None):
        """Brief steady, then restore the light state.

        color=None keeps the success green; pass C_ORANGE for the
        not-recognized / out-of-scope flash. The flash itself is
        feedback only: if the lights were on (LIGHT_ON/COLOR_*/
        BRIGHTNESS_*), they return to that state instead of going dark -
        only an explicit LIGHT_OFF (or shutdown) leaves them off.
        """
        self._stop_blink()
        self._set_all(color if color is not None else self.exec_color)
        time.sleep(hold_ms / 1000.0)
        if self._restore is not None:
            self._set_all(self._restore)
            self._state = "restored"
        else:
            self.off()

    def on(self):
        """LIGHT_ON - all 3 LEDs steady white (the 'lights' of this device)."""
        self._state = "on"
        self._stop_blink()
        self._restore = C_WHITE
        self._set_all(C_WHITE)

    def brightness(self, level):
        """LIGHT brightness 0-31 (APA102 global scale) - steady white."""
        self._state = "brightness"
        self._stop_blink()
        rgb = tuple(int(c * max(0.05, min(1.0, level / 31.0))) for c in C_WHITE)
        self._restore = rgb
        self._set_all(rgb)

    def color(self, rgb):
        """LIGHT colour - all 3 LEDs set to an (R, G, B) tuple."""
        self._state = "color"
        self._stop_blink()
        self._restore = tuple(rgb)
        self._set_all(rgb)

    def off(self):
        """All off - explicit LIGHT_OFF or power-off (the only paths
        that leave the lights dark; feedback flashes restore instead)."""
        self._state = "off"
        self._restore = None
        self._stop_blink()
        self._set_all(C_OFF)

    def close(self):
        self.off()
