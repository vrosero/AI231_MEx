"""Tone feedback for VCM - the cue beep ("yes?") and ack/error beeps.

v6: spoken responses are pre-recorded MP3s (runtime/tts.py plays them from
audio/responses/); lights and music answer with their real effect only. What
remains here are the short sine tones: the wake cue (1320 Hz, "yes?") and
the ack (880 Hz) / error (440 Hz) beep after each command.

Playback: sounddevice (venv, same lib the mic uses) with afplay as fallback,
so it works on the Pi (pipewire) and during dry-run.
"""
import logging
import os
import shutil
import subprocess
import time

import numpy as np

log = logging.getLogger("audio")


class TonePlayer:
    """Short sine tones through the same backend the TTS uses.

    tone() blocks for the tone duration (~90-120 ms) - by design: the cue
    beep plays BEFORE the command window opens, so the mic never hears it.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._backend = self._pick_backend()
        log.info("audio: tone backend=%s", self._backend or "none")

    def _pick_backend(self):
        try:
            import sounddevice as sd  # noqa: F401
            return "sounddevice"
        except Exception:
            pass
        if shutil.which("afplay"):
            return "afplay"
        return None

    def tone(self, freq=1320, ms=90, ok=True, wait=True):
        """Short sine tone (cue beep / ack / error beep)."""
        if not self.enabled:
            return False
        sr = 44100
        n = int(sr * ms / 1000)
        t = (12000 * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype(np.float32)
        try:
            if self._backend == "sounddevice":
                import sounddevice as sd
                sd.play(t, sr); sd.wait()
                return True
            if self._backend == "afplay":
                import soundfile as sf
                tmp = "/tmp/vcm_tone_%d.wav" % int(time.time() * 1000)
                sf.write(tmp, t, sr)
                subprocess.run(["afplay", tmp], check=False, timeout=10,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(tmp)
                return True
        except Exception as e:
            log.debug("audio: tone failed: %s", e)
        return False

    def close(self):
        pass