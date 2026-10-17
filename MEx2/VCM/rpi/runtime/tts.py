"""Pre-recorded spoken responses for VCM - no real-time TTS.

v6: the spoken confirmations (ALARM, TEMPERATURE, CREATE_REMINDER,
LIST_REMINDERS, CALL, MESSAGE) are PRE-GENERATED MP3 files, not synthesized
at runtime. Live piper synthesis on the Pi cost a ~10 s voice load on first
use and ~1 s of synth per sentence; playing a ready-made MP3 starts in
~0.2 s and never loads a model.

Files live in <rpi>/audio/responses/<slug>.mp3, one per sentence, generated
OFFLINE by gen_responses.py (piper en_GB-alan-medium - the same "Jarvis"
voice - run on a laptop, not on the Pi). The slug is derived from the
sentence itself (slugify below), so actions.py keeps returning plain
"spoken" text and stays the single source of truth for WHAT is said.

If a sentence has no MP3 (actions.py edited without regenerating), the
command still runs - it just answers with the ack beep only (audio.py
TonePlayer) and logs a warning naming gen_responses.py.

Playback: one-shot subprocess, backend chain mpv > ffplay > afplay (macOS).
This deliberately does NOT use the music player's mpv IPC socket - a spoken
confirmation must never take over or rewire the playlist; with pipewire the
short response simply mixes over whatever is playing (the old piper path
did the same through sounddevice).
"""

import hashlib
import logging
import os
import queue
import re
import shutil
import subprocess
import threading

log = logging.getLogger("tts")

HERE = os.path.dirname(os.path.abspath(__file__))
RESPONSES_DIR = os.path.join(HERE, "..", "audio", "responses")


def slugify(text):
    """Sentence -> deterministic, collision-free filename stem.

    'Alarm set for 6:00 AM.' -> 'alarm-set-for-6-00-am-<md5[:6]>'. The
    truncated slug keeps names readable; the md5 suffix guarantees two
    different sentences never map to one file (and the same sentence always
    maps to the same file, so gen_responses.py and tts.py agree).
    """
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:6]
    return "%s-%s" % (base, digest)


def path_for(text):
    """MP3 path that holds the spoken form of `text`."""
    return os.path.join(RESPONSES_DIR, slugify(text) + ".mp3")


class TTS:
    """Plays pre-generated response MP3s: tts.say(text) queues the file and
    a worker thread plays it.

    Same public contract as the old piper TTS, so actions.py and vcm.py
    are unchanged: say() is non-blocking, one utterance may sit queued (a
    newer say() replaces a still-queued older one), utterances never
    overlap, close() stops the worker. speak(text, wait=True) plays
    synchronously.
    """

    def __init__(self, enabled=True, volume=1.0):
        self.enabled = enabled
        self.volume = max(0.0, min(1.0, volume))
        self._q = queue.Queue(maxsize=1)
        self._proc = None
        self._warned = set()
        self._worker = (threading.Thread(target=self._loop, daemon=True,
                                         name="tts")
                        if self.enabled else None)
        if self._worker:
            self._worker.start()

    # ---------------------------------------------------------------- play
    def _pick_backend(self):
        for name in ("mpv", "ffplay", "afplay"):
            if shutil.which(name):
                return name
        return None

    def _play(self, path):
        """Blocking playback of one MP3. Returns True when it played."""
        backend = self._pick_backend()
        if not backend:
            log.warning("tts: no mpv/ffplay/afplay on PATH - cannot play %s",
                        os.path.basename(path))
            return False
        if backend == "mpv":
            cmd = ["mpv", "--really-quiet", "--no-video",
                   "--volume=%d" % round(self.volume * 100), path]
        elif backend == "ffplay":
            # ffplay has no simple volume flag - plays at file volume
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                   path]
        else:                                   # afplay (macOS)
            cmd = ["afplay", path]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._proc.wait()
            return self._proc.returncode == 0
        except Exception as e:
            log.warning("tts: playback failed (%s)", e)
            return False

    # --------------------------------------------------------------- worker
    def playing(self):
        """True while a response MP3 subprocess is still running.

        Read from the main loop (the worker sets _proc); used by vcm.py's
        self-deaf guard so the wake CNN never scores the assistant's own
        spoken responses coming back through the mic.
        """
        return self._proc is not None and self._proc.poll() is None

    def _loop(self):
        while True:
            path = self._q.get()
            if path is None:
                return
            try:
                self._play(path)
            except Exception as e:
                log.warning("tts: speak failed (%s)", e)

    # ---------------------------------------------------------------- public
    def say(self, text, replace=True):
        """Queue the pre-recorded MP3 for `text` (non-blocking).

        replace=True drops a still-queued utterance in favour of this one,
        exactly like the old piper queue. Returns False (beep-only fallback)
        when TTS is disabled or the sentence has no MP3 yet.
        """
        if not (self.enabled and text):
            return False
        path = path_for(text)
        if not os.path.isfile(path):
            if text not in self._warned:
                self._warned.add(text)
                log.warning("tts: no response MP3 for %r - run "
                            "rpi_inf/gen_responses.py (answering with the "
                            "ack beep only)", text)
            return False
        try:
            if replace:
                try:
                    self._q.get_nowait()       # drop the stale one
                except queue.Empty:
                    pass
            self._q.put_nowait(path)
            return True
        except Exception:
            return False

    def speak(self, text, wait=False):
        """say() + optional synchronous playback (used by --wav mode)."""
        if wait and self.enabled and text:
            path = path_for(text)
            if not os.path.isfile(path):
                log.warning("tts: no response MP3 for %r - run "
                            "rpi_inf/gen_responses.py", text)
                return False
            return self._play(path)
        return self.say(text)

    def close(self):
        if self._worker:
            self._q.put(None)