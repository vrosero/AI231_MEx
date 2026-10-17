"""Timer sound player for VCM — one MP3 per timer slot.

v3 change: a timer command no longer starts a dashboard countdown. Instead it
plays the PRE-RECORDED chime for that slot (the same way PLAY_MUSIC plays a
track), from audio/timer/:

    TIMER_10s -> TIMER_10s.mp3   (10-second timer chime)
    TIMER_30s -> TIMER_30s.mp3   (30-second timer chime)
    TIMER_1m  -> TIMER_1m.mp3    (1-minute timer chime)

Playback mirrors the music player's backends: mpv JSON-IPC when available,
ffplay fire-and-forget otherwise. A second timer command replaces the one
currently playing (like re-pressing a kitchen timer).
"""
import logging
import os
import subprocess

log = logging.getLogger("timer")

HERE = os.path.dirname(os.path.abspath(__file__))
TIMER_DIR = os.path.join(HERE, "..", "audio", "timer")
SOCK = os.path.join(HERE, "..", "mpv.sock")

# fine label -> mp3 file
TIMER_FILES = {
    "TIMER_10s": "TIMER_10s.mp3",
    "TIMER_30s": "TIMER_30s.mp3",
    "TIMER_1m": "TIMER_1m.mp3",
}


def _which(name):
    from shutil import which
    return which(name)


class TimerPlayer:
    """Plays the per-slot timer chime. Same backend chain as music.py."""

    def __init__(self):
        self._proc = None
        self._backend = None

    # ---- backends ---------------------------------------------------------
    def _play_mpv(self, path):
        """Replace the current mpv playlist with the timer file."""
        import json
        import socket
        import time
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(SOCK)
            s.sendall((json.dumps(
                {"command": ["loadfile", path, "replace"]}) + "\n").encode())
            s.close()
            return True
        except Exception as e:
            log.debug("timer: mpv IPC unavailable (%s)", e)
            return False

    def _play_ffplay(self, path):
        self._kill()
        try:
            self._proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            return True
        except Exception as e:
            log.error("timer: ffplay failed: %s", e)
            return False

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None

    # ---- public API -------------------------------------------------------
    def play(self, fine_label):
        """Play the chime for a TIMER_* fine label. Returns (ok, path)."""
        fn = TIMER_FILES.get(fine_label)
        if not fn:
            return False, None
        path = os.path.join(TIMER_DIR, fn)
        if not os.path.exists(path):
            log.warning("timer: missing %s", path)
            return False, path
        if self._play_mpv(path):
            self._backend = "mpv"
            return True, path
        if _which("ffplay"):
            ok = self._play_ffplay(path)
            if ok:
                self._backend = "ffplay"
            return ok, path
        log.error("timer: no mpv/ffplay available")
        return False, path

    def stop(self):
        """Stop a playing chime (STOP command)."""
        self._kill()
        try:
            import json
            import socket
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(SOCK)
            s.sendall((json.dumps({"command": ["stop"]}) + "\n").encode())
            s.close()
        except Exception:
            pass
        return True