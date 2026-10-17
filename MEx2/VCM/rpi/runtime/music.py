"""MP3 playlist player for VCM music commands.

Backed by `mpv` (installed by setup.sh). mpv is controlled over a JSON
IPC socket so play/pause/stop/next/volume are instant and non-blocking - the
voice loop never waits on the audio thread.

Commands handled here:
    PLAY_MUSIC  - start (or resume) the playlist from the current track
    PAUSE       - pause
    STOP        - stop (reset to start of playlist)
    NEXT        - advance to the next track (wraps)
    VOLUME_UP   - volume += 10%
    VOLUME_DOWN - volume -= 10%

The 5 tracks live in ../music (music_1.mp3 .. music_5.mp3).
"""
import json
import logging
import os
import subprocess
import time

log = logging.getLogger("music")

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC_DIR = os.path.join(HERE, "..", "music")
SOCK = os.path.join(HERE, "..", "mpv.sock")
VOL_STEP = 10  # percent per VOLUME_UP / VOLUME_DOWN


def _tracks():
    d = os.path.abspath(MUSIC_DIR)
    if not os.path.isdir(d):
        return []
    files = [f for f in os.listdir(d) if f.lower().endswith(".mp3")]
    files.sort(key=lambda s: int("".join(c for c in os.path.splitext(s)[0] if c.isdigit()) or 0))
    return [os.path.join(d, f) for f in files]


class MusicPlayer:
    def __init__(self, start_volume=60, audio_device=None):
        self.tracks = _tracks()
        self.index = 0
        self.volume = int(start_volume)
        self.playing = False
        self.paused = False
        self._ready = False
        self.audio_device = audio_device  # mpv --audio-device (None = auto)
        log.info("music: %d tracks in the music folder (audio_device=%s)",
                 len(self.tracks), audio_device or "auto")

    # ---- mpv lifecycle ----------------------------------------------------
    def _ensure(self):
        """Spawn mpv (if needed) and wait until its IPC socket answers.
        Falls back to ffplay (no IPC: fire-and-forget playback) if mpv is
        absent, so PLAY_MUSIC still works on systems without mpv."""
        if self._ready:
            return True
        if not self.tracks:
            log.warning("music: no mp3 files found in %s", MUSIC_DIR)
            return False
        if not os.path.exists("/usr/bin/mpv") and not _which("mpv"):
            if _which("ffplay"):
                return self._ffplay_play()
            log.error("music: mpv not installed (run setup.sh)")
            return False
        cmd = ["mpv", "--no-video", "--really-quiet", "--idle=yes",
               f"--input-ipc-server={SOCK}",
               "--volume=%d" % self.volume]
        if self.audio_device:
            cmd.insert(1, "--audio-device=%s" % self.audio_device)
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
        except Exception as e:
            log.error("music: failed to start mpv: %s", e)
            return False
        for _ in range(40):  # up to ~2s for the socket to come up
            if self._cmd({"command": ["get_property", "idle"]}) is not None:
                self._ready = True
                return True
            time.sleep(0.05)
        log.warning("music: mpv socket not ready yet")
        return False

    def _cmd(self, obj):
        """Send one JSON command to mpv over the unix socket; return reply dict
        or None on failure."""
        import socket
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(SOCK)
            s.sendall((json.dumps(obj) + "\n").encode())
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
            s.close()
            if buf:
                return json.loads(buf.split(b"\n")[0])
        except Exception:
            return None
        return None

    def _get(self, prop):
        r = self._cmd({"command": ["get_property", prop]})
        return r.get("data") if r and r.get("error") == "success" else None

    # ---- ffplay fallback (no IPC, one process per play/next) ---------------
    def _ffplay_play(self):
        """Fire-and-forget playback via ffplay. Supports play/stop/next with
        full fidelity; pause/volume degrade to no-ops (logged once)."""
        self._kill_ffplay()
        try:
            subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                 self.tracks[self.index % len(self.tracks)]],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            self._ffplay = True
            self.playing = True
            self.paused = False
            log.info("music[ffplay]: playing track %d/%d", self.index + 1,
                     len(self.tracks), extra={"vcm_bare": True})
            return True
        except Exception as e:
            log.error("music: ffplay failed: %s", e)
            return False

    def _kill_ffplay(self):
        if getattr(self, "_ffplay", False):
            try:
                subprocess.run(["pkill", "-f", "ffplay -nodisp"],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            except Exception:
                pass
            self._ffplay = False

    # ---- public API -------------------------------------------------------
    def _load_current(self):
        if not self.tracks:
            return
        path = self.tracks[self.index % len(self.tracks)]
        self._cmd({"command": ["loadfile", path, "replace"]})
        self.playing = True
        self.paused = False
        log.info("music: playing track %d/%d %s", self.index + 1, len(self.tracks),
                 os.path.basename(path), extra={"vcm_bare": True})

    def play(self):
        if not self._ensure():
            return False
        idle = self._get("idle-active")
        if idle is None:
            idle = True
        if idle:
            self._load_current()
        elif self.paused:
            self._cmd({"command": ["set_property", "pause", False]})
            self.paused = False
        return True

    def pause(self):
        if not self._ready:
            if getattr(self, "_ffplay", False):
                log.warning("music: pause not supported by ffplay fallback")
                return False
            return False
        self._cmd({"command": ["set_property", "pause", True]})
        self.paused = True
        return True

    def stop(self):
        if not self._ready:
            if getattr(self, "_ffplay", False):
                self._kill_ffplay()
                self.playing = False
                self.paused = False
                self.index = 0
                return True
            return False
        self._cmd({"command": ["stop"]})
        self.playing = False
        self.paused = False
        self.index = 0
        return True

    def next(self):
        if not self._ensure():
            return False
        self.index = (self.index + 1) % max(1, len(self.tracks))
        if getattr(self, "_ffplay", False):
            return self._ffplay_play()
        self._load_current()
        return True

    def volume_up(self):
        self._set_vol(self.volume + VOL_STEP)
        return True

    def volume_down(self):
        self._set_vol(self.volume - VOL_STEP)
        return True

    def _set_vol(self, v):
        v = max(0, min(100, int(v)))
        self.volume = v
        if self._ready:
            self._cmd({"command": ["set_property", "volume", v]})
        log.info("music: volume -> %d%%", v, extra={"vcm_bare": True})

    def state(self):
        return {
            "playing": self.playing,
            "paused": self.paused,
            "track": (os.path.basename(self.tracks[self.index % len(self.tracks)])
                      if self.tracks else None),
            "track_no": (self.index + 1 if self.tracks else 0),
            "num_tracks": len(self.tracks),
            "volume": self.volume,
        }

    def close(self):
        if self._ready:
            self._cmd({"command": ["quit"]})
            self._ready = False
        self._kill_ffplay()


def _which(name):
    import shutil
    return shutil.which(name)
