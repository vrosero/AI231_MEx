"""Generate the pre-recorded response MP3s (run OFFLINE, not on the Pi).

v6: spoken confirmations are MP3 files in <rpi>/audio/responses/, played by
runtime/tts.py. This script synthesizes them ONCE with piper (the same
"Jarvis" voice, en_GB-alan-medium) and encodes them with ffmpeg/lame, so the
Pi never loads a TTS model at runtime.

The sentences are NOT retyped here - they are produced by importing
actions.py and calling dispatch() for every spoken fine label, so the MP3s
always match exactly what the dispatcher says, even if actions.py changes.
Re-run this script after editing any "spoken" sentence:

    pip install piper-tts                       # offline machine only
    python3 runtime/gen_responses.py            # regenerates audio/responses/

LIST_REMINDERS wording depends on (N, first item), so it is enumerated
through the real store: n = 0; n = 1 with each reminder text; n = 2..CAP
with each reminder text as the FIRST item (later order doesn't change the
sentence). CAP is actions.LIST_SPOKEN_CAP - actions.py caps the spoken
number there, which bounds the file set (the true count still goes to the
log via "detail").

The wake cue ("Yes?") is the one sentence that does not come from an
action - vcm.py speaks it as the listening prompt (it replaced
the cue beep + LED blink). It is added via EXTRA_SENTENCES below.

Usage:
    python3 runtime/gen_responses.py [--voice en_GB-alan-medium] [--dry-run]
"""

import argparse
import json
import logging
import os
import sys
import wave

log = logging.getLogger("gen_responses")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import actions                                   # noqa: E402  (runtime module)
from tts import path_for, slugify                # noqa: E402

RPI = os.path.join(HERE, "..")
RESPONSES_DIR = os.path.join(RPI, "audio", "responses")
CONFIG = os.path.join(RPI, "runtime", "config.json")

# The spoken fine labels (actions.py: everything with a "spoken" result)
SPOKEN_LABELS = (list(actions.ALARM_TIME) + list(actions.TEMP_C) +
                 list(actions.REMINDER_TEXT) + ["LIST_REMINDERS", "CALL",
                                                "MESSAGE"])

# vcm.py's wake cue - not an action, but spoken through the same MP3 path
EXTRA_SENTENCES = {"Yes?": ["WAKE_CUE"]}

# Silence written before every sentence (seconds). Covers the Pi's USB DAC
# power-up transient (~100 ms) with margin; also gives the listener a beat
# after the wake word before the cue starts. Keep < tts.py's mpv startup
# gap so total latency does not grow noticeably.
LEAD_IN_S = 0.25


def spoken_sentences():
    """{sentence: [fine labels that produce it]} by driving actions.dispatch.

    Uses a dry-run context (no hardware touched; the spoken commands are
    simulated anyway). LIST_REMINDERS is driven through the real _REMINDERS
    store to get its exact wording for every enumerated case.
    """
    ctx = {"dry_run": True, "tts": None}
    out = {}

    for lbl in SPOKEN_LABELS:
        if lbl == "LIST_REMINDERS":
            continue
        res = actions.dispatch(lbl, ctx)
        sent = res.get("spoken")
        if sent:
            out.setdefault(sent, []).append(lbl)

    # LIST_REMINDERS: enumerate every sentence the store can produce.
    cap = actions.LIST_SPOKEN_CAP
    firsts = list(actions.REMINDER_TEXT.values())

    def drive(n, first=None):
        actions._REMINDERS.clear()
        if first is not None:
            actions._REMINDERS.append(first)
            for i in range(1, n):
                actions._REMINDERS.append(firsts[i % len(firsts)])
        res = actions.dispatch("LIST_REMINDERS", ctx)
        sent = res.get("spoken")
        if sent:
            tag = "LIST_REMINDERS(n=%d" % n
            if first is not None:
                tag += ",first=%s" % first
            out.setdefault(sent, []).append(tag + ")")

    drive(0)
    for first in firsts:
        drive(1, first)
        for n in range(2, cap + 1):
            drive(n, first)
    actions._REMINDERS.clear()
    return out


def pick_encoder():
    """wav->mp3 encoder: ffmpeg, then lame, else None (keep wav).

    ffmpeg's output options (-codec:a, -b:a) must come AFTER the -i input;
    before it, ffmpeg parses them as input options and fails.
    """
    import shutil
    if shutil.which("ffmpeg"):
        return ["ffmpeg", "-y", "-loglevel", "error"]
    if shutil.which("lame"):
        return ["lame", "--quiet", "-b", "96"]
    return None


def encode(wav_path, mp3_path, enc):
    import subprocess
    if enc is None:
        os.replace(wav_path, mp3_path.replace(".mp3", ".wav"))
        return mp3_path.replace(".mp3", ".wav")
    if enc[0] == "ffmpeg":
        cmd = enc + ["-i", wav_path,
                     "-codec:a", "libmp3lame", "-b:a", "96k", mp3_path]
    else:                                        # lame
        cmd = enc + [wav_path, mp3_path]
    subprocess.run(cmd, check=True)
    os.remove(wav_path)
    return mp3_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--voice", default=None,
                    help="piper voice name (default: config.json tts_voice)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the sentences/files, synthesize nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    voice = args.voice
    if not voice:
        try:
            with open(CONFIG) as f:
                voice = json.load(f).get("tts_voice", "en_GB-alan-medium")
        except Exception:
            voice = "en_GB-alan-medium"

    sentences = spoken_sentences()
    for s, tags in EXTRA_SENTENCES.items():
        sentences.setdefault(s, tags)
    log.info("%d distinct sentences (%d from actions + %d extra, voice %s)",
             len(sentences), len(sentences) - len(EXTRA_SENTENCES),
             len(EXTRA_SENTENCES), voice)

    if args.dry_run:
        for s in sorted(sentences):
            print("%-58s %s" % (slugify(s) + ".mp3", s))
        return

    try:
        from piper.voice import PiperVoice
    except ImportError:
        sys.exit("piper-tts not installed: pip install piper-tts")
    vp = os.path.join(RPI, "tts", voice + ".onnx")
    if not os.path.isfile(vp):
        sys.exit("voice model not found: %s (setup.sh step 4 downloads it)"
                 % vp)
    log.info("loading piper voice %s ...", voice)
    v = PiperVoice.load(vp)
    enc = pick_encoder()
    if enc is None:
        log.warning("no ffmpeg/lame - files will be .wav (tts.py plays "
                    "those too, but they are ~4x larger)")
    os.makedirs(RESPONSES_DIR, exist_ok=True)

    import tempfile
    manifest = {}
    total = 0
    for sent in sorted(sentences):
        mp3 = path_for(sent)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name
        chunks = list(v.synthesize(sent))
        if not chunks:
            log.warning("piper produced no audio for %r", sent)
            os.remove(wav_path)
            continue
        sr = chunks[0].sample_rate
        # LEAD_IN_S of silence before every sentence: the Pi's USB DAC
        # (AB13X) wakes when the playback stream opens and its power-up
        # transient masks the first ~100 ms of audio. Without padding that
        # ate the onset of short words (the "y" of "Yes?" -> heard as "es").
        # The dead air is eaten by the DAC warm-up; the word itself survives.
        lead = b"\x00\x00" * int(sr * LEAD_IN_S)
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(lead + b"".join(c.audio_int16_array for c in chunks))
        final = encode(wav_path, mp3, enc)
        total += 1
        manifest[os.path.basename(final)] = {
            "sentence": sent, "labels": sentences[sent]}
        log.info("ok %-52s %s", os.path.basename(final), sent)

    with open(os.path.join(RESPONSES_DIR, "manifest.json"), "w") as f:
        json.dump({"voice": voice, "count": total, "files": manifest},
                  f, indent=2, ensure_ascii=False)
    log.info("done: %d response files in %s (+ manifest.json)", total,
             os.path.abspath(RESPONSES_DIR))


if __name__ == "__main__":
    main()