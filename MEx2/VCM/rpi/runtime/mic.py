"""Microphone capture for mini-VCM on the Raspberry Pi.

Uses sounddevice (PortAudio). The mic is the ReSpeaker 2-Mics HAT (wired I2S
codec, tlv320aic3x) which shows up as an ALSA/PulseAudio input named
"seeed2micvoicec" / "bcm2835-i2s-tlv320aic3x". We auto-select it by name
substring (default "respeaker" / "seeed"), falling back to the system default
input.

NOTE: the Bluetooth speaker (XSP61) is OUTPUT-ONLY and is NOT a valid mic, so
we deliberately do not match "bluetooth" here.

Returns float32 mono at SAMPLE_RATE (12 kHz).

HARDWARE NOTE (verified on megumi-rpi): the ReSpeaker HAT's hw device only
accepts 16k/44.1k/48k — PortAudio rejects 12 kHz outright (-9997). So we
capture at CAP_RATE (16 kHz, the closest supported rate) and decimate to
SAMPLE_RATE (12 kHz) with a simple low-pass + linear resample. All pipeline
code below the mic sees the same 12 kHz frames as before.
"""
import logging
import numpy as np

log = logging.getLogger("mic")

SAMPLE_RATE = 12000
NAMES = ("respeaker", "seeed", "tlv320aic3x", "bcm2835-i2s", "pulse")


def list_inputs():
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append((i, d["name"], d["default_samplerate"]))
    return out


def select_device(prefer=()):
    """Pick the input device index. Prefer a name match, else default input.

    Substrings are tried in the order given (outer loop) BEFORE falling back
    to enumeration order (inner loop), so a specific name like "seeed" always
    beats a generic one like "usb" even when the USB device happens to
    enumerate first. (Bug fix 2026-10-17: after a reboot the AB13X USB
    headset adapter can grab a lower card index than the ReSpeaker HAT; the
    old devices-first loop then matched the generic "usb" fallback and VCM
    captured a dead headset jack instead of the HAT.)
    """
    import sounddevice as sd
    devs = sd.query_devices()
    prefer = [p.lower() for p in prefer]
    for p in prefer:                      # most specific name first
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                if p in d["name"].lower():
                    log.info("selected mic #%d: %s", i, d["name"])
                    return i
    di = sd.default.device[0]
    if isinstance(di, (list, tuple)):
        di = di[0]
    if di is not None and di >= 0:
        log.warning("no preferred mic found; using default input #%d: %s",
                    di, devs[di]["name"])
        return int(di)
    log.warning("no input device found")
    return None


def select_output(prefer=()):
    """Pick the OUTPUT device index for the confirmation beep. Prefer a name
    match (e.g. the XSP61 Bluetooth speaker), else the system default output.

    Same ordering rule as select_device: preferred substrings outer,
    devices inner, so a specific name beats a generic one regardless of
    enumeration order.
    """
    import sounddevice as sd
    devs = sd.query_devices()
    prefer = [p.lower() for p in prefer]
    for p in prefer:                  # most specific name first
        for i, d in enumerate(devs):
            if d.get("max_output_channels", 0) > 0:
                if p in d["name"].lower():
                    log.info("selected speaker #%d: %s", i, d["name"])
                    return i
    do = sd.default.device[1]
    if isinstance(do, (list, tuple)):
        do = do[0]
    if do is not None and do >= 0:
        log.warning("no preferred speaker found; using default output #%d: %s",
                    do, devs[do]["name"])
        return int(do)
    log.warning("no output device found")
    return None


def _decimate_to_12k(x, cap_rate):
    """16k (or 44.1k/48k) mono -> 12 kHz mono: low-pass then linear resample."""
    if cap_rate == SAMPLE_RATE:
        return x
    # simple moving-average low-pass sized to the rate ratio, then interp
    from numpy import convolve
    ratio = cap_rate / SAMPLE_RATE
    taps = max(1, int(round(ratio)))
    if taps > 1:
        x = convolve(x, np.ones(taps, dtype=np.float32) / taps, mode="same")
    n_out = int(round(x.size * SAMPLE_RATE / cap_rate))
    x_old = np.linspace(0.0, 1.0, num=x.size)
    x_new = np.linspace(0.0, 1.0, num=n_out)
    return np.interp(x_new, x_old, x).astype(np.float32)


def _pick_cap_rate(device):
    """Highest-priority sample rate the device actually supports, probed in
    order of preference (12k exact, then 16k, then 48k, then 44.1k)."""
    import sounddevice as sd
    for sr in (SAMPLE_RATE, 16000, 48000, 44100):
        try:
            sd.check_input_settings(device=device, samplerate=sr,
                                    channels=1, dtype="float32")
            return sr
        except Exception:
            continue
    return 16000  # last resort; will raise a clear error downstream


def capture_seconds(seconds, device=None, sr=SAMPLE_RATE):
    """Record `seconds` of mono float32 at SAMPLE_RATE (12 kHz). Returns
    np.ndarray [N]. Captures at a hardware-supported rate, then resamples."""
    import sounddevice as sd
    cap = _pick_cap_rate(device)
    n = int(seconds * cap)
    audio = sd.rec(n, samplerate=cap, channels=1, dtype="float32", device=device)
    sd.wait()
    x = audio.flatten().astype(np.float32)
    return _decimate_to_12k(x, cap)


def stream_frames(device=None, sr=SAMPLE_RATE, frame_ms=30, block_ms=100):
    """Yield mono float32 frames of `frame_ms` duration at SAMPLE_RATE (12 kHz)
    from the mic (blocking PortAudio stream). Used by the main loop for the
    wake gate. Captures at a hardware-supported rate, then resamples."""
    import sounddevice as sd
    cap = _pick_cap_rate(device)
    frame_cap = int(cap * frame_ms / 1000)
    block_cap = int(cap * block_ms / 1000)
    with sd.InputStream(samplerate=cap, channels=1, dtype="float32",
                        device=device, blocksize=block_cap) as stream:
        while True:
            data, _ = stream.read(block_cap)
            x = data.flatten().astype(np.float32)
            if cap != SAMPLE_RATE:
                x = _decimate_to_12k(x, cap)
            frame = int(SAMPLE_RATE * frame_ms / 1000)
            for i in range(0, len(x) - frame + 1, frame):
                yield x[i:i + frame]
