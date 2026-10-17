"""VCM wake gate.

Two tiers:
  1. Energy/RMS pre-gate - ALWAYS available, no model, no internet. Keeps CPU
     low when idle and gates the heavier stage.
  2. Wake-word model - the real gate: our own jarvis_cnn.onnx (log-mel CNN,
     trained from scratch on the Kaggle Jarvis set, deployed version).
     No extra deps beyond onnxruntime. wake_engine="energy" still selects a
     no-model RMS gate only (debug fallback).

CONTRACT (jarvis_cnn.onnx, verified against the training code):
  * The mic delivers float32 mono at MIC_RATE (12 kHz, ReSpeaker HAT).
  * The wake CNN was trained at 16 kHz, so each 30 ms frame is resampled
    12 -> 16 kHz and appended to a rolling 1 s buffer; the buffer is scored
    while the voiced latch is open (see CnnWakeGate below).
"""
import logging
import os

import numpy as np

log = logging.getLogger("wake")

MIC_RATE = 12000          # ReSpeaker 2-Mics HAT delivers 12 kHz
CNN_RATE = 16000          # the wake CNN's training rate
FRAME_MS = 30
FRAME = int(MIC_RATE * FRAME_MS / 1000)      # 360 mic samples


def _resample_12_to_16(frame):
    """Resample a 12 kHz mono frame to 16 kHz (linear interp, no extra deps)."""
    if frame.size < 2:
        return np.zeros(0, dtype=np.float32)
    n_out = int(round(frame.size * CNN_RATE / MIC_RATE))
    x_old = np.linspace(0.0, 1.0, num=frame.size)
    x_new = np.linspace(0.0, 1.0, num=n_out)
    return np.interp(x_new, x_old, frame).astype(np.float32)


class EnergyGate:
    """Voice-activity pre-gate: trigger when RMS exceeds threshold for a few
    consecutive frames after a quiet period (avoids re-triggering on our own
    confirmation sounds)."""
    def __init__(self, threshold=0.012, min_active=3, cooldown_ms=1500):
        self.threshold = threshold
        self.min_active = min_active
        self.cooldown = int(cooldown_ms / FRAME_MS)
        self._active = 0
        self._cooldown = 0

    def reset(self):
        self._active = 0
        self._cooldown = 0

    def process(self, frame):
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        if rms >= self.threshold:
            self._active += 1
        else:
            self._active = 0
        if self._active >= self.min_active:
            self._active = 0
            self._cooldown = self.cooldown
            return True
        return False


# --------------------------------------------------------------------------
# Wake-word CNN engine ("jarvis_cnn.onnx", trained from scratch on the
# Kaggle Jarvis set + real recordings - see model/wakeword/ in the project
# root for the training script + artifacts; the deployed ONNX ships in
# model/ next to the CRNN).
#
# CONTRACT (jarvis_cnn.onnx, verified against the training code):
#   * input  "mel": (B, 1, 40, 98) float32 log-mel spectrogram
#       - 16 kHz mono, n_fft=512, hop=160 (10 ms), 40 mel bands, Hann window
#       - center-padded STFT -> 101 frames for a 1.0 s clip -> crop/pad to 98
#       - layout is MEL-MAJOR: (mel_bin, frame), i.e. logmel(...).T
#   * output "prob": (B,) sigmoid probability that the clip contains the wake word
#   * operating point on the Pi (12 kHz mic -> 16 kHz): wake_threshold lives in
#     runtime/config.json (deployed: 0.5)
# --------------------------------------------------------------------------
CNN_RATE = 16000           # model's training rate
CNN_N_FFT = 512
CNN_HOP = 160              # 10 ms
CNN_N_MELS = 40
CNN_T_FRAMES = 98          # 1.0 s clip
CNN_MODEL = "jarvis_cnn.onnx"
CNN_LATCH = 20             # frames (~0.6 s) to keep scoring after last voiced
                           # frame - lets the 1 s window slide across the tail


def _cnn_logmel(x16k):
    """float32 mono 16 kHz -> (40, 98) log-mel, same math as train_cnn.py."""
    n_fft, hop, n_mels = CNN_N_FFT, CNN_HOP, CNN_N_MELS
    x = np.pad(x16k, (n_fft // 2, n_fft // 2))
    n_frames = (len(x) - n_fft) // hop + 1
    idx = np.arange(n_fft)[None, :] + np.arange(n_frames)[:, None] * hop
    frames = x[idx] * np.hanning(n_fft).astype(np.float32)[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2) / n_fft
    # mel filterbank (slaney-style triangle bins, identical to build_mel_bank)
    def hz2mel(h): return 2595.0 * np.log10(1.0 + h / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(20.0), hz2mel(CNN_RATE / 2), n_mels + 2)
    hzs = mel2hz(mels)
    bins = np.clip(np.floor((n_fft + 1) * hzs / CNN_RATE).astype(np.int64), 1, n_fft)
    bank = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            bank[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            bank[i, c:r] = (r - np.arange(c, r)) / (r - c)
    mel = np.log(power @ bank.T + 1e-5).astype(np.float32)   # (frames, mels)
    if mel.shape[0] < CNN_T_FRAMES:
        mel = np.pad(mel, ((0, CNN_T_FRAMES - mel.shape[0]), (0, 0)),
                     constant_values=-11.5)
    else:
        mel = mel[:CNN_T_FRAMES]
    return mel.T                                            # (40, 98) mel-major


class CnnWakeGate:
    """Energy pre-gate + the deployed 'Hello Jarvis' CNN confirmation.

    Two-tier: the RMS gate keeps idle CPU near zero, the ONNX CNN over
    log-mel confirms. Falls back to pure energy gating if the model or
    onnxruntime can't be loaded.
    """
    def __init__(self, model_dir="model", cnn_threshold=0.5,
                 persist_frames=2, **energy_kwargs):
        self.threshold = cnn_threshold
        self.energy = EnergyGate(**energy_kwargs)
        self._sess = None
        self._buf = np.zeros(0, dtype=np.float32)   # rolling 16 kHz float32
        self._latch = 0                             # voiced-tail frames left
        self._last_score = 0.0
        self._confirmed_score = 0.0                 # score of the last confirmed wake
        # Persistence: a single lucky window must not fire. Require
        # `persist_frames` CONSECUTIVE frames (30 ms each) to score >=
        # threshold. "Jarvis" sustains the score across the 1 s window; a
        # door slam or clap is one sharp frame. persist_frames <= 1 keeps
        # the old single-frame behaviour.
        self.persist = max(1, int(persist_frames))
        self._run = 0                               # consecutive >=thr frames
        try:
            import onnxruntime as ort
            path = os.path.join(model_dir, CNN_MODEL)
            if not os.path.isfile(path):
                raise FileNotFoundError("no %s in %s/" % (CNN_MODEL, model_dir))
            self._sess = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"])
            log.info("Wake Word CNN loaded: %s", CNN_MODEL)
            log.debug("wake threshold %.2f", self.threshold)
        except Exception as e:
            log.warning("wake CNN unavailable (%s) - using energy gate only", e)
            self._sess = None

    def process(self, frame):
        """One 30 ms mic frame. Rolling 1 s buffer is ALWAYS appended (cheap);
        the CNN only scores while a voiced latch is active, so idle CPU stays
        near the energy-gate floor. The latch keeps scoring ~0.6 s past the
        last voiced frame so the 1 s window slides across the whole phrase."""
        r = _resample_12_to_16(frame)
        if r.size:
            self._buf = np.concatenate([self._buf, r])[-CNN_RATE:]
        if self._sess is None:
            return self.energy.process(frame)   # energy-only fallback

        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        self._latch = CNN_LATCH if rms >= self.energy.threshold \
            else max(0, self._latch - 1)

        if self._buf.size >= CNN_RATE and self._latch > 0:
            mel = _cnn_logmel(self._buf)        # (40, 98)
            prob = float(self._sess.run(["prob"], {"mel": mel[None, None]})[0][0])
            self._last_score = prob
            if prob >= self.threshold:
                # Persistence gate: count consecutive passing frames.
                self._run += 1
                if self._run < self.persist:
                    return False                # one more frame to confirm
                self._buf = np.zeros(0, dtype=np.float32)
                self._latch = 0
                self.energy.reset()
                log.info("Jarvis is waking up...")
                self._confirmed_score = float(prob)
                self._run = 0
                return True
            # Below threshold: reset the run.
            self._run = 0
        return False

    @property
    def last_score(self):
        return self._last_score


def make_gate(config, model_dir):
    engine = str(config.get("wake_engine", "energy")).lower()
    # Energy-gate kwargs (RMS-based). Kept separate from the wake-word model
    # threshold so the two "threshold" concepts never collide.
    ek = dict(threshold=config.get("wake_rms_threshold", 0.012),
              min_active=config.get("wake_min_active_frames", 3),
              cooldown_ms=config.get("wake_cooldown_ms", 1500))
    if engine == "cnn":
        return CnnWakeGate(model_dir=model_dir,
                           cnn_threshold=config.get("wake_threshold", 0.5),
                           persist_frames=config.get("wake_persist_frames", 2),
                           **ek)
    return EnergyGate(**ek)
