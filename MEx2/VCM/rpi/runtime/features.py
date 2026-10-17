"""VCM feature extraction + ONNX inference. Self-contained (numpy + onnxruntime
only - NO torch, NO librosa, NO soundfile). Runs on the Raspberry Pi CPU.

The log-mel pipeline is byte-for-byte identical to the VCM training code
(model/vcm_model.py), so the Pi sees exactly the features the deployed CRNN
(crnn.onnx) was trained on:
    STFT (n_fft=400, hop=200) -> power -> 40-band mel (50-6000 Hz) -> log
    -> per-clip z-score -> [40, 96]  (padded with -11.5)

Two entry points:
  - logmel_from_array(x, sr): in-memory float32 mono (live mic) -> [40,96]
  - logmel_from_wav(path):    wav file -> [40,96]
Then:
  - VCMClassifier: load ONNX + meta, predict() -> dict with fine + intent
    label, confidence, and full probability vectors.

ONNX contract (crnn.onnx):
  input  logmel  [batch, 1, 40, 96]  float32
  output fine_logits   [batch, 37]   float32
  output intent_logits [batch, 20]   float32
"""
import json, os, wave
import numpy as np

# --- Feature config (FIXED; must match crnn_meta.json) ---------------
SAMPLE_RATE = 12000
ENERGY_NORM = True          # match training (vcm_model.wav_to_logmel_np): RMS-
REF_RMS = 0.1               # rescale the waveform before the STFT, so the
                            # runtime path == training path.
N_MELS = 40
N_FFT = 400
HOP = 200
MAX_FRAMES = 96
PAD_VALUE = -11.5
F_MIN, F_MAX = 50, 6000


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


_mel_cache = {}
def _get_mel_filters(sr, n_fft, n_mels, f_min, f_max):
    key = (sr, n_fft, n_mels, f_min, f_max)
    if key in _mel_cache:
        return _mel_cache[key]
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    m = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    m_hz = _mel_to_hz(m)
    bins = (m_hz * (n_fft // 2 + 1) / (sr / 2)).astype(int)
    f = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c > l:
            f[i - 1, l:c] = (freqs[l:c] - m_hz[i - 1]) / (m_hz[i] - m_hz[i - 1])
        if r > c:
            f[i - 1, c:r] = (m_hz[i + 1] - freqs[c:r]) / (m_hz[i + 1] - m_hz[i])
    f = np.clip(f, 0, 1)
    norm = f.max(axis=1, keepdims=True)
    norm[norm == 0] = 1
    f = f / norm
    _mel_cache[key] = f.astype(np.float32)
    return _mel_cache[key]


def _stft_power(x, n_fft, hop):
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames <= 0:
        return None
    idx = np.arange(n_fft)[None, :] + (np.arange(n_frames) * hop)[:, None]
    frames = x[idx] * np.hanning(n_fft).astype(np.float32)
    spec = np.fft.rfft(frames.astype(np.float64), axis=1)
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float64)


def _power_to_logmel(power, sr):
    if power is None:
        return None
    mel = _get_mel_filters(sr, N_FFT, N_MELS, F_MIN, F_MAX).astype(np.float64)
    logmel = np.log(power @ mel.T + 1e-5).astype(np.float32)
    logmel = logmel.T
    T = logmel.shape[1]
    if T >= MAX_FRAMES:
        logmel = logmel[:, :MAX_FRAMES]
    else:
        logmel = np.pad(logmel, ((0, 0), (0, MAX_FRAMES - T)), constant_values=PAD_VALUE)
    mu = logmel.mean(); sd = logmel.std()
    if sd < 1e-5:
        sd = 1e-5
    return ((logmel - mu) / sd).astype(np.float32)


def _resample_linear(x, sr, target_sr):
    if sr == target_sr:
        return x
    dur = len(x) / sr
    new_n = max(1, int(round(dur * target_sr)))
    old_pos = np.arange(len(x))
    new_pos = np.linspace(0, len(x) - 1, new_n)
    return np.interp(new_pos, old_pos, x).astype(np.float32)


def logmel_from_array(x, sr=SAMPLE_RATE):
    """In-memory float32 mono (any sr) -> [40,96] log-mel. None on failure."""
    try:
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float32)
        x = _resample_linear(x, sr, SAMPLE_RATE)
        if ENERGY_NORM:
            rms = float(np.sqrt(np.mean(x ** 2)))
            if rms > 1e-6:
                x = x * (REF_RMS / rms)
        return _power_to_logmel(_stft_power(x, N_FFT, HOP), SAMPLE_RATE)
    except Exception:
        return None


def _load_wav_mono_raw(path):
    """(x, sr) mono float32, NO resampling - used by perf.py wake."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); nch = w.getnchannels(); sw = w.getsampwidth()
        n = w.getnframes(); raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return x, sr


def _load_wav_mono(path, target_sr):
    x, sr = _load_wav_mono_raw(path)
    return _resample_linear(x, sr, target_sr)


def logmel_from_wav(path):
    """wav file -> [40,96] log-mel. None on failure."""
    try:
        x = _load_wav_mono(path, SAMPLE_RATE)
        if len(x) < HOP:
            return None
        if ENERGY_NORM:
            rms = float(np.sqrt(np.mean(x ** 2)))
            if rms > 1e-6:
                x = x * (REF_RMS / rms)
        return _power_to_logmel(_stft_power(x, N_FFT, HOP), SAMPLE_RATE)
    except Exception:
        return None


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class VCMClassifier:
    """Dual-head VCM command classifier (37 fine / 20 intent)."""

    def __init__(self, onnx_path, meta_path):
        import onnxruntime as ort
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.fine_labels = list(self.meta["fine_labels"])
        self.intent_labels = list(self.meta["intent_labels"])
        self.fine2intent = dict(self.meta.get("fine2intent", {}))
        self.fine2transcript = dict(self.meta.get("fine2transcript", {}))
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(onnx_path, sess_opts,
                                          providers=["CPUExecutionProvider"])
        self._in_name = self._sess.get_inputs()[0].name

    def _forward(self, logmel):
        x = np.asarray(logmel, dtype=np.float32)[None, None, :, :]  # [1,1,40,96]
        fine_l, intent_l = self._sess.run(None, {self._in_name: x})
        return fine_l[0], intent_l[0]

    def predict(self, logmel):
        """Return dict with fine/intent label, confidence, and full probs."""
        fine_l, intent_l = self._forward(logmel)
        fp = _softmax(fine_l)
        ip = _softmax(intent_l)
        fi = int(np.argmax(fp))
        ii = int(np.argmax(ip))
        return {
            "fine_label": self.fine_labels[fi],
            "fine_conf": float(fp[fi]),
            "fine_idx": fi,
            "fine_probs": fp.tolist(),
            "intent_label": self.intent_labels[ii],
            "intent_conf": float(ip[ii]),
            "intent_idx": ii,
            "intent_probs": ip.tolist(),
        }

    def predict_array(self, x, sr=SAMPLE_RATE):
        lm = logmel_from_array(x, sr)
        if lm is None:
            raise ValueError("could not compute log-mel from array")
        return self.predict(lm)

    def predict_wav(self, path):
        lm = logmel_from_wav(path)
        if lm is None:
            raise ValueError("could not compute log-mel from wav: %s" % path)
        return self.predict(lm)
