#!/usr/bin/env python3
"""Wake-CNN base pieces (vendored for reproducibility).

Vendored from the wake-CNN training working tree — the
exact front-end, augmentation dataset and WakeCNN that produced the deployed
rpi/model/jarvis_cnn.onnx (deployed). train_cnn.py imports from here so
Self-contained: no imports across repos.

Input contract (unchanged): (B,1,40,98) log-mel @ 16 kHz -> sigmoid prob.
"""

import argparse, json, os, random, re, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



SR = 16000          # target sample rate (spec)
CLIP_S = 1.0        # fixed clip duration (spec)
N = int(SR * CLIP_S)  # 16000 samples
N_MELS = 40
N_FFT = 512
HOP = 160           # 10 ms
WIN = 512
# frames = (N - N_FFT)//HOP + 1 = 98 ; we pad to T_FRAMES
T_FRAMES = 98

SEED = 42


# ----------------------------- data -----------------------------------------
def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    dur = len(x) / sr
    n_out = int(round(dur * SR))
    if n_out < 1:
        return np.zeros(N, np.float32)
    idx = np.linspace(0, len(x) - 1, n_out)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, len(x) - 1)
    frac = (idx - i0)[:, None]
    x = x[i0][:, None] * (1 - frac) + x[i1][:, None] * frac  # index BEFORE broadcasting
    return x.reshape(-1).astype(np.float32)


def load_wav(path: str) -> np.ndarray:
    """Load wav, resample to SR, return float32 mono in [-1, 1]."""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = _resample(x, sr)
    # pad or truncate to exactly N
    if len(x) >= N:
        start = random.randint(0, len(x) - N)
        x = x[start:start + N]
    else:
        pad = N - len(x)
        lp = random.randint(0, pad)
        x = np.pad(x, (lp, pad - lp))  # lp + (pad-lp) = pad -> exactly N
    return x.astype(np.float32)


def load_wav_fixed(path: str) -> np.ndarray:
    """Deterministic load (center crop / center pad) for validation & probe."""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = _resample(x, sr)
    if len(x) >= N:
        start = (len(x) - N) // 2
        x = x[start:start + N]
    else:
        pad = N - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
    return x.astype(np.float32)


class WWDataset(Dataset):
    """Augmentation (train only): polarity flip, gain jitter, speed perturbation,
    noise mixing at random SNR from the noise/ folder. Val stays deterministic."""

    def __init__(self, items, augment=True, noise_dir=None):
        super().__init__()
        self.items = items
        self.augment = augment
        self.noise_files = []
        if augment and noise_dir:
            self.noise_files = sorted(str(p) for p in Path(noise_dir).glob("*.wav"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        if self.augment:
            x = load_wav(path)
            if random.random() < 0.5:
                x = -x  # polarity flip (TTS phase invariance)
            # gain jitter +/-6 dB (distance / speaking volume)
            if random.random() < 0.5:
                x = x * (10.0 ** (random.uniform(-6, 6) / 20.0))
            # speed perturbation 0.9x-1.1x (speaker variation)
            if random.random() < 0.5:
                s = random.uniform(0.9, 1.1)
                n_out = int(round(len(x) / s))
                x = np.interp(np.linspace(0, len(x) - 1, n_out),
                              np.arange(len(x)), x).astype(np.float32)
                if len(x) >= N:
                    st = random.randint(0, len(x) - N)
                    x = x[st:st + N]
                else:
                    x = np.pad(x, (0, N - len(x)))
            # noise mixing at random SNR 5-20 dB
            if self.noise_files and random.random() < 0.7:
                nx, nsr = sf.read(random.choice(self.noise_files),
                                  dtype="float32", always_2d=False)
                if nx.ndim > 1:
                    nx = nx.mean(axis=1)
                if nsr != SR:
                    nx = _resample(nx, nsr)
                if len(nx) < N:
                    nx = np.pad(nx, (0, N - len(nx)))
                else:
                    st = random.randint(0, len(nx) - N)
                    nx = nx[st:st + N]
                sig_rms = float(np.sqrt(np.mean(np.square(x)))) + 1e-8
                noi_rms = float(np.sqrt(np.mean(np.square(nx)))) + 1e-8
                snr = random.uniform(5.0, 20.0)
                nx = nx * (sig_rms / noi_rms) * (10.0 ** (-snr / 20.0))
                x = x + nx
                peak = np.abs(x).max()
                if peak > 1.0:
                    x = x / peak
        else:
            x = load_wav_fixed(path)
        return torch.from_numpy(x.astype(np.float32)), float(label)


def collate(batch):
    xs = torch.stack([b[0] for b in batch])   # (B, N)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return xs, ys


# ----------------------------- features -------------------------------------
def hann(win: int) -> np.ndarray:
    return np.hanning(win).astype(np.float32)


def logmel(x: np.ndarray, win: np.ndarray, n_fft: int, hop: int,
           n_mels: int, sr: int) -> np.ndarray:
    """Compute log-mel spectrogram with a pure-numpy STFT + mel filterbank."""
    x = np.pad(x, (n_fft // 2, n_fft // 2))
    n_frames = (len(x) - n_fft) // hop + 1
    # gather frames
    idx = np.arange(n_fft)[None, :] + np.arange(n_frames)[:, None] * hop
    frames = x[idx] * win[None, :]                       # (F, fft)
    spec = np.fft.rfft(frames, n=n_fft, axis=1)          # (F, fft/2+1)
    power = (spec.real ** 2 + spec.imag ** 2) / n_fft
    mel = power @ MEL_BANK.T                             # (F, n_mels)
    return np.log(mel + 1e-5).astype(np.float32)


def build_mel_bank(n_fft: int, n_mels: int, sr: int) -> np.ndarray:
    import numpy as np
    def hz2hz(h):
        return h
    low, high = 20.0, sr / 2
    def hz2mel(h):
        return 2595.0 * np.log10(1.0 + h / 700.0)
    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(low), hz2mel(high), n_mels + 2)
    hzs = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hzs / sr).astype(np.int64)
    bins = np.clip(bins, 1, n_fft)
    bank = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        for j in range(l, c):
            if c > l:
                bank[i, j] = (j - l) / (c - l)
        for j in range(c, r):
            if r > c:
                bank[i, j] = (r - j) / (r - c)
    return bank


MEL_BANK = build_mel_bank(N_FFT, N_MELS, SR)
WIN_FN = hann(WIN)


# ----------------------------- model ----------------------------------------
class WakeCNN(nn.Module):
    def __init__(self, n_mels: int = N_MELS, t_frames: int = T_FRAMES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                # (32, M/2, T/2)
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                # (64, M/4, T/4)
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                # (128, M/8, T/8)
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, 1, M, T)
        z = self.features(x)
        return self.head(z).squeeze(-1)


