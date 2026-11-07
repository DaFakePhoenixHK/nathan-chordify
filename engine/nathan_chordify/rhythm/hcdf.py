"""Harmonic change detection (Harte, Sandler & Gasser 2006) on the harmonic stem (DESIGN §5.3).

12-bin chroma (CQT from C2, 12 bins/octave, hop 1024 @ 22.05 kHz) → 6-D tonal centroid →
Gaussian smoothing (σ = 1 frame) → ξ(t) = ‖c(t+1) − c(t−1)‖. Peaks above mean + 1.5·std are
harmonic changes; the decoder adds a +0.5 nat boundary bonus within ±60 ms of one (§5.10).
The CQT starts at C2 so the low filters stay short (≈ 0.26 s) and the change times sharp.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 22050
HOP = 1024
SIGMA = 1.0
K_STD = 1.5


def curve(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """(frame times in s, harmonic change function ξ)."""
    import librosa
    from scipy.ndimage import gaussian_filter1d
    y = np.asarray(y, dtype=np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < HOP * 4:
        return np.zeros(0), np.zeros(0)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C2"),
                                        n_octaves=6, bins_per_octave=12, n_chroma=12)
    energy = chroma.sum(axis=0)
    chroma[:, energy < 1e-3 * max(float(energy.max()), 1e-12)] = 0.0     # silence → zero centroid
    tc = librosa.feature.tonnetz(chroma=chroma)                           # [6, T]
    tc = gaussian_filter1d(tc, SIGMA, axis=1, mode="nearest")
    xi = np.zeros(tc.shape[1])
    xi[1:-1] = np.linalg.norm(tc[:, 2:] - tc[:, :-2], axis=0)
    return np.arange(len(xi)) * HOP / SR, xi


def peaks(y: np.ndarray, sr: int) -> np.ndarray:
    """Times (s) of harmonic-change peaks above mean + 1.5·std."""
    from scipy.signal import find_peaks
    times, xi = curve(y, sr)
    if len(xi) < 3:
        return np.zeros(0)
    idx, _ = find_peaks(xi, height=float(xi.mean() + K_STD * xi.std()))
    return times[idx]


def peaks_from_file(path: Path | str) -> np.ndarray:
    """HCDF peak times of an audio file (the harmonic stem)."""
    from .beats import load_mono
    y, sr = load_mono(path, SR)
    return peaks(y, sr)
