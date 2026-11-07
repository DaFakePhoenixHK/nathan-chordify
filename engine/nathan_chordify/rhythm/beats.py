"""Beat This! wrapper (DESIGN §5.3): beats, downbeats and per-beat log-odds.

Beat This! (final0, no DBN) gives frame-wise beat and downbeat *logits* at 50 fps and no
confidences. We keep, for every beat the minimal post-processor finds:

- ``beat_logit``: the peak beat logit within ±70 ms (the network's log-odds that a beat is there);
- ``downbeat_logit``: the peak downbeat logit within ±70 ms.

``beat_conf = σ(beat_logit − 4)`` (``conf_from_logit``): the frame activations, not IBI
regularity. A logit of 4 (p ≈ 0.98) maps to 0.5, the free-time threshold of §5.3; steady beats
in real and synthetic music score 6–10 (conf > 0.85), beats guessed through a rubato passage
0.5–4.5 (conf < 0.6). The logits also carry the metrical hierarchy that the tempo-octave rule
uses (on-beat vs off-beat eighths, downbeats vs half-bar downbeats; see ``meter.py``).

Beat times are refined below the 20 ms frame grid by parabolic interpolation of the logit peak.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 50                       # Beat This! frame rate (hop 441 @ 22.05 kHz)
CHECKPOINT = "final0"
PEAK_RADIUS = 3                # frames (±60 ms, the post-processor's max-pool is ±70 ms)
CONF_LOGIT_MID = 4.0           # beat_conf = σ(logit − CONF_LOGIT_MID)

_MODELS: dict[str, object] = {}


@dataclass
class TrackerOutput:
    beats: np.ndarray           # [B] seconds
    downbeats: np.ndarray       # [D] seconds (each one is also in beats)
    beat_logit: np.ndarray      # [B]
    downbeat_logit: np.ndarray  # [B] downbeat log-odds at every beat

    @property
    def beat_conf(self) -> np.ndarray:
        return conf_from_logit(self.beat_logit)

    def to_json(self) -> dict:
        return {"version": 1, "model": CHECKPOINT, "beats": self.beats.tolist(),
                "downbeats": self.downbeats.tolist(), "beat_logit": self.beat_logit.tolist(),
                "downbeat_logit": self.downbeat_logit.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "TrackerOutput":
        return cls(*(np.asarray(d[k], dtype=float) for k in ("beats", "downbeats", "beat_logit", "downbeat_logit")))


def conf_from_logit(logit: np.ndarray) -> np.ndarray:
    """Per-beat confidence in [0, 1] from the beat log-odds."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(logit, dtype=float) - CONF_LOGIT_MID)))


def load_mono(path: Path | str, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Read an audio file with soundfile → mono float32 (optionally resampled to ``sr``)."""
    import soundfile as sf
    y, fs = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr is not None and sr != fs:
        import librosa
        y = librosa.resample(y, orig_sr=fs, target_sr=sr).astype(np.float32)
        fs = sr
    return np.ascontiguousarray(y), int(fs)


def default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _model(device: str):
    if device not in _MODELS:
        from beat_this.inference import Audio2Frames
        _MODELS[device] = Audio2Frames(checkpoint_path=CHECKPOINT, device=device)
    return _MODELS[device]


def _peaks(logits: np.ndarray, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Refined peak position (frames, sub-frame) and peak value near each frame index."""
    n = len(logits)
    pos = np.empty(len(frames))
    val = np.empty(len(frames))
    for i, f in enumerate(frames):
        lo, hi = max(0, f - PEAK_RADIUS), min(n, f + PEAK_RADIUS + 1)
        if lo >= hi:
            pos[i], val[i] = f, -12.0
            continue
        k = lo + int(np.argmax(logits[lo:hi]))
        d = 0.0
        if 0 < k < n - 1:
            a, b, c = logits[k - 1], logits[k], logits[k + 1]
            den = a - 2 * b + c
            if den < 0:
                d = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        pos[i], val[i] = k + d, logits[k]
    return pos, val


def frames_to_output(beat_logits: np.ndarray, downbeat_logits: np.ndarray) -> TrackerOutput:
    """Minimal post-processing (Beat This!'s own) + refinement + per-beat log-odds."""
    import torch
    from beat_this.model.postprocessor import Postprocessor
    bl = np.asarray(beat_logits, dtype=np.float32)
    dl = np.asarray(downbeat_logits, dtype=np.float32)
    beats, downs = Postprocessor(type="minimal", fps=FPS)(torch.from_numpy(bl), torch.from_numpy(dl))
    beats, downs = np.asarray(beats, float), np.asarray(downs, float)
    if len(beats) == 0:
        e = np.zeros(0)
        return TrackerOutput(e, e.copy(), e.copy(), e.copy())
    frames = np.round(beats * FPS).astype(int)
    pos, blog = _peaks(bl, frames)
    _, dlog = _peaks(dl, np.round(pos).astype(int))
    t = pos / FPS
    order = np.argsort(t, kind="stable")
    t, blog, dlog = t[order], blog[order], dlog[order]
    # downbeats are snapped to beats by the post-processor: map them onto the refined times
    idx = np.unique([int(np.argmin(np.abs(beats[order] - d))) for d in downs]) if len(downs) else np.zeros(0, int)
    return TrackerOutput(t, t[idx] if len(idx) else np.zeros(0), blog.astype(float), dlog.astype(float))


def track_signal(signal: np.ndarray, sr: int, device: str | None = None) -> TrackerOutput:
    """Run Beat This! on a mono signal (any sample rate; it resamples to 22.05 kHz itself)."""
    import torch
    a2f = _model(device or default_device())
    with torch.inference_mode():
        bl, dl = a2f(np.asarray(signal, dtype=np.float32), sr)
    return frames_to_output(bl.float().cpu().numpy(), dl.float().cpu().numpy())


def track_file(path: Path | str, device: str | None = None) -> TrackerOutput:
    """Beat-track the mix (asset.wav), read with soundfile and averaged to mono."""
    y, sr = load_mono(path)
    return track_signal(y, sr, device)
