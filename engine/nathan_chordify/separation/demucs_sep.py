"""htdemucs_ft stem separation, in-process (DESIGN §5.2, §5.14).

The mix tensor stays on the CPU (``apply_model`` allocates whole-track buffers on the input's
device) while the model runs on CUDA; that keeps peak VRAM flat for any song length.
OOM ladder: ``segment`` 7.8 → 6 (after ``empty_cache`` and unloading the chord models) → CPU.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ..types import Progress

log = logging.getLogger(__name__)

MODEL = "htdemucs_ft"
SR = 44100
SOURCES = ("vocals", "drums", "bass", "other")
SEGMENT = 7.8            # htdemucs maximum
SEGMENT_OOM = 6.0

_sep = None              # (device, Separator): kept loaded between jobs
_lock = threading.Lock()


def _separator(device: str, segment: float, progress: Progress | None):
    """The resident ``Separator`` on ``device``, with ``segment`` and a progress callback."""
    global _sep
    from demucs.api import Separator

    def cb(d: dict) -> None:
        if progress is None or d.get("state") != "end":
            return
        n_models = max(int(d.get("models", 1)), 1)
        frac = (d.get("segment_offset", 0) / max(d.get("audio_length", 1), 1))
        pct = (d.get("model_idx_in_bag", 0) + min(frac, 1.0)) / n_models
        progress("separate", min(pct, 0.99), "Separating stems")

    if _sep is None or _sep[0] != device:
        release()
        _sep = (device, Separator(model=MODEL, device=device, shifts=1, overlap=0.25, segment=segment))
    _sep[1].update_parameter(segment=segment, callback=cb)
    return _sep[1]


def release() -> None:
    """Unload the separator (frees about 0.35 GB of weights)."""
    global _sep
    _sep = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def read_stereo(path: Path) -> np.ndarray:
    """[2, N] float32 at the file's rate (mono files are duplicated)."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path.name}: expected {SR} Hz, got {sr}")
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return np.ascontiguousarray(y[:, :2].T)


def write_flac(path: Path, y: np.ndarray, sr: int = SR) -> None:
    """[C, N] or [N] float → 16-bit FLAC, clipped to [-1, 1], written atomically."""
    data = np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0)
    if data.ndim == 2:
        data = data.T
    tmp = path.with_name(path.name + ".part")
    sf.write(str(tmp), data, sr, subtype="PCM_16", format="FLAC")
    os.replace(tmp, path)


def separate_tensor(mix: np.ndarray, device: str, progress: Progress | None = None) -> dict[str, np.ndarray]:
    """[2, N] float32 mix → {vocals, drums, bass, other} [2, N] float32, with the OOM ladder."""
    import torch
    x = torch.from_numpy(mix)            # stays on the CPU
    attempts = [(device, SEGMENT)]
    if device.startswith("cuda"):
        attempts += [(device, SEGMENT_OOM), ("cpu", SEGMENT)]
    for i, (dev, seg) in enumerate(attempts):
        try:
            with _lock, torch.inference_mode():
                _, stems = _separator(dev, seg, progress).separate_tensor(x, SR)
            return {k: stems[k].cpu().numpy() for k in SOURCES}
        except torch.cuda.OutOfMemoryError:
            if i == len(attempts) - 1:
                raise
            log.warning("demucs: CUDA out of memory (device %s, segment %.1f); next: %s",
                        dev, seg, attempts[i + 1])
            if i == 0:
                from .. import models
                models.release_gpu()
            release()
    raise AssertionError("unreachable")


def run(wav: Path, out_dir: Path, device: str, progress: Progress | None = None) -> dict[str, Path]:
    """Separate ``wav`` into ``out_dir/{vocals,drums,bass,other,harmonic}.flac``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    stems = separate_tensor(read_stereo(wav), device, progress)
    t1 = time.perf_counter()
    paths = {k: out_dir / f"{k}.flac" for k in (*SOURCES, "harmonic")}
    for k in SOURCES:
        write_flac(paths[k], stems[k])
    write_flac(paths["harmonic"], stems["bass"] + stems["other"])
    log.info("demucs %s on %s: separation %.1f s, writing %.1f s", MODEL, device, t1 - t0,
             time.perf_counter() - t1)
    return paths
