"""WP2 stem separation (DESIGN §5.2): ``separate(asset, mode, progress) -> Stems``.

Full mode (the app): htdemucs_ft → ``cache/<key>/stems/{vocals,drums,bass,other,harmonic}.flac``
(44.1 kHz stereo, 16-bit FLAC; harmonic = bass + other, clipped to [-1, 1]). ``Stems.mix`` is
the analysis WAV itself (``asset.wav``); no mix FLAC is written.

Fast mode (CLI/debug only): no separation. ``harmonic`` = ``mix`` = ``asset.wav``, ``bass`` =
the mix low-passed at 300 Hz (``stems/bass_lp300.flac``, zero-phase 4th-order Butterworth, so
no added latency). ``vocals``, ``drums`` and ``other`` are *absent*: they point at
``stems/<name>.absent``, which is never created. Check with ``available(path)`` (or
``stems.model == "none"``) before reading a stem.

Both modes are cached: when the stem files exist, nothing is recomputed.
"""
from __future__ import annotations

from pathlib import Path

from ..types import AudioAsset, Progress, Stems

MODEL = "htdemucs_ft"
ABSENT_SUFFIX = ".absent"
FAST_BASS = "bass_lp300.flac"
LOWPASS_HZ = 300.0
_FULL = ("vocals", "drums", "bass", "other", "harmonic")

__all__ = ["separate", "available", "stems_dir", "MODEL"]


def _noop(stage: str, pct: float, msg: str) -> None:
    pass


def stems_dir(asset: AudioAsset) -> Path:
    return asset.dir / "stems"


def available(path: Path) -> bool:
    """False for the absent stems of fast mode (and for missing files)."""
    return path.suffix != ABSENT_SUFFIX and path.exists()


def _default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def separate(asset: AudioAsset, mode: str = "full", progress: Progress | None = None,
             device: str | None = None) -> Stems:
    """Stems for ``asset`` (cached in ``asset.dir/stems``). ``mode`` is "full" or "fast"."""
    progress = progress or _noop
    d = stems_dir(asset)
    if mode == "full":
        paths = {k: d / f"{k}.flac" for k in _FULL}
        if not all(p.exists() for p in paths.values()):
            from . import demucs_sep
            progress("separate", 0.0, "Separating stems")
            paths = demucs_sep.run(asset.wav, d, device or _default_device(), progress)
        progress("separate", 1.0, "Stems ready")
        return Stems(mix=asset.wav, vocals=paths["vocals"], drums=paths["drums"], bass=paths["bass"],
                     other=paths["other"], harmonic=paths["harmonic"], model=MODEL)
    if mode == "fast":
        bass = d / FAST_BASS
        if not bass.exists():
            progress("separate", 0.0, "Low-passing the mix (fast mode)")
            _lowpass_bass(asset.wav, bass)
        progress("separate", 1.0, "Fast mode: no separation")
        return Stems(mix=asset.wav, vocals=d / f"vocals{ABSENT_SUFFIX}", drums=d / f"drums{ABSENT_SUFFIX}",
                     bass=bass, other=d / f"other{ABSENT_SUFFIX}", harmonic=asset.wav, model="none")
    raise ValueError(f"mode must be 'full' or 'fast', not {mode!r}")


def _lowpass_bass(wav: Path, out: Path) -> None:
    from scipy.signal import butter, sosfiltfilt

    from .demucs_sep import SR, read_stereo, write_flac
    y = read_stereo(wav)
    sos = butter(4, LOWPASS_HZ, btype="low", fs=SR, output="sos")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_flac(out, sosfiltfilt(sos, y, axis=-1))
