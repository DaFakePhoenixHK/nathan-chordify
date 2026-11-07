"""WP3 rhythm: beats, downbeats, meter and the sub-beat grid (DESIGN §5.3).

    track(asset, stems, overrides=None, progress=None) -> RhythmGrid
    check_octave(grid, main) -> RhythmGrid | None          # pipeline step 5
    build_grid(beats, downbeats, beat_conf, ...) -> RhythmGrid   # pure NumPy (tests, re-pooling)
    hcdf_peaks(stems) -> np.ndarray                         # harmonic-change times for §5.10

The raw Beat This! output is cached in ``<asset.dir>/rhythm_raw.json`` so that re-applying
overrides never re-runs the network.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..types import DecodeResult, AudioAsset, Progress, RhythmGrid, RhythmOverrides, Stems
from . import beats as _beats
from . import hcdf, meter
from .grid import OnsetEnvelope, _forced_free_intervals, build_grid, grid_from_beats, onset_envelope
from .meter import BeatSeq

__all__ = ["track", "check_octave", "build_grid", "hcdf_peaks", "OnsetEnvelope", "onset_envelope"]

RAW_CACHE = "rhythm_raw.json"
OCTAVE_MIN_CHANGES = 8      # check_octave needs at least this many main-layer changes
OCTAVE_EVEN_SHARE = 0.8


def _noop(stage: str, pct: float, msg: str) -> None:
    pass


def _tracker_output(asset: AudioAsset, device: str | None, use_cache: bool) -> _beats.TrackerOutput:
    path = asset.dir / RAW_CACHE
    if use_cache and path.exists():
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("version") == 1 and d.get("model") == _beats.CHECKPOINT:
                return _beats.TrackerOutput.from_json(d)
        except (OSError, ValueError, KeyError):
            pass
    out = _beats.track_file(asset.wav, device)
    if asset.dir.is_dir():
        try:
            path.write_text(json.dumps(out.to_json()), encoding="utf-8")
        except OSError:
            pass
    return out


def track(asset: AudioAsset, stems: Stems, overrides: RhythmOverrides | None = None,
          progress: Progress | None = None, *, device: str | None = None, use_cache: bool = True) -> RhythmGrid:
    """Beat This! on the mix → octave rule, bars, free time, sparse intros, triplet test, cells (§5.3).

    The harmonic stem gives the onsets (triplet test) and the harmonic changes (an intro that
    holds its chord inherits the song's grid, ``meter.backfill_intro``)."""
    progress = progress or _noop
    progress("rhythm", 0.0, "Tracking beats")
    raw = _tracker_output(asset, device, use_cache)
    progress("rhythm", 0.6, "Building the beat grid")
    y, sr = _beats.load_mono(stems.harmonic)
    onset = onset_envelope(y, sr)
    duration = asset.duration_s or len(y) / sr
    grid = build_grid(raw.beats, raw.downbeats, raw.beat_conf, duration=duration, beat_logit=raw.beat_logit,
                      downbeat_logit=raw.downbeat_logit, onset=onset, overrides=overrides,
                      harmonic_changes=hcdf_peaks(stems))
    progress("rhythm", 1.0, f"{grid.bpm:.1f} BPM, {len(grid.bars)} bars")
    return grid


_HCDF_CACHE: dict[tuple, np.ndarray] = {}


def hcdf_peaks(stems: Stems) -> np.ndarray:
    """Harmonic-change peak times (s) of the harmonic stem (+0.5 nat boundary bonus, §5.10).

    ``track`` needs them too (sparse intros), so they are memoised per file version."""
    path = Path(stems.harmonic)
    try:
        st = path.stat()
        key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:
        return hcdf.peaks_from_file(path)
    if key not in _HCDF_CACHE:
        if len(_HCDF_CACHE) > 32:
            _HCDF_CACHE.clear()
        _HCDF_CACHE[key] = hcdf.peaks_from_file(path)
    return _HCDF_CACHE[key].copy()


def _change_beats(grid: RhythmGrid, main: DecodeResult) -> tuple[np.ndarray, np.ndarray]:
    """Beat index of each main-layer chord change on a metered beat, and the change count per beat."""
    t = grid.beats
    starts = sorted({float(s.t0) for s in main.segments if s.layer == "main"})[1:]
    counts = np.zeros(len(t), int)
    idx = []
    for x in starts:
        j = int(np.argmin(np.abs(t - x)))
        if abs(t[j] - x) <= 0.2 * grid.ibi_ref and grid.beat_in_bar[j] > 0:
            idx.append(j)
            counts[j] += 1
    return np.asarray(idx, int), counts


def check_octave(grid: RhythmGrid, main: DecodeResult) -> RhythmGrid | None:
    """§5.3 octave check after the first main decode; None = keep the grid.

    Halves the tempo when ≥ 80% of the main-layer changes fall on even beats of their bar
    (0, 2, 4, … from the downbeat), at least 8 changes are seen, the grid is ≥ 110 BPM (the
    static rule's threshold; slower songs that change chords every 2 beats are left alone), the
    halved tempo stays ≥ 45 BPM and the main bar length is even. The beats are folded with the
    same half-beat unit grid as the octave rule (``meter.unit_fold``), with chord changes and old
    downbeats as on-beat evidence, so
    parts already at the slow level (回家's intro) are kept and odd tracked bars do not shift the
    phase. Bars are re-placed with the bar Viterbi; 2-beat bars become 4-beat bars, phased where
    more chord changes start a bar. User-edited grids are never changed.
    """
    if grid.edited or grid.bpm < meter.FOLD_MIN_BPM or grid.bpm / 2 < meter.MIN_BPM:
        return None
    idx, counts = _change_beats(grid, main)
    if len(idx) < OCTAVE_MIN_CHANGES:
        return None
    if np.mean((grid.beat_in_bar[idx] - 1) % 2 == 0) < OCTAVE_EVEN_SHARE:
        return None
    old_L = meter.mode_int([n for _, n in grid.bars]) or 4
    if old_L % 2:
        return None                       # 3-, 5-, 7-beat bars cannot be halved evenly
    B = len(grid.beats)
    old_db = np.zeros(B, bool)
    for s, _ in grid.bars:
        old_db[s] = True
    seq = BeatSeq(grid.beats.copy(), grid.beat_conf.copy(), 3.0 * counts, np.full(B, np.nan), old_db)
    seq = meter.unit_fold(seq, 2 * grid.ibi_ref)
    old_of_new = np.clip(np.searchsorted(grid.beats, seq.t + 1e-6) - 1, 0, B - 1)   # inserted → previous old beat
    compound_bar = np.zeros(B, bool)
    for j, (s, n) in enumerate(grid.bars):
        num, den = _meter_at(grid, j)
        compound_bar[s:s + n] = den == 8 and num % 3 == 0 and num // 3 == n
    half_L = old_L // 2
    L = 4 if (half_L <= 2 and not compound_bar.any()) else half_L
    free_iv = _forced_free_intervals(seq.t, grid.free)
    evidence = np.where(seq.is_db, 3.0, -3.0) + (2.0 / 3.0) * np.nan_to_num(seq.blog, nan=0.0)
    ternary = np.isin(grid.subdiv[old_of_new], (3, 6))
    return grid_from_beats(seq, free_iv, L, evidence, duration=float(grid.cell_t[-1]), ternary_beat=ternary,
                           compound_beat=compound_bar[old_of_new], free_cells_source=grid.cell_t, edited=False)


def _meter_at(grid: RhythmGrid, bar: int) -> tuple[int, int]:
    num, den = 4, 4
    for b, n, d in grid.meter:
        if b <= bar:
            num, den = n, d
    return num, den
