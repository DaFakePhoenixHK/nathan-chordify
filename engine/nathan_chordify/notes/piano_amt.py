"""Piano transcription of the "other" stem (DESIGN §5.7, WP5, M5).

    audio = soundfile.read(stems/other.flac) → mono → soxr to 16 kHz, float32   (not the package's load_audio)
    amt   = PianoTranscription(device, checkpoint_path=models/piano_transcription/CRNN_note_F1=…pth)
    notes = the package's regression post-processor on the network output (onset, offset, pitch, velocity)
    offset_pedal = offset extended to the sustain-pedal release (cut at the next strike of the same key)
    kept: velocity ≥ 20, duration ≥ 60 ms, 33 ≤ pitch ≤ 96                     (``filter_notes``)

The raw transcription (every note and pedal event) is cached as ``cache/<key>/notes.npz`` (no
pickles), so the filters can change without the GPU. ``transcribe`` returns the filtered
``NoteEvents``; ``notes.symbolic`` turns the notes ≥ MIDI 55 into the symbolic chroma and
``notes.bass`` uses the notes < MIDI 55 as the left-hand bass where the bass stem is quiet.

Package facts (piano_transcription_inference 0.0.6, verified in M5):
- its ``load_audio`` passes a positional resample argument that librosa 0.11 rejects, so audio is
  read with soundfile and resampled with soxr here;
- its checkpoint auto-download shells out to ``wget`` (absent on Windows): the checkpoint is
  prefetched by ``scripts/fetch_models.py --all`` and ``checkpoint_path`` is always passed;
- ``transcribe(audio, midi_path)`` needs the second argument (None = no MIDI file) and runs the
  network with batch size 1; the same steps are run here with a batch of 8 (≈ 5× faster, same
  notes). fp16 autocast is NOT used: it returns no notes at all.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np

from .. import config
from ..types import NoteEvents

SR = 16000
CHECKPOINT = config.MODELS_DIR / "piano_transcription" / "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
CACHE_NAME = "notes.npz"
CACHE_VERSION = 1
SOURCE = "piano_amt@other"
BATCH = 8

MIN_VELOCITY = 20                 # §5.7 filters
MIN_DUR_S = 0.06
PITCH_LO, PITCH_HI = 33, 96

_MODELS: dict[str, object] = {}


# ---------------------------------------------------------------------------------------------
# Audio and model
# ---------------------------------------------------------------------------------------------

def load_audio(path: str | Path) -> np.ndarray:
    """soundfile → mono → soxr to 16 kHz, float32 (§5.7)."""
    import soundfile as sf
    import soxr
    y, fs = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if fs != SR:
        y = soxr.resample(y, fs, SR)
    return np.ascontiguousarray(y, dtype=np.float32)


def checkpoint() -> Path:
    if not CHECKPOINT.exists() or CHECKPOINT.stat().st_size < 1.6e8:
        raise FileNotFoundError(f"piano transcription checkpoint missing: {CHECKPOINT}. "
                                r"Run .venv\Scripts\python scripts\fetch_models.py --all")
    return CHECKPOINT


def model(device: str = "cuda"):
    """The PianoTranscription object for ``device``, kept between songs."""
    if device not in _MODELS:
        ck = checkpoint()                # before the package can try its wget download
        from piano_transcription_inference import PianoTranscription
        with contextlib.redirect_stdout(io.StringIO()):
            _MODELS[device] = PianoTranscription(device=device, checkpoint_path=str(ck))
    return _MODELS[device]


def run_amt(audio: np.ndarray, device: str = "cuda", batch: int = BATCH) -> tuple[np.ndarray, np.ndarray]:
    """``PianoTranscription.transcribe`` without the MIDI file and with batched inference.

    Returns (notes [N, 4] = onset, offset, pitch, velocity sorted by onset; pedals [P, 2])."""
    from piano_transcription_inference.pytorch_utils import forward
    from piano_transcription_inference.utilities import RegressionPostProcessor
    amt = model(device)
    x = np.asarray(audio, dtype=np.float32)[None, :]
    n = x.shape[1]
    seg = amt.segment_samples
    pad = int(np.ceil(max(n, 1) / seg)) * seg - n
    x = np.concatenate([x, np.zeros((1, pad), dtype=np.float32)], axis=1)
    frames = amt.enframe(x, seg)
    with contextlib.redirect_stdout(io.StringIO()):
        out = forward(amt.model, frames, batch_size=batch)
    for k in out:
        out[k] = amt.deframe(out[k])[0:n]
    pp = RegressionPostProcessor(amt.frames_per_second, classes_num=amt.classes_num,
                                 onset_threshold=amt.onset_threshold, offset_threshold=amt.offset_threshod,
                                 frame_threshold=amt.frame_threshold,
                                 pedal_offset_threshold=amt.pedal_offset_threshold)
    notes_ev, pedal_ev = pp.output_dict_to_midi_events(out)
    notes = np.array([[e["onset_time"], e["offset_time"], e["midi_note"], e["velocity"]] for e in notes_ev],
                     dtype=np.float64).reshape(-1, 4)
    notes = notes[np.lexsort((notes[:, 2], notes[:, 0]))]
    pedals = np.array([[e.get("onset_time", e.get("osnet_time", 0.0)), e["offset_time"]] for e in pedal_ev],
                      dtype=np.float64).reshape(-1, 2)
    pedals = pedals[np.argsort(pedals[:, 0], kind="stable")]
    return notes, pedals


# ---------------------------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------------------------

def pedal_offsets(onset: np.ndarray, offset: np.ndarray, pitch: np.ndarray, pedals: np.ndarray) -> np.ndarray:
    """offset_pedal: a key released while the sustain pedal is down keeps sounding until the
    pedal is released — but never past the next strike of the same key."""
    onset, offset = np.asarray(onset, float), np.asarray(offset, float)
    pitch = np.asarray(pitch).astype(int)
    out = offset.copy()
    ped = np.asarray(pedals, float).reshape(-1, 2)
    if len(ped):
        k = np.searchsorted(ped[:, 0], offset, side="right") - 1        # last pedal pressed by the release
        ok = k >= 0
        kk = np.where(ok, k, 0)
        down = ok & (ped[kk, 1] > offset)
        out[down] = ped[kk[down], 1]
    for p in np.unique(pitch):                                          # re-strike of the same key
        idx = np.flatnonzero(pitch == p)
        idx = idx[np.argsort(onset[idx], kind="stable")]
        nxt = onset[idx[1:]]
        cur = out[idx[:-1]]
        out[idx[:-1]] = np.maximum(offset[idx[:-1]], np.minimum(cur, nxt))
    return out


def to_events(notes: np.ndarray, pedals: np.ndarray) -> NoteEvents:
    notes = np.asarray(notes, float).reshape(-1, 4)
    on, off, pitch, vel = notes[:, 0], notes[:, 1], notes[:, 2].astype(np.int64), notes[:, 3]
    return NoteEvents(onset=on, offset=off, pitch=pitch, velocity=vel,
                      offset_pedal=pedal_offsets(on, off, pitch, pedals), source=SOURCE)


def filter_notes(ne: NoteEvents, min_velocity: float = MIN_VELOCITY, min_dur: float = MIN_DUR_S,
                 lo: int = PITCH_LO, hi: int = PITCH_HI) -> NoteEvents:
    """§5.7: velocity ≥ 20, duration (key down) ≥ 60 ms, 33 ≤ pitch ≤ 96."""
    pitch = np.asarray(ne.pitch)
    m = ((np.asarray(ne.velocity) >= min_velocity) & (np.asarray(ne.offset) - np.asarray(ne.onset) >= min_dur - 1e-9)
         & (pitch >= lo) & (pitch <= hi))
    return NoteEvents(onset=np.asarray(ne.onset)[m], offset=np.asarray(ne.offset)[m], pitch=pitch[m],
                      velocity=np.asarray(ne.velocity)[m], offset_pedal=np.asarray(ne.offset_pedal)[m],
                      source=ne.source)


def empty() -> NoteEvents:
    z = np.zeros(0)
    return NoteEvents(onset=z, offset=z, pitch=np.zeros(0, dtype=np.int64), velocity=z, offset_pedal=z,
                      source=SOURCE)


# ---------------------------------------------------------------------------------------------
# Cache (cache/<key>/notes.npz)
# ---------------------------------------------------------------------------------------------

def song_dir(wav: str | Path) -> Path:
    """cache/<key>/stems/other.flac → cache/<key>; any other file → its folder."""
    p = Path(wav).resolve()
    return p.parent.parent if p.parent.name == "stems" else p.parent


def save_cache(d: str | Path, notes: np.ndarray, pedals: np.ndarray, meta: dict) -> Path:
    path = Path(d) / CACHE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, notes=np.asarray(notes, float).reshape(-1, 4),
                            pedals=np.asarray(pedals, float).reshape(-1, 2),
                            meta=np.array(json.dumps({"version": CACHE_VERSION, **meta})))
    os.replace(tmp, path)
    return path


def load_raw(d: str | Path) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """(notes [N, 4], pedals [P, 2], meta) of a cached transcription, or None."""
    path = Path(d) / CACHE_NAME
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta.get("version") != CACHE_VERSION:
            return None
        return z["notes"], z["pedals"], meta


def load_cached(d: str | Path, *, raw: bool = False) -> NoteEvents | None:
    """The cached notes of a song folder (filtered unless ``raw``), or None when not transcribed."""
    got = load_raw(d)
    if got is None:
        return None
    ne = to_events(got[0], got[1])
    return ne if raw else filter_notes(ne)


def transcribe(wav: str | Path, device: str = "cuda", *, cache_dir: str | Path | None = None,
               force: bool = False, raw: bool = False) -> NoteEvents:
    """Piano notes of ``wav`` (normally ``stems/other.flac``), cached in ``cache_dir`` (default:
    the song folder, ``song_dir(wav)``) as notes.npz. Returns the §5.7-filtered notes unless ``raw``."""
    d = Path(cache_dir) if cache_dir is not None else song_dir(wav)
    if not force:
        got = load_cached(d, raw=raw)
        if got is not None:
            return got
    audio = load_audio(wav)
    notes, pedals = run_amt(audio, device)
    save_cache(d, notes, pedals, {"input": Path(wav).name, "samples": int(len(audio)), "sr": SR,
                                  "checkpoint": CHECKPOINT.name})
    ne = to_events(notes, pedals)
    return ne if raw else filter_notes(ne)
