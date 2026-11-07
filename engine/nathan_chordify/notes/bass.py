"""Bass tracker (DESIGN §5.6): pitch-class salience, onset-segmented bass notes and P_bass per cell.

    x   = bass stem, mono, 22.05 kHz (debug fast mode: the stem is the mix low-passed at 300 Hz)
    C   = |CQT(x)|, fmin E1, 12 bins/octave, filter_scale 0.5, 6 octaves, hop 256 (11.6 ms)
    S   = Σ_h 0.8^(h−1) · C[f0 + round(12·log2 h)], h = 1..4, f0 ∈ [E1, G3]   (harmonic summation)
    sal = softmax_pc( max_octave S / max_pc(max_octave S) / 0.5 )            (per-frame max-normalised)
    notes: onsets (librosa onset strength without its extra centring lag, delta 0.03, backtracked
           ≤ 35 ms, latency-compensated) → segments; pc = argmax of the mean salience after a 30 ms
           attack skip; conf = mean per-frame max salience; segments < 60 ms merge into the previous
           note; silent segments are rests; a note ends where it first decays 25 dB below its peak.
    P_bass(g) = normalise( Σ_n ov(n, g)·conf_n·onehot(pc_n) + 0.05·(1 − Σ_n ov(n, g))/12 )

``method="pyin"`` swaps the pitch estimate for ``librosa.pyin`` (fmin 40 Hz, frame 2048, hop 256)
with the same segmentation. The CQT method passes the §5.6 acceptance test on synthetic bass lines
(see engine/tests/test_bass.py), so it is the default.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..types import BassTrack, NoteEvents, RhythmGrid, Stems

SR = 22050
HOP = 256                         # 11.6 ms
FMIN_MIDI = 28                    # E1, 41.2 Hz
N_BINS = 72                       # 6 octaves at 12 bins/octave
F0_BINS = 55 - FMIN_MIDI + 1      # f0 ∈ [E1, G3]
HARMONICS = ((0, 1.0), (12, 0.8), (19, 0.64), (24, 0.512))   # (round(12·log2 h), 0.8^(h−1))
FILTER_SCALE = 0.5
SAL_TEMP = 0.5
ATTACK_SKIP_S = 0.03
MIN_NOTE_S = 0.06
# Onset latency (detected − true), measured on 60 synthetic bass lines like those of test_bass.py
# (sawtooth E1–G3, 16ths at 65 BPM and 8ths at 130 BPM; detached, legato and staccato): mean
# −25.9 ms (−21 detached … −29 staccato; backtracking lands early). Subtracted from every onset.
ONSET_LATENCY_S = -0.026
MAX_BACKTRACK = 3                 # frames (35 ms)
ONSET_DELTA = 0.03                # librosa's 0.07 misses legato pitch changes (recall 0.61 → 0.98)
SILENCE_DBFS = -55.0              # absolute floor for a sounding note
SILENCE_REL_DB = 35.0             # ... and at most this far below the track's loud level (95th pct)
DECAY_DB = 25.0                   # a note ends where it falls this far below its own peak
P_FLOOR = 0.05
AMT_GATE_DBFS = -45.0             # M5: rolling 1 s RMS of the bass stem below this → piano left hand
AMT_SPLIT_MIDI = 55
LH_MAX_HOLD_S = 3.0               # a pedalled left-hand note stops being the bass after this, once a newer one is struck
DEFAULT_METHOD: Literal["cqt", "pyin"] = "cqt"


def _load(path) -> np.ndarray:
    from ..rhythm.beats import load_mono
    y, _ = load_mono(path, SR)
    return y


def energy_db(x: np.ndarray, n_frames: int, frame_length: int = 2048) -> np.ndarray:
    """Frame RMS in dBFS (hop 256, centred). BassTrack.energy_db uses 2048-sample frames; the note
    segmentation uses 512 so that the next attack does not leak into the end of a rest."""
    import librosa
    rms = librosa.feature.rms(y=x, frame_length=frame_length, hop_length=HOP, center=True)[0]
    return _fit(20 * np.log10(np.maximum(rms, 1e-10)), n_frames, -200.0)


def _fit(a: np.ndarray, n: int, fill: float) -> np.ndarray:
    if len(a) >= n:
        return a[:n]
    return np.concatenate([a, np.full(n - len(a), fill)])


def cqt_salience(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S [T, F0_BINS] harmonic-summed magnitude, sal [T, 12] pitch-class salience)."""
    import librosa
    C = np.abs(librosa.cqt(x, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(FMIN_MIDI), n_bins=N_BINS,
                           bins_per_octave=12, filter_scale=FILTER_SCALE)).T          # [T, 72]
    S = np.zeros((C.shape[0], F0_BINS))
    for off, w in HARMONICS:
        S += w * C[:, off:off + F0_BINS]
    return S, _pc_softmax(S)


def _pc_softmax(S: np.ndarray) -> np.ndarray:
    pcs = (np.arange(S.shape[1]) + FMIN_MIDI) % 12
    M = np.full((S.shape[0], 12), 0.0)
    for pc in range(12):
        M[:, pc] = S[:, pcs == pc].max(axis=1)
    M = M / np.maximum(M.max(axis=1, keepdims=True), 1e-12)
    z = np.exp((M - 1.0) / SAL_TEMP)
    return z / z.sum(axis=1, keepdims=True)


def pyin_salience(x: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """(S [T, F0_BINS] one-hot voiced pitch × voicing prob, sal [T, 12]) from librosa.pyin."""
    import librosa
    f0, _, vprob = librosa.pyin(x, fmin=40.0, fmax=float(librosa.midi_to_hz(57)), sr=SR,
                                frame_length=2048, hop_length=HOP, center=True)
    f0, vprob = _fit(f0, n_frames, np.nan), _fit(np.nan_to_num(vprob), n_frames, 0.0)
    S = np.zeros((n_frames, F0_BINS))
    ok = np.isfinite(f0)
    b = np.clip(np.rint(librosa.hz_to_midi(np.where(ok, f0, 100.0))).astype(int) - FMIN_MIDI, 0, F0_BINS - 1)
    S[np.flatnonzero(ok), b[ok]] = vprob[ok]
    sal = np.full((n_frames, 12), 1 / 12)
    pc = (b + FMIN_MIDI) % 12
    sal[ok] = (1 - vprob[ok, None]) / 12
    sal[np.flatnonzero(ok), pc[ok]] += vprob[ok]
    return S, sal


def detect_onsets(x: np.ndarray) -> np.ndarray:
    """Backtracked onset times (s), latency-compensated."""
    import librosa
    env = librosa.onset.onset_strength(y=x, sr=SR, hop_length=HOP, n_fft=1024, center=False)
    peak = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, backtrack=False,
                                      units="frames", delta=ONSET_DELTA)
    fr = librosa.onset.onset_backtrack(peak, env)
    fr = np.maximum(fr, peak - MAX_BACKTRACK)     # in silence the envelope is flat: don't run back into it
    return np.maximum(0.0, fr * HOP / SR - ONSET_LATENCY_S)


def segment_notes(onsets: np.ndarray, frame_t: np.ndarray, S: np.ndarray, sal: np.ndarray,
                  edb: np.ndarray, end_t: float, source: str = "bass_stem") -> list[tuple[float, float, int, float, str]]:
    """Notes between onsets (§5.6): rests where silent, short segments merged, decays trimmed."""
    loud = float(np.percentile(edb, 95)) if len(edb) else -200.0
    floor = max(SILENCE_DBFS, loud - SILENCE_REL_DB)
    on = sorted(set(float(t) for t in onsets if 0 <= t < end_t))
    first = int(np.searchsorted(frame_t, on[0])) if on else len(frame_t)
    if (not on or on[0] > MIN_NOTE_S) and first > 0 and edb[:first].max() > floor:
        on.insert(0, 0.0)                 # the line starts before the first detected onset
    bounds = on + [end_t]
    segs = [[a, b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    merged: list[list[float]] = []
    for a, b in segs:                     # < 60 ms → into the previous note
        if merged and b - a < MIN_NOTE_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if len(merged) > 1 and merged[0][1] - merged[0][0] < MIN_NOTE_S:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    pcs = (np.arange(S.shape[1]) + FMIN_MIDI) % 12
    notes = []
    for a, b in merged:
        i0, i1 = np.searchsorted(frame_t, [a, b])
        if i1 <= i0 or edb[i0:i1].max() < floor:
            continue                      # a rest
        seg_db = edb[i0:i1]
        peak = int(np.argmax(seg_db))
        dead = np.flatnonzero(seg_db[peak:] < max(floor, seg_db[peak] - DECAY_DB))
        if len(dead):                     # first decay below the threshold (the next attack leaks into
            i1 = i0 + peak + int(dead[0])  # the last centred RMS frames, so not the last loud frame)
            b = min(b, float(frame_t[i1 - 1]) + HOP / SR)
        if b - a < MIN_NOTE_S:
            continue                      # only a tail or a blip sounded
        j0 = min(int(np.searchsorted(frame_t, a + ATTACK_SKIP_S)), i1 - 1)
        m = sal[j0:i1].mean(axis=0)
        pc = int(np.argmax(m))
        conf = float(sal[j0:i1].max(axis=1).mean())
        sm = S[j0:i1].mean(axis=0)
        cand = np.flatnonzero(pcs == pc)
        midi = FMIN_MIDI + int(cand[np.argmax(sm[cand])])
        notes.append((float(a), float(b), midi, conf, source))
    return notes


def track_signal(x: np.ndarray, sr: int = SR, method: Literal["cqt", "pyin"] | None = None) -> BassTrack:
    """Bass track of a mono signal (resampled to 22.05 kHz)."""
    import librosa
    x = np.asarray(x, dtype=np.float32)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    method = method or DEFAULT_METHOD
    if len(x) < 4096:
        x = np.pad(x, (0, 4096 - len(x)))
    S, sal = cqt_salience(x)
    T = S.shape[0]
    if method == "pyin":
        S, sal = pyin_salience(x, T)
    frame_t = np.arange(T) * HOP / SR
    edb = energy_db(x, T)
    notes = segment_notes(detect_onsets(x), frame_t, S, sal, energy_db(x, T, 512), len(x) / SR)
    return BassTrack(frame_t=frame_t, salience=sal, energy_db=edb, notes=notes)


def track(stems: Stems, grid: RhythmGrid, amt: NoteEvents | None = None,
          method: Literal["cqt", "pyin"] | None = None) -> BassTrack:
    """§5.6 bass track of the bass stem; with ``amt`` (M5) the piano left hand fills quiet bass."""
    bt = track_signal(_load(stems.bass), SR, method)
    if amt is not None and len(amt.onset):
        bt.notes = _merge_left_hand(bt, amt)
    return bt


def with_left_hand(bt: BassTrack, amt: NoteEvents | None) -> BassTrack:
    """A copy of the stem's bass track with the §5.6 left-hand fallback applied (``bt`` is not
    changed). The pipeline keeps the stem-only track in its decode state and applies this at decode
    time, so ``DecoderParams.amt_bass`` can switch the fallback on and off without re-tracking."""
    if amt is None or not len(amt.onset):
        return bt
    return BassTrack(frame_t=bt.frame_t, salience=bt.salience, energy_db=bt.energy_db,
                     notes=_merge_left_hand(bt, amt))


def quiet_frames(bt: BassTrack) -> np.ndarray:
    """[T] True where the rolling 1 s RMS of the bass stem is below −45 dBFS (§5.6)."""
    win = max(1, int(round(1.0 / (HOP / SR))))
    e = np.nan_to_num(np.asarray(bt.energy_db, dtype=float), nan=-200.0, neginf=-200.0)
    p = 10 ** (e / 10)
    roll = 10 * np.log10(np.maximum(np.convolve(p, np.ones(win) / win, mode="same"), 1e-20))
    return roll < AMT_GATE_DBFS


def _runs(mask: np.ndarray, hop: float) -> list[tuple[float, float]]:
    """[t0, t1) runs of True frames (frame k covers [k·hop, (k+1)·hop)); the last run is open."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = [(s * hop, e * hop) for s, e in zip(starts, ends)]
    if m[-1]:
        out[-1] = (out[-1][0], np.inf)    # the stem is silent past its end
    return out


def left_hand_line(amt: NoteEvents, split: int = AMT_SPLIT_MIDI,
                   max_hold_s: float = LH_MAX_HOLD_S) -> list[tuple[float, float, int, float]]:
    """The left hand's bass line: at every moment the LOWEST sounding piano note below ``split``
    (sounding = from its onset to its pedal-extended offset). A note struck above a held root
    (arpeggio fills, octaves, fifths) does not move the bass; a lower note or the release of the
    root does. A held note stops counting at the first left-hand strike after it is ``max_hold_s``
    old (a long pedal held across a chord change).

    Returns (t0, t1, midi, conf) with conf = clip(velocity / 127, 0.3, 1)."""
    pitch = np.asarray(amt.pitch).astype(int)
    lh = np.flatnonzero(pitch < split)
    if not len(lh):
        return []
    on = np.asarray(amt.onset, dtype=float)[lh]
    off = np.maximum(np.asarray(amt.offset_pedal, dtype=float)[lh], on)
    pit = pitch[lh]
    vel = np.asarray(amt.velocity, dtype=float)[lh]
    order = np.lexsort((pit, on))
    on, off, pit, vel = on[order], off[order], pit[order], vel[order]
    # a note older than max_hold_s ends at the next left-hand strike after that age
    off_eff = off.copy()
    for i in range(len(on)):
        later = np.flatnonzero(on > on[i] + 1e-9)
        if len(later):
            j = later[np.searchsorted(on[later], on[i] + max_hold_s, side="left"):]
            if len(j):
                off_eff[i] = min(off_eff[i], max(on[j[0]], on[i] + max_hold_s))
    times = np.unique(np.r_[on, off_eff])
    line: list[list] = []                                   # [t0, t1, note index]
    for t0, t1 in zip(times[:-1], times[1:]):
        mid = 0.5 * (t0 + t1)
        snd = np.flatnonzero((on <= mid) & (off_eff > mid))
        if not len(snd):
            continue
        k = int(snd[np.lexsort((-on[snd], pit[snd]))[0]])   # lowest; the newest strike on a tie
        if line and line[-1][2] == k and abs(line[-1][1] - t0) < 1e-9:
            line[-1][1] = float(t1)
        else:
            line.append([float(t0), float(t1), k])
    return [(a, b, int(pit[k]), float(np.clip(vel[k] / 127.0, 0.3, 1.0))) for a, b, k in line]


def _merge_left_hand(bt: BassTrack, amt: NoteEvents) -> list[tuple[float, float, int, float, str]]:
    """§5.6: where the rolling 1 s RMS of the bass stem is < −45 dBFS, the bass line is the
    piano's left hand (AMT notes below MIDI 55, ``left_hand_line``) and stem notes starting there
    are dropped (they are bleed: kick, riser, piano rumble)."""
    hop = HOP / SR
    quiet = quiet_frames(bt)
    if not len(quiet):
        return bt.notes
    runs = _runs(quiet, hop)
    extra = []
    for t0, t1, midi, conf in left_hand_line(amt):
        for q0, q1 in runs:                               # only inside the quiet stretches
            a, b = max(t0, q0), min(t1, q1)
            if b - a >= MIN_NOTE_S:
                extra.append((float(a), float(b), int(midi), conf, "amt_lh"))
    if not runs:
        return bt.notes
    keep = [n for n in bt.notes if not quiet[min(int(n[0] / hop), len(quiet) - 1)]]
    return sorted(keep + extra)


def cell_probs(bass: BassTrack, grid: RhythmGrid) -> np.ndarray:
    """P_bass [G, 12] (§5.6): overlap-weighted note pitch classes with a 0.05 uniform floor."""
    ct = np.asarray(grid.cell_t, dtype=float)
    G = len(ct) - 1
    a, b = ct[:-1], ct[1:]
    dur = np.maximum(b - a, 1e-9)
    P = np.zeros((G, 12))
    cover = np.zeros(G)
    for t0, t1, midi, conf, _ in bass.notes:
        g0 = max(0, int(np.searchsorted(ct, t0, side="right")) - 1)
        g1 = min(G, int(np.searchsorted(ct, t1, side="left")))
        if g1 <= g0:
            continue
        ov = np.clip((np.minimum(b[g0:g1], t1) - np.maximum(a[g0:g1], t0)) / dur[g0:g1], 0.0, 1.0)
        P[g0:g1, int(midi) % 12] += ov * conf
        cover[g0:g1] += ov
    P += (P_FLOOR * np.clip(1.0 - cover, 0.0, 1.0) / 12)[:, None]
    s = P.sum(axis=1, keepdims=True)
    P = np.where(s > 0, P / np.maximum(s, 1e-12), 1 / 12)
    return P
