"""Bars, subdivision and the sub-beat cell grid (DESIGN §5.3), plus user overrides.

``build_grid`` is pure NumPy: it takes beat lists (and optionally the tracker's per-beat log-odds
and an onset envelope of the harmonic stem) and returns a ``RhythmGrid``.

Conventions (resolved where §4/§5.3 are silent):
- The grid covers [0, duration]. Time before the first beat and after the end of the last beat
  (the last beat lasts one more IBI) is free time when ≥ 0.1 s, and is listed in ``free``;
  shorter edges are absorbed by the first / last cell.
- Beat i owns the interval [beats[i], beats[i+1]). A free region spans whole beat intervals; its
  beats have bar_of_beat −1 and beat_in_bar 0. ``cell_pos`` 5 ⇔ the cell lies in a free region.
- Pickup beats (before the first downbeat of a metered run) have bar −1 and their position in a
  virtual bar of the main length (a one-beat pickup in 4/4 is beat 4).
- Meter labels: (n, 4) for an n-beat bar; (3n, 8) for compound bars; a ternary 2-beat bar is 6/8,
  and a ternary 4-beat bar with beats ≥ 0.75 s is 12/8 (slow gospel triplet feel). Swing at
  faster tempos stays 4/4 with 3 cells per beat.
- Triplet test per 8-bar block: r3 = mean onset strength at 1/3, 2/3 of each beat ÷ mean at
  1/4, 1/2, 3/4, sampled with ±min(25 ms, ibi/24) so the 1/4 and 1/3 windows never overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import RhythmGrid, RhythmOverrides
from . import meter
from .meter import BeatSeq, TempoInfo

ONSET_SR = 22050
ONSET_HOP = 256
FREE_CELL_S = 0.2
FREE_SNAP_S = 0.06
MIN_FREE_CELL_S = 0.05
EDGE_MIN_S = 0.1
TRIPLET_R3 = 1.2
BLOCK_BARS = 8
ALLOWED_SUBDIV = (2, 3, 4, 6)


@dataclass
class OnsetEnvelope:
    """Onset strength of the harmonic stem (frame i is centred at i / fps)."""
    env: np.ndarray
    fps: float

    def at(self, t: np.ndarray, tol: float | np.ndarray = 0.025) -> np.ndarray:
        """Max onset strength within ±tol of each time."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        tol = np.broadcast_to(np.asarray(tol, dtype=float), t.shape)
        n = len(self.env)
        lo = np.clip(np.floor((t - tol) * self.fps).astype(int), 0, n - 1)
        hi = np.clip(np.ceil((t + tol) * self.fps).astype(int), 0, n - 1)
        return np.array([self.env[a:b + 1].max() if b >= a else 0.0 for a, b in zip(lo, hi)])

    def strongest(self, t: np.ndarray, tol: float = FREE_SNAP_S) -> np.ndarray:
        """Time of the strongest onset within ±tol of each time (t itself if the window is flat)."""
        out = np.array(t, dtype=float)
        n = len(self.env)
        for i, x in enumerate(out):
            a = max(0, int(np.floor((x - tol) * self.fps)))
            b = min(n - 1, int(np.ceil((x + tol) * self.fps)))
            if b > a and self.env[a:b + 1].max() > self.env[a:b + 1].min():
                out[i] = (a + int(np.argmax(self.env[a:b + 1]))) / self.fps
        return out


def onset_envelope(y: np.ndarray, sr: int) -> OnsetEnvelope:
    """librosa onset strength at 22.05 kHz, hop 256 (11.6 ms), n_fft 1024, aligned to the onsets."""
    import librosa
    if sr != ONSET_SR:
        y = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=ONSET_SR)
    # center=False: librosa's own centring shift adds n_fft/2 of lag on top of the centred STFT
    # (≈ 40 ms peak lag measured); without it the peaks sit within ±10 ms of the true onsets.
    env = librosa.onset.onset_strength(y=np.asarray(y, dtype=np.float32), sr=ONSET_SR, hop_length=ONSET_HOP,
                                       n_fft=1024, center=False)
    return OnsetEnvelope(env.astype(float), ONSET_SR / ONSET_HOP)


def triplet_ratio(onset: OnsetEnvelope, t: np.ndarray, beat_idx: np.ndarray) -> float:
    """r3 of §5.3 over the given beats (each with a following beat). 1.0 when there is no signal."""
    beat_idx = np.asarray([i for i in beat_idx if i + 1 < len(t)], dtype=int)
    if len(beat_idx) == 0:
        return 1.0
    b0, ibi = t[beat_idx], t[beat_idx + 1] - t[beat_idx]
    tol = np.minimum(0.025, ibi / 24)

    def mean_at(fracs):
        return float(np.mean([onset.at(b0 + f * ibi, tol) for f in fracs]))
    num, den = mean_at((1 / 3, 2 / 3)), mean_at((0.25, 0.5, 0.75))
    floor = 0.05 * float(np.percentile(onset.env, 95)) + 1e-9
    if num < floor and den < floor:
        return 1.0
    return num / max(den, floor)


def subdiv_for(ternary: bool, ibi: float) -> int:
    """§5.3: (6 if ibi ≥ 0.9 s else 3) if ternary else (4 if ibi ≥ 0.75 s else 2)."""
    if ternary:
        return 6 if ibi >= 0.9 else 3
    return 4 if ibi >= 0.75 else 2


def _free_cell_bounds(a: float, b: float, onset: OnsetEnvelope | None, source: np.ndarray | None) -> np.ndarray:
    """Interior boundaries of a free region: 0.2 s cells snapped to the strongest onset (±60 ms)."""
    if source is not None:
        inner = source[(source > a + 1e-6) & (source < b - 1e-6)]
        if len(inner):
            return np.asarray(inner, float)
    n = max(1, int(round((b - a) / FREE_CELL_S)))
    nominal = a + (b - a) * np.arange(1, n) / n
    if onset is not None and len(nominal):
        nominal = onset.strongest(nominal)
    out, last = [], a
    for x in np.sort(nominal):
        if x - last >= MIN_FREE_CELL_S and b - x >= MIN_FREE_CELL_S:
            out.append(x)
            last = x
    return np.asarray(out, float)


def _metered_runs(free_iv: np.ndarray, B: int) -> list[tuple[int, int]]:
    """[first, last] beat index of each metered run (a beat is metered if its interval is)."""
    metered = np.ones(B, bool)
    if B > 1:
        metered[:-1] = ~free_iv
        metered[-1] = not free_iv[-1]
    return [(int(s), int(e) - 1) for s, e in meter.runs_of(metered)]


def _complete_bars(db: list[int], run: tuple[int, int], L: int) -> list[int]:
    """Fill a pickup of ≥ L beats and an over-long last bar with bars of L (missing downbeats)."""
    a, b = run
    db = sorted(d for d in set(db) if a <= d <= b)
    if not db:
        return list(range(a, b + 1, L)) if b - a + 1 >= L else []
    while db[0] - a >= L:
        db.insert(0, db[0] - L)
    while b - db[-1] + 1 > 2 * L - 1:
        db.append(db[-1] + L)
    return db


def _forced_free_intervals(t: np.ndarray, regions: list[tuple[float, float]]) -> np.ndarray:
    if len(t) < 2:
        return np.zeros(0, bool)
    mid = (t[:-1] + t[1:]) / 2
    out = np.zeros(len(mid), bool)
    for a, b in regions:
        out |= (mid >= a) & (mid <= b)
    return out


def _apply_moved(seq: BeatSeq, moved: dict[int, float]) -> BeatSeq:
    if not moved:
        return seq
    t = seq.t.copy()
    for i, x in moved.items():
        if 0 <= int(i) < len(t):
            t[int(i)] = float(x)
    order = np.argsort(t, kind="stable")
    return BeatSeq(t[order], seq.conf[order], seq.blog[order], seq.dlog[order], seq.is_db[order])


def _tapped(seq: BeatSeq, tapped: list[float]) -> BeatSeq:
    """Tapped beats replace the tracker; tracker downbeats are snapped onto them as evidence."""
    t = np.sort(np.asarray(tapped, dtype=float))
    out = BeatSeq.make(t, np.ones(len(t)), [], None, None)
    if len(t) > 1 and seq.is_db.any():
        tol = 0.35 * float(np.median(np.diff(t)))
        for d in seq.t[seq.is_db]:
            j = int(np.argmin(np.abs(t - d)))
            if abs(t[j] - d) <= tol:
                out.is_db[j] = True
    return out


def _meter_label(n: int, compound: bool, ternary: bool, ibi: float) -> tuple[int, int]:
    if compound:
        return 3 * n, 8
    if ternary and n == 2:
        return 6, 8
    if ternary and n == 4 and ibi >= 0.75:
        return 12, 8
    return n, 4


def _beats_per_bar(num: int, den: int) -> tuple[int, bool]:
    """Beats per bar and compound flag of a user meter."""
    if den == 8 and num % 3 == 0 and num >= 6:
        return num // 3, True
    return max(1, int(num)), False


def build_grid(beats, downbeats, beat_conf, *, duration: float | None = None, beat_logit=None,
               downbeat_logit=None, onset: OnsetEnvelope | None = None,
               overrides: RhythmOverrides | None = None, harmonic_changes=None) -> RhythmGrid:
    """Beat lists (+ optional log-odds and onset envelope) → RhythmGrid (§5.3), overrides re-applied.

    ``harmonic_changes`` (HCDF peak times, s) lets a sparse intro that holds its chord inherit the
    song's grid (``meter.backfill_intro``); without it a free intro stays free."""
    ov = overrides or RhythmOverrides()
    seq = BeatSeq.make(beats, beat_conf, downbeats, beat_logit, downbeat_logit)
    tf = float(ov.tempo_factor or 1.0)
    intro = None
    if ov.tapped_beats:
        seq = _tapped(seq, ov.tapped_beats)
        seq, info = meter.normalize_tempo(seq, tf, auto=False)
    else:
        seq, info = meter.normalize_tempo(seq, tf, auto=True)
        seq, intro = meter.backfill_intro(seq, harmonic_changes)
    seq = _apply_moved(seq, ov.moved_beats)
    free_iv = meter.free_intervals(seq) if len(seq) > 1 else np.zeros(0, bool)
    if len(seq) > 1 and intro is not None:
        free_iv &= ~_forced_free_intervals(seq.t, [intro])     # a sparse intro in tempo is metered
    if len(seq) > 1 and ov.free:
        free_iv = free_iv | _forced_free_intervals(seq.t, ov.free)
    L = meter.bar_length(seq, info)
    edited = bool(ov.tapped_beats or tf != 1.0 or ov.downbeat_shift or ov.moved_beats or ov.meter
                  or ov.subdiv or ov.free)
    return grid_from_beats(seq, free_iv, L, meter.bar_evidence(seq), info=info, duration=duration,
                           onset=onset, overrides=ov, edited=edited)


def grid_from_beats(seq: BeatSeq, free_iv: np.ndarray, L: int, evidence: np.ndarray, *,
                    info: TempoInfo | None = None, duration: float | None = None,
                    onset: OnsetEnvelope | None = None, ternary_beat: np.ndarray | None = None,
                    compound_beat: np.ndarray | None = None, free_cells_source: np.ndarray | None = None,
                    overrides: RhythmOverrides | None = None, edited: bool = False) -> RhythmGrid:
    """Bars (Viterbi), downbeat shift, meter/subdiv overrides, triplet test and cells."""
    ov = overrides or RhythmOverrides()
    t = seq.t
    B = len(t)
    if B < 2:
        return _all_free_grid(t, seq.conf, duration, onset, free_cells_source, edited)
    ibi = np.diff(t)
    ibi_ref = float(np.median(ibi[~free_iv])) if (~free_iv).any() else float(np.median(ibi))
    runs = _metered_runs(free_iv, B)

    # --- downbeats per metered run --------------------------------------------------------------
    db: list[int] = []
    for a, b in runs:
        d = [a + int(i) for i in meter.viterbi_bars(evidence[a:b + 1], L)]
        if ov.downbeat_shift:
            d = [x + int(ov.downbeat_shift) for x in d]
        db += _complete_bars(d, (a, b), L)

    # --- meter overrides (from bar k onward; applied in list order) -------------------------------
    compound = np.zeros(B, bool) if compound_beat is None else np.asarray(compound_beat, bool).copy()
    if info is not None and info.compound:
        compound[:] = True
    forced = np.zeros((B, 2), int)
    for from_bar, num, den in ov.meter:
        bars = _bars_from(db, runs)
        if not (0 <= from_bar < len(bars)):
            continue
        s = bars[from_bar][0]
        n, comp = _beats_per_bar(int(num), int(den))
        db = [d for d in db if d < s]
        for a, b in runs:
            if b < s:
                continue
            db += list(range(max(a, s), b + 1, n))
        compound[s:] = comp
        forced[s:] = (num, den)
        L = n

    bars = _bars_from(db, runs)
    bar_of_beat = np.full(B, -1, int)
    beat_in_bar = np.zeros(B, int)
    bar_len_of_beat = np.full(B, L, int)
    for j, (s, n) in enumerate(bars):
        bar_of_beat[s:s + n] = j
        beat_in_bar[s:s + n] = np.arange(1, n + 1)
        bar_len_of_beat[s:s + n] = n
    for a, b in runs:                      # pickups: position in a virtual bar of length L
        first = next((s for s, _ in bars if a <= s <= b), b + 1)
        p = first - a
        if p > 0:
            beat_in_bar[a:first] = np.arange(max(1, L - p + 1), max(1, L - p + 1) + p)

    # --- sections (8-bar blocks) and the triplet test -------------------------------------------
    block = np.zeros(B, int)
    for j, (s, n) in enumerate(bars):
        block[s:s + n] = j // BLOCK_BARS
    nxt = len(bars) // BLOCK_BARS if bars else 0
    for i in range(B - 1, -1, -1):          # pickups / free beats join the block of the next bar
        if bar_of_beat[i] >= 0:
            nxt = block[i]
        else:
            block[i] = nxt
    metered_beat = beat_in_bar > 0
    subdiv = np.zeros(B, int)
    ternary = np.zeros(B, bool)
    blk_ibi = np.full(B, ibi_ref)
    for blk in np.unique(block):
        members = np.flatnonzero(block == blk)
        mb = members[metered_beat[members] & (members < B - 1)]
        local = float(np.median(ibi[mb])) if len(mb) else ibi_ref
        if compound[members].any():
            tern = True
        elif ternary_beat is not None:
            tern = bool(np.mean(np.asarray(ternary_beat)[members]) > 0.5)
        elif onset is not None and len(mb):
            tern = triplet_ratio(onset, t, mb) > TRIPLET_R3
        else:
            tern = False
        ternary[members] = tern
        blk_ibi[members] = local
        subdiv[members] = subdiv_for(tern, local)
    for from_bar, cpb in ov.subdiv:
        if 0 <= from_bar < len(bars):
            v = min(ALLOWED_SUBDIV, key=lambda a: abs(a - int(cpb)))
            subdiv[bars[from_bar][0]:] = v

    # --- meter list -------------------------------------------------------------------------------
    meter_list: list[tuple[int, int, int]] = []
    run_last = {max(s for s, _ in bars if a <= s <= b) for a, b in runs if any(a <= s <= b for s, _ in bars)}
    for j, (s, n) in enumerate(bars):
        lab = tuple(forced[s]) if forced[s, 0] else _meter_label(n, bool(compound[s]), bool(ternary[s]), blk_ibi[s])
        if meter_list and s in run_last and n < L and not forced[s, 0]:
            continue                        # a bar cut short by free time or the song end is not a meter change
        if not meter_list or meter_list[-1][1:] != lab:
            meter_list.append((j, int(lab[0]), int(lab[1])))

    # --- cells ------------------------------------------------------------------------------------
    cell_t, cell_beat, cell_pos, free = _cells(t, free_iv, subdiv, beat_in_bar, bar_of_beat, bar_len_of_beat,
                                               L, duration, onset, free_cells_source)
    return RhythmGrid(beats=t.copy(), beat_conf=seq.conf.copy(), bar_of_beat=bar_of_beat,
                      beat_in_bar=beat_in_bar, bars=[(int(s), int(n)) for s, n in bars], meter=meter_list,
                      bpm=60.0 / ibi_ref, ibi_ref=ibi_ref, subdiv=subdiv, free=free, cell_t=cell_t,
                      cell_beat=cell_beat, cell_pos=cell_pos, edited=edited)


def _bars_from(db: list[int], runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    bars = []
    for a, b in runs:
        d = sorted(x for x in set(db) if a <= x <= b)
        for x, y in zip(d, d[1:] + [b + 1]):
            bars.append((x, y - x))
    return bars


def _beat_pos(i: int, beat_in_bar: np.ndarray, bar_of_beat: np.ndarray, bar_len: np.ndarray) -> int:
    k, n = beat_in_bar[i], bar_len[i]
    if k == 1 and bar_of_beat[i] >= 0:
        return 0
    if n % 2 == 0 and k == n // 2 + 1:
        return 1
    return 2


def _sub_pos(j: int, s: int) -> int:
    """cell_pos of sub-cell j > 0 of a beat split into s cells."""
    if s in (3, 6):
        return 3 if (j * 3) % s == 0 else 4
    return 3 if 2 * j == s else 4


def _cells(t, free_iv, subdiv, beat_in_bar, bar_of_beat, bar_len, L, duration, onset, source):
    B = len(t)
    ibi = np.diff(t)
    last_len = float(ibi[-1])
    end_metered = t[-1] + last_len
    if duration is not None:
        end_metered = min(end_metered, float(duration))
    last_ok = (not free_iv[-1]) and end_metered - t[-1] >= 0.3 * last_len
    song_end = float(duration) if duration is not None else (end_metered if last_ok else t[-1])

    # pieces: ("beat", i, a, b) | ("free", a, b, start_beat_idx or None)
    pieces: list[tuple] = []

    def add_free(a, b, start_idx):
        if b - a <= 1e-9:
            return
        if pieces and pieces[-1][0] == "free" and abs(pieces[-1][2] - a) < 1e-9:
            pieces[-1] = ("free", pieces[-1][1], b, pieces[-1][3])
        else:
            pieces.append(("free", a, b, start_idx))

    if t[0] >= EDGE_MIN_S:
        add_free(0.0, t[0], None)
    for i in range(B - 1):
        if free_iv[i]:
            add_free(t[i], t[i + 1], i)
        else:
            pieces.append(("beat", i, t[i], t[i + 1]))
    if last_ok:
        pieces.append(("beat", B - 1, t[-1], end_metered))
        tail_start = end_metered
    else:
        tail_start = t[-1]
    if song_end - tail_start >= EDGE_MIN_S or (pieces and pieces[-1][0] == "free" and song_end > tail_start):
        add_free(tail_start, song_end, B - 1 if not last_ok else None)

    ct, cb, cp, free = [], [], [], []
    for p in pieces:
        if p[0] == "beat":
            _, i, a, b = p
            s = int(subdiv[i])
            for j in range(s):
                ct.append(a + (b - a) * j / s)
                cb.append(i + j / s)
                cp.append(_beat_pos(i, beat_in_bar, bar_of_beat, bar_len) if j == 0 else _sub_pos(j, s))
        else:
            _, a, b, si = p
            free.append((float(a), float(b)))
            inner = _free_cell_bounds(a, b, onset, source)
            ct.append(a)
            cb.append(float(si) if si is not None and abs(t[si] - a) < 1e-9 else np.nan)
            cp.append(5)
            for x in inner:
                ct.append(float(x))
                cb.append(np.nan)
                cp.append(5)
    last = pieces[-1]
    end = last[3] if last[0] == "beat" else last[2]
    ct.append(max(end, song_end))                         # absorb a tiny tail
    if 0.0 < ct[0] < EDGE_MIN_S:
        ct[0] = 0.0                                       # ... and a tiny pre-roll
    cb.append(last[1] + 1.0 if last[0] == "beat" else np.nan)
    return (np.asarray(ct, float), np.asarray(cb, float), np.asarray(cp, np.int8), free)


def _all_free_grid(t, conf, duration, onset, source, edited) -> RhythmGrid:
    """No usable beats: the whole song is one free-time region."""
    end = float(duration) if duration is not None else (float(t[-1]) + 1.0 if len(t) else 1.0)
    inner = _free_cell_bounds(0.0, end, onset, source)
    ct = np.concatenate([[0.0], inner, [end]])
    B = len(t)
    ibi_ref = 0.6                         # free-time "beats" are groups of 3 cells (§5.10)
    return RhythmGrid(beats=np.asarray(t, float), beat_conf=np.asarray(conf, float), bar_of_beat=np.full(B, -1),
                      beat_in_bar=np.zeros(B, int), bars=[], meter=[], bpm=60.0 / ibi_ref, ibi_ref=ibi_ref,
                      subdiv=np.full(B, 4), free=[(0.0, end)], cell_t=ct, cell_beat=np.full(len(ct), np.nan),
                      cell_pos=np.full(len(ct) - 1, 5, np.int8), edited=edited)
