"""Tempo octave, bars and free time (DESIGN §5.3).

Four steps, all on beat lists (no audio):

1. ``normalize_tempo`` picks one beat period P for the song and brings every beat to that level.
   Beat This! can switch metrical level inside a song: on 回家 it tracks the 65 BPM intro and a
   quiet middle part at the quarter note but most of the song at the eighth note (130 BPM), with
   "downbeats" every 4 eighths. So the §5.3 octave rule is applied with these refinements:
   - P_dom = the dominant inter-beat interval. Fold to 2·P_dom when P_dom is ≥ 110 BPM, the
     halved tempo stays ≥ 45 BPM, and either the tracked bars have 8 beats (the §5.3 rule), or
     the beat logits show an on-beat/off-beat hierarchy (alternate beats differ by ≥ 2 logits),
     or, without logits, part of the song is already tracked at 2·P_dom. Below 45 BPM the tempo
     is doubled.
   - Compound meter (6/8, 12/8; beat = dotted quarter): bars of 6/12 tracked beats at ≥ 110 BPM
     (beat = 3 tracked beats), or a slower P_dom with ≥ 12 tracked intervals at ⅓ or ⅔ of it and
     (almost) none at ½ (活出愛: the tracker follows eighths for 30 s, then dotted quarters).
   - Every steady stretch is then put on a unit grid of u = P/m (m = 2 half beats, or 3 eighths
     for compound meter). Beats at units ≢ 0 (mod m) are sub-beats: an m-state Viterbi chooses the
     phase from the beat logits and tracker downbeats, with beats already at the beat level forced
     on, and drops the sub-beats. Gaps of 2–4 beats inside a steady stretch are filled (missed
     beats); with P halved this also doubles the tempo. Near-duplicate beats (< u/2 apart, a
     tracker glitch) are merged first. So the whole song sits at one tempo octave.
2. ``bar_length`` + ``viterbi_bars`` place the downbeats: a Viterbi over "beats since the last
   downbeat" scored by the downbeat log-odds. Bars of the main length L are free, any other
   length costs BAR_PENALTY, so real mixed-meter bars survive and spurious downbeats do not. When
   tracked bars come out at 2 beats and consecutive downbeat logits alternate strong/weak (the
   half-bar downbeats of 回家), L = 4 and the Viterbi keeps the strong phase.
3. ``free_intervals``: sliding 8-beat windows, free when mean(beat_conf) < 0.5 or the IBI spread
   > 0.15 (min of std/mean and the robust 1.4826·MAD/median, so one tracking glitch does not open
   a free region); windows are centred on the interval; edges grow over beats with conf < 0.75
   (Beat This! tends to anticipate the groove by a beat at the end of a rubato intro). Free time
   is for rubato only: once the tracker has been sure of the beat, an interval inside a run of
   ≥ 8 steady intervals (each within ±8 % of the local median beat) is never free, however unsure
   the tracker gets (活出愛's soft verses, logits 2–5), and edges grow at most 4 beats into such a
   run. Before that first sure stretch (an intro) low confidence alone still means free time:
   Beat This! extrapolates a steady but unsure pulse through a rubato intro as well.
4. ``backfill_intro``: a leading free stretch whose tracked beats mostly (≥ 60 %) sit 1–4 song
   beats apart (a sparse intro is often tracked at half tempo) and whose harmony is held (≤ 0.35 harmonic changes per beat) is a sparse intro, not
   rubato (Here I Bow: a held D pad for 4 bars, beats with logits ≈ 1 and two gaps). Its beats are
   replaced by the grid of the first stable run extended backwards (snapping to tracked beats
   within ±10 % of a beat), and the stretch is metered.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FOLD_MIN_BPM = 110.0      # fold (or compound) only when the tracked level is at least this fast
MIN_BPM = 45.0            # double below this; never fold below it
HALF_BAR_MAX_BPM = 90.0   # 2-beat bars below this tempo are half bars after a fold (no logits)
HIER_MIN = 2.0            # median logit contrast of alternate beats that makes them eighths
HIER_CONSIST = 0.7        # ... with this share of beat pairs agreeing
SWITCH_COST = 8.0         # parity Viterbi: cost of breaking the on/off alternation
DB_ON_BONUS = 2.0         # parity Viterbi: a tracker downbeat is an on-beat
BAR_PENALTY = 8.0         # bar Viterbi: cost of a bar whose length is not L (> a half-bar logit, ≈ 5–7.5)
ALT_MIN = 2.5             # median |Δ downbeat logit| of consecutive downbeats → half-bar downbeats
UNIT_TOL = 0.3            # |ibi/u − k| tolerance on the half-beat unit grid
FREE_WINDOW = 8
FREE_CONF = 0.5
FREE_CV = 0.15
FREE_EDGE_CONF = 0.75     # free regions grow over adjacent beats that are not clearly confident
NO_LOGIT_DB, NO_LOGIT_OTHER = 6.0, -3.0   # bar evidence without downbeat logits
UNKNOWN_DLOG = -6.0       # inserted beats: "probably not a downbeat"
SUB_TOL3, SUB_TOL2 = 0.15, 0.12   # relative tolerance of a ⅓ / ⅔ / ½ sub-beat interval
COMPOUND_SUB_MIN = 12     # tracked ⅓/⅔-beat intervals that make a < 110 BPM song compound ...
COMPOUND_SUB_SHARE = 0.05  # ... and at least this share of all intervals ...
COMPOUND_SUB_RATIO = 3.0   # ... and this many times the ½-beat intervals
DEDUP_UNITS = 0.5         # beats closer than this many units are one beat (a tracker glitch)
STEADY_TOL = 0.08         # a steady interval is within ±8 % of the local median beat ...
STEADY_MIN = 8            # ... and a run of this many steady intervals is in tempo (never free)
STEADY_BAND = (0.7, 1.4)  # local median / song beat for a run to count as the song's tempo
FREE_EDGE_MAX = 4         # free regions grow at most this many beats into a steady run
BACKFILL_MIN_RUN = 8      # intervals of the stable run an intro is extrapolated from
BACKFILL_SHARE = 0.6      # share of the intro's tracked intervals that are 1–4 song beats ...
BACKFILL_TOL = 0.1        # ... within ±10 % of a beat
BACKFILL_SNAP = 0.1       # backfilled beats snap to tracked beats within ±this many beats
BACKFILL_CONF = 0.5       # confidence of an inferred intro beat
BACKFILL_MAX_CHANGES = 0.35  # harmonic changes per beat an intro may have and still be held


@dataclass
class BeatSeq:
    """Beats with their per-beat evidence (NaN logits = unknown)."""
    t: np.ndarray
    conf: np.ndarray
    blog: np.ndarray
    dlog: np.ndarray
    is_db: np.ndarray

    @classmethod
    def make(cls, beats, conf, downbeats=(), blog=None, dlog=None, tol: float = 0.07) -> "BeatSeq":
        t = np.asarray(beats, dtype=float)
        order = np.argsort(t, kind="stable")
        t = t[order]
        n = len(t)

        def arr(x, fill):
            return np.full(n, fill) if x is None else np.asarray(x, dtype=float)[order]
        is_db = np.zeros(n, bool)
        for d in np.asarray(downbeats, dtype=float):
            if n:
                j = int(np.argmin(np.abs(t - d)))
                if abs(t[j] - d) <= tol:
                    is_db[j] = True
        conf = np.clip(arr(conf, 1.0), 0.0, 1.0)
        return cls(t, conf, arr(blog, np.nan), arr(dlog, np.nan), is_db)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def has_blog(self) -> bool:
        return bool(np.isfinite(self.blog).any())

    @property
    def has_dlog(self) -> bool:
        return bool(np.isfinite(self.dlog).any())

    def take(self, idx) -> "BeatSeq":
        return BeatSeq(self.t[idx], self.conf[idx], self.blog[idx], self.dlog[idx], self.is_db[idx])

    def with_inserted(self, times: list[float], conf: list[float]) -> "BeatSeq":
        """Add inferred beats (unknown logits, not downbeats) and re-sort."""
        if not times:
            return self
        k = len(times)
        t = np.concatenate([self.t, times])
        order = np.argsort(t, kind="stable")
        cat = lambda a, b: np.concatenate([a, b])[order]  # noqa: E731
        return BeatSeq(t[order], cat(self.conf, np.asarray(conf, float)), cat(self.blog, np.full(k, np.nan)),
                       cat(self.dlog, np.full(k, np.nan)), cat(self.is_db, np.zeros(k, bool)))


@dataclass
class TempoInfo:
    p_dom: float              # dominant tracked inter-beat interval (s)
    period: float             # chosen beat period (s)
    compound: bool            # beats are dotted quarters made from 3 tracked beats
    folded: bool              # beats were dropped by the octave rule
    doubled: bool             # beats were inserted by the octave rule (< 45 BPM or tempo_factor 2)
    tracked_bpb: int | None
    hierarchy: float | None   # logit contrast of alternate beats (None without logits)

    @property
    def units(self) -> int:
        """Sub-beat units per beat of the unit grid: 3 eighths (compound) or 2 half beats."""
        return 3 if self.compound else 2


# ------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------

def mode_int(x) -> int | None:
    x = np.asarray(x, dtype=int)
    x = x[x > 0]
    return int(np.bincount(x).argmax()) if len(x) else None


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def dominant_period(ibi: np.ndarray) -> float:
    """Mode of the inter-beat intervals on a log scale (±0.1 octave), refined by a median."""
    ibi = np.asarray(ibi, dtype=float)
    ibi = ibi[ibi > 0.08]
    if len(ibi) == 0:
        return 0.5
    x = np.log2(ibi)
    if len(x) > 3000:
        x = x[:: int(np.ceil(len(x) / 3000))]
    counts = (np.abs(x[:, None] - x[None, :]) < 0.1).sum(axis=1)
    c = x[int(np.argmax(counts))]
    return float(2 ** np.median(x[np.abs(x - c) < 0.1]))


def tracked_bpb(seq: BeatSeq) -> int | None:
    idx = np.flatnonzero(seq.is_db)
    d = np.diff(idx)
    return mode_int(d[(d >= 1) & (d <= 16)])


def robust_cv(iv: np.ndarray) -> float:
    """min(std/mean, 1.4826·MAD/median): sustained irregularity, not one glitch."""
    iv = np.asarray(iv, dtype=float)
    if len(iv) < 2:
        return 0.0
    cv = float(np.std(iv) / max(np.mean(iv), 1e-9))
    med = float(np.median(iv))
    mad = 1.4826 * float(np.median(np.abs(iv - med))) / max(med, 1e-9)
    return min(cv, mad)


# ------------------------------------------------------------------------------------------------
# 1. tempo octave
# ------------------------------------------------------------------------------------------------

def hierarchy(seq: BeatSeq, period: float) -> tuple[float, float] | None:
    """Logit contrast of alternate beats in steady runs at ``period`` (on-beat vs off-beat eighths).

    Returns (median contrast, share of pairs agreeing), pooled over runs weighted by length, or
    None without beat logits.
    """
    if not seq.has_blog or len(seq) < 5:
        return None
    r = np.diff(seq.t) / period
    contrast, consist, weight = [], [], []
    for s, e in runs_of(np.abs(r - 1) < 0.2):
        if e - s < 4:
            continue
        lg = seq.blog[s:e + 1]
        if not np.isfinite(lg).all():
            continue
        x = (lg[:-1] - lg[1:]) * np.where(np.arange(e - s) % 2 == 0, 1.0, -1.0)
        c = float(np.median(x))
        contrast.append(abs(c))
        consist.append(float(np.mean(np.sign(x) == np.sign(c))) if c != 0 else 0.0)
        weight.append(e - s)
    if not weight:
        return None
    w = np.asarray(weight, float)
    return float(np.average(contrast, weights=w)), float(np.average(consist, weights=w))


def subbeat_counts(ibi: np.ndarray, period: float) -> tuple[int, int]:
    """(# intervals at ⅓ or ⅔ of ``period``, # intervals at ½ of it): ternary vs binary sub-beats."""
    r = np.asarray(ibi, dtype=float) / period
    n3 = int(np.sum((np.abs(3 * r - 1) < SUB_TOL3) | (np.abs(1.5 * r - 1) < SUB_TOL2)))
    n2 = int(np.sum(np.abs(2 * r - 1) < SUB_TOL2))
    return n3, n2


def ternary_subbeats(ibi: np.ndarray, period: float) -> bool:
    """Part of the song is tracked at the eighths of a dotted-quarter beat ``period``."""
    n3, n2 = subbeat_counts(ibi, period)
    return n3 >= max(COMPOUND_SUB_MIN, COMPOUND_SUB_SHARE * len(ibi)) and n3 >= COMPOUND_SUB_RATIO * n2


def choose_period(seq: BeatSeq, tempo_factor: float = 1.0, auto: bool = True) -> TempoInfo:
    """The §5.3 tempo-octave decision (see the module docstring)."""
    ibi = np.diff(seq.t)
    p_dom = dominant_period(ibi)
    bpm = 60.0 / p_dom
    bpb = tracked_bpb(seq)
    h = hierarchy(seq, p_dom) if auto else None
    period, compound, folded = p_dom, False, False
    if auto:
        if bpm >= FOLD_MIN_BPM and bpb in (6, 12):
            period, compound, folded = 3 * p_dom, True, True
        elif MIN_BPM <= bpm < FOLD_MIN_BPM and ternary_subbeats(ibi, p_dom):
            compound = folded = True          # dotted-quarter beats; the eighth-tracked parts fold by 3
        elif bpm >= FOLD_MIN_BPM and bpm / 2 >= MIN_BPM:
            slow = int(np.sum(np.abs(ibi / (2 * p_dom) - 1) < 0.2))
            if bpb == 8:
                fold = True
            elif h is not None:
                fold = h[0] >= HIER_MIN and h[1] >= HIER_CONSIST
            else:
                fold = slow >= 8
            if fold:
                period, folded = 2 * p_dom, True
        elif bpm < MIN_BPM:
            period = p_dom / 2
    if tempo_factor and tempo_factor != 1.0:
        period /= tempo_factor
        folded = folded or tempo_factor < 1
    return TempoInfo(p_dom, period, compound, folded, period < p_dom * 0.75, bpb,
                     None if h is None else h[0])


def _phase_viterbi(seq: BeatSeq, s: int, units: np.ndarray, m: int = 2) -> np.ndarray:
    """Phase of beats s..s+len(units) on the unit grid: 0 = on the beat, 1..m−1 = sub-beats.

    An interval of k units moves the phase by k (mod m); breaking that costs SWITCH_COST. Beats
    whose neighbouring intervals are all whole beats are beats (phase 0)."""
    n = len(units) + 1
    off = units % m != 0
    adj = np.zeros(n, bool)
    adj[:-1] |= off
    adj[1:] |= off
    lg = seq.blog[s:s + n]
    z = np.zeros(n)
    if np.isfinite(lg[adj]).any():
        med = float(np.nanmedian(lg[adj]))
        z = np.nan_to_num(lg - med, nan=0.0)
    z = z + DB_ON_BONUS * seq.is_db[s:s + n]
    em = np.empty((n, m))
    em[:, 0] = z
    em[:, 1:] = -z[:, None]
    em[~adj, 1:] = -np.inf                 # beats between whole-beat intervals are beats, not sub-beats
    score = em[0].copy()
    back = np.zeros((n, m), int)
    for j in range(1, n):
        step = int(units[j - 1]) % m
        new = np.empty(m)
        for st in range(m):
            prev_ok = (st - step) % m      # predecessor phase that keeps the unit count
            other = score - SWITCH_COST
            other[prev_ok] = -np.inf
            alt = int(np.argmax(other))
            if score[prev_ok] >= other[alt]:
                back[j, st], best = prev_ok, score[prev_ok]
            else:
                back[j, st], best = alt, other[alt]
            new[st] = best + em[j, st]
        score = new
    states = np.zeros(n, int)
    states[-1] = int(np.argmax(score))
    for j in range(n - 1, 0, -1):
        states[j - 1] = back[j, states[j]]
    return states


def dedup(seq: BeatSeq, min_gap: float) -> BeatSeq:
    """Merge beats closer than ``min_gap`` (a tracker glitch): keep the one with the stronger logit."""
    if len(seq) < 2:
        return seq
    strength = np.nan_to_num(seq.blog, nan=0.0) + seq.conf + DB_ON_BONUS * seq.is_db
    keep = np.ones(len(seq), bool)
    last = 0
    for i in range(1, len(seq)):
        if seq.t[i] - seq.t[last] >= min_gap:
            last = i
        elif strength[i] > strength[last]:
            keep[last] = False
            last = i
        else:
            keep[i] = False
    return seq if keep.all() else seq.take(np.flatnonzero(keep))


def unit_fold(seq: BeatSeq, period: float, m: int = 2) -> BeatSeq:
    """Bring every steady stretch to the beat level ``period`` (drop sub-beats, fill missed beats).

    ``m`` = units per beat: 2 (half beats) or 3 (the eighths of a dotted-quarter beat)."""
    if len(seq) < 3:
        return seq
    u = period / m
    r = np.diff(seq.t) / u
    k = np.rint(r)
    ok = (np.abs(r - k) <= UNIT_TOL) & (k >= 1) & (k <= 4 * m)
    keep = np.ones(len(seq), bool)
    ins_t, ins_c = [], []
    for s, e in runs_of(ok):
        units = k[s:e].astype(int)
        if (units % m != 0).any():
            keep[s:e + 1] &= _phase_viterbi(seq, s, units, m) == 0
        if e - s < 4:
            continue
        cum = np.concatenate([[0], np.cumsum(units)])
        kept = [i for i in range(s, e + 1) if keep[i]]
        for a, b in zip(kept[:-1], kept[1:]):
            d = cum[b - s] - cum[a - s]
            if d % m == 0 and 2 * m <= d <= 4 * m:
                nb = d // m
                for j in range(1, nb):
                    ins_t.append(seq.t[a] + (seq.t[b] - seq.t[a]) * j / nb)
                    ins_c.append(min(seq.conf[a], seq.conf[b]))
    return seq.take(np.flatnonzero(keep)).with_inserted(ins_t, ins_c)


def normalize_tempo(seq: BeatSeq, tempo_factor: float = 1.0, auto: bool = True) -> tuple[BeatSeq, TempoInfo]:
    """Step 1 of the module docstring. ``auto=False`` (tapped beats) applies only tempo_factor."""
    if len(seq) < 4:
        p = float(np.median(np.diff(seq.t))) if len(seq) > 1 else 0.5
        return seq, TempoInfo(p, p, False, False, False, None, None)
    info = choose_period(seq, tempo_factor, auto)
    tf = tempo_factor if tempo_factor else 1.0
    out = seq
    if auto:
        base = info.period * tf           # the beat before the user's tempo factor
        out = dedup(out, DEDUP_UNITS * base / info.units)
        out = unit_fold(out, base, info.units)
        if tf != 1.0:
            out = unit_fold(out, info.period)
    elif tf != 1.0:
        out = unit_fold(out, info.period)
    if len(out) > 1:
        info.period = float(np.median(np.diff(out.t)))
    return out, info


# ------------------------------------------------------------------------------------------------
# 2. bars
# ------------------------------------------------------------------------------------------------

def bar_length(seq: BeatSeq, info: TempoInfo | None = None) -> int:
    """Main bar length L in beats (mixed meters are handled by the bar Viterbi)."""
    idx = np.flatnonzero(seq.is_db)
    d = np.diff(idx)
    L = mode_int(d[(d >= 1) & (d <= 16)])
    compound = bool(info and info.compound)
    if L is None:
        return 2 if compound else 4
    if L == 2:                             # half bars (of 4/4, or of 12/8 when compound)?
        if seq.has_dlog:
            s = np.nan_to_num(seq.dlog[idx], nan=UNKNOWN_DLOG)
            if len(s) >= 4 and float(np.median(np.abs(np.diff(s)))) >= ALT_MIN:
                L = 4
        elif not compound and info is not None and info.folded and 60.0 / info.period < HALF_BAR_MAX_BPM:
            L = 4
    if L == 1 or (L == 8 and not compound):
        L = 4
    return L


def bar_evidence(seq: BeatSeq) -> np.ndarray:
    """Downbeat log-odds per beat for the bar Viterbi."""
    if seq.has_dlog:
        return np.clip(np.nan_to_num(seq.dlog, nan=UNKNOWN_DLOG), -12.0, 12.0)
    return np.where(seq.is_db, NO_LOGIT_DB, NO_LOGIT_OTHER)


def viterbi_bars(evidence: np.ndarray, L: int, penalty: float = BAR_PENALTY) -> np.ndarray:
    """Downbeat indices for one metered run.

    State k = beats since the last downbeat (0..2L−1). A bar of length L is free, any other
    length costs ``penalty``; a pickup (start in any state < L) and a short last bar are free.
    """
    n = len(evidence)
    if n == 0:
        return np.zeros(0, int)
    K = 2 * L
    score = np.full(K, -np.inf)
    score[:L] = 0.0
    score[0] += evidence[0]
    back = np.zeros((n, K), int)
    trans_cost = np.where(np.arange(K) + 1 == L, 0.0, -penalty)    # from state k' → 0
    for i in range(1, n):
        cand = score + trans_cost
        j = int(np.argmax(cand))
        new = np.full(K, -np.inf)
        new[0] = cand[j] + evidence[i]
        back[i, 0] = j
        new[1:] = score[:-1]
        back[i, 1:] = np.arange(K - 1)
        score = new
    st = int(np.argmax(score))
    states = np.zeros(n, int)
    states[-1] = st
    for i in range(n - 1, 0, -1):
        st = back[i, st]
        states[i - 1] = st
    return np.flatnonzero(states == 0)


# ------------------------------------------------------------------------------------------------
# 3. free time
# ------------------------------------------------------------------------------------------------

def steady_runs(seq: BeatSeq) -> np.ndarray:
    """[B−1] bool: interval i lies in a run of ≥ STEADY_MIN steady intervals (the music is in tempo).

    Steady = within ±STEADY_TOL of the median of its 9 neighbours, at a local beat within
    STEADY_BAND of the song's median beat (a stretch tracked at another level does not count)."""
    ibi = np.diff(seq.t)
    n = len(ibi)
    out = np.zeros(n, bool)
    if n < STEADY_MIN:
        return out
    song = float(np.median(ibi))
    h = STEADY_MIN // 2
    med = np.array([np.median(ibi[max(0, i - h):i + h + 1]) for i in range(n)])
    st = (np.abs(ibi / med - 1) < STEADY_TOL) & (med > STEADY_BAND[0] * song) & (med < STEADY_BAND[1] * song)
    for s, e in runs_of(st):
        if e - s >= STEADY_MIN:
            out[s:e] = True
    return out


def free_intervals(seq: BeatSeq, window: int = FREE_WINDOW) -> np.ndarray:
    """[B−1] bool: beat interval i (beats i → i+1) is in free time (step 3 of the module docstring)."""
    B = len(seq)
    if B < 2:
        return np.zeros(0, bool)
    ibi = np.diff(seq.t)
    unsure = np.zeros(B - 1, bool)
    uneven = np.zeros(B - 1, bool)
    for k in range(B - 1):
        lo = max(0, min(k - window // 2 + 1, B - window))
        hi = min(B, lo + window)
        unsure[k] = float(np.mean(seq.conf[lo:hi])) < FREE_CONF
        uneven[k] = robust_cv(ibi[lo:hi - 1]) > FREE_CV
    # Once the tracker has been sure of the beat, a steady run is in tempo however unsure it gets
    # (soft passages). Before that (an intro) Beat This! extrapolates a steady but unsure pulse
    # through rubato too, so there low confidence alone still means free time.
    sure = np.flatnonzero(~unsure & ~uneven)
    kept = steady_runs(seq)
    if len(sure):
        kept[:sure[0]] = False
    else:
        kept[:] = False
    flag = (unsure | uneven) & ~kept
    low = seq.conf < FREE_EDGE_CONF
    for s, e in runs_of(flag):              # intervals s..e-1 are free; extend over unsure beats
        grown = 0
        while e < B - 1 and low[e] and (not kept[e] or grown < FREE_EDGE_MAX):
            grown += int(kept[e])
            flag[e] = True
            e += 1
        grown = 0
        while s > 0 and low[s - 1] and (not kept[s - 1] or grown < FREE_EDGE_MAX):
            s -= 1
            grown += int(kept[s])
            flag[s] = True
    return flag


# ------------------------------------------------------------------------------------------------
# 4. sparse intros
# ------------------------------------------------------------------------------------------------

def backfill_intro(seq: BeatSeq, changes: np.ndarray | None = None) -> tuple[BeatSeq, tuple[float, float] | None]:
    """Step 4 of the module docstring: a sparse intro in tempo inherits the song's grid.

    The intro is everything before the groove's lead-in: the steady beats (one song beat apart)
    leading into the first stable metered run, from the first of them the tracker is sure of
    (conf ≥ FREE_EDGE_CONF). ``changes``: harmonic-change times (``hcdf.peaks``). Beat This!
    extrapolates a steady, unsure pulse through a rubato intro as well, so the tracker cannot tell
    the two apart; the harmony can: free time is there so chords can change off the beat, and an
    intro that holds its chord (≤ BACKFILL_MAX_CHANGES harmonic changes per beat; a rubato intro
    changes chord at every attack) loses nothing on the grid. Without ``changes`` nothing is
    backfilled.

    Returns the new beats and the (start, end) span to keep metered, or (seq, None) when the song
    does not start in free time, has no stable run of ≥ BACKFILL_MIN_RUN intervals after it, the
    intro's tracked beats are not mostly whole song beats apart, or its harmony moves."""
    B = len(seq)
    if changes is None or B < BACKFILL_MIN_RUN + 3:
        return seq, None
    free = free_intervals(seq)
    if not free[0]:
        return seq, None
    run = next(((s, e) for s, e in runs_of(~free) if e - s >= BACKFILL_MIN_RUN), None)
    if run is None or run[0] < 2:
        return seq, None
    a = int(run[0])                          # first beat of the stable run
    ibi = np.diff(seq.t)
    P = float(np.median(ibi[a:a + BACKFILL_MIN_RUN]))
    c = a                                    # first beat of the groove's sure, steady lead-in
    while c > 0 and abs(ibi[c - 1] / P - 1) < BACKFILL_TOL:
        c -= 1
    while c < a and seq.conf[c] < FREE_EDGE_CONF:
        c += 1
    r = ibi[:c] / P                          # whole beats apart (1–4: a sparse intro misses beats)
    on_grid = (np.abs(r - np.rint(r)) < BACKFILL_TOL) & (np.rint(r) >= 1) & (np.rint(r) <= 4)
    if c < 2 or np.mean(on_grid) < BACKFILL_SHARE:
        return seq, None
    ch = np.asarray(changes, dtype=float)
    lo, hi = seq.t[0] + 0.25 * P, seq.t[c] - 0.25 * P
    if np.sum((ch > lo) & (ch < hi)) > BACKFILL_MAX_CHANGES * (seq.t[c] - seq.t[0]) / P:
        return seq, None                     # the chords move: a rubato intro stays free
    tracked = seq.t[:c]
    times, src = [], []
    x = float(seq.t[c])
    while x - P >= seq.t[0] - 0.5 * P:
        x -= P
        j = int(np.argmin(np.abs(tracked - x)))
        if abs(tracked[j] - x) <= BACKFILL_SNAP * P:
            x = float(tracked[j])            # a tracked beat on the grid: keep it (and its evidence)
            src.append(j)
        else:
            src.append(-1)
        times.append(x)
    if not times:
        return seq, None
    keep = sorted(j for j in src if j >= 0) + list(range(c, B))
    ins = [t for t, j in zip(times, src) if j < 0]
    out = seq.take(np.asarray(keep, int)).with_inserted(ins, [BACKFILL_CONF] * len(ins))
    return out, (float(min(times)), float(seq.t[a]))
