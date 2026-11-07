"""WP7 decoding: semi-Markov Viterbi + forward–backward on the sub-beat grid (DESIGN §5.10).

The model is defined on *units*: cells for the detail layer, beats (3 cells in free time) for
the main layer (E summed over each beat's cells, E_short unused). A path is a sequence of
segments (i, j, c) over units [i, j) with label c ∈ C; its score is

    Σ_segments [ seg(i, j, c) + D(j − i, c, i) ]  +  Σ_boundaries T(cp, c, i)

    seg(i, j, c) = (Ecum_s if dur_beats(i, j) ≤ ½ else Ecum)[j, c] − (same)[i, c]
    D(n, c, i)   = w_dur · lognormal_logpdf(dur_beats, median 2.0, σ 0.9)
                   + pt_bonus [c ∈ PT and dur ≤ 1 beat] + rep_bonus [cls(c) = majority(rep[i:i+n])]
    T(cp, c, i)  = −0.3 if cp = c (re-entry, merged later), otherwise
                   −λ[cell_pos(i)] + w_prior·prior(cp, c, key(i)) + 0.6 [1 ≤ |Δbass| ≤ 2]
                   + 0.5 [HCDF peak within 60 ms]

with λ = λ_scale·lam (detail) or λ_scale·lam_main_mult·lam (main). Segments are 1 … Lmax units.

Viterbi (max-sum) and forward–backward (log-sum-exp) share the same recursions; both are
vectorised over the candidates and the segment lengths, looping only over the segment end j:

    Δ[i, c] = max_cp δ[i, cp] + T(cp, c, i)                   (O(|C|²) per boundary)
    δ[j, c] = max_n start[j − n, c] + seg + D                  (O(Lmax·|C|) per end)

Posteriors come from start/end probabilities: post[g, c] = Σ_{i ≤ g} P(c starts at i) −
Σ_{j ≤ g} P(c ends at j); pb[g] = P(a *change* at g) (re-entries excluded).

M2: the forward–backward can run at a temperature ``params.POST_TEMP``: every score term is
divided by it, so the Viterbi path (and the reported score, rescaled) is unchanged and only the
marginals are tempered (temperature scaling). 2.0 meets the §5.9 ECE gate on mini-D2 (0.165 →
0.085) but ruins the Detail slider's ranking, so it ships at 1.0 (see params.py and M2.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..theory import priors as PR
from ..theory.vocab import is_passing_type
from ..types import Candidate, DecodeResult, DecoderParams, Evidence, KeyTrack, RhythmGrid, Segment, Vocab
from . import params as P
from .evidence import cell_beats

NEG = -1e30
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_EPS_B = 1e-6


def lognormal_logpdf(d: np.ndarray | float, median: float, sigma: float) -> np.ndarray:
    """log of the log-normal density of a duration d (beats) with the given median and σ."""
    d = np.maximum(np.asarray(d, dtype=float), 1e-9)
    ld = np.log(d)
    return -ld - math.log(sigma) - _LOG_SQRT_2PI - (ld - math.log(median)) ** 2 / (2.0 * sigma * sigma)


# =============================================================================================
# Units (cells or beats)
# =============================================================================================

@dataclass
class Units:
    c0: np.ndarray            # [U] first cell
    c1: np.ndarray            # [U] end cell (exclusive)
    beats: np.ndarray         # [U] length in beats (Δt / ibi_ref in free time)
    pos: np.ndarray           # [U] metric position code of the first cell (λ index)
    beat_idx: np.ndarray      # [U] beat index (local key lookup)


def beat_of_cells(grid: RhythmGrid) -> np.ndarray:
    """Beat index of every cell (free-time cells: the last beat at or before the cell)."""
    ct = np.asarray(grid.cell_t, dtype=float)
    cb = np.asarray(grid.cell_beat, dtype=float)[:-1]
    B = max(len(grid.beats), 1)
    out = np.empty(len(cb), dtype=np.int64)
    fin = np.isfinite(cb)
    out[fin] = np.floor(cb[fin] + 1e-6).astype(np.int64)
    if (~fin).any():
        out[~fin] = np.searchsorted(np.asarray(grid.beats, dtype=float), ct[:-1][~fin] + 1e-6, side="right") - 1
    return np.clip(out, 0, B - 1)


def detail_units(grid: RhythmGrid) -> Units:
    G = len(grid.cell_t) - 1
    c = np.arange(G)
    return Units(c, c + 1, cell_beats(grid), np.asarray(grid.cell_pos, dtype=np.int64), beat_of_cells(grid))


def main_units(grid: RhythmGrid) -> Units:
    """Beats as units; in free time, groups of 3 cells (0.6 s) act as beats (§5.10)."""
    cb = np.asarray(grid.cell_beat, dtype=float)[:-1]
    G = len(cb)
    starts: list[int] = []
    prev_key = None
    run = -1
    k_in_run = 0
    for g in range(G):
        if np.isfinite(cb[g]):
            key = ("m", int(math.floor(cb[g] + 1e-6)))
        else:
            if prev_key is None or prev_key[0] != "f":
                run += 1
                k_in_run = 0
            key = ("f", run, k_in_run // P.FREE_BEAT_CELLS)
            k_in_run += 1
        if key != prev_key:
            starts.append(g)
        prev_key = key
    c0 = np.array(starts, dtype=np.int64)
    c1 = np.r_[c0[1:], G]
    u = cell_beats(grid)
    beats = np.add.reduceat(u, c0)
    pos = np.asarray(grid.cell_pos, dtype=np.int64)[c0]
    pos = np.where(np.isin(pos, (3, 4)), 2, pos)
    return Units(c0, c1, beats, pos, beat_of_cells(grid)[c0])


def _lmax(grid: RhythmGrid, level: str) -> int:
    bpb = max([b for _, b in grid.bars] or [4])
    sub = int(np.max(grid.subdiv)) if len(grid.subdiv) else 4
    L = P.LMAX_BARS * bpb * (sub if level == "detail" else 1)
    free = np.asarray(grid.cell_pos) == 5
    if free.any():
        dt = float(np.median(np.diff(np.asarray(grid.cell_t))[free]))
        per = dt * (1 if level == "detail" else P.FREE_BEAT_CELLS)
        L = max(L, int(math.ceil(P.LMAX_FREE_S / max(per, 1e-3))))
    return max(int(L), 1)


# =============================================================================================
# The scoring model
# =============================================================================================

# --- full-vocabulary prior tables (M2 speed-up) ----------------------------------------------------
# prior(cp, c, key) depends only on the chords' degrees relative to the key (theory.priors), so for
# a key with tonic t it equals prior(cp − t, c − t, C major | C minor). One [V, V] table per mode,
# computed once (≈ 3.6 M prior calls) and cached on disk under a hash of the theory sources and the
# vocabulary, replaces the per-song O(|C|²) Python loop (≈ 1 s per key and lattice).

_TABLES: dict[int, np.ndarray] = {}
_SHIFT: np.ndarray | None = None


def _table_id(vocab: Vocab) -> str:
    import hashlib
    import inspect

    from ..theory import templates, vocab as TV
    h = hashlib.sha1()
    for mod in (PR, templates, TV):
        h.update(inspect.getsource(mod).encode())
    h.update(repr([(c.root, c.quality, c.bass) for c in vocab.cands]).encode())
    return h.hexdigest()[:12]


def _shift_index(vocab: Vocab) -> np.ndarray:
    """[12, V]: index of each candidate transposed DOWN by t semitones."""
    global _SHIFT
    if _SHIFT is None or _SHIFT.shape[1] != len(vocab.cands):
        idx = {(c.root, c.quality, c.bass): i for i, c in enumerate(vocab.cands)}
        out = np.zeros((12, len(vocab.cands)), dtype=np.int64)
        for i, c in enumerate(vocab.cands):
            for t in range(12):
                out[t, i] = i if c.root < 0 else idx[((c.root - t) % 12, c.quality, (c.bass - t) % 12)]
        _SHIFT = out
    return _SHIFT


def prior_table(vocab: Vocab, mode: int) -> np.ndarray:
    """Uncapped prior(a, b, key) over the vocabulary for key C major (mode 0) or C minor (mode 1)."""
    t = _TABLES.get(mode)
    if t is not None and t.shape[0] == len(vocab.cands):
        return t
    from .. import config
    path = config.MODELS_DIR / "derived" / f"prior_table_{_table_id(vocab)}_{mode}.npy"
    if path.exists():
        t = np.load(path)
    else:
        trip = [(c.root, c.quality, c.bass) for c in vocab.cands]
        key = 0 if mode == 0 else 12
        V = len(trip)
        t = np.zeros((V, V), dtype=np.float32)
        for a in range(V):
            ta = trip[a]
            if ta[1] == "N":
                continue
            row = t[a]
            for b in range(V):
                if a != b:
                    row[b] = PR.prior(ta, trip[b], key, 1e9)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".part.npy")
        np.save(tmp, t)
        tmp.replace(path)
    t.setflags(write=False)
    _TABLES[mode] = t
    return t


def prior_matrix_for(C: np.ndarray, vocab: Vocab, key: int, cap: float) -> np.ndarray:
    """prior(cp, c, key) for every pair of candidates C (vocabulary indices), capped at ``cap``."""
    tonic, mode = key % 12, int(key >= 12)
    s = _shift_index(vocab)[tonic][np.asarray(C, dtype=np.int64)]
    M = np.minimum(prior_table(vocab, mode)[np.ix_(s, s)].astype(float), cap)
    np.fill_diagonal(M, 0.0)
    return M


def _step_bass(b1: int, b2: int) -> float:
    if b1 < 0 or b2 < 0:
        return 0.0
    d = abs(b1 - b2) % 12
    return P.STEP_BASS_BONUS if 1 <= min(d, 12 - d) <= 2 else 0.0


class Lattice:
    """The §5.10 scores at one level; also used by ``layers`` for local rescoring (salience)."""

    def __init__(self, ev: Evidence, grid: RhythmGrid, key: KeyTrack | None, vocab: Vocab,
                 p: DecoderParams, level: Literal["detail", "main"] = "detail",
                 rep: np.ndarray | None = None, hcdf: np.ndarray | None = None, tau: float = 1.0):
        self.p, self.grid, self.vocab, self.level = p, grid, vocab, level
        # every score term is divided by tau (forward–backward temperature, P.POST_TEMP in decode);
        # the Viterbi path is invariant to it, the marginals are tempered
        self.tau = float(tau)
        self.s = s = 1.0 / self.tau
        self.t_self = P.T_SELF * s
        self.C = np.asarray(ev.C, dtype=np.int64)
        self.K = len(self.C)
        self.units = un = detail_units(grid) if level == "detail" else main_units(grid)
        self.U = len(un.c0)
        if level == "detail":
            E, Es = np.asarray(ev.E, dtype=float), np.asarray(ev.E_short, dtype=float)
        else:
            E = np.add.reduceat(np.asarray(ev.E, dtype=float), un.c0, axis=0)
            Es = E
        z = np.zeros((1, self.K))
        self.Ecum = np.vstack([z, np.cumsum(E, axis=0) * s])
        self.use_short = level == "detail" and not np.array_equal(E, Es)
        self.Escum = np.vstack([z, np.cumsum(Es, axis=0) * s]) if self.use_short else self.Ecum
        self.Ucum = np.r_[0.0, np.cumsum(un.beats)]
        mult = p.lam_scale * (p.lam_main_mult if level == "main" else 1.0)
        self.lam = np.array([p.lam.get(int(q), p.lam.get(2, 3.0)) for q in un.pos]) * mult * s
        self.hb = np.zeros(self.U)
        if hcdf is not None and len(hcdf):
            peaks = np.sort(np.asarray(hcdf, dtype=float))
            t = np.asarray(grid.cell_t, dtype=float)[un.c0]
            k = np.searchsorted(peaks, t)
            lo = peaks[np.clip(k - 1, 0, len(peaks) - 1)]
            hi = peaks[np.clip(k, 0, len(peaks) - 1)]
            near = np.minimum(np.abs(lo - t), np.abs(hi - t))
            self.hb[near <= P.HCDF_WINDOW_S] = P.HCDF_BONUS * s
        if key is not None and len(key.per_beat):
            pb_ = np.asarray(key.per_beat, dtype=np.int64)
            self.keys = pb_[np.clip(un.beat_idx, 0, len(pb_) - 1)]
        else:
            self.keys = np.full(self.U, -1, dtype=np.int64)
        self.cands: list[Candidate] = [vocab.cands[c] for c in self.C]
        self.triples = tuple((c.root, c.quality, c.bass) for c in self.cands)
        self.pt = np.array([is_passing_type(c) for c in self.cands], dtype=bool)
        self.ptb = p.pt_bonus * self.pt.astype(float) * s
        self.cls = np.array([c.cls for c in self.cands], dtype=np.int64)
        self.bass = np.array([c.bass for c in self.cands], dtype=np.int64)
        self.col = {int(c): k for k, c in enumerate(self.C)}
        self.Lmax = _lmax(grid, level)
        self.rep = None
        if rep is not None:
            r = np.asarray(rep, dtype=np.int64)
            if level == "main":
                r = np.array([_majority(r[a:b]) for a, b in zip(un.c0, un.c1)], dtype=np.int64)
            vals = np.unique(r[r >= 0])
            if len(vals):
                oh = (r[:, None] == vals[None, :]).astype(np.int32)
                self.rep = (vals, np.vstack([np.zeros((1, len(vals)), np.int32), np.cumsum(oh, axis=0)]))
        d = np.abs(self.bass[:, None] - self.bass[None, :]) % 12
        d = np.minimum(d, 12 - d)
        self.stepbass = np.where((d >= 1) & (d <= 2) & (self.bass[:, None] >= 0) & (self.bass[None, :] >= 0),
                                 P.STEP_BASS_BONUS * s, 0.0)
        self._pm: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # --- transitions ---------------------------------------------------------------------------
    def prior_matrix(self, key: int) -> tuple[np.ndarray, np.ndarray]:
        """(M, exp M) with M[cp, c] = w_prior·prior + 0.6·stepbass for cp ≠ c; diagonal excluded."""
        hit = self._pm.get(key)
        if hit is None:
            M = self.stepbass.copy()
            if key >= 0 and self.p.w_prior:
                M = M + self.s * self.p.w_prior * prior_matrix_for(self.C, self.vocab, int(key), float(self.p.prior_cap))
            np.fill_diagonal(M, NEG)
            X = np.exp(M)
            np.fill_diagonal(X, 0.0)
            hit = self._pm[key] = (M, X)
        return hit

    def trans_cand(self, cp: Candidate, c: Candidate, i: int) -> float:
        """T(cp, c, i) for any two vocabulary candidates (boundaries, rescoring)."""
        if (cp.root, cp.quality, cp.bass) == (c.root, c.quality, c.bass):
            return self.t_self
        i = min(max(i, 0), self.U - 1)
        k = int(self.keys[i])
        pr = PR.prior(cp, c, k, self.p.prior_cap) if (k >= 0 and self.p.w_prior) else 0.0
        return float(-self.lam[i] + self.hb[i] + self.s * (self.p.w_prior * pr + _step_bass(cp.bass, c.bass)))

    def trans(self, kp: int, k: int, i: int) -> float:
        """T between two columns of C at boundary unit i."""
        return self.trans_cand(self.cands[kp], self.cands[k], i)

    def _vit_trans(self, d: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
        M, _ = self.prior_matrix(int(self.keys[i]))
        S = d[:, None] + M
        cp = S.argmax(axis=0)
        best = S[cp, np.arange(self.K)] - self.lam[i] + self.hb[i]
        selfv = d + self.t_self
        use = selfv >= best
        return np.where(use, selfv, best), np.where(use, np.arange(self.K), cp)

    def _fwd_trans(self, a: np.ndarray, i: int) -> np.ndarray:
        _, X = self.prior_matrix(int(self.keys[i]))
        m = a.max()
        with np.errstate(divide="ignore"):
            v = np.log(np.exp(a - m) @ X) + m - self.lam[i] + self.hb[i]
        return np.logaddexp(v, a + self.t_self)

    def _bwd_trans(self, b: np.ndarray, i: int) -> np.ndarray:
        _, X = self.prior_matrix(int(self.keys[i]))
        m = b.max()
        with np.errstate(divide="ignore"):
            v = np.log(X @ np.exp(b - m)) + m - self.lam[i] + self.hb[i]
        return np.logaddexp(v, b + self.t_self)

    # --- segments ------------------------------------------------------------------------------
    def _rep_majority(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Strict-majority consensus class of rep over each [start, end) (−2 = none)."""
        vals, cum = self.rep
        counts = cum[ends] - cum[starts]
        if counts.ndim == 1:
            counts = counts[None, :]
        best = counts.argmax(axis=1)
        cnt = counts[np.arange(len(counts)), best]
        return np.where(2 * cnt > (ends - starts), vals[best], -2)

    def _finish(self, out: np.ndarray, dur: np.ndarray, starts, ends) -> np.ndarray:
        """Add D(n, c, i) to segment sums ``out`` [n, K] (dur [n] beats; §5.10)."""
        p = self.p
        out += (self.s * p.w_dur * lognormal_logpdf(dur, p.dur_median_beats, p.dur_sigma))[:, None]
        if p.pt_bonus:
            m1 = dur <= P.PT_MAX_BEATS + _EPS_B
            if m1.any():
                out[m1] += self.ptb
        if self.rep is not None:
            maj = self._rep_majority(starts, ends)
            out += self.s * p.rep_bonus * (self.cls[None, :] == maj[:, None])
        return out

    def sd(self, i_lo: int, j: int) -> np.ndarray:
        """seg + D for all starts i ∈ [i_lo, j) of segments ending at j: [j − i_lo, K]."""
        out = self.Ecum[j] - self.Ecum[i_lo:j]
        dur = self.Ucum[j] - self.Ucum[i_lo:j]
        if self.use_short:
            sm = dur <= P.SHORT_BEATS + _EPS_B
            if sm.any():
                out[sm] = self.Escum[j] - self.Escum[i_lo:j][sm]
        return self._finish(out, dur, np.arange(i_lo, j), j)

    def sd_from(self, i: int, j_hi: int) -> np.ndarray:
        """seg + D for all ends j ∈ (i, j_hi] of segments starting at i: [j_hi − i, K]."""
        out = self.Ecum[i + 1:j_hi + 1] - self.Ecum[i]
        dur = self.Ucum[i + 1:j_hi + 1] - self.Ucum[i]
        if self.use_short:
            sm = dur <= P.SHORT_BEATS + _EPS_B
            if sm.any():
                out[sm] = self.Escum[i + 1:j_hi + 1][sm] - self.Escum[i]
        return self._finish(out, dur, i, np.arange(i + 1, j_hi + 1))

    def seg_score(self, k: int, i: int, j: int) -> float:
        """seg + D of one segment of column k over units [i, j) (any length; used for rescoring)."""
        dur = self.Ucum[j] - self.Ucum[i]
        cum = self.Escum if (self.use_short and dur <= P.SHORT_BEATS + _EPS_B) else self.Ecum
        p = self.p
        v = cum[j, k] - cum[i, k] + self.s * p.w_dur * float(lognormal_logpdf(dur, p.dur_median_beats, p.dur_sigma))
        if p.pt_bonus and dur <= P.PT_MAX_BEATS + _EPS_B:
            v += self.ptb[k]
        if self.rep is not None:
            v += self.s * p.rep_bonus * float(self.cls[k] == self._rep_majority(np.array([i]), np.array([j]))[0])
        return float(v)

    def path_score(self, segs: list[tuple[int, int, int]]) -> float:
        """Score of consecutive segments (i, j, k): Σ seg + D + the transitions between them."""
        s = 0.0
        for n, (i, j, k) in enumerate(segs):
            s += self.seg_score(k, i, j)
            if n:
                s += self.trans(segs[n - 1][2], k, i)
        return s



def _majority(x: np.ndarray) -> int:
    x = x[x >= 0]
    if not len(x):
        return -1
    v, c = np.unique(x, return_counts=True)
    k = int(c.argmax())
    return int(v[k]) if 2 * c[k] > len(x) else -1


# =============================================================================================
# Viterbi + forward–backward
# =============================================================================================

@dataclass
class _Run:
    path: list[tuple[int, int, int]]   # (i, j, k) in units, k = column of C
    score: float
    post: np.ndarray | None            # [U, K]
    pb: np.ndarray | None              # [U]
    logZ: float | None


def _run(lat: Lattice, r0: int, r1: int, L0: np.ndarray, Rv: np.ndarray, left_k: int | None,
         marginals: bool = True) -> _Run:
    U, K, Lmax = lat.U, lat.K, lat.Lmax
    start_v = np.full((U + 1, K), NEG)
    start_v[r0] = L0
    delta = np.full((U + 1, K), NEG)
    bp_n = np.zeros((U + 1, K), dtype=np.int32)
    bp_c = np.zeros((U + 1, K), dtype=np.int32)
    ar = np.arange(K)
    if marginals:
        start_f = np.full((U + 1, K), NEG)
        start_f[r0] = L0
        alpha = np.full((U + 1, K), NEG)
    for j in range(r0 + 1, r1 + 1):
        i_lo = max(r0, j - Lmax)
        sd = lat.sd(i_lo, j)
        tot = start_v[i_lo:j] + sd
        r = tot.argmax(axis=0)
        delta[j] = tot[r, ar]
        bp_n[j] = j - (i_lo + r)
        if marginals:
            a = start_f[i_lo:j] + sd
            m = a.max(axis=0)
            alpha[j] = m + np.log(np.exp(a - m).sum(axis=0))
        if j < r1:
            start_v[j], bp_c[j] = lat._vit_trans(delta[j], j)
            if marginals:
                start_f[j] = lat._fwd_trans(alpha[j], j)

    final = delta[r1] + Rv
    k = int(final.argmax())
    score = float(final[k])
    path: list[tuple[int, int, int]] = []
    j = r1
    while j > r0:
        n = int(bp_n[j, k])
        i = j - n
        path.append((i, j, k))
        if i <= r0:
            break
        k = int(bp_c[i, k])
        j = i
    path.reverse()
    if not marginals:
        return _Run(path, score, None, None, None)

    # backward: βs[i, c] = log Σ_n exp(seg + D + β[i + n, c]) (a segment of c starts at i),
    #           β[i, cp] = log Σ_c exp(T(cp, c, i) + βs[i, c])     (a segment of cp ends at i)
    beta = np.full((U + 1, K), NEG)
    beta[r1] = Rv
    betas = np.full((U + 1, K), NEG)
    for i in range(r1 - 1, r0 - 1, -1):
        j_hi = min(r1, i + Lmax)
        x = lat.sd_from(i, j_hi) + beta[i + 1:j_hi + 1]
        m = x.max(axis=0)
        betas[i] = m + np.log(np.exp(x - m).sum(axis=0))
        if i > r0:
            beta[i] = lat._bwd_trans(betas[i], i)

    fz = alpha[r1] + Rv
    logZ = float(fz.max() + np.log(np.exp(fz - fz.max()).sum()))
    S = np.exp(start_f[r0:r1] + betas[r0:r1] - logZ)              # P(c starts at i)
    Eend = np.exp(alpha[r0 + 1:r1 + 1] + beta[r0 + 1:r1 + 1] - logZ)   # P(c ends at j)
    ended = np.vstack([np.zeros((1, K)), np.cumsum(Eend, axis=0)[:-1]])
    post_r = np.clip(np.cumsum(S, axis=0) - ended, 0.0, None)
    post_r /= np.maximum(post_r.sum(axis=1, keepdims=True), 1e-300)
    pb_r = np.zeros(r1 - r0)
    if r1 - r0 > 1:
        selfp = np.exp(alpha[r0 + 1:r1] + lat.t_self + betas[r0 + 1:r1] - logZ).sum(axis=1)
        pb_r[1:] = np.clip(S[1:].sum(axis=1) - selfp, 0.0, 1.0)
    pb_r[0] = 1.0 if left_k is None else float(np.clip(1.0 - S[0, left_k], 0.0, 1.0))
    post = np.zeros((U, K))
    post[r0:r1] = post_r
    pb = np.zeros(U)
    pb[r0:r1] = pb_r
    return _Run(path, score, post, pb, logZ)


# =============================================================================================
# Public entry point
# =============================================================================================

def class_marginals(post: np.ndarray, C: np.ndarray, vocab: Vocab) -> np.ndarray:
    """post_cls [G, n_cls] = Σ_{c ∈ class} post[g, c] (vocabulary class ids)."""
    cls = np.array([vocab.cands[c].cls for c in C], dtype=np.int64)
    out = np.zeros((post.shape[0], len(vocab.classes)))
    order = np.argsort(cls, kind="stable")
    cs = cls[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    out[:, cs[starts]] = np.add.reduceat(post[:, order], starts, axis=1)
    return out


def _region_units(lat: Lattice, region: tuple[int, int] | None) -> tuple[int, int]:
    if region is None:
        return 0, lat.U
    g0, g1 = int(region[0]), int(region[1])
    r0 = int(np.searchsorted(lat.units.c0, g0, side="left"))
    r1 = int(np.searchsorted(lat.units.c1, g1, side="right"))
    r0, r1 = max(0, min(r0, lat.U - 1)), max(1, min(r1, lat.U))
    return r0, max(r1, r0 + 1)


def decode(ev: Evidence, grid: RhythmGrid, key: KeyTrack | None, vocab: Vocab, p: DecoderParams,
           level: Literal["detail", "main"] = "detail",
           region: tuple[int, int] | None = None,
           boundary: tuple[int | None, int | None] = (None, None),
           rep: np.ndarray | None = None, *,
           hcdf: np.ndarray | None = None, marginals: bool = True) -> DecodeResult:
    """Decode the detail (cells) or main (beats) layer (§5.10).

    ``region`` is a cell range [r0, r1) (re-scan); ``boundary`` = (left, right) vocabulary
    candidates fixed just outside it: L0[c] = T(left, c, r0), the last segment adds
    T(c, right, r1). ``rep`` is the consensus class per cell (M5, −1 = none). ``hcdf`` (keyword,
    not in the §4 contract yet) are HCDF peak times (s) for the +0.5 boundary bonus.

    Returns segments in cell coordinates (consecutive re-entries merged, ``cand`` = vocabulary
    index, ``layer`` = "main" until ``layers.build`` assigns layers, ``conf`` = class-level mean
    posterior), ``post`` [G, K], ``post_cls`` [G, n_cls], ``pb`` [G] and the Viterbi score. Rows
    outside the region are zero.
    """
    tau = P.POST_TEMP if marginals else 1.0
    lat = Lattice(ev, grid, key, vocab, p, level, rep=rep, hcdf=hcdf, tau=tau)
    r0, r1 = _region_units(lat, region)
    left, right = boundary
    L0 = np.zeros(lat.K)
    left_k = None
    if left is not None:
        lc = vocab.cands[left]
        L0 = np.array([lat.trans_cand(lc, c, r0) for c in lat.cands])
        left_k = lat.col.get(int(left))
    Rv = np.zeros(lat.K)
    if right is not None and r1 < lat.U:
        rc = vocab.cands[right]
        Rv = np.array([lat.trans_cand(c, rc, r1) for c in lat.cands])
    run = _run(lat, r0, r1, L0, Rv, left_k, marginals)

    G = len(grid.cell_t) - 1
    un = lat.units
    if marginals:
        counts = un.c1 - un.c0
        post = np.repeat(run.post, counts, axis=0)
        pb = np.zeros(G)
        pb[un.c0] = run.pb
        post_cls = class_marginals(post, lat.C, vocab)
    else:
        post = np.zeros((G, lat.K))
        pb = np.zeros(G)
        post_cls = np.zeros((G, len(vocab.classes)))

    ct = np.asarray(grid.cell_t, dtype=float)
    segments: list[Segment] = []
    for i, j, k in run.path:
        c0, c1 = int(un.c0[i]), int(un.c1[j - 1])
        if segments and segments[-1].cand == int(lat.C[k]) and segments[-1].c1 == c0:
            segments[-1].c1, segments[-1].t1 = c1, float(ct[c1])
            continue
        cand = int(lat.C[k])
        segments.append(Segment(c0=c0, c1=c1, t0=float(ct[c0]), t1=float(ct[c1]), cand=cand,
                                cls=vocab.cands[cand].cls, layer="main"))
    for s in segments:
        if marginals:
            s.conf = float(post_cls[s.c0:s.c1, s.cls].mean())
    return DecodeResult(segments=segments, post=post, post_cls=post_cls, pb=pb, score=run.score * lat.tau)


def rescan_region(grid: RhythmGrid, t0: float, t1: float, bars: int = 2) -> tuple[int, int]:
    """Cell range for "Re-scan this section": the selection widened by ± ``bars`` bars (§5.11)."""
    ct = np.asarray(grid.cell_t, dtype=float)
    beats = np.asarray(grid.beats, dtype=float)
    if grid.bars and len(beats):
        bt = np.array([beats[min(b, len(beats) - 1)] for b, _ in grid.bars])
        a = int(np.searchsorted(bt, t0, side="right") - 1)
        b = int(np.searchsorted(bt, t1, side="left"))
        s = bt[max(a - bars, 0)] if a - bars >= 0 else ct[0]
        e = bt[b + bars] if b + bars < len(bt) else ct[-1]
    else:
        span = bars * 4 * float(grid.ibi_ref)
        s, e = t0 - span, t1 + span
    r0 = int(np.clip(np.searchsorted(ct, s, side="right") - 1, 0, len(ct) - 2))
    r1 = int(np.clip(np.searchsorted(ct, e, side="left"), r0 + 1, len(ct) - 1))
    return r0, r1
