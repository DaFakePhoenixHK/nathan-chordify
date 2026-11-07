"""WP7 fusion: calibrated cell evidence (DESIGN §5.9) and candidate pruning (§5.8).

``build`` turns the model outputs, the symbolic chroma and the bass track into ``Evidence``:

1. **One source per model** (§5.9 step 1). A model's head probabilities are averaged over its
   inputs first (p = 0.6·p@harmonic + 0.4·p@mix). Each model source is then written as a set of
   per-frame *columns* (logs of head classes, floored where §5.9 floors them) and a coefficient
   matrix A [V, M] with raw_s(t, c) = cols(t) · A[c]. Every nonlinear step happens per frame,
   so interpolating and integrating the columns (step 2) is exactly the same as interpolating and
   integrating raw_s(t, c), at a fraction of the cost (≈ 100 columns instead of |C|).
2. **Time** (§5.9 step 2): columns are linearly interpolated onto the 11.6 ms timeline
   (``timeline``/``to_timeline``) and integrated over each cell with overlap weights (``pool``).
3. **Calibration** (§5.9 step 3): ℓ_s = max(clip, log_softmax_C(raw_s / T_s)); the bass source
   ``bas`` runs its softmax over the 12 bass pcs.
4. **Weighted mean, per beat** (§5.9 step 4): B = Σ w_s ℓ_s / Σ w_s, E = κ_b·(Δg/ibi_g)·B, and
   E_short with w_bas and w_sym multiplied by (1 + w_short) (numerator *and* denominator, so it
   stays a weighted mean).

Sources plug in through ``register_source`` (frame-level model sources: lv now; ace and btc are
implemented to the §5.9 formulas and switch on as soon as their adapters emit evidence named
"ace@…"/"btc@…"); ``sym`` and ``bas`` are cell-level and built in.

``Evidence.parts`` holds the calibrated ℓ_s [G, K] per source under the source name ("lv",
"bas", …). Keys containing ":" are auxiliary arrays used by ``post``/``calibrate`` (they are not
[G, K] sources): ``aux:raw:<src>`` (uncalibrated raw scores, for ``calibrate.fit``),
``aux:lv:root`` [G,12] (lv root-head mass), ``aux:lv:tones`` [G,12] (lv@harmonic per-pc tone
presence), ``aux:sym:x`` [G,12].
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp

from ..theory import templates as TT
from ..types import BassTrack, ChordEvidence, DecodeResult, DecoderParams, Evidence, RhythmGrid, Vocab
from . import params as P

EPS = 1e-12
AUX = "aux:"


# =============================================================================================
# Grid and timeline helpers
# =============================================================================================

def cell_beats(grid: RhythmGrid) -> np.ndarray:
    """Δg / ibi_g per cell (beats): the metered cell length in beats, Δt / ibi_ref in free time."""
    cb = np.asarray(grid.cell_beat, dtype=float)
    ct = np.asarray(grid.cell_t, dtype=float)
    u = np.diff(cb)
    bad = ~np.isfinite(u)
    if bad.any():
        u[bad] = np.diff(ct)[bad] / max(float(grid.ibi_ref), 1e-6)
    return np.maximum(u, 1e-6)


def timeline(t_end: float, hop: float = P.HOP_S) -> np.ndarray:
    """Frame centres k·hop, k = 0 … ceil(t_end / hop): the 11.6 ms evidence timeline (§5.9)."""
    n = int(np.ceil(max(float(t_end), 0.0) / hop)) + 1
    return np.arange(n) * hop


def to_timeline(frame_t: np.ndarray, X: np.ndarray, tl_t: np.ndarray) -> np.ndarray:
    """Linear interpolation of frame scores X [T, M] (at frame centres ``frame_t``, ascending)
    onto ``tl_t``; held constant beyond both ends (§5.9 step 2)."""
    ft = np.asarray(frame_t, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if len(ft) == 1:
        return np.repeat(X, len(tl_t), axis=0)
    j = np.clip(np.searchsorted(ft, tl_t, side="right"), 1, len(ft) - 1)
    t0, t1 = ft[j - 1], ft[j]
    w = np.clip((tl_t - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)[:, None]
    return X[j - 1] * (1.0 - w) + X[j] * w


def pool(cell_t: np.ndarray, tl_t: np.ndarray, X_tl: np.ndarray, hop: float = P.HOP_S) -> np.ndarray:
    """Overlap-weighted mean of timeline frames over each cell [cell_t[g], cell_t[g+1]) (§5.9 step 2).

    Frame k covers [t_k − hop/2, t_k + hop/2); the step function is integrated exactly through
    its running integral, so partial frames at cell edges get their overlap weight."""
    X_tl = np.asarray(X_tl, dtype=float)
    if X_tl.ndim == 1:
        X_tl = X_tl[:, None]
    n = len(tl_t)
    e0 = float(tl_t[0]) - hop / 2
    F = np.zeros((n + 1, X_tl.shape[1]))
    np.cumsum(X_tl * hop, axis=0, out=F[1:])

    def integral(t: np.ndarray) -> np.ndarray:
        x = (t - e0) / hop
        k = np.clip(np.floor(x).astype(np.int64), 0, n - 1)
        return F[k] + X_tl[k] * ((x - k) * hop)[:, None]

    ct = np.asarray(cell_t, dtype=float)
    dt = np.maximum(np.diff(ct), 1e-12)
    return (integral(ct[1:]) - integral(ct[:-1])) / dt[:, None]


def cell_mean(frame_t: np.ndarray, X: np.ndarray, cell_t: np.ndarray) -> np.ndarray:
    """``to_timeline`` then ``pool``: frame scores → cell means [G, M]."""
    tl = timeline(float(cell_t[-1]))
    return pool(cell_t, tl, to_timeline(frame_t, X, tl))


# =============================================================================================
# Source registry (§5.9 step 1)
# =============================================================================================

@dataclass
class FrameSource:
    """A model source as linear building blocks: raw_s(t, c) = cols[t] · A[c]."""
    t: np.ndarray                     # [T] frame centres (s), offset-corrected
    cols: np.ndarray                  # [T, M] per-frame log columns
    A: np.ndarray                     # [V, M] coefficients over the full vocabulary
    aux: dict[str, np.ndarray] = field(default_factory=dict)   # [T, 12] side info (probabilities)


SourceFn = Callable[[dict[str, ChordEvidence], Vocab], FrameSource]
FRAME_SOURCES: dict[str, SourceFn] = {}


def register_source(name: str, fn: SourceFn) -> None:
    """Plug a frame-level model source in (``name`` = the part of ChordEvidence.name before "@")."""
    FRAME_SOURCES[name] = fn


def split_name(name: str) -> tuple[str, str]:
    """"lv@harmonic" → ("lv", "harmonic"); a bare name is the mix."""
    model, _, inp = name.partition("@")
    return model, (inp or "mix")


def merge_inputs(by_input: dict[str, ChordEvidence],
                 get: Callable[[ChordEvidence], dict[str, np.ndarray]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """p = Σ_i w_i·p@i / Σ_i w_i over the available inputs (0.6 harmonic, 0.4 mix; §5.9 step 1).

    The reference frames are those of the highest-weight input; other inputs are interpolated
    onto them (linearly, in probability)."""
    names = sorted(by_input, key=lambda k: -P.INPUT_WEIGHTS.get(k, 0.5))
    ref = by_input[names[0]]
    t = np.asarray(ref.frame_t, dtype=float)
    acc: dict[str, np.ndarray] = {}
    wsum = 0.0
    for nm in names:
        ev = by_input[nm]
        w = P.INPUT_WEIGHTS.get(nm, 0.5)
        arrs = get(ev)
        same = len(ev.frame_t) == len(t) and np.allclose(ev.frame_t, t)
        for k, a in arrs.items():
            a = np.asarray(a, dtype=float)
            if not same:
                a = to_timeline(ev.frame_t, a, t)
            acc[k] = acc.get(k, 0.0) + w * a
        wsum += w
    return t, {k: v / wsum for k, v in acc.items()}


def _vocab_key(vocab: Vocab) -> tuple:
    return tuple((c.root, c.quality, c.bass) for c in vocab.cands)


_A_CACHE: dict[tuple[str, int], tuple[Vocab, np.ndarray]] = {}


def _cached_A(name: str, vocab: Vocab, make: Callable[[Vocab], np.ndarray]) -> np.ndarray:
    key = (name, id(vocab))
    hit = _A_CACHE.get(key)
    if hit is not None and hit[0] is vocab:
        return hit[1]
    A = make(vocab)
    _A_CACHE[key] = (vocab, A)
    return A


# --- lv-chordia (§5.9 lv) ---------------------------------------------------------------------

LV_TRIAD = ("maj", "min", "sus4", "sus2", "dim", "aug")       # triad head: 1 + type·12 + root
TRIAD_TONES = {"maj": (0, 4, 7), "min": (0, 3, 7), "sus4": (0, 5, 7), "sus2": (0, 2, 7),
               "dim": (0, 3, 6), "aug": (0, 4, 8)}
LV_EXT: dict[str, tuple[str, ...]] = {
    "seventh": ("none", "7", "b7", "bb7"),
    "ninth": ("none", "9", "#9", "b9"),
    "eleventh": ("none", "11", "#11"),
    "thirteenth": ("none", "13", "b13"),
}
OPT9 = "none|9"     # an optional 9th: log(p_none + p_9)

#: Internal quality → (lv triad class, {extension head: class}); unlisted heads are "none".
#: "5" has no lv triad class (lv's training chops power chords to "unknown"): it scores
#: log(2·min(p_maj, p_min)) at its root, i.e. "no third" is supported only when the network
#: splits its third between major and minor. (M2 fix: the v1 score log((p_maj + p_min)/2) cost
#: only log 2 below a confidently major chord, so "5" was the runner-up in every cell and won
#: whenever a prior, PT or bass term nudged it — E5/B, F#5/C#, G#5/D# on 回家.)
#: Optional 9ths (m11, 13sus4, 13 — templates.py marks them "?") score log(p_none + p_9).
LV_QUALITY: dict[str, tuple[str, dict[str, str]]] = {
    "maj": ("maj", {}), "min": ("min", {}), "dim": ("dim", {}), "aug": ("aug", {}),
    "sus2": ("sus2", {}), "sus4": ("sus4", {}), "5": ("maj|min", {}),
    "6": ("maj", {"thirteenth": "13"}), "m6": ("min", {"thirteenth": "13"}),
    "7": ("maj", {"seventh": "b7"}), "maj7": ("maj", {"seventh": "7"}),
    "m7": ("min", {"seventh": "b7"}), "m7b5": ("dim", {"seventh": "b7"}),
    "dim7": ("dim", {"seventh": "bb7"}), "mM7": ("min", {"seventh": "7"}),
    "+maj7": ("aug", {"seventh": "7"}), "7sus4": ("sus4", {"seventh": "b7"}),
    "+7": ("aug", {"seventh": "b7"}), "add9": ("maj", {"ninth": "9"}),
    "madd9": ("min", {"ninth": "9"}), "6/9": ("maj", {"ninth": "9", "thirteenth": "13"}),
    "9": ("maj", {"seventh": "b7", "ninth": "9"}), "maj9": ("maj", {"seventh": "7", "ninth": "9"}),
    "m9": ("min", {"seventh": "b7", "ninth": "9"}),
    "m11": ("min", {"seventh": "b7", "ninth": OPT9, "eleventh": "11"}),
    "9sus4": ("sus4", {"seventh": "b7", "ninth": "9"}),
    "13sus4": ("sus4", {"seventh": "b7", "ninth": OPT9, "thirteenth": "13"}),
    "13": ("maj", {"seventh": "b7", "ninth": OPT9, "thirteenth": "13"}),
    "7b9": ("maj", {"seventh": "b7", "ninth": "b9"}), "7#9": ("maj", {"seventh": "b7", "ninth": "#9"}),
    "7b13": ("maj", {"seventh": "b7", "thirteenth": "b13"}),
    "maj7#11": ("maj", {"seventh": "7", "eleventh": "#11"}),
}
assert set(LV_QUALITY) == set(TT.QUALITIES)

# column layout of the lv source
_LV_TRI0, _LV_5, _LV_BASS = 0, 73, 85
_LV_EXT0: dict[str, int] = {}
_off = 98
for _h, _cl in LV_EXT.items():
    _LV_EXT0[_h] = _off
    _off += len(_cl)
_LV_OPT9 = _off
LV_NCOLS = _off + 1


def lv_triad_index(root: int, triad: str) -> int:
    """Index in the lv triad head [T, 73]: 0 = N, 1 + type·12 + root."""
    return 1 + LV_TRIAD.index(triad) * 12 + root % 12


def lv_coefficients(vocab: Vocab) -> np.ndarray:
    """A_lv [V, 113]: lv(t,c) = log p_triad + 0.5·log p_bass + Σ_h max(log 1e-4, log p_h)."""
    A = np.zeros((len(vocab.cands), LV_NCOLS))
    for i, c in enumerate(vocab.cands):
        if c.quality == TT.N:
            A[i, _LV_TRI0] = 1.0
            A[i, _LV_BASS] = P.MODEL_BASS_WEIGHT
            for h in LV_EXT:
                A[i, _LV_EXT0[h]] = 1.0
            continue
        triad, ext = LV_QUALITY[c.quality]
        if triad == "maj|min":
            A[i, _LV_5 + c.root] = 1.0
        else:
            A[i, _LV_TRI0 + lv_triad_index(c.root, triad)] = 1.0
        A[i, _LV_BASS + 1 + c.bass] = P.MODEL_BASS_WEIGHT
        for h, classes in LV_EXT.items():
            cl = ext.get(h, "none")
            if cl == OPT9:
                A[i, _LV_OPT9] = 1.0
            else:
                A[i, _LV_EXT0[h] + classes.index(cl)] = 1.0
    return A


def lv_columns(heads: dict[str, np.ndarray]) -> np.ndarray:
    """Per-frame lv columns [T, 113] from (merged) head probabilities."""
    tri = np.asarray(heads["triad"], dtype=float)
    T_ = tri.shape[0]
    out = np.empty((T_, LV_NCOLS))
    out[:, _LV_TRI0:_LV_TRI0 + 73] = np.log(np.maximum(tri, EPS))
    pmaj = tri[:, 1:13]
    pmin = tri[:, 13:25]
    out[:, _LV_5:_LV_5 + 12] = np.log(np.maximum(2.0 * np.minimum(pmaj, pmin), EPS))
    out[:, _LV_BASS:_LV_BASS + 13] = np.log(np.maximum(np.asarray(heads["bass"], dtype=float), EPS))
    floor = np.log(P.EXT_FLOOR)
    for h, classes in LV_EXT.items():
        ph = np.asarray(heads[h], dtype=float)
        if ph.shape[1] != len(classes):
            raise ValueError(f"lv head {h!r} has {ph.shape[1]} classes, expected {len(classes)}")
        out[:, _LV_EXT0[h]:_LV_EXT0[h] + len(classes)] = np.maximum(np.log(np.maximum(ph, EPS)), floor)
    p9 = np.asarray(heads["ninth"], dtype=float)
    out[:, _LV_OPT9] = np.maximum(np.log(np.maximum(p9[:, 0] + p9[:, 1], EPS)), floor)
    return out


def lv_root_mass(heads: dict[str, np.ndarray]) -> np.ndarray:
    """Σ over triad types of the triad head: the model's root mass [T, 12] (§5.12 step 6)."""
    tri = np.asarray(heads["triad"], dtype=float)
    return tri[:, 1:].reshape(len(tri), len(LV_TRIAD), 12).sum(axis=1)


def lv_tone_presence(heads: dict[str, np.ndarray]) -> np.ndarray:
    """Per-pc tone presence [T, 12] implied by the lv heads: triad tones of every (root, type)
    plus each extension class at the root mass (used for the "added tone is supported" tests of
    §5.12 steps 3 and 5)."""
    tri = np.asarray(heads["triad"], dtype=float)
    T_ = len(tri)
    pres = np.zeros((T_, 12))
    rm = np.zeros((T_, 12))
    for ti, typ in enumerate(LV_TRIAD):
        blk = tri[:, 1 + ti * 12:1 + (ti + 1) * 12]
        rm += blk
        for iv in TRIAD_TONES[typ]:
            pres += np.roll(blk, iv, axis=1)          # root r → pc r + iv
    for h, classes in LV_EXT.items():
        ph = np.asarray(heads[h], dtype=float)
        for k, cl in enumerate(classes):
            if cl != "none":
                pres += ph[:, k:k + 1] * np.roll(rm, TT.degree_semitones(cl), axis=1)
    return np.clip(pres, 0.0, 1.0)


_LV_HEADS = ("triad", "bass") + tuple(LV_EXT)


def _lv_source(by_input: dict[str, ChordEvidence], vocab: Vocab) -> FrameSource:
    t, heads = merge_inputs(by_input, lambda e: {h: e.heads[h] for h in _LV_HEADS})
    aux = {"root": lv_root_mass(heads)}
    harm = by_input.get("harmonic")
    hh = {h: harm.heads[h] for h in _LV_HEADS} if harm is not None else heads
    tones = lv_tone_presence(hh)
    if harm is not None and (len(harm.frame_t) != len(t) or not np.allclose(harm.frame_t, t)):
        tones = to_timeline(harm.frame_t, tones, t)
    aux["tones"] = tones
    return FrameSource(t, lv_columns(heads), _cached_A("lv", vocab, lv_coefficients), aux)


register_source("lv", _lv_source)


# --- consonance-ACE (§5.9 ace; M5) ------------------------------------------------------------

_TAU_LEVELS = (0.85, 0.5, 0.10)


def ace_coefficients(vocab: Vocab) -> np.ndarray:
    """A_ace [V, 62]: log root + 0.5·log bass + 0.5·Σ_pc floored Bernoulli(τ_c(pc)) note terms."""
    from ..theory import vocab as TV
    tau = TV.tau_matrix(vocab.cands, mode="ace")
    A = np.zeros((len(vocab.cands), 26 + 36))
    for i, c in enumerate(vocab.cands):
        A[i, 12 if c.quality == TT.N else c.root] = 1.0
        A[i, 13 + (12 if c.quality == TT.N else c.bass)] = P.MODEL_BASS_WEIGHT
        lev = np.abs(tau[i][:, None] - np.array(_TAU_LEVELS)[None, :]).argmin(axis=1)
        A[i, 26 + lev * 12 + np.arange(12)] = 0.5
    return A


def _ace_source(by_input: dict[str, ChordEvidence], vocab: Vocab) -> FrameSource:
    t, h = merge_inputs(by_input, lambda e: {"root": e.root, "bass": e.bass, "notes": e.notes})
    n = np.clip(h["notes"], EPS, 1 - 1e-9)
    floor = np.log(P.EXT_FLOOR)
    note_cols = [np.maximum(tau * np.log(n) + (1 - tau) * np.log1p(-n), floor) for tau in _TAU_LEVELS]
    cols = np.hstack([np.log(np.maximum(h["root"], EPS)), np.log(np.maximum(h["bass"], EPS))] + note_cols)
    return FrameSource(t, cols, _cached_A("ace", vocab, ace_coefficients))


register_source("ace", _ace_source)


# --- BTC-CL (§5.9 btc; M5) --------------------------------------------------------------------

def btc_label_sets(labels: list[str], vocab: Vocab) -> tuple[list[tuple[tuple[int, ...], float]], np.ndarray]:
    """M(r, q): the nearest BTC labels of each candidate (§5.9 btc; the bass is ignored): the
    labels on the same root whose tone set differs least from the candidate's template
    (symmetric difference; 0 = exact). Ties are averaged, so the column is
    log(mean_{l ∈ M} p_l): a "5" chord gets log((p_maj + p_min) / 2) as in the lv source, and a
    fallback can never outscore the exact chord. N maps to the N/X labels.
    Returns the distinct (label set, weight) pairs and the set index of every candidate."""
    from ..theory.harte import parse_harte_full
    parsed: list[tuple[int, frozenset[int]] | None] = []
    n_idx: list[int] = []
    for i, lab in enumerate(labels):
        try:
            pc = parse_harte_full(lab, strict=False)
        except Exception:                      # unknown shorthand: never matched
            parsed.append(None)
            continue
        if pc.quality == TT.N:
            n_idx.append(i)
            parsed.append(None)
        else:
            parsed.append((pc.root, TT.tones(pc.quality)))
    sets: dict[tuple[tuple[int, ...], float], int] = {}
    which = np.zeros(len(vocab.cands), dtype=np.int64)
    memo: dict[tuple[int, str], tuple[tuple[int, ...], float]] = {}
    for ci, c in enumerate(vocab.cands):
        key = (c.root, c.quality)
        if key not in memo:
            best: tuple[int, ...] = ()
            if c.quality != TT.N:
                tq = TT.tones(c.quality)
                dist = {i: len(tq ^ x[1]) for i, x in enumerate(parsed) if x is not None and x[0] == c.root}
                if dist:
                    d0 = min(dist.values())
                    best = tuple(i for i, d in dist.items() if d == d0)
            s = best or tuple(n_idx)
            memo[key] = (s, 1.0 / max(len(s), 1))
        which[ci] = sets.setdefault(memo[key], len(sets))
    return list(sets), which


def _btc_source(by_input: dict[str, ChordEvidence], vocab: Vocab) -> FrameSource:
    first = next(iter(by_input.values()))
    labels = list(first.labels or [])
    t, h = merge_inputs(by_input, lambda e: {"p": e.label_probs})
    sets, which = btc_label_sets(labels, vocab)
    cols = np.stack([np.log(np.maximum(wt * h["p"][:, list(s)].sum(axis=1), EPS)) if s
                     else np.full(len(t), np.log(EPS)) for s, wt in sets], axis=1)
    A = np.zeros((len(vocab.cands), len(sets)))
    A[np.arange(len(vocab.cands)), which] = 1.0
    return FrameSource(t, cols, A)


register_source("btc", _btc_source)


# =============================================================================================
# Cell-level sources: sym and bas
# =============================================================================================

def sym_raw(x: np.ndarray, vocab: Vocab, C: np.ndarray) -> np.ndarray:
    """sym(g,c) = Σ_pc [x·log τ' + (1 − x)·log(1 − τ')] with τ' = Vocab.tau (§5.9 sym)."""
    tau = np.clip(np.asarray(vocab.tau, dtype=float)[C], 1e-6, 1 - 1e-6)
    l1, l0 = np.log(tau), np.log1p(-tau)
    return np.asarray(x, dtype=float) @ (l1 - l0).T + l0.sum(axis=1)[None, :]


def sym_activity(x: np.ndarray) -> np.ndarray:
    """Per-cell activity of the ``sym`` source (M5): min(1, Σ_pc x / SYM_ACT_FULL).

    x is a decaying presence (τ = 0.4 s, §5.7): it falls to ≈ 0 about a second after the piano's
    last onset, during rests, and wherever the piano is silent. An all-zero x is not "no chord":
    as a full member of the weighted mean it would pull every held chord toward N and three-note
    templates. Like ``bas`` over a silent bass stem (M2 fix 3), the source fades out instead."""
    return np.clip(np.asarray(x, dtype=float).sum(axis=1) / max(P.SYM_ACT_FULL, 1e-9), 0.0, 1.0)


def p_bass_cells(bass: BassTrack | np.ndarray, grid: RhythmGrid) -> np.ndarray:
    """P_bass [G, 12] (§5.6). Uses ``notes.bass.cell_probs`` when WP5 provides it; otherwise
    the §5.6 formula on the bass notes (overlap · conf, plus 0.05 · uncovered / 12)."""
    if isinstance(bass, np.ndarray):
        Pb = np.asarray(bass, dtype=float)
    else:
        try:
            from ..notes.bass import cell_probs        # WP5
            Pb = np.asarray(cell_probs(bass, grid), dtype=float)
        except ImportError:
            Pb = _p_bass_from_notes(bass, grid)
    return Pb / np.maximum(Pb.sum(axis=1, keepdims=True), EPS)


def _p_bass_from_notes(bass: BassTrack, grid: RhythmGrid) -> np.ndarray:
    ct = np.asarray(grid.cell_t, dtype=float)
    G = len(ct) - 1
    a, b = ct[:-1], ct[1:]
    dt = np.maximum(b - a, 1e-9)
    acc = np.zeros((G, 12))
    cover = np.zeros(G)
    if bass.notes:
        for t0, t1, midi, conf, _src in bass.notes:
            ov = np.clip(np.minimum(b, t1) - np.maximum(a, t0), 0.0, None) / dt
            acc[:, int(midi) % 12] += ov * conf
            cover += ov
    elif bass.salience is not None and len(bass.frame_t):
        sal = cell_mean(bass.frame_t, bass.salience, ct)
        return sal + 1e-6
    acc += 0.05 * np.clip(1.0 - cover, 0.0, 1.0)[:, None] / 12.0
    return acc + 1e-9


def calibrate_scores(raw: np.ndarray, T: float, clip: float) -> np.ndarray:
    """ℓ = max(clip, log_softmax(raw / T)) along the last axis (§5.9 step 3)."""
    z = raw / max(float(T), 1e-6)
    return np.maximum(z - logsumexp(z, axis=-1, keepdims=True), clip)


def bass_coverage(bass: BassTrack | np.ndarray, grid: RhythmGrid) -> np.ndarray:
    """Fraction of each cell covered by tracked bass notes [G] (1 for a precomputed P_bass).

    Used as the per-cell *activity* of the ``bas`` source (M2 fix): where the bass stem is
    silent (回家's piano-only verse and pre-chorus, 0–110 s), the tracker has no notes and P_bass
    is the flat floor. A flat source carries no information but, as a member of the weighted
    mean with w_bas = 2, it used to shrink every other source's contribution to 1/3 there."""
    ct = np.asarray(grid.cell_t, dtype=float)
    G = len(ct) - 1
    if isinstance(bass, np.ndarray):
        return np.ones(G)
    a, b = ct[:-1], ct[1:]
    dt = np.maximum(b - a, 1e-9)
    cover = np.zeros(G)
    cover_lh = np.zeros(G)                      # M5: piano left-hand notes (§5.6 fallback)
    for t0, t1, _midi, _conf, src in bass.notes:
        g0 = max(0, int(np.searchsorted(ct, t0, side="right")) - 1)
        g1 = min(G, int(np.searchsorted(ct, t1, side="left")))
        if g1 > g0:
            ov = np.clip((np.minimum(b[g0:g1], t1) - np.maximum(a[g0:g1], t0)) / dt[g0:g1], 0.0, 1.0)
            (cover_lh if src == "amt_lh" else cover)[g0:g1] += ov
    if not bass.notes and bass.salience is not None and len(bass.frame_t):
        return np.ones(G)                       # salience-only track (no segmentation): trust it
    # the stem-level weight describes the bass STEM; the left hand lives exactly where it is quiet
    return np.clip(np.clip(cover, 0.0, 1.0) * bass_level_weight(bass, grid) + cover_lh, 0.0, 1.0)


def bass_level_weight(bass: BassTrack, grid: RhythmGrid) -> np.ndarray:
    """Per-cell reliability of the bass track from the stem level relative to the song's own loud
    bass level (95th percentile of energy_db): a soft step BASS_REL_DB below it (M2 fix).

    On 回家 the bass stem carries a non-tonal rumble near G1 (kick / riser bleed) at about −50 dBFS
    from 111 to 116.5 s, 30 dB under the real bass (−18 dBFS from 117 s); the tracker turns it into
    one held G note, which outvoted lv's E–D#–C#–B walk-down (C#°7/G, B+7/G). Quiet but real bass
    lines (−10 … −20 dB under the loud level) keep a weight near 1."""
    e = np.asarray(bass.energy_db, dtype=float)
    ft = np.asarray(bass.frame_t, dtype=float)
    if not len(e) or not len(ft):
        return np.ones(len(grid.cell_t) - 1)
    loud = float(np.percentile(e[np.isfinite(e)], 95)) if np.isfinite(e).any() else 0.0
    lev = cell_mean(ft, np.nan_to_num(e, nan=-120.0, neginf=-120.0), np.asarray(grid.cell_t, dtype=float))[:, 0]
    return 1.0 / (1.0 + np.exp(-(lev - (loud - P.BASS_REL_DB)) / P.BASS_REL_WIDTH_DB))


# =============================================================================================
# build
# =============================================================================================

def _by_model(evid: list[ChordEvidence]) -> dict[str, dict[str, ChordEvidence]]:
    by_model: dict[str, dict[str, ChordEvidence]] = {}
    for e in evid:
        m, inp = split_name(e.name)
        by_model.setdefault(m, {})[inp] = e
    return by_model


def build(grid: RhythmGrid, evid: list[ChordEvidence], sym: np.ndarray | None,
          bass: BassTrack | np.ndarray | None, vocab: Vocab, C: np.ndarray,
          p: DecoderParams) -> Evidence:
    """Calibrated cell evidence over the candidate set C (§5.9). ``bass`` may also be a
    precomputed P_bass [G, 12].

    M2 changes (see eval/reports/M2.md):
    - each source's log-softmax (§5.9 step 3) runs over the FULL vocabulary, not the pruned C.
      Over C it differs only by a per-cell constant, which cancels in every path comparison and
      bracket difference (only the −6 clip sees it); over the vocabulary the calibration no longer
      depends on pruning and ``subset`` can slice any C out of one build;
    - the weighted mean is taken per cell over the sources *active* in that cell (``aux:act:<s>``
      [G], 1 by default; ``bas``: the bass-note coverage of the cell, see ``bass_coverage``;
      ``sym`` (M5): the symbolic presence of the cell, see ``sym_activity``).
    ``aux:raw:<s>`` keeps the uncalibrated scores over the full vocabulary (for ``calibrate.fit``)."""
    C = np.asarray(C, dtype=np.int64)
    V = len(vocab.cands)
    full = build_full(grid, evid, sym, bass, vocab, p)
    if len(C) == V and np.array_equal(C, np.arange(V)):
        return full
    return subset(full, C)


def build_full(grid: RhythmGrid, evid: list[ChordEvidence], sym: np.ndarray | None,
               bass: BassTrack | np.ndarray | None, vocab: Vocab, p: DecoderParams) -> Evidence:
    """``build`` over the whole vocabulary (C = arange(V))."""
    V = len(vocab.cands)
    C = np.arange(V, dtype=np.int64)
    ct = np.asarray(grid.cell_t, dtype=float)
    G = len(ct) - 1
    u = cell_beats(grid)
    tl = timeline(float(ct[-1]))
    temp = lambda s: p.temp.get(s, P.DEFAULT_TEMP)      # noqa: E731

    parts: dict[str, np.ndarray] = {}
    aux: dict[str, np.ndarray] = {}
    act: dict[str, np.ndarray] = {}
    for model, by_input in _by_model(evid).items():
        fn = FRAME_SOURCES.get(model)
        if fn is None:
            warnings.warn(f"no evidence source registered for {model!r}; ignored", stacklevel=2)
            continue
        fs = fn(by_input, vocab)
        cols = pool(ct, tl, to_timeline(fs.t, fs.cols, tl))
        raw = cols @ fs.A.T
        aux[f"{AUX}raw:{model}"] = raw
        parts[model] = calibrate_scores(raw, temp(model), p.clip)
        for k, v in fs.aux.items():
            aux[f"{AUX}{model}:{k}"] = pool(ct, tl, to_timeline(fs.t, v, tl))

    if sym is not None:
        x = np.asarray(sym, dtype=float)
        raw = sym_raw(x, vocab, C)
        aux[f"{AUX}raw:sym"] = raw
        aux[f"{AUX}sym:x"] = x
        parts["sym"] = calibrate_scores(raw, temp("sym"), p.clip)
        act["sym"] = sym_activity(x)

    if bass is not None:
        Pb = p_bass_cells(bass, grid)
        raw12 = np.log(P.BAS_MIX * Pb + (1.0 - P.BAS_MIX) / 12.0)
        aux[f"{AUX}raw:bas"] = raw12
        l12 = calibrate_scores(raw12, temp("bas"), p.clip)
        bpc = np.array([c.bass for c in vocab.cands])
        lb = np.full((G, V), np.log(1.0 / 12.0))       # N: uniform over the 12 pcs
        ok = bpc >= 0
        lb[:, ok] = l12[:, bpc[ok]]
        parts["bas"] = lb
        act["bas"] = bass_coverage(bass, grid)
    for s, a in act.items():
        aux[f"{AUX}act:{s}"] = a
    aux[f"{AUX}beats"] = u

    E, E_short = _fuse(parts, act, u, p)
    parts.update(aux)
    return Evidence(C=C, E=E, E_short=E_short, parts=parts)


def _fuse(parts: dict[str, np.ndarray], act: dict[str, np.ndarray], u: np.ndarray,
          p: DecoderParams) -> tuple[np.ndarray, np.ndarray]:
    """E, E_short = κ_b·(Δg/ibi)·Σ w_s·a_s·ℓ_s / Σ w_s·a_s (per cell; a_s = source activity)."""
    G = len(u)

    def mean(boost: float) -> np.ndarray:
        num = None
        den = np.zeros(G)
        for s, l in parts.items():
            w = p.w_src.get(s, 1.0) * (boost if s in ("bas", "sym") else 1.0)
            wg = w * act[s] if s in act else np.full(G, w)
            num = wg[:, None] * l if num is None else num + wg[:, None] * l
            den += wg
        if num is None:
            return np.zeros((G, 0))
        return num / np.maximum(den, 1e-12)[:, None]

    scale = p.kappa_b * u[:, None]
    E = scale * mean(1.0)
    E_short = scale * mean(1.0 + p.w_short) if ("bas" in parts or "sym" in parts) else E.copy()
    return E, E_short


def subset(ev: Evidence, C: np.ndarray) -> Evidence:
    """The evidence of candidates C (vocabulary indices) sliced out of a build over a superset."""
    C = np.asarray(C, dtype=np.int64)
    pos = {int(c): k for k, c in enumerate(np.asarray(ev.C))}
    idx = np.array([pos[int(c)] for c in C], dtype=np.int64)
    K0 = len(ev.C)
    parts = {}
    for k, v in ev.parts.items():
        if v.ndim == 2 and v.shape[1] == K0 and (":" not in k or k.startswith(f"{AUX}raw:")) \
                and not k == f"{AUX}raw:bas":
            parts[k] = v[:, idx]
        else:
            parts[k] = v
    return Evidence(C=C, E=ev.E[:, idx], E_short=ev.E_short[:, idx], parts=parts)


def source_activity(ev: Evidence, name: str, c0: int, c1: int) -> float:
    """Mean activity of a source over cells [c0, c1) (1 for always-active sources)."""
    a = ev.parts.get(f"{AUX}act:{name}")
    if a is None or c1 <= c0:
        return 1.0
    return float(np.mean(a[c0:c1]))


def sources(ev: Evidence) -> list[str]:
    """Names of the calibrated [G, K] sources in ``ev.parts`` (auxiliary keys excluded)."""
    return [k for k in ev.parts if ":" not in k]


def bracket(ev: Evidence, grid: RhythmGrid, p: DecoderParams) -> np.ndarray:
    """The per-beat bracket B(g, c) = E / (κ_b·Δg/ibi_g) (§5.9 step 4)."""
    return ev.E / (p.kappa_b * cell_beats(grid)[:, None])


# =============================================================================================
# prune (§5.8)
# =============================================================================================

@lru_cache(maxsize=4)
def _index_of(vkey: tuple) -> dict[tuple[int, str, int], int]:
    return {t: i for i, t in enumerate(vkey)}


def _vocab_index(vocab: Vocab) -> dict[tuple[int, str, int], int]:
    return _index_of(_vocab_key(vocab))


def bridges(main: DecodeResult, vocab: Vocab) -> set[int]:
    """Idiom completion (§5.8 pass 2): for every adjacent pair of main chords A → B add
    #x°7 → x (the °7 a semitone below root(B)), V7/B, A and B over each pc of the stepwise bass
    path, the first-inversion major triad on each path pc (F#/A# in B → F#/A# → G#m; an
    addition to §5.8, see the WP7 report), Bsus4, Aaug and A+7 — each with its whole class."""
    idx = _vocab_index(vocab)
    out: set[int] = set()
    segs = sorted(main.segments, key=lambda s: s.c0)

    def add(r: int, q: str, b: int) -> None:
        i = idx.get((r % 12, q, b % 12))
        if i is not None:
            out.update(vocab.classes[vocab.cands[i].cls])

    for sa, sb in zip(segs, segs[1:]):
        A, B = vocab.cands[sa.cand], vocab.cands[sb.cand]
        if A.quality == TT.N or B.quality == TT.N or sa.cand == sb.cand:
            continue
        add(B.root - 1, "dim7", B.root - 1)
        add(B.root + 7, "7", B.root + 7)
        add(B.root, "sus4", B.root)
        add(A.root, "aug", A.root)
        add(A.root, "+7", A.root)
        d = (B.bass - A.bass) % 12
        d = d - 12 if d > 6 else d
        if 2 <= abs(d) <= 4:
            step = 1 if d > 0 else -1
            for k in range(1, abs(d)):
                x = (A.bass + step * k) % 12
                add(A.root, A.quality, x)
                add(B.root, B.quality, x)
                add(x - 4, "maj", x)
    return out


def prune(ev_all: Evidence, vocab: Vocab, main: DecodeResult | None, p: DecoderParams) -> np.ndarray:
    """Candidate set C (vocabulary indices, sorted) per §5.8.

    1. C0: every member of the top-``prune_top`` classes (ranked by their best member's E) in
       each cell, and — the ±2 neighbourhood — of the top classes by E summed over cell ± 2.
    2. With ``main``: idiom-completion bridges (``bridges``), never dropped.
    3. Cap ``prune_cap``: whole classes are dropped from the bottom of the ranking. N is always
       kept. A class that does not fit is skipped, and smaller classes further down may still fit.

    Ranking (M2 fix): first every class that is the best in at least one cell, then the rest;
    inside each tier by the *soft cell count* Σ_g exp(B(g, class) − max B(g)) (B = per-beat
    bracket: how many cells the class wins, near-misses counted partially). The v1 rank — the
    best margin a class reaches in ANY cell — ties every class that wins a cell at 0 (48 classes
    on 回家), and the old loop stopped at the first class that did not fit: on 回家 it dropped
    B major (the tonic, top-1 in 145 of 1,086 cells) while the B7 bridge survived, so B was
    decoded as B7, B5/F# or E5/B. The soft count alone is not enough either: a passing chord
    wins only its own 2–4 cells and would lose its place to never-winning variants of the long
    chords (10,000 Reasons: Em/B dropped, the B°7 bridge decoded over a B bass instead).
    """
    C_all = np.asarray(ev_all.C, dtype=np.int64)
    E = np.asarray(ev_all.E, dtype=float)
    G = E.shape[0]
    cls_col = np.array([vocab.cands[c].cls for c in C_all])
    order = np.argsort(cls_col, kind="stable")
    cs = cls_col[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    cls_ids = cs[starts]
    Ecls = np.maximum.reduceat(E[:, order], starts, axis=1)          # [G, n_cls present]
    n_top = min(int(p.prune_top), Ecls.shape[1])

    keep = set(np.unique(np.argpartition(-Ecls, n_top - 1, axis=1)[:, :n_top]).tolist())
    cum = np.vstack([np.zeros((1, Ecls.shape[1])), np.cumsum(Ecls, axis=0)])
    g = np.arange(G)
    win = cum[np.minimum(g + 3, G)] - cum[np.maximum(g - 2, 0)]
    keep |= set(np.unique(np.argpartition(-win, n_top - 1, axis=1)[:, :n_top]).tolist())

    rel = Ecls - Ecls.max(axis=1, keepdims=True)
    margin = rel.max(axis=0)
    per_beat = np.maximum(float(p.kappa_b) * _cell_beats_from_E(ev_all, G), 1e-9)
    soft = np.exp(rel / per_beat[:, None]).sum(axis=0)
    wins = margin >= -1e-9                       # best class in at least one cell
    ranked = sorted(keep, key=lambda k: (not wins[k], -soft[k], -margin[k]))
    in_all = set(C_all.tolist())

    out: set[int] = {i for i, c in enumerate(vocab.cands) if c.quality == TT.N}
    if main is not None:
        out |= {i for i in bridges(main, vocab) if i in in_all}
    for k in ranked:
        members = [m for m in vocab.classes[cls_ids[k]] if m in in_all and m not in out]
        if len(out) + len(members) > p.prune_cap:
            continue
        out.update(members)
    return np.array(sorted(out), dtype=np.int64)


def _cell_beats_from_E(ev: Evidence, G: int) -> np.ndarray:
    """Δg/ibi per cell when the evidence carries it (``aux:beats``), else 1 (ranking only)."""
    u = ev.parts.get(f"{AUX}beats")
    return np.asarray(u, dtype=float) if u is not None and len(u) == G else np.ones(G)
