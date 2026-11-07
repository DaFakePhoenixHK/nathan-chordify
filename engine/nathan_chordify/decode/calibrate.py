"""WP7 calibration (DESIGN §5.9): temperatures T_s by NLL, and the break-even table.

``fit`` needs the uncalibrated scores that ``evidence.build`` keeps in ``ev.parts`` under
``aux:raw:<src>`` and per-cell ground truth (vocabulary indices). The likelihood is class-level
(§5.8): a cell's target is every member of the GT chord's equivalence class present in C. The
bass source is fitted on the GT bass pc over its 12-way softmax.

``break_even_table`` measures, with the real decoder, the per-beat bracket advantage δ* (nats
of B(g, P) − B(g, A) per beat) at which a passing chord P is decoded, for the patterns A–P–B
and A–P–A, P lengths ¼, ½ and 1 beat, and each metric position of P's start (§5.9 targets:
≈0.7 for a half-beat A–P–B, ≈1.3 for a half-beat A–P–A). ``synthetic_grid`` builds the metered
grid it (and the unit tests) run on.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ..types import DecoderParams, Evidence, KeyTrack, RhythmGrid, Segment, Vocab
from . import params as P
from .evidence import cell_beats


# =============================================================================================
# Temperatures
# =============================================================================================

def gt_cells(gt: np.ndarray | list[Segment], G: int) -> np.ndarray:
    """Ground truth as one vocabulary index per cell (−1 = unlabelled)."""
    if isinstance(gt, np.ndarray):
        return gt.astype(np.int64)
    out = np.full(G, -1, dtype=np.int64)
    for s in gt:
        out[s.c0:s.c1] = s.cand
    return out


def _nll(raw: np.ndarray, mask: np.ndarray, w: np.ndarray, T: float) -> float:
    z = raw / T
    tgt = logsumexp(np.where(mask, z, -np.inf), axis=1)
    return float(np.sum(w * (logsumexp(z, axis=1) - tgt)))


def fit(evidence_sets: list[Evidence], gt: list[np.ndarray | list[Segment]], *,
        vocab: Vocab | None = None, sources: list[str] | None = None,
        bounds: tuple[float, float] = (0.05, 20.0),
        weights: list[np.ndarray] | None = None) -> dict[str, float]:
    """Temperature T_s per source minimising the class-level NLL of the GT over all cells.

    ``evidence_sets[i]`` and ``gt[i]`` describe one song; ``weights[i]`` [G] optionally weights
    cells (e.g. by their length in beats)."""
    if vocab is None:
        from ..theory.vocab import candidates
        vocab = candidates()
    names = sources or sorted({k.split(":", 2)[2] for ev in evidence_sets for k in ev.parts
                               if k.startswith("aux:raw:")})
    out: dict[str, float] = {}
    for s in names:
        raws, masks, ws = [], [], []
        for n, (ev, g) in enumerate(zip(evidence_sets, gt)):
            raw = ev.parts.get(f"aux:raw:{s}")
            if raw is None:
                continue
            cells = gt_cells(g, raw.shape[0])
            ok = cells >= 0
            if s == "bas":
                bpc = np.array([vocab.cands[c].bass if c >= 0 else -1 for c in cells])
                ok &= bpc >= 0
                mask = np.zeros(raw.shape, dtype=bool)
                mask[np.flatnonzero(ok), bpc[ok]] = True
            else:
                ccls = np.array([vocab.cands[c].cls for c in ev.C])
                gcls = np.array([vocab.cands[c].cls if c >= 0 else -1 for c in cells])
                mask = ccls[None, :] == gcls[:, None]
                ok &= mask.any(axis=1)
            w = np.ones(raw.shape[0]) if weights is None else np.asarray(weights[n], dtype=float)
            raws.append(raw[ok])
            masks.append(mask[ok])
            ws.append(w[ok])
        if not raws or sum(len(r) for r in raws) == 0:
            continue
        def obj(logT: float) -> float:
            T = float(np.exp(logT))
            return sum(_nll(r, m, w, T) for r, m, w in zip(raws, masks, ws))
        res = minimize_scalar(obj, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
        out[s] = float(np.exp(res.x))
    return out


# =============================================================================================
# Synthetic grid
# =============================================================================================

def synthetic_grid(bpm: float = 65.0, n_bars: int = 4, beats_per_bar: int = 4, subdiv: int = 4,
                   t0: float = 0.0) -> RhythmGrid:
    """A metered grid with constant tempo, meter and subdivision (cell_pos as in §5.3)."""
    ibi = 60.0 / bpm
    B = n_bars * beats_per_bar
    beats = t0 + np.arange(B) * ibi
    G = B * subdiv
    pos = np.empty(G, dtype=np.int8)
    for g in range(G):
        b, k = divmod(g, subdiv)
        if k == 0:
            bib = b % beats_per_bar
            pos[g] = 0 if bib == 0 else (1 if beats_per_bar % 2 == 0 and bib == beats_per_bar // 2 else 2)
        elif subdiv in (2, 3):
            pos[g] = 3
        elif subdiv == 4:
            pos[g] = 3 if k == 2 else 4
        else:
            pos[g] = 3 if k % 2 == 0 else 4
    return RhythmGrid(
        beats=beats, beat_conf=np.ones(B), bar_of_beat=np.arange(B) // beats_per_bar,
        beat_in_bar=np.arange(B) % beats_per_bar + 1,
        bars=[(b * beats_per_bar, beats_per_bar) for b in range(n_bars)],
        meter=[(0, beats_per_bar, 4)], bpm=float(bpm), ibi_ref=ibi,
        subdiv=np.full(B, subdiv, dtype=np.int64), free=[],
        cell_t=t0 + np.arange(G + 1) * ibi / subdiv, cell_beat=np.arange(G + 1) / subdiv,
        cell_pos=pos)


# =============================================================================================
# Break-even table
# =============================================================================================

#: Neutral chords (key C, priors off): A = C, P = G#+ (passing-type), B = F; no bass steps.
NEUTRAL = {"key": 0, "A": (0, "maj", 0), "P": (8, "aug", 8), "B": (5, "maj", 5)}
#: 回家 idioms (key B, full priors): B → F#/A# → G#m (A–P–B) and B → E/B → B (A–P–A).
HUI_JIA = {"key": 11, "A": (11, "maj", 11), "P": (6, "maj", 10), "B": (8, "min", 8),
           "P_apa": (4, "maj", 11)}
POSITIONS = {0: "downbeat", 1: "mid-bar beat", 2: "beat", 3: "half", 4: "quarter"}


def _start_cell(pos: int, subdiv: int) -> int:
    """P's first cell in bar 2 of the synthetic grid for a metric position."""
    bar2 = 8 * subdiv
    return {0: bar2, 1: bar2 + 2 * subdiv, 2: bar2 + subdiv,
            3: bar2 + subdiv + subdiv // 2, 4: bar2 + subdiv + 1}[pos]


def _p_decoded(delta: float, pattern: str, n_cells: int, s0: int, grid: RhythmGrid, vocab: Vocab,
               C: np.ndarray, cols: dict[str, int], key: KeyTrack, p: DecoderParams,
               context: int, other: float) -> bool:
    from .hsmm import decode
    G = len(grid.cell_t) - 1
    K = len(C)
    Bk = np.full((G, K), other)
    s1 = s0 + n_cells
    a, b, pc = cols["A"], cols["B"], cols["P"]
    Bk[:s0, a] = 0.0
    if pattern == "A-P-B":
        Bk[s1:, b] = 0.0
        Bk[s0:s1, b] = 0.0
    else:
        Bk[s1:, a] = 0.0
    Bk[s0:s1, a] = 0.0
    Bk[s0:s1, pc] = delta
    E = p.kappa_b * cell_beats(grid)[:, None] * Bk
    ev = Evidence(C=C, E=E, E_short=E, parts={})
    r0, r1 = s0 - context, s1 + context
    res = decode(ev, grid, key, vocab, p, "detail", region=(r0, r1), marginals=False)
    covered = sum(min(s.c1, s1) - max(s.c0, s0) for s in res.segments
                  if s.cand == int(C[pc]) and s.c1 > s0 and s.c0 < s1)
    return covered * 2 >= n_cells


def break_even(pattern: str, length_beats: float, pos: int, p: DecoderParams | None = None,
               chords: dict | None = None, *, bpm: float = 65.0, subdiv: int = 4,
               context_beats: int = 4, other: float = -3.0, lo: float = -4.0, hi: float = 10.0,
               tol: float = 0.005) -> float:
    """δ* (nats per beat) for one pattern / P length / metric position (bisection on δ).

    ``chords`` = {"key", "A", "P", "B"[, "P_apa"]} as (root, quality, bass) triples; None →
    ``NEUTRAL`` with w_prior = 0 (the structural cost of λ, the duration prior and the PT
    bonus). All other candidates sit at ``other`` nats/beat below. Returns +inf if P is never
    decoded below ``hi`` and −inf if it is decoded even at ``lo``."""
    from ..theory.vocab import candidates
    p = p or DecoderParams()
    if chords is None:
        chords, p = NEUTRAL, replace(p, w_prior=0.0)
    vocab = candidates()
    idx = {(c.root, c.quality, c.bass): i for i, c in enumerate(vocab.cands)}
    pt = chords.get("P_apa", chords["P"]) if pattern == "A-P-A" else chords["P"]
    trip = {"A": chords["A"], "P": pt, "B": chords["B"]}
    vidx = {k: idx[v] for k, v in trip.items()}
    C = np.array(sorted({0, *vidx.values()}), dtype=np.int64)
    cols = {k: int(np.flatnonzero(C == v)[0]) for k, v in vidx.items()}
    grid = synthetic_grid(bpm, n_bars=4, subdiv=subdiv)
    key = KeyTrack(chords["key"] % 12, "major", 1.0, np.full(len(grid.beats), chords["key"]), [])
    n_cells = max(1, int(round(length_beats * subdiv)))
    s0 = _start_cell(pos, subdiv)
    ctx = context_beats * subdiv
    f = lambda d: _p_decoded(d, pattern, n_cells, s0, grid, vocab, C, cols, key, p, ctx, other)  # noqa: E731
    if f(lo):
        return float("-inf")
    if not f(hi):
        return float("inf")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if f(mid) else (mid, hi)
    return round(0.5 * (lo + hi), 3)


def break_even_table(p: DecoderParams | None = None, chords: dict | None = None, *,
                     patterns: tuple[str, ...] = ("A-P-B", "A-P-A"),
                     lengths: tuple[float, ...] = (0.25, 0.5, 1.0),
                     positions: tuple[int, ...] = (0, 1, 2, 3, 4), **kw) -> list[dict]:
    """Rows {pattern, length, pos, position, delta} of the §5.9 break-even table."""
    rows = []
    for pat in patterns:
        for ln in lengths:
            for pos in positions:
                if pos == 4 and ln >= 0.5:
                    continue            # a ≥½-beat chord starting on a 16th: same costs as pos 3
                rows.append({"pattern": pat, "length": ln, "pos": pos, "position": POSITIONS[pos],
                             "delta": break_even(pat, ln, pos, p, chords, **kw)})
    return rows


def fit_lam_scale(target: float = 0.7, p: DecoderParams | None = None, chords: dict | None = None, *,
                  pattern: str = "A-P-B", length_beats: float = 0.5, pos: int = 2,
                  lo: float = 0.2, hi: float = 2.0, tol: float = 1e-3) -> float:
    """λ_scale at which δ*(pattern, length, pos) = target (κ_b fixed; §5.9 "λ_scale, the priors
    and the PT bonus are tuned relative to it"). δ* rises monotonically with λ_scale."""
    p = p or DecoderParams()
    f = lambda s: break_even(pattern, length_beats, pos, replace(p, lam_scale=s), chords)  # noqa: E731
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return round(0.5 * (lo + hi), 4)


def format_table(rows: list[dict]) -> str:
    """A plain-text rendering of ``break_even_table`` rows."""
    lines = [f"{'pattern':8} {'P len':>6}  {'P starts on':13} {'δ* (nats/beat)':>14}"]
    for r in rows:
        lines.append(f"{r['pattern']:8} {r['length']:>6.2f}  {r['position']:13} {r['delta']:>14.3f}")
    return "\n".join(lines)
