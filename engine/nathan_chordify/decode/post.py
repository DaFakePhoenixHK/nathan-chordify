"""WP7 post-processing (DESIGN §5.12 steps 1–6 and 11). Spelling, numbers and voicings
(steps 7–10, 12) belong to ``theory``; here only cand / cls / layer / role / min_detail /
alternatives change.

Neighbours: a main segment's neighbours are the adjacent main segments; a passing segment's
neighbours are the adjacent pieces of the "Everything" timeline (main segments with the
passing overlays carved in).

1. **Bass events** (review A8). P (a passing segment of at most 2 beats or a main segment of at
   most 1 beat, and shorter than N — M2 fix: without it a 4-beat I next to vi7 became vi7/1 and a
   2-beat main iii next to Imaj7 became Imaj7/3, because the longer chord's tones contain the
   shorter one's) is a bass event of a neighbour N when only the bass differs:
   bass(P) ≠ bass(N), P's non-bass pcs ⊆ pcs(N) *and* N's non-bass pcs ⊆ pcs(P). (The second
   half is our reading of "with only the bass different"; without it every I → vi change would
   become I6/vi.) The bass must also be *new*: when bass(P) is the other neighbour's bass, the
   bass is just held over (or anticipated) under the new chord — the transition phantom of
   step 2 (B → Emaj7 read as Emaj7/B), which step 1 would otherwise swallow. P is relabelled to N over P's bass (the smallest vocabulary chord on N's root
   whose template contains N's and has that bass as a chord tone: C over B → Cmaj7/B, shown as
   C/B through step 5) and gets ``role = "bass_event"``; it is never relabelled again. Shorter
   than min(½ beat, ``bass_event_max_s``) it is a **bass passing tone**: N's main segment is
   extended over it and P stays as a passing overlay with min_detail 1.0 (Everything).
2. **Transition phantoms** (A6): a passing P ≤ ½ beat with pcs(P) ⊆ pcs(A) ∪ pcs(B),
   bass(P) ∈ {bass(A), bass(B)} and no new onset outside A ∪ B (``onsets`` [G,12], e.g. the
   symbolic dch, when given) is merged into the neighbour sharing the most pcs.
3. **A–P–A neighbours** (A7) are never deleted; P ≤ 1 beat between two pieces of the same class
   keeps its min_detail when whitelisted, else min_detail += ``demote`` (cap 1). Whitelist: a
   different root over the same bass or a bass ±1–2 semitones away; IV / V (any bass) / vii°
   of A; a same-root sus4 / sus2 / add9 / 7 whose added tone is supported (symbolic presence or
   lv@harmonic tone presence ≥ 0.5). Same-root neighbours need the support test even though
   they share A's bass — otherwise the support rule could never apply.
4. **Merge** adjacent segments with the same label (and drop overlays equal to their host).
5. **7th only in the bass**: ``seventh_in_upper`` decides it per segment (the Segment type has
   no field to store it yet; see the WP7 report).
6. **Display member** of each class: the member with the best w_prior·(prior(L, m) +
   prior(m, R)) + log(root-head mass of m's root), the prior taken on m's root-position proxy
   (members share bass and pcs; see ``_member_score``).
11. **Alternatives**: the top 3 other classes by posterior mass, each with its display member.
"""
from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass

import numpy as np

from ..theory import priors as PR
from ..theory import templates as TT
from ..theory.simplify import seventh_interval
from ..types import BassTrack, Candidate, DecoderParams, Evidence, KeyTrack, RhythmGrid, Segment, Vocab
from . import params as P
from .evidence import cell_beats
from .hsmm import beat_of_cells

Piece = tuple[int, int, Segment]


@dataclass
class _Ctx:
    grid: RhythmGrid
    vocab: Vocab
    ev: Evidence
    key: KeyTrack | None
    p: DecoderParams
    ucum: np.ndarray
    ct: np.ndarray
    beat_of: np.ndarray
    index: dict[tuple[int, str, int], int]

    def beats(self, s: Segment) -> float:
        return float(self.ucum[s.c1] - self.ucum[s.c0])

    def secs(self, s: Segment) -> float:
        return float(self.ct[s.c1] - self.ct[s.c0])

    def key_at(self, g: int) -> int:
        k = self.key
        if k is None or not len(k.per_beat):
            return -1
        b = int(self.beat_of[min(max(g, 0), len(self.beat_of) - 1)])
        return int(k.per_beat[min(b, len(k.per_beat) - 1)])

    def cand(self, s: Segment) -> Candidate:
        return self.vocab.cands[s.cand]

    def set_cand(self, s: Segment, i: int) -> None:
        s.cand, s.cls = int(i), self.vocab.cands[i].cls

    def set_span(self, s: Segment, c0: int, c1: int) -> None:
        s.c0, s.c1, s.t0, s.t1 = int(c0), int(c1), float(self.ct[c0]), float(self.ct[c1])

    def aux_mean(self, name: str, s: Segment) -> np.ndarray | None:
        a = self.ev.parts.get(name)
        if a is None or s.c1 <= s.c0:
            return None
        return np.asarray(a[s.c0:s.c1], dtype=float).mean(axis=0)


def _ctx(grid, vocab, ev, key, p) -> _Ctx:
    return _Ctx(grid, vocab, ev, key, p, np.r_[0.0, np.cumsum(cell_beats(grid))],
                np.asarray(grid.cell_t, dtype=float), beat_of_cells(grid),
                {(c.root, c.quality, c.bass): i for i, c in enumerate(vocab.cands)})


# =============================================================================================
# Timeline helpers
# =============================================================================================

def _mains(segs: list[Segment]) -> list[Segment]:
    return sorted((s for s in segs if s.layer == "main"), key=lambda s: s.c0)


def _overlays(segs: list[Segment]) -> list[Segment]:
    return sorted((s for s in segs if s.layer == "passing"), key=lambda s: s.c0)


def pieces(segs: list[Segment]) -> list[Piece]:
    """The Everything timeline: (c0, c1, segment) with the passing overlays carved into the main
    layer. Overlapping overlays keep the earlier one."""
    over: list[Segment] = []
    for o in _overlays(segs):
        if not over or o.c0 >= over[-1].c1:
            over.append(o)
    out: list[Piece] = [(o.c0, o.c1, o) for o in over]
    for m in _mains(segs):
        cur = m.c0
        for o in over:
            if o.c1 <= cur or o.c0 >= m.c1:
                continue
            if o.c0 > cur:
                out.append((cur, o.c0, m))
            cur = max(cur, o.c1)
        if cur < m.c1:
            out.append((cur, m.c1, m))
    return sorted(out, key=lambda x: x[0])


def neighbours(s: Segment, segs: list[Segment], pcs: list[Piece] | None = None
               ) -> tuple[Segment | None, Segment | None]:
    """Adjacent main segments for a main segment; adjacent Everything pieces for an overlay."""
    if s.layer == "main":
        ms = _mains(segs)
        prev = next((m for m in reversed(ms) if m.c1 == s.c0 and m is not s), None)
        nxt = next((m for m in ms if m.c0 == s.c1 and m is not s), None)
        return prev, nxt
    pcs = pieces(segs) if pcs is None else pcs
    prev = next((x[2] for x in reversed(pcs) if x[1] == s.c0 and x[2] is not s), None)
    nxt = next((x[2] for x in pcs if x[0] == s.c1 and x[2] is not s), None)
    return prev, nxt


def _absorb(ctx: _Ctx, segs: list[Segment], host: Segment, c0: int, c1: int) -> None:
    """Make main segment ``host`` cover cells [c0, c1), trimming the other main segments."""
    lo, hi = min(host.c0, c0), max(host.c1, c1)
    for m in _mains(segs):
        if m is host or m.c1 <= lo or m.c0 >= hi:
            continue
        if m.c0 < lo:
            ctx.set_span(m, m.c0, lo)
        elif m.c1 > hi:
            ctx.set_span(m, hi, m.c1)
        else:
            segs.remove(m)
    ctx.set_span(host, lo, hi)


def _host(segs: list[Segment], s: Segment) -> Segment | None:
    """The main segment with the largest overlap with s."""
    best, ov = None, 0
    for m in _mains(segs):
        o = min(m.c1, s.c1) - max(m.c0, s.c0)
        if o > ov:
            best, ov = m, o
    return best


# =============================================================================================
# Step 1: bass events
# =============================================================================================

def upper(c: Candidate) -> frozenset[int]:
    """Non-bass pitch classes."""
    return c.pcs - {c.bass}


def is_bass_event(pc: Candidate, nc: Candidate, other: Candidate | None = None) -> bool:
    """Only the bass differs between P and its neighbour N, and P's bass is not just the other
    neighbour's bass held over or anticipated (see the module doc)."""
    if TT.N in (pc.quality, nc.quality) or pc.bass == nc.bass:
        return False
    if other is not None and other.quality != TT.N and other.bass == pc.bass:
        return False
    return upper(pc) <= nc.pcs and upper(nc) <= pc.pcs and len(upper(pc)) >= 2


def over_bass(vocab: Vocab, index: dict, n: Candidate, bass: int) -> int | None:
    """N's chord over another bass: (root, quality, bass) itself, else the smallest quality on
    N's root whose template contains N's and has the bass as a non-extension chord tone."""
    i = index.get((n.root, n.quality, bass % 12))
    if i is not None:
        return i
    iv = (bass - n.root) % 12
    need = TT.tones(n.quality)
    best = None
    for q in TT.QUALITIES:
        if need <= TT.tones(q) and iv in TT.chord_bass_intervals(q):
            j = index.get((n.root, q, bass % 12))
            if j is not None and (best is None or len(TT.tones(q)) < best[0]):
                best = (len(TT.tones(q)), j)
    return None if best is None else best[1]


def is_bass_passing_tone(s: Segment, grid: RhythmGrid, p: DecoderParams) -> bool:
    """A bass event shorter than min(½ beat, bass_event_max_s): goes to ``bass_line`` with
    role "passing_tone" (Detailed and up) and to the grid as N/bass at Everything."""
    if s.role != "bass_event":
        return False
    u = cell_beats(grid)
    ct = np.asarray(grid.cell_t, dtype=float)
    return float(u[s.c0:s.c1].sum()) < 0.5 - 1e-6 and float(ct[s.c1] - ct[s.c0]) < p.bass_event_max_s - 1e-9


def _bass_events(segs: list[Segment], ctx: _Ctx) -> list[Segment]:
    order = _overlays(segs) + _mains(segs)
    for x in order:
        if x not in segs or x.role != "chord":
            continue
        xc = ctx.cand(x)
        limit = P.PASSING_MAX_BEATS if x.layer == "passing" else P.BASS_EVENT_MAX_BEATS
        if xc.quality == TT.N or ctx.beats(x) > limit + 1e-6:
            continue                    # M2: a bass move under a HELD chord, not a chord of its own
        nb = neighbours(x, segs)
        for side, n in enumerate(nb):
            if n is None or n.role != "chord" or ctx.beats(x) >= ctx.beats(n) - 1e-6:
                continue                # … that is held longer than the move itself
            nc = ctx.cand(n)
            other = nb[1 - side]
            if not is_bass_event(xc, nc, ctx.cand(other) if other is not None else None):
                continue
            new = over_bass(ctx.vocab, ctx.index, nc, xc.bass)
            if new is None:
                continue
            x.alternatives = [(x.cand, x.conf)] + list(x.alternatives)
            ctx.set_cand(x, new)
            x.role = "bass_event"
            short = ctx.beats(x) < 0.5 - 1e-6 and ctx.secs(x) < ctx.p.bass_event_max_s - 1e-9
            if short:
                main_n = n if n.layer == "main" else None
                if x.layer == "main" and main_n is not None:
                    x.layer = "passing"
                    _absorb(ctx, segs, main_n, x.c0, x.c1)
                elif x.layer == "passing" and main_n is not None:
                    _absorb(ctx, segs, main_n, x.c0, x.c1)
                x.layer, x.min_detail = "passing", 1.0
            break
    return segs


# =============================================================================================
# Step 2: transition phantoms
# =============================================================================================

def _phantoms(segs: list[Segment], ctx: _Ctx, onsets: np.ndarray | None) -> list[Segment]:
    changed = True
    while changed:
        changed = False
        pcs_ = pieces(segs)
        for x in _overlays(segs):
            if x.role != "chord" or ctx.beats(x) > P.PHANTOM_MAX_BEATS + 1e-6:
                continue
            a, b = neighbours(x, segs, pcs_)
            if a is None or b is None:
                continue
            xc, ac, bc = ctx.cand(x), ctx.cand(a), ctx.cand(b)
            if TT.N in (xc.quality, ac.quality, bc.quality):
                continue
            both = ac.pcs | bc.pcs
            if not (xc.pcs <= both and xc.bass in (ac.bass, bc.bass)):
                continue
            if onsets is not None:
                outside = [pc for pc in range(12) if pc not in both]
                if outside and np.any(np.asarray(onsets)[x.c0:x.c1][:, outside] > P.ONSET_MIN):
                    continue
            tgt = a if len(xc.pcs & ac.pcs) >= len(xc.pcs & bc.pcs) else b
            segs.remove(x)
            if tgt.layer == "passing":
                ctx.set_span(tgt, min(tgt.c0, x.c0), max(tgt.c1, x.c1))
            else:
                _absorb(ctx, segs, tgt, x.c0, x.c1)
            changed = True
            break
    return segs


# =============================================================================================
# Step 3: A–P–A neighbours
# =============================================================================================

_ADDED_TONE = {"sus4": 5, "sus2": 2, "add9": 2, "madd9": 2, "7": 10, "m7": 10}
_IV_CORES = frozenset({"maj", "maj7", "6", "7", "min", "m6", "m7"})
_V_CORES = frozenset({"maj", "7", "sus4", "7sus4"})


def supported(ctx: _Ctx, s: Segment, pc: int) -> bool:
    """Symbolic presence ≥ 0.5 or lv@harmonic tone presence ≥ 0.5 for pc over s."""
    for name in ("aux:sym:x", "aux:lv:tones"):
        m = ctx.aux_mean(name, s)
        if m is not None and m[pc % 12] >= P.SUPPORT_MIN:
            return True
    return False


def neighbour_whitelisted(ctx: _Ctx, x: Segment, a: Segment) -> bool:
    xc, ac = ctx.cand(x), ctx.cand(a)
    if TT.N in (xc.quality, ac.quality):
        return False
    if xc.root == ac.root:
        iv = _ADDED_TONE.get(xc.quality)
        return iv is not None and supported(ctx, x, xc.root + iv)
    d = abs(xc.bass - ac.bass) % 12
    if min(d, 12 - d) <= 2:                                   # pedal neighbour / bass ±1–2
        return True
    r = (xc.root - ac.root) % 12
    core = TT.CORE.get(xc.quality, xc.quality)
    if r == 5 and core in _IV_CORES:                         # IV (iv) of A
        return True
    if r == 7 and core in _V_CORES:                          # V of A, any bass (V/7)
        return True
    return r == 11 and xc.quality in TT.DIM_FAMILY           # vii°/A


def _demote(segs: list[Segment], ctx: _Ctx) -> None:
    pcs_ = pieces(segs)
    for x in _overlays(segs):
        if x.role != "chord" or ctx.beats(x) > P.APA_MAX_BEATS + 1e-6:
            continue
        a, b = neighbours(x, segs, pcs_)
        if a is None or b is None or a.cls != b.cls or x.cls == a.cls:
            continue
        if not neighbour_whitelisted(ctx, x, a):
            x.min_detail = min(1.0, (x.min_detail if x.min_detail is not None else 0.0) + ctx.p.demote)


# =============================================================================================
# Step 4: merge
# =============================================================================================

def _merge(segs: list[Segment], ctx: _Ctx) -> list[Segment]:
    out: list[Segment] = []
    for layer in (_mains(segs), _overlays(segs)):
        cur: Segment | None = None
        for s in layer:
            if cur is not None and cur.c1 == s.c0 and cur.cand == s.cand and cur.role == s.role:
                wa, wb = ctx.beats(cur), ctx.beats(s)
                cur.conf = (cur.conf * wa + s.conf * wb) / max(wa + wb, 1e-9)
                if s.salience is not None:
                    cur.salience = max(cur.salience or s.salience, s.salience)
                if s.min_detail is not None:
                    cur.min_detail = min(cur.min_detail if cur.min_detail is not None else 1.0, s.min_detail)
                cur.alternatives = list(cur.alternatives) + list(s.alternatives)
                ctx.set_span(cur, cur.c0, s.c1)
                continue
            cur = s
            out.append(s)
    mains = _mains(out)
    keep = []
    for s in out:
        if s.layer == "passing" and s.role == "chord":
            h = next((m for m in mains if m.c0 <= s.c0 and m.c1 >= s.c1), None)
            if h is not None and h.cand == s.cand:
                continue                                   # identical to its host: merged
        keep.append(s)
    return keep


# =============================================================================================
# Step 5: 7th only in the bass
# =============================================================================================

def seventh_in_upper(seg: Segment, ev: Evidence, vocab: Vocab) -> bool:
    """False when the bass is the chord's 7th and the upper evidence for it is < 0.5 (§5.12
    step 5): show the triad over the bass (C/B, Am/G#) with the 7th chord as an alternative —
    pass the result to ``theory.simplify.label(..., seventh_in_upper=…)``. Bass events whose
    7th exists only to carry the moving bass are always False."""
    c = vocab.cands[seg.cand]
    if c.quality == TT.N:
        return True
    sev = seventh_interval(c.quality)
    if sev is None or (c.bass - c.root) % 12 != sev:
        return True
    if seg.role == "bass_event":
        return False
    vals = []
    for name in ("aux:sym:x", "aux:lv:tones"):
        a = ev.parts.get(name)
        if a is not None and seg.c1 > seg.c0:
            vals.append(float(np.asarray(a[seg.c0:seg.c1, c.bass]).mean()))
    return (max(vals) >= P.SUPPORT_MIN) if vals else True


# =============================================================================================
# Steps 6 and 11: display member and alternatives
# =============================================================================================

def _member_score(ctx: _Ctx, m: Candidate, l: Candidate | None, r: Candidate | None,
                  key: int, rootmass: np.ndarray | None) -> float:
    """Prior given the neighbours + log root-head mass. Members of a class share bass and pcs,
    so they differ only in how they are *named*: the prior is evaluated on m's root-position
    proxy (m.root, m.quality, m.root). Otherwise the prior's "stepwise bass with a slash chord"
    rule would prefer any inverted spelling (D#°7/C over C°7 before C#m)."""
    s = 0.0
    if key >= 0 and ctx.p.w_prior:
        mp = (m.root, m.quality, m.root)
        if l is not None:
            s += ctx.p.w_prior * PR.prior(l, mp, key, ctx.p.prior_cap)
        if r is not None:
            s += ctx.p.w_prior * PR.prior(mp, r, key, ctx.p.prior_cap)
    if rootmass is not None:
        s += math.log(float(rootmass[m.root]) + 1e-3)
    return s


_UPPER = {"9sus4": "maj", "13sus4": "maj7"}      # theory.simplify.UPPER_STRUCTURE


def upper_structure_member(vocab: Vocab, i: int) -> int:
    """For a root-position 9sus4 / 13sus4, the class member that names its upper structure (the
    major / maj7 chord on the ♭7 over the root: G9sus4 → F/G, F#13sus4 → Emaj7/F#), else i."""
    c = vocab.cands[i]
    q = _UPPER.get(c.quality)
    if q is None or c.root != c.bass:
        return i
    for m in vocab.classes[c.cls]:
        mc = vocab.cands[m]
        if mc.quality == q and mc.root == (c.root + 10) % 12 and mc.bass == c.bass:
            return m
    return i


def display_member(ctx: _Ctx, cand: int, l: Candidate | None, r: Candidate | None, key: int,
                   rootmass: np.ndarray | None) -> int:
    """The class member of ``cand`` with the best prior given the neighbours + root-head mass.

    M2: a root-position 9sus4 / 13sus4 winner is named by its upper structure when the class has
    it (§5.12 step 9, piano default): the label already reads "F/G", and the canonical Harte used
    for evaluation and exports must agree with it (the root-position proxy of ``_member_score``
    also hid F/G's IV/5 → I prior, so G9sus4 used to win the class on the sus rules)."""
    members = ctx.vocab.classes[ctx.vocab.cands[cand].cls]
    if len(members) <= 1:
        return cand
    best, bs = cand, _member_score(ctx, ctx.vocab.cands[cand], l, r, key, rootmass) + 1e-9
    for m in members:
        s = _member_score(ctx, ctx.vocab.cands[m], l, r, key, rootmass)
        if s > bs:
            best, bs = m, s
    return upper_structure_member(ctx.vocab, best)


def _display(segs: list[Segment], ctx: _Ctx) -> None:
    pcs_ = pieces(segs)
    for s in segs:
        a, b = neighbours(s, segs, pcs_)
        l = ctx.cand(a) if a is not None else None
        r = ctx.cand(b) if b is not None else None
        k = ctx.key_at(s.c0)
        rm = ctx.aux_mean("aux:lv:root", s)
        if s.role == "chord" and ctx.cand(s).quality != TT.N:
            ctx.set_cand(s, display_member(ctx, s.cand, l, r, k, rm))
        best: dict[int, tuple[int, float]] = {}
        for cand, conf in s.alternatives:
            cl = ctx.vocab.cands[cand].cls
            if cl == s.cls:
                continue
            if cl not in best or conf > best[cl][1]:
                best[cl] = (display_member(ctx, cand, l, r, k, rm), float(conf))
        s.alternatives = sorted(best.values(), key=lambda x: -x[1])[:P.N_ALTERNATIVES]


# =============================================================================================
# Entry point
# =============================================================================================

def finalize(segments: list[Segment], bass: BassTrack | None, key: KeyTrack | None, grid: RhythmGrid,
             vocab: Vocab, ev: Evidence, p: DecoderParams, *,
             onsets: np.ndarray | None = None) -> list[Segment]:
    """§5.12 steps 1–6 and 11 on the output of ``layers.build`` (returns new Segment objects).

    ``bass`` is accepted for the §4 contract; the bass events here are read from the segment
    labels (the bass track already shaped them through the ``bas`` source). ``onsets`` [G, 12]
    (keyword, not in the §4 contract yet) is the new-onset feature for the phantom rule."""
    ctx = _ctx(grid, vocab, ev, key, p)
    segs = [copy(s) for s in segments]
    for s in segs:
        s.alternatives = list(s.alternatives)
    segs = _bass_events(segs, ctx)
    segs = _phantoms(segs, ctx, onsets)
    _demote(segs, ctx)
    segs = _merge(segs, ctx)
    _display(segs, ctx)
    return sorted(segs, key=lambda s: (s.c0, s.layer != "main"))
