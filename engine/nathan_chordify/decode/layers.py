"""WP7 layers (DESIGN §5.11): main vs passing, promotion, salience, class-level conf, min_detail.

For each run R of the detail path, M is the main segment with the largest overlap (review A19):

- same chord (same root, same bass, same §5.11 core quality — or the same equivalence class):
  R is part of the main layer; when R covers ≥ 50 % of M and adds an extension (its template
  strictly contains M's), M takes R's richer label;
- otherwise, R ≤ 2 beats → a **passing** candidate;
- otherwise (a long disagreement) both are compared on the DETAIL lattice: R is promoted into
  the main layer (M is split) when mean post_cls_detail[R, cls(R)] > mean post_cls_detail[R,
  cls(M)] + 0.10; else M keeps the span and gets R's class as an alternative.

Passing candidates get
    salience = score(detail) − score(best merge into a neighbour) + trigram(L, P, R)
    conf     = mean_{g ∈ P} post_cls_detail[g, cls(P)]                (class level, §5.8)
    min_detail = clamp(σ(a·logit(conf) + b·salience + c), 0.05, 1)    ((a, b, c) = p.slider)
Nothing is deleted here; ``post.finalize`` may demote neighbour chords (§5.12 step 3).
"""
from __future__ import annotations

import math
from copy import copy

import numpy as np

from ..theory import priors as PR
from ..theory import templates as TT
from ..types import Candidate, DecodeResult, DecoderParams, Evidence, KeyTrack, RhythmGrid, Segment, Vocab
from . import params as P
from .evidence import cell_beats
from .hsmm import Lattice, beat_of_cells


def core(q: str) -> str:
    """§5.11 core quality: strips only the 9/11/13 extensions and alterations."""
    return TT.CORE.get(q, q)


def same_chord(a: Candidate, b: Candidate) -> bool:
    """Same root, same bass and the same core quality (sus, 6, 7, aug, dim are real changes)."""
    if a.quality == TT.N or b.quality == TT.N:
        return a.quality == b.quality
    return a.root == b.root and a.bass == b.bass and core(a.quality) == core(b.quality)


def adds_extension(r: Candidate, m: Candidate) -> bool:
    """R is M with extensions added: same chord, and R's template strictly contains M's."""
    return (same_chord(r, m) and r.quality != m.quality and r.quality != TT.N
            and TT.tones(m.quality) < TT.tones(r.quality))


def min_detail(conf: float, salience: float, p: DecoderParams) -> float:
    """§5.11 slider mapping: clamp(σ(a·logit(conf) + b·salience + c), 0.05, 1.00)."""
    a, b, c = p.slider
    q = min(max(conf, 1e-4), 1 - 1e-4)
    z = a * math.log(q / (1 - q)) + b * salience + c
    lo, hi = P.MIN_DETAIL_RANGE
    return float(min(max(1.0 / (1.0 + math.exp(-z)), lo), hi))


def class_alternatives(res: DecodeResult, C: np.ndarray, vocab: Vocab, c0: int, c1: int,
                       own_cls: int, n: int = P.N_ALTERNATIVES) -> list[tuple[int, float]]:
    """The top-n other classes by mean posterior over cells [c0, c1), each as (best member, conf)."""
    pc = res.post_cls[c0:c1].mean(axis=0).copy()
    pc[own_cls] = 0.0
    top = [int(k) for k in np.argsort(-pc)[:n] if pc[k] > 1e-3]
    if not top:
        return []
    pm = res.post[c0:c1].mean(axis=0)
    cls_of_col = np.array([vocab.cands[c].cls for c in C])
    out = []
    for k in top:
        cols = np.flatnonzero(cls_of_col == k)
        if len(cols):
            out.append((int(C[cols[np.argmax(pm[cols])]]), float(pc[k])))
    return out


def build(detail: DecodeResult, main: DecodeResult, ev: Evidence, grid: RhythmGrid,
          key: KeyTrack | None, vocab: Vocab, p: DecoderParams, *,
          rep: np.ndarray | None = None, hcdf: np.ndarray | None = None) -> list[Segment]:
    """Main + passing segments (§5.11). ``rep``/``hcdf`` should match the detail decode."""
    ucum = np.r_[0.0, np.cumsum(cell_beats(grid))]
    ct = np.asarray(grid.cell_t, dtype=float)
    beat_of = beat_of_cells(grid)

    def beats(c0: int, c1: int) -> float:
        return float(ucum[c1] - ucum[c0])

    def key_at(g: int) -> int:
        if key is None or not len(key.per_beat):
            return -1
        return int(key.per_beat[min(int(beat_of[min(g, len(beat_of) - 1)]), len(key.per_beat) - 1)])

    runs = sorted(detail.segments, key=lambda s: s.c0)
    mains = sorted((copy(s) for s in main.segments), key=lambda s: s.c0)
    main_cls = {id(m): m.cls for m in mains}            # the class the main lattice decided
    for m in mains:
        m.layer, m.salience, m.min_detail, m.alternatives = "main", None, None, []

    def overlap(a: Segment, b: Segment) -> float:
        lo, hi = max(a.c0, b.c0), min(a.c1, b.c1)
        return beats(lo, hi) if hi > lo else 0.0

    passing_idx: list[int] = []
    promoted: list[Segment] = []
    extra_alts: dict[int, list[tuple[int, float]]] = {}
    richer: dict[int, tuple[float, Segment]] = {}
    for idx, R in enumerate(runs):
        cands = [m for m in mains if m.c0 < R.c1 and m.c1 > R.c0]
        if not cands:
            continue
        M = max(cands, key=lambda m: overlap(R, m))
        rc, mc = vocab.cands[R.cand], vocab.cands[M.cand]
        if R.cls == M.cls or same_chord(rc, mc):
            ov = overlap(R, M)
            if adds_extension(rc, mc) and ov >= P.RICHER_COVER * beats(M.c0, M.c1):
                if id(M) not in richer or ov > richer[id(M)][0]:
                    richer[id(M)] = (ov, R)
            continue
        if beats(R.c0, R.c1) <= P.PASSING_MAX_BEATS:
            if rc.quality != TT.N:
                passing_idx.append(idx)
            continue
        pr = float(detail.post_cls[R.c0:R.c1, R.cls].mean())
        pm = float(detail.post_cls[R.c0:R.c1, M.cls].mean())
        if pr > pm + P.PROMOTE_MARGIN:
            promoted.append(R)
        else:
            extra_alts.setdefault(id(M), []).append((R.cand, pr))

    # richer labels, alternatives and conf of the main segments (main lattice)
    out_main: list[Segment] = []
    for m in mains:
        ocls = main_cls[id(m)]
        m.conf = float(main.post_cls[m.c0:m.c1, ocls].mean())
        alts = class_alternatives(main, ev.C, vocab, m.c0, m.c1, ocls)
        for a in extra_alts.get(id(m), []):
            if all(vocab.cands[x].cls != vocab.cands[a[0]].cls for x, _ in alts):
                alts.append(a)
        m.alternatives = alts
        if id(m) in richer:
            R = richer[id(m)][1]
            m.cand, m.cls = R.cand, R.cls
        out_main.append(m)

    # promotions: carve R out of the main layer
    for R in promoted:
        new: list[Segment] = []
        for m in out_main:
            if m.c1 <= R.c0 or m.c0 >= R.c1:
                new.append(m)
                continue
            if m.c0 < R.c0:
                a = copy(m)
                a.c1, a.t1 = R.c0, float(ct[R.c0])
                a.alternatives = list(m.alternatives)
                new.append(a)
            if m.c1 > R.c1:
                b = copy(m)
                b.c0, b.t0 = R.c1, float(ct[R.c1])
                b.alternatives = list(m.alternatives)
                new.append(b)
        s = copy(R)
        s.layer, s.salience, s.min_detail = "main", None, None
        s.conf = float(detail.post_cls[R.c0:R.c1, R.cls].mean())
        s.alternatives = class_alternatives(detail, ev.C, vocab, R.c0, R.c1, R.cls)
        new.append(s)
        out_main = sorted(new, key=lambda x: x.c0)

    # passing candidates: salience, class-level conf, min_detail
    lat = Lattice(ev, grid, key, vocab, p, "detail", rep=rep, hcdf=hcdf) if passing_idx else None
    out_pass: list[Segment] = []
    for idx in passing_idx:
        Pseg = runs[idx]
        L = runs[idx - 1] if idx > 0 else None
        Rn = runs[idx + 1] if idx + 1 < len(runs) else None
        sal = _salience(lat, L, Pseg, Rn)
        k = key_at(Pseg.c0)
        if L is not None and Rn is not None and k >= 0:
            sal += PR.trigram(vocab.cands[L.cand], vocab.cands[Pseg.cand], vocab.cands[Rn.cand], k)
        conf = float(detail.post_cls[Pseg.c0:Pseg.c1, Pseg.cls].mean())
        s = Segment(c0=Pseg.c0, c1=Pseg.c1, t0=float(ct[Pseg.c0]), t1=float(ct[Pseg.c1]),
                    cand=Pseg.cand, cls=Pseg.cls, layer="passing", conf=conf, salience=float(sal),
                    min_detail=min_detail(conf, sal, p),
                    alternatives=class_alternatives(detail, ev.C, vocab, Pseg.c0, Pseg.c1, Pseg.cls))
        out_pass.append(s)

    return sorted(out_main + out_pass, key=lambda s: (s.c0, s.layer != "main"))


def _salience(lat: Lattice | None, L: Segment | None, Pseg: Segment, Rn: Segment | None) -> float:
    """score(detail) − score(best merge of P into a neighbour), rescoring only these 3 segments."""
    if lat is None or (L is None and Rn is None):
        return 0.0
    col = lat.col
    kP = col[Pseg.cand]
    cur: list[tuple[int, int, int]] = []
    if L is not None:
        cur.append((L.c0, L.c1, col[L.cand]))
    cur.append((Pseg.c0, Pseg.c1, kP))
    if Rn is not None:
        cur.append((Rn.c0, Rn.c1, col[Rn.cand]))
    options: list[list[tuple[int, int, int]]] = []
    if L is not None and Rn is not None and L.cand == Rn.cand:
        options.append([(L.c0, Rn.c1, col[L.cand])])
    else:
        if L is not None:
            options.append([(L.c0, Pseg.c1, col[L.cand])] + ([(Rn.c0, Rn.c1, col[Rn.cand])] if Rn else []))
        if Rn is not None:
            options.append(([(L.c0, L.c1, col[L.cand])] if L else []) + [(Pseg.c0, Rn.c1, col[Rn.cand])])
    return lat.path_score(cur) - max(lat.path_score(o) for o in options)
