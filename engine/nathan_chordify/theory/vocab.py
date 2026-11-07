"""Chord candidates, template matrix and equivalence classes (DESIGN §5.8, §5.9; review A5).

``candidates()`` builds every (root, quality, bass) candidate:

- 12 roots × 32 qualities × the allowed basses: every non-extension chord tone (root first), plus
  the non-chord basses +2, +5, +10 for maj, min, sus2, sus4, add9, maj7, m7 and 6 (§5.8 v1.1);
- N = (-1, "N", -1) is candidate 0.

``Vocab.tau`` is the [V,12] template matrix used by the symbolic source (§5.9 *sym*):
τ = 0.85 for required tones, 0.5 for optional tones, 0.10 for absent tones; the bass pc gets
0.5 unless it is also an upper (template) tone, because the right hand need not play a non-chord
bass. So Em/A has τ(A) = 0.5, while B/D# keeps τ(D#) = 0.85 (D# is the 3rd of B). N rows are 0.1
everywhere. ``tau_matrix(cands, mode="ace")`` gives the variant for a notes head that also hears
the bass (the bass pc counts as required).

**Equivalence classes** (§5.8 / A5): candidates with the same bass pc and the same pc set
(template tones ∪ bass) form a class; two classes are then merged when they share the bass and
one's set equals the other's minus a single optional perfect 5th. Design choice: the removed 5th
must be the optional P5 of a *root-position* member of the larger class (root = bass). That
gives every example in §5.8 (E6 = C#m7/E, Csus2 = Gsus4/C, F#9sus4 = C#m7/F# merged with E/F#,
Emaj7/F# = F#13sus4, B+7 = B7b13 no 5, the dim7 and aug rotations) while keeping transitive
merges from chaining distinct chords together (without it F#7sus4 would join E/F# through the
5th of C#m7/F#). Merging uses union-find, so classes partition the candidates.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..types import Candidate, Vocab
from . import templates as T

N_INDEX = 0
TAU_REQUIRED, TAU_OPTIONAL, TAU_ABSENT, TAU_BASS = 0.85, 0.5, 0.10, 0.5
TAU_N = 0.1

PT_QUALITIES = T.PT_QUALITIES

Triple = tuple[int, str, int]


def is_passing_type(c) -> bool:
    """Passing-type set PT (§5.8): dim, dim7, m7b5, aug, +7, +maj7, sus2, sus4, 7sus4, 9sus4,
    13sus4, and any chord whose bass ≠ root. N is not passing-type."""
    if c.quality == T.N:
        return False
    return c.quality in PT_QUALITIES or c.bass != c.root


def cand_pcs(root: int, quality: str, bass: int) -> frozenset[int]:
    if quality == T.N:
        return frozenset()
    return frozenset((root + i) % 12 for i in T.tones(quality)) | {bass % 12}


def _triples() -> list[Triple]:
    out: list[Triple] = [(-1, T.N, -1)]
    for root in range(12):
        for q in T.QUALITIES:
            for iv in T.allowed_bass_intervals(q):
                out.append((root, q, (root + iv) % 12))
    return out


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _classes(triples: list[Triple]) -> list[int]:
    """Class id per candidate (ids numbered by first member; N is class 0)."""
    keys = [(cand_pcs(*t), t[2]) for t in triples]
    base: dict[tuple[frozenset[int], int], int] = {}
    for i, k in enumerate(keys):
        base.setdefault(k, i)
    parent = list(range(len(triples)))
    for i, k in enumerate(keys):                       # same (pc set, bass) → same class
        a, b = _find(parent, i), _find(parent, base[k])
        if a != b:
            parent[max(a, b)] = min(a, b)
    for i, (root, q, bass) in enumerate(triples):      # merge across an optional P5
        if q == T.N or root != bass or 7 not in T.optional(q):
            continue
        pcs, _ = keys[i]
        smaller = (pcs - {(root + 7) % 12}, bass)
        j = base.get(smaller)
        if j is None:
            continue
        a, b = _find(parent, i), _find(parent, j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = [_find(parent, i) for i in range(len(triples))]
    ids: dict[int, int] = {}
    return [ids.setdefault(r, len(ids)) for r in roots]


def tau_row(root: int, quality: str, bass: int, mode: str = "sym") -> np.ndarray:
    """Template probabilities over the 12 pcs for one candidate (see module doc)."""
    row = np.full(12, TAU_ABSENT if quality != T.N else TAU_N, dtype=np.float64)
    if quality == T.N:
        return row
    for t in T.TEMPLATE[quality]:
        row[(root + t.interval) % 12] = TAU_OPTIONAL if t.optional else TAU_REQUIRED
    b = bass % 12
    upper = (b - root) % 12 in T.tones(quality)
    if mode == "sym":
        if not upper:
            row[b] = TAU_BASS
    elif mode == "ace":
        row[b] = max(row[b], TAU_REQUIRED)
    else:
        raise ValueError("mode must be 'sym' or 'ace'")
    return row


def tau_matrix(cands: list[Candidate], mode: str = "sym") -> np.ndarray:
    return np.stack([tau_row(c.root, c.quality, c.bass, mode) for c in cands])


@lru_cache(maxsize=1)
def candidates() -> Vocab:
    """The full candidate vocabulary with τ and equivalence classes (cached; do not mutate)."""
    triples = _triples()
    cls = _classes(triples)
    cands = [Candidate(r, q, b, cand_pcs(r, q, b), cls[i]) for i, (r, q, b) in enumerate(triples)]
    classes: list[list[int]] = [[] for _ in range(max(cls) + 1)]
    for i, c in enumerate(cls):
        classes[c].append(i)
    tau = tau_matrix(cands, "sym")
    tau.setflags(write=False)
    return Vocab(cands, tau, classes)


@lru_cache(maxsize=1)
def _index() -> dict[Triple, int]:
    return {(c.root, c.quality, c.bass): i for i, c in enumerate(candidates().cands)}


def index_of(root: int, quality: str, bass: int) -> int | None:
    """Index of a candidate in ``candidates().cands`` (None when it is not in the vocabulary)."""
    if quality == T.N:
        return N_INDEX
    return _index().get((root % 12, quality, bass % 12))


def class_members(i: int) -> list[Candidate]:
    """The candidates in the same equivalence class as candidate ``i``."""
    v = candidates()
    return [v.cands[j] for j in v.classes[v.cands[i].cls]]


def equivalents(root: int, quality: str, bass: int) -> list[Triple]:
    """Other members of a chord's class, as triples (for the JSON ``equivalents`` field)."""
    i = index_of(root, quality, bass)
    if i is None:
        return []
    return [(c.root, c.quality, c.bass) for c in class_members(i)
            if (c.root, c.quality, c.bass) != (root % 12, quality, bass % 12)]


def same_class(a: Triple, b: Triple) -> bool:
    ia, ib = index_of(*a), index_of(*b)
    if ia is None or ib is None:
        return False
    v = candidates()
    return v.cands[ia].cls == v.cands[ib].cls
