"""WP6 vocabulary, template matrix and equivalence classes (shared/test_vectors/classes.json)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from nathan_chordify.theory import harte as H
from nathan_chordify.theory import templates as T
from nathan_chordify.theory import vocab as V
from nathan_chordify.types import Candidate, Vocab

VEC = Path(__file__).resolve().parents[2] / "shared" / "test_vectors"
CLASSES = json.loads((VEC / "classes.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def vocab() -> Vocab:
    return V.candidates()


def cls_of(label: str) -> int:
    r, q, b = H.parse_harte(label)
    i = V.index_of(r, q, b)
    assert i is not None, f"{label} is not a candidate"
    return V.candidates().cands[i].cls


def test_size_and_n(vocab):
    # 12 roots × Σ_q |allowed basses(q)| + N
    per_root = sum(len(T.allowed_bass_intervals(q)) for q in T.QUALITIES)
    assert per_root == 139
    assert len(vocab.cands) == 12 * per_root + 1 == 1669
    n = vocab.cands[V.N_INDEX]
    assert (n.root, n.quality, n.bass) == (-1, "N", -1)
    assert vocab.classes[n.cls] == [V.N_INDEX]
    assert isinstance(n, Candidate)


def test_candidates_unique_and_allowed(vocab):
    triples = [(c.root, c.quality, c.bass) for c in vocab.cands]
    assert len(set(triples)) == len(triples)
    for c in vocab.cands[1:]:
        iv = (c.bass - c.root) % 12
        assert iv in T.allowed_bass_intervals(c.quality)
        assert c.pcs == frozenset((c.root + i) % 12 for i in T.tones(c.quality)) | {c.bass}
        assert V.index_of(c.root, c.quality, c.bass) == triples.index((c.root, c.quality, c.bass))


def test_allowed_basses():
    assert T.allowed_bass_intervals("maj") == (0, 4, 7, 2, 5, 10)
    assert T.allowed_bass_intervals("maj7") == (0, 4, 7, 11, 2, 5, 10)
    assert T.allowed_bass_intervals("m7") == (0, 3, 7, 10, 2, 5)
    assert T.allowed_bass_intervals("6") == (0, 4, 7, 9, 2, 5, 10)
    assert T.allowed_bass_intervals("add9") == (0, 4, 7, 2, 5, 10)   # +2 is the 9th: allowed as non-chord bass
    assert T.allowed_bass_intervals("9") == (0, 4, 7, 10)            # never the 9th
    assert T.allowed_bass_intervals("13") == (0, 4, 7, 10)           # never the 13th
    assert T.allowed_bass_intervals("m11") == (0, 3, 7, 10)          # never the 11th
    assert T.allowed_bass_intervals("dim7") == (0, 3, 6, 9)
    assert T.allowed_bass_intervals("aug") == (0, 4, 8)
    assert T.allowed_bass_intervals("sus4") == (0, 5, 7, 2, 10)
    assert T.allowed_bass_intervals("5") == (0, 7)
    # the §5.8 examples are candidates
    for label in ("F:maj/2", "E:min/4", "F#:maj/4", "G:maj/b7", "C#:min7/4", "E:maj7/2", "B:maj6/2"):
        assert V.index_of(*H.parse_harte(label)) is not None, label
    assert V.index_of(0, "maj", 11) is None          # C/B only as Cmaj7/B
    assert V.index_of(0, "maj7", 11) is not None


def test_template_roles():
    assert T.required("maj") == {0, 4} and T.optional("maj") == {7}
    for q in ("dim", "aug", "m7b5", "dim7", "+7", "+maj7", "5"):
        assert not T.optional(q), q              # their 5th is required
    assert T.optional("m11") == {2, 7}
    assert T.optional("13sus4") == {2, 7}
    assert T.optional("13") == {2, 7}
    assert T.tones("13") == {0, 2, 4, 7, 9, 10}
    assert T.tones("7b13") == {0, 4, 7, 8, 10}
    assert T.tones("maj7#11") == {0, 4, 6, 7, 11}
    assert T.role("7", 10) == "required" and T.role("7", 7) == "optional" and T.role("7", 1) == "absent"
    assert T.absent("maj") == frozenset(range(12)) - {0, 4, 7}


def test_tau_rows(vocab):
    tau = vocab.tau
    assert tau.shape == (len(vocab.cands), 12)
    assert np.all(tau[V.N_INDEX] == 0.1)
    row = V.tau_row(0, "7", 0)                       # C7
    assert row[0] == row[4] == row[10] == 0.85
    assert row[7] == 0.5
    assert row[1] == row[11] == 0.10
    em_a = V.tau_row(4, "min", 9)                    # Em/A: non-chord bass A gets 0.5
    assert em_a[9] == 0.5 and em_a[4] == em_a[7] == 0.85 and em_a[11] == 0.5
    b_d = V.tau_row(11, "maj", 3)                    # B/D#: D# is an upper tone, stays required
    assert b_d[3] == 0.85
    f_g = V.tau_row(5, "maj", 7)                     # F/G
    assert f_g[7] == 0.5 and f_g[5] == 0.85
    assert V.tau_row(5, "maj", 7, mode="ace")[7] == 0.85
    i = V.index_of(4, "min", 9)
    assert np.array_equal(tau[i], em_a)
    assert set(np.unique(tau)) <= {0.1, 0.5, 0.85}
    with pytest.raises(ValueError):
        tau[0, 0] = 1.0                             # the cached matrix is read-only


def test_classes_partition(vocab):
    seen = sorted(i for members in vocab.classes for i in members)
    assert seen == list(range(len(vocab.cands)))
    for cid, members in enumerate(vocab.classes):
        assert members and all(vocab.cands[i].cls == cid for i in members)
        assert len({vocab.cands[i].bass for i in members}) == 1      # a class shares its bass
    # same (pc set, bass) → same class
    by_key: dict = {}
    for c in vocab.cands[1:]:
        by_key.setdefault((c.pcs, c.bass), set()).add(c.cls)
    assert all(len(s) == 1 for s in by_key.values())
    sizes = Counter(len(m) for m in vocab.classes)
    assert max(sizes) <= 5


@pytest.mark.parametrize("group", CLASSES["same"], ids=lambda g: " = ".join(g))
def test_same_class(group):
    ids = {cls_of(label) for label in group}
    assert len(ids) == 1, group


@pytest.mark.parametrize("pair", CLASSES["different"], ids=lambda p: " != ".join(p))
def test_different_class(pair):
    assert cls_of(pair[0]) != cls_of(pair[1])


def test_dim7_and_aug_rotations():
    for bass in range(12):
        dim = {cls_of(H.to_harte((bass + k) % 12, "dim7", bass)) for k in (0, 3, 6, 9)}
        assert len(dim) == 1
        aug = {cls_of(H.to_harte((bass + k) % 12, "aug", bass)) for k in (0, 4, 8)}
        assert len(aug) == 1
        assert cls_of(H.to_harte(bass, "dim7", bass)) != cls_of(H.to_harte((bass + 1) % 12, "dim7", (bass + 1) % 12))


def test_equivalents_helper():
    eq = V.equivalents(4, "6", 4)                    # E6
    assert (1, "m7", 4) in eq                        # C#m7/E
    assert (4, "6", 4) not in eq
    assert V.same_class((6, "9sus4", 6), (4, "maj", 6))
    assert not V.same_class((6, "7sus4", 6), (4, "maj", 6))
    assert V.equivalents(0, "maj", 11) == []         # not a candidate


def test_passing_type_set():
    pt = {"dim", "dim7", "m7b5", "aug", "+7", "+maj7", "sus2", "sus4", "7sus4", "9sus4", "13sus4"}
    assert V.PT_QUALITIES == pt
    vocab = V.candidates()
    for c in vocab.cands:
        expected = c.quality != "N" and (c.quality in pt or c.bass != c.root)
        assert V.is_passing_type(c) == expected
    assert V.is_passing_type(vocab.cands[V.index_of(6, "maj", 10)])      # F#/A#
    assert not V.is_passing_type(vocab.cands[V.index_of(11, "maj", 11)])  # B
