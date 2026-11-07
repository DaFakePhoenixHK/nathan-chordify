"""WP6 Nashville / 級數 and Roman numerals (shared/test_vectors/numbers.json)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nathan_chordify.theory import harte as H
from nathan_chordify.theory import numbers as NB
from nathan_chordify.theory import simplify as SM
from nathan_chordify.theory import spelling as S
from nathan_chordify.theory import vocab as V

VEC = Path(__file__).resolve().parents[2] / "shared" / "test_vectors"
DATA = json.loads((VEC / "numbers.json").read_text(encoding="utf-8"))


def key_index(name: str) -> int:
    minor = name.endswith("m")
    pc = S.parse_note(name[:-1] if minor else name).pc
    return pc + (12 if minor else 0)


@pytest.mark.parametrize("v", DATA["vectors"],
                         ids=lambda v: f"{v['chord']} in {v['key']}" + (f" next={v['next']}" if "next" in v else ""))
def test_number_vectors(v):
    r, q, b = H.parse_harte(v["chord"])
    nxt = H.parse_harte(v["next"])[0] if "next" in v else None
    key = key_index(v["key"])
    assert NB.nashville(r, q, b, key, next_root=nxt) == v["nashville"]
    assert NB.roman(r, q, b, key, next_root=nxt) == v["roman"]


def test_design_examples():
    B = 11
    assert NB.nashville(11, "maj", 11, B) == "1"
    assert NB.nashville(6, "maj", 10, B) == "5/7"
    assert NB.nashville(8, "min", 8, B) == "6m"
    assert NB.nashville(4, "maj7", 4, B) == "4maj7"
    assert NB.nashville(9, "7", 9, B) == "b7(7)"
    assert NB.nashville(3, "7", 10, B) == "3(7)/7"
    assert NB.nashville(0, "dim7", 0, B) == "#1°7"
    assert NB.nashville(11, "+7", 11, B) == "1+(7)"
    assert NB.roman(11, "+7", 11, B) == "I+7"
    deg = NB.degrees(11, "+7", 11, B)
    assert deg == {"root": 0, "bass": 0, "nashville": "1+(7)", "roman": "I+7"}
    assert NB.degrees(-1, "N", -1, B)["nashville"] == "N"


def test_numbers_follow_levels_and_notation():
    C = 0
    assert NB.nashville(7, "9", 7, C, level="standard") == "5(7)"
    assert NB.nashville(7, "9", 7, C, level="simple") == "5"
    assert NB.nashville(7, "9sus4", 7, C) == "4/5"                    # upper structure F/G
    assert NB.nashville(7, "9sus4", 7, C, upper_structure=False) == "5(9sus4)"
    assert NB.nashville(4, "min", 9, 7, level="standard") == "6m"     # Em/A in G: no slash at Standard
    assert NB.nashville(0, "maj7", 11, C, seventh_in_upper=False) == "1/7"
    assert NB.roman(0, "maj7", 11, C, seventh_in_upper=False) == "I/7̂"
    assert NB.nashville(0, "maj7", 0, C, notation=SM.Notation(maj7="Δ7")) == "1Δ7"
    assert NB.roman(0, "maj7", 4, C, notation=SM.Notation(maj7="Δ7")) == "IΔ6/5"
    assert NB.roman(11, "m7b5", 11, C, notation=SM.Notation(symbols=False)) == "viim7b5"


def test_minor_key_numbering():
    Am = 21
    assert NB.nashville(9, "min", 9, Am) == "6m"            # relative-major 級數
    assert NB.roman(9, "min", 9, Am) == "i"
    assert NB.nashville(4, "7", 4, Am) == "3(7)"
    assert NB.roman(4, "7", 4, Am) == "V7"
    assert NB.roman(5, "maj", 5, Am) == "bVI"
    assert NB.roman(10, "maj", 10, Am) == "bII"             # Neapolitan


def test_numbers_are_transposition_invariant():
    vocab = V.candidates()
    for key in (0, 11, 18, 21):
        for c in vocab.cands[1::11]:
            for t in (1, 5, 8):
                k2 = (key % 12 + t) % 12 + (12 if key >= 12 else 0)
                r2, b2 = (c.root + t) % 12, (c.bass + t) % 12
                assert NB.nashville(c.root, c.quality, c.bass, key) == NB.nashville(r2, c.quality, b2, k2)
                assert NB.roman(c.root, c.quality, c.bass, key) == NB.roman(r2, c.quality, b2, k2)


def test_degree_str():
    n = S.parse_note
    assert NB.degree_str(n("B#"), n("B")) == "#1"
    assert NB.degree_str(n("A"), n("B")) == "b7"
    assert NB.degree_str(n("F##"), n("B")) == "#5"
    assert NB.degree_str(n("Cb"), n("Db")) == "b7"
