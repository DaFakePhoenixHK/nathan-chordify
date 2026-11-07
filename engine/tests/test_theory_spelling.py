"""WP6 key-aware spelling (shared/test_vectors/spelling.json) and its global invariants."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nathan_chordify.theory import harte as H
from nathan_chordify.theory import simplify as SM
from nathan_chordify.theory import spelling as S
from nathan_chordify.theory import vocab as V

VEC = Path(__file__).resolve().parents[2] / "shared" / "test_vectors"
DATA = json.loads((VEC / "spelling.json").read_text(encoding="utf-8"))


def key_index(name: str) -> int:
    minor = name.endswith("m")
    pc = S.parse_note(name[:-1] if minor else name).pc
    return pc + (12 if minor else 0)


def vid(v: dict) -> str:
    extra = "".join(f" {k}={v[k]}" for k in ("t", "pref", "theoretical", "next") if k in v)
    return f"{v['chord']} in {v['key']}{extra}"


@pytest.mark.parametrize("v", DATA["key_names"], ids=lambda v: f"{v[0]}+{v[1]} {v[2]}{' theo' if v[3] else ''}")
def test_key_names(v):
    key, t, pref, theo, expected = v
    assert S.key_name(key, t, pref, theo) == expected


@pytest.mark.parametrize("v", DATA["vectors"], ids=vid)
def test_spelling_vectors(v):
    r, q, b = H.parse_harte(v["chord"])
    nxt = H.parse_harte(v["next"])[0] if "next" in v else None
    key = key_index(v["key"])
    kw = dict(transpose=v.get("t", 0), pref=v.get("pref", "auto"),
              theoretical=v.get("theoretical", False))
    assert SM.label(r, q, b, key, next_root=nxt, **kw) == v["label"]
    if "expect_key" in v:
        assert S.key_name(key, kw["transpose"], kw["pref"], kw["theoretical"]) == v["expect_key"]


def test_vector_count():
    assert len(DATA["vectors"]) + len(DATA["key_names"]) >= 300


ODD = re.compile(r"(^|/)(B#|E#|Cb|Fb)")
DOUBLE = re.compile(r"(##|bb)")


@pytest.mark.parametrize("key", range(24))
def test_no_double_accidentals_or_odd_roots(key):
    """A18: never a double sharp/flat; no B#/E#/Cb/Fb roots or basses by default; names sound right."""
    vocab = V.candidates()
    for t in (0, 1, 5, 7):
        for c in vocab.cands[1::7]:
            sp = S.spell(c.root, c.quality, c.bass, key, transpose=t)
            assert S.parse_note(sp.root).pc == (c.root + t) % 12
            assert S.parse_note(sp.bass).pc == (c.bass + t) % 12
            for name in (sp.root, sp.bass, *sp.tones):
                assert not DOUBLE.search(name[1:]), (sp, key, t)
            assert sp.root not in S.ODD_NAMES and sp.bass not in S.ODD_NAMES, (sp, key, t)
            lab = SM.label(c.root, c.quality, c.bass, key, transpose=t)
            assert not ODD.search(lab), lab
            assert not DOUBLE.search(lab), lab


def test_spelled_tones_follow_intervals():
    sp = S.spell(11, "+7", 11, key=11)               # B+7 in B: F## shown as G
    assert sp.tones == ("B", "D#", "G", "A")
    sp = S.spell(3, "7", 7, key=11)                  # D#7/F## → D#7/G
    assert (sp.root, sp.bass) == ("D#", "G")
    assert str(sp.func_bass) == "F##"
    sp = S.spell(0, "dim7", 0, key=11)               # #1°7 in B
    assert sp.root == "C" and str(sp.func_root) == "B#"
    assert sp.tones == ("C", "Eb", "Gb", "A")        # spelled from the displayed root; Bbb → A
    sp = S.spell(0, "dim7", 0, key=11, theoretical=True)
    assert sp.root == "B#" and sp.tones == ("B#", "D#", "F#", "A")
    sp = S.spell(4, "maj", 8, key=4)                 # E/G#, not E/Ab
    assert sp.bass == "G#"
    sp = S.spell(4, "maj", 8, key=3)                 # even in Eb major
    assert sp.bass == "G#"


def test_key_signature_and_tonic():
    assert S.key_signature(S.parse_note("B"), False) == 5
    assert S.key_signature(S.parse_note("Gb"), False) == -6
    assert S.key_signature(S.parse_note("G#"), True) == 5
    assert S.key_signature(S.parse_note("G#"), False) == 8           # theoretical key
    assert str(S.tonic_note(6, False)) == "F#"                       # tie → sharps
    assert str(S.tonic_note(3, True)) == "D#"                        # D#m (6#) vs Ebm (6b) → sharps
    with pytest.raises(ValueError):
        S.key_name(24)
    with pytest.raises(ValueError):
        S.tonic_note(0, False, pref="bogus")


def test_note_helpers():
    assert str(S.simple(S.parse_note("F##"))) == "G"
    assert str(S.simple(S.parse_note("Bbb"))) == "A"
    assert str(S.simple(S.parse_note("E##"))) == "F#"
    assert str(S.simple(S.parse_note("Cbb"))) == "Bb"
    assert str(S.display(S.parse_note("B#"))) == "C"
    assert str(S.display(S.parse_note("Fb"))) == "E"
    assert str(S.display(S.parse_note("Fb"), theoretical=True)) == "Fb"
    assert str(S.parse_note("D#").up("b3")) == "F#"
    assert str(S.parse_note("D#").up("3")) == "F##"
    assert str(S.parse_note("Ab").up("#5")) == "E"
    assert S.pc_of("Cx") == 2 and S.pc_of("E♭") == 3
    with pytest.raises(ValueError):
        S.parse_note("H")


def test_spell_candidate_and_n():
    vocab = V.candidates()
    c = vocab.cands[V.index_of(6, "maj", 10)]
    assert S.spell_candidate(c, key=11).bass == "A#"
    n = S.spell(-1, "N", -1, key=11)
    assert n.root == "N" and n.key == "B"
