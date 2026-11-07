"""WP6 simplify levels, slash visibility, upper structures (shared/test_vectors/simplify.json)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nathan_chordify.theory import harte as H
from nathan_chordify.theory import simplify as SM
from nathan_chordify.theory import spelling as S
from nathan_chordify.theory import templates as T
from nathan_chordify.theory import vocab as V

VEC = Path(__file__).resolve().parents[2] / "shared" / "test_vectors"
DATA = json.loads((VEC / "simplify.json").read_text(encoding="utf-8"))


def key_index(name: str) -> int:
    minor = name.endswith("m")
    pc = S.parse_note(name[:-1] if minor else name).pc
    return pc + (12 if minor else 0)


def vid(v):
    opts = "".join(f" {k}={v[k]}" for k in ("slash", "upper_structure", "seventh_in_upper", "notation") if k in v)
    return f"{v['chord']} {v['level']}{opts}"


@pytest.mark.parametrize("v", DATA["vectors"], ids=vid)
def test_simplify_vectors(v):
    r, q, b = H.parse_harte(v["chord"])
    kw = {k: v[k] for k in ("slash", "upper_structure", "seventh_in_upper") if k in v}
    if "notation" in v:
        kw["notation"] = SM.Notation(**v["notation"])
    assert SM.label(r, q, b, key_index(v["key"]), level=v["level"], **kw) == v["label"]


def test_level_table_is_complete():
    for lv in SM.LEVELS:
        assert set(SM.LEVEL_MAP[lv]) == set(T.QUALITIES)
        assert set(SM.LEVEL_MAP[lv].values()) <= set(T.QUALITIES)
    simple_out = set(SM.LEVEL_MAP["simple"].values())
    assert simple_out == {"maj", "min", "dim"}


def test_labels_triplet_matches_json_example():
    assert SM.labels(11, "+7", 11, 11) == {"simple": "B", "standard": "B+", "full": "B+7"}


def test_bass_visibility_rules():
    vocab = V.candidates()
    for c in vocab.cands[1:]:
        iv = (c.bass - c.root) % 12
        simple = SM.display_chord(c.root, c.quality, c.bass, "simple")
        std = SM.display_chord(c.root, c.quality, c.bass, "standard")
        full = SM.display_chord(c.root, c.quality, c.bass, "full")
        assert simple.bass is None
        if iv == 0:
            assert full.bass is None or full.kind == "upper"
            continue
        assert full.bass == c.bass
        chord_tone = iv in T.chord_bass_intervals(c.quality)
        assert (std.bass is not None) == chord_tone
        assert SM.display_chord(c.root, c.quality, c.bass, "simple", slash=True).bass == c.bass
        assert SM.display_chord(c.root, c.quality, c.bass, "full", slash=False).bass is None


def test_upper_structure_only_in_root_position():
    dc = SM.display_chord(7, "9sus4", 7, "full")
    assert (dc.root, dc.quality, dc.bass, dc.kind) == (5, "maj", 7, "upper")
    dc = SM.display_chord(6, "13sus4", 6, "standard")
    assert (dc.root, dc.quality, dc.bass, dc.kind) == (4, "maj7", 6, "upper")
    dc = SM.display_chord(7, "9sus4", 0, "full")      # G9sus4/C keeps its name
    assert dc.kind == "inversion" and dc.quality == "9sus4"
    assert SM.display_chord(7, "9sus4", 7, "simple").quality == "maj"
    assert SM.display_chord(7, "9sus4", 7, "simple").root == 7


def test_seventh_only_in_bass_map_is_triadic():
    for q, tq in SM.SEVENTH_ONLY_IN_BASS.items():
        assert SM.seventh_interval(q) is not None, q
        assert SM.seventh_interval(tq) is None, (q, tq)
    assert SM.seventh_interval("dim7") is None           # bb7 is not treated as a 7th


def test_bad_level_and_notation():
    with pytest.raises(ValueError):
        SM.display_chord(0, "maj", 0, "medium")
    with pytest.raises(ValueError):
        SM.Notation(maj7="ma7")
    assert SM.label(-1, "N", -1, 0) == "N"


def test_every_label_parses_back_to_the_displayed_chord():
    vocab = V.candidates()
    for key in (0, 11, 13):
        for c in vocab.cands[1::5]:
            for lv in SM.LEVELS:
                lab = SM.label(c.root, c.quality, c.bass, key, level=lv)
                dc = SM.display_chord(c.root, c.quality, c.bass, lv)
                parsed = H.parse_symbol(lab)
                bass = dc.bass if dc.bass is not None else dc.root
                assert parsed == (dc.root, dc.quality, bass), (c, lv, lab)
