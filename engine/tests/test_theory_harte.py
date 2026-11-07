"""WP6 Harte labels and typed chord symbols (shared/test_vectors/harte.json)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nathan_chordify.theory import harte as H
from nathan_chordify.theory import templates as T
from nathan_chordify.theory import vocab as V

VEC = Path(__file__).resolve().parents[2] / "shared" / "test_vectors"


def load(name: str):
    return json.loads((VEC / f"{name}.json").read_text(encoding="utf-8"))


DATA = load("harte")


def test_every_quality_has_a_harte_label():
    assert {v["quality"] for v in DATA["qualities"]} == set(T.QUALITIES)
    assert len(T.QUALITIES) == 32


@pytest.mark.parametrize("v", DATA["qualities"], ids=lambda v: v["quality"])
def test_quality_harte_round_trip(v):
    assert H.to_harte(0, v["quality"], 0) == v["harte"]
    assert H.parse_harte(v["harte"]) == (0, v["quality"], 0)
    # the Harte intervals equal the template tones
    p = H.parse_harte_full(v["harte"])
    assert p.quality == v["quality"]


@pytest.mark.parametrize("v", DATA["labels"], ids=lambda v: v["label"])
def test_parse_harte_labels(v):
    assert list(H.parse_harte(v["label"])) == v["triple"]


@pytest.mark.parametrize("v", DATA["symbols"], ids=lambda v: v["text"])
def test_parse_typed_symbols(v):
    assert list(H.parse_symbol(v["text"])) == v["triple"]
    assert H.symbol_to_harte(v["text"]) == v["harte"]
    assert list(H.parse_harte(v["harte"])) == v["triple"]


@pytest.mark.parametrize("text", DATA["errors"])
def test_bad_input_raises(text):
    with pytest.raises(ValueError):
        H.parse_any(text)


def test_every_candidate_round_trips_through_harte_and_symbol():
    for c in V.candidates().cands:
        label = H.to_harte(c.root, c.quality, c.bass)
        assert H.parse_harte(label) == (c.root, c.quality, c.bass), label
        sym = H.harte_to_symbol(label)
        assert H.parse_symbol(sym) == (c.root, c.quality, c.bass), (label, sym)


def test_harte_bass_degrees_use_chord_tone_names():
    assert H.to_harte(0, "aug", 8) == "C:aug/#5"
    assert H.to_harte(0, "dim7", 9) == "C:dim7/bb7"
    assert H.to_harte(0, "m7b5", 6) == "C:hdim7/b5"
    assert H.to_harte(4, "maj", 6) == "E:maj/2"           # non-chord +2
    assert H.to_harte(4, "min", 9) == "E:min/4"           # non-chord +5
    assert H.to_harte(7, "maj", 5) == "G:maj/b7"          # non-chord +10
    assert H.to_harte(1, "m7", 4, "C#") == "C#:min7/b3"


def test_harte_in_key_uses_key_spelling():
    assert H.to_harte_in_key(3, "7", 10, key=11) == "D#:7/5"
    assert H.to_harte_in_key(0, "dim7", 0, key=11) == "C:dim7"
    assert H.to_harte_in_key(0, "dim7", 0, key=11, theoretical=True) == "B#:dim7"
    assert H.to_harte_in_key(10, "maj", 10, key=0) == "Bb:maj"
    assert H.to_harte_in_key(11, "+7", 11, key=11, transpose=1) == "C:aug(b7)"
    assert H.to_harte_in_key(-1, "N", -1, key=0) == "N"


def test_root_name_must_match_pc():
    with pytest.raises(ValueError):
        H.to_harte(0, "maj", 0, root_name="D")


def test_quality_from_intervals():
    assert H.quality_from_intervals({0, 4, 7}) == "maj"
    assert H.quality_from_intervals({0, 4}) == "maj"              # omitted optional 5th
    assert H.quality_from_intervals({0, 4, 10}) == "7"
    assert H.quality_from_intervals({0, 2, 4, 10}) == "9"
    assert H.quality_from_intervals({0, 4, 8, 10}) == "+7"
    with pytest.raises(ValueError):
        H.quality_from_intervals({0, 1, 2})
    assert H.quality_from_intervals({0, 2, 4, 5, 7, 10}, strict=False) in T.QUALITIES  # dominant 11


def test_n_and_x():
    assert H.parse_harte("N") == H.NO_CHORD
    assert H.parse_symbol("N.C.") == H.NO_CHORD
    assert H.to_harte(-1, "N", -1) == "N"
    assert H.harte_to_symbol("N") == "N"


def test_symbol_keeps_written_spelling():
    assert H.harte_to_symbol("F#:maj/3") == "F#/A#"
    assert H.harte_to_symbol("Gb:maj/3") == "Gb/Bb"
    assert H.harte_to_symbol("D#:7/3") == "D#7/G"          # double sharp simplified
    assert H.harte_to_symbol("E:maj/2") == "E/F#"
    assert H.harte_to_symbol("B:aug(b7)") == "B+7"
