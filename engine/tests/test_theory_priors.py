"""WP6 key/idiom priors (§5.10) and trigrams (§6.4)."""
from __future__ import annotations

import pytest

from nathan_chordify.theory import harte as H
from nathan_chordify.theory import priors as P
from nathan_chordify.theory import vocab as V

B, C, Am = 11, 0, 21


def ch(label: str):
    return H.parse_harte(label)


@pytest.mark.parametrize("cp, c, key, expected", [
    # 回家 (key B)
    ("F#:maj/3", "G#:min", B, 2.5),         # diatonic + bass step A# → G# with a slash chord
    ("A:7", "E:min6", B, 1.0),              # ♭VII7 → iv (borrowed iv, A10)
    ("B:sus4", "B:maj", B, 3.0),            # sus resolution + diatonic (capped)
    ("E:maj7", "A:7", B, 0.8),              # IV → ♭VII
    ("B:aug(b7)", "E:maj7", B, 2.5),        # dominant → target + I+ → IV
    ("B:maj", "B:aug(b7)", B, 1.0),         # I → I+7
    ("F#:sus4(b7,9)", "B:maj", B, 2.5),     # IV/5-type → I + diatonic
    ("E:maj/2", "B:maj", B, 2.5),           # E/F# → B
    ("C:dim7", "C#:min7", B, 1.5),          # passing #1°7 → ii7
    ("D#:7/5", "G#:min", B, 3.0),           # V7/vi with bass step (capped)
    ("C#:min7", "D#:min7", B, 1.0),         # diatonic rising m7
    ("A:7", "B:maj", B, 0.8),               # ♭VII → I
    ("G:maj", "A:maj", B, 0.8),             # ♭VI → ♭VII
    ("B:maj", "C:maj", B, -2.5),            # chromatic semitone, nothing matched
    ("B:maj", "D:min", B, -1.0),            # non-diatonic target
    # key C
    ("C#:7", "C:maj", C, 1.5),              # tritone substitution ♭II7 → I
    ("E:min7", "Eb:min7", C, 1.0),          # chromatic descending m7
    ("A:min", "A:minmaj7", C, 1.0),         # line cliché
    ("A:minmaj7", "A:min7", C, 1.0),
    ("C:maj", "C:aug", C, 2.0),             # line cliché + I → I+
    ("G:7", "C:maj", C, 2.5),               # V7 → I + diatonic
    ("C:maj", "F:min", C, 1.5),             # C acts as V of F (major acting as V)
    # minor key (relative major mapping; V7 is diatonic in minor)
    ("E:7", "A:min", Am, 2.5),
    ("A:min", "E:7", Am, 1.0),
])
def test_prior(cp, c, key, expected):
    assert P.prior(ch(cp), ch(c), key) == pytest.approx(expected)


def test_prior_cap_and_n():
    assert P.prior(ch("D#:7/5"), ch("G#:min"), B, cap=2.0) == 2.0
    assert P.prior(ch("N"), ch("B:maj"), B) == 0.0
    assert P.prior(ch("B:maj"), ch("B:maj"), B) == 0.0


def test_prior_accepts_candidates():
    v = V.candidates()
    a = v.cands[V.index_of(9, "7", 9)]
    b = v.cands[V.index_of(4, "m6", 4)]
    assert P.prior(a, b, B) == pytest.approx(1.0)


def test_prior_bounds_over_vocab():
    v = V.candidates()
    sample = v.cands[1::37]
    for a in sample:
        for b in sample:
            x = P.prior(a, b, B)
            assert -2.5 <= x <= 3.0


@pytest.mark.parametrize("l, p, r, key, expected, name", [
    ("B:maj", "B:aug(b7)", "E:maj7", B, 1.5, "augmented passing"),
    ("E:maj7", "A:7", "E:min6", B, 1.5, "borrowed"),
    ("B:maj", "F#:maj/3", "G#:min", B, 3.0, "descending canon"),
    ("C#:min7", "B:maj/3", "E:maj", B, 3.0, "walk-up"),
    ("G#:min7", "A#:min7", "B:maj", B, 3.0, "rising m7"),
    ("E:maj", "F#:maj", "G#:min", B, 3.0, "walk-up"),
    ("D:min7", "C#:7", "C:maj", C, 3.0, "tritone substitution"),
    ("C:maj", "C#:dim7", "D:min", C, 3.0, "passing diminished"),
    ("F:maj", "F#:dim7", "C:maj/5", C, 3.0, "passing diminished"),
    ("F:maj7", "G:maj", "E:min7", C, 1.5, "royal road"),
    ("G:maj/b7", "E:min7", "A:min", C, 1.5, "royal road"),
    ("Ab:maj", "Bb:maj", "C:maj", C, 3.0, "bVI-bVII-I"),
    ("C:maj", "C:maj7/7", "C:maj/b7", C, 3.0, "chromatic bass"),
    ("C:maj7/7", "C:7/b7", "F:maj/3", C, 3.0, "chromatic bass"),
    ("A:min", "A:minmaj7", "A:min7", C, 1.5, "line cliche"),
    ("C:maj", "C:aug", "C:maj6", C, 1.5, "line cliche"),
    ("E:min7", "Eb:min7", "D:min7", C, 3.0, "chromatic m7"),
    ("C:maj", "E:7/3", "A:min", C, 1.5, "secondary dominant"),
    ("C:maj", "G:sus4", "G:maj", C, 1.5, "sus resolution"),
    ("C:maj", "F:maj/2", "C:maj", C, 1.5, "four over five"),
    ("C:maj", "F:maj/5", "C:maj", C, 1.5, "pedal neighbour"),
    ("A:min", "A:min/b7", "F:maj", C, 3.0, "descending canon"),
    ("F:maj", "C:maj/3", "D:min", C, 3.0, "descending canon"),
    ("D:min", "C:maj/3", "F:maj", C, 3.0, "walk-up"),
    ("F:maj", "G:maj", "A:min", Am, 3.0, "walk-up"),          # minor key via relative major
    ("C:maj", "F#:maj/5", "Bb:maj", C, 0.0, None),
    ("C:maj", "D:min", "E:min", C, 1.5, None),                 # only the bass line (C → D → E)
])
def test_trigram(l, p, r, key, expected, name):
    assert P.idiom(ch(l), ch(p), ch(r), key) == name
    assert P.trigram(ch(l), ch(p), ch(r), key) == pytest.approx(expected)


def test_trigram_bass_line_only():
    assert P.bass_monotone(ch("C:maj"), ch("D:min"), ch("E:min"))
    assert not P.bass_monotone(ch("C:maj"), ch("D:min"), ch("C:maj"))
    assert not P.bass_monotone(ch("C:maj"), ch("C:maj"), ch("D:min"))
    assert not P.bass_monotone(ch("C:maj"), ch("F:maj"), ch("G:maj"))
    assert P.trigram(ch("N"), ch("C:maj"), ch("D:min"), C) == 0.0
