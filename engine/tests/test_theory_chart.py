"""WP6 bar-text charts (DESIGN §0.1): parse, format, round trips."""
from __future__ import annotations

import numpy as np
import pytest

from nathan_chordify.theory import chart as CH
from nathan_chordify.types import Chart, ChartChord, RhythmGrid, Segment

EXAMPLE = """
[Verse 1]
| B . . B+7 | Emaj7 . A7 Em6 |
"""


def test_design_example():
    ch = CH.parse(EXAMPLE)
    assert [s.label for s in ch.sections] == ["Verse 1"]
    bar1, bar2 = ch.sections[0].bars
    assert bar1 == [ChartChord(0.0, "B:maj"), ChartChord(3.0, "B:aug(b7)")]
    assert bar2 == [ChartChord(0.0, "E:maj7"), ChartChord(2.0, "A:7"), ChartChord(3.0, "E:min6")]


def test_even_split_brackets_repeat_unsure_and_n():
    ch = CH.parse("| B E | % | C#m7 . (D#m7 E) . | N . F#/A#? . | B . (. B+7) . |")
    bars = ch.bars
    assert bars[0] == [ChartChord(None, "B:maj"), ChartChord(None, "E:maj")]
    assert bars[1] == bars[0]
    assert bars[2] == [ChartChord(0.0, "C#:min7"), ChartChord(2.0, "D#:min7"), ChartChord(2.5, "E:maj")]
    assert bars[3] == [ChartChord(0.0, "N"), ChartChord(2.0, "F#:maj/3")]
    assert bars[4] == [ChartChord(0.0, "B:maj"), ChartChord(2.5, "B:aug(b7)")]
    assert ch.sections[0].label is None
    assert ch.sections[0].unsure == {(3, 1)}          # F#/A#? is bar 3, chord 1
    pos, harte = bars[2][2]                           # bars hold (beat_pos, harte) pairs
    assert (pos, harte) == (2.5, "E:maj")


def test_repeat_copies_unsure_flags():
    ch = CH.parse("| Em/G? . . . | % |")
    assert ch.sections[0].unsure == {(0, 0), (1, 0)}
    assert CH.format(ch) == "| Em/G? . . . | Em/G? . . . |\n"


def test_triplet_brackets_and_held_bar():
    ch = CH.parse("| (A B C) . . . | . |")
    a, b, c = ch.bars[0]
    assert (a.pos, b.pos, c.pos) == (0.0, pytest.approx(1 / 3), pytest.approx(2 / 3))
    assert ch.bars[1] == []


def test_typed_symbols_are_stored_as_harte():
    ch = CH.parse("| E/F# Em/A C#m7/E D#7/G |")       # 4 tokens in 4/4 = one per beat
    assert [c.harte for c in ch.bars[0]] == ["E:maj/2", "E:min/4", "C#:min7/b3", "D#:7/3"]
    assert [c.pos for c in ch.bars[0]] == [0.0, 1.0, 2.0, 3.0]


def test_section_on_same_line_and_trailing_notes():
    ch = CH.parse("[Intro] | Emaj7 D#7sus4 | G#m/D# |   (L–M)\n[Chorus]\n| G#m F# |")
    assert [s.label for s in ch.sections] == ["Intro", "Chorus"]
    assert ch.sections[0].bars[0] == [ChartChord(None, "E:maj7"), ChartChord(None, "D#:sus4(b7)")]
    assert ch.sections[0].bars[1] == [ChartChord(None, "G#:min/5")]


ROUND_TRIPS = [
    EXAMPLE,
    "| B E | % | C#m7 . (D#m7 E) . | N . F#/A#? . |",
    "[A]\n| B . . . | B . E . | (B E) . . . | . | B |\n[B]\n| G#m7/F# . Em/A . |",
    "| C . F/G . | Cmaj7/B . C7/Bb . | F/A . Fm6/Ab . | C/G . (G7sus4 G7) . |",
    "| F#11 . E/F# . | C°7 . Cø . | Cm(maj7) . C6/9 . | Bbmaj7#11 . C13sus4 . |",
    "| B . . (. . . B+7) | Emaj7 (. A7) . Em6 |",
]


@pytest.mark.parametrize("text", ROUND_TRIPS)
def test_round_trip(text):
    ch = CH.parse(text)
    out = CH.format(ch)
    again = CH.parse(out)
    assert again == ch
    assert CH.format(again) == out                   # formatting is stable


@pytest.mark.parametrize("bad", [
    "| B E A F# G |",            # five chords in a 4-beat bar
    "| B . . . . |",
    "| % |",                     # nothing to repeat
    "| Bxyz |",
    "| (A7 Em6 |",
    "B E",                       # no bar lines
    "| () . . . |",
])
def test_errors(bad):
    with pytest.raises(CH.ChartError):
        CH.parse(bad)


def test_three_four():
    ch = CH.parse("| G . D/F# | Em . . |", beats_per_bar=3)
    assert ch.beats_per_bar == 3
    assert ch.bars[0] == [ChartChord(0.0, "G:maj"), ChartChord(2.0, "D:maj/3")]
    assert CH.parse(CH.format(ch), beats_per_bar=3) == ch


def _grid(n_bars=2, bpb=4, subdiv=2):
    n_beats = n_bars * bpb
    beats = np.arange(n_beats) * 0.5
    cells = n_beats * subdiv
    cell_beat = np.arange(cells + 1) / subdiv
    return RhythmGrid(
        beats=beats, beat_conf=np.ones(n_beats), bar_of_beat=np.repeat(np.arange(n_bars), bpb),
        beat_in_bar=np.tile(np.arange(1, bpb + 1), n_bars),
        bars=[(i * bpb, bpb) for i in range(n_bars)], meter=[(0, bpb, 4)], bpm=120.0, ibi_ref=0.5,
        subdiv=np.full(n_beats, subdiv), free=[], cell_t=cell_beat * 0.5, cell_beat=cell_beat,
        cell_pos=np.zeros(cells, dtype=np.int8))


def _seg(c0, c1):
    return Segment(c0=c0, c1=c1, t0=c0 * 0.25, t1=c1 * 0.25, cand=0, cls=0, layer="main")


def test_format_from_segments():
    grid = _grid()
    segs = [_seg(0, 6), _seg(6, 7), _seg(7, 16)]     # B for 3 beats, B+7 half a beat, Emaj7 from 3.5
    text = CH.format(segs, grid, ["B:maj", "B:aug(b7)", "E:maj7"],
                     sections=[{"label": "Pre-chorus", "start": 0.0}])
    assert text == "[Pre-chorus]\n| B . . (B+7 Emaj7) | . |\n"
    ch = CH.parse(text)
    assert ch.bars[0][-1] == ChartChord(3.5, "E:maj7")


def test_format_requires_labels():
    with pytest.raises(TypeError):
        CH.format([_seg(0, 4)], _grid())


def test_align_empty_chart():
    assert CH.align(Chart(), np.arange(8.0)) == []
