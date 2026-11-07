"""sheets.align: synthetic evidence — a known chart in another key finds its transposition and
"good"; the same chords in a shuffled order are "different"; helpers round-trip."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nathan_chordify.sheets import align as A
from nathan_chordify.theory.vocab import candidates

SEQ_C = ["C:maj", "G:maj", "A:min", "F:maj", "C:maj", "G:maj", "F:maj", "G:maj",      # verse (8 bars)
         "F:maj", "G:maj", "E:min", "A:min", "F:maj", "G:maj", "C:maj", "C:maj"]        # chorus (8 bars)
BPB = 4
IBI = 0.5


def synth_audio(bar_chords: list[str], key_index: int = 0, p: float = 0.9, n_lead_bars: int = 2) -> A.Audio:
    """One cell per beat; each bar's chord family holds ``p`` of the posterior mass; silence before."""
    V = len(candidates().cands)
    n_beats = (len(bar_chords) + n_lead_bars) * BPB
    beats = np.arange(n_beats + 1) * IBI
    E = np.full((n_beats, V), math.log((1 - p) / (V - 1)))
    for bar, h in enumerate(bar_chords):
        members = A.nearest_family(A.family(h))
        for b in range(BPB):
            g = (bar + n_lead_bars) * BPB + b
            E[g, members] = math.log(p / len(members))
    n_fam = A.nearest_family(("N",))
    for g in range(n_lead_bars * BPB):
        E[g, n_fam] = math.log(p / len(n_fam))
    return A.Audio("yt-TESTTESTTES", beats, float(beats[-1]), beats.copy(), E, "synthetic", key_index, [], "test")


def chart_text(bar_chords: list[str], transpose: int, labels=("Verse 1", "Chorus")) -> str:
    half = len(bar_chords) // 2
    out = []
    for i, lab in enumerate(labels):
        out.append(f"[{lab}]")
        bars = [A.transpose_harte(h, transpose) for h in bar_chords[i * half:(i + 1) * half]]
        syms = [A.H.harte_to_symbol(h) for h in bars]
        out.append("| " + " | ".join(syms[:4]) + " |")
        out.append("| " + " | ".join(syms[4:]) + " |")
    return "\n".join(out) + "\n"


def test_chart_in_another_key_is_found_with_the_right_transposition():
    audio = synth_audio(SEQ_C + SEQ_C[8:])                  # verse, chorus, chorus again
    text = chart_text(SEQ_C, 2)                             # the chart is written in D
    al = A.align_chart_text(text, audio, BPB, written_key=2)
    assert al.transpose == -2 and al.verdict == "good" and al.score > 0.6 and al.beat_scale == 1.0
    assert al.coverage > 0.8 and [n for n, _, _ in al.form if n] == ["Verse 1", "Chorus", "Chorus"]
    # the aligned chords sit on the right bars, spelled in the recording's key (C major)
    first = next(c for c in al.chords if c[2] != "N")
    assert first[2] == "C:maj" and abs(first[0] - 2 * BPB * IBI) < 1e-6
    hartes = [h for _, _, h in al.chords]
    assert hartes[:4] == ["C:maj", "G:maj", "A:min", "F:maj"]
    assert al.sections[0][1] == "Verse 1" and abs(al.sections[0][0] - 4.0) < 1e-6
    assert al.chart_text(text).splitlines()[1] == "| C | G | Am | F |"


def test_shuffled_chart_is_a_different_song():
    audio = synth_audio(SEQ_C + SEQ_C[8:])
    rng = np.random.default_rng(3)
    shuffled = list(SEQ_C)
    while shuffled == SEQ_C:
        rng.shuffle(shuffled)
    al = A.align_chart_text(chart_text(shuffled, 0), audio, BPB)
    assert al.verdict == "different" and al.score < 0.4


def test_wrong_written_key_hint_does_not_mislead():
    audio = synth_audio(SEQ_C)
    al = A.align_chart_text(chart_text(SEQ_C, 5), audio, BPB, written_key=7)   # the page says G, the chart is in F
    assert al.transpose == -5 and al.verdict == "good"


def test_chart_beat_is_two_recording_beats():
    audio = synth_audio(SEQ_C + SEQ_C)
    # the same chords written with 2-beat bars (half-note beats): each chart beat = 2 recording beats
    short = chart_text(SEQ_C, 0).replace("| ", "| ").replace(" |", " |")
    al2 = A.align_chart_text(short, audio, 2)
    assert al2.transpose == 0 and al2.verdict == "good" and al2.beat_scale == 2.0


def test_split_templates_and_transpose_chart_text():
    text = "[Verse]\n| C . G . | Am |\n[Chorus]\n| F | G |\n[Verse]\n| C . G . | Am |\n[Chorus]\n| C | D |\n"
    tpl = A.split_templates(text)
    assert [n for n, _ in tpl] == ["Verse", "Chorus", "Chorus (2)"]
    moved = A.transpose_chart_text("| C . G/B . | Am7 |\n", 2, 2)
    assert moved.strip() == "| D . A/C# | Bm7 |"                 # trailing dots are implied
    assert A.transpose_harte("C:maj/3", 1) == "C#:maj/3" and A.transpose_harte("N", 3) == "N"


def test_chart_beats_marks_unknown_bars():
    cb = A.chart_beats("[Intro]\n| N? | C |\n", 4)
    assert cb.beat_label[:4] == ["X"] * 4 and cb.beat_label[4:] == ["C:maj"] * 4 and cb.K == 8


def test_dtw_identity_path_is_regular():
    C = np.full((6, 6), 3.0)
    np.fill_diagonal(C, 0.0)
    path = A.dtw(C)
    assert path == [(k, k) for k in range(6)]
    assert A._regularity(path) == 1.0
    T = A.chart_beat_times(path, 6, np.arange(7) * 0.5)
    assert np.allclose(T, np.arange(7) * 0.5)


def test_audio_from_analysis_fallback():
    from nathan_chordify import schema
    from nathan_chordify.server.stub import fixture_paths
    a = schema.Analysis.model_validate_json(fixture_paths()[0].read_text("utf-8"))
    audio = A.audio_from_analysis(a)
    assert audio.source == "analysis" and audio.E.shape[1] == len(candidates().cands)
    assert audio.key_index == 11 and len(audio.beats) > 100
    # the chord in a cell is the favourite family of that cell
    c = next(ch for ch in a.chords if ch.layer == "main" and ch.harte != "N" and ch.end - ch.start > 1.0)
    g = int(np.searchsorted(audio.cell_t, 0.5 * (c.start + c.end), side="right") - 1)
    assert int(np.argmax(audio.E[g])) in set(A.nearest_family(A.family(c.harte)).tolist())


@pytest.mark.parametrize("score, fit, cov, want", [(0.8, 0.9, 0.9, "good"), (0.5, 0.65, 0.9, "good"),
                                                   (0.5, 0.4, 0.9, "different"), (0.3, 0.7, 0.9, "maybe"),
                                                   (0.3, 0.5, 0.9, "different"), (0.8, 0.9, 0.3, "maybe"),
                                                   (0.1, 0.9, 0.9, "different")])
def test_verdict_thresholds(score, fit, cov, want):
    assert A.verdict_of(score, fit, cov) == want
