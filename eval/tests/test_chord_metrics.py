"""Tests for eval/chord_metrics.py on hand-made cases (120 BPM: a beat is 0.5 s)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval import chord_metrics as cm  # noqa: E402

BPM = 120.0
SPB = 60.0 / BPM

# | C . . G/B | Am . . (Am Am/G) | F . . (F C/E) | Dm7 . G . | C . . E7/G# | Am . . . | F/G . . . | C . . . |
REF = [(3, "C:maj"), (1, "G:maj/3"), (3.5, "A:min"), (0.5, "A:min/b7"), (3.5, "F:maj"), (0.5, "C:maj/3"),
       (2, "D:min7"), (2, "G:maj"), (3, "C:maj"), (1, "E:7/3"), (4, "A:min"), (4, "F:maj/2"), (4, "C:maj")]
TAGS = {"G:maj/3": "descending_canon", "A:min/b7": "descending_canon", "C:maj/3": "descending_canon",
        "E:7/3": "secondary_dominant", "F:maj/2": "four_over_five"}


def lab(seq, spb=SPB):
    """[(beats, label)] → (intervals, labels)."""
    edges = np.concatenate([[0.0], np.cumsum([b for b, _ in seq])]) * spb
    return np.column_stack([edges[:-1], edges[1:]]), [x for _, x in seq]


def beats_for(seq, spb=SPB):
    return np.arange(0, sum(b for b, _ in seq) + 1e-9, 1.0) * spb


def score(est_seq, ref_seq=REF, spb=SPB, tags=None):
    ref_iv, ref_lab = lab(ref_seq, spb)
    est_iv, est_lab = lab(est_seq, spb) if not isinstance(est_seq, tuple) else est_seq
    return cm.evaluate_all(ref_iv, ref_lab, est_iv, est_lab, beats_for(ref_seq, spb), tags)


def ref_tags():
    return [TAGS.get(x) for _, x in REF]


def without(seq, label):
    """Merge every `label` segment into the chord before it (a detector that misses it)."""
    out = []
    for b, x in seq:
        if x == label and out:
            out[-1] = (out[-1][0] + b, out[-1][1])
        else:
            out.append((b, x))
    return out


def test_perfect_prediction_scores_one():
    res = score(REF, tags=ref_tags())
    for k, v in res["standard"].items():
        assert v == pytest.approx(1.0), k
    for d in ("shortseg_1beat", "shortseg_0.6s"):
        for lv in cm.LEVELS:
            s = res[d][lv]
            assert (s["precision"], s["recall"], s["f1"]) == (1.0, 1.0, 1.0)
            assert s["misses"] == []
    assert res["shortseg_1beat"]["root"]["n_ref"] == 4            # G/B, Am/G, C/E, E7/G#
    assert res["flicker"]["count"] == 0 and res["flicker"]["rate"] == 0.0
    assert res["boundary_100ms"]["f"] == 1.0 and res["boundary_qbeat"]["f"] == 1.0
    assert res["bass"]["acc"] == 1.0 and res["bass"]["seg_acc"] == 1.0
    assert res["ooc_bass"]["acc"] == 1.0 and res["ooc_bass"]["n_seg"] == 2   # Am/G and F/G
    bins = res["duration_bins"]
    assert bins["(0,0.5]"]["n"] == 2 and bins["(0.5,1]"]["n"] == 2
    assert all(row[lv]["acc"] == 1.0 for row in bins.values() if row["n"] for lv in cm.LEVELS)
    assert all(row["recall"]["sevenths_inv"] == 1.0 for row in res["idioms"].values())


def test_missing_half_beat_passing_chord():
    res = score(without(REF, "C:maj/3"), tags=ref_tags())          # F . . (F C/E): C/E missed
    short = res["shortseg_1beat"]["majmin_inv"]
    assert (short["hits"], short["n_ref"]) == (3, 4) and short["recall"] == 0.75
    assert short["misses"] == [[pytest.approx(5.75), pytest.approx(6.0), "C:maj/3"]]
    assert short["precision"] == 1.0
    assert res["standard"]["majmin"] > 0.98                      # 0.25 s of 20 s
    assert res["duration_bins"]["(0,0.5]"]["majmin_inv"]["acc"] == pytest.approx(0.5)
    assert res["idioms"]["descending_canon"]["recall"]["majmin_inv"] == pytest.approx(2 / 3)
    assert res["idioms"]["secondary_dominant"]["recall"]["majmin_inv"] == 1.0
    assert res["flicker"]["count"] == 0


def test_flickery_prediction_raises_flicker_rate():
    est = []
    for b, x in REF:                                              # a 0.2-beat blip inside every long chord
        if b >= 2:
            est += [(b / 2, x), (0.2, "D:maj"), (b / 2 - 0.2, x)]
        else:
            est.append((b, x))
    res = score(est)
    n_blips = sum(1 for b, _ in REF if b >= 2)
    assert res["flicker"]["count"] == n_blips
    assert res["flicker"]["rate"] == pytest.approx(n_blips / (sum(b for b, _ in REF) * SPB / 60))
    assert res["flicker"]["rate"] > 2
    assert score(REF)["flicker"]["rate"] == 0.0
    s = res["shortseg_1beat"]["majmin_inv"]
    assert s["recall"] == 1.0 and s["precision"] < 0.5
    assert res["boundary_100ms"]["precision"] < 0.5


def test_slash_bass_errors_hit_inv_and_bass_not_majmin():
    est = [(b, x.split("/")[0]) for b, x in REF]                  # every slash bass dropped
    res = score(est)
    std = res["standard"]
    assert std["majmin"] == pytest.approx(1.0)
    assert std["root"] == pytest.approx(1.0)
    assert std["majmin_inv"] < 0.95 and std["sevenths_inv"] < 0.95
    assert res["bass"]["acc"] < 0.95
    assert res["ooc_bass"]["acc"] == 0.0                          # Am/G and F/G both wrong
    # Am/G → Am merges into the Am before it, so even the root level loses that onset
    assert res["shortseg_1beat"]["root"]["recall"] == 0.75
    assert res["shortseg_1beat"]["majmin_inv"]["recall"] == 0.0


def test_out_of_gamut_passing_chords_use_fallbacks():
    assert cm.compare("B:aug(b7)", "B:aug", "majmin_inv") == 1.0   # triads_inv fallback
    assert cm.compare("B:aug(b7)", "B:aug", "sevenths_inv") == 0.0  # tetrads_inv fallback
    assert cm.compare("B:aug(b7)", "B:7", "majmin_inv") == 0.0
    assert cm.compare("C#:dim7", "C#:dim7", "sevenths_inv") == 1.0
    assert cm.compare("A:min/b7", "A:min7/b7", "majmin_inv") == 1.0
    assert cm.has_out_of_chord_bass("F:maj/2") and cm.has_out_of_chord_bass("E:min/4")
    assert not cm.has_out_of_chord_bass("G:maj/3") and not cm.has_out_of_chord_bass("C:maj(9)/2")


def test_onset_tolerance_and_overlap():
    ref = [(3, "C:maj"), (1, "G:maj/3"), (4, "A:min")]
    late = (np.array([[0, 1.6], [1.6, 2.1], [2.1, 4.0]]), ["C:maj", "G:maj/3", "A:min"])  # 100 ms late
    assert score(late, ref)["shortseg_1beat"]["majmin_inv"]["recall"] == 1.0           # tol 125 ms
    later = (np.array([[0, 1.7], [1.7, 2.2], [2.2, 4.0]]), ["C:maj", "G:maj/3", "A:min"])  # 200 ms late
    assert score(later, ref)["shortseg_1beat"]["majmin_inv"]["recall"] == 0.0
    slow_late = (np.array([[0, 3.2], [3.2, 4.2], [4.2, 8.0]]), ["C:maj", "G:maj/3", "A:min"])
    assert score(slow_late, ref, spb=1.0)["shortseg_1beat"]["majmin_inv"]["recall"] == 1.0  # 60 BPM: tol 250 ms


def test_short_definitions_differ_at_slow_tempo():
    res = score(REF, spb=1.0)                                     # 60 BPM: a beat is 1 s
    assert res["shortseg_1beat"]["root"]["n_ref"] == 4
    assert res["shortseg_0.6s"]["root"]["n_ref"] == 2             # only the half-beat chords


def test_boundary_windows():
    plain = [(4, "C:maj"), (4, "F:maj"), (4, "G:maj"), (4, "C:maj")]   # changes 2 s apart at 120 BPM
    for spb, shift, f100, fq in ((SPB, 0.04, 1.0, 1.0), (SPB, 0.2, 0.0, 0.0), (1.0, 0.2, 0.0, 1.0)):
        ref_iv, ref_lab = lab(plain, spb)
        est_iv = ref_iv + shift
        est_iv[0, 0] = 0.0
        beats = beats_for(plain, spb)
        assert cm.boundary_f(ref_iv, ref_lab, est_iv, ref_lab, beats, window=0.1)["f"] == pytest.approx(f100)
        assert cm.boundary_f(ref_iv, ref_lab, est_iv, ref_lab, beats, beat_window=0.25)["f"] == pytest.approx(fq)
    # with half-beat chords, a 0.2 s shift lands near a neighbouring change: F is low but not 0
    ref_iv, ref_lab = lab(REF)
    est_iv = ref_iv + 0.2
    est_iv[0, 0] = 0.0
    assert 0 < cm.boundary_f(ref_iv, ref_lab, est_iv, ref_lab, beats_for(REF), window=0.1)["f"] < 0.3


def test_gaps_and_partial_estimates():
    ref_iv, ref_lab = lab(REF)
    est_iv, est_lab = ref_iv[2:6], ref_lab[2:6]                   # covers only part of the song
    res = cm.evaluate_all(ref_iv, ref_lab, est_iv, est_lab, beats_for(REF))
    assert 0 < res["standard"]["majmin"] < 0.5
    assert res["shortseg_1beat"]["majmin_inv"]["hits"] == 2       # Am/G and C/E
    empty = cm.evaluate_all(ref_iv, ref_lab, np.zeros((0, 2)), [], beats_for(REF))
    assert empty["standard"]["root"] == 0.0 and empty["shortseg_1beat"]["root"]["f1"] == 0.0
    assert math.isnan(empty["shortseg_1beat"]["root"]["precision"])


def test_beat_helpers():
    beats = np.array([1.0, 1.5, 2.0, 3.0])
    assert np.allclose(cm.beat_position([0.5, 1.25, 2.5, 4.0], beats), [-1.0, 0.5, 2.5, 4.0])
    assert np.allclose(cm.beat_period([1.2, 2.6], beats), [0.5, 1.0])
    assert np.allclose(cm.onset_tolerance([1.2, 2.6], beats), [0.125, 0.25])
