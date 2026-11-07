"""WP6 piano voicings (shared/test_vectors/voicing.json) and the A17 rules on the whole vocab."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nathan_chordify.theory import harte as H
from nathan_chordify.theory import templates as T
from nathan_chordify.theory import vocab as V
from nathan_chordify.theory import voicing as VO

VEC = Path(__file__).resolve().parents[2] / "shared" / "test_vectors"
DATA = json.loads((VEC / "voicing.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("v", DATA["single"], ids=lambda v: v["chord"] + (" rootless" if v.get("rootless") else ""))
def test_single_voicings(v):
    r, q, b = H.parse_harte(v["chord"])
    vc = VO.voice(r, q, b, rootless=v.get("rootless", False))
    assert list(vc.lh) == v["lh"]
    if "rh" in v:
        assert list(vc.rh) == v["rh"]
    if "rh_pcs" in v:
        assert sorted(n % 12 for n in vc.rh) == sorted(v["rh_pcs"])


@pytest.mark.parametrize("v", DATA["sequences"], ids=lambda v: " ".join(v["chords"]))
def test_voice_leading_sequences(v):
    chords = [H.parse_harte(c) for c in v["chords"]]
    got = VO.voice_sequence(chords)
    assert [g.as_dict() for g in got] == v["voicings"]


@pytest.mark.parametrize("v", DATA["transpose"], ids=lambda v: f"{v['chord']}+{v['t']}")
def test_transpose_voicings(v):
    r, q, b = H.parse_harte(v["chord"])
    out = VO.transpose_voicings([VO.voice(r, q, b)], v["t"])
    assert list(out[0].lh) == v["lh"] and list(out[0].rh) == v["rh"]


def test_b_over_d_sharp_keeps_b_in_rh():
    vc = VO.voice(11, "maj", 3)                      # B/D#
    assert vc.lh[0] % 12 == 3
    pcs = [n % 12 for n in vc.rh]
    assert pcs.count(11) == 2 and 6 in pcs and 3 not in pcs


@pytest.mark.parametrize("rootless", [False, True])
def test_rules_hold_for_every_candidate(rootless):
    vocab = V.candidates()
    prev = None
    for c in vocab.cands[1:]:
        vc = VO.voice(c.root, c.quality, c.bass, prev, rootless=rootless)
        prev = vc
        assert len(vc.lh) == 1 and vc.lh[0] % 12 == c.bass
        assert VO.LH_RANGE[0] <= vc.lh[0] <= VO.LH_RANGE[1]
        assert 2 <= len(vc.rh) <= 4
        assert all(VO.RH_RANGE[0] <= n <= VO.RH_RANGE[1] for n in vc.rh)
        assert list(vc.rh) == sorted(set(vc.rh))
        assert vc.rh[-1] - vc.rh[0] <= VO.MAX_SPAN
        pcs = {n % 12 for n in vc.rh}
        chord = {(c.root + i) % 12 for i in T.tones(c.quality)}
        assert pcs <= chord | {c.root}
        iv = (c.bass - c.root) % 12
        if iv and iv in T.tones(c.quality):         # inversion: root kept, bass pc dropped
            assert c.root in pcs and c.bass not in pcs, c
        elif iv:                                    # non-chord bass: the upper chord in the RH
            assert c.bass not in pcs
            assert {(c.root + i) % 12 for i in T.required(c.quality)} - {(c.root + 7) % 12} <= pcs | {c.root}
        third_or_sus = [i for i in (3, 4, 2, 5) if i in T.required(c.quality)]
        if third_or_sus and (c.root + third_or_sus[0]) % 12 != c.bass:
            assert (c.root + third_or_sus[0]) % 12 in pcs, c   # the 3rd (or sus tone) always stays
        sev = [i for i in (10, 11) if i in T.tones(c.quality)]
        if sev and (c.root + sev[0]) % 12 != c.bass:
            assert (c.root + sev[0]) % 12 in pcs, c            # and the 7th


def test_top_note_term_is_a_bonus():
    prev = (60, 64, 67)
    smooth = (60, 65, 69)                           # top moves 2: bonus applies
    assert VO.rh_cost(smooth, prev) == VO._movement(prev, smooth) - 2
    jump = (64, 69, 72)
    assert VO.rh_cost(jump, prev) == VO._movement(prev, jump)
    assert VO.rh_cost((58, 59, 64), None) >= 3      # minor 2nd below MIDI 60 costs +3


def test_lh_nearest_previous():
    assert VO.choose_lh(4, 50) == 52                # E: 40 or 52, nearest 50
    assert VO.choose_lh(4, 41) == 40
    assert VO.choose_lh(2, None) == 38              # tie at 44 → lower
    assert VO.choose_lh(11, None) == 47


def test_n_and_empty():
    assert VO.voice(-1, "N", -1) == VO.EMPTY
    seq = VO.voice_sequence([(0, "maj", 0), (-1, "N", -1), (5, "maj", 5)])
    assert seq[1] == VO.EMPTY and seq[2].rh                        # N does not reset voice leading
    assert VO.rh_pitch_classes(-1, "N", -1) == []


def test_transpose_shifts_whole_song():
    song = VO.voice_sequence([(11, "maj", 11), (4, "maj7", 4), (6, "maj", 10), (8, "min", 8)])
    for t in range(12):
        out = VO.transpose_voicings(song, t)
        shifts = {o.rh[0] - s.rh[0] for o, s in zip(out, song)} | {o.lh[0] - s.lh[0] for o, s in zip(out, song)}
        assert len(shifts) == 1                     # one shift for the whole song
        assert (shifts.pop() - t) % 12 == 0
