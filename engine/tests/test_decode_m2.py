"""M2 regression tests for the evidence / pruning / post-processing fixes and the decode state.

Each test pins one bug found on real lv-chordia output (回家, mini-D2): see eval/reports/M2.md.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nathan_chordify import pipeline
from nathan_chordify.decode import calibrate as cal
from nathan_chordify.decode import evidence as evm
from nathan_chordify.decode import post
from nathan_chordify.theory.vocab import candidates, index_of
from nathan_chordify.types import (BassTrack, DecoderParams, Evidence, KeyChange, KeyTrack, Segment)

from test_decode import HEADS, SUB, bass_track, key_track, lv_evidence, run, triple, vidx


def test_power_chord_needs_an_ambiguous_third():
    """'5' scores log(2·min(p_maj, p_min)): far below a confident major triad, and above both
    triads only when the network splits its third evenly."""
    tri = np.full((2, 73), 1e-6)
    tri[0, evm.lv_triad_index(0, "maj")] = 0.98            # confident C major
    tri[1, evm.lv_triad_index(0, "maj")] = 0.45            # C, third unclear
    tri[1, evm.lv_triad_index(0, "min")] = 0.45
    heads = {"triad": tri / tri.sum(1, keepdims=True), "bass": np.full((2, 13), 1 / 13)}
    for h, n in HEADS.items():
        if h not in heads:
            v = np.zeros((2, n))
            v[:, 0] = 1.0
            heads[h] = v
    cols = evm.lv_columns(heads)
    A = evm.lv_coefficients(candidates())
    raw = cols @ A.T
    c, c5 = index_of(0, "maj", 0), index_of(0, "5", 0)
    assert raw[0, c] - raw[0, c5] > 3.0                     # v1 mapping: only log 2 ≈ 0.69
    assert raw[1, c5] > raw[1, c]


def test_prune_keeps_the_class_that_wins_most_cells():
    """With the cap binding, a class that is top-1 in most cells must survive even when dozens of
    one-cell winners tie with it on the best margin (回家: B major was dropped, B7 kept)."""
    vocab = candidates()
    V = len(vocab.cands)
    G = 60
    rng = np.random.default_rng(0)
    E = np.full((G, V), -60.0) + rng.normal(0, 0.1, (G, V))
    big = index_of(11, "maj", 11)
    E[:, big] = -1.0
    small = [i for i in range(1, V) if len(vocab.classes[vocab.cands[i].cls]) == 1 and i != big][:G]
    for g, i in enumerate(small):                            # 60 one-cell winners, margin 0
        if g % 2 == 0:
            E[g, i] = 0.0
            E[g, big] = -0.5
    ev = Evidence(C=np.arange(V), E=E, E_short=E, parts={})
    p = replace(DecoderParams(), prune_cap=20, prune_top=3)
    C = evm.prune(ev, vocab, None, p)
    assert big in set(C.tolist())
    assert len(C) <= 20


def test_silent_bass_stem_drops_out_of_the_mean():
    """Cells without bass notes take the lv source alone: a flat bas must not scale lv down."""
    grid = cal.synthetic_grid(65, n_bars=2, subdiv=SUB)
    B, E = (11, "maj", 11), (4, "maj", 4)
    evid = lv_evidence(grid, [(0, 4, [(B, 1.0)]), (4, 8, [(E, 1.0)])])
    bass = bass_track(grid, [(0, 4, 11)])                    # bass only in bar 1
    p = DecoderParams()
    ev = evm.build(grid, evid, None, bass, candidates(), np.arange(len(candidates().cands)), p)
    act = ev.parts["aux:act:bas"]
    assert act[:4 * SUB].min() > 0.9 and act[5 * SUB:].max() == 0.0
    Bk = evm.bracket(ev, grid, p)
    assert np.allclose(Bk[5 * SUB:], ev.parts["lv"][5 * SUB:])


def test_quiet_bass_rumble_is_down_weighted():
    """Bass-stem content far below the song's own bass level (kick / riser bleed) gets ~0 weight."""
    grid = cal.synthetic_grid(65, n_bars=2, subdiv=SUB)
    ft = np.arange(0, grid.cell_t[-1], 256 / 22050)
    e = np.where(ft < grid.cell_t[-1] / 2, -18.0, -50.0)
    bt = BassTrack(frame_t=ft, salience=np.full((len(ft), 12), 1 / 12), energy_db=e,
                   notes=[(0.0, float(grid.cell_t[-1]), 43, 0.2, "bass_stem")])
    w = evm.bass_coverage(bt, grid)
    G = len(grid.cell_t) - 1
    assert w[: G // 2 - 2].min() > 0.95 and w[G // 2 + 2:].max() < 0.1


def test_long_chord_is_not_a_bass_event_of_its_neighbour():
    """I (4 beats) → vi7 (4 beats): vi7 contains I's tones, but I must stay I (not vi7/1)."""
    grid = cal.synthetic_grid(65, n_bars=3, subdiv=SUB)
    C, Am7, F = (0, "maj", 0), (9, "m7", 9), (5, "maj", 5)
    evid = lv_evidence(grid, [(0, 4, [(C, 1.0)]), (4, 8, [(Am7, 1.0)]), (8, 12, [(F, 1.0)])])
    r = run(grid, evid, key_track(grid, 0), DecoderParams())
    mains = [s for s in r.final if s.layer == "main"]
    assert [triple(r.vocab, s) for s in mains] == [C, Am7, F]
    assert all(s.role == "chord" for s in mains)


def test_nine_sus_four_is_named_by_its_upper_structure():
    vocab = candidates()
    g9 = index_of(7, "9sus4", 7)
    fg = index_of(5, "maj", 7)
    assert vocab.cands[g9].cls == vocab.cands[fg].cls
    assert post.upper_structure_member(vocab, g9) == fg
    assert post.upper_structure_member(vocab, fg) == fg


def test_full_vocabulary_calibration_matches_subset():
    """build(C) is a slice of build(all): bracket differences are those of a rebuild over C."""
    grid = cal.synthetic_grid(65, n_bars=2, subdiv=SUB)
    evid = lv_evidence(grid, [(0, 8, [((0, "maj", 0), 0.7), ((9, "min", 9), 0.3)])])
    vocab = candidates()
    p = DecoderParams()
    full = evm.build(grid, evid, None, None, vocab, np.arange(len(vocab.cands)), p)
    C = np.array([0, index_of(0, "maj", 0), index_of(9, "min", 9)])
    sub = evm.build(grid, evid, None, None, vocab, C, p)
    assert np.allclose(sub.E, full.E[:, C])
    assert sub.parts["aux:raw:lv"].shape[1] == 3


def test_decode_state_round_trip(tmp_path):
    grid = cal.synthetic_grid(65, n_bars=2, subdiv=SUB)
    bt = bass_track(grid, [(0, 4, 11), (4, 8, 4)])
    bt.frame_t = np.arange(10) * 0.0116
    bt.salience = np.full((10, 12), 1 / 12)
    bt.energy_db = np.full(10, -20.0)
    key = KeyTrack(11, "major", 0.8, np.full(len(grid.beats), 11),
                   [KeyChange(t=1.0, bar=1, from_key=11, to_key=1, kind="step_up")])
    st = pipeline.DecodeState(dir=tmp_path, key="test-key", meta={"title": "t", "evidence": []},
                              grid=grid, bass=bt, key0=key, hcdf=np.array([0.5, 1.0]))
    pipeline.save_state(st)
    back = pipeline.load_state(tmp_path)
    assert back.key == "test-key" and back.meta["title"] == "t"
    assert np.allclose(back.grid.cell_t, grid.cell_t) and back.grid.bars == grid.bars
    assert back.bass.notes == bt.notes
    assert back.key0.changes[0].to_key == 1 and back.key0.global_mode == "major"
    assert np.allclose(back.hcdf, [0.5, 1.0])


def test_two_beat_main_iii_is_not_imaj7_over_its_third():
    """Royal road: Gmaj7 (4 beats) → Bm (2 beats, main) → … : Bm = Gmaj7 minus its root, but a
    main chord of 2 beats is a chord, not a bass move under Gmaj7 (it used to become Gmaj7/B)."""
    grid = cal.synthetic_grid(65, n_bars=3, subdiv=SUB)
    Gmaj7, Bm, Em = (7, "maj7", 7), (11, "min", 11), (4, "min", 4)
    evid = lv_evidence(grid, [(0, 4, [(Gmaj7, 1.0)]), (4, 6, [(Bm, 1.0)]), (6, 12, [(Em, 1.0)])])
    r = run(grid, evid, key_track(grid, 7), DecoderParams())
    mains = [s for s in r.final if s.layer == "main"]
    assert [triple(r.vocab, s) for s in mains] == [Gmaj7, Bm, Em]
    assert all(s.role == "chord" for s in mains)


def test_prior_table_matches_prior_on_random_pairs():
    """The transposed full-vocabulary prior table equals theory.priors.prior for every key."""
    from nathan_chordify.decode import hsmm
    from nathan_chordify.theory import priors as PR
    vocab = candidates()
    rng = np.random.default_rng(3)
    C = np.unique(rng.integers(0, len(vocab.cands), 120))
    for key in (0, 4, 11, 12, 18, 23):
        M = hsmm.prior_matrix_for(C, vocab, key, 3.0)
        for a, b in rng.integers(0, len(C), (150, 2)):
            if a == b:
                continue
            ta, tb = vocab.cands[C[a]], vocab.cands[C[b]]
            assert M[a, b] == pytest.approx(PR.prior(ta, tb, key, 3.0), abs=1e-5), (ta, tb, key)
