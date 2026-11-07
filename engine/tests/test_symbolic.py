"""Symbolic chroma (DESIGN §5.7) and its use as the ``sym`` evidence source (§5.9) — synthetic notes.

Covers τ (decay), β (onset bonus and its 50 ms look-back), θ (saturating presence, no
max-normalisation), α (the onset feature dch), the pedal-extended offsets and the MIDI-55 split,
plus the M5 wiring: the ``sym`` activity, the left-hand bass activity, and ``pipeline._Run``.
"""
from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from nathan_chordify import pipeline
from nathan_chordify.decode import calibrate as cal
from nathan_chordify.decode import evidence as evm
from nathan_chordify.decode import params as prm
from nathan_chordify.notes import bass as nb
from nathan_chordify.notes import piano_amt as amt
from nathan_chordify.notes.symbolic import SymParams, cell_chroma, raw_chroma
from nathan_chordify.theory.vocab import candidates, index_of
from nathan_chordify.types import BassTrack, DecoderParams, NoteEvents

from test_decode import SUB, key_track, lv_evidence

P0 = SymParams()
DT = 0.25                                       # cell length used by the hand-made grids


def grid_of(n: int, dt: float = DT):
    return SimpleNamespace(cell_t=np.arange(n + 1) * dt)


def notes(*rows, pedals=()) -> NoteEvents:
    """rows = (onset, offset, pitch, velocity); offset_pedal from ``pedals`` [(down, up)]."""
    a = np.array(rows, dtype=float).reshape(-1, 4)
    return amt.to_events(a, np.array(pedals, dtype=float).reshape(-1, 2))


def decay_integral(on, a, b, tau=P0.tau):
    lo, hi = max(a, on), b
    return tau * (math.exp(-(lo - on) / tau) - math.exp(-(hi - on) / tau)) if hi > lo else 0.0


# ---------------------------------------------------------------------------------------------
# τ, β, θ, α
# ---------------------------------------------------------------------------------------------

def test_decay_integral_matches_tau_formula():
    ne = notes((0.0, 10.0, 60, 127))                           # C4, v = 1, held
    ch = raw_chroma(ne, grid_of(8).cell_t)
    for g in range(8):
        a, b = g * DT, (g + 1) * DT
        want = decay_integral(0.0, a, b) / DT + (P0.beta if g == 0 else 0.0)
        assert ch[g, 0] == pytest.approx(want, rel=1e-9)
    # after the onset cell, consecutive cells shrink by exp(−Δ/τ)
    assert ch[3, 0] / ch[2, 0] == pytest.approx(math.exp(-DT / P0.tau), rel=1e-9)
    assert np.count_nonzero(ch[:, 1:]) == 0


def test_tau_controls_the_memory():
    ne = notes((0.0, 10.0, 64, 127))
    slow = raw_chroma(ne, grid_of(8).cell_t, SymParams(tau=1.6))
    fast = raw_chroma(ne, grid_of(8).cell_t, SymParams(tau=0.2))
    assert slow[6, 4] > 10 * fast[6, 4]


def test_onset_bonus_cell_and_50ms_lookback():
    # onset 30 ms before the edge of cell 2 (t = 0.5): beta counts in cell 1 (where it falls)
    # AND in cell 2 (on ∈ [a − 0.05, b)); not in cell 3
    ne = notes((0.47, 0.48, 67, 127))                         # a 10 ms blip: almost no sustain
    ch = raw_chroma(ne, grid_of(4).cell_t)
    assert ch[1, 7] >= P0.beta and ch[2, 7] == pytest.approx(P0.beta)
    assert ch[0, 7] == 0 and ch[3, 7] == 0
    # 80 ms before the edge: only its own cell
    ch2 = raw_chroma(notes((0.42, 0.43, 67, 127)), grid_of(4).cell_t)
    assert ch2[1, 7] > 0 and ch2[2, 7] == 0


def test_velocity_scales_linearly_and_theta_saturates_without_normalisation():
    g = grid_of(2)
    loud = raw_chroma(notes((0.0, 1.0, 60, 127)), g.cell_t)
    soft = raw_chroma(notes((0.0, 1.0, 60, 127 * 0.25)), g.cell_t)
    assert soft[0, 0] == pytest.approx(0.25 * loud[0, 0])
    x, _ = cell_chroma(notes((0.0, 1.0, 60, 127 * 0.25)), g)
    assert x[0, 0] == pytest.approx(1 - math.exp(-soft[0, 0] / P0.theta))
    assert x[0, 0] < 0.9                                       # a soft note alone is not "fully present"
    both, _ = cell_chroma(notes((0.0, 1.0, 60, 127), (0.0, 1.0, 72, 127)), g)   # C4 + C5: same pc adds up
    one, _ = cell_chroma(notes((0.0, 1.0, 60, 127)), g)
    assert one[0, 0] < both[0, 0] < 1.0
    bigtheta, _ = cell_chroma(notes((0.0, 1.0, 60, 127)), g, SymParams(theta=1.0))
    assert bigtheta[0, 0] < one[0, 0]


def test_dch_is_an_onset_feature():
    # E4 held from 0; G4 struck at 1.0 s
    ne = notes((0.0, 3.0, 64, 100), (1.0, 3.0, 67, 100))
    x, dch = cell_chroma(ne, grid_of(10))
    g_on = 4                                                   # cell [1.0, 1.25)
    assert dch[0, 4] > prm.ONSET_MIN                           # the first strike
    assert np.all(dch[1:, 4] == 0)                             # a decaying held note is no new onset (e^{−Δ/τ} < α)
    assert dch[g_on, 7] > prm.ONSET_MIN and np.all(dch[:g_on - 1, 7] == 0)
    assert x[g_on, 7] > 0.9
    # α = 0 → dch = ch
    _, d0 = cell_chroma(ne, grid_of(10), SymParams(alpha=0.0))
    assert np.allclose(d0, raw_chroma(ne, grid_of(10).cell_t))


# ---------------------------------------------------------------------------------------------
# pedal and split
# ---------------------------------------------------------------------------------------------

def test_sustain_ends_at_the_pedal_extended_offset():
    dry = notes((0.0, 0.3, 60, 127))
    wet = notes((0.0, 0.3, 60, 127), pedals=[(0.1, 1.2)])
    assert wet.offset_pedal[0] == pytest.approx(1.2) and dry.offset_pedal[0] == pytest.approx(0.3)
    cd = raw_chroma(dry, grid_of(8).cell_t)
    cw = raw_chroma(wet, grid_of(8).cell_t)
    assert cd[2, 0] == 0 and cw[2, 0] > 0 and cw[4, 0] > 0      # sounds on under the pedal …
    assert cw[5, 0] == 0                                        # … and stops at the release (1.2 s)
    assert cw[4, 0] == pytest.approx(decay_integral(0.0, 1.0, 1.2) / DT)


def test_midi_55_split():
    ne = notes((0.0, 1.0, 54, 127), (0.0, 1.0, 55, 127))       # F#3 → left hand, G3 → symbolic
    x, _ = cell_chroma(ne, grid_of(2))
    assert x[0, 6] == 0 and x[0, 7] > 0.9
    lh = nb.left_hand_line(ne)
    assert [m for _, _, m, _ in lh] == [54]
    x2, _ = cell_chroma(ne, grid_of(2), SymParams(split_midi=50))
    assert x2[0, 6] > 0.9


def test_no_notes_gives_zero_chroma():
    x, dch = cell_chroma(None, grid_of(3))
    assert x.shape == (3, 12) and not x.any() and not dch.any()
    x, _ = cell_chroma(amt.empty(), grid_of(3))
    assert not x.any()


# ---------------------------------------------------------------------------------------------
# the sym source (§5.9) and the M5 activities
# ---------------------------------------------------------------------------------------------

def test_sym_source_prefers_the_played_chord_and_fades_out_in_silence():
    vocab = candidates()
    grid = cal.synthetic_grid(65, n_bars=1, subdiv=SUB)
    G = len(grid.cell_t) - 1
    x = np.zeros((G, 12))
    x[:8, [0, 4, 8]] = 0.95                                    # C+ played in the first 2 beats, then silence
    p = DecoderParams()
    ev = evm.build_full(grid, [], x, None, vocab, p)
    caug, cmaj, cmaj7 = index_of(0, "aug", 0), index_of(0, "maj", 0), index_of(4, "maj7", 4)
    assert ev.parts["sym"][0, caug] > ev.parts["sym"][0, cmaj] + 2.0
    assert ev.parts["sym"][0, caug] > ev.parts["sym"][0, cmaj7]
    act = ev.parts["aux:act:sym"]
    assert np.all(act[:8] == 1.0) and np.all(act[8:] == 0.0)
    assert np.allclose(ev.parts["aux:sym:x"], x)
    # where the source is inactive the fused evidence equals the evidence without it
    evid = lv_evidence(grid, [(0, 4, [((0, "maj", 0), 1.0)])])
    with_sym = evm.build_full(grid, evid, x, None, vocab, p)
    without = evm.build_full(grid, evid, None, None, vocab, p)
    assert np.allclose(with_sym.E[8:], without.E[8:])
    assert not np.allclose(with_sym.E[:8], without.E[:8])


def test_left_hand_bass_notes_are_not_scaled_by_the_stem_level():
    grid = cal.synthetic_grid(65, n_bars=1, subdiv=SUB)
    ibi = grid.ibi_ref
    ft = np.arange(0, 4 * ibi, 0.0116)
    e = np.where(ft < 2 * ibi, -15.0, -70.0)                   # loud bass, then a silent stem
    bt = BassTrack(frame_t=ft, salience=np.full((len(ft), 12), 1 / 12), energy_db=e,
                   notes=[(0.0, 2 * ibi, 35, 0.9, "bass_stem"), (2 * ibi, 4 * ibi, 40, 0.9, "amt_lh")])
    act = evm.bass_coverage(bt, grid)
    assert act[:4].min() > 0.95 and act[-4:].min() > 0.95      # the left hand stays active where the stem is quiet
    bt.notes[1] = (2 * ibi, 4 * ibi, 40, 0.9, "bass_stem")     # a stem note there is bleed: faded out
    assert evm.bass_coverage(bt, grid)[-4:].max() < 0.05


def test_pipeline_run_switches_sym_and_left_hand():
    grid = cal.synthetic_grid(65, n_bars=2, subdiv=SUB)
    ibi = grid.ibi_ref
    evid = lv_evidence(grid, [(0, 8, [((11, "maj", 11), 1.0)])])
    ft = np.arange(0, 8 * ibi, 0.0116)
    bt = BassTrack(frame_t=ft, salience=np.full((len(ft), 12), 1 / 12), energy_db=np.full(len(ft), -80.0),
                   notes=[])                                    # silent bass stem (回家's piano-only verse)
    ne = notes((0.0, 3.0, 35, 90), (0.0, 3.0, 63, 90), (0.0, 3.0, 66, 90), (0.0, 3.0, 71, 90))   # B1 | D# F# B
    key = key_track(grid, 11)
    on = pipeline._Run(grid, evid, bt, key, candidates(), replace(DecoderParams(), use_sym=True, amt_bass=True),
                       None, notes=ne)
    on.build_all()
    assert "sym" in on.ev_all.parts and on.onsets.shape == (len(grid.cell_t) - 1, 12)
    assert any(n[4] == "amt_lh" and n[2] == 35 for n in on.bass_eff.notes)
    assert on.bass.notes == []                                   # the stem track itself is untouched
    off = pipeline._Run(grid, evid, bt, key, candidates(), replace(DecoderParams(), use_sym=False, amt_bass=False),
                        None, notes=ne)
    off.build_all()
    assert "sym" not in off.ev_all.parts and off.onsets is None and off.bass_eff is bt
    defaults = DecoderParams()
    assert defaults.use_sym is False and defaults.amt_bass is False   # §13.6 gate: shipped OFF (M5)
