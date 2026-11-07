"""WP5 (M1) tests: the bass tracker and P_bass (DESIGN §5.6).

Acceptance (§5.6): synthetic bass lines (harmonic-rich sawtooth tones, E1–G3) with 16th notes at
65 BPM and 8th notes at 130 BPM → ≥ 90% pitch-class accuracy per note and onset error ≤ 40 ms.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from nathan_chordify.notes import bass as nb
from nathan_chordify.rhythm import build_grid
from nathan_chordify.types import BassTrack, NoteEvents, Stems

SR = 22050


def saw_note(midi, dur, amp, rng):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = 440 * 2 ** ((midi - 69) / 12)
    y = np.zeros(n)
    for h in range(1, 40):
        if h * f0 > SR / 2 - 500:
            break
        y += np.sin(2 * np.pi * h * f0 * t + rng.uniform(0, 2 * np.pi)) / h
    env = np.exp(-t / 0.6) * np.minimum(1, t / 0.004)
    r = min(n, int(0.015 * SR))
    env[-r:] *= np.linspace(1, 0, r)
    return amp * y * env


def bass_line(bpm, per_beat, n, seed, gap=0.92, lo=28, hi=55, repeat=0.2, rests=()):
    """Random-walk line over E1–G3; `gap` = sounding fraction of each step; `rests` = silent steps."""
    rng = np.random.default_rng(seed)
    step = 60 / bpm / per_beat
    t0 = 0.3
    notes, midi = [], int(rng.integers(lo, hi + 1))
    for i in range(n):
        if i and rng.random() > repeat:
            midi = int(np.clip(midi + rng.choice([-7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 12, -12]), lo, hi))
        if i not in rests:
            notes.append((t0 + i * step, t0 + (i + gap) * step, midi))
    y = np.zeros(int((t0 + n * step + 0.5) * SR))
    for a, b, m in notes:
        x = saw_note(m, b - a, rng.uniform(0.15, 0.3), rng)
        i = int(a * SR)
        y[i:i + len(x)] += x
    y += 1e-3 * rng.standard_normal(len(y))
    return y.astype(np.float32), notes


def score(bt: BassTrack, notes):
    """(pc accuracy per note, onset recall within 40 ms, signed onset errors of the hits)."""
    on = np.array([n[0] for n in bt.notes])
    pc_ok, errs = 0, []
    for a, b, m in notes:
        if len(on):
            j = int(np.argmin(np.abs(on - a)))
            if abs(on[j] - a) <= 0.04:
                errs.append(on[j] - a)
        mid = (a + b) / 2
        cov = [n for n in bt.notes if n[0] <= mid < n[1]]
        pc_ok += bool(cov) and cov[0][2] % 12 == m % 12
    return pc_ok / len(notes), len(errs) / len(notes), np.asarray(errs)


def grid_for(duration, bpm=65.0):
    t = 0.3 + np.arange(int(duration * bpm / 60)) * 60 / bpm
    return build_grid(t, t[::4], np.full(len(t), 0.95), duration=duration)


# ------------------------------------------------------------------------------------------------
# §5.6 acceptance
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bpm,per_beat", [(65, 4), (130, 2)], ids=["16ths@65", "8ths@130"])
@pytest.mark.parametrize("gap", [0.92, 1.0], ids=["detached", "legato"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_acceptance_cqt(bpm, per_beat, gap, seed):
    y, notes = bass_line(bpm, per_beat, 64, seed=seed + 10 * per_beat + int(gap * 100))
    bt = nb.track_signal(y, SR)
    pc_acc, recall, errs = score(bt, notes)
    assert pc_acc >= 0.9
    assert recall >= 0.9
    assert np.mean(np.abs(errs)) <= 0.04
    assert abs(np.mean(errs)) <= 0.015            # the latency constant removes the bias
    assert all(src == "bass_stem" for *_, src in bt.notes)


def test_pyin_fallback_also_works():
    y, notes = bass_line(65, 4, 48, seed=3)
    bt = nb.track_signal(y, SR, method="pyin")
    pc_acc, recall, errs = score(bt, notes)
    assert pc_acc >= 0.9 and recall >= 0.9 and np.mean(np.abs(errs)) <= 0.04


def test_repeated_notes_are_separate_onsets():
    """Pumping eighths on one root (worship bass) still give one onset per note."""
    y, notes = bass_line(130, 2, 48, seed=5, repeat=0.85, gap=0.85)
    bt = nb.track_signal(y, SR)
    pc_acc, recall, _ = score(bt, notes)
    assert pc_acc >= 0.9 and recall >= 0.9


def test_piano_bleed_does_not_move_the_bass():
    rng = np.random.default_rng(9)
    y, notes = bass_line(65, 4, 48, seed=9, gap=0.95)
    beat = 60 / 65
    for k in range(int(len(y) / SR / beat)):
        root = int(rng.integers(60, 72))
        for m in (root, root + 4, root + 7):
            x = saw_note(m, beat * 0.95, 0.05, rng)
            i = int((0.3 + k * beat) * SR)
            y[i:i + len(x)] += x[:max(0, len(y) - i)]
    pc_acc, recall, _ = score(nb.track_signal(y, SR), notes)
    assert pc_acc >= 0.9 and recall >= 0.9


# ------------------------------------------------------------------------------------------------
# BassTrack structure, rests, segmentation
# ------------------------------------------------------------------------------------------------

def test_track_structure_and_rests():
    y, notes = bass_line(65, 4, 40, seed=4, rests=set(range(12, 24)))
    bt = nb.track_signal(y, SR)
    T = len(bt.frame_t)
    assert bt.salience.shape == (T, 12) and bt.energy_db.shape == (T,)
    assert np.allclose(bt.salience.sum(axis=1), 1.0)
    assert bt.frame_t[1] == pytest.approx(256 / SR)                 # 11.6 ms hop
    gap = (notes[11][1] + 0.05, notes[12][0] - 0.05)                # the rest between steps 11 and 24
    assert not [n for n in bt.notes if gap[0] < n[0] < gap[1]]
    assert not [n for n in bt.notes if n[0] < gap[1] and n[1] > gap[0] + 0.15]
    assert all(28 <= n[2] <= 55 for n in bt.notes)
    assert all(0 < n[3] <= 1 for n in bt.notes)


def test_short_segments_merge_into_previous_note():
    T = 200
    frame_t = np.arange(T) * nb.HOP / nb.SR
    S = np.zeros((T, nb.F0_BINS))
    S[:, 0] = 1.0                                                   # E1 throughout
    sal = nb._pc_softmax(S)
    edb = np.full(T, -20.0)
    notes = nb.segment_notes(np.array([0.2, 0.5, 0.54, 1.0]), frame_t, S, sal, edb, end_t=2.0)
    assert [round(n[0], 2) for n in notes] == [0.0, 0.2, 0.54, 1.0]   # 0.50–0.54 (40 ms) merged back
    assert notes[1][1] == pytest.approx(0.54)
    assert all(n[2] % 12 == 4 for n in notes)                       # E


def test_attack_skip_ignores_previous_note_leak():
    """The first 30 ms after an onset are not used for the pitch class."""
    T = 100
    frame_t = np.arange(T) * nb.HOP / nb.SR
    S = np.zeros((T, nb.F0_BINS))
    S[:, 0] = 1.0                                                   # E1
    on = 40
    S[on:on + 4, 0], S[on:on + 4, 5] = 0.0, 1.0                     # A1 leaks for 4 frames (0–35 ms)
    S[on + 4:on + 7, 0], S[on + 4:on + 7, 3] = 0.0, 1.0             # then G1 for 3 frames
    onsets = np.array([0.0, frame_t[on], frame_t[on + 7]])
    notes = nb.segment_notes(onsets, frame_t, S, nb._pc_softmax(S), np.full(T, -20.0), 1.1)
    assert [n[2] % 12 for n in notes] == [4, 7, 4]                  # without the skip the middle note is A


# ------------------------------------------------------------------------------------------------
# P_bass per cell
# ------------------------------------------------------------------------------------------------

def test_cell_probs_overlap_weighting_and_floor():
    t = np.arange(9) * 1.0                                          # 60 BPM → subdiv 4, cells of 0.25 s
    grid = build_grid(t, t[::4], np.full(9, 0.95), duration=9.0)
    T = 10
    bt = BassTrack(frame_t=np.arange(T) * 0.0116, salience=np.full((T, 12), 1 / 12), energy_db=np.zeros(T),
                   notes=[(0.0, 1.0, 40, 0.8, "bass_stem"),            # E fills cells 0-3
                          (1.0, 1.125, 43, 0.8, "bass_stem"),          # G: first half of cell 4
                          (1.125, 1.25, 45, 0.4, "bass_stem"),         # A: second half of cell 4
                          (1.25, 1.375, 47, 0.6, "bass_stem")])        # B: half of cell 5, rest empty
    P = nb.cell_probs(bt, grid)
    g0 = int(np.searchsorted(grid.cell_t, 0.0))
    assert P.shape == (len(grid.cell_t) - 1, 12) and np.allclose(P.sum(axis=1), 1.0)
    assert P[g0, 4] == pytest.approx(1.0)                           # fully covered: the floor vanishes
    assert P[g0 + 4, 7] == pytest.approx(2 / 3) and P[g0 + 4, 9] == pytest.approx(1 / 3)
    expect_b = 0.5 * 0.6 / (0.5 * 0.6 + 0.05 * 0.5)
    assert P[g0 + 5, 11] == pytest.approx(expect_b + (1 - expect_b) / 12)
    assert np.allclose(P[g0 + 7], 1 / 12)                           # no bass: uniform


def test_cell_probs_follow_the_line():
    y, notes = bass_line(65, 2, 32, seed=6)                         # 8ths at 65: two notes per beat
    bt = nb.track_signal(y, SR)
    grid = grid_for(len(y) / SR)
    P = nb.cell_probs(bt, grid)
    ok = 0
    for a, b, m in notes:
        mid = (a + b) / 2
        g = int(np.searchsorted(grid.cell_t, mid, side="right")) - 1
        ok += int(np.argmax(P[g])) == m % 12
    assert ok / len(notes) >= 0.9


def test_track_reads_the_bass_stem_and_uses_the_left_hand_where_quiet(tmp_path):
    y, notes = bass_line(65, 4, 40, seed=8, rests=set(range(20, 40)))
    path = tmp_path / "bass.flac"
    sf.write(path, np.stack([y, y], axis=1), SR, subtype="PCM_16")
    stems = Stems(mix=path, vocals=path, drums=path, bass=path, other=path, harmonic=path, model="none")
    grid = grid_for(len(y) / SR)
    quiet_t0 = notes[-1][1] + 1.6                                   # > 1 s after the stem goes silent
    amt = NoteEvents(onset=np.array([quiet_t0, quiet_t0, quiet_t0 + 0.5, 1.0]),
                     offset=np.array([quiet_t0 + 0.4, quiet_t0 + 0.4, quiet_t0 + 0.9, 1.3]),
                     pitch=np.array([43, 50, 45, 40]), velocity=np.array([80, 70, 90, 90]),
                     offset_pedal=np.array([quiet_t0 + 0.45, quiet_t0 + 0.45, quiet_t0 + 1.0, 1.3]),
                     source="piano_amt@other")
    bt = nb.track(stems, grid)
    assert all(src == "bass_stem" for *_, src in bt.notes)
    bt2 = nb.track(stems, grid, amt)
    lh = [n for n in bt2.notes if n[4] == "amt_lh"]
    assert [(round(n[0], 3), n[2]) for n in lh] == [(round(quiet_t0, 3), 43), (round(quiet_t0 + 0.5, 3), 45)]
    assert lh[0][1] == pytest.approx(quiet_t0 + 0.45)
    assert len([n for n in bt2.notes if n[4] == "bass_stem"]) == len(bt.notes)   # the 1.0 s AMT note is ignored
