"""WP3 tests: tempo octave, bars, meter, free time, triplet test, cells, overrides, check_octave, HCDF.

Pure-NumPy tests drive ``rhythm.build_grid`` with synthetic beat lists. GPU tests render musical
synthetic grooves (kick/snare/hat + bass + keys; bare clicks fool Beat This!) and run the tracker.
The 回家 tests use cache/yt-Y7boUFbM76k when present (never downloaded here).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from nathan_chordify import config, rhythm
from nathan_chordify.rhythm import OnsetEnvelope, beats as rb, build_grid, check_octave, hcdf
from nathan_chordify.rhythm.grid import triplet_ratio
from nathan_chordify.types import AudioAsset, DecodeResult, RhythmOverrides, Segment, Stems

SR = 44100
HUIJIA = config.CACHE_DIR / "yt-Y7boUFbM76k"


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


gpu = pytest.mark.skipif(not _cuda(), reason="needs CUDA for Beat This!")


# ------------------------------------------------------------------------------------------------
# synthetic audio
# ------------------------------------------------------------------------------------------------

def _env(n, attack, decay):
    t = np.arange(n) / SR
    e = np.exp(-t / decay)
    a = int(attack * SR)
    if a:
        e[:a] *= np.linspace(0, 1, a)
    return e


def _kick():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(50 + 90 * np.exp(-t / 0.04)) / SR
    return 0.9 * np.sin(ph) * _env(n, 0.001, 0.12)


def _snare(rng):
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    noise = np.diff(rng.standard_normal(n), prepend=0)
    return 0.35 * noise * _env(n, 0.001, 0.06) + 0.3 * np.sin(2 * np.pi * 185 * t) * _env(n, 0.001, 0.05)


def _hat(rng, amp):
    n = int(0.08 * SR)
    return amp * np.diff(np.diff(rng.standard_normal(n + 2))) * _env(n, 0.0005, 0.015)


def _tone(midi, dur, amp, kind="saw", decay=None):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    y = np.zeros(n)
    for h in range(1, 13):
        if h * f0 > SR / 2 - 1000:
            break
        y += (1.0 / h if kind == "saw" else 1.0 / h ** 2) * np.sin(2 * np.pi * h * f0 * t + 0.3 * h)
    e = _env(n, 0.005, decay if decay else dur * 0.6)
    r = min(n, int(0.02 * SR))
    e[-r:] *= np.linspace(1, 0, r)
    return amp * y * e


def _add(buf, x, t):
    i = int(round(t * SR))
    if 0 <= i < len(buf):
        j = min(len(buf), i + len(x))
        buf[i:j] += x[:j - i]


CHORDS = [(48, [60, 64, 67]), (43, [59, 62, 67]), (45, [57, 60, 64]), (41, [57, 60, 65])]


def groove(beat_times, bpb, sub, seed=0, tail=3.0):
    """(mix, harmonic) of a groove: drums by meter, bass on every beat, keys comping on `sub`."""
    rng = np.random.default_rng(seed)
    bt = np.asarray(beat_times, float)
    n = int((bt[-1] + tail) * SR)
    dr, harm = np.zeros(n), np.zeros(n)
    K, S = _kick(), _snare(rng)
    for i, t in enumerate(bt):
        ibi = bt[i + 1] - t if i + 1 < len(bt) else bt[i] - bt[i - 1]
        pos = i % bpb
        root, triad = CHORDS[(i // bpb) % 4]
        if bpb == 4:
            _add(dr, K if pos in (0, 2) else S, t)
        elif bpb == 3:
            _add(dr, K if pos == 0 else 0.6 * S, t)
        else:
            _add(dr, K if pos == 0 else S, t)
        for f in sub:
            _add(dr, _hat(rng, 0.12 if f == 0 else 0.08), t + f * ibi)
            for m in triad:
                _add(harm, _tone(m, ibi * 0.45, 0.05 if f == 0 else 0.035, "tri", 0.25), t + f * ibi)
        _add(harm, _tone(root - 12 if root > 40 else root, ibi * 0.9, 0.25 if pos == 0 else 0.18), t)
    pk = np.abs(dr + harm).max()
    return ((dr + harm) / pk * 0.8).astype(np.float32), (harm / pk * 0.8).astype(np.float32)


def steady(bpm, n, t0=0.5):
    return t0 + np.arange(n) * 60.0 / bpm


def rubato_then_groove(seed=3, bpm=76.0, n_groove=48):
    """Piano chords at random IBIs (no drums), then a steady 4/4 groove. Returns mix, harm, start."""
    rng = np.random.default_rng(seed)
    intro = 0.5 + np.concatenate([[0], np.cumsum(rng.uniform(0.6, 1.9, 9))])
    start = float(intro[-1] + 1.2)
    head = np.zeros(int(start * SR))
    for i, t in enumerate(intro):
        root, triad = CHORDS[i % 4]
        for m in [root] + triad + [triad[0] + 12]:
            _add(head, _tone(m, 1.6, 0.07, "tri", 0.8), t)
    head *= 0.8 / np.abs(head).max()
    mix, harm = groove(steady(bpm, n_groove, t0=0.0), 4, [0, 0.5], seed=4)
    return (np.concatenate([head, mix]).astype(np.float32), np.concatenate([head, harm]).astype(np.float32), start)


def from_audio(mix, harm, **kw):
    """What ``rhythm.track`` does: tracker on the mix, onsets and harmonic changes of the harmonic part."""
    out = rb.track_signal(mix, SR)
    return build_grid(out.beats, out.downbeats, out.beat_conf, duration=len(mix) / SR,
                      beat_logit=out.beat_logit, downbeat_logit=out.downbeat_logit,
                      onset=rhythm.onset_envelope(harm, SR), harmonic_changes=hcdf.peaks(harm, SR), **kw)


# ------------------------------------------------------------------------------------------------
# helpers on grids
# ------------------------------------------------------------------------------------------------

def bar_lengths(g):
    return np.array([n for _, n in g.bars])


def median_bar_s(g):
    return float(np.median([g.beats[s + n] - g.beats[s] for s, n in g.bars if s + n < len(g.beats)]))


def metered_subdiv(g):
    return set(np.unique(g.subdiv[g.beat_in_bar > 0]).tolist())


def inner_free(g):
    """Free regions strictly inside the tracked music (not the pre-roll / ring-out)."""
    a, b = g.beats[0] + 0.1, g.beats[-1] - 0.1
    return [(x, y) for x, y in g.free if x > a and y < b]


def assert_cells_consistent(g, duration=None):
    ct, cb, cp = g.cell_t, g.cell_beat, g.cell_pos
    assert len(cp) == len(ct) - 1 and len(cb) == len(ct)
    assert np.all(np.diff(ct) > 0)
    if duration is not None:
        assert ct[-1] == pytest.approx(duration, abs=1e-6)
    free_cell = cp == 5
    for x, y in g.free:
        inside = (ct[:-1] >= x - 1e-9) & (ct[1:] <= y + 1e-9)
        assert np.all(free_cell[inside])
    covered = np.zeros(len(cp), bool)
    for x, y in g.free:
        covered |= (ct[:-1] >= x - 1e-9) & (ct[1:] <= y + 1e-9)
    assert np.array_equal(covered, free_cell)
    # metered cells per beat == subdiv
    starts = np.flatnonzero(~free_cell & np.isfinite(cb[:-1]) & (np.abs(cb[:-1] - np.round(cb[:-1])) < 1e-9))
    for c in starts:
        i = int(round(cb[c]))
        assert np.sum((cb[:-1] >= i - 1e-9) & (cb[:-1] < i + 1 - 1e-9) & ~free_cell) == g.subdiv[i]


def synth_beats(bpm, n, every, t0=1.0, conf=0.95):
    t = t0 + np.arange(n) * 60.0 / bpm
    return t, t[::every], np.full(n, conf)


# ------------------------------------------------------------------------------------------------
# pure NumPy: tempo octave
# ------------------------------------------------------------------------------------------------

def test_octave_rule_bpb8_folds_130_to_65():
    t, db, c = synth_beats(130, 128, 8)
    g = build_grid(t, db, c, duration=t[-1] + 1.0)
    assert g.bpm == pytest.approx(65, abs=0.5)
    assert np.all(bar_lengths(g)[:-1] == 4)
    assert median_bar_s(g) == pytest.approx(240 / 65, abs=0.02)
    assert metered_subdiv(g) == {4}
    assert g.meter == [(0, 4, 4)]
    assert_cells_consistent(g, t[-1] + 1.0)


def test_hierarchy_folds_eighths_and_merges_half_bars():
    """回家 pattern: 130 BPM eighths, "downbeats" every 4 eighths, alternating logits."""
    t, db, c = synth_beats(130, 128, 4)
    blog = np.where(np.arange(128) % 2 == 0, 9.0, 3.0)
    dlog = np.full(128, -10.0)
    dlog[0::8], dlog[4::8] = 10.0, 5.0
    g = build_grid(t, db, c, beat_logit=blog, downbeat_logit=dlog)
    assert g.bpm == pytest.approx(65, abs=0.5)
    assert np.all(bar_lengths(g)[:-1] == 4)
    assert median_bar_s(g) == pytest.approx(240 / 65, abs=0.02)
    assert np.allclose(g.beats[[s for s, _ in g.bars]][:10], t[0::8][:10])


def test_no_fold_at_130_without_evidence():
    t, db, c = synth_beats(130, 96, 4)
    g = build_grid(t, db, c, beat_logit=np.full(96, 8.0), downbeat_logit=np.where(np.arange(96) % 4 == 0, 10.0, -10.0))
    assert g.bpm == pytest.approx(130, abs=1)
    assert np.all(bar_lengths(g)[:-1] == 4)
    g2 = build_grid(t, db, c)                       # no logits, no slow regions: stays too
    assert g2.bpm == pytest.approx(130, abs=1)


@pytest.mark.parametrize("with_logits", [True, False])
def test_mixed_tracking_levels_are_unified(with_logits):
    """Intro tracked at 65 BPM, then eighths at 130 with half-bar downbeats and one missed eighth."""
    q = 60 / 65
    slow = 1.0 + np.arange(16) * q
    fast = slow[-1] + q / 2 * np.arange(1, 97)
    t = np.concatenate([slow, fast])
    on = np.concatenate([np.ones(16, bool), np.arange(1, 97) % 2 == 0])
    j = np.arange(1, 97)                            # eighths after the intro; the next bar starts at j = 2
    idx_bar = np.concatenate([np.arange(16) % 4 == 0, (j - 2) % 8 == 0])
    idx_half = np.concatenate([np.zeros(16, bool), (j - 2) % 8 == 4])
    drop = 16 + 40                                  # one missed off-beat eighth
    keep = np.ones(len(t), bool)
    keep[drop] = False
    assert not on[drop]
    t, on, idx_bar, idx_half = t[keep], on[keep], idx_bar[keep], idx_half[keep]
    db = t[idx_bar | idx_half]
    kw = {}
    if with_logits:
        kw = dict(beat_logit=np.where(on, 9.0, 2.5),
                  downbeat_logit=np.where(idx_bar, 10.0, np.where(idx_half, 5.5, -10.0)))
    g = build_grid(t, db, np.full(len(t), 0.95), **kw)
    assert g.bpm == pytest.approx(65, abs=0.5)
    assert len(g.beats) == 16 + 48
    expected = 1.0 + np.arange(len(g.bars)) * 4 * q  # bar lines continue the intro's phase
    assert np.allclose(g.beats[[s for s, _ in g.bars]], expected, atol=0.02)
    assert not inner_free(g)


def test_compound_rule_6_8():
    t, db, c = synth_beats(180, 96, 6)
    g = build_grid(t, db, c)
    assert g.bpm == pytest.approx(60, abs=0.5)
    assert np.all(bar_lengths(g)[:-1] == 2)
    assert g.meter == [(0, 6, 8)]
    assert metered_subdiv(g) == {6}


def test_slow_tempo_is_doubled():
    t, db, c = synth_beats(40, 32, 2)
    g = build_grid(t, db, c)
    assert g.bpm == pytest.approx(80, abs=0.5)
    assert np.all(bar_lengths(g)[:-1] == 4)


def test_missed_beat_is_filled_not_free():
    t, db, c = synth_beats(90, 64, 4)
    keep = np.ones(64, bool)
    keep[22] = False
    g = build_grid(t[keep], db, c[keep])
    assert len(g.beats) == 64 and np.allclose(g.beats, t, atol=1e-6)
    assert not inner_free(g)
    assert np.all(bar_lengths(g)[:-1] == 4)


def test_one_glitch_does_not_open_free_time():
    t, db, c = synth_beats(90, 64, 4)
    t = t.copy()
    t[30:] += 0.2                                   # one IBI 30% too long (not a missed beat)
    g = build_grid(t, t[0::4], c)
    assert not inner_free(g)


def _conf(logit):
    return rb.conf_from_logit(np.asarray(logit, float))


def twelve_eight_tracker(seed=0):
    """活出愛-like tracker output for a 12/8 song (dotted quarter q = 0.95 s, eighth e = q/3):
    A) 30 s tracked at eighths, every 3rd eighth stronger, "downbeats" every 6 eighths (half bars);
    B) 20 s of eighths with weak eighths sometimes missed; C) dotted quarters for the rest, mostly
    unsure (logit 3, conf 0.27) with a confident stretch, a near-duplicate glitch beat, downbeats
    every 2 beats. Returns (beats, downbeats, blog, dlog, bar-start times, q)."""
    rng = np.random.default_rng(seed)
    q = 0.951
    e = q / 3
    t0 = 1.14                                       # first downbeat; one dotted-quarter pickup before it
    times, blog, dlog = [], [], []

    def add(t, bl, dl):
        times.append(t + rng.normal(0, 0.006))
        blog.append(bl)
        dlog.append(dl)
    pickup = t0 - q
    for k in range(3):                              # pickup eighths
        add(pickup + k * e, 10.5 if k == 0 else 8.5, -10.0)
    nA = 32                                         # A: 32 dotted quarters = 8 bars of 12/8
    for b in range(nA):
        for k in range(3):
            on = k == 0
            dl = (8.0 if b % 4 == 0 else 5.5) if (on and b % 2 == 0) else -10.0
            add(t0 + (3 * b + k) * e, 10.5 if on else 8.5, dl)
    nB = 20                                         # B: weak eighths at logit ~1, some missed
    for b in range(nA, nA + nB):
        for k in range(3):
            if k and rng.random() < 0.35:
                continue
            on = k == 0
            add(t0 + (3 * b + k) * e, 8.5 if on else 1.0, (7.0 if on and b % 2 == 0 else -10.0))
    nC = 180                                        # C: dotted quarters, unsure except beats 60..100
    for b in range(nA + nB, nA + nB + nC):
        sure = 60 <= b - nA - nB < 100
        add(t0 + 3 * b * e, 7.0 if sure else 3.0, 4.0 if b % 2 == 0 else -9.0)
        if b == nA + nB + 130:
            add(t0 + 3 * b * e + 0.06, 0.5, -9.0)    # tracker glitch: a second peak 60 ms later
    t = np.asarray(times)
    blog, dlog = np.asarray(blog), np.asarray(dlog)
    db = t[dlog > 0]
    bar_starts = t0 + 4 * q * np.arange((nA + nB + nC) // 4)
    return t, db, blog, dlog, bar_starts, q


def test_twelve_eight_commits_to_dotted_quarters():
    """12/8 tracked partly at eighths, partly at dotted quarters (活出愛): one tempo octave (the
    dotted quarter), compound bars all the way, no 2/4 ↔ 4/4 ↔ 3/4 flips and no free time."""
    t, db, blog, dlog, bar_starts, q = twelve_eight_tracker()
    g = build_grid(t, db, _conf(blog), duration=t[-1] + 1.0, beat_logit=blog, downbeat_logit=dlog)
    assert g.bpm == pytest.approx(60 / q, abs=0.5)
    ibi = np.diff(g.beats)
    assert np.all(np.abs(ibi / q - 1) < 0.1)        # every beat is a dotted quarter (no eighths left)
    assert not inner_free(g)
    assert all(d == 8 and n in (6, 12) for _, n, d in g.meter) and len(g.meter) == 1
    assert len(set(bar_lengths(g)[:-1].tolist())) == 1
    starts = g.beats[[s for s, _ in g.bars]]
    assert all(np.min(np.abs(starts - x)) < 0.05 for x in bar_starts[:40])  # bar lines on 12/8 bars (or halves)
    assert metered_subdiv(g) == {6}
    assert_cells_consistent(g, t[-1] + 1.0)


def test_six_eight_at_eighths_still_folds_with_logits():
    """The ≥ 110 BPM compound rule now folds with the 3-phase Viterbi: the phase follows the
    stronger eighths and the downbeats even when the first tracked beat is a sub-beat."""
    e = 60 / 180
    t = 0.5 + np.arange(1, 97) * e                  # starts on the 2nd eighth of a bar
    pos = np.arange(1, 97) % 3
    blog = np.where(pos == 0, 10.0, 8.0)
    dlog = np.where(np.arange(1, 97) % 6 == 0, 8.0, -10.0)
    g = build_grid(t, t[dlog > 0], _conf(blog), beat_logit=blog, downbeat_logit=dlog)
    assert g.bpm == pytest.approx(60, abs=0.5)
    assert np.allclose(g.beats, t[pos == 0], atol=1e-6)
    assert g.meter == [(0, 6, 8)]


def slow_intro_then_groove(P=0.8676, n_groove=64):
    """Here I Bow-like: a 4-bar intro on one held chord that the tracker follows only at half tempo,
    off the beat and with a gap (logit ≈ 1), then a confident 4/4 groove. Returns tracker output,
    the groove start g0 and the intro start."""
    start = 0.54
    g0 = start + 16 * P
    intro = [start] + [4.2 + 2 * P * j for j in range(4)] + [g0 - 2 * P, g0 - P]
    groove = g0 + P * np.arange(n_groove)
    t = np.concatenate([intro, groove])
    blog = np.concatenate([np.full(len(intro), 1.0), np.full(n_groove, 7.0)])
    dlog = np.concatenate([[3.0], np.full(len(intro) - 1, -1.0),
                           np.where(np.arange(n_groove) % 4 == 0, 8.0, -8.0)])
    return t, blog, dlog, g0, start


def test_sparse_intro_inherits_the_song_tempo():
    t, blog, dlog, g0, start = slow_intro_then_groove()
    P = 0.8676
    g = build_grid(t, t[dlog > 0], _conf(blog), duration=t[-1] + 2.0, beat_logit=blog, downbeat_logit=dlog,
                   harmonic_changes=np.array([g0 + 4 * P * k for k in range(16)]))   # held chord until the groove
    assert g.bpm == pytest.approx(60 / P, abs=0.5)
    assert g.free[0] == (0.0, pytest.approx(start))                      # only the silent pre-roll
    assert not inner_free(g)
    assert np.allclose(g.beats[:17], start + P * np.arange(17), atol=0.03)  # backfilled at the song tempo
    assert np.all(g.beat_in_bar[:16] > 0)
    starts = g.beats[[s for s, _ in g.bars]]
    assert np.min(np.abs(starts - start)) < 0.03 and np.min(np.abs(starts - g0)) < 0.03   # 4 whole bars
    assert np.all(bar_lengths(g)[:-1] == 4)
    assert_cells_consistent(g, t[-1] + 2.0)


def test_rubato_intro_with_moving_chords_stays_free():
    t, blog, dlog, g0, start = slow_intro_then_groove()
    rng = np.random.default_rng(1)
    moving = np.sort(rng.uniform(start + 0.5, g0 - 0.5, 10))               # chord changes off the grid
    for changes in (moving, None):                                        # None: no harmony → no backfill
        g = build_grid(t, t[dlog > 0], _conf(blog), duration=t[-1] + 2.0, beat_logit=blog,
                       downbeat_logit=dlog, harmonic_changes=changes)
        assert g.free[0][0] == 0.0 and g.free[0][1] >= g0 - 1.8
        assert np.all(g.beat_in_bar[g.beats < g0 - 1.8] == 0)
        assert g.bpm == pytest.approx(60 / 0.8676, abs=0.5)


def test_unsure_steady_passage_after_the_groove_is_metered():
    """A soft passage (logit 2.5) in the middle of a confident song is in tempo, not free time;
    an irregular passage still is."""
    t, db, c = synth_beats(72, 120, 4)
    blog = np.full(120, 7.0)
    blog[40:80] = 2.5
    dlog = np.where(np.arange(120) % 4 == 0, 6.0, -8.0)
    g = build_grid(t, db, _conf(blog), duration=t[-1] + 1.0, beat_logit=blog, downbeat_logit=dlog)
    assert not inner_free(g) and len(g.bars) == 30
    t2 = t.copy()
    rng = np.random.default_rng(2)
    t2[50:70] += np.cumsum(rng.uniform(-0.25, 0.25, 20))                # rubato: the tempo wanders
    t2[70:] += t2[69] - t[69]
    g2 = build_grid(t2, t2[::4], _conf(blog), duration=t2[-1] + 1.0, beat_logit=blog, downbeat_logit=dlog)
    assert inner_free(g2)


# ------------------------------------------------------------------------------------------------
# pure NumPy: bars, free time, cells, overrides
# ------------------------------------------------------------------------------------------------

def test_pickup_and_mixed_meter():
    t = 1.0 + np.arange(40) * 0.6
    db_idx = [1, 5, 9, 11, 15, 19, 23, 27, 31, 35, 39]           # one-beat pickup, then a 2/4 bar
    g = build_grid(t, t[db_idx], np.full(40, 0.95))
    assert g.bar_of_beat[0] == -1 and g.beat_in_bar[0] == 4
    assert g.bars[:4] == [(1, 4), (5, 4), (9, 2), (11, 4)]
    assert g.meter == [(0, 4, 4), (2, 2, 4), (3, 4, 4)]
    assert g.beat_in_bar[9] == 1 and g.beat_in_bar[10] == 2 and g.beat_in_bar[11] == 1


def test_spurious_and_missing_downbeats_are_cleaned():
    t = 1.0 + np.arange(48) * 0.6
    db_idx = [0, 4, 8, 10, 12, 20, 24, 28, 32, 36, 40, 44]       # spurious at 10, missing at 16
    g = build_grid(t, t[db_idx], np.full(48, 0.95))
    assert [s for s, _ in g.bars] == list(range(0, 48, 4))


def test_free_time_rubato_intro_numpy():
    rng = np.random.default_rng(7)
    intro = 0.4 + np.concatenate([[0], np.cumsum(rng.uniform(0.5, 1.8, 10))])
    start = intro[-1] + 1.0
    groove_t = start + np.arange(48) * 0.75
    t = np.concatenate([intro, groove_t])
    conf = np.concatenate([np.full(len(intro), 0.2), np.full(48, 0.95)])
    g = build_grid(t, groove_t[::4], conf, duration=groove_t[-1] + 2.0)
    assert g.free[0][0] == 0.0
    assert g.free[0][1] == pytest.approx(start, abs=0.76)
    assert not inner_free(g) or all(b <= start + 0.8 for _, b in inner_free(g))
    assert np.all(g.beat_in_bar[:len(intro) - 1] == 0) and np.all(g.bar_of_beat[:len(intro) - 1] == -1)
    assert g.bpm == pytest.approx(80, abs=1)
    assert_cells_consistent(g, groove_t[-1] + 2.0)


def test_cell_layout_65_bpm():
    t, db, c = synth_beats(65, 33, 4)
    dur = t[-1] + 60 / 65 + 2.0
    g = build_grid(t, db, c, duration=dur)
    assert_cells_consistent(g, dur)
    assert g.free[0] == (0.0, pytest.approx(1.0)) and g.free[-1][1] == pytest.approx(dur)
    first = int(np.searchsorted(g.cell_t, t[0] - 1e-9))
    assert g.cell_pos[first:first + 16].tolist() == [0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4]
    assert np.allclose(g.cell_beat[first:first + 5], [0, 0.25, 0.5, 0.75, 1.0])
    assert np.isnan(g.cell_beat[0])                 # pre-roll
    n_metered = int(np.sum(g.cell_pos != 5))
    assert n_metered == 4 * 33                     # 32 beat intervals + the last beat's extrapolated one
    assert g.ibi_ref == pytest.approx(60 / 65)


def test_cell_pos_3_4_and_ternary():
    t, db, c = synth_beats(100, 30, 3)
    g = build_grid(t, db, c, overrides=RhythmOverrides(subdiv=[(0, 3)]))
    first = int(np.searchsorted(g.cell_t, t[0] - 1e-9))
    assert g.cell_pos[first:first + 9].tolist() == [0, 3, 3, 2, 3, 3, 2, 3, 3]
    g6 = build_grid(t, db, c, overrides=RhythmOverrides(subdiv=[(0, 6)]))
    first = int(np.searchsorted(g6.cell_t, t[0] - 1e-9))
    assert g6.cell_pos[first:first + 6].tolist() == [0, 4, 3, 4, 3, 4]


def test_overrides():
    t, db, c = synth_beats(65, 64, 4)
    base = build_grid(t, db, c)
    assert not base.edited
    g = build_grid(t, db, c, overrides=RhythmOverrides(tempo_factor=2.0))
    assert g.edited and g.bpm == pytest.approx(130, abs=0.5) and len(g.beats) == 2 * 64 - 1
    t2, db2, c2 = synth_beats(130, 96, 4)
    g = build_grid(t2, db2, c2, overrides=RhythmOverrides(tempo_factor=0.5))
    assert g.bpm == pytest.approx(65, abs=0.5) and np.all(bar_lengths(g)[:-1] == 4)
    g = build_grid(t, db, c, overrides=RhythmOverrides(downbeat_shift=1))
    assert g.bars[0][0] == 1 and g.bar_of_beat[0] == -1
    g = build_grid(t, db, c, overrides=RhythmOverrides(meter=[(2, 3, 4)]))
    assert g.bars[:4] == [(0, 4), (4, 4), (8, 3), (11, 3)] and (2, 3, 4) in g.meter
    g = build_grid(t, db, c, overrides=RhythmOverrides(meter=[(1, 6, 8)]))
    assert g.bars[1:3] == [(4, 2), (6, 2)] and (1, 6, 8) in g.meter
    g = build_grid(t, db, c, overrides=RhythmOverrides(subdiv=[(1, 3)]))
    assert np.all(g.subdiv[:4] == 4) and np.all(g.subdiv[4:] == 3)
    assert_cells_consistent(g)
    g = build_grid(t, db, c, overrides=RhythmOverrides(free=[(t[10] + 0.1, t[20] - 0.1)]))
    assert any(a == pytest.approx(t[10]) and b == pytest.approx(t[20]) for a, b in g.free)
    assert np.all(g.beat_in_bar[10:20] == 0)
    assert_cells_consistent(g)
    tap = t[::2] + 0.01
    g = build_grid(t, db, c, overrides=RhythmOverrides(tapped_beats=list(tap)))
    assert np.allclose(g.beats, tap) and np.all(g.beat_conf == 1.0)
    g = build_grid(t, db, c, overrides=RhythmOverrides(moved_beats={5: float(t[5] + 0.05)}))
    assert g.beats[5] == pytest.approx(t[5] + 0.05) and g.edited


def test_no_beats_is_all_free():
    g = build_grid(np.array([]), np.array([]), np.array([]), duration=10.0)
    assert g.free == [(0.0, 10.0)] and np.all(g.cell_pos == 5) and g.cell_t[-1] == 10.0
    assert np.all(np.diff(g.cell_t) > 0.1)


# ------------------------------------------------------------------------------------------------
# triplet test, check_octave, HCDF
# ------------------------------------------------------------------------------------------------

def _impulse_env(beats, fracs, fps=86.13, dur=None):
    dur = dur or beats[-1] + 1
    env = np.zeros(int(dur * fps) + 1)
    ibi = np.diff(beats)
    for b, d in zip(beats[:-1], ibi):
        for f in fracs:
            env[int(round((b + f * d) * fps))] = 1.0
    return OnsetEnvelope(env, fps)


@pytest.mark.parametrize("bpm", [65, 100, 130])
def test_triplet_ratio(bpm):
    b = 1.0 + np.arange(17) * 60 / bpm
    idx = np.arange(16)
    assert triplet_ratio(_impulse_env(b, (0, 0.5)), b, idx) < 1.0
    assert triplet_ratio(_impulse_env(b, (0, 0.25, 0.5, 0.75)), b, idx) < 1.0
    assert triplet_ratio(_impulse_env(b, (0, 1 / 3, 2 / 3)), b, idx) > 1.2
    assert triplet_ratio(_impulse_env(b, (0, 2 / 3)), b, idx) > 1.2          # swung 8ths


def _main(times):
    segs = [Segment(c0=0, c1=0, t0=float(x), t1=float(x) + 0.5, cand=0, cls=0, layer="main") for x in times]
    return DecodeResult(segments=segs, post=np.zeros((0, 0)), post_cls=np.zeros((0, 0)), pb=np.zeros(0), score=0.0)


def test_check_octave_halves_and_merges_bars():
    t, db, c = synth_beats(130, 129, 4)
    g = build_grid(t, db, c, duration=t[-1] + 1.0)
    assert g.bpm == pytest.approx(130, abs=1)
    # chords change on real downbeats (every 8 eighths), sometimes on the real beat 3 (+4) or 2 (+2)
    ch = [t[0]] + [t[k] for k in range(8, 128, 8)] + [t[k] for k in range(12, 128, 16)] + [t[18], t[50]]
    new = check_octave(g, _main(ch))
    assert new is not None
    assert new.bpm == pytest.approx(65, abs=0.5)
    assert np.all(bar_lengths(new)[:-1] == 4)
    assert np.allclose(new.beats[[s for s, _ in new.bars]][:8], t[0::8][:8])
    assert_cells_consistent(new, t[-1] + 1.0)
    assert check_octave(new, _main(ch)) is None                          # 65 BPM: guarded


def test_check_octave_keeps_grid_when_changes_hit_odd_beats():
    t, db, c = synth_beats(130, 129, 4)
    g = build_grid(t, db, c)
    ch = [t[k] for k in range(0, 128, 4)] + [t[k] for k in range(1, 128, 8)] + [t[k] for k in range(3, 128, 16)]
    assert check_octave(g, _main(ch)) is None
    slow = build_grid(*synth_beats(100, 64, 4))
    assert check_octave(slow, _main(slow.beats[::2])) is None           # < 110 BPM: never halved
    edited = build_grid(t, db, c, overrides=RhythmOverrides(moved_beats={3: float(t[3])}))
    assert check_octave(edited, _main(t[::8])) is None


def test_hcdf_peaks_on_chord_changes():
    prog = [(48, [60, 64, 67]), (43, [59, 62, 67]), (45, [57, 60, 64]), (41, [57, 60, 65]),
            (50, [57, 62, 65]), (40, [56, 59, 64]), (45, [57, 60, 64]), (44, [56, 60, 63])]
    times, durs = [0.5], []
    for i in range(24):
        durs.append(0.46 if i % 6 == 5 else 1.38)
        times.append(times[-1] + durs[-1])
    y = np.zeros(int((times[-1] + 1) * SR))
    for i in range(24):
        root, tri = prog[i % len(prog)]
        _add(y, _tone(root - 12 if root > 47 else root, durs[i], 0.2), times[i])
        k = 0
        while k * 0.23 < durs[i] - 0.05:
            for m in tri:
                _add(y, _tone(m, 0.4, 0.05, "tri", 0.3), times[i] + k * 0.23)
            k += 1
    pk = hcdf.peaks(y, SR)
    ch = np.array(times[1:24])
    hits = [np.min(np.abs(pk - x)) <= 0.06 for x in ch]
    assert np.mean(hits) >= 0.7
    spurious = [np.min(np.abs(ch - p)) > 0.1 for p in pk if 0.7 < p < times[-1] - 0.2]
    assert np.mean(spurious) <= 0.2


# ------------------------------------------------------------------------------------------------
# GPU: Beat This! on synthetic grooves
# ------------------------------------------------------------------------------------------------

@gpu
def test_groove_65_bpm_4_4():
    mix, harm = groove(steady(65, 40), 4, [0, 0.25, 0.5, 0.75])
    g = from_audio(mix, harm)
    assert g.bpm == pytest.approx(65, abs=0.7)
    assert np.mean(bar_lengths(g)[:-1] == 4) >= 0.9
    assert median_bar_s(g) == pytest.approx(240 / 65, abs=0.05)
    assert metered_subdiv(g) == {4}
    assert g.meter == [(0, 4, 4)]
    assert not inner_free(g)
    assert_cells_consistent(g, len(mix) / SR)


@gpu
def test_groove_130_bpm_4_4():
    mix, harm = groove(steady(130, 64), 4, [0, 0.5])
    g = from_audio(mix, harm)
    assert g.bpm == pytest.approx(130, abs=1.5)
    assert np.mean(bar_lengths(g)[:-1] == 4) >= 0.9
    assert metered_subdiv(g) == {2}
    assert g.meter == [(0, 4, 4)]
    assert not inner_free(g)


@gpu
def test_groove_waltz_3_4():
    mix, harm = groove(steady(100, 48), 3, [0, 0.5])
    g = from_audio(mix, harm)
    assert g.bpm == pytest.approx(100, abs=1.5)
    assert np.mean(bar_lengths(g)[:-1] == 3) >= 0.9
    assert g.meter == [(0, 3, 4)]
    assert metered_subdiv(g) == {2}


@gpu
def test_groove_6_8():
    mix, harm = groove(steady(60, 32), 2, [0, 1 / 3, 2 / 3])
    g = from_audio(mix, harm)
    assert g.bpm == pytest.approx(60, abs=1.5)
    assert np.mean(bar_lengths(g)[:-1] == 2) >= 0.9
    assert g.meter == [(0, 6, 8)]
    assert metered_subdiv(g) == {6}


@gpu
def test_groove_swung_eighths():
    mix, harm = groove(steady(100, 48), 4, [0, 2 / 3])
    g = from_audio(mix, harm)
    assert g.bpm == pytest.approx(100, abs=1.5)
    assert metered_subdiv(g) == {3}
    assert g.meter == [(0, 4, 4)]


@gpu
def test_rubato_intro_is_free_time():
    """Rubato piano intro, then a 76 BPM groove. Beat This! sometimes locks onto the groove a few
    beats early when the last intro chords happen to fit its grid, so 6 of 8 seeds must end the free
    region within 1.25 beats of the groove, and every seed must cover at least half the intro."""
    beat = 60 / 76
    good = 0
    for seed in range(8):
        mix, harm, start = rubato_then_groove(seed=seed)
        g = from_audio(mix, harm)
        assert g.free[0][0] == 0.0 and g.free[0][1] >= 0.5 * start
        good += abs(g.free[0][1] - start) <= 1.25 * beat
        assert not [r for r in g.free[1:] if r[1] < g.beats[-1] - 0.1]
        assert g.bpm == pytest.approx(76, abs=1.5)
        assert metered_subdiv(g) == {4}
        assert_cells_consistent(g, len(mix) / SR)
    assert good >= 6


@gpu
def test_track_end_to_end_and_raw_cache(tmp_path, monkeypatch):
    mix, harm = groove(steady(65, 24), 4, [0, 0.5])
    paths = {}
    for name, y in [("audio", mix), ("harmonic", harm), ("bass", harm), ("other", harm), ("drums", mix), ("vocals", 0 * mix)]:
        paths[name] = tmp_path / f"{name}.wav"
        sf.write(paths[name], np.stack([y, y], axis=1), SR, subtype="PCM_16")
    asset = AudioAsset(key="test", dir=tmp_path, wav=paths["audio"], play=tmp_path / "play.m4a",
                       duration_s=len(mix) / SR, title="t", artist=None, source_type="upload", source_url=None)
    stems = Stems(mix=paths["audio"], vocals=paths["vocals"], drums=paths["drums"], bass=paths["bass"],
                  other=paths["other"], harmonic=paths["harmonic"], model="none")
    calls = []
    g = rhythm.track(asset, stems, progress=lambda s, p, m: calls.append((s, p)))
    assert g.bpm == pytest.approx(65, abs=0.7) and metered_subdiv(g) == {4}
    assert calls[0] == ("rhythm", 0.0) and calls[-1] == ("rhythm", 1.0)
    raw = json.loads((tmp_path / rhythm.RAW_CACHE).read_text(encoding="utf-8"))
    assert len(raw["beats"]) == len(raw["beat_logit"]) == len(raw["downbeat_logit"])

    def boom(*a, **k):
        raise AssertionError("the cached tracker output must be reused")
    monkeypatch.setattr(rb, "track_file", boom)
    g2 = rhythm.track(asset, stems, RhythmOverrides(tempo_factor=2.0))
    assert g2.bpm == pytest.approx(130, abs=1.5) and g2.edited


# ------------------------------------------------------------------------------------------------
# 回家 (real song; skipped when its cache is missing)
# ------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not (HUIJIA / "spike" / "rhythm.json").exists(), reason="回家 spike output not cached")
def test_huijia_spike_beats_fold_to_65_without_logits():
    d = json.loads((HUIJIA / "spike" / "rhythm.json").read_text(encoding="utf-8"))
    b = np.asarray(d["beats"])
    g = build_grid(b, np.asarray(d["downbeats"]), np.ones(len(b)), duration=246.93)
    assert g.bpm == pytest.approx(65.2, abs=1.0)
    assert median_bar_s(g) == pytest.approx(3.68, abs=0.08)
    assert np.mean(bar_lengths(g) == 4) >= 0.9
    starts = g.beats[[s for s, _ in g.bars]]
    assert np.min(np.abs(starts - 32.42)) < 0.05 and np.min(np.abs(starts - 36.12)) < 0.05


@gpu
@pytest.mark.skipif(not (HUIJIA / "audio.wav").exists() or not (HUIJIA / "stems" / "harmonic.flac").exists(),
                    reason="回家 audio not cached")
def test_huijia_beat_this_folds_to_65():
    out = rb.track_file(HUIJIA / "audio.wav")
    assert 60 / np.median(np.diff(out.beats)) > 110          # the tracker itself reports eighths
    y, sr = rb.load_mono(HUIJIA / "stems" / "harmonic.flac")
    g = build_grid(out.beats, out.downbeats, out.beat_conf, duration=len(y) / sr, beat_logit=out.beat_logit,
                   downbeat_logit=out.downbeat_logit, onset=rhythm.onset_envelope(y, sr))
    assert g.bpm == pytest.approx(65.2, abs=1.0)
    assert median_bar_s(g) == pytest.approx(3.68, abs=0.05)
    assert np.mean(bar_lengths(g) == 4) >= 0.95
    assert metered_subdiv(g) == {4}
    assert g.meter[0] == (0, 4, 4)
    starts = g.beats[[s for s, _ in g.bars]]
    assert np.min(np.abs(starts - 32.42)) < 0.05 and np.min(np.abs(starts - 36.12)) < 0.05
    assert_cells_consistent(g, len(y) / sr)


@pytest.mark.skipif(not (HUIJIA / "spike" / "b1_lv_crf.json").exists() or not (HUIJIA / "spike" / "rhythm.json").exists(),
                    reason="回家 spike output not cached")
def test_huijia_check_octave_fixes_an_unfolded_grid():
    """Without the octave rule the grid is 130 BPM with half bars; the lv-chordia changes fix it."""
    from nathan_chordify.rhythm import meter
    from nathan_chordify.rhythm.grid import grid_from_beats
    d = json.loads((HUIJIA / "spike" / "rhythm.json").read_text(encoding="utf-8"))
    segs = json.loads((HUIJIA / "spike" / "b1_lv_crf.json").read_text(encoding="utf-8"))
    seq = meter.BeatSeq.make(d["beats"], np.ones(len(d["beats"])), d["downbeats"])
    g = grid_from_beats(seq, meter.free_intervals(seq), 4, meter.bar_evidence(seq), duration=246.93)
    assert g.bpm > 120
    new = check_octave(g, _main([s["start_time"] for s in segs]))
    assert new is not None
    assert new.bpm == pytest.approx(65.2, abs=1.0)
    assert median_bar_s(new) == pytest.approx(3.68, abs=0.08)
    assert np.mean(bar_lengths(new) == 4) >= 0.9
    starts = new.beats[[s for s, _ in new.bars]]
    assert np.min(np.abs(starts - 6.6)) < 0.05 and np.min(np.abs(starts - 32.42)) < 0.05   # intro kept at 65
    assert_cells_consistent(new, 246.93)


# ------------------------------------------------------------------------------------------------
# D1 worship songs: the cached Beat This! output (cache/<key>/rhythm_raw.json); skipped when missing
# ------------------------------------------------------------------------------------------------

HUOCHU_AI = "yt-lfP-3JLVvaw"        # 活出愛 Live Out Love: 12/8, dotted quarter ≈ 63
HERE_I_BOW = "yt-hvHi7N8kc8s"       # Here I Bow: 4/4 ≈ 69, a held-chord intro of 4 bars


def _song_grid(key: str, with_audio: bool = False):
    d = config.CACHE_DIR / key
    raw_path, meta_path = d / rhythm.RAW_CACHE, d / "meta.json"
    if not raw_path.exists() or not meta_path.exists():
        pytest.skip(f"{key}: tracker output not cached")
    raw = rb.TrackerOutput.from_json(json.loads(raw_path.read_text(encoding="utf-8")))
    kw = {}
    if with_audio:
        harm = d / "stems" / "harmonic.flac"
        if not harm.exists():
            pytest.skip(f"{key}: harmonic stem not cached")
        y, sr = rb.load_mono(harm)
        kw = dict(onset=rhythm.onset_envelope(y, sr), harmonic_changes=hcdf.peaks(y, sr))
    duration = float(json.loads(meta_path.read_text(encoding="utf-8"))["duration_s"])
    return build_grid(raw.beats, raw.downbeats, raw.beat_conf, duration=duration, beat_logit=raw.beat_logit,
                      downbeat_logit=raw.downbeat_logit, **kw), duration


def test_huochu_ai_is_12_8_at_dotted_quarters():
    """活出愛: the tracker follows eighths for 30 s, then dotted quarters with logits 2–5. The grid
    used to flip eighths ↔ dotted quarters, open ~10 free-time gaps and change bar lengths
    (2/4 ↔ 4/4 ↔ 3/4); now one tempo octave, compound bars, metered through the scored song."""
    g, duration = _song_grid(HUOCHU_AI)
    assert g.bpm == pytest.approx(63.0, abs=1.0)
    ibi = np.diff(g.beats)
    assert np.mean(np.abs(ibi / (60 / 63.0) - 1) < 0.12) >= 0.98             # dotted quarters only
    assert all(den == 8 for _, _, den in g.meter) and len(g.meter) <= 3
    L = np.bincount(bar_lengths(g)).argmax()
    assert L in (2, 4) and np.mean(bar_lengths(g) == L) >= 0.97
    assert not [r for r in g.free if r[1] > 2.0 and r[0] < 258.0]           # metered until the tag
    starts = g.beats[[s for s, _ in g.bars]]
    for x in (1.14, 31.64, 62.08, 96.37, 126.87, 157.34, 164.96):           # section starts of the chart
        assert np.min(np.abs(starts - x)) < 0.1
    assert_cells_consistent(g, duration)


def test_here_i_bow_intro_inherits_the_song_tempo():
    """Here I Bow: the 14 s intro (one held D) was free time with stray beats; it is 4 bars at the
    song's tempo, backfilled from the groove, and the grid is metered from the first sound."""
    g, duration = _song_grid(HERE_I_BOW, with_audio=True)
    assert g.bpm == pytest.approx(69.0, abs=1.0)
    assert g.beats[0] < 0.6 and g.free[0] == (0.0, pytest.approx(g.beats[0]))
    assert not [r for r in g.free if r[1] > 0.6 and r[0] < 230.0]
    assert np.all(np.abs(np.diff(g.beats[:20]) / 0.868 - 1) < 0.1)          # no half-tempo intro
    starts = g.beats[[s for s, _ in g.bars]]
    for x in (0.54, 14.42, 42.25, 73.56, 101.38, 132.69, 160.51):          # intro + chart section starts
        assert np.min(np.abs(starts - x)) < 0.1
    assert np.all(bar_lengths(g)[:-1] == 4) and g.meter == [(0, 4, 4)]
    assert_cells_consistent(g, duration)


@pytest.mark.parametrize("key, bpm", [("yt-Y7boUFbM76k", 65.1), ("yt-XtwIT8JjddM", 72.6), ("yt-WLtqi4s29LQ", 66.0)])
def test_simple_meter_songs_keep_their_grid(key, bpm):
    """回家, 10,000 Reasons, 立志擺上 stay 4/4 at the same tempo (no compound, no octave change)."""
    g, duration = _song_grid(key)
    assert g.bpm == pytest.approx(bpm, abs=0.7)
    assert g.meter[0] == (0, 4, 4) and all(den == 4 for _, _, den in g.meter)
    assert np.mean(bar_lengths(g)[:-1] == 4) >= 0.95            # (10,000 Reasons has real 2/4 and 6/4 bars)
    assert len(inner_free(g)) <= 1
    assert_cells_consistent(g, duration)
