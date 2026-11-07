"""WP6 key tests (DESIGN §5.4): profiles, local-key HMM, chord-based refinement, truck-driver test."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from nathan_chordify.theory import key as K
from nathan_chordify.types import BassTrack, RhythmGrid, Segment, Stems

B_MAJ, C_MAJ, CS_MAJ, E_MAJ, E_MIN = 11, 0, 1, 4, 16
IBI = 0.92


def make_grid(n_beats: int, ibi: float = IBI, bpb: int = 4, t0: float = 1.0) -> RhythmGrid:
    b = np.arange(n_beats)
    beats = t0 + b * ibi
    cells = t0 + np.arange(4 * n_beats + 1) * ibi / 4
    return RhythmGrid(beats=beats, beat_conf=np.ones(n_beats), bar_of_beat=b // bpb, beat_in_bar=b % bpb + 1,
                      bars=[(i, bpb) for i in range(0, n_beats, bpb)], meter=[(0, bpb, 4)], bpm=60 / ibi,
                      ibi_ref=ibi, subdiv=np.full(n_beats, 4), free=[], cell_t=cells,
                      cell_beat=np.arange(4 * n_beats + 1) / 4, cell_pos=np.zeros(4 * n_beats, np.int8))


TRIAD = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10)}


def chord_chroma(root: int, quality: str, rng, melody_scale: tuple[int, ...] | None = None) -> np.ndarray:
    v = np.full(12, 0.05) + 0.05 * rng.random(12)
    for i in TRIAD[quality]:
        v[(root + i) % 12] += 1.0
    if melody_scale is not None:
        v[rng.choice(melody_scale)] += 0.6
    return v / v.max()


MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)


def progression_chroma(key_tonic: int, n_bars: int, rng, prog=((0, "maj"), (7, "maj"), (9, "min"), (5, "maj"))):
    """I–V–vi–IV (one chord per 4-beat bar) with melody notes from the scale."""
    scale = tuple((key_tonic + s) % 12 for s in MAJOR_SCALE)
    rows = []
    for bar in range(n_bars):
        deg, q = prog[bar % len(prog)]
        for _ in range(4):
            rows.append(chord_chroma(key_tonic + deg, q, rng, scale))
    return np.array(rows)


def test_transition_matrix_rows_and_favoured_moves():
    A = K.transition_matrix()
    assert np.allclose(A.sum(axis=1), 1.0)
    assert np.allclose(np.diag(A), K.P_STAY)
    row = A[B_MAJ]
    assert np.isclose(row[CS_MAJ], 2 * row[K.make_key(B_MAJ + 3, False)])     # +2 same mode favoured
    assert np.isclose(row[C_MAJ], row[K.make_key(B_MAJ + 7, False)])          # +1 and V both favoured
    assert np.isclose(row[K.make_key(8, True)], row[K.make_key(11, True)])    # relative and parallel


def test_b_major_global_and_local():
    rng = np.random.default_rng(0)
    bc = progression_chroma(11, 48, rng)
    grid = make_grid(len(bc))
    kt = K.estimate_from_chroma(bc, grid)
    assert (kt.global_tonic, kt.global_mode) == (11, "major")
    assert kt.global_conf > 0.5
    assert np.all(kt.per_beat == B_MAJ)
    assert kt.changes == []


def test_bass_histogram_term_prefers_the_tonic():
    """An ambiguous chroma (B major vs G# minor) is decided by a bass that sits on B."""
    rng = np.random.default_rng(1)
    bc = progression_chroma(11, 16, rng, prog=((0, "maj"), (9, "min")))
    n = 1000
    sal = np.full((n, 12), 0.01)
    sal[: n // 2, 11] = 1.0
    sal[n // 2:, 4] = 0.6
    sal /= sal.sum(axis=1, keepdims=True)
    bass = BassTrack(frame_t=np.arange(n) * 0.0116, salience=sal, energy_db=np.full(n, -20.0), notes=[])
    hist = K.bass_histogram(bass)
    assert hist.argmax() == 11
    s = K.global_scores(bc.sum(axis=0), hist)
    assert s[B_MAJ] > s[K.make_key(8, True)]


def test_minor_key_with_harmonic_dominant():
    rng = np.random.default_rng(2)
    prog = ((0, "min"), (5, "min"), (7, "7"), (0, "min"))
    scale = tuple((4 + s) % 12 for s in (0, 2, 3, 5, 7, 8, 11))
    rows = [chord_chroma(4 + d, q, rng, scale) for bar in range(32) for d, q in [prog[bar % 4]] for _ in range(4)]
    kt = K.estimate_from_chroma(np.array(rows), make_grid(len(rows)))
    assert (kt.global_tonic, kt.global_mode) == (4, "minor")


def test_step_up_modulation_is_found():
    rng = np.random.default_rng(3)
    bc = np.vstack([progression_chroma(11, 40, rng), progression_chroma(0, 24, rng)])
    grid = make_grid(len(bc))
    kt = K.estimate_from_chroma(bc, grid)
    assert (kt.global_tonic, kt.global_mode) == (11, "major")
    assert len(kt.changes) == 1
    ch = kt.changes[0]
    assert (ch.from_key, ch.to_key, ch.kind) == (B_MAJ, C_MAJ, "step_up")
    assert abs(ch.t - grid.beats[160]) <= 4 * IBI
    assert abs(ch.bar - 40) <= 1
    assert np.all(kt.per_beat[:150] == B_MAJ) and np.all(kt.per_beat[170:] == C_MAJ)


def test_short_key_excursions_are_merged():
    rng = np.random.default_rng(4)
    bc = np.vstack([progression_chroma(11, 20, rng), progression_chroma(4, 2, rng), progression_chroma(11, 20, rng)])
    kt = K.estimate_from_chroma(bc, make_grid(len(bc)))
    assert all(r[2] - r[1] >= K.MIN_SEG_BEATS for r in K.runs(kt.per_beat))
    assert np.all(kt.per_beat == B_MAJ)


def _tone(midi: float, dur: float, sr: int) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    f = 440 * 2 ** ((midi - 69) / 12)
    y = sum(0.6 ** (h - 1) * np.sin(2 * np.pi * f * h * t) for h in range(1, 7) if f * h < sr / 2)
    return (y * np.exp(-t / 1.5)).astype(np.float32)


def test_estimate_from_audio(tmp_path):
    """End to end on a rendered B-major progression: chroma_cqt → beat chroma → HMM."""
    sr, bar = 22050, 4 * IBI
    voicings = [(59, [63, 66, 71]), (54, [61, 66, 70]), (56, [63, 68, 71]), (52, [64, 68, 71])]  # B F# G#m E
    n_bars = 16
    y = np.zeros(int((n_bars * bar + 2) * sr), np.float32)
    for i in range(n_bars):
        b, upper = voicings[i % 4]
        s = int((1.0 + i * bar) * sr)
        for m in [b - 12] + upper:
            x = _tone(m, bar, sr)
            y[s:s + x.size] += 0.15 * x
    path = tmp_path / "harmonic.flac"
    sf.write(str(path), y, sr, subtype="PCM_16")
    stems = Stems(mix=path, vocals=path, drums=path, bass=path, other=path, harmonic=path, model="none")
    grid = make_grid(4 * n_bars)
    kt = K.estimate(stems, grid, None)
    assert (kt.global_tonic, kt.global_mode) == (11, "major")
    assert len(kt.per_beat) == 4 * n_bars


# --- refine (steps 3–4) -------------------------------------------------------------------------

def _cand(root: int, quality: str, bass: int | None = None) -> int:
    from nathan_chordify.theory import vocab
    bass = root if bass is None else bass
    for i, c in enumerate(vocab.candidates().cands):
        if (c.root, c.quality, c.bass) == (root % 12, quality, bass % 12):
            return i
    raise KeyError((root, quality, bass))


def _segments(chords: list[tuple[int, str]], grid: RhythmGrid, beats_per_chord: int = 4) -> list[Segment]:
    from nathan_chordify.theory import vocab
    cands = vocab.candidates().cands
    out = []
    for i, (r, q) in enumerate(chords):
        b0, b1 = i * beats_per_chord, (i + 1) * beats_per_chord
        t1 = grid.beats[b1] if b1 < len(grid.beats) else grid.beats[-1] + grid.ibi_ref
        ci = _cand(r, q)
        out.append(Segment(c0=4 * b0, c1=4 * b1, t0=float(grid.beats[b0]), t1=float(t1), cand=ci,
                           cls=cands[ci].cls, layer="main", conf=0.9))
    return out


def _flat_key(n: int, k: int) -> K.KeyTrack:
    return K.KeyTrack(global_tonic=k % 12, global_mode="minor" if k >= 12 else "major", global_conf=0.8,
                      per_beat=np.full(n, k, dtype=np.int64), changes=[])


def test_refine_chord_key_beats_the_profile_key():
    """The profile says E (IV); the chords (B, F#, G#m, E with F# major) say B."""
    prog = [(11, "maj"), (6, "maj"), (8, "min"), (4, "maj")] * 8
    grid = make_grid(4 * len(prog))
    kt = K.refine(_flat_key(len(grid.beats), E_MAJ), _segments(prog, grid), grid, None)
    assert np.all(kt.per_beat == B_MAJ)
    assert (kt.global_tonic, kt.global_mode) == (11, "major")


def test_refine_keeps_relative_key_ties():
    """C major and A minor fit C–F–G–Am equally: the profile key stays."""
    prog = [(0, "maj"), (5, "maj"), (7, "maj"), (9, "min")] * 8
    grid = make_grid(4 * len(prog))
    kt = K.refine(_flat_key(len(grid.beats), C_MAJ), _segments(prog, grid), grid, None)
    assert np.all(kt.per_beat == C_MAJ) and kt.changes == []


def test_truck_driver_step_up_at_section_boundary():
    """Verse + 3 choruses in B, last chorus up a whole step (C#); the HMM missed it."""
    verse = [(11, "maj"), (4, "maj"), (8, "min"), (6, "maj")] * 2
    chorus = [(4, "maj"), (6, "maj"), (8, "min"), (11, "maj"), (4, "maj"), (6, "maj"), (11, "maj"), (11, "maj")]
    up = [((r + 2) % 12, q) for r, q in chorus]
    prog = verse + chorus + chorus + chorus + up
    grid = make_grid(4 * len(prog))
    kt = K.refine(_flat_key(len(grid.beats), B_MAJ), _segments(prog, grid), grid, None)
    a = 4 * 8 * 4                           # first beat of the last 8-bar block
    assert np.all(kt.per_beat[:a] == B_MAJ) and np.all(kt.per_beat[a:] == CS_MAJ)
    assert len(kt.changes) == 1
    ch = kt.changes[0]
    assert (ch.from_key, ch.to_key, ch.kind, ch.bar) == (B_MAJ, CS_MAJ, "step_up", 32)
    assert (kt.global_tonic, kt.global_mode) == (11, "major")


def test_truck_driver_uses_given_sections_and_snaps_an_early_change():
    chorus = [(4, "maj"), (6, "maj"), (8, "min"), (11, "maj")] * 2
    up = [((r + 1) % 12, q) for r, q in chorus]
    prog = chorus * 3 + up                    # 4 sections of 8 bars, step-up by a semitone
    grid = make_grid(4 * len(prog))
    a = 3 * 32
    per_beat = np.full(len(grid.beats), B_MAJ, dtype=np.int64)
    per_beat[a - 6:] = C_MAJ                  # the HMM switched 6 beats early
    key = K.KeyTrack(11, "major", 0.8, per_beat, [])
    sections = [{"label": "Chorus", "start": float(grid.beats[i * 32]), "end": 0.0} for i in range(4)]
    kt = K.refine(key, _segments(prog, grid), grid, sections)
    assert np.all(kt.per_beat[:a] == B_MAJ) and np.all(kt.per_beat[a:] == C_MAJ)
    assert [c.kind for c in kt.changes] == ["step_up"]


def test_no_step_up_for_a_plain_repeat():
    chorus = [(4, "maj"), (6, "maj"), (8, "min"), (11, "maj")] * 2
    prog = chorus * 4
    grid = make_grid(4 * len(prog))
    kt = K.refine(_flat_key(len(grid.beats), B_MAJ), _segments(prog, grid), grid, None)
    assert kt.changes == [] and np.all(kt.per_beat == B_MAJ)


# --- worship-style dwelling on IV / vi / ii: the home key must hold (eval/reports/M2.md §7.3) -----

G_SM, C_MAJ_, G_MAJ, A_MIN = 20, 0, 7, 21


def bars_chroma(key_tonic: int, bars, rng, scale=MAJOR_SCALE) -> np.ndarray:
    """One 4-beat bar per entry: a (degree, quality) chord, or a list of them splitting the bar."""
    sc = tuple((key_tonic + s) % 12 for s in scale)
    rows = []
    for bar in bars:
        parts = bar if isinstance(bar, list) else [bar]
        for i in range(4):
            deg, q = parts[i * len(parts) // 4]
            rows.append(chord_chroma(key_tonic + deg, q, rng, sc))
    return np.array(rows)


POP = [(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")]                       # I–V–vi–IV
DWELL = {
    "vi with V/vi": [[(9, "min"), (4, "7")], (9, "min"), [(9, "min"), (4, "7")], (9, "min")] * 2,
    "IV with borrowed bVII and iv": [(5, "maj"), [(5, "maj"), (10, "7")], (5, "maj"), [(5, "maj"), (5, "min")]] * 2,
    "ii": [(2, "min"), (2, "min"), [(2, "min"), (5, "maj")], (2, "min")] * 2,
}


def test_dwelling_on_iv_vi_ii_keeps_the_home_key():
    """8 bars (32 beats) on vi, IV (with A7 / Em) or ii inside B major: no local key change."""
    for name, dwell in DWELL.items():
        rng = np.random.default_rng(10)
        bc = bars_chroma(11, POP * 4 + dwell + POP * 4, rng)
        kt = K.estimate_from_chroma(bc, make_grid(len(bc)))
        assert (kt.global_tonic, kt.global_mode) == (11, "major"), name
        assert np.all(kt.per_beat == B_MAJ) and kt.changes == [], (name, K.runs(kt.per_beat))


def test_genuine_modulation_is_found():
    """B major for 24 bars, then G major (a chromatic-mediant move, not a step-up) for 24 bars."""
    rng = np.random.default_rng(11)
    bc = np.vstack([bars_chroma(11, POP * 6, rng), bars_chroma(7, POP * 6, rng)])
    grid = make_grid(len(bc))
    kt = K.estimate_from_chroma(bc, grid)
    assert len(kt.changes) == 1
    ch = kt.changes[0]
    assert (ch.from_key, ch.to_key, ch.kind) == (B_MAJ, G_MAJ, "modulation")
    assert abs(ch.bar - 24) <= 1
    assert np.all(kt.per_beat[:88] == B_MAJ) and np.all(kt.per_beat[104:] == G_MAJ)


def test_refine_overrides_profile_drift_to_vi_and_iv():
    """The 回家 failure: the profile path reads the intro as G#m and the pre-chorus as E; the
    chords (D#7→G#m tonicisation, A7 / Em borrowed) are B major throughout."""
    intro = [(4, "maj"), (3, "7"), (8, "min"), (4, "maj"), (3, "7"), (8, "min"), (4, "maj"), (6, "maj")]
    pre = [(11, "maj"), (4, "maj"), (9, "7"), (4, "min"), (11, "maj"), (4, "maj"), (9, "7"), (4, "min"),
           (1, "min"), (3, "min"), (4, "maj"), (6, "maj")]
    chorus = [(8, "min"), (6, "maj"), (4, "maj"), (6, "maj"), (11, "maj"), (4, "maj"), (6, "maj"), (11, "maj")]
    prog = intro + pre * 2 + chorus * 3 + [(11, "maj"), (11, "7"), (9, "maj"), (1, "min"), (4, "maj"), (11, "maj")]
    grid = make_grid(4 * len(prog))
    per_beat = np.full(len(grid.beats), B_MAJ, dtype=np.int64)
    per_beat[:32] = G_SM
    per_beat[32:128] = E_MAJ
    per_beat[-24:] = K.make_key(1, True)
    kt = K.refine(K.KeyTrack(11, "major", 0.5, per_beat, []), _segments(prog, grid), grid, None)
    assert np.all(kt.per_beat == B_MAJ) and kt.changes == [], K.runs(kt.per_beat)
    assert (kt.global_tonic, kt.global_mode) == (11, "major")


def test_refine_relative_minor_needs_sustained_cadences():
    """A 4-bar V/vi → vi tonicisation stays in B; 16 bars of G#m cadences (i–iv–V7–i) are a
    genuine move to the relative minor."""
    verse = [(11, "maj"), (6, "maj"), (8, "min"), (4, "maj")] * 4
    tonicise = [(3, "7"), (8, "min"), (3, "7"), (8, "min")]
    grid = make_grid(4 * (2 * len(verse) + len(tonicise)))
    prog = verse + tonicise + verse
    kt = K.refine(_flat_key(len(grid.beats), B_MAJ), _segments(prog, grid), grid, None)
    assert np.all(kt.per_beat == B_MAJ) and kt.changes == []

    minor = [(8, "min"), (1, "min"), (3, "7"), (8, "min")] * 4
    prog = verse + minor + verse
    grid = make_grid(4 * len(prog))
    kt = K.refine(_flat_key(len(grid.beats), B_MAJ), _segments(prog, grid), grid, None)
    a, b = 4 * len(verse), 4 * (len(verse) + len(minor))
    assert np.all(kt.per_beat[a + 4:b - 4] == G_SM), K.runs(kt.per_beat)
    assert np.all(kt.per_beat[:a - 4] == B_MAJ) and np.all(kt.per_beat[b + 4:] == B_MAJ)
    assert [c.kind for c in kt.changes] == ["relative", "relative"]


def test_refine_finds_a_genuine_modulation():
    """The profile missed it; 16 bars of G-major chords after 16 bars of B major are a modulation."""
    prog = [(11, "maj"), (6, "maj"), (8, "min"), (4, "maj")] * 4 + [(7, "maj"), (2, "maj"), (4, "min"), (0, "maj")] * 4
    grid = make_grid(4 * len(prog))
    kt = K.refine(_flat_key(len(grid.beats), B_MAJ), _segments(prog, grid), grid, None)
    assert np.all(kt.per_beat[:60] == B_MAJ) and np.all(kt.per_beat[68:] == G_MAJ), K.runs(kt.per_beat)
    assert [(c.from_key, c.to_key, c.kind) for c in kt.changes] == [(B_MAJ, G_MAJ, "modulation")]


def test_refine_minor_key_with_dominant_seventh():
    """E minor (i–iv–V7–i, VI–VII–V7–i): stays E minor rather than drifting to G major."""
    prog = [(4, "min"), (9, "min"), (11, "7"), (4, "min"), (0, "maj"), (2, "maj"), (11, "7"), (4, "min")] * 6
    grid = make_grid(4 * len(prog))
    kt = K.refine(_flat_key(len(grid.beats), E_MIN), _segments(prog, grid), grid, None)
    assert np.all(kt.per_beat == E_MIN) and kt.changes == []
    assert (kt.global_tonic, kt.global_mode) == (4, "minor")


def test_refine_relative_key_loop_follows_the_profile():
    """Am–F–C–G fits A minor and C major almost equally: the profile key (A minor) decides."""
    prog = [(9, "min"), (5, "maj"), (0, "maj"), (7, "maj")] * 16
    grid = make_grid(4 * len(prog))
    kt = K.refine(_flat_key(len(grid.beats), A_MIN), _segments(prog, grid), grid, None)
    assert np.all(kt.per_beat == A_MIN) and kt.changes == []


def test_close_keys_are_left_to_the_chords():
    """The profile HMM damps moves to keys one accidental away (IV, V, ii, iii, vi); step-ups
    (+1 = 5 accidentals, +2 = 2) and distant keys keep the design's transition weights."""
    assert K.close_keys(B_MAJ) == {B_MAJ, G_SM, E_MAJ, K.make_key(1, True), K.make_key(6, False), K.make_key(3, True)}
    assert K.close_keys(E_MIN) == {E_MIN, G_MAJ, K.make_key(0, False), A_MIN, K.make_key(2, False), K.make_key(11, True)}
    L, base = K.profile_log_transitions(), np.log(K.transition_matrix())
    for k in (C_MAJ, CS_MAJ, G_MAJ):
        assert L[B_MAJ, k] == base[B_MAJ, k]
    assert np.isclose(L[B_MAJ, E_MAJ], base[B_MAJ, E_MAJ] - K.CLOSE_PENALTY)
    assert L[B_MAJ, E_MAJ] < L[B_MAJ, CS_MAJ] - 10


def test_whole_step_up_is_found():
    rng = np.random.default_rng(12)
    bc = np.vstack([bars_chroma(11, POP * 10, rng), bars_chroma(1, POP * 6, rng)])
    kt = K.estimate_from_chroma(bc, make_grid(len(bc)))
    assert [(c.from_key, c.to_key, c.kind) for c in kt.changes] == [(B_MAJ, CS_MAJ, "step_up")]
    assert abs(kt.changes[0].bar - 40) <= 1
