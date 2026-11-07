"""train/song_mix.py: progressions, meters, labels, features and the pool format."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train import chords as C  # noqa: E402
from train import piano as P  # noqa: E402
from train import song_mix as S  # noqa: E402

needs_salamander = pytest.mark.skipif(not P.available(), reason="Salamander piano not downloaded")


def _beats(chords) -> float:
    return sum(c.beats for c in chords)


def test_spec_of_maps_symbols_and_simplifies():
    assert S.spec_of("G/B").label == C.chord("G/B").label
    assert S.spec_of("Cmaj7").pcs == frozenset({0, 4, 7, 11})
    assert S.spec_of("N") is None and S.spec_of("X") is None
    # an out-of-vocabulary chord falls back to something in the vocabulary (or None), never raises
    out = S.spec_of("C:min7(11)")
    assert out is None or out.root == 0


def test_refit_meter_keeps_bars_and_order():
    ch = [S.PChord(C.chord("C"), 2.0), S.PChord(C.chord("G"), 2.0), S.PChord(C.chord("Am"), 4.0)]
    out = S._refit_meter(ch, 4, 3)
    assert _beats(out) == pytest.approx(6.0)                 # two 4/4 bars → two 3/4 bars
    assert [c.spec.label for c in out] == ["C:maj", "G:maj", "A:min"]
    assert S._refit_meter(ch, 4, 4) is ch


def test_loop_to_fills_exactly_and_starts_mid_progression():
    ch = [S.PChord(C.chord("C"), 4.0), S.PChord(C.chord("F"), 4.0)]
    out = S._loop_to(ch, 10.0, start=4.0)
    assert _beats(out) == pytest.approx(10.0)
    assert out[0].spec.label == "F:maj"


@pytest.mark.parametrize("bpb", [3, 4])
def test_diatonic_progression_is_in_vocab_and_bar_aligned(bpb):
    rng = np.random.default_rng(3)
    for _ in range(20):
        prog = S.diatonic_progression(rng, int(rng.integers(12)), bpb, 6)
        assert _beats(prog) == pytest.approx(6 * bpb)
        assert all(c.beats > 0 for c in prog)
        assert all(c.spec.index >= 0 for c in prog)


def test_chart_parsing_and_meter_header(tmp_path):
    text = "# Meter 3/4\n[Verse]\n| C . G | Am | F . . |\n| G/B . C |\n"
    bpb = S._chart_meter(text)
    assert bpb == 3
    chords = S._parse_chart_text(text, bpb)
    assert _beats(chords) == pytest.approx(12.0)
    assert chords[0].spec.label == "C:maj" and chords[0].beats == 2.0
    assert S._chart_meter("# Time: 12/8\n| C |") == 4
    assert S._chart_meter("| C |") == 4


def test_plan_song_meters_tempi_and_timeline():
    rng = np.random.default_rng(11)
    meters = set()
    for _ in range(30):
        plan = S.plan_song(rng, charts_ok=False)
        meters.add(plan.meter)
        if plan.meter == "12/8":
            assert 40 <= plan.bpm <= 75
        else:
            assert 55 <= plan.bpm <= 140
        assert 15 <= plan.info["duration"] <= 30
        t = [s.t0 for s in plan.segs] + [plan.segs[-1].t1]
        assert np.all(np.diff(t) > 0)
        assert plan.segs[0].t0 == pytest.approx(plan.lead)
        assert plan.sections[0][0] == 0.0
    assert meters == {"4/4", "3/4", "12/8"}


def test_frame_labels_follow_segments():
    a, b = C.chord("C"), C.chord("G/B")
    segs = [S.Seg(0, 4, 1.0, 3.0, a, False), S.Seg(4, 5, 3.0, 3.5, b, True)]
    plan = S.SongPlan("4/4", 120.0, 1.0, segs, 3.8, [(0.0, {})], {"duration": 20.0})
    F = S.WIN_FRAMES
    root, bass, notes = S.frame_labels(plan, F)
    t = S.frame_times(F)
    assert np.all(root[t < 1.0] == 12) and np.all(bass[t < 1.0] == 12) and np.all(notes[t < 1.0] == 0)
    m = (t >= 1.0) & (t < 3.0)
    assert np.all(root[m] == 0) and np.all(bass[m] == 0)
    assert set(np.flatnonzero(S.notes_matrix(notes[m][:1])[0])) == {0, 4, 7}
    m2 = (t >= 3.0) & (t < 3.8)
    assert np.all(root[m2] == 7) and np.all(bass[m2] == 11)
    assert np.all(root[t >= 3.8] == 12)


def test_features_shape_and_peak_envelope():
    sr = S.SR
    y = np.zeros(int(5 * sr), np.float32)
    y[sr:2 * sr] = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
    cqt, env = S.features(y)
    assert cqt.shape == (S.N_BINS, S.WIN_FRAMES)            # padded to ACE's 20 s window
    assert env.shape == (S.WIN_FRAMES,)
    assert env.max() == pytest.approx(0.5, rel=1e-3)
    t = S.frame_times(S.WIN_FRAMES)
    assert env[t < 0.9].max() == 0.0 and env[t > 2.1].max() == 0.0
    # A4 (440 Hz, MIDI 69) = bin 2·(69 − 24) = 90 from C1 (MIDI 24)
    assert int(cqt[:, (t > 1.2) & (t < 1.8)].mean(axis=1).argmax()) == 90


def test_crop_normalisation_matches_ace_window():
    """Dividing the stored CQT by the crop's peak = ACE's per-window peak normalisation."""
    from nathan_chordify.models import ace
    rng = np.random.default_rng(0)
    y = (0.3 * rng.standard_normal(int(22 * S.SR))).astype(np.float32)
    cqt, env = S.features(y)
    a = 30
    from train import finetune_ace as FT
    item = {"cqt": cqt.astype(np.float16), "env": env.astype(np.float16), "root": np.zeros(cqt.shape[1], np.int8),
            "bass": np.zeros(cqt.shape[1], np.int8), "notes": np.zeros(cqt.shape[1], np.uint16)}
    x, *_ = FT.crop(item, a)
    ref = ace.Ace().features(y, a, S.WIN_SAMPLES)
    mid = slice(60, S.WIN_FRAMES - 60)                       # away from the window edges (CQT support)
    rel = np.abs(x[:, mid] - ref[:, mid]).mean() / ref[:, mid].mean()
    assert rel < 0.02


@needs_salamander
def test_make_item_and_shard_roundtrip(tmp_path):
    it = S.make_item(0, 99, "val", S.get_piano(), charts_ok=bool(S.chart_files("val")))
    F = it["cqt"].shape[1]
    assert F >= S.WIN_FRAMES and it["cqt"].shape[0] == S.N_BINS
    assert len(it["root"]) == len(it["bass"]) == len(it["notes"]) == len(it["env"]) == F
    assert (it["root"] < 12).mean() > 0.5
    chord_frames = it["root"] < 12
    assert np.all(it["notes"][chord_frames] > 0)
    # the bass pc is always one of the chord's notes
    nm = S.notes_matrix(it["notes"][chord_frames])
    assert np.all(nm[np.arange(len(nm)), it["bass"][chord_frames]] == 1)
    r = S.write_shard("val", 99, 0, [0], out_dir=tmp_path)
    z = S.load_shard(Path(r["path"]))
    assert z["offsets"].tolist() == [0, F]
    assert z["cqt"].dtype == np.float16 and z["meta"][0]["id"] == "val-99-0"
    np.testing.assert_array_equal(z["root"], it["root"])
    assert S.write_shard("val", 99, 0, [0], out_dir=tmp_path)["skipped"]
    st = S.stats(tmp_path)
    assert st["val"]["items"] == 1 and st["val"]["chords"] == len(it["meta"]["chords"])
    json.dumps(st)


def test_rerhythm_keeps_order_and_bar_alignment():
    seq = [S.PChord(C.chord(x), 4.0) for x in ("C", "G/B", "Am", "F", "C/E", "Dm7", "G")]
    for bpb in (3, 4):
        rng = np.random.default_rng(bpb)
        for _ in range(20):
            out = S.rerhythm(seq, bpb, rng)
            labels = [c.spec.label for c in out]
            assert labels == [c.spec.label for c in seq]          # chord order is exact
            assert all(1.0 <= c.beats <= 2 * bpb for c in out)
            assert all(c.passing == (c.beats <= 1.0) for c in out)


def test_chart_split_keeps_alternative_versions_with_their_song(tmp_path, monkeypatch):
    (tmp_path / "alt").mkdir()
    for k in range(10):
        (tmp_path / f"song{k}.chart.txt").write_text("| C | G |\n", "utf-8")
    (tmp_path / "alt" / "song0.other.chart.txt").write_text("| C | F |\n", "utf-8")
    (tmp_path / "alt" / "song3.other.chart.txt").write_text("| C | F |\n", "utf-8")
    monkeypatch.setattr(S, "CHART_DIR", tmp_path)
    val = {f.name for f in S.chart_files("val")}
    train = {f.name for f in S.chart_files("train")}
    want = {f"song{k}.chart.txt" for k in range(10) if S.is_val_slug(f"song{k}")}
    want |= {f"song{k}.other.chart.txt" for k in (0, 3) if S.is_val_slug(f"song{k}")}
    assert val == want
    assert not val & train and len(val) + len(train) == 12


def test_piano_focus_changes_only_the_arrangement():
    seen = set()
    for i in range(30):
        a = S.plan_song(np.random.default_rng([5, i]), "train", False, True, np.random.default_rng([5, i, 7]))
        b = S.plan_song(np.random.default_rng([5, i]), "train", False)
        assert [s.spec for s in a.segs] == [s.spec for s in b.segs] and a.bpm == b.bpm
        arr = a.info["arrangement"]
        seen.add(arr)
        if arr == "solo":
            assert all(sec[k] is None for _, sec in a.sections for k in S.LAYERS)
        elif arr == "vocal":
            assert all(sec["vocal"] is not None and sec["drums"] is None for _, sec in a.sections)
    assert seen == {"solo", "vocal", "band"}
    assert b.info["arrangement"] == "mixed"
