"""train/song_mix_v2.py (FT2 recipe): allowed chords, register near middle C, simple mixes, rests."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nathan_chordify.theory import templates as T  # noqa: E402
from train import chords as C  # noqa: E402
from train import piano as P  # noqa: E402
from train import song_mix as SM  # noqa: E402
from train import song_mix_v2 as V  # noqa: E402

needs_salamander = pytest.mark.skipif(not P.available(), reason="Salamander piano not downloaded")


def test_quality_map_covers_every_other_quality():
    assert set(V.ALLOWED) | set(V.QUALITY_MAP) == set(T.QUALITIES)
    assert not set(V.ALLOWED) & set(V.QUALITY_MAP)
    assert set(V.QUALITY_MAP.values()) <= set(V.ALLOWED)


@pytest.mark.parametrize("text,want", [("C6", "C:maj"), ("Am6", "A:min"), ("C+", "C:maj"), ("Bm7b5", "B:dim"),
                                       ("G13", "G:9"), ("Cmaj7", "C:maj7"), ("G/B", "G:maj/3"), ("C", "C:maj")])
def test_allowed_spec_maps_to_the_nearest_allowed_chord(text, want):
    assert V.allowed_spec(text).label == want


def test_every_vocabulary_chord_maps_into_the_allowed_list():
    for c in C.enumerate_chords():
        m = V.to_allowed(c)
        assert m.quality in V.ALLOWED and m.root == c.root


def test_plans_use_allowed_chords_mostly_plain_with_rests_and_arrangements():
    q, arr, n, slash, rests = Counter(), Counter(), 0, 0, 0
    for i in range(150):
        plan = V.plan_song(np.random.default_rng([9, i]), "train", charts_ok=bool(SM.chart_files("train")))
        arr[plan.info["arrangement"]] += 1
        rests += plan.info["rests"]
        assert plan.segs[-1].spec is not None and plan.segs[0].t0 == pytest.approx(plan.lead)
        for s in plan.segs:
            if s.spec is None:
                continue
            n += 1
            q[s.spec.quality] += 1
            slash += s.spec.bass != s.spec.root
        if plan.info["arrangement"] == "solo":
            assert all(sec[k] is None for _, sec in plan.sections for k in V.LAYERS)
    assert set(q) <= set(V.ALLOWED)
    assert 0.7 <= (q["maj"] + q["min"]) / n <= 0.9
    assert 0.05 <= slash / n <= 0.2
    assert set(arr) == {"solo", "soft", "band"} and arr["band"] < arr["solo"]
    assert rests > 10


def test_voicings_stay_near_middle_c():
    rng = np.random.default_rng(4)
    for text in ("C", "G/B", "Am7", "Fmaj7", "F/G", "D7sus4", "Bdim7", "Ebadd9", "C#m9"):
        spec = C.chord(text)
        for layout in ("lh_rh", "two_hand", "one_hand"):
            for _ in range(30):
                v = V.voicing(spec, rng, layout)
                if v is None:
                    continue
                assert not V.check_voicing(spec, v)
                assert v.notes[0] >= V.LH_MIN and v.notes[0] % 12 == spec.bass
                assert v.notes[-1] <= V.RH_RANGE[1]
                assert len({m % 12 for m in v.lh}) == len(v.lh)            # no left-hand octave doubling
                if v.lh:
                    assert v.rh[0] >= V.RH_RANGE[0] and v.lh[-1] < v.rh[0]
    vs = V.voicings(C.chord("G"), rng, 8)
    assert vs and all(48 <= v.notes[0] <= 59 or 43 <= v.notes[0] for v in vs)


def test_frame_labels_rest_decay_and_n():
    a, b = C.chord("C"), C.chord("F")
    segs = [SM.Seg(0, 2, 1.0, 2.0, a, False), SM.Seg(2, 4, 2.0, 3.0, None, False), SM.Seg(4, 6, 3.0, 4.0, b, False)]
    plan = SM.SongPlan("4/4", 120.0, 1.0, segs, 4.3, [(0.0, {})], {"duration": 20.0})
    F = SM.WIN_FRAMES
    root, bass, notes = V.frame_labels(plan, F)
    t = SM.frame_times(F)
    assert np.all(root[t < 1.0] == 12)
    assert np.all(root[(t >= 1.0) & (t < 2.0 + V.REST_DECAY_S)] == 0)          # C rings 0.25 s into the rest
    assert np.all(root[(t >= 2.0 + V.REST_DECAY_S + 0.01) & (t < 3.0)] == 12)
    assert np.all(root[(t >= 3.0) & (t < 4.3)] == 5)
    assert np.all(root[t >= 4.3] == 12) and np.all(notes[root == 12] == 0)


@needs_salamander
def test_make_item_shapes_register_and_shard(tmp_path):
    it = V.make_item(1, 31, "val", SM.get_piano())
    m, F = it["meta"], it["cqt"].shape[1]
    assert F >= SM.WIN_FRAMES and len(it["root"]) == F
    assert m["recipe"] == "ft2" and m["arrangement"] in V.ARRANGEMENT_P
    reg = m["register"]
    assert reg["bass_min"] >= V.LH_MIN and reg["rh_max"] <= V.RH_RANGE[1]
    assert m["n_frames_N"] == int((it["root"] == 12).sum())
    assert all(lab == "N" or V.allowed_spec(lab).label == lab for _, _, lab, _ in m["chords"])
    if m.get("synth_bass"):
        assert V.SYNTH_BASS[0] <= m["synth_bass"][0]
    r = V.write_shard("val", 31, 0, [1], out_dir=tmp_path)
    z = SM.load_shard(Path(r["path"]))
    np.testing.assert_array_equal(z["root"], it["root"])
    st = V.stats(tmp_path)
    assert st["val"]["items"] == 1 and st["val"]["outside_allowed"] == []
    from train import finetune_ace as FT                    # the FT1 trainer reads the v2 pool unchanged
    d = FT.PoolData("val", tmp_path)
    x, root, bass, notes = FT.crop(d.items[0], 0)
    assert x.shape == (144, SM.WIN_FRAMES)
