"""train/chords.py: enumeration, labels, voicing rules, rendering, progressions, dataset, CLI."""
from __future__ import annotations

import csv
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nathan_chordify.theory import harte as H  # noqa: E402
from nathan_chordify.theory import templates as T  # noqa: E402
from nathan_chordify.theory import vocab as V  # noqa: E402
from train import chords as C  # noqa: E402
from train import piano as P  # noqa: E402

SR = C.SR
needs_salamander = pytest.mark.skipif(not P.available(), reason="Salamander piano not downloaded")
DRY = {"reverb": None, "tune_cents": 0.0, "peak_db": -3.0}


def _chroma(y: np.ndarray, t0: float, t1: float, fmin_note: str | None = "C3") -> np.ndarray:
    import librosa
    seg = y.mean(axis=1)[int(t0 * SR):int(t1 * SR)]
    kw = dict(fmin=librosa.note_to_hz(fmin_note), n_octaves=4) if fmin_note else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return librosa.feature.chroma_cqt(y=seg, sr=SR, hop_length=4096, bins_per_octave=36, **kw).mean(axis=1)


# ---------------------------------------------------------------------------------- enumeration

def test_enumerate_chords_matches_vocab_and_roundtrips():
    chords = C.enumerate_chords()
    assert len(chords) == 1668 == len(V.candidates().cands) - 1
    assert len({c.label for c in chords}) == len(chords)
    assert {c.quality for c in chords} == set(T.QUALITIES)
    for c in chords:
        assert H.parse_harte(c.label) == (c.root, c.quality, c.bass)
        assert c.pcs == V.cand_pcs(c.root, c.quality, c.bass)
        assert C.chord(c.label) is c
    inv = Counter(c.inversion for c in chords)
    # foreign basses: +2/+5/+10 on maj, min, maj7, 6 (3 each), sus2, sus4, add9, m7 (2 each)
    assert inv["root"] == 12 * 32 and inv["non_chord"] == 12 * 20
    assert inv["chord_tone"] == 1668 - 12 * 52


def test_symbols_and_lookup():
    assert C.chord("C#m7/E").label == "C#:min7/b3"
    assert C.chord("E/F#").inversion == "non_chord" and C.chord("G/B").inversion == "chord_tone"
    assert "A:min7/b3" in C.equivalents(C.chord("C6"))
    with pytest.raises(KeyError):
        C.lookup(0, "dim", 2)                        # dim takes no foreign bass
    with pytest.raises(ValueError):
        C.chord("N")


# -------------------------------------------------------------------------------------- voicings

def test_voicings_obey_the_rules_for_every_chord():
    rng = np.random.default_rng(11)
    layouts, positions = Counter(), Counter()
    for c in C.enumerate_chords():
        vs = C.voicings(c, rng, 3)
        assert len(vs) == 3, c.label
        for v in vs:
            assert C.check_voicing(c, v) == [], (c.label, v)
            notes = v.notes
            assert 21 <= notes[0] and notes[-1] <= 108 and notes[0] % 12 == c.bass
            assert (c.required | {c.bass}) <= v.pcs <= c.pcs
            assert all(C.LH_RANGE[0] <= m <= C.LH_RANGE[1] for m in v.lh)
            assert 3 <= len(v.rh) <= 5 and all(C.RH_RANGE[0] <= m <= C.RH_RANGE[1] for m in v.rh)
            layouts[v.layout] += 1
            positions[v.position] += 1
    assert set(layouts) == set(C.LAYOUTS) and set(positions) == set(C.POSITIONS)


def test_check_voicing_catches_mistakes():
    c = C.chord("C/E")
    good = C.Voicing((40,), (60, 67, 72), "lh_rh", "close")
    assert C.check_voicing(c, good) == []
    assert C.check_voicing(c, C.Voicing((36,), (64, 67, 72), "lh_rh", "close"))      # bass not lowest
    assert C.check_voicing(c, C.Voicing((40,), (60, 62, 67), "lh_rh", "close"))      # D is foreign
    assert C.check_voicing(C.chord("C7"), C.Voicing((36,), (64, 67, 72), "lh_rh", "close"))  # no b7
    assert C.check_voicing(c, C.Voicing((40, 43), (60, 67, 72), "lh_rh", "close"))   # muddy LH 3rd
    assert C.check_voicing(C.chord("Em/A"), C.Voicing((), (57, 64, 67, 71), "one_hand", "close"))


def test_voicings_are_deterministic_and_varied():
    c = C.chord("Bbmaj7/D")
    a = C.voicings(c, np.random.default_rng(5), 12)
    b = C.voicings(c, np.random.default_rng(5), 12)
    assert a == b and len({(v.lh, v.rh) for v in a}) == 12
    assert len({v.rh[-1] for v in a}) >= 4                                   # different top notes
    one = C.voicings(c, np.random.default_rng(1), 5, layout="one_hand")
    assert one and all(not v.lh and v.rh[0] % 12 == c.bass for v in one)


def test_count_voicings():
    cnt = C.count_voicings(C.chord("C"))
    assert cnt["two_hand"] > 1000 and cnt["one_hand"] == 6
    assert C.count_voicings(C.chord("Em/A"))["one_hand"] == 0
    st = C.voicing_stats([C.chord("C"), C.chord("C:maj7/3")])
    assert st["chords"] == 2 and st["total"] == sum(st["per_quality"].values())


# ------------------------------------------------------------------------------------- rendering

@needs_salamander
@pytest.mark.parametrize("label", ["C:maj", "A:min", "E:7", "F#:dim7", "Bb:maj7", "G:sus4",
                                   "Ab:aug", "B:hdim7", "Eb:7(#9)", "D:min11"])
def test_chord_chroma_peaks_at_its_pitch_classes(label):
    """The right hand alone in one octave (C4–B4): the top-k chroma bins are the chord's tones."""
    c = C.chord(label)
    rh = tuple(sorted(next(m for m in range(60, 72) if m % 12 == p) for p in c.upper))
    if len(rh) > 5:
        rh = tuple(m for m in rh if m % 12 not in c.optional)
    v = C.Voicing((), rh, "one_hand", "close") if rh[0] % 12 == c.bass else None
    notes = [P.PianoNote(m, 75, 0.05, 1.5) for m in rh]
    y = P.get_piano().render(notes, release_samples=False)
    ch = _chroma(y, 0.15, 1.4)
    pcs = sorted({m % 12 for m in rh})
    assert sorted(np.argsort(ch)[-len(pcs):].tolist()) == pcs
    if v is not None:
        assert C.check_voicing(c, v) == []


@needs_salamander
@pytest.mark.parametrize("label", ["C:maj/3", "A:min7/b3", "E:min/4", "F:maj/2", "D:7/b7", "G:sus4(b7,9)"])
def test_rendered_example_label_and_chroma(label):
    c = C.chord(label)
    rng = np.random.default_rng(4)
    audio, lab = C.render_example(c, None, "block", rng, fx=DRY, duration=1.6)
    assert audio.ndim == 2 and audio.shape[1] == 2 and audio.dtype == np.float32
    assert lab["label"] == label and H.parse_harte(lab["label"]) == (c.root, c.quality, c.bass)
    assert set(lab["sounding_pcs"]) <= set(lab["pcs"]) and c.bass in lab["sounding_pcs"]
    assert min(n["pitch"] for n in lab["notes"]) % 12 == c.bass
    assert all(30 <= n["velocity"] <= 110 for n in lab["notes"])
    assert abs(lab["duration"] - len(audio) / SR) < 1e-4 and lab["t1"] - lab["t0"] == pytest.approx(1.6)
    ch = _chroma(audio, lab["t0"] + 0.1, lab["t1"], None)
    assert ch[lab["sounding_pcs"]].sum() / ch.sum() > 0.45                    # most energy on its tones


@needs_salamander
def test_every_style_plays_every_tone_and_is_deterministic():
    c = C.chord("C:maj7/3")
    for style in C.STYLES:
        rng = np.random.default_rng(9)
        v = C.voicings(c, rng, 1, layout="lh_rh")[0]
        a1, l1 = C.render_example(c, v, style, np.random.default_rng(1))
        a2, l2 = C.render_example(c, v, style, np.random.default_rng(1))
        assert np.array_equal(a1, a2) and l1 == l2
        assert {n["pitch"] for n in l1["notes"]} == set(v.notes), style
        assert l1["style"] == style and np.abs(a1).max() <= 10 ** (-0.5 / 20) + 1e-6


@needs_salamander
def test_render_progression_segments():
    rng = np.random.default_rng(3)
    name, key, chords = C.random_progression(rng, name="huijia_prechorus", key=11, n_chords=6)
    assert len(chords) == 6 and all(isinstance(pc.chord, C.ChordSpec) for pc in chords)
    audio, info = C.render_progression(chords, 72.0, rng, style="pulse")
    segs = info["segments"]
    assert [s["label"] for s in segs] == [pc.chord.label for pc in chords]
    assert all(a["t1"] == b["t0"] for a, b in zip(segs, segs[1:]))
    spb = 60 / 72.0
    for s, pc in zip(segs, chords):
        assert s["t1"] - s["t0"] == pytest.approx(pc.beats * spb, abs=1e-4)
        played = [n for n in info["notes"] if s["t0"] - 0.03 <= n["onset"] < s["t1"] - 0.01]
        assert {n["pitch"] % 12 for n in played} <= set(s["pcs"])
    assert info["duration"] == pytest.approx(len(audio) / SR, abs=1e-4) and info["duration"] > segs[-1]["t1"]
    a2, i2 = C.render_progression(["C", "G/B", ("Am", 2), ("Am/G", 2), "F"], 90.0, np.random.default_rng(0))
    assert [s["symbol"] for s in i2["segments"]] == ["C", "G/B", "Am", "Am/G", "F"]


def test_progression_library_covers_every_quality_in_every_key():
    lib = C.progression_library()
    assert {"walk_up", "four_over_five", "sus_resolution", "passing_diminished", "secondary_dominant",
            "axis_v7", "huijia_prechorus"} <= set(lib)
    assert {pc.chord.quality for p in lib.values() for pc in p} == set(T.QUALITIES)
    for p in lib.values():
        for pc in p:
            for k in range(12):
                assert pc.chord.transpose(k).quality == pc.chord.quality


@needs_salamander
def test_iter_dataset_is_deterministic():
    a = list(C.iter_dataset(3, seed=21, progression_share=0.5))
    b = list(C.iter_dataset(3, seed=21, progression_share=0.5))
    for x, y in zip(a, b):
        assert np.array_equal(x["audio"], y["audio"])
        assert {k: v for k, v in x.items() if k != "audio"} == {k: v for k, v in y.items() if k != "audio"}
    c = next(C.iter_dataset(1, seed=22))
    assert c["id"] != a[0]["id"]
    assert len(c["audio"]) != len(a[0]["audio"]) or not np.array_equal(c["audio"], a[0]["audio"])
    one = C.make_example(1, 21, progression_share=0.5)
    assert np.array_equal(one["audio"], a[1]["audio"])


@needs_salamander
def test_preview_cli(tmp_path):
    assert C.main(["--preview", "3", "--out", str(tmp_path), "--seed", "7"]) == 0
    flacs = sorted(tmp_path.glob("*.flac"))
    assert len(flacs) == 3
    rows = list(csv.DictReader(open(tmp_path / "labels.csv", encoding="utf-8")))
    assert {r["file"] for r in rows} == {f.name for f in flacs}
    assert {r["kind"] for r in rows} == {"chord", "progression"}
    for r in rows:
        H.parse_harte(r["harte"])
    import soundfile as sf
    info = sf.info(str(flacs[0]))
    assert info.samplerate == 44100 and info.channels == 2


def test_stats_quick(capsys):
    st = C.print_stats(quick=True)
    out = capsys.readouterr().out
    assert "1668" in out and st["chords"] == 1668 // 12 and st["total"] > 100_000
