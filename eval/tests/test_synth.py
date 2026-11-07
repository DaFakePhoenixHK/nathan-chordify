"""Tests for eval/synth/render.py: DSL parsing, labels/beats alignment, determinism, audio sanity."""
import json
import sys
from pathlib import Path

import mir_eval
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.synth import render as R  # noqa: E402

FAST = dict(bpm=120.0, rt60=0.6, humanize_ms=3.0, seed=7)
SMALL = "| C . . *G/B | Am . . (Am *Am/G) |"


def _quick(chart=SMALL, **kw) -> R.Rendered:
    return R.render(chart, R.RenderParams(**{**FAST, **kw}))


# ------------------------------------------------------------------------------------------- DSL

@pytest.mark.parametrize("symbol, harte", [
    ("B", "B:maj"), ("B+7", "B:aug(b7)"), ("Emaj7", "E:maj7"), ("Em6", "E:min6"), ("A7", "A:7"),
    ("G#m7/F#", "G#:min7/b7"), ("Em/A", "E:min/4"), ("E/F#", "E:maj/2"), ("F#11", "F#:sus4(b7,9)"),
    ("D#7/A#", "D#:7/5"), ("C#m7/E", "C#:min7/b3"), ("C#dim7", "C#:dim7"), ("Bb7", "Bb:7"),
    ("Gsus4", "G:sus4"), ("F/G", "F:maj/2"), ("C6/9", "C:maj6(9)"), ("E7/G#", "E:7/3"),
    ("B/F#", "B:maj/5"), ("Cm(maj7)", "C:minmaj7"), ("F#add9", "F#:maj(9)"), ("D#7sus4", "D#:sus4(b7)"),
    ("N", "N"), ("Em/G?", "E:min/b3"), ("C:maj7/3", "C:maj7/3"),
])
def test_symbol_to_harte(symbol, harte):
    assert R.symbol_to_harte(symbol) == harte
    mir_eval.chord.encode(harte)


def test_bad_symbol_raises():
    with pytest.raises(ValueError):
        R.symbol_to_harte("Hmaj7")
    with pytest.raises(ValueError):
        R.symbol_to_harte("Cfoo")


def test_chord_tones_keep_slash_bass_separate():
    assert R.chord_tones("E:min/4") == (4, (4, 7, 11), 9)          # Em/A: A is only the bass
    root, tones, bass = R.chord_tones("F#:sus4(b7,9)")              # F#11 = E/F#
    assert (root, bass) == (6, 6) and set(tones) == {6, 11, 1, 4, 8}


def test_parse_chart_dots():
    chords = R.parse_chart("| B . . B+7 | Emaj7 . A7 Em6 |")
    assert [(c.beats, c.label) for c in chords] == [
        (3.0, "B:maj"), (1.0, "B:aug(b7)"), (2.0, "E:maj7"), (1.0, "A:7"), (1.0, "E:min6")]


def test_parse_chart_even_split_brackets_repeat():
    assert [c.beats for c in R.parse_chart("| B E |")] == [2.0, 2.0]
    assert [round(c.beats, 6) for c in R.parse_chart("| C F G |")] == [1.333333] * 3
    chords = R.parse_chart("[Verse]\n| C . . (C *C+) | % | (M) note")
    assert [(c.beats, c.label, c.passing) for c in chords] == [
        (3.0, "C:maj", False), (0.5, "C:maj", False), (0.5, "C:aug", True),
        (3.0, "C:maj", False), (0.5, "C:maj", False), (0.5, "C:aug", True)]
    # a dot continues the previous bar's chord; brackets may hold a dot
    assert [(c.beats, c.label) for c in R.parse_chart("| C . . . | . . F (. G) |")] == [
        (6.0, "C:maj"), (1.5, "F:maj"), (0.5, "G:maj")]
    # with dots, a short bar lets its last chord run to the bar end
    assert [c.beats for c in R.parse_chart("| C . F |")] == [2.0, 2.0]


def test_parse_chart_markers_and_errors():
    chords = R.parse_chart("| B . *A7@borrowed_iv *Em6 | N . . . |", default_tag="huijia")
    assert [(c.passing, c.tags) for c in chords] == [
        (False, ()), (True, ("borrowed_iv",)), (True, ("huijia",)), (False, ())]
    assert chords[-1].label == "N"
    with pytest.raises(ValueError):
        R.parse_chart("| C . F G A |")       # five beats in 4/4
    with pytest.raises(ValueError):
        R.parse_chart("| . C . . |")          # dot before any chord
    with pytest.raises(ValueError):
        R.parse_chart("no bars here")


def test_parse_progression_list():
    chords = R.parse_progression([(2, "C"), (1, "G/B", "*"), (1, "A:min", "*walk"), (4, "F")], default_tag="x")
    assert [(c.beats, c.label, c.passing, c.tags) for c in chords] == [
        (2.0, "C:maj", False, ()), (1.0, "G:maj/3", True, ("x",)),
        (1.0, "A:min", True, ("walk",)), (4.0, "F:maj", False, ())]


def test_transpose_idiom():
    canon = R.IDIOMS_BY_NAME["descending_canon"]
    labels = [c.label for c in canon.chords("D")]
    assert labels[:3] == ["D:maj", "A:maj/3", "B:min"]
    assert "| D . . *A/C# |" in canon.chart_in("D")
    huijia = R.IDIOMS_BY_NAME["huijia_prechorus"]
    assert [c.label for c in huijia.chords()][:4] == ["B:maj", "B:maj", "B:aug(b7)", "E:maj7"]
    assert [c.label for c in huijia.chords("Eb")][2] == "Eb:aug(b7)"


def test_idiom_library_has_half_and_one_beat_passing_chords():
    assert len(R.IDIOMS) == 12
    for idiom in R.IDIOMS:
        chords = idiom.chords()
        passing = {c.beats for c in chords if c.passing}
        assert {0.5, 1.0} <= passing, idiom.name
        assert sum(c.beats for c in chords) % 4 == 0, idiom.name
        assert all(c.tags for c in chords if c.passing), idiom.name


def test_mini_d2_plan():
    specs = R.mini_d2_items()
    assert len(specs) == 72 and len({s.item_id for s in specs}) == 72
    assert {s.params.bpm for s in specs} == {65.0, 110.0}
    keys = [s.key for s in specs]
    assert all(keys.count(k) == 6 for k in R.KEYS)                 # every key 3 idioms × 2 tempos
    assert "B" in {s.key for s in specs if s.idiom.name == "huijia_prechorus"}
    for s in specs:
        beats = sum(c.beats for c in s.idiom.chords()) * s.params.repeats
        assert s.params.lead_in + beats * 60 / s.params.bpm + s.params.tail <= R.MAX_SECONDS
        assert -18 <= s.params.pad_db <= 0 and 0.5 <= s.params.rt60 <= 3.0
    pads = {s.params.pad_db for s in specs if s.idiom.name == "walk_up"}
    assert pads == set(R.PAD_SWEEP)
    assert [s.params.seed for s in R.mini_d2_items()] == [s.params.seed for s in specs]


# ---------------------------------------------------------------------------------------- render

def test_render_is_deterministic():
    a, b = _quick(), _quick()
    assert np.array_equal(a.audio, b.audio)
    assert a.segments == b.segments
    assert not np.array_equal(a.audio, _quick(seed=8).audio)


def test_render_format_and_duration():
    r = _quick()
    p = r.params
    assert r.audio.ndim == 2 and r.audio.shape[1] == 2 and r.audio.dtype == np.float32
    assert np.all(np.isfinite(r.audio)) and 0.5 < np.max(np.abs(r.audio)) <= 0.9
    assert abs(r.duration - (p.lead_in + 8 * 60 / p.bpm + p.tail)) < 1e-3
    left, right = r.audio[:, 0], r.audio[:, 1]
    assert np.corrcoef(left, right)[0, 1] > 0.5                    # mono-compatible stereo


def test_labels_align_with_beats():
    r = _quick()
    spb = 60 / r.params.bpm
    assert np.allclose(np.diff(r.beats), spb, atol=2e-6)
    assert r.beats[0] == r.segments[1]["t0"] == r.params.lead_in
    assert list(r.beat_pos) == [1, 2, 3, 4, 1, 2, 3, 4]
    segs = r.segments
    assert segs[0]["t0"] == 0.0 and segs[0]["label"] == "N" and segs[-1]["label"] == "N"
    assert abs(segs[-1]["t1"] - r.duration) < 1e-6
    assert all(a["t1"] == b["t0"] for a, b in zip(segs, segs[1:]))   # contiguous
    grid = r.params.lead_in + spb * np.arange(0, 8.5, 0.5)
    for s in segs[1:-1]:
        assert np.min(np.abs(grid - s["t0"])) < 1e-6                  # every change on the half-beat grid
    assert [(s["label"], s["beats"], s["passing"]) for s in segs[1:-1]] == [
        ("C:maj", 3.0, False), ("G:maj/3", 1.0, True), ("A:min", 3.5, False), ("A:min/b7", 0.5, True)]


def test_swing_moves_offbeat_boundaries():
    r = _quick(swing=0.62)
    spb = 60 / r.params.bpm
    am_g = [s for s in r.segments if s["label"] == "A:min/b7"][0]
    assert abs(am_g["t0"] - (r.params.lead_in + (7 + 0.62) * spb)) < 1e-6
    assert np.allclose(np.diff(r.beats), spb, atol=2e-6)              # beats stay straight


def test_compound_meter():
    r = _quick("| C . | G/B . | Am . | F . |", meter="6/8", bpm=60.0)
    assert list(r.beat_pos) == [1, 2] * 4
    assert len(r.segments) == 6


def test_pad_sustains_previous_chord_over_passing():
    r = _quick(pad_db=0.0)
    pad = [n for n in r.notes if n.part == "pad"]
    g_b = [s for s in r.segments if s["label"] == "G:maj/3"][0]
    sounding = {n.midi % 12 for n in pad if n.t0 <= g_b["t0"] + 1e-6 and n.t1 >= g_b["t1"] - 1e-6}
    assert sounding == {0, 4, 7}                                      # still C major under G/B
    assert not any(g_b["t0"] - 1e-6 <= n.t0 < g_b["t1"] for n in pad)


def test_piano_restrikes_and_bass_follows_slash_chords():
    r = _quick(comping="block")
    for s in r.segments[1:-1]:
        onset = [n for n in r.notes if abs(n.t0 - s["t0"]) < 0.02]
        root, tones, bass = R.chord_tones(s["label"])
        rh = {n.midi % 12 for n in onset if n.part == "piano_rh"}
        assert rh and rh <= set(tones)
        assert [n.midi % 12 for n in onset if n.part == "bass"] == [bass]


def test_bass_audio_pitch_matches_slash_bass():
    # no pad, drums, vocal or left hand; the right hand sits above 160 Hz
    r = _quick("| G/B . . . |", pad_db=None, drums_db=None, vocal_db=None, piano_lh=False, rt60=0.0)
    x = r.audio[:, 0].astype(float)
    seg = x[int(0.7 * R.SR): int(2.0 * R.SR)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8 * len(seg)))
    freqs = np.fft.rfftfreq(8 * len(seg), 1 / R.SR)
    band = (freqs > 30) & (freqs < 160)
    f_peak = freqs[band][np.argmax(spec[band])]
    assert round(12 * np.log2(f_peak / 440) + 69) % 12 == 11           # B in the bass
    assert r.segments[1]["label"] == "G:maj/3"


def test_write_item_roundtrip(tmp_path):
    r = _quick()
    folder = R.write_item(r, tmp_path, "demo", {"idiom": "test"})
    iv, labels = mir_eval.io.load_labeled_intervals(str(folder / "demo.lab"))
    assert labels == [s["label"] for s in r.segments]
    assert np.allclose(iv[:, 0], [s["t0"] for s in r.segments])
    beats = np.loadtxt(folder / "demo.beats", ndmin=2)
    assert np.allclose(beats[:, 0], r.beats) and list(beats[:, 1]) == list(r.beat_pos)
    audio, sr = sf.read(folder / "demo.wav")
    assert sr == 44100 and audio.shape == r.audio.shape
    info = json.loads((folder / "demo.json").read_text(encoding="utf-8"))
    assert info["idiom"] == "test" and [s["passing"] for s in info["segments"]] == [s["passing"] for s in r.segments]


def test_cli_renders_one_idiom(tmp_path):
    assert R.main(["--set", "mini-d2", "--only", "four_over_five", "--out", str(tmp_path)]) == 0
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert len(index["items"]) == 6
    for it in index["items"]:
        folder = tmp_path / it["id"]
        assert {f.suffix for f in folder.iterdir()} == {".wav", ".lab", ".beats", ".json"}
        assert it["duration"] <= R.MAX_SECONDS and it["passing_beats"] == [0.5, 1.0]


# ------------------------------------------------------------------------- sampled piano (optional)

def _salamander_available() -> bool:
    try:
        from train.piano import available
    except ImportError:
        return False
    return available()


def test_piano_option_is_checked():
    assert R.RenderParams().piano == "additive"
    with pytest.raises(ValueError):
        R.render(SMALL, R.RenderParams(**FAST, piano="steinway"))


@pytest.mark.skipif(not _salamander_available(), reason="Salamander piano not downloaded")
def test_salamander_piano_renders_same_timeline(tmp_path):
    add = _quick()
    sal = _quick(piano="salamander")
    assert sal.audio.shape == add.audio.shape and sal.segments == add.segments
    assert np.all(np.isfinite(sal.audio)) and 0.5 < np.max(np.abs(sal.audio)) <= 0.9
    assert not np.array_equal(sal.audio, add.audio)
    assert np.array_equal(sal.audio, _quick(piano="salamander").audio)        # deterministic
    assert R.main(["--chart", "| C . G/B . |", "--piano", "salamander", "--out", str(tmp_path),
                   "--name", "sal"]) == 0
    info = json.loads((tmp_path / "sal" / "sal.json").read_text(encoding="utf-8"))
    assert info["params"]["piano"] == "salamander"
