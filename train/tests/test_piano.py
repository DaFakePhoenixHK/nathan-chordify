"""train/piano.py: SFZ parsing, pitch, velocity, damping/pedal, determinism, effects and speed."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train import piano as P  # noqa: E402

SR = P.SR
needs_salamander = pytest.mark.skipif(not P.available(), reason="Salamander piano not downloaded")


def _peak_freq(x: np.ndarray, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), 8 * len(x)))
    f = np.fft.rfftfreq(8 * len(x), 1 / SR)
    band = (f > lo) & (f < hi)
    return float(f[band][np.argmax(spec[band])])


def _cents(f: float, midi: float) -> float:
    return 1200 * np.log2(f / (440 * 2 ** ((midi - 69) / 12)))


# ------------------------------------------------------------------------------------- parsing

def test_key_number():
    assert P.key_number("60") == 60 and P.key_number("c4") == 60 and P.key_number("C#4") == 61
    assert P.key_number("db4") == 61 and P.key_number("a0") == 21 and P.key_number("c-1") == 0
    assert P.key_number("-1") == -1
    with pytest.raises(ValueError):
        P.key_number("h2")


@pytest.fixture
def tiny_sfz(tmp_path):
    """A two-layer, one-sample bank: a 440 Hz sine at key 69 (soft) and a louder copy."""
    d = tmp_path / "bank"
    (d / "my samples").mkdir(parents=True)
    t = np.arange(int(2.0 * 48000)) / 48000
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-t / 1.5)).astype(np.float32)
    sf.write(d / "my samples" / "a4 soft.flac", np.stack([tone, tone], 1) * 0.3, 48000)
    sf.write(d / "my samples" / "a4 loud.flac", np.stack([tone, tone], 1), 48000)
    sfz = d / "tiny.sfz"
    sfz.write_text(
        "// a comment line\n"
        "#define $REL 0.5\n"
        "<control> default_path=my samples/\n"
        "<global> amp_veltrack=50 /* block\n comment */\n"
        "<group> ampeg_release=$REL lokey=a3 hikey=c6 pitch_keycenter=69\n"
        "<region> sample=a4 soft.flac lovel=1 hivel=64 tune=0\n"
        "<region> sample=a4 loud.flac lovel=65 hivel=127 volume=-1\n"
        "<group> trigger=release rt_decay=6\n"
        "<region> sample=a4 soft.flac key=69\n", encoding="utf-8")
    return sfz


def test_parse_synthetic_sfz(tiny_sfz):
    regs = P.parse_sfz(tiny_sfz)
    assert len(regs) == 3
    soft, loud, rel = regs
    assert soft.sample.name == "a4 soft.flac" and soft.sample.exists()
    assert (soft.lokey, soft.hikey, soft.keycenter) == (57, 84, 69)
    assert (soft.lovel, soft.hivel, loud.lovel, loud.hivel) == (1, 64, 65, 127)
    assert soft.amp_veltrack == 50 and soft.ampeg_release == 0.5 and loud.volume == -1
    assert rel.trigger == "release" and rel.rt_decay == 6 and (rel.lokey, rel.hikey, rel.keycenter) == (69, 69, 69)
    assert rel.ampeg_release == 0.001                      # a new <group> resets the group opcodes


def test_synthetic_sampler_pitch_and_layers(tiny_sfz):
    pn = P.Piano(tiny_sfz, release_samples=False, pedal_noise=False)
    assert [r.sample.name for r, _ in pn.pick(69, 40)] == ["a4 soft.flac"]
    assert [r.sample.name for r, _ in pn.pick(69, 100)] == ["a4 loud.flac"]
    assert len(pn.pick(69, 64, "xfade")) == 2
    y = pn.render([(71, 100, 0.0, 1.0)])                   # two semitones up: 493.9 Hz
    assert y.shape[1] == 2 and y.dtype == np.float32
    assert abs(_cents(_peak_freq(y[:SR, 0], 300, 800), 71)) < 3
    soft = np.abs(pn.render([(69, 30, 0.0, 1.0)])).max()
    loud = np.abs(pn.render([(69, 120, 0.0, 1.0)])).max()
    assert loud > 3 * soft


# ---------------------------------------------------------------------------------- Salamander

@needs_salamander
def test_parse_salamander():
    pn = P.get_piano()
    assert len(pn.regions) == 641 and len(pn.note_regions) == 480
    assert pn.keys[0] == P.A0 and pn.keys[-1] == P.C8
    assert all(len(pn.layers(k)) == 16 for k in range(P.A0, P.C8 + 1))
    assert {r.ampeg_release for r in pn.layers(60)} == {1.0} and pn.layers(100)[0].ampeg_release == 5.0
    assert len(pn.release_regions) == 157 and len(pn.pedal_down_regions) == 2 and len(pn.pedal_up_regions) == 2
    assert pn.layers(108)[0].tune < pn.layers(105)[0].tune     # the C8 correction is applied


@needs_salamander
@pytest.mark.parametrize("midi", [37, 45, 62, 69, 86, 98])
def test_single_note_pitch(midi):
    """Low/mid/high keys (sampled and resampled ones): the strongest spectral peak is the
    fundamental or (for low keys) a harmonic, and the fundamental is within 25 cents."""
    y = P.get_piano().render([(midi, 80, 0.0, 1.5)], release_samples=False).mean(axis=1)
    seg = y[int(0.05 * SR): int(1.2 * SR)].astype(np.float64)
    f0 = 440 * 2 ** ((midi - 69) / 12)
    f_dom = _peak_freq(seg, 20, 6000)
    h = round(f_dom / f0)
    assert 1 <= h <= 4 and abs(_cents(f_dom / h, midi)) < 30
    if midi >= 50:
        assert h == 1
    f_fund = _peak_freq(seg, f0 * 2 ** (-0.5 / 12), f0 * 2 ** (0.5 / 12))
    assert abs(_cents(f_fund, midi)) < 25


@needs_salamander
def test_velocity_note_off_and_pedal():
    pn = P.get_piano()
    rms = lambda y, a, b: float(np.sqrt(np.mean(y[int(a * SR):int(b * SR)] ** 2)))  # noqa: E731
    soft = pn.render([(60, 35, 0.0, 1.0)], release_samples=False)
    loud = pn.render([(60, 105, 0.0, 1.0)], release_samples=False)
    assert rms(loud, 0, 0.5) > 2 * rms(soft, 0, 0.5)                    # > 6 dB louder
    dry = pn.render([(48, 80, 0.0, 0.5)], duration=2.5, release_samples=False, pedal_noise=False)
    held = pn.render([(48, 80, 0.0, 0.5)], pedal=[(0.0, 2.0)], duration=2.5, release_samples=False,
                     pedal_noise=False)
    assert rms(held, 1.2, 1.8) > 30 * rms(dry, 1.2, 1.8)                 # damped vs ringing
    assert rms(held, 2.4, 2.5) < 0.1 * rms(held, 1.2, 1.8)                # pedal up damps it
    assert P.pedal_intervals(True, [P.PianoNote(60, 80, 1.0, 2.0)]) == [(0.95, 3.0)]


@needs_salamander
def test_release_samples_and_restrike():
    pn = P.get_piano()
    plain = pn.render([(60, 80, 0.0, 1.0)], release_samples=False, duration=3.0)
    rel = pn.render([(60, 80, 0.0, 1.0)], release_samples=True, duration=3.0)
    diff = rel - plain
    assert np.abs(diff[: int(0.99 * SR)]).max() == 0                      # nothing before the key-up
    assert np.abs(diff[int(1.0 * SR):]).max() > 1e-4                      # release noise after it
    twice = pn.render([(60, 80, 0.0, 2.0), (60, 80, 0.5, 1.0)], release_samples=False)
    assert np.all(np.isfinite(twice)) and np.abs(twice).max() < 1.5


@needs_salamander
def test_deterministic_and_effects():
    pn = P.get_piano()
    notes = [(43, 70, 0.0, 1.0), (59, 64, 0.01, 1.0), (62, 66, 0.0, 1.0), (67, 72, 0.02, 1.0)]
    a = pn.render(notes, pedal=True, seed=3, reverb=(1.2, 0.3))
    b = pn.render(notes, pedal=True, seed=3, reverb=(1.2, 0.3))
    assert np.array_equal(a, b)
    assert a.shape[0] > pn.render(notes, pedal=True, seed=3).shape[0]     # reverb adds a tail
    loud = pn.render(notes, gain_db=30.0, limit=True)
    assert np.abs(loud).max() <= 10 ** (-1 / 20) + 1e-6
    x = pn.render(notes, velocity_mode="xfade")
    assert np.all(np.isfinite(x)) and np.abs(x).max() > 0.01


@needs_salamander
def test_global_detune_keeps_note_times():
    pn = P.get_piano()
    notes = [(69, 80, 0.5, 1.0)]
    plain = pn.render(notes, duration=2.0, release_samples=False).mean(axis=1)
    sharp = pn.render(notes, duration=2.0, release_samples=False, tune_cents=50.0).mean(axis=1)
    assert len(plain) == len(sharp) == int(np.ceil(2.0 * SR))
    f = _peak_freq(sharp[int(0.6 * SR):int(1.4 * SR)].astype(np.float64), 400, 500)
    f0 = _peak_freq(plain[int(0.6 * SR):int(1.4 * SR)].astype(np.float64), 400, 500)
    assert 1200 * np.log2(f / f0) == pytest.approx(50.0, abs=4.0)
    onset = lambda y: np.argmax(np.abs(y) > 0.05 * np.abs(y).max()) / SR  # noqa: E731
    assert abs(onset(sharp) - onset(plain)) < 0.003


@needs_salamander
def test_render_speed_two_second_chord():
    pn = P.get_piano()
    notes = [(48, 80, 0.0, 2.0), (64, 70, 0.0, 2.0), (67, 72, 0.0, 2.0), (72, 75, 0.0, 2.0)]
    pn.preload(keys=[48, 64, 67, 72])
    pn.render(notes)                                                       # warm the caches
    times = []
    for _ in range(5):
        t = time.perf_counter()
        pn.render(notes)
        times.append(time.perf_counter() - t)
    assert min(times) < 0.050, times
