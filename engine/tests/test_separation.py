"""WP2 tests: stem separation (DESIGN §5.2) — fast mode, htdemucs_ft full mode and the cache."""
from __future__ import annotations

import time

import numpy as np
import pytest
import soundfile as sf

from nathan_chordify import separation
from nathan_chordify.separation import demucs_sep
from nathan_chordify.types import AudioAsset

SR = 44100
DUR = 10.0


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


needs_cuda = pytest.mark.skipif(not _cuda(), reason="needs CUDA")


def synth_mix(dur: float = DUR, sr: int = SR) -> np.ndarray:
    """[N, 2] stereo: bass line, piano-ish chords, a vibrato 'voice' and noise-burst drums."""
    rng = np.random.default_rng(0)
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros((t.size, 2), np.float32)
    for i, (b, chord) in enumerate([(55.0, (261.6, 329.6, 392.0)), (49.0, (246.9, 293.7, 392.0))] * int(dur // 4 + 1)):
        s, e = int(i * 2 * sr), int((i + 1) * 2 * sr)
        if s >= t.size:
            break
        tt = t[s:e] - t[s]
        env = np.exp(-tt / 1.0)
        y[s:e, 0] += 0.25 * np.sin(2 * np.pi * b * tt) * env
        y[s:e, 1] += 0.25 * np.sin(2 * np.pi * b * tt) * env
        for k, f in enumerate(chord):
            y[s:e, k % 2] += 0.08 * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * env
    voice = 0.1 * np.sin(2 * np.pi * (440 * t + 3 * np.sin(2 * np.pi * 5 * t)))
    y += voice[:, None]
    for beat in np.arange(0, dur, 0.5):
        s = int(beat * sr)
        n = min(int(0.05 * sr), t.size - s)
        y[s:s + n] += (0.2 * rng.standard_normal((n, 2)) * np.exp(-np.arange(n) / (0.01 * sr))[:, None])
    return np.clip(y, -1, 1)


def make_asset(tmp_path, y: np.ndarray) -> AudioAsset:
    d = tmp_path / "cache" / "sha256-test"
    d.mkdir(parents=True)
    wav = d / "audio.wav"
    sf.write(str(wav), y, SR, subtype="PCM_16")
    return AudioAsset(key="sha256-test", dir=d, wav=wav, play=d / "play.m4a", duration_s=len(y) / SR,
                      title="synthetic", artist=None, source_type="upload", source_url=None)


def test_fast_mode_lowpass_and_absent_stems(tmp_path):
    t = np.arange(int(4 * SR)) / SR
    low, high = np.sin(2 * np.pi * 80 * t), np.sin(2 * np.pi * 2000 * t)
    y = 0.4 * np.stack([low + high, low + high], axis=1)
    asset = make_asset(tmp_path, y)
    stems = separation.separate(asset, "fast")
    assert stems.model == "none"
    assert stems.mix == asset.wav and stems.harmonic == asset.wav
    assert stems.bass.exists() and sf.info(str(stems.bass)).subtype == "PCM_16"
    for p in (stems.vocals, stems.drums, stems.other):
        assert not separation.available(p) and not p.exists()
    assert separation.available(stems.bass) and separation.available(stems.harmonic)
    b, sr = sf.read(str(stems.bass), always_2d=True)
    assert sr == SR and b.shape == y.shape
    mid = b[SR:-SR, 0]
    spec = np.abs(np.fft.rfft(mid))
    freqs = np.fft.rfftfreq(mid.size, 1 / SR)
    a80, a2k = spec[np.argmin(np.abs(freqs - 80))], spec[np.argmin(np.abs(freqs - 2000))]
    assert 20 * np.log10(a80 / a2k) > 40                       # 2 kHz is gone, 80 Hz kept
    assert np.abs(b[SR:-SR, 0] - 0.4 * low[SR:-SR]).max() < 0.02   # zero phase: no delay


def test_bad_mode(tmp_path):
    asset = make_asset(tmp_path, np.zeros((SR, 2)))
    with pytest.raises(ValueError):
        separation.separate(asset, "medium")


@needs_cuda
def test_full_mode_writes_flac_stems_and_caches(tmp_path, monkeypatch):
    import torch
    y = synth_mix()
    asset = make_asset(tmp_path, y)
    calls = []
    demucs_sep._sep = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    stems = separation.separate(asset, "full", lambda s, p, m: calls.append(p), device="cuda")
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()
    print(f"\nhtdemucs_ft on {DUR:.0f} s (cold, incl. model load): {wall:.1f} s, peak VRAM {peak / 2**30:.2f} GB")

    assert stems.model == "htdemucs_ft" and stems.mix == asset.wav
    paths = [stems.vocals, stems.drums, stems.bass, stems.other, stems.harmonic]
    assert [p.name for p in paths] == ["vocals.flac", "drums.flac", "bass.flac", "other.flac", "harmonic.flac"]
    assert sorted(p.name for p in (asset.dir / "stems").iterdir()) == sorted(p.name for p in paths)
    for p in paths:
        info = sf.info(str(p))
        assert (info.format, info.subtype, info.samplerate, info.channels) == ("FLAC", "PCM_16", SR, 2)
        assert info.frames == len(y)
    bass, _ = sf.read(str(stems.bass))
    other, _ = sf.read(str(stems.other))
    harm, _ = sf.read(str(stems.harmonic))
    assert np.abs(harm - np.clip(bass + other, -1, 1)).max() < 2e-4
    assert np.sqrt(np.mean(bass ** 2)) > 0.01                    # the 50 Hz line lands in the bass stem
    assert calls and calls[-1] == 1.0 and all(0 <= p <= 1 for p in calls)

    # second call: served from the cache, demucs is not run
    monkeypatch.setattr(demucs_sep, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-ran")))
    again = separation.separate(asset, "full")
    assert again == stems

    # warm run (model resident): the steady-state separation time
    t0 = time.perf_counter()
    demucs_sep.separate_tensor(np.ascontiguousarray(y.T.astype(np.float32)), "cuda")
    print(f"htdemucs_ft warm on {DUR:.0f} s: {time.perf_counter() - t0:.1f} s")


@needs_cuda
def test_oom_ladder_falls_back(monkeypatch):
    """A CUDA OOM at segment 7.8 retries at segment 6 (after unloading the chord models)."""
    import torch
    seen = []

    class FakeSep:
        def __init__(self, dev, seg):
            self.dev, self.seg = dev, seg

        def separate_tensor(self, x, sr):
            seen.append((self.dev, self.seg))
            if self.seg > 7:
                raise torch.cuda.OutOfMemoryError("fake")
            return x, {k: x.clone() for k in demucs_sep.SOURCES}

    monkeypatch.setattr(demucs_sep, "_separator", lambda dev, seg, progress: FakeSep(dev, seg))
    out = demucs_sep.separate_tensor(np.zeros((2, 100), np.float32), "cuda")
    assert seen == [("cuda", 7.8), ("cuda", 6.0)]
    assert set(out) == set(demucs_sep.SOURCES)
