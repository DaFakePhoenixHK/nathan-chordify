"""WP4a tests: lv-chordia adapter (DESIGN §5.5) on synthetic piano-like progressions."""
from __future__ import annotations

import time

import numpy as np
import pytest
import soundfile as sf

from nathan_chordify.models import base
from nathan_chordify.models import lv_chordia as lv

torch = pytest.importorskip("torch")
CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")

SR = 44100
CHORD_S = 2.0
# name → (root pc, triad label, bass midi, upper voicing)
CHORDS = {"C": (0, "C:maj", 36, [60, 64, 67, 72]), "G": (7, "G:maj", 43, [59, 62, 67, 71]),
          "Am": (9, "A:min", 33, [60, 64, 69, 72]), "F": (5, "F:maj", 41, [60, 65, 69, 72])}
PROG = ["C", "G", "Am", "F"]


def piano_tone(midi: int, dur: float, sr: int = SR) -> np.ndarray:
    """Harmonic-rich decaying tone (8 partials, slight inharmonicity, 5 ms attack)."""
    t = np.arange(int(dur * sr)) / sr
    f = 440.0 * 2 ** ((midi - 69) / 12)
    y = np.zeros_like(t)
    for k in range(1, 9):
        fk = f * k * np.sqrt(1 + 1e-4 * k * k)
        if fk < sr / 2:
            y += 0.6 ** (k - 1) * np.sin(2 * np.pi * fk * t) * np.exp(-t * (0.8 + 0.3 * k))
    return (y * np.minimum(t / 0.005, 1.0)).astype(np.float32)


def render(seq: list[str], sr: int = SR, chord_s: float = CHORD_S) -> np.ndarray:
    """Upper voicing struck on each beat-pair (every 1 s) and a bass note per chord."""
    y = np.zeros(int((len(seq) * chord_s + 1.0) * sr), np.float32)
    for i, name in enumerate(seq):
        _, _, bass, upper = CHORDS[name]
        s = int(i * chord_s * sr)
        for strike in np.arange(0.0, chord_s, 1.0):
            s2 = s + int(strike * sr)
            for m in upper:
                x = 0.12 * piano_tone(m, chord_s - strike, sr)
                y[s2:s2 + x.size] += x
        x = 0.3 * piano_tone(bass, chord_s, sr)
        y[s:s + x.size] += x
    return y / max(1.0, np.abs(y).max() / 0.9)


@pytest.fixture(scope="module")
def prog_wav(tmp_path_factory):
    y = render(PROG * 3)
    p = tmp_path_factory.mktemp("lv") / "audio.wav"
    sf.write(str(p), np.stack([y, y], axis=1), SR, subtype="PCM_16")
    return p


@pytest.fixture(scope="module")
def evidence(prog_wav):
    m = lv.LvChordia()
    m.run(prog_wav, DEVICE)                 # warm-up (model load, librosa filter caches)
    ev = m.run(prog_wav, DEVICE)
    print(f"\nlv-chordia on {m.stats['audio_s']:.0f} s: CQT {m.stats['cqt_s']:.2f} s, "
          f"inference {m.stats['infer_s']:.2f} s ({DEVICE})")
    return ev


def test_weights_present_and_loaded():
    paths = lv.check_weights()
    assert len(paths) == 5 and all(p.stat().st_size > 1_000_000 for p in paths)
    nets = lv.load(DEVICE)
    assert len(nets) == 5 and all(n.finalized for n in nets)


def test_windows_tile_exactly():
    for n in (1, 50, 100, 101, 999, 5168, 5169, 25_840):
        ws = lv.windows(n, win=100 if n < 5000 else 5168, crop=10 if n < 5000 else 215)
        kept = np.concatenate([np.arange(k0, k1) for _, _, k0, k1 in ws])
        assert np.array_equal(kept, np.arange(n))
        assert all(s <= k0 < k1 <= e for s, e, k0, k1 in ws)


def test_heads_shapes_and_labels(evidence):
    ev = evidence
    assert ev.name == "lv@mix"
    T = len(ev.frame_t)
    assert set(ev.heads) == set(lv.HEADS)
    for h, labels in ev.head_labels.items():
        assert ev.heads[h].shape == (T, len(labels))
        assert np.allclose(ev.heads[h].sum(axis=1), 1.0, atol=1e-4)
    assert [len(ev.head_labels[h]) for h in lv.HEADS] == [73, 13, 4, 4, 3, 3]
    assert ev.root.shape == (T, 13) and ev.bass.shape == (T, 13)
    assert np.allclose(ev.root.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(ev.root[:, 12], ev.heads["triad"][:, 0])
    assert np.allclose(ev.bass[:, :12], ev.heads["bass"][:, 1:])
    assert np.allclose(np.diff(ev.frame_t), lv.HOP / lv.SR)
    assert ev.frame_t[0] == pytest.approx(-ev.offset_s)


def _truth(frame_t: np.ndarray, seq: list[str]):
    idx = np.floor(frame_t / CHORD_S).astype(int)
    inside = (idx >= 0) & (idx < len(seq))
    dist = np.abs(frame_t - np.round(frame_t / CHORD_S) * CHORD_S)
    return idx, inside & (dist > 0.3)


def test_progression_root_triad_and_bass(evidence):
    ev, seq = evidence, PROG * 3
    idx, keep = _truth(ev.frame_t, seq)
    want_triad = np.array([lv.HEAD_LABELS["triad"].index(CHORDS[seq[i]][1]) for i in idx[keep]])
    want_root = np.array([CHORDS[seq[i]][0] for i in idx[keep]])
    triad_acc = np.mean(ev.heads["triad"][keep].argmax(axis=1) == want_triad)
    root_acc = np.mean(ev.root[keep].argmax(axis=1) == want_root)
    bass_acc = np.mean(ev.bass[keep].argmax(axis=1) == want_root)
    print(f"\nframe accuracy away from boundaries: triad {triad_acc:.3f}, root {root_acc:.3f}, bass {bass_acc:.3f}")
    assert triad_acc >= 0.9 and root_acc >= 0.9
    assert bass_acc >= 0.8


def test_label_change_latency_is_small(evidence):
    """Rough latency check on synthetic audio (the real offset is measured on D2 in M2)."""
    ev = evidence
    lab = ev.heads["triad"].argmax(axis=1)
    changes = ev.frame_t[1:][np.diff(lab) != 0]
    truth = np.arange(1, 12) * CHORD_S
    errs = [changes[np.abs(changes - t).argmin()] - t for t in truth]
    med = float(np.median(errs))
    print(f"\nmedian label-change offset vs true boundaries: {1000 * med:+.1f} ms")
    assert abs(med) < 0.1


def test_evidence_cache_roundtrip(evidence, tmp_path, monkeypatch):
    path = base.evidence_path(tmp_path, evidence.name, lv.LvChordia.adapter_version)
    assert path.name == "lv@mix.v1.npz"
    base.save_evidence(evidence, path)
    back = base.load_evidence(path)
    assert back.name == evidence.name and back.offset_s == evidence.offset_s
    assert back.head_labels == evidence.head_labels and back.notes is None and back.labels is None
    for h in lv.HEADS:
        assert np.array_equal(back.heads[h], evidence.heads[h])
    assert np.array_equal(back.root, evidence.root) and np.array_equal(back.frame_t, evidence.frame_t)

    m = lv.LvChordia()
    monkeypatch.setattr(m, "run", lambda wav, device: (_ for _ in ()).throw(AssertionError("not cached")))
    assert m.run_cached(tmp_path / "audio.wav", DEVICE, tmp_path).name == "lv@mix"


def test_input_names():
    from pathlib import Path
    assert base.input_name(Path("cache/k/audio.wav")) == "mix"
    assert base.input_name(Path("cache/k/stems/harmonic.flac")) == "harmonic"


def test_chunked_inference_matches_single_window():
    y = render(PROG * 8, sr=lv.SR)                         # 64 s
    cqt = lv.cqt_features(y)
    m = lv.LvChordia()
    m.chunk_s, m.overlap_s = 1000.0, 10.0
    whole = m.infer(cqt, DEVICE)
    m.chunk_s, m.overlap_s = 20.0, 4.0
    chunked = m.infer(cqt, DEVICE)
    agree = np.mean(whole["triad"].argmax(1) == chunked["triad"].argmax(1))
    assert agree >= 0.95, agree


@needs_cuda
def test_ten_minutes_in_chunks_with_flat_vram(tmp_path):
    one_min = render(PROG * 7 + PROG[:2], sr=lv.SR)[: 60 * lv.SR]
    y10 = np.tile(one_min, 10)
    p = tmp_path / "long.wav"
    sf.write(str(p), y10, lv.SR, subtype="PCM_16")
    m = lv.LvChordia()
    lv.load("cuda")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    ev = m.run(p, "cuda")
    wall = time.perf_counter() - t0
    peak10 = torch.cuda.max_memory_allocated() - base_mem

    cqt2 = lv.cqt_features(y10[: 120 * lv.SR])            # one full 120 s window
    torch.cuda.reset_peak_memory_stats()
    m.infer(cqt2, "cuda")
    peak2 = torch.cuda.max_memory_allocated() - base_mem
    print(f"\n10-min file: {len(ev.frame_t)} frames, total {wall:.1f} s (CQT {m.stats['cqt_s']:.1f} s, "
          f"inference {m.stats['infer_s']:.1f} s); peak extra VRAM 10 min {peak10 / 2**20:.0f} MiB, "
          f"one 120 s window {peak2 / 2**20:.0f} MiB; resident weights {base_mem / 2**20:.0f} MiB")
    assert len(ev.frame_t) == int(np.ceil(y10.size / lv.HOP)) or abs(len(ev.frame_t) - y10.size / lv.HOP) <= 1
    assert peak10 <= 1.1 * peak2 + 32 * 2**20
    lab = ev.heads["triad"].argmax(axis=1)
    assert lab[-200:].max() > 0                             # the last chunk is real output, not zeros
