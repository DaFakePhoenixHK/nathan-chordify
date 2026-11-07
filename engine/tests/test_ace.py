"""WP4b tests: consonance-ACE adapter (DESIGN §5.5) on synthetic piano-like progressions."""
from __future__ import annotations

import time

import numpy as np
import pytest
import soundfile as sf

from nathan_chordify.models import ace
from nathan_chordify.models import base

torch = pytest.importorskip("torch")
CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
needs_repo = pytest.mark.skipif(not ace.checkpoint_path().is_file(),
                                reason="third_party/consonance-ACE is not cloned")
pytestmark = needs_repo

SR = 44100
CHORD_S = 2.0
# name → (root pc, bass midi, upper voicing)
CHORDS = {"C": (0, 36, [60, 64, 67, 72]), "G": (7, 43, [59, 62, 67, 71]),
          "Am": (9, 33, [60, 64, 69, 72]), "F": (5, 41, [60, 65, 69, 72])}
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
    """Upper voicing and bass note struck every second (a left-hand pulse).

    ACE answers "N" where the bass has decayed (a bass struck once per 2 s chord decays ~10 dB
    by the second strike): its 12 root classes still rank right on 99–100 % of frames, but N wins
    the argmax on 13–23 % of them. A re-struck bass, as in real accompaniment, avoids that."""
    y = np.zeros(int((len(seq) * chord_s + 1.0) * sr), np.float32)
    for i, name in enumerate(seq):
        _, bass, upper = CHORDS[name]
        s = int(i * chord_s * sr)
        for strike in np.arange(0.0, chord_s, 1.0):
            s2 = s + int(strike * sr)
            for m in upper:
                x = 0.12 * piano_tone(m, chord_s - strike, sr)
                y[s2:s2 + x.size] += x
            x = 0.3 * piano_tone(bass, chord_s - strike, sr)
            y[s2:s2 + x.size] += x
    return y / max(1.0, np.abs(y).max() / 0.9)


def _truth(frame_t: np.ndarray, seq: list[str], margin: float = 0.3):
    idx = np.floor(frame_t / CHORD_S).astype(int)
    inside = (idx >= 0) & (idx < len(seq))
    dist = np.abs(frame_t - np.round(frame_t / CHORD_S) * CHORD_S)
    return np.clip(idx, 0, len(seq) - 1), inside & (dist > margin)


@pytest.fixture(scope="module")
def prog_wav(tmp_path_factory):
    y = render(PROG * 4)                                   # 33 s: two windows
    p = tmp_path_factory.mktemp("ace") / "audio.wav"
    sf.write(str(p), np.stack([y, 0.8 * y], axis=1), SR, subtype="PCM_16")
    return p


@pytest.fixture(scope="module")
def evidence(prog_wav):
    m = ace.Ace()
    m.run(prog_wav, DEVICE)                                # warm-up (import, weights, librosa filters)
    ev = m.run(prog_wav, DEVICE)
    print(f"\nconsonance-ACE on {m.stats['audio_s']:.0f} s: {m.stats['windows']} windows, CQT "
          f"{m.stats['cqt_s']:.2f} s, inference {m.stats['infer_s']:.2f} s ({DEVICE})")
    return ev


# ---------------------------------------------------------------------------------------------
# weights and imports
# ---------------------------------------------------------------------------------------------

def test_checkpoint_loads_with_weights_only():
    p = ace.check_checkpoint()
    assert p.stat().st_size == ace.CKPT_BYTES
    ck = torch.load(p, map_location="cpu", weights_only=True)
    hp = ck["hyper_parameters"]
    assert hp["input_dim"] == ace.N_BINS and hp["positional_encoding"] is True
    assert {k: hp["vocabularies"][k] for k in ("root", "bass", "onehot")} == {"root": 13, "bass": 13, "onehot": 12}
    model = ace.load(DEVICE)                               # strict=True inside
    sd = model.state_dict()
    assert set(sd) == set(ck["state_dict"])
    k = "output_root.weight"
    assert torch.equal(sd[k].cpu(), ck["state_dict"][k])
    assert not model.training
    assert ace.load(DEVICE) is model                       # resident between runs


def test_ace_resolves_to_third_party_and_order():
    mods = ace.import_ace()
    import sys
    from pathlib import Path
    import ACE
    assert ace.repo_dir().resolve() in Path(ACE.__file__).resolve().parents
    assert "ACE.mir_evaluation" in sys.modules and mods["model"].__name__ == "ConformerDecomposedModel"
    # our eval package must not shadow ACE's top-level ``metrics`` (review-feasibility issue 3)
    m = sys.modules.get("metrics")
    assert m is None or ace.repo_dir().resolve() in Path(list(m.__path__)[0]).resolve().parents


def test_release_gpu_unloads():
    from nathan_chordify import models
    ace.load(DEVICE)
    assert ace._models
    models.release_gpu()
    assert not ace._models
    ace.load(DEVICE)


# ---------------------------------------------------------------------------------------------
# windows and features
# ---------------------------------------------------------------------------------------------

def test_windows_tile_exactly():
    for n in (1, 50, 861, 862, 863, 1508, 1509, 2000, 10_635, 25_840):
        ws = ace.windows(n, win=862, crop=108)
        kept = np.concatenate([np.arange(k0, k1) for _, _, k0, k1 in ws])
        assert np.array_equal(kept, np.arange(n))
        assert all(e - s == 862 for s, e, _, _ in ws)
        assert all(s <= k0 < k1 <= e for s, e, k0, k1 in ws)
        if n > 862:
            assert ws[-1][1] == n                          # last window aligned to the end
            for (s, e, k0, k1), (s2, _, k2, _) in zip(ws, ws[1:]):
                assert k1 - s2 >= 108 and e - k1 >= 108    # ≥ crop dropped at both interior edges


def test_window_constants():
    L, win, crop = ace.Ace().window_frames()
    assert (L, win, crop) == (441_000, 862, 108)
    assert ace.HOP / ace.SR == pytest.approx(0.02322, abs=1e-5)


def test_features_match_ace_reference(prog_wav):
    """Our window features are exactly ACE's AudioChunkProcessor + CQTransform output."""
    mods = ace.import_ace(audio_processor=True)
    m = ace.Ace()
    L, win, _ = m.window_frames()
    ref = mods["chunker"](audio_path=prog_wav, target_sample_rate=ace.SR, hop_length=ace.HOP,
                          max_sequence_length=m.chunk_s, device="cpu",
                          transform=mods["cqt"](ace.SR, ace.HOP), normalize=True)
    y = ace.load_audio(prog_wav)
    assert np.allclose(y, ref.base_signal[0].numpy(), atol=1e-6)
    for start in (0, 646):
        ours = m.features(y, start, L)
        theirs = ref.process_chunk(onset=(start * ace.HOP + 0.5) / ace.SR).numpy()[0]
        assert ours.shape == theirs.shape == (ace.N_BINS, win)
        assert np.allclose(ours, theirs, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------------

def test_shapes_and_conventions(evidence):
    ev = evidence
    assert ev.name == "ace@mix"
    T = len(ev.frame_t)
    assert ev.root.shape == (T, 13) and ev.bass.shape == (T, 13) and ev.notes.shape == (T, 12)
    assert np.allclose(ev.root.sum(axis=1), 1.0, atol=1e-4) and np.allclose(ev.bass.sum(axis=1), 1.0, atol=1e-4)
    assert ev.notes.min() >= 0.0 and ev.notes.max() <= 1.0
    assert np.allclose(np.diff(ev.frame_t), ace.HOP / ace.SR)
    assert ev.frame_t[0] == pytest.approx(-ev.offset_s)
    assert ev.head_labels["root"][12] == "N" and ev.head_labels["root"][0] == "C"
    assert ev.label_probs is None and ev.labels is None


def test_progression_root_and_bass(evidence):
    ev, seq = evidence, PROG * 4
    idx, keep = _truth(ev.frame_t, seq)
    want = np.array([CHORDS[seq[i]][0] for i in idx[keep]])
    root_acc = np.mean(ev.root[keep].argmax(axis=1) == want)
    bass_acc = np.mean(ev.bass[keep].argmax(axis=1) == want)
    tones = {"C": (0, 4, 7), "G": (7, 11, 2), "Am": (9, 0, 4), "F": (5, 9, 0)}
    note_hit = np.mean([np.all(ev.notes[k][list(tones[seq[i]])] > 0.5)
                        for k, i in zip(np.flatnonzero(keep), idx[keep])])
    print(f"\nframe accuracy away from boundaries: root {root_acc:.3f}, bass {bass_acc:.3f}, "
          f"all triad tones > 0.5 {note_hit:.3f}")
    assert root_acc >= 0.9 and bass_acc >= 0.9
    assert note_hit >= 0.8


def test_label_change_latency_is_small(evidence):
    """Rough latency check on synthetic audio (the real offset is measured on mini-D2)."""
    ev = evidence
    lab = ev.root.argmax(axis=1)
    changes = ev.frame_t[1:][np.diff(lab) != 0]
    truth = np.arange(1, 16) * CHORD_S
    errs = [changes[np.abs(changes - t).argmin()] - t for t in truth]
    med = float(np.median(errs))
    print(f"\nmedian root-change offset vs true boundaries: {1000 * med:+.1f} ms")
    assert abs(med) < 0.1


def test_evidence_cache_roundtrip(evidence, tmp_path, monkeypatch):
    path = base.evidence_path(tmp_path, evidence.name, ace.Ace.adapter_version)
    assert path.name == "ace@mix.v1.npz"
    base.save_evidence(evidence, path)
    back = base.load_evidence(path)
    assert back.name == evidence.name and back.offset_s == evidence.offset_s and back.heads == {}
    for k in ("frame_t", "root", "bass", "notes"):
        assert np.array_equal(getattr(back, k), getattr(evidence, k))
    m = ace.Ace()
    monkeypatch.setattr(m, "run", lambda wav, device: (_ for _ in ()).throw(AssertionError("not cached")))
    assert m.run_cached(tmp_path / "audio.wav", DEVICE, tmp_path).name == "ace@mix"
    assert m.evidence_name(tmp_path / "stems" / "harmonic.flac") == "ace@harmonic"


def test_chunked_matches_ace_reference_and_whole(tmp_path):
    """Overlapped chunking vs ACE's own inference (AudioChunkProcessor, 20 s chunks without
    overlap, as ACE/inference.py) and vs one long window."""
    mods = ace.import_ace(audio_processor=True)
    seq = PROG * 8                                          # 65 s
    y = render(seq, sr=ace.SR)
    wav = tmp_path / "prog.wav"
    sf.write(str(wav), y, ace.SR, subtype="FLOAT")
    m = ace.Ace()
    root, bass, _ = m.infer(ace.load_audio(wav), DEVICE)
    n = len(root)
    t_ours = np.arange(n) * ace.HOP / ace.SR

    # ACE's reference: chunk i covers [20 i, 20 i + 20) s; its frame j sits at 20 i + j·hop/sr
    model = ace.load(DEVICE)
    chunker = mods["chunker"](audio_path=wav, target_sample_rate=ace.SR, hop_length=ace.HOP,
                              max_sequence_length=m.chunk_s, device="cpu",
                              transform=mods["cqt"](ace.SR, ace.HOP), normalize=True)
    ref_t, ref_p = [], []
    for i in range(int(np.ceil(y.size / ace.SR / m.chunk_s))):
        feats = chunker.process_chunk(onset=i * m.chunk_s)[None].to(DEVICE)     # [1, 1, 144, 862]
        with torch.inference_mode():
            r = torch.softmax(model(feats)["root"], -1)[0].cpu().numpy()
        tj = i * m.chunk_s + np.arange(len(r)) * ace.HOP / ace.SR
        ok = tj < (i + 1) * m.chunk_s
        ref_t.append(tj[ok])
        ref_p.append(r[ok])
    ref_t, ref_p = np.concatenate(ref_t), np.concatenate(ref_p)
    near = np.clip(np.searchsorted(ref_t, t_ours), 0, len(ref_t) - 1)
    aligned = np.abs(ref_t[near] - t_ours) < 1e-3          # the same frame time (chunk 0, and every
    ref_root = ref_p[near]                                  # frame whose chunk starts on the hop grid)
    first = np.arange(n) < ace.windows(n, 862, 108)[0][3]   # our first window == ACE's first chunk
    assert aligned[first].all()                            # (batch 4 vs 1: cuDNN TF32 differences ~2e-4)
    assert np.allclose(root[first], ref_root[first], atol=2e-3)
    assert np.array_equal(root[first].argmax(1), ref_root[first].argmax(1))
    a, b = root.argmax(1), ref_root.argmax(1)
    agree_ref = float(np.mean(a == b))
    both = (a < 12) & (b < 12)                              # disagreements are N-vs-chord, not chord-vs-chord
    agree_ref_chords = float(np.mean(a[both] == b[both]))

    m1 = ace.Ace()
    m1.chunk_s = 70.0                                       # one 70 s window (positional encoding ≤ 5000 frames)
    w_root, w_bass, _ = m1.infer(ace.load_audio(wav), DEVICE)
    agree_whole = float(np.mean(root.argmax(1) == w_root.argmax(1)))
    agree_bass = float(np.mean(bass.argmax(1) == w_bass.argmax(1)))
    idx, keep = _truth(t_ours, seq)
    want = np.array([CHORDS[seq[i]][0] for i in idx[keep]])
    acc = {"chunked": float(np.mean(root[keep].argmax(1) == want)),
           "ace_ref": float(np.mean(ref_root[keep].argmax(1) == want)),
           "whole": float(np.mean(w_root[keep].argmax(1) == want))}
    print(f"\nroot argmax agreement: chunked vs ACE reference {agree_ref:.3f} ({agree_ref_chords:.3f} where "
          f"neither says N), chunked vs one window {agree_whole:.3f} (bass {agree_bass:.3f}); root accuracy {acc}")
    assert agree_ref >= 0.9 and agree_ref_chords >= 0.98
    assert agree_whole >= 0.95 and agree_bass >= 0.95
    assert acc["chunked"] >= acc["ace_ref"] - 0.01


@needs_cuda
def test_ten_minutes_in_chunks_with_flat_vram(tmp_path):
    one_min = render(PROG * 7 + PROG[:2], sr=ace.SR)[: 60 * ace.SR]
    y10 = np.tile(one_min, 10)
    p = tmp_path / "long.wav"
    sf.write(str(p), y10, ace.SR, subtype="PCM_16")
    m = ace.Ace()
    ace.load("cuda")

    torch.cuda.synchronize()
    base_mem = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ev = m.run(p, "cuda")
    wall = time.perf_counter() - t0
    peak10 = torch.cuda.max_memory_allocated() - base_mem
    n_win = m.stats["windows"]

    torch.cuda.reset_peak_memory_stats()
    m.infer(y10[: 70 * ace.SR], "cuda")                     # ≥ one full batch of windows
    peak1 = torch.cuda.max_memory_allocated() - base_mem
    print(f"\n10-min file: {len(ev.frame_t)} frames, {n_win} windows, total {wall:.1f} s (CQT "
          f"{m.stats['cqt_s']:.1f} s, inference {m.stats['infer_s']:.1f} s); max_memory_allocated extra: "
          f"10 min {peak10 / 2**20:.0f} MiB, 70 s {peak1 / 2**20:.0f} MiB; resident weights {base_mem / 2**20:.0f} MiB")
    assert len(ev.frame_t) == ace.n_frames(y10.size)
    assert peak10 <= 1.1 * peak1 + 32 * 2**20
    assert ev.root[-200:, :12].sum(axis=1).min() > 0.5      # the last window is real output, not zeros


# ---------------------------------------------------------------------------------------------
# fusion: the evidence is what decode/evidence.py's "ace" source expects (§5.9)
# ---------------------------------------------------------------------------------------------

def test_plugs_into_fusion(evidence):
    from nathan_chordify.decode import calibrate as CAL
    from nathan_chordify.decode import evidence as EV
    from nathan_chordify.theory.vocab import candidates
    from nathan_chordify.types import DecoderParams
    vocab = candidates()
    seq = PROG * 4
    grid = CAL.synthetic_grid(bpm=60.0, n_bars=len(seq) // 2, subdiv=4)          # 2 beats per chord
    harm = base.ChordEvidence(**{**evidence.__dict__, "name": "ace@harmonic"})
    ev = EV.build_full(grid, [evidence, harm], None, None, vocab, DecoderParams())
    assert "ace" in EV.sources(ev) and ev.parts["ace"].shape == (len(grid.cell_t) - 1, len(vocab.cands))
    best = ev.parts["ace"].argmax(axis=1)
    mid = (grid.cell_t[:-1] + grid.cell_t[1:]) / 2
    idx, keep = _truth(mid, seq)
    roots = np.array([vocab.cands[c].root for c in best])
    want = np.array([CHORDS[seq[i]][0] for i in idx])
    acc = float(np.mean(roots[keep] == want[keep]))
    print(f"\nace source: argmax-candidate root accuracy per cell {acc:.3f}")
    assert acc >= 0.9
