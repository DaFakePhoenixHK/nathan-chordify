"""WP7 tests on SYNTHETIC evidence: lv-style head probabilities over a 65 BPM grid, subdiv 4.

Real lv-chordia output (D2 renders, 回家) is the WP7 acceptance set in M2 (DESIGN §14.2); these
unit tests check the decoder mechanics: exact decoding without flicker, recovery of soft
half-beat passing chords at the break-even advantage, A–P–A demotion, bass events, class-level
confidence, speed and normalisation.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from nathan_chordify.decode import calibrate as cal
from nathan_chordify.decode import evidence as evm
from nathan_chordify.decode import hsmm, layers, post
from nathan_chordify.decode import params as prm
from nathan_chordify.theory.vocab import candidates
from nathan_chordify.types import BassTrack, ChordEvidence, DecoderParams, Evidence, KeyTrack

LV_HOP = 512 / 22050                                   # lv-chordia's native hop (23 ms)
HEADS = {"triad": 73, "bass": 13, "seventh": 4, "ninth": 4, "eleventh": 3, "thirteenth": 3}
N = (-1, "N", -1)
KEY_B, KEY_C = 11, 0
SUB = 4                                                # cells per beat

# chords as (root pc, quality, bass pc)
B, E, Gsm, Fs, Csm = (11, "maj", 11), (4, "maj", 4), (8, "min", 8), (6, "maj", 6), (1, "min", 1)
Fs_As, E_B, B_Cs = (6, "maj", 10), (4, "maj", 11), (11, "maj", 1)
C, F, Gs_aug = (0, "maj", 0), (5, "maj", 5), (8, "aug", 8)
DIM7_ON_C = [(0, "dim7", 0), (3, "dim7", 0), (6, "dim7", 0), (9, "dim7", 0)]   # one class


# =============================================================================================
# Synthetic lv evidence
# =============================================================================================

def _targets(chord) -> dict[str, int]:
    r, q, b = chord
    if q == "N":
        return {h: 0 for h in HEADS}
    triad, ext = evm.LV_QUALITY[q]
    out = {"triad": evm.lv_triad_index(r, "maj" if triad == "maj|min" else triad), "bass": 1 + b}
    for h, classes in evm.LV_EXT.items():
        cl = ext.get(h, "none")
        out[h] = classes.index("none" if cl == evm.OPT9 else cl)
    return out


def head_probs(mix, s: float = 6.0) -> dict[str, np.ndarray]:
    """A probability mixture of one-hot softmaxes (logit s on each chord's head classes)."""
    out = {h: np.zeros(n) for h, n in HEADS.items()}
    for chord, w in mix:
        for h, i in _targets(chord).items():
            v = np.zeros(HEADS[h])
            v[i] = s
            e = np.exp(v - v.max())
            out[h] += w * e / e.sum()
    return out


def lv_evidence(grid, spans, s: float = 6.0) -> list[ChordEvidence]:
    """spans = [(beat0, beat1, [(chord, weight), …])]; later spans override earlier ones.
    Both inputs (harmonic, mix) carry the same heads."""
    T = int(np.ceil(grid.cell_t[-1] / LV_HOP)) + 1
    t = (np.arange(T) + 0.5) * LV_HOP
    bt = t / grid.ibi_ref
    heads = {h: np.zeros((T, n)) for h, n in HEADS.items()}
    hp = head_probs([(N, 1.0)], s)
    for h in heads:
        heads[h][:] = hp[h]
    for b0, b1, mix in spans:
        if mix is None:                                  # placeholder, filled by tune_mix
            continue
        m = (bt >= b0) & (bt < b1)
        hp = head_probs(mix, s)
        for h in heads:
            heads[h][m] = hp[h]
    return [ChordEvidence(name=nm, frame_t=t, offset_s=0.0, root=None, bass=None, notes=None,
                          heads={h: v.copy() for h, v in heads.items()}, head_labels={},
                          label_probs=None, labels=None)
            for nm in ("lv@harmonic", "lv@mix")]


def bass_track(grid, notes) -> BassTrack:
    """notes = [(beat0, beat1, pc)] → a BassTrack with conf-0.9 notes (latency-free)."""
    ibi = grid.ibi_ref
    return BassTrack(frame_t=np.zeros(0), salience=np.zeros((0, 12)), energy_db=np.zeros(0),
                     notes=[(b0 * ibi, b1 * ibi, 36 + pc, 0.9, "bass_stem") for b0, b1, pc in notes])


def key_track(grid, k: int) -> KeyTrack:
    return KeyTrack(k % 12, "major", 1.0, np.full(len(grid.beats), k), [])


def vidx(chord) -> int:
    v = candidates()
    return next(i for i, c in enumerate(v.cands) if (c.root, c.quality, c.bass) == tuple(chord))


def cells(b0: float, b1: float) -> tuple[int, int]:
    return int(round(b0 * SUB)), int(round(b1 * SUB))


def advantage(grid, evid, p, a, pch, b0, b1, bass=None) -> float:
    """Mean per-beat bracket advantage B(g, P) − B(g, A) over the cells of [b0, b1) beats."""
    vocab = candidates()
    ia, ip = vidx(a), vidx(pch)
    Cs = np.array(sorted({0, ia, ip}))
    ev = evm.build(grid, evid, None, bass, vocab, Cs, p)
    Bk = evm.bracket(ev, grid, p)
    c0, c1 = cells(b0, b1)
    ka, kp = int(np.flatnonzero(Cs == ia)[0]), int(np.flatnonzero(Cs == ip)[0])
    return float((Bk[c0:c1, kp] - Bk[c0:c1, ka]).mean())


def tune_mix(grid, spans, i, a, pch, target, p, bass=None) -> list:
    """Set the weight of P in spans[i] (mixed with A) so that its advantage is ``target``."""
    b0, b1, _ = spans[i]
    lo, hi = 0.0, 1.0
    for _ in range(30):
        w = 0.5 * (lo + hi)
        spans[i] = (b0, b1, [(pch, w), (a, 1 - w)])
        if advantage(grid, lv_evidence(grid, spans), p, a, pch, b0, b1, bass) < target:
            lo = w
        else:
            hi = w
    spans[i] = (b0, b1, [(pch, hi), (a, 1 - hi)])
    return spans


def run(grid, evid, key, p, bass=None):
    """The §5 orchestration steps 3–9 (no M5 parts): prune → main0 → prune with idiom
    completion → build → detail + main → layers → post."""
    vocab = candidates()
    ev_all = evm.build(grid, evid, None, bass, vocab, np.arange(len(vocab.cands)), p)
    C0 = evm.prune(ev_all, vocab, None, p)
    main0 = hsmm.decode(evm.build(grid, evid, None, bass, vocab, C0, p), grid, key, vocab, p, "main")
    C = evm.prune(ev_all, vocab, main0, p)
    ev = evm.build(grid, evid, None, bass, vocab, C, p)
    detail = hsmm.decode(ev, grid, key, vocab, p, "detail")
    main = hsmm.decode(ev, grid, key, vocab, p, "main")
    segs = layers.build(detail, main, ev, grid, key, vocab, p)
    final = post.finalize(segs, bass, key, grid, vocab, ev, p)
    return SimpleNamespace(vocab=vocab, ev=ev, C=C, detail=detail, main=main, segs=segs, final=final)


def triple(vocab, s):
    c = vocab.cands[s.cand]
    return (c.root, c.quality, c.bass)


def passing(r):
    return [s for s in r.final if s.layer == "passing"]


def mains(r):
    return [s for s in r.final if s.layer == "main"]


# =============================================================================================
# (1) a clear progression decodes exactly, without flicker
# =============================================================================================

def test_clear_progression_exact_no_flicker():
    grid = cal.synthetic_grid(65, n_bars=5, subdiv=SUB)
    prog = [B, E, Gsm, Fs, B]
    spans = [(4 * i, 4 * i + 4, [(ch, 1.0)]) for i, ch in enumerate(prog)]
    p = DecoderParams()
    r = run(grid, lv_evidence(grid, spans), key_track(grid, KEY_B), p)
    for res in (r.detail, r.main):
        assert [triple(r.vocab, s) for s in res.segments] == prog
        assert [(s.c0, s.c1) for s in res.segments] == [(16 * i, 16 * i + 16) for i in range(5)]
    assert passing(r) == []
    assert [triple(r.vocab, s) for s in mains(r)] == prog
    assert all(s.conf > 0.95 for s in mains(r))
    assert len(r.C) <= p.prune_cap


# =============================================================================================
# (2) soft half-beat passing chord at the break-even advantage vs noise blips
# =============================================================================================

def test_soft_passing_chord_recovered_noise_rejected_hui_jia():
    """回家's B → F#/A# → G#m with the half-beat F#/A# at ~0.7 nats/beat over B (default
    params, full priors) is kept; half-beat blips at ~0.2 nats/beat are not."""
    grid = cal.synthetic_grid(65, n_bars=6, subdiv=SUB)
    p = DecoderParams()
    spans = [(0, 4, [(B, 1.0)]), (4, 8, [(Gsm, 1.0)]), (8, 12, [(E, 1.0)]), (12, 16, [(Fs, 1.0)]),
             (16, 24, [(B, 1.0)]),
             (3.5, 4.0, None),            # P: F#/A# over B (A–P–B)
             (5.0, 5.5, None),            # noise: C#m inside G#m (A–P–A, diatonic)
             (9.5, 10.0, None),           # noise: F inside E (A–P–A, chromatic)
             (15.5, 16.0, None)]          # noise: D between F# and B (A–P–B)
    plan = {5: (B, Fs_As, 0.7), 6: (Gsm, Csm, 0.2), 7: (E, F, 0.2), 8: (Fs, (2, "maj", 2), 0.2)}
    for i, (a, pch, target) in plan.items():
        spans = tune_mix(grid, spans, i, a, pch, target, p)
    evid = lv_evidence(grid, spans)
    for i, (a, pch, target) in plan.items():
        assert advantage(grid, evid, p, a, pch, spans[i][0], spans[i][1]) == pytest.approx(target, abs=0.02)

    r = run(grid, evid, key_track(grid, KEY_B), p)
    assert [triple(r.vocab, s) for s in mains(r)] == [B, Gsm, E, Fs, B]
    ps = passing(r)
    hit = [s for s in ps if (s.c0, s.c1) == cells(3.5, 4.0)]
    assert len(hit) == 1
    s = hit[0]
    assert r.vocab.cands[s.cand].cls == r.vocab.cands[vidx(Fs_As)].cls
    # M2 slider fit (confidence only): Standard needs conf ≳ 0.95; a passing chord AT the 0.7-nat
    # break-even is shown from Detailed (the v1 slider showed it from Standard)
    assert s.role == "chord" and s.min_detail is not None and s.min_detail <= 0.8   # shown from Detailed
    assert s.conf > 0.5 and s.salience > 0
    for b0, b1 in [(5.0, 5.5), (9.5, 10.0), (15.5, 16.0)]:
        c0, c1 = cells(b0, b1)
        assert not [x for x in ps if x.c0 < c1 and x.c1 > c0], (b0, b1)


def test_soft_passing_chord_at_break_even_neutral():
    """Structural check (no priors, no idiom): with λ_scale fitted so that the half-beat
    A–P–B break-even is 0.7 nats/beat, a passing G#+ between C and F slightly above it (0.8)
    is recovered, one slightly below it (0.6) and a 0.2 blip are not."""
    p = replace(DecoderParams(), w_prior=0.0)
    p = replace(p, lam_scale=cal.fit_lam_scale(0.7, p))
    grid = cal.synthetic_grid(65, n_bars=6, subdiv=SUB)
    spans = [(0, 4, [(C, 1.0)]), (4, 8, [(F, 1.0)]), (8, 12, [(C, 1.0)]), (12, 16, [(F, 1.0)]),
             (16, 20, [(C, 1.0)]), (20, 24, [(F, 1.0)]),
             (3.5, 4.0, None), (11.5, 12.0, None), (19.5, 20.0, None)]
    plan = {6: 0.8, 7: 0.6, 8: 0.2}
    for i, target in plan.items():
        spans = tune_mix(grid, spans, i, C, Gs_aug, target, p)
    r = run(grid, lv_evidence(grid, spans), key_track(grid, KEY_C), p)
    found = {(s.c0, s.c1): s for s in passing(r)}
    assert cells(3.5, 4.0) in found
    assert r.vocab.cands[found[cells(3.5, 4.0)].cand].cls == r.vocab.cands[vidx(Gs_aug)].cls
    assert cells(11.5, 12.0) not in found and cells(19.5, 20.0) not in found


# =============================================================================================
# (3) A–P–A neighbour: demoted, never deleted
# =============================================================================================

def test_apa_neighbour_demoted_not_deleted():
    grid = cal.synthetic_grid(65, n_bars=4, subdiv=SUB)
    p = DecoderParams()
    spans = [(0, 16, [(B, 1.0)]),
             (5.0, 5.5, [(Gsm, 1.0)]),        # B – G#m – B: not whitelisted → demoted
             (13.0, 13.5, [(E_B, 1.0)])]      # B – E/B – B: pedal neighbour → kept
    r = run(grid, lv_evidence(grid, spans), key_track(grid, KEY_B), p)
    before = {(s.c0, s.c1): s for s in r.segs if s.layer == "passing"}
    after = {(s.c0, s.c1): s for s in passing(r)}
    g, e = cells(5.0, 5.5), cells(13.0, 13.5)
    assert g in before and g in after                       # never deleted
    assert triple(r.vocab, after[g]) == Gsm
    assert after[g].min_detail == pytest.approx(min(1.0, before[g].min_detail + p.demote))
    assert e in before and e in after
    assert r.vocab.cands[after[e].cand].cls == r.vocab.cands[vidx(E_B)].cls
    assert after[e].min_detail == pytest.approx(before[e].min_detail)
    assert [triple(r.vocab, s) for s in mains(r)] == [B]


# =============================================================================================
# (4) bass event: only the bass moves for < 0.3 s
# =============================================================================================

def test_bass_passing_tone_keeps_chord_label():
    grid = cal.synthetic_grid(65, n_bars=3, subdiv=SUB)
    # M2 defaults give the bass less say in ≤ ½-beat segments (w_short 0.3, w_bas 1.7), so a
    # 0.23 s note seen ONLY by the bass tracker is no longer decoded on its own; this test is
    # about the post-processing of such a cell, so it keeps the v1 fusion weights.
    p = replace(DecoderParams(), w_short=1.0, w_src={**DecoderParams().w_src, "bas": 2.0})
    spans = [(0, 8, [(B, 1.0)]), (8, 12, [(E, 1.0)])]       # the upper structure holds B
    bt = bass_track(grid, [(0, 5.25, 11), (5.25, 5.5, 1), (5.5, 8, 11), (8, 12, 4)])   # C# for 0.23 s
    r = run(grid, lv_evidence(grid, spans), key_track(grid, KEY_B), p, bass=bt)
    ev_cells = cells(5.25, 5.5)
    det = [s for s in r.detail.segments if (s.c0, s.c1) == ev_cells]
    assert det and triple(r.vocab, det[0]) == B_Cs           # the decoder saw B/C# for one cell
    be = [s for s in r.final if s.role == "bass_event"]
    assert len(be) == 1
    s = be[0]
    assert (s.c0, s.c1) == ev_cells and s.t1 - s.t0 < 0.3
    assert triple(r.vocab, s) == B_Cs                        # B's label with the slash: B/C#
    assert s.layer == "passing" and s.min_detail == 1.0      # grid at Everything only
    assert post.is_bass_passing_tone(s, grid, p)
    host = [m for m in mains(r) if m.c0 <= s.c0 and m.c1 >= s.c1]
    assert len(host) == 1 and triple(r.vocab, host[0]) == B  # the chord keeps its label
    assert [triple(r.vocab, m) for m in mains(r)] == [B, E]
    assert [x for x in passing(r) if x.role == "chord"] == []


# =============================================================================================
# (5) equivalence class: a passing dim7 gets class-level conf
# =============================================================================================

def test_passing_dim7_class_level_conf():
    """B → C°7 → C#m with the network unsure which of the four dim7 spellings over C it is."""
    grid = cal.synthetic_grid(65, n_bars=3, subdiv=SUB)
    p = DecoderParams()
    spans = [(0, 4, [(B, 1.0)]), (4, 12, [(Csm, 1.0)]),
             (3.5, 4.0, [(d, 0.25) for d in DIM7_ON_C])]
    r = run(grid, lv_evidence(grid, spans), key_track(grid, KEY_B), p)
    c0, c1 = cells(3.5, 4.0)
    hit = [s for s in passing(r) if (s.c0, s.c1) == (c0, c1)]
    assert len(hit) == 1
    s = hit[0]
    cls = r.vocab.cands[vidx(DIM7_ON_C[0])].cls
    assert s.cls == cls
    cols = [int(np.flatnonzero(r.C == vidx(d))[0]) for d in DIM7_ON_C]
    member = r.detail.post[c0:c1][:, cols].mean(axis=0)
    assert s.conf == pytest.approx(member.sum(), abs=1e-6)     # class = Σ members
    assert s.conf > 0.6 and member.max() < 0.8 * s.conf      # not split 4 ways
    assert triple(r.vocab, s) == (0, "dim7", 0)              # named #1°7 → 2m (resolves up)
    assert s.min_detail <= 0.5


# =============================================================================================
# (6) speed and normalisation; exactness on a tiny lattice
# =============================================================================================

def _random_evidence(G: int, K: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    vocab = candidates()
    C = np.r_[0, np.sort(rng.choice(np.arange(1, len(vocab.cands)), K - 1, replace=False))]
    Bk = -3.0 + 0.7 * rng.standard_normal((G, K))
    for blk in range(0, G, 16):
        Bk[blk:blk + 16, rng.integers(K)] = 0.0
    E = 10.0 * 0.25 * Bk
    Es = E + 0.5 * rng.standard_normal((G, K))
    return vocab, Evidence(C=C, E=E, E_short=Es, parts={})


def test_decode_speed_and_marginals():
    G, K = 1008, 250                                      # 63 bars of 16 cells
    grid = cal.synthetic_grid(65, n_bars=G // 16, subdiv=SUB)
    vocab, ev = _random_evidence(G, K)
    key = key_track(grid, KEY_B)
    p = DecoderParams()
    t0 = time.perf_counter()
    res = hsmm.decode(ev, grid, key, vocab, p, "detail")
    dt = time.perf_counter() - t0
    print(f"\ndecode detail {G} cells x {K} candidates (Lmax 64): {dt:.2f} s (prior matrix incl.)")
    t0 = time.perf_counter()
    hsmm.decode(ev, grid, key, vocab, p, "detail")
    print(f"  again with the cached prior matrix: {time.perf_counter() - t0:.2f} s")
    assert dt < 4.0
    assert res.post.shape == (G, K) and res.post_cls.shape == (G, len(vocab.classes))
    np.testing.assert_allclose(res.post.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(res.post_cls.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((res.pb >= 0) & (res.pb <= 1)) and res.pb[0] == 1.0
    assert sum(s.c1 - s.c0 for s in res.segments) == G
    t0 = time.perf_counter()
    m = hsmm.decode(ev, grid, key, vocab, p, "main")
    print(f"  main layer ({G // 4} beats): {time.perf_counter() - t0:.2f} s")
    np.testing.assert_allclose(m.post.sum(axis=1), 1.0, atol=1e-6)


def test_tiny_lattice_matches_brute_force():
    """Viterbi score, cell marginals and boundary probabilities equal exhaustive enumeration."""
    grid = cal.synthetic_grid(65, n_bars=1, subdiv=2)          # 8 cells
    rng = np.random.default_rng(3)
    vocab = candidates()
    C = np.array([vidx(B), vidx(Fs_As), vidx(Gsm)])
    E = rng.normal(0, 2, (8, 3))
    ev = Evidence(C=C, E=E, E_short=E + rng.normal(0, 1, (8, 3)), parts={})
    p = DecoderParams()
    key = key_track(grid, KEY_B)
    res = hsmm.decode(ev, grid, key, vocab, p, "detail")
    lat = hsmm.Lattice(ev, grid, key, vocab, p, "detail")
    G = 8
    scores, post, pb = [], np.zeros((G, 3)), np.zeros(G)
    paths = []
    for cuts in itertools.product([0, 1], repeat=G - 1):
        bounds = [0] + [i + 1 for i, c in enumerate(cuts) if c] + [G]
        segs = list(zip(bounds[:-1], bounds[1:]))
        for labels in itertools.product(range(3), repeat=len(segs)):
            path = [(i, j, k) for (i, j), k in zip(segs, labels)]
            paths.append(path)
            scores.append(lat.path_score(path))
    scores = np.array(scores)
    w = np.exp((scores - scores.max()) / prm.POST_TEMP)      # marginals are tempered (M2), the path is not
    w /= w.sum()
    for path, wp in zip(paths, w):
        for n, (i, j, k) in enumerate(path):
            post[i:j, k] += wp
            if n and path[n - 1][2] != k:
                pb[i] += wp
    pb[0] = 1.0
    assert res.score == pytest.approx(scores.max(), abs=1e-9)
    np.testing.assert_allclose(res.post, post, atol=1e-9)
    np.testing.assert_allclose(res.pb, pb, atol=1e-9)


def test_region_rescan_with_boundaries():
    grid = cal.synthetic_grid(65, n_bars=6, subdiv=SUB)
    spans = [(0, 8, [(B, 1.0)]), (8, 24, [(Gsm, 1.0)])]
    p = DecoderParams()
    key = key_track(grid, KEY_B)
    vocab = candidates()
    evid = lv_evidence(grid, spans)
    C = np.array(sorted({0, vidx(B), vidx(Gsm), vidx(Fs_As), vidx(E)}))
    ev = evm.build(grid, evid, None, None, vocab, C, p)
    # a selection inside bar 2, widened by ±1 bar → bars 1–3 = cells [16, 64) of 96
    r0, r1 = hsmm.rescan_region(grid, 2.0 * grid.ibi_ref * 4, 2.5 * grid.ibi_ref * 4, bars=1)
    assert (r0, r1) == (16, 64)
    res = hsmm.decode(ev, grid, key, vocab, prm.fast_rescan(p), "detail", region=(r0, r1),
                      boundary=(vidx(B), vidx(Gsm)))
    assert res.segments[0].c0 == r0 and res.segments[-1].c1 == r1
    assert [triple(vocab, s) for s in res.segments] == [B, Gsm]
    np.testing.assert_allclose(res.post[r0:r1].sum(axis=1), 1.0, atol=1e-6)
    assert np.all(res.post[:r0] == 0) and np.all(res.post[r1:] == 0)


# =============================================================================================
# Evidence details
# =============================================================================================

def test_pool_is_overlap_weighted_mean():
    tl = evm.timeline(1.0)
    X = np.arange(len(tl), dtype=float)[:, None]
    cell_t = np.array([0.0, 0.1, 0.35, 1.0])
    got = evm.pool(cell_t, tl, X)[:, 0]
    # the step function's exact mean over [a, b): integrate numerically on a fine grid
    fine = np.linspace(0, 1.0, 200001)
    k = np.clip(np.floor(fine / prm.HOP_S + 0.5).astype(int), 0, len(tl) - 1)
    for g in range(3):
        m = (fine >= cell_t[g]) & (fine < cell_t[g + 1])
        assert got[g] == pytest.approx(k[m].mean(), abs=2e-3)


def test_lv_quality_mapping_prefers_the_true_chord():
    """Clean heads for each quality make that quality's candidate the best on its root/bass."""
    vocab = candidates()
    grid = cal.synthetic_grid(65, n_bars=1, subdiv=SUB)
    from nathan_chordify.theory import templates as TT
    for q in TT.QUALITIES:
        if q == "5":
            continue                                   # no lv triad class (scored as maj|min)
        ch = (2, q, 2)
        evid = lv_evidence(grid, [(0, 4, [(ch, 1.0)])])
        same = [i for i, c in enumerate(vocab.cands) if c.root == 2 and c.bass == 2]
        ev = evm.build(grid, evid, None, None, vocab, np.array([0] + same), DecoderParams())
        best = int(ev.C[ev.E[8].argmax()])
        bc = vocab.cands[best]
        assert bc.cls == vocab.cands[vidx(ch)].cls or bc.quality == q, (q, bc.quality)


def test_break_even_table_default_params():
    """Prints the §5.9 break-even table with the default DecoderParams (targets: 0.7 for a
    half-beat A–P–B, 1.3 for a half-beat A–P–A) and the λ_scale that would hit 0.7."""
    p = DecoderParams()
    rows = cal.break_even_table(p)
    print("\nBreak-even δ* (nats/beat), default params, neutral chords (C – G#+ – F, w_prior 0):")
    print(cal.format_table(rows))
    rows_h = cal.break_even_table(p, cal.HUI_JIA, lengths=(0.5,))
    print("\n回家 idioms with full priors (B – F#/A# – G#m; B – E/B – B):")
    print(cal.format_table(rows_h))
    s = cal.fit_lam_scale(0.7, p)
    rows_t = cal.break_even_table(replace(p, lam_scale=s), lengths=(0.5,))
    print(f"\nλ_scale = {s} puts the half-beat A–P–B at 0.7:")
    print(cal.format_table(rows_t))
    half = {(r["pattern"], r["pos"]): r["delta"] for r in rows if r["length"] == 0.5}
    # the default lam_scale (0.76) is fitted to the §5.9 targets
    assert half[("A-P-B", 2)] == pytest.approx(0.7, abs=0.05)
    assert all(abs(d - 1.3) <= 0.2 for (pat, _), d in half.items() if pat == "A-P-A")
    assert all(abs(r["delta"] - 0.7) < 0.05 for r in rows_t if r["pattern"] == "A-P-B")
    assert all(abs(r["delta"] - 1.3) <= 0.2 for r in rows_t if r["pattern"] == "A-P-A")


def test_calibrate_fit_recovers_temperature():
    """NLL fit on synthetic raw scores drawn at a known temperature."""
    rng = np.random.default_rng(1)
    vocab = candidates()
    C = np.array([m[0] for m in vocab.classes[:60]])          # one member per class
    G, K = 4000, len(C)
    T_true = 2.5
    logits = rng.normal(0, 3.0, (G, K))
    pr = np.exp(logits - logits.max(axis=1, keepdims=True))
    pr /= pr.sum(axis=1, keepdims=True)
    gt = np.array([C[rng.choice(K, p=row)] for row in pr])
    cls = [vocab.cands[c].cls for c in C]
    assert len(set(cls)) == len(cls)
    ev = Evidence(C=C, E=np.zeros((G, K)), E_short=np.zeros((G, K)),
                  parts={"aux:raw:lv": logits * T_true})
    T = cal.fit([ev], [gt], vocab=vocab)["lv"]
    assert T == pytest.approx(T_true, rel=0.1)


# =============================================================================================
# More mechanics: rep / HCDF / boundaries, free time, layers, post rules, other sources
# =============================================================================================

def test_tiny_lattice_brute_force_with_rep_hcdf_and_boundaries():
    grid = cal.synthetic_grid(65, n_bars=2, subdiv=2)          # 16 cells; region of 7
    rng = np.random.default_rng(7)
    vocab = candidates()
    C = np.array([vidx(B), vidx(Fs_As), vidx(Gsm)])
    G = 16
    Ev = rng.normal(0, 2, (G, 3))
    ev = Evidence(C=C, E=Ev, E_short=Ev + rng.normal(0, 1, (G, 3)), parts={})
    rep = rng.choice([-1, vocab.cands[C[0]].cls, vocab.cands[C[2]].cls], size=G)
    hcdf = np.array([grid.cell_t[5] + 0.03, grid.cell_t[9] - 0.05])
    p = replace(DecoderParams(), rep_bonus=1.0)
    key = key_track(grid, KEY_B)
    r0, r1 = 4, 11
    left, right = vidx(E), vidx(Gsm)
    res = hsmm.decode(ev, grid, key, vocab, p, "detail", region=(r0, r1), boundary=(left, right),
                      rep=rep, hcdf=hcdf)
    lat = hsmm.Lattice(ev, grid, key, vocab, p, "detail", rep=rep, hcdf=hcdf)
    assert lat.hb[5] == prm.HCDF_BONUS and lat.hb[9] == prm.HCDF_BONUS and lat.hb[6] == 0
    lc, rc = vocab.cands[left], vocab.cands[right]
    n = r1 - r0
    scores, paths = [], []
    for cuts in itertools.product([0, 1], repeat=n - 1):
        bounds = [r0] + [r0 + i + 1 for i, c in enumerate(cuts) if c] + [r1]
        segs = list(zip(bounds[:-1], bounds[1:]))
        for labels in itertools.product(range(3), repeat=len(segs)):
            path = [(i, j, k) for (i, j), k in zip(segs, labels)]
            sc = (lat.trans_cand(lc, lat.cands[path[0][2]], r0) + lat.path_score(path)
                  + lat.trans_cand(lat.cands[path[-1][2]], rc, r1))
            paths.append(path)
            scores.append(sc)
    scores = np.array(scores)
    w = np.exp((scores - scores.max()) / prm.POST_TEMP)      # marginals are tempered (M2), the path is not
    w /= w.sum()
    post_bf = np.zeros((G, 3))
    for path, wp in zip(paths, w):
        for i, j, k in path:
            post_bf[i:j, k] += wp
    assert res.score == pytest.approx(scores.max(), abs=1e-9)
    np.testing.assert_allclose(res.post, post_bf, atol=1e-9)


def test_free_time_units_and_decode():
    grid = cal.synthetic_grid(65, n_bars=2, subdiv=SUB)
    grid.cell_beat = grid.cell_beat.astype(float)
    grid.cell_beat[0:6] = np.nan                                 # a 6-cell rubato intro
    grid.cell_pos[0:6] = 5
    grid.free = [(0.0, float(grid.cell_t[6]))]
    un = hsmm.main_units(grid)
    assert list(un.c0[:4]) == [0, 3, 6, 8]                       # 3 free cells act as a beat
    assert un.pos[0] == 5 and un.pos[3] == 1                     # beat 3 of 4/4 = mid-bar
    np.testing.assert_allclose(evm.cell_beats(grid)[:6], 0.25)
    assert hsmm._lmax(grid, "detail") >= 64
    spans = [(0, 4, [(B, 1.0)]), (4, 8, [(E, 1.0)])]
    r = run(grid, lv_evidence(grid, spans), key_track(grid, KEY_B), DecoderParams())
    assert [triple(r.vocab, s) for s in mains(r)] == [B, E]
    np.testing.assert_allclose(r.main.post.sum(axis=1), 1.0, atol=1e-6)


def _seg(vocab, c0, c1, chord, layer="main", **kw):
    from nathan_chordify.types import Segment
    t = lambda c: c * (60 / 65) / SUB                           # noqa: E731
    i = vidx(chord)
    return Segment(c0=c0, c1=c1, t0=t(c0), t1=t(c1), cand=i, cls=vocab.cands[i].cls, layer=layer, **kw)


def test_post_phantoms_merged_real_passing_chords_survive():
    vocab = candidates()
    grid = cal.synthetic_grid(65, n_bars=4, subdiv=SUB)
    p = DecoderParams()
    key = key_track(grid, KEY_B)
    ev = Evidence(C=np.arange(len(vocab.cands)), E=np.zeros((64, 1)), E_short=np.zeros((64, 1)), parts={})
    Emaj7, Emaj7_B, Bsus4, Baug7 = (4, "maj7", 4), (4, "maj7", 11), (11, "sus4", 11), (11, "+7", 11)
    segs = [_seg(vocab, 0, 16, B), _seg(vocab, 16, 32, Emaj7), _seg(vocab, 32, 48, B), _seg(vocab, 48, 64, Gsm),
            _seg(vocab, 14, 16, Bsus4, "passing", min_detail=0.3),     # phantom → merged into B
            _seg(vocab, 16, 18, Emaj7_B, "passing", min_detail=0.3),   # held bass → merged into Emaj7
            _seg(vocab, 30, 32, Baug7, "passing", min_detail=0.3),     # B+7: G (F##) is new → survives
            _seg(vocab, 46, 48, Fs_As, "passing", min_detail=0.3)]     # F#/A#: survives
    out = post.finalize(segs, None, key, grid, vocab, ev, p)
    ms = [(s.c0, s.c1, triple(vocab, s)) for s in out if s.layer == "main"]
    ps = [(s.c0, s.c1, triple(vocab, s)) for s in out if s.layer == "passing"]
    assert ms == [(0, 16, B), (16, 32, Emaj7), (32, 48, B), (48, 64, Gsm)]
    assert sorted(ps) == [(30, 32, Baug7), (46, 48, Fs_As)]


def test_post_bass_event_relabel_and_seventh_in_bass():
    from nathan_chordify.types import Segment
    vocab = candidates()
    grid = cal.synthetic_grid(65, n_bars=3, subdiv=SUB)
    p = DecoderParams()
    key = key_track(grid, KEY_C)
    G = 48
    tones = np.zeros((G, 12))
    ev = Evidence(C=np.arange(len(vocab.cands)), E=np.zeros((G, 1)), E_short=np.zeros((G, 1)),
                  parts={"aux:lv:tones": tones})
    Em_B, Am = (4, "min", 11), (9, "min", 9)
    # C | Em/B (only the bass moved: C over B) | Am — a 2-beat passing slash chord
    segs = [_seg(vocab, 0, 16, C), _seg(vocab, 16, 48, Am),
            _seg(vocab, 8, 16, Em_B, "passing", min_detail=0.4, conf=0.7)]
    out = post.finalize(segs, None, key, grid, vocab, ev, p)
    be = [s for s in out if s.role == "bass_event"]
    assert len(be) == 1
    s = be[0]
    assert triple(vocab, s) == (0, "maj7", 11)                  # C over B (shown C/B)
    assert not post.seventh_in_upper(s, ev, vocab)
    assert not post.is_bass_passing_tone(s, grid, p)            # 2 beats: a real slash chord
    assert s.min_detail == 0.4 and s.layer == "passing"
    assert vocab.cands[s.alternatives[0][0]].cls == vocab.cands[vidx(Em_B)].cls
    x = Segment(0, 16, 0.0, 1.0, vidx((0, "maj7", 11)), 0, "main")   # a decoded Cmaj7/B
    tones[:16, 11] = 0.2
    assert not post.seventh_in_upper(x, ev, vocab)
    tones[:16, 11] = 0.8
    assert post.seventh_in_upper(x, ev, vocab)


def test_layers_richer_label_promotion_and_alternatives():
    from nathan_chordify.types import DecodeResult, Segment
    vocab = candidates()
    grid = cal.synthetic_grid(65, n_bars=4, subdiv=SUB)
    p = DecoderParams()
    key = key_track(grid, KEY_B)
    Badd9, Gsm7 = (11, "add9", 11), (8, "m7", 8)
    C = np.array(sorted({0, vidx(B), vidx(Badd9), vidx(E), vidx(Csm), vidx(Gsm), vidx(Gsm7)}))
    col = {int(c): k for k, c in enumerate(C)}
    G, K = 64, len(C)
    ct = grid.cell_t

    def dr(spec, post_):
        segs = [Segment(c0, c1, float(ct[c0]), float(ct[c1]), vidx(ch), vocab.cands[vidx(ch)].cls, "main")
                for c0, c1, ch in spec]
        return DecodeResult(segs, post_, hsmm.class_marginals(post_, C, vocab), np.zeros(G), 0.0)

    pm = np.zeros((G, K))
    pm[0:16, col[vidx(B)]] = 1
    pm[16:40, col[vidx(E)]] = 1
    pm[40:64, col[vidx(Gsm)]] = 1
    main = dr([(0, 16, B), (16, 40, E), (40, 64, Gsm)], pm)
    pd = np.zeros((G, K))
    pd[0:4, col[vidx(B)]] = 1
    pd[4:16, col[vidx(Badd9)]] = 1                               # richer, covers 75 % of M
    pd[16:20, col[vidx(E)]] = 1
    pd[20:40, col[vidx(Csm)]] = 0.8                              # 5 beats, clearly C#m → promoted
    pd[20:40, col[vidx(E)]] = 0.2
    pd[40:44, col[vidx(Gsm)]] = 1
    pd[44:64, col[vidx(Gsm7)]] = 0.5                             # 5 beats, G#m7 vs G#m → kept
    pd[44:64, col[vidx(Gsm)]] = 0.45
    pd[44:64, col[vidx(E)]] = 0.05
    detail = dr([(0, 4, B), (4, 16, Badd9), (16, 20, E), (20, 40, Csm), (40, 44, Gsm), (44, 64, Gsm7)], pd)
    ev = Evidence(C=C, E=np.zeros((G, K)), E_short=np.zeros((G, K)), parts={})
    segs = layers.build(detail, main, ev, grid, key, vocab, p)
    got = [(s.c0, s.c1, triple(vocab, s), s.layer) for s in segs]
    assert got == [(0, 16, Badd9, "main"), (16, 20, E, "main"), (20, 40, Csm, "main"), (40, 64, Gsm, "main")]
    assert any(vocab.cands[c].cls == vocab.cands[vidx(Gsm7)].cls for c, _ in segs[-1].alternatives)
    assert segs[2].conf == pytest.approx(0.8)


def test_ace_btc_sym_sources_rank_the_true_chord():
    vocab = candidates()
    grid = cal.synthetic_grid(65, n_bars=1, subdiv=SUB)
    p = DecoderParams()
    T = 200
    t = (np.arange(T) + 0.5) * LV_HOP
    allC = np.arange(len(vocab.cands))
    e_cls = vocab.cands[vidx(E)].cls

    root = np.full((T, 13), 0.01)
    root[:, 4] = 0.9
    bass = root.copy()
    notes = np.full((T, 12), 0.03)
    notes[:, [4, 8, 11]] = 0.95
    ace = ChordEvidence("ace@harmonic", t, 0.0, root, bass, notes, {}, {}, None, None)
    ev = evm.build(grid, [ace], None, None, vocab, allC, p)
    assert vocab.cands[int(ev.C[ev.E[8].argmax()])].cls == e_cls

    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    labels = ["N"] + [f"{r}:{q}" for r in names for q in ("maj", "min", "7", "maj7", "min7")]
    lp = np.full((T, len(labels)), 0.002)
    lp[:, labels.index("E:maj")] = 0.9
    btc = ChordEvidence("btc@harmonic", t, 0.0, None, None, None, {}, {}, lp, labels)
    ev = evm.build(grid, [btc], None, None, vocab, allC, p)
    top = vocab.cands[int(ev.C[ev.E[8].argmax()])]
    assert (top.root, top.quality) == (4, "maj")                # BTC ignores the bass

    x = np.full((16, 12), 0.02)
    x[:, [4, 8, 11]] = 0.95
    ev = evm.build(grid, [], x, None, vocab, allC, p)
    assert vocab.cands[int(ev.C[ev.E[8].argmax()])].pcs == frozenset({4, 8, 11})
    assert evm.sources(ev) == ["sym"]
