"""Symbolic chroma from the piano transcription (DESIGN §5.7, WP5, M5).

Per cell g = [a, b), Δg = b − a, for the notes at or above the left-hand split (MIDI 55), with
v_n = velocity / 127 and off'_n = offset_pedal:

    ch[g, pc]  = Σ_{n: pc(n) = pc} v_n · (1/Δg) ∫_{max(a, on_n)}^{min(b, off'_n)} exp(−(t − on_n)/τ) dt
               + β · v_n · 1[on_n ∈ [a − 0.05, b)]                           τ = 0.4 s, β = 0.5
    x[g, pc]   = 1 − exp(−ch[g, pc] / θ)                                      θ = 0.25 (saturating presence)
    dch[g, pc] = max(0, ch[g, pc] − α · ch[g − 1, pc])                       α = 0.7 (onset feature)

``x`` feeds the ``sym`` evidence source (§5.9) and the "added tone is supported" tests of §5.12
steps 3 and 5 (``aux:sym:x``); ``dch`` is the new-onset feature of the phantom rule (§5.12 step 2).
The notes below the split are the left hand: they go to the bass tracker (§5.6), not here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import NoteEvents, RhythmGrid


@dataclass(frozen=True)
class SymParams:
    tau: float = 0.4              # s, decay of a note's weight after its onset
    beta: float = 0.5             # onset bonus (× velocity) for the cell the onset falls in
    theta: float = 0.25           # presence saturation
    alpha: float = 0.7            # dch = max(0, ch − α·ch_prev)
    onset_pre_s: float = 0.05     # an onset up to 50 ms before a cell still counts for it
    split_midi: int = 55          # pitch < split → left hand (bass, §5.6); ≥ split → symbolic chroma


def cell_chroma(notes: NoteEvents | None, grid: RhythmGrid, p: SymParams = SymParams()) -> tuple[np.ndarray, np.ndarray]:
    """(x [G, 12] saturating presence, dch [G, 12] onset feature) on the cells of ``grid``."""
    ct = np.asarray(grid.cell_t, dtype=float)
    G = len(ct) - 1
    ch = raw_chroma(notes, ct, p)
    x = 1.0 - np.exp(-ch / max(p.theta, 1e-9))
    prev = np.vstack([np.zeros((1, 12)), ch[:-1]]) if G else ch
    dch = np.maximum(0.0, ch - p.alpha * prev)
    return x, dch


def raw_chroma(notes: NoteEvents | None, cell_t: np.ndarray, p: SymParams = SymParams()) -> np.ndarray:
    """ch [G, 12] of §5.7 (before saturation) on cell edges ``cell_t``."""
    ct = np.asarray(cell_t, dtype=float)
    G = max(len(ct) - 1, 0)
    ch = np.zeros((G, 12))
    if notes is None or G == 0 or not len(notes.onset):
        return ch
    a, b = ct[:-1], ct[1:]
    dg = np.maximum(b - a, 1e-9)
    on = np.asarray(notes.onset, dtype=float)
    off = np.maximum(np.asarray(notes.offset_pedal, dtype=float), on)
    v = np.asarray(notes.velocity, dtype=float) / 127.0
    pc = np.asarray(notes.pitch).astype(int) % 12
    upper = np.asarray(notes.pitch) >= p.split_midi
    tau = max(p.tau, 1e-9)
    for n in np.flatnonzero(upper):
        # decaying sustain over the cells overlapping [on, off')
        g0 = max(0, int(np.searchsorted(ct, on[n], side="right")) - 1)
        g1 = min(G, int(np.searchsorted(ct, off[n], side="left")))
        if g1 > g0:
            lo = np.maximum(a[g0:g1], on[n])
            hi = np.minimum(b[g0:g1], off[n])
            seg = np.where(hi > lo, tau * (np.exp(-(lo - on[n]) / tau) - np.exp(-(hi - on[n]) / tau)), 0.0)
            ch[g0:g1, pc[n]] += v[n] * seg / dg[g0:g1]
        # onset term: every cell with on ∈ [a − 0.05, b)
        k0 = int(np.searchsorted(b, on[n], side="right"))                  # first cell with b > on
        k1 = int(np.searchsorted(a, on[n] + p.onset_pre_s, side="right"))  # cells with a ≤ on + 0.05
        if k1 > k0:
            ch[k0:k1, pc[n]] += p.beta * v[n]
    return ch

