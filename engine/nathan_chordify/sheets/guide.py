"""Use an aligned sheet to refine the detected chords.

Two modes (the server applies either as ONE undoable edit group, like a GT chart import):

- **refine** (default): a chart-guided re-decode from the cached decode state. The sheet is a
  prior, not the truth: at every cell under an aligned sheet chord, the candidates of the sheet
  chord's *family* (same root, triad type and bass) get a bonus of ``W_SHEET`` × κ_b per beat
  (``W_SHEET_ROOT`` for the same root with another triad / bass) in the fused evidence, and the
  decoder runs as usual (main layer + the passing layer the engine already makes). The sheet
  wins where the audio is ambiguous (colour chords, slash basses); a clearly different audio
  chord still wins, and such a sheet chord is FLAGGED (detected kept) instead of applied. The
  bonus stops ``EDGE_BEATS`` short of each sheet chord's edges so the audio places the boundary.
  Songs without a decode state fall back to ``refine_from_analysis``: the sheet replaces a
  detected chord only where the detection was unsure (``conf`` below ``LOW_CONF``).
- **sheet**: the aligned sheet chords as written (a GT-style import).

``diff_ops`` turns a target main-layer timeline into ``set_chord`` range ops only where it differs
from the current one, so unchanged chords keep their ids and the edit log stays small.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import schema
from ..theory import harte as H
from ..theory import simplify as SM
from .align import family

W_SHEET = 0.3            # bonus per beat (× κ_b nats) for the sheet chord's family: moderate, see the module doc
W_SHEET_ROOT = 0.15      # the same root with another triad type / bass
EDGE_BEATS = 0.25        # no bonus this close to a sheet chord's edges
FLAG_MIN_SHARE = 0.5     # a sheet chord whose family lost over this share of its span is flagged
LOW_CONF = 0.6           # fallback mode: a detected chord below this confidence yields to the sheet
MIN_PIECE_S = 0.12       # shorter differences are ignored (boundary jitter)

Chord = tuple[float, float, str]          # (t0, t1, harte)


@dataclass
class Refined:
    main: list[Chord]                      # the refined main-layer timeline
    flagged: list[dict]                    # [{t0, t1, detected, sheet}] (Harte; the server spells them)
    method: str                            # "state" | "analysis"
    analysis: schema.Analysis | None = None
    display: schema.Display | None = None
    n_sheet: int = 0
    n_agree: int = 0
    notes: list[str] = field(default_factory=list)


def _triple(h: str) -> tuple[int, str, int]:
    return H.parse_harte(h, strict=False)


def same_family(a: str, b: str) -> bool:
    return family(a) == family(b)


def same_chord(a: str, b: str) -> bool:
    return _triple(a) == _triple(b)


# ---------------------------------------------------------------------------------- the bonus

def sheet_bonus(grid, vocab, chords: list[Chord], kappa_b: float, w_sheet: float = W_SHEET,
                w_root: float = W_SHEET_ROOT, edge_beats: float = EDGE_BEATS) -> np.ndarray:
    """[G, V] evidence bonus (nats) for the aligned sheet chords: w_sheet·κ_b per beat for the
    sheet chord's family, w_root·κ_b for the same root, 0 near the chord's edges and for N."""
    from .align import nearest_family
    from ..decode.evidence import cell_beats
    ct = np.asarray(grid.cell_t, dtype=float)
    G = len(ct) - 1
    V = len(vocab.cands)
    u = cell_beats(grid)
    roots = np.array([c.root for c in vocab.cands])
    bonus = np.zeros((G, V))
    ibi = float(grid.ibi_ref) if grid.ibi_ref else 0.5
    for t0, t1, h in chords:
        r, q, b = _triple(h)
        if r < 0 or q == "N":
            continue
        pad = min(edge_beats * ibi, 0.2 * (t1 - t0))
        a, z = t0 + pad, t1 - pad
        if z <= a:
            continue
        g0 = int(np.searchsorted(ct, a, side="right") - 1)
        g1 = int(np.searchsorted(ct, z, side="left"))
        if g1 <= g0:
            continue
        g0, g1 = max(g0, 0), min(g1, G)
        fam = nearest_family(family(h))
        same_root = np.flatnonzero(roots == r)
        for g in range(g0, g1):
            ov = (min(ct[g + 1], z) - max(ct[g], a)) / max(ct[g + 1] - ct[g], 1e-6)
            if ov <= 0:
                continue
            w = kappa_b * u[g] * ov
            bonus[g, same_root] = np.maximum(bonus[g, same_root], w_root * w)
            bonus[g, fam] = np.maximum(bonus[g, fam], w_sheet * w)
    return bonus


# ---------------------------------------------------------------------------- comparing layers

def main_timeline(a: schema.Analysis) -> list[Chord]:
    return [(c.start, c.end, c.harte) for c in sorted(a.chords, key=lambda c: c.start) if c.layer == "main"]


def _overlaps(timeline: list[Chord], t0: float, t1: float) -> list[tuple[float, float, str]]:
    out = []
    for a, b, h in timeline:
        lo, hi = max(a, t0), min(b, t1)
        if hi - lo > 1e-6:
            out.append((lo, hi, h))
    return out


def flags(sheet: list[Chord], main: list[Chord], min_share: float = FLAG_MIN_SHARE) -> tuple[list[dict], int]:
    """Sheet chords whose family the decoded main layer rejected over ≥ ``min_share`` of their span
    (→ {t0, t1, detected, sheet}), and the number of sheet chords the main layer agrees with."""
    flagged: list[dict] = []
    agree = 0
    for t0, t1, h in sheet:
        if _triple(h)[0] < 0:
            continue
        pieces = _overlaps(main, t0, t1)
        span = max(t1 - t0, 1e-6)
        bad = [(a, b, d) for a, b, d in pieces if not same_family(d, h)]
        share = sum(b - a for a, b, _ in bad) / span
        if share >= min_share and bad:
            det = max(bad, key=lambda x: x[1] - x[0])[2]
            flagged.append({"t0": round(t0, 3), "t1": round(t1, 3), "detected": det, "sheet": h})
        elif pieces:
            agree += 1
    return flagged, agree


# ------------------------------------------------------------------------------------ refine

def refine_state(st, chords: list[Chord], params=None, w_sheet: float = W_SHEET, w_root: float = W_SHEET_ROOT,
                 analysis: schema.Analysis | None = None) -> Refined:
    """Chart-guided re-decode from a decode state (seconds, no GPU)."""
    from .. import pipeline
    t = time.perf_counter()
    p = pipeline.params_from(params)
    run = pipeline.run_from_state(st, p)
    bonus = sheet_bonus(run.grid, run.vocab, chords, p.kappa_b, w_sheet, w_root)
    run.ev_all.E = run.ev_all.E + bonus
    run.ev_all.E_short = run.ev_all.E_short + bonus
    main0 = run.first_main()
    segs, ev, key = pipeline._finish(run, main0)
    timing = {"refine": round(time.perf_counter() - t, 2)}
    new_a, display = pipeline._assemble(st, p, st.meta.get("mode", "full"), run, segs, ev, key, timing)
    main = main_timeline(new_a)
    flagged, agree = flags(chords, main)
    return Refined(main, flagged, "state", new_a, display, len([c for c in chords if _triple(c[2])[0] >= 0]), agree)


def refine_from_analysis(a: schema.Analysis, chords: list[Chord], low_conf: float = LOW_CONF) -> Refined:
    """Without a decode state: the sheet replaces a detected main chord only where the detection
    was unsure (conf < low_conf) and disagrees; confident disagreements are flagged."""
    cur = [(c.start, c.end, c.harte, c.conf) for c in sorted(a.chords, key=lambda c: c.start) if c.layer == "main"]
    main: list[Chord] = []
    flagged: list[dict] = []
    agree = 0
    changes: list[Chord] = []
    for t0, t1, h in chords:
        if _triple(h)[0] < 0:
            continue
        pieces = [(max(s, t0), min(e, t1), d, conf) for s, e, d, conf in cur if min(e, t1) - max(s, t0) > 1e-6]
        bad = [(s, e, d, conf) for s, e, d, conf in pieces if not same_family(d, h)]
        span = max(t1 - t0, 1e-6)
        bad_share = sum(e - s for s, e, _, _ in bad) / span
        if not bad:
            agree += 1
            continue
        sure = [x for x in bad if x[3] >= low_conf]
        if sum(e - s for s, e, _, _ in sure) / span >= FLAG_MIN_SHARE:
            det = max(sure, key=lambda x: x[1] - x[0])[2]
            flagged.append({"t0": round(t0, 3), "t1": round(t1, 3), "detected": det, "sheet": h})
            continue
        for s, e, d, conf in bad:
            if conf < low_conf and e - s >= MIN_PIECE_S:
                changes.append((s, e, h))
        if bad_share < FLAG_MIN_SHARE:
            agree += 1
    # the refined timeline = the current one with the changes carved in
    main = [(s, e, d) for s, e, d, _ in cur]
    for s, e, h in changes:
        main = _carve(main, s, e, h)
    n_sheet = len([c for c in chords if _triple(c[2])[0] >= 0])
    return Refined(main, flagged, "analysis", None, None, n_sheet, agree, notes=[f"{len(changes)} unsure chords took the sheet's chord"])


def _carve(timeline: list[Chord], t0: float, t1: float, h: str) -> list[Chord]:
    out: list[Chord] = []
    for a, b, d in timeline:
        if b <= t0 + 1e-6 or a >= t1 - 1e-6:
            out.append((a, b, d))
            continue
        if a < t0 - 1e-6:
            out.append((a, t0, d))
        if b > t1 + 1e-6:
            out.append((t1, b, d))
    out.append((t0, t1, h))
    return sorted(out)


# --------------------------------------------------------------------------------------- ops

def diff_ops(current: list[Chord], target: list[Chord], group: str, prefix: str, conf: float = 1.0,
             min_piece_s: float = MIN_PIECE_S) -> tuple[list[dict], int, int]:
    """``set_chord`` range ops (one group) where ``target`` differs from ``current`` (both main
    timelines) → (ops, n_changed pieces, n_kept target chords)."""
    ops: list[dict] = []
    pieces: list[Chord] = []
    kept = 0
    for t0, t1, h in target:
        if _triple(h)[0] < 0 and h != "N":
            continue
        diffs = [(a, b) for a, b, d in _overlaps(current, t0, t1) if not same_chord(d, h)]
        covered = sum(b - a for a, b, _ in _overlaps(current, t0, t1))
        if covered < (t1 - t0) - 1e-6:                            # the current layer has a hole here
            holes = _holes(current, t0, t1)
            diffs += holes
        if not diffs:
            kept += 1
            continue
        merged: list[list[float]] = []
        for a, b in sorted(diffs):
            if merged and a <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        emitted = False
        for a, b in merged:
            if b - a >= min_piece_s:
                pieces.append((a, b, h))
                emitted = True
        if not emitted:
            kept += 1                                                 # only boundary jitter: nothing to change
    n = 0
    for a, b, h in pieces:
        ops.append({"id": f"{prefix}.{n:04d}", "kind": "set_chord",
                    "target": {"t0": round(a, 4), "t1": round(b, 4), "mode": "range", "layer": "main"},
                    "value": {"harte": h, "group": group, "conf": conf}})
        n += 1
    return ops, len(pieces), kept


def _holes(current: list[Chord], t0: float, t1: float) -> list[tuple[float, float]]:
    out = []
    cur = t0
    for a, b, _ in sorted(_overlaps(current, t0, t1)):
        if a > cur + 1e-6:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < t1 - 1e-6:
        out.append((cur, t1))
    return out


def sheet_ops(chords: list[Chord], group: str, prefix: str, conf: float = 1.0) -> list[dict]:
    """The aligned sheet chords as written: one ``set_chord`` range op each."""
    ops = []
    for n, (t0, t1, h) in enumerate(chords):
        if _triple(h)[0] < 0 and h != "N":
            continue
        ops.append({"id": f"{prefix}.{n:04d}", "kind": "set_chord",
                    "target": {"t0": round(t0, 4), "t1": round(t1, 4), "mode": "range", "layer": "main"},
                    "value": {"harte": h, "group": group, "conf": conf}})
    return ops


def section_ops(sections: list[tuple[float, str]], group: str, prefix: str) -> list[dict]:
    return [{"id": f"{prefix}.s{n:03d}", "kind": "set_section", "target": {"t0": round(t, 4)},
             "value": {"label": label, "group": group}} for n, (t, label) in enumerate(sections)]


def full_label(harte: str, key: int, next_root: int | None = None) -> str:
    """The full-level label at transpose 0 (as display.variants["0:auto"] spells it)."""
    r, q, b = _triple(harte)
    if r < 0 or q == "N":
        return "N"
    return SM.label(r, q, b, key, level="full", next_root=next_root)
