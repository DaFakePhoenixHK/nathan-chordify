"""Calibration on D2 (DESIGN §5.9, §13.2): temperatures, reliability/ECE, break-even, offsets.

    python -m eval.calibration --set D2 [--fit] [--write models/calibration/cal-m2-0001.json]

- ``fit_temperatures``: T_s per active source (lv, bas) by class-level NLL against the exact
  D2 labels (``decode.calibrate.fit``), on every cell whose GT label covers ≥ 75 % of it.
  Cells are weighted by their length in beats; ``bas`` cells also by the source's activity
  (bass-note coverage × level, §5.9 M2 notes), so silent cells don't count.
- ``reliability``: top-label reliability diagram and ECE (10 bins, beat-weighted) of the
  class posteriors of the DETAIL lattice (forward–backward) against the GT class per cell.
- ``observed_advantage``: for every GT passing chord P (D2 json "passing") and the chord A before
  it, the mean per-beat bracket advantage B(g, P) − B(g, A) over P's cells — what real lv/bass
  evidence delivers, to be read against the break-even δ* of ``decode.calibrate``.
- ``lv_offset``: the lv adapter's latency (§5.5): median time from each GT chord change to the
  crossing of the old and new chords' lv scores (adapter acceptance: |offset| < 20 ms).

States come from ``eval.engine_eval`` (``prepare`` first).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import chord_metrics as cm
from . import engine_eval as EE

ROOT = Path(__file__).resolve().parents[1]


def _vocab():
    from nathan_chordify.theory.vocab import candidates
    return candidates()


def item_truth(item, st) -> dict:
    iv, lab = cm.load_lab(item.lab)
    return {"iv": iv, "lab": lab, "cells": EE.gt_cells(st.grid, iv, lab)}


def build_raw(st, p):
    """Full-vocabulary evidence of one item with temperatures ``p.temp`` (T = 1 when empty)."""
    from nathan_chordify.decode import evidence as EV
    return EV.build_full(st.grid, st.evid, None, st.bass, _vocab(), p)


# ---------------------------------------------------------------------------------------------
# temperatures
# ---------------------------------------------------------------------------------------------

def fit_temperatures(states: dict, items: dict, p=None) -> dict[str, float]:
    from nathan_chordify.decode import calibrate as CAL
    from nathan_chordify.decode.evidence import cell_beats
    from nathan_chordify.types import DecoderParams
    p = replace(p or DecoderParams(), temp={})
    evs, gts, w_lv, w_bas = [], [], [], []
    for iid, st in states.items():
        ev = build_raw(st, p)
        evs.append(ev)
        gts.append(item_truth(items[iid], st)["cells"])
        u = cell_beats(st.grid)
        w_lv.append(u)
        w_bas.append(u * ev.parts.get("aux:act:bas", np.ones_like(u)))
    out = {}
    out.update(CAL.fit(evs, gts, vocab=_vocab(), sources=["lv"], weights=w_lv))
    out.update(CAL.fit(evs, gts, vocab=_vocab(), sources=["bas"], weights=w_bas))
    return {k: round(v, 4) for k, v in out.items()}


def nll_table(states: dict, items: dict, temps: dict[str, float]) -> list[dict]:
    """Mean class-level NLL per cell (nats) at T = 1 and at the fitted T, per source."""
    from nathan_chordify.decode import calibrate as CAL
    from nathan_chordify.decode.evidence import cell_beats
    from nathan_chordify.types import DecoderParams
    V = _vocab()
    rows = []
    for s in ("lv", "bas"):
        tot = {1.0: 0.0, temps.get(s, 1.0): 0.0}
        wsum = 0.0
        for iid, st in states.items():
            ev = build_raw(st, replace(DecoderParams(), temp={}))
            raw = ev.parts.get(f"aux:raw:{s}")
            if raw is None:
                continue
            cells = item_truth(items[iid], st)["cells"]
            ok = cells >= 0
            if s == "bas":
                bpc = np.array([V.cands[c].bass if c >= 0 else -1 for c in cells])
                ok &= bpc >= 0
                mask = np.zeros(raw.shape, bool)
                mask[np.flatnonzero(ok), bpc[ok]] = True
                w = cell_beats(st.grid) * ev.parts.get("aux:act:bas", 1.0)
            else:
                ccls = np.array([c.cls for c in V.cands])
                gcls = np.array([V.cands[c].cls if c >= 0 else -1 for c in cells])
                mask = ccls[None, :] == gcls[:, None]
                ok &= mask.any(axis=1)
                w = cell_beats(st.grid)
            w = np.asarray(w, dtype=float)
            for T in tot:
                tot[T] += CAL._nll(raw[ok], mask[ok], w[ok], T)
            wsum += float(w[ok].sum())
        rows.append({"source": s, "T_fit": temps.get(s, 1.0),
                     "nll_T1": tot[1.0] / max(wsum, 1e-9), "nll_fit": tot[temps.get(s, 1.0)] / max(wsum, 1e-9),
                     "beats": wsum})
    return rows


# ---------------------------------------------------------------------------------------------
# reliability / ECE (detail lattice)
# ---------------------------------------------------------------------------------------------

def detail_posteriors(st, p):
    """(run, ev, detail DecodeResult) with the same pruning as the pipeline (key0, no refine)."""
    from nathan_chordify import pipeline
    from nathan_chordify.decode import evidence as EV
    from nathan_chordify.decode import hsmm
    run = pipeline.run_from_state(st, p)
    main0 = run.first_main()
    C = EV.prune(run.ev_all, run.vocab, main0, p)
    ev = run.sub(C)
    det = hsmm.decode(ev, st.grid, st.key0, run.vocab, p, "detail", hcdf=run.hcdf)
    return run, ev, det


def reliability(states: dict, items: dict, p, n_bins: int = 10) -> dict:
    from nathan_chordify.decode.evidence import cell_beats
    V = _vocab()
    confs, hits, ws = [], [], []
    for iid, st in states.items():
        _, _, det = detail_posteriors(st, p)
        cells = item_truth(items[iid], st)["cells"]
        ok = cells >= 0
        pc = det.post_cls[ok]
        g = np.array([V.cands[c].cls for c in cells[ok]])
        k = pc.argmax(axis=1)
        confs.append(pc[np.arange(len(k)), k])
        hits.append((k == g).astype(float))
        ws.append(cell_beats(st.grid)[ok])
    conf, hit, w = np.concatenate(confs), np.concatenate(hits), np.concatenate(ws)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1]) if b else (conf >= 0) & (conf <= edges[1])
        if not m.any():
            rows.append({"bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}", "n_beats": 0.0, "conf": None, "acc": None})
            continue
        wb = w[m]
        c, a = float(np.average(conf[m], weights=wb)), float(np.average(hit[m], weights=wb))
        ece += wb.sum() / w.sum() * abs(a - c)
        rows.append({"bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}", "n_beats": float(wb.sum()), "conf": c, "acc": a})
    return {"ece": float(ece), "acc": float(np.average(hit, weights=w)), "mean_conf": float(np.average(conf, weights=w)),
            "bins": rows}


# ---------------------------------------------------------------------------------------------
# observed advantage of real evidence for GT passing chords
# ---------------------------------------------------------------------------------------------

def observed_advantage(states: dict, items: dict, p) -> list[dict]:
    from nathan_chordify.decode.evidence import cell_beats
    from nathan_chordify.theory.harte import parse_harte
    from nathan_chordify.theory.vocab import index_of
    rows = []
    for iid, st in states.items():
        it = items[iid]
        ev = build_raw(st, p)
        Bk = ev.E / (p.kappa_b * cell_beats(st.grid)[:, None])
        segs = it.meta.get("segments") or []
        ct = np.asarray(st.grid.cell_t, dtype=float)
        for k, s in enumerate(segs):
            if not s.get("passing") or k == 0:
                continue
            a, b = segs[k - 1], segs[k + 1] if k + 1 < len(segs) else None
            ip, ia = index_of(*parse_harte(s["label"], strict=False)), index_of(*parse_harte(a["label"], strict=False))
            if ip is None or ia is None:
                continue
            share = (np.clip(np.minimum(ct[1:], s["t1"]) - np.maximum(ct[:-1], s["t0"]), 0, None)
                     / np.maximum(np.diff(ct), 1e-9))
            g = np.flatnonzero(share >= 0.75)
            if not len(g):
                continue
            pattern = "A-P-A" if (b is not None and b["label"] == a["label"]) else "A-P-B"
            rows.append({"item": iid, "idiom": it.meta.get("idiom"), "label": s["label"], "prev": a["label"],
                         "beats": float(s["beats"]), "pattern": pattern,
                         "delta": float(np.mean(Bk[g, ip] - Bk[g, ia]))})
    return rows


def advantage_summary(rows: list[dict], table: dict | None = None) -> list[dict]:
    """Median / quartiles of δ per (pattern, length), and the share above δ* (beat position)."""
    out = []
    for pat in ("A-P-B", "A-P-A"):
        for ln in (0.5, 1.0):
            d = np.array([r["delta"] for r in rows if r["pattern"] == pat and abs(r["beats"] - ln) < 1e-6])
            if not len(d):
                continue
            star = (table or {}).get((pat, ln))
            out.append({"pattern": pat, "length": ln, "n": len(d), "q25": float(np.percentile(d, 25)),
                        "median": float(np.median(d)), "q75": float(np.percentile(d, 75)),
                        "delta_star": star, "share_above": None if star is None else float(np.mean(d > star))})
    return out


# ---------------------------------------------------------------------------------------------
# lv latency
# ---------------------------------------------------------------------------------------------

def lv_offset(states: dict, items: dict, search_s: float = 0.25) -> dict:
    """Median (crossing time − GT change time) of the lv scores of the old vs new chord."""
    from nathan_chordify.decode import evidence as EV
    from nathan_chordify.theory.harte import parse_harte
    from nathan_chordify.theory.vocab import candidates, index_of
    V = candidates()
    A = EV.lv_coefficients(V)
    lags = []
    for iid, st in states.items():
        by = {EV.split_name(e.name)[1]: e for e in st.evid if e.name.startswith("lv@")}
        if not by:
            continue
        t, heads = EV.merge_inputs(by, lambda e: {h: e.heads[h] for h in EV._LV_HEADS})
        cols = EV.lv_columns(heads)
        iv, lab = cm.load_lab(items[iid].lab)
        for k in range(1, len(lab)):
            if "N" in (lab[k - 1], lab[k]):
                continue
            i0, i1 = index_of(*parse_harte(lab[k - 1], strict=False)), index_of(*parse_harte(lab[k], strict=False))
            if i0 is None or i1 is None or i0 == i1:
                continue
            tb = iv[k, 0]
            m = (t >= tb - search_s) & (t <= tb + search_s)
            if m.sum() < 3:
                continue
            d = cols[m] @ (A[i1] - A[i0])
            tt = t[m]
            j = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
            if len(j):
                jj = j[np.argmin(np.abs(tt[j] - tb))]
                x = tt[jj] + (tt[jj + 1] - tt[jj]) * (-d[jj]) / max(d[jj + 1] - d[jj], 1e-9)
                lags.append(x - tb)
    lags = np.array(lags)
    return {"n": int(len(lags)), "median_s": float(np.median(lags)) if len(lags) else float("nan"),
            "iqr_s": [float(np.percentile(lags, 25)), float(np.percentile(lags, 75))] if len(lags) else None}


# ---------------------------------------------------------------------------------------------
# break-even
# ---------------------------------------------------------------------------------------------

def break_even(p) -> dict:
    from nathan_chordify.decode import calibrate as CAL
    neutral = CAL.break_even_table(p)
    hui = CAL.break_even_table(p, CAL.HUI_JIA, lengths=(0.5, 1.0))
    return {"neutral": neutral, "huijia": hui}


def delta_star_lookup(rows: list[dict], pos: int = 2) -> dict:
    """{(pattern, length): δ*} at one metric position (default: P starts on a beat)."""
    return {(r["pattern"], r["length"]): r["delta"] for r in rows if r["pos"] == pos}


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def _fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main(argv: list[str] | None = None) -> int:
    from nathan_chordify.types import DecoderParams
    ap = argparse.ArgumentParser(prog="python -m eval.calibration")
    ap.add_argument("--set", default="D2")
    ap.add_argument("--fit", action="store_true", help="fit T_s (otherwise use the DecoderParams defaults)")
    ap.add_argument("--params", help="JSON DecoderParams overrides")
    ap.add_argument("--out", help="write the results as JSON")
    a = ap.parse_args(argv)
    items = {it.id: it for it in EE._items(a.set)}
    states = EE.load_states(items.values())
    p = DecoderParams()
    if a.params:
        p = replace(p, **json.loads(a.params))
    res: dict = {"n_items": len(states)}
    if a.fit:
        temps = fit_temperatures(states, items, p)
        p = replace(p, temp=temps)
        res["temps"] = temps
        res["nll"] = nll_table(states, items, temps)
    res["temp_used"] = dict(p.temp)
    res["lv_offset"] = lv_offset(states, items)
    rel = reliability(states, items, p)
    res["reliability"] = rel
    be = break_even(p)
    res["break_even"] = be
    adv = observed_advantage(states, items, p)
    res["advantage"] = advantage_summary(adv, delta_star_lookup(be["neutral"]))
    res["advantage_rows"] = adv
    print(f"items: {len(states)}; temperatures used: {p.temp}")
    for r in res.get("nll", []):
        print(f"  NLL {r['source']}: T=1 {r['nll_T1']:.3f} → T={r['T_fit']} {r['nll_fit']:.3f} nats/beat over {r['beats']:.0f} beats")
    print(f"lv offset: {res['lv_offset']}")
    print(f"ECE {rel['ece']:.3f} (acc {rel['acc']:.3f}, mean conf {rel['mean_conf']:.3f})")
    for r in rel["bins"]:
        print(f"  {r['bin']}: beats {r['n_beats']:.0f} conf {_fmt(r['conf'])} acc {_fmt(r['acc'])}")
    from nathan_chordify.decode import calibrate as CAL
    print(CAL.format_table(be["neutral"]))
    print(CAL.format_table(be["huijia"]))
    for r in res["advantage"]:
        print(f"  {r['pattern']} {r['length']} beat: n {r['n']} δ median {r['median']:.2f} [{r['q25']:.2f}, {r['q75']:.2f}]"
              f"  δ* {_fmt(r['delta_star'])}  share above {_fmt(r['share_above'])}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
