"""WP4b: consonance-ACE as a second chord-model source, measured against "without ACE".

    python -m eval.ace_eval prepare [--force]   # ACE evidence (mix + harmonic) for every mini-D2 item and 回家
    python -m eval.ace_eval offset              # ACE (and lv) latency on mini-D2 (§5.5)
    python -m eval.ace_eval fit                 # T_ace by class-level NLL on mini-D2 (§5.9)
    python -m eval.ace_eval eval [--temp-ace T] [--w-ace 1.0 0.5 ...]   # with vs without ACE (§13.6 M5 gate)
    python -m eval.ace_eval all                 # prepare, offset, fit, eval → eval/reports/ace_eval.json

``pipeline.py`` is NOT edited (another work package owns it): ACE evidence is added to the cached
decode states by wrapping ``pipeline.load_state`` (``with_ace``), so ``pipeline.redecode`` — and
through it ``engine_eval.engine_prediction`` and the checklist — decode with lv + ACE + bass
exactly as the pipeline will once it runs the adapter. Both arms of every comparison run in the
same process against the same decoder code and the same cached states.

ACE evidence lands next to the lv evidence: ``eval/data/cache/<key>/evidence/ace@{mix,harmonic}.v1.npz``
for mini-D2, ``cache/yt-Y7boUFbM76k/evidence/…`` for 回家 (its audio is cached; nothing is downloaded).

Metrics (Detailed = Detail 0.8): mini-D2 majmin_inv, sevenths_inv, short-segment (≤ 1 beat) P / R /
F1 at majmin_inv and sevenths_inv, flicker / min, bass accuracy; the 回家 DRAFT checklist and its
false-positive gate. B2 (ACE alone, its own argmax decoding with --chord-min-duration 0.5, from
our overlapped evidence on the mix) is reported for context.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import chord_metrics as cm
from . import engine_eval as EE

ROOT = Path(__file__).resolve().parents[1]
HUIJIA = "yt-Y7boUFbM76k"
REPORT = ROOT / "eval" / "reports" / "ace_eval.json"
DETAIL = 0.8
SUMMARY = (("majmin_inv", ("standard", "majmin_inv")), ("sevenths_inv", ("standard", "sevenths_inv")),
           ("root", ("standard", "root")),
           ("short F1 mm_inv", ("shortseg_1beat", "majmin_inv", "f1")),
           ("short P mm_inv", ("shortseg_1beat", "majmin_inv", "precision")),
           ("short R mm_inv", ("shortseg_1beat", "majmin_inv", "recall")),
           ("short F1 sev_inv", ("shortseg_1beat", "sevenths_inv", "f1")),
           ("flicker/min", ("flicker", "rate")), ("bass acc", ("bass", "acc")),
           ("ooc bass acc", ("ooc_bass", "acc")), ("boundary F 1/4 beat", ("boundary_qbeat", "f")))


def huijia_dir() -> Path:
    from nathan_chordify import config
    return config.CACHE_DIR / HUIJIA


def d2_items():
    return EE._items("D2")


# ---------------------------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------------------------

def inputs(d: Path) -> list[Path]:
    """The adapter inputs of a cache folder: the mix, then the harmonic stem when present."""
    out = [d / "audio.wav"]
    h = d / "stems" / "harmonic.flac"
    if h.exists():
        out.append(h)
    return out


def prepare_dir(d: Path, device: str = "cuda", force: bool = False) -> dict:
    """ACE evidence for one cache folder (cached as evidence/ace@<input>.v<ver>.npz)."""
    from nathan_chordify.models import ace
    from nathan_chordify.models.base import evidence_path
    m = ace.Ace()
    stats = {}
    for w in inputs(d):
        path = evidence_path(d, m.evidence_name(w), m.adapter_version)
        if force and path.exists():
            path.unlink()
        t = time.perf_counter()
        m.run_cached(w, device, d)
        stats[m.evidence_name(w)] = {"wall_s": round(time.perf_counter() - t, 3),
                                     **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.stats.items()}}
    return stats


def ace_evidence(d: Path) -> list:
    from nathan_chordify.models import ace
    from nathan_chordify.models.base import evidence_path, load_evidence
    m = ace.Ace()
    out = []
    for w in inputs(d):
        p = evidence_path(d, m.evidence_name(w), m.adapter_version)
        if p.exists():
            out.append(load_evidence(p))
    return out


@contextlib.contextmanager
def with_ace(enabled: bool = True):
    """While active, ``pipeline.load_state`` appends the folder's ACE evidence to the state
    (``enabled=False``: removes any, i.e. the lv-only baseline)."""
    from nathan_chordify import pipeline
    orig = pipeline.load_state

    def load_state(target):
        st = orig(target)
        evid = [e for e in st.evid if not e.name.startswith("ace@")]
        st.evid = evid + (ace_evidence(st.dir) if enabled else [])
        return st

    pipeline.load_state = load_state
    try:
        yield
    finally:
        pipeline.load_state = orig


def prepare(device: str = "cuda", force: bool = False) -> dict:
    import torch
    items = d2_items()
    t0 = time.perf_counter()
    for n, it in enumerate(items):
        EE.prepare_item(it)                                  # no-op when the decode state exists
        prepare_dir(EE.item_dir(it), device, force)
    d2_s = time.perf_counter() - t0
    # 回家: timed with uncached runs (weights already resident), then cached
    from nathan_chordify.models import ace
    hd = huijia_dir()
    if not (hd / "audio.wav").exists():
        raise FileNotFoundError(f"{hd}/audio.wav missing (never downloaded here: run the pipeline on 回家 first)")
    t = time.perf_counter()
    ace.load(device)
    timing: dict = {"first_load_s": round(time.perf_counter() - t, 2)}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    m = ace.Ace()
    for w in inputs(hd):
        t = time.perf_counter()
        m.run(w, device)
        timing[m.evidence_name(w)] = {"wall_s": round(time.perf_counter() - t, 3),
                                      **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.stats.items()}}
    if torch.cuda.is_available():
        timing["peak_vram_extra_mib"] = round((torch.cuda.max_memory_allocated() - base) / 2**20, 1)
        timing["resident_weights_mib"] = round(base / 2**20, 1)
        timing["max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    prepare_dir(hd, device, force)
    return {"d2_items": len(items), "d2_s": round(d2_s, 1), "huijia": timing}


# ---------------------------------------------------------------------------------------------
# offset (§5.5) and temperature (§5.9)
# ---------------------------------------------------------------------------------------------

def states_with_ace(items) -> dict:
    with with_ace(True):
        return EE.load_states(items)


def source_offset(states: dict, items: dict, model: str = "ace", search_s: float = 0.25) -> dict:
    """Median time from each D2 chord change to the crossing of the old and new chord's raw
    scores of one model source (``calibration.lv_offset`` for any registered frame source)."""
    from nathan_chordify.decode import evidence as EV
    from nathan_chordify.theory.harte import parse_harte
    from nathan_chordify.theory.vocab import candidates, index_of
    V = candidates()
    lags = []
    for iid, st in states.items():
        by = {EV.split_name(e.name)[1]: e for e in st.evid if e.name.startswith(model + "@")}
        if not by:
            continue
        fs = EV.FRAME_SOURCES[model](by, V)
        t, cols, A = np.asarray(fs.t), fs.cols, fs.A
        iv, lab = cm.load_lab(items[iid].lab)
        for k in range(1, len(lab)):
            if "N" in (lab[k - 1], lab[k]):
                continue
            i0 = index_of(*parse_harte(lab[k - 1], strict=False))
            i1 = index_of(*parse_harte(lab[k], strict=False))
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
                lags.append(tt[jj] + (tt[jj + 1] - tt[jj]) * (-d[jj]) / max(d[jj + 1] - d[jj], 1e-9) - tb)
    lags = np.array(lags)
    if not len(lags):
        return {"n": 0}
    return {"n": int(len(lags)), "median_ms": round(1000 * float(np.median(lags)), 1),
            "iqr_ms": [round(1000 * float(np.percentile(lags, q)), 1) for q in (25, 75)],
            "mean_abs_ms": round(1000 * float(np.mean(np.abs(lags))), 1)}


def offsets(items=None) -> dict:
    items = items or d2_items()
    by_id = {it.id: it for it in items}
    st = states_with_ace(items)
    return {"ace": source_offset(st, by_id, "ace"), "lv": source_offset(st, by_id, "lv")}


def fit_temperature(items=None, sources=("ace",)) -> dict:
    """T_s by class-level NLL on mini-D2 (cells ≥ 75 % covered by one GT chord, beat-weighted),
    with the NLL per beat at T = 1 and at the fit."""
    from nathan_chordify.decode import calibrate as CAL
    from nathan_chordify.decode import evidence as EV
    from nathan_chordify.decode.evidence import cell_beats
    from nathan_chordify.theory.vocab import candidates
    from nathan_chordify.types import DecoderParams
    items = items or d2_items()
    by_id = {it.id: it for it in items}
    V = candidates()
    p = replace(DecoderParams(), temp={})
    evs, gts, ws = [], [], []
    for iid, st in states_with_ace(items).items():
        evs.append(EV.build_full(st.grid, st.evid, None, st.bass, V, p))
        iv, lab = cm.load_lab(by_id[iid].lab)
        gts.append(EE.gt_cells(st.grid, iv, lab))
        ws.append(cell_beats(st.grid))
    temps = CAL.fit(evs, gts, vocab=V, sources=list(sources), weights=ws)
    out = {}
    ccls = np.array([c.cls for c in V.cands])
    for s, T in temps.items():
        tot = {1.0: 0.0, T: 0.0}
        wsum = 0.0
        for ev, g, w in zip(evs, gts, ws):
            raw = ev.parts[f"aux:raw:{s}"]
            gcls = np.array([V.cands[c].cls if c >= 0 else -1 for c in g])
            mask = ccls[None, :] == gcls[:, None]
            ok = (g >= 0) & mask.any(axis=1)
            for tt in tot:
                tot[tt] += CAL._nll(raw[ok], mask[ok], w[ok], tt)
            wsum += float(w[ok].sum())
        out[s] = {"T": round(T, 4), "nll_T1": round(tot[1.0] / wsum, 4), "nll_fit": round(tot[T] / wsum, 4),
                  "beats": round(wsum, 1)}
    return out


# ---------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------

def _get(d: dict, path) -> float:
    for k in path:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return float(d) if isinstance(d, (int, float)) else float("nan")


def summary(agg: dict) -> dict:
    return {name: round(_get(agg, path), 4) for name, path in SUMMARY}


def score_d2(params: dict | None, ace_on: bool, items=None, predictor=None) -> tuple[dict, dict]:
    """(summary, aggregate) over mini-D2 at Detailed."""
    from . import run_eval as RE
    items = items or d2_items()
    results = {}
    with with_ace(ace_on):
        for it in items:
            est = predictor(it) if predictor else EE.engine_prediction(it, DETAIL, params)
            ref_iv, ref_lab = cm.load_lab(it.lab)
            results[it.id] = cm.evaluate_all(ref_iv, ref_lab, est[0], est[1], RE.item_beats(it, ref_iv),
                                             RE.ref_tags(it, ref_iv, ref_lab))
    agg = RE.aggregate(results)
    return summary(agg), agg


def score_huijia(params: dict | None, ace_on: bool) -> dict:
    from nathan_chordify import pipeline
    from . import checklist as CL
    with with_ace(ace_on):
        t = time.perf_counter()
        an = pipeline.redecode(huijia_dir(), params or None)
        dt = time.perf_counter() - t
    res = CL.evaluate(HUIJIA, CL.shown_from_analysis(an, DETAIL))
    return {"checklist": round(res["score"], 4), "n": res["n"], "fp_per_min": round(res["fp"]["per_min"], 3),
            "fp": res["fp"]["chords"], "redecode_s": round(dt, 2),
            "rows": [{"n": r["n"], "name": r["name"], "where": r["where"], "score": r["score"],
                      "shown": [p.get("shown", p.get("shown_bass")) for p in r["parts"]]} for r in res["rows"]],
            "n_main": an.stats.n_main, "n_passing": an.stats.n_passing}


# --- B2: ACE alone with its own decoding (context) ----------------------------------------------

def b2_labels(ev, min_dur: float = 0.5) -> tuple[np.ndarray, list[str]]:
    """ACE's own inference decoding (root/bass argmax, notes > 0.5, ACE.mir_evaluation.decode_chord,
    remove_short_chords(min_dur), merge) on our evidence."""
    from nathan_chordify.models import ace
    ace.import_ace()
    from ACE.mir_evaluation import decode_chord, remove_short_chords
    r, b = ev.root.argmax(1), ev.bass.argmax(1)
    nb = ev.notes > 0.5
    memo: dict = {}
    labs = []
    for k in range(len(r)):
        key = (int(r[k]), int(b[k]), nb[k].tobytes())
        if key not in memo:
            try:
                memo[key] = decode_chord(int(r[k]), int(b[k]), ev.notes[k], 0.5)
            except Exception:                      # harte-library rejects a few note sets
                memo[key] = "X"
        labs.append(memo[key])
    hop = float(ev.frame_t[1] - ev.frame_t[0])
    t0 = np.asarray(ev.frame_t) - hop / 2
    iv, lab = [], []
    for k, x in enumerate(labs):
        if lab and lab[-1] == x:
            iv[-1][1] = t0[k] + hop
        else:
            iv.append([max(0.0, t0[k]), t0[k] + hop])
            lab.append(x)
    iv2, lab2 = remove_short_chords([tuple(v) for v in iv], lab, min_duration=min_dur)
    out_iv, out_lab = [], []
    for (a, e), x in zip(iv2, lab2):
        if out_lab and out_lab[-1] == x:
            out_iv[-1][1] = e
        else:
            out_iv.append([a, e])
            out_lab.append(x)
    return np.array(out_iv, dtype=float).reshape(-1, 2), out_lab


def b2_prediction(item):
    from nathan_chordify.models import ace
    from nathan_chordify.models.base import evidence_path, load_evidence
    d = EE.item_dir(item)
    return b2_labels(load_evidence(evidence_path(d, "ace@mix", ace.Ace.adapter_version)))


def b2_huijia() -> dict:
    from nathan_chordify.models import ace
    from nathan_chordify.models.base import evidence_path, load_evidence
    from . import checklist as CL
    iv, lab = b2_labels(load_evidence(evidence_path(huijia_dir(), "ace@mix", ace.Ace.adapter_version)))
    res = CL.evaluate(HUIJIA, CL.shown_from_lab(iv, lab))
    return {"checklist": round(res["score"], 4), "fp_per_min": round(res["fp"]["per_min"], 3)}


# ---------------------------------------------------------------------------------------------
# the comparison and the gate
# ---------------------------------------------------------------------------------------------

def base_params(temp_ace: float | None, w_ace: float | None = None) -> dict:
    from nathan_chordify.types import DecoderParams
    p = DecoderParams()
    temp = dict(p.temp)
    if temp_ace is not None:
        temp["ace"] = float(temp_ace)
    w = dict(p.w_src)
    if w_ace is not None:
        w["ace"] = float(w_ace)
    return {"temp": temp, "w_src": w}


def gate(base: dict, with_ace_: dict) -> dict:
    """§13.6 M5 component gate on mini-D2 (no D1 yet, so no LOSO): keep only if short-segment F1
    rises by ≥ 0.02 with no majmin_inv loss."""
    d_f1 = with_ace_["short F1 mm_inv"] - base["short F1 mm_inv"]
    d_mm = with_ace_["majmin_inv"] - base["majmin_inv"]
    return {"d_short_f1": round(d_f1, 4), "d_majmin_inv": round(d_mm, 4),
            "pass": bool(d_f1 >= 0.02 - 1e-9 and d_mm >= -1e-9)}


def evaluate(temp_ace: float | None, w_list: list[float], with_b2: bool = True) -> dict:
    items = d2_items()
    out: dict = {"detail": DETAIL, "temp_ace": temp_ace, "d2": {}, "huijia": {}}
    p0 = base_params(temp_ace)
    t = time.perf_counter()
    out["d2"]["without_ace"], _ = score_d2(p0, False, items)
    out["huijia"]["without_ace"] = score_huijia(p0, False)
    print(f"without ACE: {out['d2']['without_ace']}  回家 {out['huijia']['without_ace']['checklist']} "
          f"({time.perf_counter() - t:.0f} s)", flush=True)
    for w in w_list:
        p = base_params(temp_ace, w)
        name = f"with_ace_w{w:g}"
        t = time.perf_counter()
        out["d2"][name], agg = score_d2(p, True, items)
        out["huijia"][name] = score_huijia(p, True)
        out["d2"][name + "_idioms"] = {k: round(v["recall"]["majmin_inv"], 3) for k, v in agg["idioms"].items()}
        out["gate_" + name] = gate(out["d2"]["without_ace"], out["d2"][name])
        print(f"{name}: {out['d2'][name]}  回家 {out['huijia'][name]['checklist']} "
              f"fp {out['huijia'][name]['fp_per_min']}/min  gate {out['gate_' + name]} "
              f"({time.perf_counter() - t:.0f} s)", flush=True)
    _, agg0 = score_d2(p0, False, items)
    out["d2"]["without_ace_idioms"] = {k: round(v["recall"]["majmin_inv"], 3) for k, v in agg0["idioms"].items()}
    if with_b2:
        out["d2"]["B2_ace_alone"], _ = score_d2(None, False, items, predictor=b2_prediction)
        out["huijia"]["B2_ace_alone"] = b2_huijia()
        print(f"B2 (ACE alone): {out['d2']['B2_ace_alone']}  回家 {out['huijia']['B2_ace_alone']}", flush=True)
    return out


def _write(res: dict) -> None:
    old = json.loads(REPORT.read_text("utf-8")) if REPORT.exists() else {}
    old.update(res)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(old, indent=1, ensure_ascii=False, default=str), "utf-8")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m eval.ace_eval")
    ap.add_argument("cmd", choices=["prepare", "offset", "fit", "eval", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="re-run ACE even when evidence is cached")
    ap.add_argument("--temp-ace", type=float, help="T_ace (default: DecoderParams().temp['ace'], else the fit)")
    ap.add_argument("--w-ace", type=float, nargs="+", default=[1.0], help="ACE source weight(s) to evaluate")
    ap.add_argument("--no-b2", action="store_true")
    a = ap.parse_args(argv)
    res: dict = {}
    if a.cmd in ("prepare", "all"):
        res["prepare"] = prepare(a.device, a.force)
        print(json.dumps(res["prepare"], indent=1))
    if a.cmd in ("offset", "all"):
        res["offset"] = offsets()
        print("offsets:", res["offset"])
    if a.cmd in ("fit", "all"):
        res["fit"] = fit_temperature()
        print("T fit:", res["fit"])
    if a.cmd in ("eval", "all"):
        from nathan_chordify.types import DecoderParams
        T = a.temp_ace if a.temp_ace is not None else DecoderParams().temp.get("ace")
        if T is None:
            T = (res.get("fit") or fit_temperature())["ace"]["T"]
        res["eval"] = evaluate(T, a.w_ace, not a.no_b2)
    _write(res)
    print(f"→ {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
