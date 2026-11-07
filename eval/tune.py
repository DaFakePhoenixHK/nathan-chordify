"""Decoder tuning on mini-D2 (DESIGN §13.5, M2) with leave-one-idiom-out validation.

    python -m eval.tune --set D2 --samples 64 --workers 6 --out eval/reports/tune_m2.json

What is tuned: λ_scale, w_bas, the PT bonus, w_prior and w_short (Sobol samples, then coordinate
descent on the best), and the slider mapping (a, b, c) of §5.11, fitted per configuration on
the passing candidates labelled hit/miss by the short-segment matcher (majmin_inv). κ_b stays
fixed and the temperatures T_s come from ``eval.calibration`` (fitted before any tuning).

Objective per item set (§13.5, every metric at Detailed = 0.8 unless noted):
    J = 0.4·sevenths_inv + 0.4·shortseg_F1(majmin_inv, ≤ 1 beat) + 0.2·bass_acc
        − 0.02·max(0, flicker/min − 2) − Σ_{s ∈ 0.5, 0.8, 1.0} max(0, prec*(s) − 0.05 − prec(s))
with prec* = 0.85 / 0.70 / 0.50 the passing-chord precision targets per Detail stop.

Validation: leave-one-idiom-out — for each of the 12 idioms, the configuration with the best J
on the other 11 is scored on the held-out idiom; the report gives the pooled held-out metrics
next to the in-sample ones. There is no real annotated D1 yet: everything here is tuned on
synthetic audio and may not transfer (see eval/reports/M2.md).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from . import chord_metrics as cm

ROOT = Path(__file__).resolve().parents[1]
DETAILS = (0.0, 0.5, 0.8, 1.0)
PREC_TARGET = {0.5: 0.85, 0.8: 0.70, 1.0: 0.50}
LOGIT = {0.5: 0.0, 0.8: math.log(0.8 / 0.2)}
WILSON_Z = 0.0             # slider fit: z of the precision lower bound (0 = point estimate; 1.0 was tried, worse LOIO)
SLIDER_USE_SALIENCE = False   # on mini-D2 salience does not rank hits (AUC 0.55, sign unstable across folds)


# ---------------------------------------------------------------------------------------------
# decoded records (no JSON assembly: labels in canonical Harte, which is all mir_eval needs)
# ---------------------------------------------------------------------------------------------

@dataclass
class Rec:
    t0: float
    t1: float
    harte: str
    passing: bool
    conf: float
    sal: float
    demoted: bool
    fixed: float | None      # min_detail that does not depend on the slider (bass passing tones)


def records(segs, vocab, p) -> list[Rec]:
    from nathan_chordify.decode import layers
    from nathan_chordify.theory.harte import to_harte
    out = []
    for s in segs:
        c = vocab.cands[s.cand]
        lab = to_harte(c.root, c.quality, c.bass)
        if s.layer != "passing":
            out.append(Rec(s.t0, s.t1, lab, False, s.conf, 0.0, False, None))
            continue
        sal = float(s.salience or 0.0)
        md0 = layers.min_detail(s.conf, sal, p)
        # post.finalize step 1: a bass passing tone is a bass_event pinned at min_detail 1.0
        fixed = 1.0 if (s.role == "bass_event" and s.min_detail is not None and s.min_detail >= 1.0 - 1e-9) else None
        demoted = fixed is None and s.min_detail is not None and s.min_detail >= md0 + 0.29
        out.append(Rec(s.t0, s.t1, lab, True, float(s.conf), sal, bool(demoted), fixed))
    return out


def min_detail(r: Rec, slider, demote: float = 0.3) -> float:
    if r.fixed is not None:
        return r.fixed
    a, b, c = slider
    q = min(max(r.conf, 1e-4), 1 - 1e-4)
    z = a * math.log(q / (1 - q)) + b * r.sal + c
    md = min(max(1.0 / (1.0 + math.exp(-max(min(z, 50), -50))), 0.05), 1.0)
    return min(1.0, md + demote) if r.demoted else md


def flatten(recs: list[Rec], detail: float, slider) -> tuple[np.ndarray, list[str]]:
    """export.flatten on records: main chords with the passing chords visible at ``detail``."""
    mains = sorted((r for r in recs if not r.passing), key=lambda r: r.t0)
    over = []
    if detail > 0:
        for o in sorted((r for r in recs if r.passing and detail >= min_detail(r, slider)), key=lambda r: r.t0):
            if not over or o.t0 >= over[-1][1] - 1e-6:
                over.append((o.t0, o.t1, o.harte))
    out = list(over)
    for m in mains:
        cur = m.t0
        for o0, o1, _ in over:
            if o1 <= cur + 1e-6 or o0 >= m.t1 - 1e-6:
                continue
            if o0 > cur + 1e-6:
                out.append((cur, o0, m.harte))
            cur = max(cur, o1)
        if cur < m.t1 - 1e-6:
            out.append((cur, m.t1, m.harte))
    out.sort()
    merged = []
    for t0, t1, h in out:
        if merged and merged[-1][2] == h and abs(merged[-1][1] - t0) < 1e-3:
            merged[-1] = (merged[-1][0], t1, h)
        else:
            merged.append((t0, t1, h))
    iv = np.array([[a, b] for a, b, _ in merged], dtype=float).reshape(-1, 2)
    return iv, [h for _, _, h in merged]


# ---------------------------------------------------------------------------------------------
# per-item work (runs in worker processes)
# ---------------------------------------------------------------------------------------------

_W: dict = {}


def _init(set_name: str):
    from . import engine_eval as EE
    items = {it.id: it for it in EE._items(set_name)}
    _W["items"] = items
    _W["states"] = EE.load_states(items.values())
    _W["truth"] = {}
    for iid, it in items.items():
        iv, lab = cm.load_lab(it.lab)
        beats = cm.load_beats(it.beats) if it.beats else None
        ref = cm.segments(*cm._fill_gaps(iv, lab))
        _W["truth"][iid] = {"iv": iv, "lab": lab, "beats": beats, "ref": ref,
                            "ref_tol": cm.onset_tolerance(ref.start, beats)}


def decode_item(args) -> tuple[str, list[Rec], list[tuple[float, float, bool, bool, float | None]], float]:
    """(item id, records, passing candidates (conf, sal, demoted, hit, fixed), seconds)."""
    iid, pdict = args
    from nathan_chordify import pipeline
    from nathan_chordify.types import DecoderParams
    t = time.perf_counter()
    p = DecoderParams(**pdict)
    st = _W["states"][iid]
    run, segs, ev, key = pipeline.decode_state(st, p)
    recs = records(segs, run.vocab, p)
    tr = _W["truth"][iid]
    cands = []
    for r in recs:
        if r.passing:
            est = cm.Segments(np.array([r.t0]), np.array([r.t1]), [r.harte])
            hit = cm._est_matched(0, est, tr["ref"], tr["ref_tol"], "majmin_inv")
            cands.append((r.conf, r.sal, r.demoted, bool(hit), r.fixed))
    return iid, recs, cands, time.perf_counter() - t


def item_stats(iid: str, recs: list[Rec], cands, slider, details=DETAILS) -> dict:
    """Sufficient statistics of one item at each Detail stop."""
    tr = _W["truth"][iid]
    out = {}
    for d in details:
        iv, lab = flatten(recs, d, slider)
        if not len(lab):
            iv, lab = np.array([[tr["iv"][0, 0], tr["iv"][-1, 1]]]), ["N"]
        args = (tr["iv"], tr["lab"], iv, lab)
        std = cm.standard_scores(*args)
        ss = cm.short_segment_prf(*args, tr["beats"], max_beats=1.0, levels=("majmin_inv", "sevenths_inv"))
        fl = cm.flicker_stats(*args, tr["beats"])
        ba = cm.bass_accuracy(*args)
        shown = [(c, s, dm, h, fx) for c, s, dm, h, fx in cands
                 if d > 0 and d >= min_detail(Rec(0, 0, "", True, c, s, dm, fx), slider)]
        out[d] = {"root": std["root"], "majmin": std["majmin"], "majmin_inv": std["majmin_inv"],
                  "sevenths_inv": std["sevenths_inv"], "seg": std["seg"],
                  "ss": {lv: {k: ss[lv][k] for k in ("n_ref", "hits", "n_est", "tp")} for lv in ss},
                  "flicker": (fl["count"], fl["minutes"]), "bass": (ba["correct_s"], ba["scored_s"]),
                  "pass": (sum(1 for x in shown if x[3]), len(shown))}
    return out


def _stats_job(args):
    iid, recs, cands, slider, details = args
    return iid, item_stats(iid, recs, cands, slider, details)


# ---------------------------------------------------------------------------------------------
# aggregation and the objective
# ---------------------------------------------------------------------------------------------

def _r(a, b):
    return a / b if b else float("nan")


def aggregate(stats: dict[str, dict], ids, details=DETAILS) -> dict:
    out = {}
    for d in details:
        rows = [stats[i][d] for i in ids if i in stats]
        if not rows:
            continue
        agg = {k: float(np.nanmean([r[k] for r in rows])) for k in ("root", "majmin", "majmin_inv", "sevenths_inv", "seg")}
        for lv in ("majmin_inv", "sevenths_inv"):
            c = {f: sum(r["ss"][lv][f] for r in rows) for f in ("n_ref", "hits", "n_est", "tp")}
            p, rc = _r(c["tp"], c["n_est"]), _r(c["hits"], c["n_ref"])
            agg[f"short_{lv}"] = {"P": p, "R": rc, "F1": cm._f1(p, rc), **c}
        n, m = sum(r["flicker"][0] for r in rows), sum(r["flicker"][1] for r in rows)
        agg["flicker"] = _r(n, m)
        agg["bass_acc"] = _r(sum(r["bass"][0] for r in rows), sum(r["bass"][1] for r in rows))
        h, n = sum(r["pass"][0] for r in rows), sum(r["pass"][1] for r in rows)
        agg["pass_prec"] = _r(h, n)
        agg["n_pass"] = n
        out[d] = agg
    return out


def objective(agg: dict) -> float:
    a = agg[0.8]
    f1 = a["short_majmin_inv"]["F1"]
    j = 0.4 * a["sevenths_inv"] + 0.4 * (0.0 if math.isnan(f1) else f1) + 0.2 * a["bass_acc"]
    j -= 0.02 * max(0.0, a["flicker"] - 2.0)
    for s, tgt in PREC_TARGET.items():
        pr = agg[s]["pass_prec"]
        if not math.isnan(pr):
            j -= max(0.0, tgt - 0.05 - pr)
    return float(j)


# ---------------------------------------------------------------------------------------------
# slider fit (§5.11)
# ---------------------------------------------------------------------------------------------

def _logit(c):
    c = np.clip(np.asarray(c, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(c / (1 - c))


def fit_slider(cands: list, prev=(-2.0, -0.5, 1.5)) -> tuple[float, float, float]:
    """(a, b, c) so that passing chords shown at Standard / Detailed reach precision ≈ 0.85 / 0.70.

    1. Direction: logistic regression of hit on (logit(conf), salience) (light L2).
    2. z = α·logit(conf) + β·salience; on Standard the shown set is {z ≥ z½}, on Detailed
       {z ≥ z⅘}, where each threshold is the lowest whose shown set (with the A–P–A demotion of
       §5.12 step 3 applied) reaches the stop's precision target.
    3. σ(a·logit(conf) + b·sal + c) ≤ s ⇔ z ≥ (c − logit(s))/λ with a = −λα, b = −λβ: two
       thresholds fix λ = logit(0.8)/(z½ − z⅘) and c = λ·z½."""
    rows = [(c, s, dm, h) for c, s, dm, h, fx in cands if fx is None]
    if len(rows) < 10 or len({h for *_, h in rows}) < 2:
        return prev
    X = np.c_[_logit([r[0] for r in rows]), [r[1] for r in rows]]
    y = np.array([float(r[3]) for r in rows])
    dem = np.array([bool(r[2]) for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b0 = np.zeros(2), 0.0
    for _ in range(500):                                      # Newton steps on the L2 log-loss
        pz = 1 / (1 + np.exp(-(Z @ w + b0)))
        g = np.r_[Z.T @ (pz - y) + 0.1 * w, np.sum(pz - y)]
        Wd = pz * (1 - pz)
        Hm = np.zeros((3, 3))
        Hm[:2, :2] = Z.T @ (Z * Wd[:, None]) + 0.1 * np.eye(2)
        Hm[:2, 2] = Hm[2, :2] = Z.T @ Wd
        Hm[2, 2] = Wd.sum() + 1e-9
        step = np.linalg.solve(Hm, g)
        w -= step[:2]
        b0 -= step[2]
        if np.abs(step).max() < 1e-8:
            break
    alpha, beta = w / sd                                       # direction in raw units
    if not SLIDER_USE_SALIENCE:
        alpha, beta = 1.0, 0.0                                 # confidence only (monotone, 1-D)
    if alpha <= 0 and beta <= 0:
        return prev
    z = X @ np.array([alpha, beta])
    order = np.sort(np.unique(z))[::-1]

    zmax = float(order[0])

    def wilson_lb(k: int, n: int, zc: float = WILSON_Z) -> float:
        if n == 0:
            return 0.0
        ph = k / n
        d = 1 + zc * zc / n
        return (ph + zc * zc / (2 * n) - zc * math.sqrt(ph * (1 - ph) / n + zc * zc / (4 * n * n))) / d

    def thr(target: float, s: float, lam_c=None) -> float:
        """Lowest threshold whose shown set reaches the target (Wilson lower bound of its
        precision ≥ target − 0.05, the tolerance of the objective — a point estimate on a few
        shown candidates overfits: the leave-one-idiom-out precision at Standard fell to 0.27);
        above every z (nothing shown at this stop) when no threshold does."""
        best = zmax + 1.0
        for t in order:                                        # descending: lower t shows more
            shown = z >= t
            if lam_c is not None:                              # demoted ones need s ≥ md + 0.3
                lam, c = lam_c
                md = 1 / (1 + np.exp(-np.clip(c - lam * z, -50, 50)))
                shown &= ~dem | (md + 0.3 <= s)
            n = int(shown.sum())
            if n and (wilson_lb(int(y[shown].sum()), n) if WILSON_Z > 0 else y[shown].mean()) >= target - 0.05:
                best = t
        return float(best)

    def solve(z_half: float, z_det: float) -> tuple[float, float]:
        if z_det >= z_half - 1e-6:                             # Detailed must show at least Standard
            z_half = z_det + 1e-3 * max(1.0, abs(z_det)) if z_half > zmax else z_half
            z_det = min(z_det, z_half - 1e-3 * max(1.0, abs(z_half)))
        lam = LOGIT[0.8] / (z_half - z_det)
        return lam, lam * z_half

    lam, c = solve(thr(PREC_TARGET[0.5], 0.5), thr(PREC_TARGET[0.8], 0.8))
    for _ in range(2):                                         # refine with the demotion in place
        lam, c = solve(thr(PREC_TARGET[0.5], 0.5, (lam, c)), thr(PREC_TARGET[0.8], 0.8, (lam, c)))
    return (round(float(-lam * alpha), 4), round(float(-lam * beta), 4), round(float(c), 4))


# ---------------------------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------------------------

SPACE = {                     # name: (lo, hi); w_bas is w_src["bas"]
    "lam_scale": (0.4, 1.6),
    "w_bas": (1.0, 3.0),
    "pt_bonus": (0.0, 2.5),
    "w_prior": (0.0, 2.0),
    "w_short": (0.0, 2.0),
}


def to_params(x: dict, base) -> dict:
    from nathan_chordify.types import DecoderParams
    p = replace(base, temp=dict(base.temp), w_src=dict(base.w_src), lam=dict(base.lam))
    for k, v in x.items():
        if k == "w_bas":
            p.w_src["bas"] = float(v)
        else:
            setattr(p, k, float(v))
    d = asdict(p)
    d["slider"] = tuple(d["slider"])
    return d


class Tuner:
    def __init__(self, set_name: str, workers: int, base):
        self.set_name, self.base = set_name, base
        _init(set_name)
        self.ids = sorted(_W["states"])
        self.idiom = {i: _W["items"][i].meta.get("idiom", "?") for i in self.ids}
        self.pool = mp.Pool(workers, initializer=_init, initargs=(set_name,)) if workers > 1 else None
        self.cache: dict[str, dict] = {}

    def _map(self, fn, jobs):
        return self.pool.map(fn, jobs, chunksize=1) if self.pool else list(map(fn, jobs))

    def evaluate(self, x: dict) -> dict:
        key = json.dumps({k: round(float(v), 5) for k, v in sorted(x.items())})
        if key in self.cache:
            return self.cache[key]
        pd = to_params(x, self.base)
        dec = self._map(decode_item, [(i, pd) for i in self.ids])
        recs = {i: r for i, r, _, _ in dec}
        cands = {i: c for i, _, c, _ in dec}
        allc = [c for i in self.ids for c in cands[i]]
        slider = fit_slider(allc, tuple(self.base.slider))
        st = dict(self._map(_stats_job, [(i, recs[i], cands[i], slider, DETAILS) for i in self.ids]))
        agg = aggregate(st, self.ids)
        res = {"x": x, "slider": slider, "J": objective(agg), "agg": agg, "stats": st, "cands": cands,
               "recs": recs, "secs": float(sum(d[3] for d in dec))}
        self.cache[key] = res
        return res

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()


def sobol(n: int, seed: int = 7) -> list[dict]:
    from scipy.stats import qmc
    names = list(SPACE)
    s = qmc.Sobol(len(names), scramble=True, seed=seed).random(n)
    return [{k: SPACE[k][0] + u * (SPACE[k][1] - SPACE[k][0]) for k, u in zip(names, row)} for row in s]


def coordinate_descent(t: Tuner, x0: dict, rounds: int = 2, log=print) -> dict:
    best = t.evaluate(x0)
    for r in range(rounds):
        for k, (lo, hi) in SPACE.items():
            span = (hi - lo) / (4 * (r + 1))
            for v in (best["x"][k] - span, best["x"][k] + span):
                v = float(min(max(v, lo), hi))
                x = dict(best["x"], **{k: v})
                res = t.evaluate(x)
                if res["J"] > best["J"] + 1e-4:
                    best = res
                    log(f"  CD {k}={v:.3f}: J {res['J']:.4f}")
    return best


def loio(t: Tuner, results: list[dict]) -> dict:
    """Leave-one-idiom-out: per held-out idiom, the config with the best J on the rest (slider
    refitted on the rest), scored on the held-out items."""
    idioms = sorted(set(t.idiom.values()))
    held_stats: dict[str, dict] = {}
    picks = []
    for idm in idioms:
        train = [i for i in t.ids if t.idiom[i] != idm]
        test = [i for i in t.ids if t.idiom[i] == idm]
        scored = [(objective(aggregate(r["stats"], train)), n) for n, r in enumerate(results)]
        jbest, n = max(scored)
        r = results[n]
        slider = fit_slider([c for i in train for c in r["cands"][i]], tuple(t.base.slider))
        st = dict(t._map(_stats_job, [(i, r["recs"][i], r["cands"][i], slider, DETAILS) for i in test]))
        held_stats.update(st)
        picks.append({"held_out": idm, "config": n, "x": r["x"], "slider": slider, "J_train": jbest,
                      "J_test": objective(aggregate(st, test))})
    agg = aggregate(held_stats, t.ids)
    return {"picks": picks, "agg": agg, "J": objective(agg)}


def _summ(agg: dict) -> str:
    a = agg[0.8]
    return (f"sev_inv {a['sevenths_inv']:.3f} mm_inv {a['majmin_inv']:.3f} shortF1 {a['short_majmin_inv']['F1']:.3f} "
            f"(P {a['short_majmin_inv']['P']:.2f} R {a['short_majmin_inv']['R']:.2f}) flick {a['flicker']:.2f} "
            f"bass {a['bass_acc']:.3f} prec .5/.8/1 {agg[0.5]['pass_prec']:.2f}/{agg[0.8]['pass_prec']:.2f}/"
            f"{agg[1.0]['pass_prec']:.2f}")


def main(argv: list[str] | None = None) -> int:
    from nathan_chordify.types import DecoderParams
    ap = argparse.ArgumentParser(prog="python -m eval.tune")
    ap.add_argument("--set", default="D2")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--base", help="JSON DecoderParams overrides for the base (e.g. fitted temps)")
    ap.add_argument("--from-run", dest="from_run", help="re-score the --top best configs of an earlier tune JSON")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "eval" / "reports" / "tune_m2.json"))
    a = ap.parse_args(argv)
    base = DecoderParams()
    if a.base:
        base = replace(base, **json.loads(a.base))
    t0 = time.perf_counter()
    t = Tuner(a.set, a.workers, base)
    print(f"{len(t.ids)} items; base temps {base.temp}", flush=True)
    x_def = {"lam_scale": base.lam_scale, "w_bas": base.w_src.get("bas", 2.0), "pt_bonus": base.pt_bonus,
             "w_prior": base.w_prior, "w_short": base.w_short}
    r = t.evaluate(x_def)
    print(f"defaults: J {r['J']:.4f}  {_summ(r['agg'])}  slider {r['slider']}  ({r['secs']:.0f} cpu-s)", flush=True)
    results = [r]
    xs = sobol(a.samples)
    if a.from_run:                       # stage 2: re-score the best configs of an earlier run
        prev = json.loads(Path(a.from_run).read_text("utf-8"))
        xs = [c["x"] for c in sorted(prev["all"], key=lambda c: -(c["J"] if c["J"] is not None else -9))[:a.top]]
    for n, x in enumerate(xs):
        r = t.evaluate(x)
        results.append(r)
        print(f"[{n + 1}/{len(xs)}] J {r['J']:.4f} " + " ".join(f"{k}={v:.2f}" for k, v in x.items())
              + f"  | {_summ(r['agg'])}", flush=True)
    best = max(results, key=lambda r: r["J"])
    print(f"best Sobol: J {best['J']:.4f} {best['x']}", flush=True)
    best = coordinate_descent(t, best["x"], a.rounds)
    results = list(t.cache.values())
    cv = loio(t, results)
    print(f"final (in-sample): J {best['J']:.4f}  {_summ(best['agg'])}  slider {best['slider']}")
    print(f"LOIO held-out:     J {cv['J']:.4f}  {_summ(cv['agg'])}")
    out = {"base": {k: v for k, v in asdict(base).items()}, "best": {k: best[k] for k in ("x", "slider", "J", "agg")},
           "default": {k: results[0][k] for k in ("x", "slider", "J", "agg")},
           "loio": cv, "n_configs": len(results),
           "all": [{"x": r["x"], "J": r["J"], "slider": r["slider"]} for r in results],
           "minutes": (time.perf_counter() - t0) / 60}
    Path(a.out).write_text(json.dumps(out, indent=1, default=lambda o: o if not isinstance(o, float) or math.isfinite(o) else None), "utf-8")
    t.close()
    print(f"wrote {a.out} in {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
