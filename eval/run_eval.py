"""Score chord predictions against reference sets and write a report (DESIGN §13.7).

    python -m eval.run_eval --set eval/data/d2 --pred <dir>              # <dir>/<id>.lab or <dir>/<id>/<id>.lab
    python -m eval.run_eval --set D2 --pred @oracle                       # sanity check: every score is 1
    python -m eval.run_eval --set D2 --pred @drop-passing                 # main chords only (Chordify-like)
    python -m eval.run_eval --set D2 --pred "tool.exe {audio} {out}"      # command template, one run per item

A set is a folder of items, each with ``<id>.lab`` and ``<id>.beats`` (and, for D2, ``<id>.json``
with passing flags and idiom tags; audio as ``<id>.wav``/``.flac``). Several sets: ``--set a,b``.
Command templates may use {audio}, {out}, {beats} and {id}.

The report (markdown + JSON) goes to eval/reports/<stamp>_<set>_<pred>.{md,json}. It compares the
headline numbers with the previous report for the same set and prediction name.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import chord_metrics as cm

ROOT = Path(__file__).resolve().parents[1]
SET_ALIASES = {"d2": "eval/data/d2", "mini-d2": "eval/data/d2", "d1": "eval/gt"}
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
SUMMARY_KEYS = (
    ("root", ("standard", "root")), ("majmin", ("standard", "majmin")),
    ("majmin_inv", ("standard", "majmin_inv")), ("sevenths_inv", ("standard", "sevenths_inv")),
    ("tetrads_inv", ("standard", "tetrads_inv")), ("seg", ("standard", "seg")),
    ("short R majmin_inv (<=1 beat)", ("shortseg_1beat", "majmin_inv", "recall")),
    ("short P majmin_inv (<=1 beat)", ("shortseg_1beat", "majmin_inv", "precision")),
    ("short F1 majmin_inv (<=1 beat)", ("shortseg_1beat", "majmin_inv", "f1")),
    ("short F1 sevenths_inv (<=1 beat)", ("shortseg_1beat", "sevenths_inv", "f1")),
    ("short R majmin_inv (<=0.6 s)", ("shortseg_0.6s", "majmin_inv", "recall")),
    ("flicker / min", ("flicker", "rate")),
    ("boundary F 100 ms", ("boundary_100ms", "f")), ("boundary F 1/4 beat", ("boundary_qbeat", "f")),
    ("bass acc", ("bass", "acc")), ("out-of-chord bass acc", ("ooc_bass", "acc")),
)


# ----------------------------------------------------------------------------------------- items

@dataclass
class Item:
    id: str
    set: str
    lab: Path
    beats: Path | None
    audio: Path | None
    meta: dict = field(default_factory=dict)


def resolve_set(name: str) -> Path:
    path = Path(SET_ALIASES.get(name.lower(), name))
    for cand in (path, ROOT / path):
        if cand.is_dir():
            return cand
    raise SystemExit(f"set not found: {name}")


def discover(set_dir: Path) -> list[Item]:
    """Every <id>.lab under set_dir with its beats / audio / json siblings."""
    items = []
    for lab in sorted(set_dir.rglob("*.lab")):
        stem, folder = lab.stem, lab.parent
        meta_path = folder / f"{stem}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        beats = folder / f"{stem}.beats"
        audio = next((folder / f"{stem}{ext}" for ext in AUDIO_EXTS if (folder / f"{stem}{ext}").exists()), None)
        items.append(Item(stem, set_dir.name, lab, beats if beats.exists() else None, audio, meta))
    return items


def ref_tags(item: Item, ref_iv: np.ndarray, ref_lab: list[str]) -> list | None:
    """Idiom tags per reference row, from the item's json (D2) when it matches the .lab."""
    segs = item.meta.get("segments")
    if not segs or len(segs) != len(ref_lab):
        return None
    if any(abs(s["t0"] - t0) > 1e-3 or s["label"] != lab for s, (t0, _), lab in zip(segs, ref_iv, ref_lab)):
        return None
    return [s.get("tags") or None for s in segs]


def item_beats(item: Item, ref_iv: np.ndarray) -> np.ndarray:
    if item.beats is not None:
        return cm.load_beats(item.beats)
    bpm = float(item.meta.get("bpm") or 120.0)  # no tapped beats: a steady grid (flagged in the report)
    return np.arange(ref_iv[0, 0], ref_iv[-1, 1] + 1e-9, 60.0 / bpm)


# ----------------------------------------------------------------------------------- predictions

def drop_passing(item: Item) -> tuple[np.ndarray, list[str]]:
    """The reference with passing chords merged into the chord before them (a main-layer-only
    prediction, like Chordify). Without json flags, chords of <= 1 beat count as passing."""
    iv, lab = cm.load_lab(item.lab)
    segs = item.meta.get("segments")
    if segs and len(segs) == len(lab):
        passing = [bool(s.get("passing")) for s in segs]
    else:
        beats = item_beats(item, iv)
        dur = cm.beat_position(iv[:, 1], beats) - cm.beat_position(iv[:, 0], beats)
        passing = [d <= 1.0 + 1e-6 and cm._is_chord(x) for d, x in zip(dur, lab)]
    out_iv, out_lab = [], []
    for (t0, t1), x, p in zip(iv, lab, passing):
        if p and out_iv:
            out_iv[-1][1] = t1
        else:
            out_iv.append([t0, t1])
            out_lab.append(x)
    return np.array(out_iv), out_lab


def find_prediction(pred_dir: Path, item_id: str) -> Path | None:
    for cand in (pred_dir / f"{item_id}.lab", pred_dir / item_id / f"{item_id}.lab"):
        if cand.exists():
            return cand
    labs = sorted((pred_dir / item_id).glob("*.lab")) if (pred_dir / item_id).is_dir() else []
    return labs[0] if len(labs) == 1 else None


PARAMS: dict = {}          # DecoderParams overrides for @engine (--params)


def predict(item: Item, pred: str, work_dir: Path):
    """(intervals, labels) for one item, or None when the prediction is missing."""
    if pred == "@oracle":
        return cm.load_lab(item.lab)
    if pred == "@drop-passing":
        return drop_passing(item)
    if pred == "@engine" or pred.startswith("@engine:") or pred == "@engine-simple":
        # our pipeline (eval/engine_eval.py): the flat timeline at a Detail stop; B5 = Detail 0.
        # TODO(M4): use the item's tapped GT beats for D1 instead of our beat tracker.
        from . import engine_eval
        detail = 0.0 if pred == "@engine-simple" else (float(pred.split(":", 1)[1]) if ":" in pred else 0.8)
        return engine_eval.engine_prediction(item, detail, PARAMS or None)
    if pred == "@lv-crf":
        from . import engine_eval
        return engine_eval.b1_prediction(item)
    path = Path(pred)
    if path.is_dir():
        found = find_prediction(path, item.id)
        return cm.load_lab(found) if found else None
    if "{" not in pred:
        raise SystemExit(f"--pred is neither a directory, a builtin (@oracle, @drop-passing, @engine[:detail], "
                         f"@engine-simple, @lv-crf) nor a command template: {pred}")
    out = work_dir / f"{item.id}.lab"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = pred.format(audio=item.audio or "", out=out, beats=item.beats or "", id=item.id)
    subprocess.run(cmd, shell=True, check=True)
    return cm.load_lab(out) if out.exists() else None


# ----------------------------------------------------------------------------------- aggregation

def _get(d: dict, path) -> float:
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return float("nan")
        d = d[k]
    return float(d) if isinstance(d, (int, float)) else float("nan")


def _nanmean(xs) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _ratio(a, b) -> float:
    return float(a) / float(b) if b else float("nan")


def aggregate(results: dict[str, dict]) -> dict:
    """Pool item results: standard metrics are item means; counts and seconds are pooled."""
    rs = list(results.values())
    agg: dict = {"n_items": len(rs)}
    agg["standard"] = {k: _nanmean([r["standard"][k] for r in rs])
                       for k in cm.STANDARD_METRICS + cm.SEGMENTATION_METRICS}
    for key in ("shortseg_1beat", "shortseg_0.6s"):
        agg[key] = {}
        for lv in cm.LEVELS:
            c = {f: sum(r[key][lv][f] for r in rs) for f in ("n_ref", "hits", "n_est", "tp")}
            p, rec = _ratio(c["tp"], c["n_est"]), _ratio(c["hits"], c["n_ref"])
            agg[key][lv] = {**c, "precision": p, "recall": rec, "f1": cm._f1(p, rec)}
    agg["duration_bins"] = {}
    for name, _, _ in cm.DURATION_BINS:
        row = {"n": sum(r["duration_bins"][name]["n"] for r in rs)}
        for lv in cm.LEVELS:
            correct = sum(r["duration_bins"][name][lv]["correct_s"] for r in rs)
            scored = sum(r["duration_bins"][name][lv]["scored_s"] for r in rs)
            row[lv] = {"acc": _ratio(correct, scored), "correct_s": correct, "scored_s": scored}
        agg["duration_bins"][name] = row
    for key in ("boundary_100ms", "boundary_qbeat"):
        c = {f: sum(r[key][f] for r in rs) for f in ("n_ref", "n_est", "n_match")}
        p, rec = _ratio(c["n_match"], c["n_est"]), _ratio(c["n_match"], c["n_ref"])
        agg[key] = {**c, "precision": p, "recall": rec, "f": cm._f1(p, rec)}
    count, minutes = sum(r["flicker"]["count"] for r in rs), sum(r["flicker"]["minutes"] for r in rs)
    agg["flicker"] = {"count": count, "minutes": minutes, "rate": _ratio(count, minutes)}
    for key in ("bass", "ooc_bass"):
        correct, scored = sum(r[key]["correct_s"] for r in rs), sum(r[key]["scored_s"] for r in rs)
        agg[key] = {"acc": _ratio(correct, scored), "n_seg": sum(r[key]["n_seg"] for r in rs)}
    agg["bass"]["seg_acc"] = _ratio(sum(r["bass"]["seg_correct"] for r in rs), agg["bass"]["n_seg"])
    idioms: dict[str, dict] = {}
    for r in rs:
        for tag, row in r.get("idioms", {}).items():
            acc = idioms.setdefault(tag, {"n": 0, "hits": {lv: 0 for lv in cm.LEVELS}})
            acc["n"] += row["n"]
            for lv in cm.LEVELS:
                acc["hits"][lv] += row["hits"][lv]
    for row in idioms.values():
        row["recall"] = {lv: _ratio(row["hits"][lv], row["n"]) for lv in cm.LEVELS}
    agg["idioms"] = dict(sorted(idioms.items()))
    agg["summary"] = {name: _get(agg, path) for name, path in SUMMARY_KEYS}
    return agg


def grouped(results: dict[str, dict], items: dict[str, Item], key: str) -> dict:
    groups: dict[str, dict] = {}
    for item_id, r in results.items():
        g = items[item_id].meta.get(key)
        if g is not None:
            groups.setdefault(str(g), {})[item_id] = r
    def order(g: str):
        try:
            return (0, float(g), g)
        except ValueError:
            return (1, 0.0, g)

    return {g: aggregate(groups[g])["summary"] for g in sorted(groups, key=order)}


# --------------------------------------------------------------------------------------- reports

def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(c) if not isinstance(c, str) else c for c in row) + " |" for row in rows]
    return "\n".join(lines)


def previous_report(out_dir: Path, sets: str, pred: str, exclude: Path) -> dict | None:
    for path in sorted(out_dir.glob("*.json"), reverse=True):
        if path == exclude:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("sets") == sets and data.get("pred") == pred:
            data["_path"] = path.name
            return data
    return None


def _est_label_at(est, t: float) -> str:
    if est is None:
        return "–"
    iv, lab = est
    k = np.flatnonzero((iv[:, 0] <= t) & (iv[:, 1] > t))
    return lab[k[0]] if len(k) else "N"


def write_report(out_dir: Path, sets: str, pred: str, name: str, results: dict, items: dict,
                 missing: list[str], estimates: dict, notes: list[str]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{stamp}_{sets}_{name}")[:120]
    md_path, json_path = out_dir / f"{slug}.md", out_dir / f"{slug}.json"
    agg = aggregate(results) if results else {}
    prev = previous_report(out_dir, sets, pred, json_path)
    by_bpm, by_idiom = grouped(results, items, "bpm"), grouped(results, items, "idiom")

    listen = []
    for item_id, r in results.items():
        for t0, t1, label in r["shortseg_1beat"]["majmin_inv"]["misses"]:
            listen.append([item_id, t0, t1, label, _est_label_at(estimates.get(item_id), (t0 + t1) / 2)])
    listen = listen[:20]

    md = [f"# Chord evaluation: {sets} vs {pred}", "",
          f"Created {datetime.now():%Y-%m-%d %H:%M}. Items scored: {len(results)}; missing predictions: {len(missing)}."]
    md += [f"- {n}" for n in notes]
    if results:
        prev_sum = (prev or {}).get("aggregate", {}).get("summary", {})
        rows = []
        for key, value in agg["summary"].items():
            old = prev_sum.get(key)
            delta = "" if old is None or math.isnan(value) else f"{value - old:+.3f}"
            rows.append([key, value, delta])
        md += ["", "## Summary", "", _table(["metric", "value", f"Δ vs {prev['_path'] if prev else 'last run'}"], rows)]
        md += ["", "## Standard metrics (mir_eval, mean over items)", "",
               _table(["metric", "value"], [[k, v] for k, v in agg["standard"].items()])]
        rows = [[d, lv, s["n_ref"], s["recall"], s["n_est"], s["precision"], s["f1"]]
                for d in ("shortseg_1beat", "shortseg_0.6s") for lv, s in agg[d].items()]
        md += ["", "## Short segments (pooled)", "",
               "Hit = >= 50 % overlap, onset within ±max(70 ms, ¼ beat), label match at the level.", "",
               _table(["definition", "level", "n ref", "recall", "n est", "precision", "F1"], rows)]
        rows = [[b, row["n"]] + [row[lv]["acc"] for lv in cm.LEVELS] for b, row in agg["duration_bins"].items()]
        md += ["", "## Accuracy by reference duration (beats, seconds-weighted)", "",
               _table(["bin", "n"] + list(cm.LEVELS), rows)]
        if agg["idioms"]:
            rows = [[tag, row["n"]] + [row["recall"][lv] for lv in cm.LEVELS] for tag, row in agg["idioms"].items()]
            md += ["", "## Idiom recall (tagged reference chords)", "", _table(["idiom", "n"] + list(cm.LEVELS), rows)]
        cols = ["majmin_inv", "short R majmin_inv (<=1 beat)", "short F1 sevenths_inv (<=1 beat)", "flicker / min", "bass acc"]
        for title, groups in (("By tempo (BPM)", by_bpm), ("By idiom", by_idiom)):
            if groups:
                md += ["", f"## {title}", "", _table(["group"] + cols, [[g] + [s[c] for c in cols] for g, s in groups.items()])]
        rows = []
        for item_id, r in results.items():
            ss = r["shortseg_1beat"]["majmin_inv"]
            rows.append([item_id, r["standard"]["root"], r["standard"]["majmin_inv"], r["standard"]["sevenths_inv"],
                         f"{ss['hits']}/{ss['n_ref']}", ss["precision"], r["flicker"]["rate"], r["bass"]["acc"]])
        md += ["", "## Per item", "",
               _table(["item", "root", "majmin_inv", "sevenths_inv", "short hits", "short P", "flicker/min", "bass"], rows)]
        if listen:
            md += ["", "## Listen list (missed short chords, majmin_inv, first 20)", "",
                   _table(["item", "t0", "t1", "reference", "estimate at mid"], listen)]
    if missing:
        md += ["", "## Missing predictions", "", ", ".join(missing)]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    data = {"created": stamp, "sets": sets, "pred": pred, "name": name, "n_items": len(results),
            "missing": missing, "notes": notes, "aggregate": agg, "by_bpm": by_bpm, "by_idiom": by_idiom,
            "listen": listen, "previous": prev["_path"] if prev else None, "items": results}
    json_path.write_text(json.dumps(_clean(data), indent=1, ensure_ascii=False), encoding="utf-8")
    return md_path, json_path


# ------------------------------------------------------------------------------------------- main

def run(sets: str, pred: str, out_dir: Path, name: str | None = None, limit: int | None = None,
        verbose: bool = True) -> tuple[Path, Path, dict]:
    items: dict[str, Item] = {}
    for s in sets.split(","):
        for it in discover(resolve_set(s.strip())):
            items.setdefault(it.id, it)
    if not items:
        raise SystemExit(f"no .lab files found in {sets}")
    if limit:
        items = dict(list(items.items())[:limit])
    name = name or (pred.lstrip("@") if pred.startswith("@") else Path(pred).name or "pred")
    work = out_dir / "pred" / re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    results, estimates, missing, notes = {}, {}, [], []
    for item_id, it in items.items():
        est = predict(it, pred, work)
        if est is None:
            missing.append(item_id)
            continue
        ref_iv, ref_lab = cm.load_lab(it.lab)
        if it.beats is None:
            notes.append(f"{item_id}: no beats file, a steady grid was used")
        res = cm.evaluate_all(ref_iv, ref_lab, est[0], est[1], item_beats(it, ref_iv), ref_tags(it, ref_iv, ref_lab))
        results[item_id], estimates[item_id] = res, est
        if verbose:
            ss = res["shortseg_1beat"]["majmin_inv"]
            print(f"{item_id:36s} majmin_inv {_fmt(res['standard']['majmin_inv'])}  "
                  f"short {ss['hits']}/{ss['n_ref']}  flicker {_fmt(res['flicker']['rate'], 2)}/min", flush=True)
    md, js = write_report(out_dir, sets, pred, name, results, items, missing, estimates, notes)
    return md, js, aggregate(results) if results else {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.run_eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", required=True, help="set folder(s), comma-separated; aliases: D2, mini-d2, D1")
    ap.add_argument("--pred", required=True, help="prediction folder, @oracle, @drop-passing, @engine[:detail], "
                    "@engine-simple (B5), @lv-crf (B1) or a command template")
    ap.add_argument("--params", help="JSON DecoderParams overrides for @engine (re-decodes from cached states)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "reports"), help="report folder")
    ap.add_argument("--name", help="short name for the prediction (report file name)")
    ap.add_argument("--limit", type=int, help="score only the first N items")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.params:
        PARAMS.update(json.loads(a.params))
    md, js, agg = run(a.set, a.pred, Path(a.out), a.name, a.limit, not a.quiet)
    if agg:
        print("\n" + "\n".join(f"{k:36s} {_fmt(v)}" for k, v in agg["summary"].items()))
    print(f"\nreport: {md}\n        {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
