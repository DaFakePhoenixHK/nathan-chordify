"""Score the engine on the D1 songs of eval/songs.yaml (draft GT from published charts, 2026-11-01).

    python -m eval.d1_eval [--out eval/reports/D1-draft] [--keys yt-a,yt-b]

Predictions per song (audio from the app cache, cache/<key>/):
- ``detailed``: the current engine re-decoded from the cached decode state (``pipeline.redecode``),
  flat timeline at Detail 0.8 (what the app shows at "Detailed");
- ``simple``: the same decode at Detail 0 (main chords only, the Chordify-like view);
- ``b1``: baseline B1, lv-chordia with its own decoder (``chord_recognition(..., "submission")``),
  cached in eval/data/cache/_b1/<key>.json.

Writes <out>.json (every metric of ``chord_metrics.evaluate_all`` per song and prediction) and
prints a summary table; the prose report eval/reports/D1-draft.md is written by hand from it.
GT stretches labelled X (not covered by the published chart) are left out of every score.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import chord_metrics as cm

ROOT = Path(__file__).resolve().parents[1]
PREDS = ("detailed", "simple", "b1")


def songs(keys: list[str] | None = None) -> list[dict]:
    import yaml
    data = yaml.safe_load((ROOT / "eval" / "songs.yaml").read_text("utf-8"))
    out = [s for s in data.get("songs", []) if isinstance(s, dict) and s.get("set") == "D1"]
    return [s for s in out if not keys or s["key"] in keys]


def engine_preds(key: str) -> dict[str, tuple[np.ndarray, list[str]]]:
    from nathan_chordify import pipeline
    from .engine_eval import lab_from_analysis
    a = pipeline.redecode(key)
    return {"detailed": lab_from_analysis(a, 0.8), "simple": lab_from_analysis(a, 0.0)}


def b1_pred(key: str) -> tuple[np.ndarray, list[str]]:
    from .engine_eval import B1_DIR
    B1_DIR.mkdir(parents=True, exist_ok=True)
    out = B1_DIR / f"{key}.json"
    if not out.exists():
        from lv_chordia.chord_recognition import chord_recognition
        res = chord_recognition(str((ROOT / "cache" / key / "audio.wav").resolve()), "submission")
        out.write_text(json.dumps(res), "utf-8")
    res = json.loads(out.read_text("utf-8"))
    iv = np.array([[r["start_time"], r["end_time"]] for r in res], dtype=float).reshape(-1, 2)
    return iv, [r["chord"] for r in res]


def _scored(ref_iv: np.ndarray, ref_lab: list[str]) -> float:
    return float(sum(t1 - t0 for (t0, t1), lab in zip(ref_iv, ref_lab) if lab != "X"))


def _clean(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return _clean(x.item())
    return x


def overall(per_song: dict[str, dict]) -> dict:
    """Pooled over songs: duration-weighted standard scores (by scored GT seconds), pooled counts
    for short segments and flicker, pooled seconds for bass."""
    out: dict[str, dict] = {}
    for p in PREDS:
        rows = [(s["scored_s"], s["metrics"][p]) for s in per_song.values() if p in s["metrics"]]
        if not rows:
            continue
        o: dict = {}
        for name in ("root", "majmin", "majmin_inv", "sevenths_inv", "tetrads_inv", "seg"):
            vals = [(w, m["standard"][name]) for w, m in rows if m["standard"][name] is not None
                    and not (isinstance(m["standard"][name], float) and math.isnan(m["standard"][name]))]
            o[name] = sum(w * v for w, v in vals) / max(sum(w for w, _ in vals), 1e-9)
        ss = [m["shortseg_1beat"]["majmin_inv"] for _, m in rows]
        n_ref, hits = sum(x["n_ref"] for x in ss), sum(x["hits"] for x in ss)
        n_est, tp = sum(x["n_est"] for x in ss), sum(x["tp"] for x in ss)
        P, R = (tp / n_est if n_est else float("nan")), (hits / n_ref if n_ref else float("nan"))
        o["short_P"], o["short_R"] = P, R
        o["short_F1"] = 2 * P * R / (P + R) if P + R > 0 else 0.0
        o["short_n_ref"] = n_ref
        fl = [m["flicker"] for _, m in rows]
        o["flicker_per_min"] = sum(f["count"] for f in fl) / max(sum(f["minutes"] for f in fl), 1e-9)
        bs = [m["bass"] for _, m in rows]
        o["bass_acc"] = sum(b["correct_s"] for b in bs) / max(sum(b["scored_s"] for b in bs), 1e-9)
        out[p] = o
    return out


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m eval.d1_eval")
    ap.add_argument("--out", default="eval/reports/D1-draft")
    ap.add_argument("--keys", default="")
    a = ap.parse_args(argv)
    per_song: dict[str, dict] = {}
    for s in songs([k for k in a.keys.split(",") if k]):
        key = s["key"]
        ref_iv, ref_lab = cm.load_lab(ROOT / s["lab"])
        beats = cm.load_beats(ROOT / s["beats"])
        preds = engine_preds(key)
        preds["b1"] = b1_pred(key)
        res = {p: cm.evaluate_all(ref_iv, ref_lab, iv, lab, beats) for p, (iv, lab) in preds.items()}
        per_song[key] = {"title": s.get("title"), "key_sig": s.get("key_sig"), "reviewed": s.get("reviewed"),
                         "scored_s": _scored(ref_iv, ref_lab), "metrics": res}
        print(f"{key} {s.get('title')}")
        for p in PREDS:
            m = res[p]
            st, ss = m["standard"], m["shortseg_1beat"]["majmin_inv"]
            print(f"  {p:<9} root {st['root']:.3f} majmin {st['majmin']:.3f} majmin_inv {st['majmin_inv']:.3f} "
                  f"7ths_inv {st['sevenths_inv']:.3f} | short P/R/F1 {ss['precision']:.2f}/{ss['recall']:.2f}/"
                  f"{ss['f1']:.2f} (n={ss['n_ref']}) | flicker {m['flicker']['rate']:.2f}/min | bass {m['bass']['acc']:.3f}")
    ov = overall(_clean(per_song))
    print("OVERALL")
    for p, o in ov.items():
        print(f"  {p:<9} " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in o.items()))
    out = ROOT / f"{a.out}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean({"songs": per_song, "overall": ov}), ensure_ascii=False, indent=1), "utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
