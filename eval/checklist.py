"""Passing-chord checklist and false-positive gate for 回家 (DESIGN §13.1 steps 6–7, review E2).

    python -m eval.checklist yt-Y7boUFbM76k                     # our cached analysis.json (Detailed)
    python -m eval.checklist yt-Y7boUFbM76k --pred b1           # baseline B1 (lv-chordia + its CRF)
    python -m eval.checklist yt-Y7boUFbM76k --pred some.lab     # any .lab file

Inputs: ``eval/gt/<key>.checklist.json`` (one entry per scored occurrence; see its "rules") and
``eval/gt/<key>.chart.txt`` (the draft bar chart, for the false-positive gate). Both are DRAFT
GT until Nathan reviews them; reports must say so (§13.1 step 5).

A prediction is a list of ``Shown(t0, t1, harte, passing)``: the chords visible at Detailed.
For our engine that is ``export.flatten(chords, 0.8)``, with ``passing`` = the passing layer;
for a prediction without layers (B1, a .lab) every chord of at most 1 beat counts as "passing"
for the false-positive gate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "eval" / "gt"
BAR_S = 4 * 60.0 / 65.4            # 回家's nominal bar (3.67 s); section starts come from the chart
SHORT_S = 60.0 / 65.4              # 1 beat: "passing" for predictions without layers


@dataclass
class Shown:
    t0: float
    t1: float
    harte: str
    passing: bool


# ---------------------------------------------------------------------------------------------
# chord matching
# ---------------------------------------------------------------------------------------------

def _triple(label: str) -> tuple[int, str, int]:
    from nathan_chordify.theory import harte as H
    try:
        return H.parse_harte(label, strict=False)
    except ValueError:
        return H.parse_symbol(label)


def _sym(label: str) -> tuple[int, str, int]:
    """A checklist label (lead-sheet symbol such as "D#7/A#") → (root, quality, bass)."""
    from nathan_chordify.theory import harte as H
    return H.parse_symbol(label)


def family(q: str) -> str:
    from nathan_chordify.decode.evidence import LV_QUALITY
    if q == "N":
        return "N"
    fam = LV_QUALITY[q][0]
    return "5" if fam == "maj|min" else fam


def _cls(t: tuple[int, str, int]) -> int | None:
    from nathan_chordify.theory.vocab import candidates, index_of
    i = index_of(*t)
    return None if i is None else candidates().cands[i].cls


def match(expected: tuple[int, str, int], shown: tuple[int, str, int], accept: dict | None = None) -> float:
    """1 for root + triad family + bass (or the same equivalence class); an exact 'accept' label
    scores its value; else 0."""
    if shown[1] == "N" or expected[1] == "N":
        return float(shown[1] == expected[1])
    if (shown[0] == expected[0] and shown[2] == expected[2] and family(shown[1]) == family(expected[1])):
        return 1.0
    ce, cs = _cls(expected), _cls(shown)
    if ce is not None and ce == cs:
        return 1.0
    best = 0.0
    for lab, val in (accept or {}).items():
        if _sym(lab) == shown:
            best = max(best, float(val))
    return best


def _best_in(window: tuple[float, float], exp: tuple[int, str, int], shown: list[Shown],
             parsed: list[tuple[int, str, int]], accept: dict | None) -> tuple[float, str | None]:
    a, b = window
    best, lab = 0.0, None
    for s, t in zip(shown, parsed):
        if s.t1 > a and s.t0 < b:
            v = match(exp, t, accept)
            if v > best:
                best, lab = v, s.harte
    return best, lab


# ---------------------------------------------------------------------------------------------
# draft chart → timeline (for the FP gate)
# ---------------------------------------------------------------------------------------------

_SEC_RE = re.compile(r"^\s*\[([^\]@]+?)\s*@\s*([\d.]+)\s*\]")


def chart_timeline(path: Path) -> list[tuple[float, float, str, str]]:
    """(t0, t1, harte, conf) per chord of the draft chart: section start + 3.67 s per bar,
    positions inside a bar from the chart (even split when it gives none)."""
    from nathan_chordify.theory import chart as CH
    out = []
    start = None
    bar_i = 0
    for raw in path.read_text("utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip() if raw.lstrip().startswith("#") else raw.rstrip()
        if not line.strip():
            continue
        m = _SEC_RE.match(line)
        if m:
            start, bar_i = float(m.group(2)), 0
            line = line[m.end():]
            if not line.strip():
                continue
        if "|" not in line or start is None:
            continue
        conf = "H" if "(H)" in line else ("L" if "(L)" in line else "M")
        body = line[:line.rfind("|") + 1]
        chart = CH.parse(body)
        for bar in chart.bars:
            b0 = start + bar_i * BAR_S
            n = len(bar)
            pos = [c.pos if c.pos is not None else 4.0 * k / n for k, c in enumerate(bar)]
            for k, c in enumerate(bar):
                t0 = b0 + pos[k] * BAR_S / 4
                t1 = b0 + (pos[k + 1] * BAR_S / 4 if k + 1 < n else BAR_S)
                out.append((t0, t1, c.harte, conf))
            bar_i += 1
    return out


# ---------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------

def score(shown: list[Shown], checklist: dict, timeline: list[tuple[float, float, str, str]] | None = None) -> dict:
    parsed = [_triple(s.harte) for s in shown]
    tol = float(checklist["rules"].get("point_tol_s", 0.3))
    rows = []
    for it in checklist["items"]:
        if it.get("scored", True) is False:
            continue
        acc = it.get("accept")
        occ: list[tuple[float, float, list[str], str]] = []
        for t in it.get("at", []):
            w = it.get("window", [-tol, tol])
            occ.append((t + w[0], t + w[1], it["expect"], f"{t:g}"))
        for a, b, exp in it.get("ranges", []):
            occ.append((a, b, exp, f"{a:g}–{b:g}"))
        for a, b, exp, where in occ:
            parts = []
            for lab in exp:
                v, got = _best_in((a, b), _sym(lab), shown, parsed, acc)
                parts.append({"expect": lab, "score": v, "shown": got})
            rows.append({"n": it["n"], "name": it["name"], "where": where,
                         "score": float(np.mean([p["score"] for p in parts])), "parts": parts})
        if "bass_walk" in it:
            hits, parts = 0, []
            for a, b, lab in it["bass_walk"]:
                want = _sym(lab)[2]
                q0, q1 = a + 0.25 * (b - a), b - 0.25 * (b - a)
                dur = {}
                for s, t in zip(shown, parsed):
                    ov = min(s.t1, q1) - max(s.t0, q0)
                    if ov > 0 and t[2] >= 0:
                        dur[t[2]] = dur.get(t[2], 0.0) + ov
                got = max(dur, key=dur.get) if dur else -1
                ok = got == want and dur.get(got, 0.0) >= 0.5 * (q1 - q0)
                hits += ok
                parts.append({"expect": lab, "score": float(ok), "shown_bass": got})
            rows.append({"n": it["n"], "name": it["name"], "where": f"{it['bass_walk'][0][0]:g}–{it['bass_walk'][-1][1]:g}",
                         "score": hits / len(it["bass_walk"]), "parts": parts})
    out = {"n": len(rows), "score": float(np.mean([r["score"] for r in rows])) if rows else float("nan"),
           "rows": rows}
    if timeline is not None:
        out["fp"] = false_positives(shown, parsed, timeline)
    return out


def false_positives(shown: list[Shown], parsed, timeline) -> dict:
    """Passing chords at Detailed inside H bars that match no overlapping draft-GT chord."""
    H = [(a, b, _triple(h)) for a, b, h, c in timeline if c == "H"]
    h_minutes = sum(b - a for a, b, _ in H) / 60.0
    bad = []
    for s, t in zip(shown, parsed):
        if not s.passing or t[1] == "N":
            continue
        mid = 0.5 * (s.t0 + s.t1)
        if not any(a <= mid < b for a, b, _ in H):
            continue
        if not any(b > s.t0 and a < s.t1 and match(g, t) > 0 for a, b, g in H):
            bad.append([round(s.t0, 2), s.harte])
    return {"count": len(bad), "h_minutes": round(h_minutes, 3),
            "per_min": len(bad) / h_minutes if h_minutes else float("nan"), "chords": bad}


# ---------------------------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------------------------

def shown_from_analysis(analysis, detail: float = 0.8) -> list[Shown]:
    from nathan_chordify.export import flatten
    return [Shown(c.start, c.end, c.harte, c.layer == "passing") for c in flatten(analysis.chords, detail)]


def shown_from_lab(intervals, labels) -> list[Shown]:
    return [Shown(float(a), float(b), lab, (b - a) <= SHORT_S + 1e-6 and lab not in ("N", "X"))
            for (a, b), lab in zip(np.asarray(intervals).reshape(-1, 2), labels)]


def b1_shown(key: str) -> list[Shown]:
    from nathan_chordify import config
    p = config.CACHE_DIR / key / "spike" / "b1_lv_crf.json"
    res = json.loads(p.read_text("utf-8"))
    return shown_from_lab([[r["start_time"], r["end_time"]] for r in res], [r["chord"] for r in res])


def load(key: str) -> tuple[dict, list]:
    cl = json.loads((GT_DIR / f"{key}.checklist.json").read_text("utf-8"))
    tl = chart_timeline(GT_DIR / f"{key}.chart.txt")
    return cl, tl


def evaluate(key: str, shown: list[Shown]) -> dict:
    cl, tl = load(key)
    return score(shown, cl, tl)


def format_result(res: dict, title: str) -> str:
    lines = [f"{title}: checklist {res['score']:.3f} over {res['n']} occurrences (DRAFT GT)"]
    if "fp" in res:
        fp = res["fp"]
        lines.append(f"  false positives in H bars: {fp['count']} in {fp['h_minutes']:.2f} min = {fp['per_min']:.2f}/min")
    for r in res["rows"]:
        got = ", ".join(f"{p['expect']}→{p.get('shown', p.get('shown_bass'))}:{p['score']:g}" for p in r["parts"])
        lines.append(f"  #{r['n']:<2} {r['name'][:34]:34s} @{r['where']:>11s}  {r['score']:.2f}  [{got}]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.checklist")
    ap.add_argument("key")
    ap.add_argument("--pred", default="engine", help="engine | b1 | <path to .lab>")
    ap.add_argument("--detail", type=float, default=0.8)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.pred == "engine":
        from nathan_chordify import pipeline
        shown = shown_from_analysis(pipeline.load_cached(a.key), a.detail)
    elif a.pred == "b1":
        shown = b1_shown(a.key)
    else:
        from .chord_metrics import load_lab
        iv, lab = load_lab(a.pred)
        shown = shown_from_lab(iv, lab)
    res = evaluate(a.key, shown)
    if isinstance(sys.stdout, type(sys.__stdout__)) and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False) if a.json else format_result(res, f"{a.key} [{a.pred}]"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
