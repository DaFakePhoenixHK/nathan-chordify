"""Draft ground truth from published charts, aligned to the recording (D1-draft, 2026-11-01).

    python -m eval.build_gt <cache key> [--explore] [--no-write]
    python -m eval.build_gt all

Input: ``data/sheets/<key>/gt_source.txt`` — the chord chart read from a free sheet / chart page,
written as section *templates* in the bar-text format of ``theory.chart`` plus a few directives:

    @title  立志擺上 (建道神學院新祢呈敬拜隊, arr. 古丹青)
    @key    Eb major            # the recording's key (checked with the engine's key detector)
    @source 編曲版主領譜 PDF, spcl.abs.edu (see sources.txt)
    @grid   engine              # GT beats = the engine's beats; "downbeats 2" = raw downbeats split in 2
    @form   Intro, Verse 1, Chorus > Chorus 2, ...    # recording order; "Template > Label" renames
    [Intro]
    | Abmaj7 | Gm7 | ... |   (H)

``N?`` in a bar = a stretch the published chart does not cover (an instrumental intro, a tag): it
is written as ``X`` in the .lab, which mir_eval and ``chord_metrics`` leave out of every score.

What it does (no listening):
1. expands the templates in ``@form`` order into one chart (``[Label]`` sections);
2. turns the chart into a list of *chart beats* (the chord sounding on each beat);
3. scores every chart chord on every GT beat with the engine's main-layer evidence (the fused,
   calibrated lv-chordia + consonance-ACE scores of ``decode.evidence``), summed over the cells of
   the beat. A chart chord is scored by its *family* (root + triad type + bass): Ebadd9, Eb and
   Ebmaj7 score the same, so the alignment follows where the harmony changes, not the colour;
4. aligns chart beats to GT beats with a DTW (one chart beat per GT beat preferred; a chart beat
   may stretch over extra GT beats — fermatas, ritardandos, tracker extra beats — or two chart
   beats may share one GT beat, each at a cost);
5. writes ``eval/gt/<key>.lab`` (start end Harte), ``.beats`` (time, beat-in-bar of the chart),
   ``.chart.txt`` (the chart with ``[Section @start]`` anchors; an existing hand-made chart is
   kept) and the ``eval/songs.yaml`` entry (``reviewed: false``, ``status: draft``).

``--explore`` instead searches the best *order* of the templates (a section-level Viterbi over the
same evidence) and prints it, to help decide ``@form`` when the chart page does not say it.

Spot check: per GT beat, the GT chord against the engine's Simple chord (Detail 0) at root and
major/minor level — printed, and stored in songs.yaml (``engine_agreement``).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
SHEETS = ROOT / "data" / "sheets"
GT_DIR = ROOT / "eval" / "gt"
SONGS_YAML = ROOT / "eval" / "songs.yaml"
DRAFT_MARK = "# DRAFT ground truth (eval/build_gt.py)"

# DTW costs (nats). Chord costs are per-beat negative log posteriors over the song's chart chords
# (temperature TAU), capped at CAP so one odd beat cannot pull the whole path.
TAU = 3.0
CAP = 6.0
PEN_STRETCH = 1.5        # a GT beat that repeats the previous chart beat
PEN_SQUEEZE = 2.5        # two chart beats on one GT beat
LEAD = 0.4               # per GT beat before the chart starts / after it ends
PEN_SECTION = 4.0        # --explore: starting a new section
X_COST = 1.0             # a chart beat marked unknown ("N?" → X): any chord is as good as any other

_FAMILY = {
    "maj": "maj", "add9": "maj", "maj7": "maj", "6": "maj", "6/9": "maj", "7": "maj", "9": "maj",
    "13": "maj", "maj9": "maj", "7b9": "maj", "7#9": "maj", "7b13": "maj", "maj7#11": "maj",
    "min": "min", "m7": "min", "m6": "min", "m9": "min", "m11": "min", "madd9": "min", "mM7": "min",
    "dim": "dim", "dim7": "dim", "m7b5": "dim",
    "aug": "aug", "+7": "aug", "+maj7": "aug",
    "sus2": "sus2", "sus4": "sus4", "7sus4": "sus4", "9sus4": "sus4", "13sus4": "sus4",
    "5": "5", "N": "N",
}


# ------------------------------------------------------------------------------------- source

@dataclass
class Source:
    key: str
    meta: dict[str, str]
    templates: dict[str, list[str]]        # name → its raw lines (bars + notes), in order
    form: list[tuple[str, str]]            # (template, label)
    path: Path


def read_source(key: str, path: Path | None = None) -> Source:
    path = path or SHEETS / key / "gt_source.txt"
    meta: dict[str, str] = {}
    templates: dict[str, list[str]] = {}
    cur: str | None = None
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            k, _, v = line[1:].partition(" ")
            meta[k.strip()] = v.split(" #")[0].strip()
            continue
        m = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
        if m:
            cur = m.group(1).strip()
            templates[cur] = []
            line = m.group(2).strip()
            if not line:
                continue
        if cur is None:
            raise ValueError(f"{path}: bars before the first [Section]")
        templates[cur].append(line)
    form: list[tuple[str, str]] = []
    for item in (meta.get("form") or ", ".join(templates)).split(","):
        item = item.strip()
        if not item:
            continue
        name, _, label = item.partition(">")
        name, label = name.strip(), (label.strip() or name.strip())
        if name not in templates:
            raise ValueError(f"{path}: @form names unknown section {name!r}")
        form.append((name, label))
    return Source(key, meta, templates, form, path)


def form_text(src: Source, anchors: list[float] | None = None) -> str:
    """The recording-order chart (bar text); with anchors, ``[Label @t]``."""
    out = []
    for i, (name, label) in enumerate(src.form):
        a = f" @{anchors[i]:.2f}" if anchors is not None and anchors[i] is not None else ""
        out.append(f"[{label}{a}]")
        out += src.templates[name]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------- chart beats

@dataclass
class ChartBeats:
    events: list                           # theory.chart.Aligned in chart-beat units (t0 = beat0)
    beat_label: list[str]                  # dominant Harte label per chart beat
    bar_of_beat: np.ndarray
    section_of_bar: list[int]
    labels: list[str]                      # section labels
    bpb: int

    @property
    def K(self) -> int:
        return len(self.beat_label)


def chart_beats(text: str, bpb: int = 4) -> ChartBeats:
    from nathan_chordify.theory import chart as CH
    ac = CH.parse_annotated(text, bpb)
    chart = ac.chart
    n_bars = len(chart.bars)
    K = n_bars * bpb
    lay = CH.layout(chart, np.arange(K + 1, dtype=float), 0.0, None)
    ev = CH.align_chart(chart, np.arange(K + 1, dtype=float), 0.0, None, ac.bar_conf, lay)
    for a in ev:
        if a.harte == "N" and a.unsure:
            a.harte = "X"                      # "N?" = not covered by the chart → unknown
    lab = ["N"] * K
    best = np.zeros(K)
    for a in ev:
        for k in range(int(math.floor(a.beat0 + 1e-6)), int(math.ceil(a.beat1 - 1e-6))):
            ov = min(a.beat1, k + 1) - max(a.beat0, k)
            if ov > best[k] + 1e-9:
                best[k], lab[k] = ov, a.harte
    return ChartBeats(ev, lab, np.repeat(np.arange(n_bars), bpb), lay.section_of_bar,
                      [s.label for s in chart.sections], bpb)


# ------------------------------------------------------------------------------------ families

def family(harte: str) -> tuple:
    from nathan_chordify.theory.harte import parse_harte
    if harte == "X":
        return ("X",)
    r, q, b = parse_harte(harte, strict=False)
    if q == "N":
        return ("N",)
    return (r, _FAMILY.get(q, q), b)


_FAM_CACHE: dict = {}


def family_members() -> dict[tuple, np.ndarray]:
    """Vocabulary candidate indices per family (root, triad type, bass pc)."""
    if not _FAM_CACHE:
        from nathan_chordify.theory.vocab import candidates
        groups: dict[tuple, list[int]] = {}
        for i, c in enumerate(candidates().cands):
            f = ("N",) if c.quality == "N" else (c.root, _FAMILY.get(c.quality, c.quality), c.bass)
            groups.setdefault(f, []).append(i)
        _FAM_CACHE.update({k: np.array(v) for k, v in groups.items()})
    return _FAM_CACHE


def _nearest_family(f: tuple) -> np.ndarray:
    """Members of a family; a family missing from the vocabulary falls back to root position,
    then to the plain major/minor triad of the root."""
    fm = family_members()
    if f in fm:
        return fm[f]
    if len(f) == 3:
        r, t, b = f
        for alt in ((r, t, r), (r, "maj" if t not in ("min", "dim") else "min", b),
                    (r, "maj" if t not in ("min", "dim") else "min", r)):
            if alt in fm:
                return fm[alt]
    return fm[("N",)]


# ------------------------------------------------------------------------------------ evidence

@dataclass
class Audio:
    key: str
    beats: np.ndarray                      # GT beat times (s), the grid the chart is laid on
    duration: float
    cell_t: np.ndarray
    E: np.ndarray                          # [G, V] main-layer evidence (all candidates)
    grid_note: str


def _fill_downbeats(d: np.ndarray) -> np.ndarray:
    """Raw downbeats with gaps of ~2× / 3× the median spacing filled evenly."""
    d = np.asarray(sorted(d), dtype=float)
    med = float(np.median(np.diff(d)))
    out = [d[0]]
    for a, b in zip(d[:-1], d[1:]):
        n = int(round((b - a) / med))
        if n >= 2 and abs((b - a) / n - med) < 0.15 * med:
            out += [a + (b - a) * i / n for i in range(1, n)]
        out.append(b)
    return np.array(out)


def gt_grid(key: str, how: str, grid_beats: np.ndarray, duration: float) -> tuple[np.ndarray, str]:
    """GT beat grid. ``engine``: the engine's beats; ``downbeats N``: the raw tracker's downbeats
    (gaps filled) split into N beats each — for songs the tracker follows at another level."""
    how = (how or "engine").strip()
    if how == "engine":
        b = np.asarray(grid_beats, dtype=float)
        note = "engine beats (rhythm grid of the current analysis)"
    elif how.startswith("downbeats"):
        n = int(how.split()[1])
        raw = json.loads((ROOT / "cache" / key / "rhythm_raw.json").read_text("utf-8"))
        d = _fill_downbeats(np.array(raw["downbeats"], dtype=float))
        b = np.concatenate([np.linspace(a, c, n, endpoint=False) for a, c in zip(d[:-1], d[1:])] + [d[-1:]])
        note = f"raw beat-tracker downbeats (gaps filled) split into {n}"
    else:
        raise ValueError(f"unknown @grid {how!r}")
    # extend to cover the whole recording with the edge periods
    p0, p1 = float(b[1] - b[0]), float(b[-1] - b[-2])
    pre = np.arange(b[0] - p0, -1e-9, -p0)[::-1] if b[0] - p0 > 0.05 else np.zeros(0)
    post = np.arange(b[-1] + p1, duration - 0.05, p1)
    return np.concatenate([pre, b, post]), note


def load_audio(key: str, grid: str = "engine") -> Audio:
    from nathan_chordify import pipeline
    st = pipeline.load_state(key)
    run = pipeline.run_from_state(st, pipeline.params_from(None))
    dur = float(st.meta.get("duration_s") or st.grid.cell_t[-1])
    beats, note = gt_grid(key, grid, st.grid.beats, dur)
    return Audio(key, beats, dur, np.asarray(st.grid.cell_t, float), np.asarray(run.ev_all.E, float), note)


def beat_family_scores(audio: Audio, beats: np.ndarray, fams: list[tuple]) -> np.ndarray:
    """[J, F] per-beat family scores: logsumexp over a family's candidates, summed over the
    cells of each beat by overlap (beats = J+1 boundaries)."""
    ct = audio.cell_t
    G = len(ct) - 1
    fam_cell = np.stack([logsumexp(audio.E[:, _nearest_family(f)], axis=1) for f in fams], axis=1)   # [G, F]
    J = len(beats) - 1
    W = np.zeros((J, G))
    lo = np.searchsorted(ct, beats[:-1], side="right") - 1
    hi = np.searchsorted(ct, beats[1:], side="left")
    for j in range(J):
        for g in range(max(lo[j], 0), min(hi[j], G)):
            ov = min(ct[g + 1], beats[j + 1]) - max(ct[g], beats[j])
            if ov > 0:
                W[j, g] = ov / (ct[g + 1] - ct[g])
    return W @ fam_cell


def cost_matrix(S: np.ndarray, fam_idx: np.ndarray) -> np.ndarray:
    """[K, J] cost of chart beat k (family column fam_idx[k]; -1 = unknown X) on GT beat j."""
    logp = S / TAU - logsumexp(S / TAU, axis=1, keepdims=True)          # [J, F]
    C = np.minimum(-logp[:, np.maximum(fam_idx, 0)].T, CAP)
    C[np.asarray(fam_idx) < 0] = X_COST
    return C


def _families(labels) -> tuple[list[tuple], dict[tuple, int]]:
    fams = sorted({family(h) for h in labels if h != "X"} | {("N",)}, key=str)
    fi = {f: i for i, f in enumerate(fams)}
    fi[("X",)] = -1
    return fams, fi


# ----------------------------------------------------------------------------------------- DTW

def dtw(C: np.ndarray) -> list[tuple[int, int]]:
    """Path [(k, j)] through the [K, J] cost matrix. Steps: (k-1, j-1) free, (k, j-1) stretch,
    (k-1, j) squeeze. The chart may start after GT beat 0 and end before the last GT beat
    (LEAD per skipped beat)."""
    K, J = C.shape
    INF = 1e18
    D = np.full((K, J), INF)
    P = np.zeros((K, J), dtype=np.int8)                  # 0 start, 1 diag, 2 stretch, 3 squeeze
    D[0, :] = C[0, :] + LEAD * np.arange(J)
    for j in range(1, J):
        s = D[0, j - 1] + PEN_STRETCH + C[0, j]
        if s < D[0, j]:
            D[0, j], P[0, j] = s, 2
    for k in range(1, K):
        diag = np.full(J, INF)
        diag[1:] = D[k - 1, :-1]
        sq = D[k - 1, :] + PEN_SQUEEZE
        best = np.minimum(diag, sq)
        how = np.where(diag <= sq, 1, 3).astype(np.int8)
        row = best + C[k]
        # stretch is sequential along j
        for j in range(1, J):
            s = row[j - 1] + PEN_STRETCH + C[k, j]
            if s < row[j]:
                row[j], how[j] = s, 2
        D[k], P[k] = row, how
    end = D[K - 1] + LEAD * (J - 1 - np.arange(J))
    j = int(np.argmin(end))
    k = K - 1
    path = [(k, j)]
    while not (k == 0 and P[k, j] == 0):
        p = P[k, j]
        if p == 1:
            k, j = k - 1, j - 1
        elif p == 2:
            j -= 1
        else:
            k -= 1
        path.append((k, j))
    return path[::-1]


def chart_beat_times(path: list[tuple[int, int]], K: int, beats: np.ndarray) -> np.ndarray:
    """Start time of every chart beat (K+1 values, the last = end of the last chart beat)."""
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for k, j in path:
        first.setdefault(k, j)
        last[k] = j
    T = np.zeros(K + 1)
    k = 0
    while k < K:
        j = first[k]
        group = [kk for kk in range(k, K) if first[kk] == j]       # chart beats squeezed into GT beat j
        n = len(group)
        t0, t1 = beats[j], beats[min(j + 1, len(beats) - 1)]
        if t1 <= t0:
            t1 = t0 + float(np.median(np.diff(beats)))
        for i, kk in enumerate(group):
            T[kk] = t0 + (t1 - t0) * i / n
        k += n
    jl = last[K - 1]
    T[K] = beats[jl + 1] if jl + 1 < len(beats) else beats[jl] + float(np.median(np.diff(beats)))
    return T


def at(T: np.ndarray, x: float) -> float:
    """Time at fractional chart beat x."""
    k = min(int(math.floor(x + 1e-9)), len(T) - 2)
    return float(T[k] + (x - k) * (T[k + 1] - T[k]))


# ------------------------------------------------------------------------------------- explore

def explore(src: Source, audio: Audio, x_cost: float = 2.0, lead: float = 3.0) -> list[tuple[str, float, float, int, int]]:
    """Best concatenation of the templates (any order, repeats allowed) for the recording: a
    section-level Viterbi over the same per-beat costs. Before the first and after the last section
    the recording may idle (``lead`` per beat); unknown (X) chart beats cost ``x_cost`` here so
    that all-unknown templates do not win by default."""
    bpb = int(src.meta.get("bpb", 4))
    names = list(src.templates)
    tb = {n: chart_beats(f"[{n}]\n" + "\n".join(src.templates[n]) + "\n", bpb) for n in names}
    fams, fi = _families([lab for n in names for lab in tb[n].beat_label])
    S = beat_family_scores(audio, audio.beats, fams)
    states = [(n, p) for n in names for p in range(tb[n].K)]
    st_f = np.array([fi[family(tb[n].beat_label[p])] for n, p in states])
    n_s = len(states)
    PRE, POST = n_s, n_s + 1
    is_start = np.array([p == 0 for n, p in states])
    ends = np.array([i for i, (n, p) in enumerate(states) if p == tb[n].K - 1])
    J = len(audio.beats) - 1
    INF = 1e18
    cost = cost_matrix(S, st_f).T                                       # [J, n_s]
    cost[:, st_f < 0] = x_cost
    prev_idx = np.maximum(np.arange(n_s) - 1, 0)
    D = np.full(n_s + 2, INF)
    D[PRE] = lead
    D[:n_s][is_start] = cost[0, is_start]
    back = np.zeros((J, n_s + 2), dtype=np.int64)
    for j in range(1, J):
        e = int(ends[np.argmin(D[ends])])
        src_new, v_new = (e, D[e] + PEN_SECTION) if D[e] + PEN_SECTION < D[PRE] else (PRE, D[PRE])
        diag = np.where(is_start, INF, D[prev_idx])
        stay = D[:n_s] + PEN_STRETCH
        new = np.where(is_start, v_new, INF)
        stack = np.stack([diag, stay, new])
        w = np.argmin(stack, axis=0)
        nd = stack[w, np.arange(n_s)] + cost[j]
        back[j, :n_s] = np.where(w == 0, np.arange(n_s) - 1, np.where(w == 1, np.arange(n_s), src_new))
        back[j, PRE] = PRE
        src_post, v_post = (e, D[e]) if D[e] < D[POST] else (POST, D[POST])
        back[j, POST] = src_post
        D = np.concatenate([nd, [D[PRE] + lead, v_post + lead]])
    e = int(ends[np.argmin(D[ends])])
    s = e if D[e] <= D[POST] else POST
    seq = []
    for j in range(J - 1, -1, -1):
        seq.append(s)
        if j > 0:
            s = int(back[j, s])
    seq = seq[::-1]
    out = []
    for j, s in enumerate(seq):
        if s >= n_s:
            continue
        n, p = states[s]
        if p == 0 and (j == 0 or seq[j - 1] != s):
            out.append([n, audio.beats[j], audio.beats[j + 1], 1, tb[n].K])
        elif out:
            out[-1][2] = audio.beats[j + 1]
            out[-1][3] += 1
    return [tuple(o) for o in out]


# --------------------------------------------------------------------------------------- build

@dataclass
class Built:
    lab: list[tuple[float, float, str]]
    beats: list[tuple[float, int]]
    anchors: list[float]
    agreement: dict
    warnings: list[str] = field(default_factory=list)
    stretch: list[tuple[float, int]] = field(default_factory=list)     # (time, extra GT beats)
    squeeze: list[tuple[float, int]] = field(default_factory=list)


def engine_simple(key: str) -> tuple[np.ndarray, list[str]]:
    """The engine's Simple chords (Detail 0) from the cached analysis.json."""
    from nathan_chordify import schema
    from nathan_chordify.export import flatten
    a = schema.Analysis.model_validate_json((ROOT / "cache" / key / "analysis.json").read_text("utf-8"))
    flat = flatten(a.chords, 0.0)
    return np.array([[c.start, c.end] for c in flat], float).reshape(-1, 2), [c.harte for c in flat]


def agreement(lab, key: str) -> dict:
    """Per GT beat (midpoint of each chord-beat piece): GT chord vs the engine's Simple chord."""
    from eval import chord_metrics as cm
    iv, labels = engine_simple(key)
    n = ok_root = ok_mm = 0
    worst: list[tuple[float, str, str]] = []
    for t0, t1, h in lab:
        if h in ("N", "X"):
            continue
        for tm in np.arange(t0 + 0.25, t1, 0.5):
            i = int(np.searchsorted(iv[:, 0], tm, side="right") - 1)
            est = labels[i] if 0 <= i < len(labels) and iv[i, 1] > tm else "N"
            n += 1
            r = cm.compare(h, est, "root")
            m = cm.compare(h, est, "majmin")
            if m < 0:                                   # outside maj/min (sus, dim...): use the root
                m = r
            ok_root += r == 1
            ok_mm += m == 1
            if r != 1:
                worst.append((round(float(tm), 1), h, est))
    return {"samples_0.5s": n, "root": round(ok_root / max(n, 1), 3), "majmin": round(ok_mm / max(n, 1), 3),
            "root_disagree": worst}


def build(src: Source, audio: Audio | None = None) -> Built:
    bpb = int(src.meta.get("bpb", 4))
    text = form_text(src)
    cb = chart_beats(text, bpb)
    audio = audio or load_audio(src.key, src.meta.get("grid", "engine"))
    fams, fi = _families(cb.beat_label)
    S = beat_family_scores(audio, audio.beats, fams)
    C = cost_matrix(S, np.array([fi[family(h)] for h in cb.beat_label]))
    path = dtw(C)
    T = chart_beat_times(path, cb.K, audio.beats)
    built = Built([], [], [], {})
    # stretches / squeezes (for the report)
    from collections import Counter
    per_k = Counter(k for k, _ in path)
    per_j = Counter(j for _, j in path)
    for k, n in sorted(per_k.items()):
        if n > 1:
            built.stretch.append((round(float(T[k]), 2), n - 1))
    for j, n in sorted(per_j.items()):
        if n > 1:
            built.squeeze.append((round(float(audio.beats[j]), 2), n - 1))
    # chords
    lab: list[tuple[float, float, str]] = []
    for a in cb.events:
        t0, t1 = at(T, a.beat0), at(T, a.beat1)
        if t1 - t0 < 0.02:
            continue
        lab.append((round(t0, 3), round(t1, 3), a.harte))
    lab.sort()
    merged: list[tuple[float, float, str]] = []
    for t0, t1, h in lab:              # re-struck chords (e.g. "%" bars) stay separate; unknown stretches merge
        if merged and h == "X" and merged[-1][2] == "X" and abs(merged[-1][1] - t0) < 1e-3:
            merged[-1] = (merged[-1][0], t1, "X")
        else:
            merged.append((t0, t1, h))
    if merged and merged[0][0] > 0.05:
        merged.insert(0, (0.0, merged[0][0], "N"))
    if merged and audio.duration - merged[-1][1] > 0.05:
        merged.append((merged[-1][1], round(audio.duration, 3), "N"))
    built.lab = merged
    # beats: chart beats with the beat in the bar
    built.beats = [(round(float(T[k]), 4), int(k % bpb) + 1) for k in range(cb.K)]
    # section anchors
    first_bar: dict[int, int] = {}
    for bar, si in enumerate(cb.section_of_bar):
        first_bar.setdefault(si, bar)
    built.anchors = [round(float(T[first_bar[si] * bpb]), 2) for si in range(len(cb.labels))]
    built.agreement = agreement(built.lab, src.key)
    return built


def write(src: Source, b: Built, audio: Audio) -> None:
    from nathan_chordify.theory import chart as CH
    from nathan_chordify.server.gt import update_songs_yaml
    key = src.key
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    GT_DIR.mkdir(parents=True, exist_ok=True)
    (GT_DIR / f"{key}.lab").write_text("".join(f"{t0:.3f}\t{t1:.3f}\t{h}\n" for t0, t1, h in b.lab), "utf-8")
    head = (f"{DRAFT_MARK} {now} — NOT human-reviewed. Beats of the chart laid on: {audio.grid_note}.\n"
            f"# time\tbeat-in-bar (chart bars of {src.meta.get('bpb', 4)} beats)\n")
    (GT_DIR / f"{key}.beats").write_text(head + "".join(f"{t:.4f}\t{n}\n" for t, n in b.beats), "utf-8")
    chart_p = GT_DIR / f"{key}.chart.txt"
    body = form_text(src, b.anchors)
    CH.parse_annotated(body, int(src.meta.get("bpb", 4)))           # must round-trip
    hdr = [f"{DRAFT_MARK} {now} — NOT human-reviewed.",
           f"# {src.meta.get('title', key)} — {key}",
           f"# Key {src.meta.get('key', '?')}; sheet/chart: {src.meta.get('source', '?')}",
           f"# Meter {src.meta.get('meter', '4/4')}; one token = one beat ({src.meta.get('beat', 'quarter note')}).",
           "# Sections are in the order the RECORDING plays them; '@t' = start time (s) found by aligning",
           "# the chart to the engine's evidence (eval/build_gt.py). (H)/(M)/(L) = confidence of the line.",
           "# 'N?' = a stretch the published chart does not cover: X (unknown, not scored) in the .lab.",
           f"# Source file with notes: data/sheets/{key}/gt_source.txt", ""]
    aligned_copy = None
    if chart_p.exists() and not chart_p.read_text("utf-8").startswith(DRAFT_MARK):
        aligned_copy = GT_DIR / f"{key}.chart.aligned.txt"           # keep the hand-made chart
        aligned_copy.write_text("\n".join(hdr) + body, "utf-8")
    else:
        chart_p.write_text("\n".join(hdr) + body, "utf-8")
    entry = {"key": key, "title": src.meta.get("title", key), "set": "D1", "reviewed": False,
             "status": "draft", "lab": f"eval/gt/{key}.lab", "beats": f"eval/gt/{key}.beats",
             "chart": f"eval/gt/{key}.chart.txt", "tapped_beats": False,
             "source_url": f"https://www.youtube.com/watch?v={key[3:]}" if key.startswith("yt-") else None,
             "key_sig": src.meta.get("key"), "meter": src.meta.get("meter", "4/4"),
             "sheet_source": src.meta.get("source"), "sheet_files": f"data/sheets/{key}/",
             "aligned_by": "eval/build_gt.py (DTW of the chart to the engine's main-layer evidence)",
             "grid": audio.grid_note, "confidence": src.meta.get("confidence"),
             "n_chords": sum(1 for _, _, h in b.lab if h not in ("N", "X")),
             "unknown_s": round(sum(t1 - t0 for t0, t1, h in b.lab if h == "X"), 1),
             "engine_agreement": {k: v for k, v in b.agreement.items() if k != "root_disagree"},
             "updated": now}
    if aligned_copy is not None:
        entry["chart_aligned"] = f"eval/gt/{aligned_copy.name}"
    update_songs_yaml(ROOT / "eval", entry)


def report(src: Source, b: Built) -> str:
    out = [f"{src.key}: {len(b.lab)} segments, {len(b.beats)} chart beats; sections:"]
    out += [f"  {lab:<22} @{t:7.2f}" for (_, lab), t in zip(src.form, b.anchors)]
    out.append(f"  stretched chart beats (time, extra GT beats): {b.stretch[:20]}{' …' if len(b.stretch) > 20 else ''}")
    out.append(f"  squeezed GT beats (time, extra chart beats): {b.squeeze[:20]}{' …' if len(b.squeeze) > 20 else ''}")
    ag = b.agreement
    out.append(f"  engine Simple agrees: root {ag['root']:.1%}, maj/min {ag['majmin']:.1%} of {ag['samples_0.5s']} half-second samples")
    # runs of root disagreement (≥ 2 s)
    runs = []
    for t, h, e in ag["root_disagree"]:
        if runs and t - runs[-1][1] <= 0.6 and runs[-1][2] == h:
            runs[-1][1] = t
        else:
            runs.append([t, t, h, e])
    long_ = [r for r in runs if r[1] - r[0] >= 1.9]
    out.append("  longest disagreements: " + "; ".join(f"{r[0]:.1f}-{r[1] + 0.5:.1f}s chart {r[2]} vs engine {r[3]}"
                                                     for r in sorted(long_, key=lambda r: r[0] - r[1])[:8]))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m eval.build_gt")
    ap.add_argument("key", help="cache key with data/sheets/<key>/gt_source.txt, or 'all'")
    ap.add_argument("--explore", action="store_true", help="search the section order instead of building")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    keys = sorted(p.parent.name for p in SHEETS.glob("*/gt_source.txt")) if a.key == "all" else [a.key]
    for key in keys:
        src = read_source(key)
        audio = load_audio(key, src.meta.get("grid", "engine"))
        if a.explore:
            for n, t0, t1, used, k in explore(src, audio):
                print(f"  {n:<20} {t0:7.2f}-{t1:7.2f}  {used:3d} GT beats for {k:3d} chart beats")
            continue
        b = build(src, audio)
        print(report(src, b))
        if not a.no_write:
            write(src, b, audio)
            print(f"  wrote eval/gt/{key}.lab/.beats/.chart.txt and eval/songs.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
