"""Line a chord chart up with a recording, without human listening (moved from eval/build_gt.py,
2026-11-03, plus the 12 transpositions and the match verdict).

The chart is a bar-text chart (``theory.chart``); its sections are *templates* that a recording
may play in any order and repeat (charts list each section once; recordings repeat the chorus).

1. Chart chords are scored by *family* (root + triad type + bass): Ebadd9, Eb and Ebmaj7 score
   the same, so the alignment follows where the harmony changes, not the colour. The score of a
   family on a beat is the log-sum-exp of the engine's fused, calibrated main-layer evidence
   (``decode.evidence``, read from the cached decode state) over the family's candidates, summed
   over the cells of the beat. Without a decode state (songs analysed before it existed, tests),
   the detected chords of ``analysis.json`` stand in as a soft one-hot evidence.
2. ``explore``: a section-level Viterbi finds the ORDER and REPEATS of the templates on the beat
   grid (idle beats before, between and after sections cost ``IDLE`` each), for every
   transposition of the chart (the written key is only a hint). The transposition whose path
   fits best wins.
3. The formed chart (templates in that order, unknown ``N?`` bars for the gaps) is aligned beat
   by beat with a DTW (stretch / squeeze / lead costs), which gives every chart chord its time.
4. ``fit`` = mean posterior (at temperature 1, over the chart's own families) of the aligned
   chord on each chart beat; the same for a chart with its chord events shuffled (order destroyed,
   chord histogram kept) is the baseline. ``score = (fit − fit_shuffled) / (1 − fit_shuffled)``;
   verdict "good" / "maybe" (another version or arrangement) / "different" (a different song).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .. import schema
from ..theory import chart as CH
from ..theory import harte as H
from ..theory import simplify as SM
from ..theory.display import key_index as _key_index

# DTW costs (nats). Chord costs are per-beat negative log posteriors over the song's chart chords
# (temperature TAU), capped at CAP so one odd beat cannot pull the whole path.
TAU = 3.0
CAP = 6.0
PEN_STRETCH = 1.5        # a GT beat that repeats the previous chart beat
PEN_SQUEEZE = 2.5        # two chart beats on one GT beat
LEAD = 0.4               # per GT beat before the chart starts / after it ends (DTW)
PEN_SECTION = 4.0        # explore: starting a new section after another
X_COST = 1.0             # a chart beat marked unknown ("N?" → X): any chord is as good as any other
IDLE = 2.0               # explore: a recording beat with no chart section on it
MIN_SECTION_BEATS = 4    # explore: a template shorter than this is padded (one bar at least)
GOOD_SCORE = 0.40        # verdict thresholds on the coverage-weighted normalised score (module doc)
MAYBE_SCORE = 0.25
GOOD_FIT = 0.45          # ... and on the raw fit
MAYBE_FIT = 0.60
COVER_FULL = 0.6         # a chart covering this share of the recording's beats counts as covering it
PSEUDO_P = 0.75          # audio_from_analysis: posterior mass of the detected family in a cell

_FAMILY = {
    "maj": "maj", "add9": "maj", "maj7": "maj", "6": "maj", "6/9": "maj", "7": "maj", "9": "maj",
    "13": "maj", "maj9": "maj", "7b9": "maj", "7#9": "maj", "7b13": "maj", "maj7#11": "maj",
    "min": "min", "m7": "min", "m6": "min", "m9": "min", "m11": "min", "madd9": "min", "mM7": "min",
    "dim": "dim", "dim7": "dim", "m7b5": "dim",
    "aug": "aug", "+7": "aug", "+maj7": "aug",
    "sus2": "sus2", "sus4": "sus4", "7sus4": "sus4", "9sus4": "sus4", "13sus4": "sus4",
    "5": "5", "N": "N",
}


# ------------------------------------------------------------------------------------ families

def family(harte: str) -> tuple:
    if harte == "X":
        return ("X",)
    r, q, b = H.parse_harte(harte, strict=False)
    if q == "N":
        return ("N",)
    return (r, _FAMILY.get(q, q), b)


def transpose_harte(harte: str, t: int) -> str:
    """A Harte label moved up ``t`` semitones (default root names); N / X unchanged."""
    if harte in ("N", "X") or t % 12 == 0:
        return harte
    r, q, b = H.parse_harte(harte, strict=False)
    if r < 0:
        return harte
    return H.to_harte((r + t) % 12, q, (b + t) % 12)


_FAM_CACHE: dict = {}


def family_members() -> dict[tuple, np.ndarray]:
    """Vocabulary candidate indices per family (root, triad type, bass pc)."""
    if not _FAM_CACHE:
        from ..theory.vocab import candidates
        groups: dict[tuple, list[int]] = {}
        for i, c in enumerate(candidates().cands):
            f = ("N",) if c.quality == "N" else (c.root, _FAMILY.get(c.quality, c.quality), c.bass)
            groups.setdefault(f, []).append(i)
        _FAM_CACHE.update({k: np.array(v) for k, v in groups.items()})
    return _FAM_CACHE


def nearest_family(f: tuple) -> np.ndarray:
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


# --------------------------------------------------------------------------------- chart beats

@dataclass
class ChartBeats:
    events: list                           # theory.chart.Aligned in chart-beat units (t0 = beat0)
    beat_label: list[str]                  # dominant Harte label per chart beat
    bar_of_beat: np.ndarray
    section_of_bar: list[int]
    labels: list[str | None]               # section labels
    bpb: int

    @property
    def K(self) -> int:
        return len(self.beat_label)


def chart_beats(text: str, bpb: int = 4) -> ChartBeats:
    """A chart laid on its own integer beat grid: one label per chart beat (``N?`` → X)."""
    ac = CH.parse_annotated(text, bpb)
    chart = ac.chart
    n_bars = len(chart.bars)
    K = max(n_bars * bpb, 1)
    lay = CH.layout(chart, np.arange(K + 1, dtype=float), 0.0, None)
    ev = CH.align_chart(chart, np.arange(K + 1, dtype=float), 0.0, None, ac.bar_conf, lay)
    for a in ev:
        if a.harte == "N" and a.unsure:
            a.harte = "X"                      # "N?" = not covered by the chart → unknown
    lab = ["N"] * K
    best = np.zeros(K)
    for a in ev:
        for k in range(int(math.floor(a.beat0 + 1e-6)), min(K, int(math.ceil(a.beat1 - 1e-6)))):
            ov = min(a.beat1, k + 1) - max(a.beat0, k)
            if ov > best[k] + 1e-9:
                best[k], lab[k] = ov, a.harte
    return ChartBeats(ev, lab, np.repeat(np.arange(n_bars), bpb), lay.section_of_bar,
                      [s.label for s in chart.sections], bpb)


_LABEL_LINE = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$")


def split_templates(text: str) -> list[tuple[str, list[str]]]:
    """A chart's sections as (name, bar lines) templates. Unlabelled bars before the first label
    are "Part"; a repeated label with the same bars is one template; a repeated label with other
    bars gets a number ("Chorus (2)")."""
    out: list[tuple[str, list[str]]] = []
    cur: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LABEL_LINE.match(line)
        if m:
            cur = re.sub(r"\s*@.*$", "", m.group(1).strip()) or "Part"
            out.append((cur, []))
            line = m.group(2).strip()
            if "|" not in line:
                continue
        if cur is None:
            cur = "Part"
            out.append((cur, []))
        if "|" in line:
            out[-1][1].append(line)
    out = [(n, ls) for n, ls in out if ls]
    merged: list[tuple[str, list[str]]] = []
    for name, lines in out:
        if any(n == name and ls == lines for n, ls in merged):
            continue                                       # the same section written twice
        k, nm = 2, name
        while any(n == nm for n, _ in merged):
            nm = f"{name} ({k})"
            k += 1
        merged.append((nm, lines))
    return merged


def transpose_chart_text(text: str, t: int, key: int | None = None, bpb: int = 4) -> str:
    """The bar text moved up ``t`` semitones (chords spelled in ``key`` when given; section labels
    kept, anchors and confidence notes dropped)."""
    ac = CH.parse_annotated(text, bpb)
    chart = ac.chart
    from ..types import ChartChord
    for s in chart.sections:
        for bi, bar in enumerate(s.bars):
            new = []
            for c in bar:
                if c.harte in ("N", "X"):
                    new.append(c)
                    continue
                r, q, b = H.parse_harte(c.harte, strict=False)
                r2, b2 = (r + t) % 12, (b + t) % 12
                h = H.to_harte_in_key(r2, q, b2, key) if key is not None else H.to_harte(r2, q, b2)
                new.append(ChartChord(c.pos, h))
            s.bars[bi] = new
    return CH.format_chart(chart)


# ------------------------------------------------------------------------------------ evidence

@dataclass
class Audio:
    key: str
    beats: np.ndarray                      # beat times (s), the grid the chart is laid on
    duration: float
    cell_t: np.ndarray                     # [G+1]
    E: np.ndarray                          # [G, V] main-layer evidence over the whole vocabulary
    grid_note: str
    key_index: int = 0                     # recording key 0..23 (0–11 major, 12–23 minor)
    key_segments: list[tuple[float, float, int]] = field(default_factory=list)
    source: str = "state"                  # "state" (decode evidence) | "analysis" (detected chords)

    def key_at(self, t: float) -> int:
        for a, b, k in self.key_segments:
            if a <= t < b:
                return k
        return self.key_index


def _extend_beats(b: np.ndarray, duration: float) -> np.ndarray:
    """The beat list extended to cover the whole recording with the edge periods."""
    b = np.asarray(b, dtype=float)
    if len(b) < 2:
        step = 0.5 if duration <= 0 else max(duration / max(int(duration / 0.5), 1), 0.2)
        return np.arange(0.0, duration + 1e-9, step)
    p0, p1 = float(b[1] - b[0]), float(b[-1] - b[-2])
    pre = np.arange(b[0] - p0, -1e-9, -p0)[::-1] if b[0] - p0 > 0.05 else np.zeros(0)
    post = np.arange(b[-1] + p1, duration - 0.05, p1) if p1 > 1e-3 else np.zeros(0)
    return np.concatenate([pre, b, post])


def _key_segments(a: schema.Analysis) -> list[tuple[float, float, int]]:
    return [(s.start, s.end, _key_index(s.tonic, s.mode)) for s in a.key.segments]


def audio_from_state(st, analysis: schema.Analysis | None = None, params=None) -> Audio:
    """The engine's fused main-layer evidence from a decode state (``pipeline.load_state``)."""
    from .. import pipeline
    run = pipeline.run_from_state(st, pipeline.params_from(params))
    dur = float(st.meta.get("duration_s") or st.grid.cell_t[-1])
    beats = _extend_beats(st.grid.beats, dur)
    key = st.key0.global_tonic + (12 if st.key0.global_mode == "minor" else 0)
    segs: list[tuple[float, float, int]] = []
    if analysis is not None:
        key = _key_index(analysis.key.global_.tonic, analysis.key.global_.mode)
        segs = _key_segments(analysis)
    return Audio(st.key, beats, dur, np.asarray(st.grid.cell_t, float), np.asarray(run.ev_all.E, float),
                 "engine beats (decode state)", key, segs, "state")


def audio_from_analysis(a: schema.Analysis, cells_per_beat: int = 2) -> Audio:
    """Pseudo-evidence from the detected chords (for songs without a decode state): every cell gives
    ``PSEUDO_P`` of its posterior mass to the family of the chord sounding there, the rest spread
    over the vocabulary (weighted by the chord's confidence)."""
    from ..theory.vocab import candidates
    V = len(candidates().cands)
    dur = float(a.source.duration)
    det = np.asarray(sorted(b.t for b in a.beats if b.bar >= 0), dtype=float)
    beats = _extend_beats(det, dur)
    cell_t = np.unique(np.concatenate([np.linspace(beats[i], beats[i + 1], cells_per_beat, endpoint=False)
                                       for i in range(len(beats) - 1)] + [beats[-1:]]))
    G = len(cell_t) - 1
    E = np.full((G, V), math.log((1 - PSEUDO_P) / max(V - 1, 1)))
    mains = sorted((c for c in a.chords if c.layer == "main"), key=lambda c: c.start)
    mid = 0.5 * (cell_t[:-1] + cell_t[1:])
    idx = np.searchsorted([c.start for c in mains], mid, side="right") - 1
    for g in range(G):
        i = int(idx[g])
        if i < 0 or mains[i].end <= mid[g]:
            continue
        c = mains[i]
        f = family(c.harte)
        p = PSEUDO_P * (0.5 + 0.5 * float(c.conf))
        members = nearest_family(f)
        E[g, members] = math.log(max(p, 1e-6) / len(members))
    key = _key_index(a.key.global_.tonic, a.key.global_.mode)
    return Audio(a.source.key, beats, dur, cell_t, E, "detected beats (analysis.json)", key, _key_segments(a), "analysis")


def load_audio(song_dir: Path, analysis: schema.Analysis | None = None, params=None) -> Audio:
    """The best available evidence for a song folder: its decode state, else its analysis."""
    from .. import pipeline
    d = Path(song_dir)
    if analysis is None and (d / "analysis.json").exists():
        analysis = schema.Analysis.model_validate_json((d / "analysis.json").read_text("utf-8"))
    if (d / pipeline.STATE_NAME).exists():
        try:
            return audio_from_state(pipeline.load_state(d), analysis, params)
        except Exception:  # noqa: BLE001 - an unreadable state (old version): fall back to the chords
            pass
    if analysis is None:
        raise FileNotFoundError(f"no analysis in {d}")
    return audio_from_analysis(analysis)


def beat_family_scores(audio: Audio, beats: np.ndarray, fams: list[tuple]) -> np.ndarray:
    """[J, F] per-beat family scores: logsumexp over a family's candidates, summed over the
    cells of each beat by overlap (beats = J+1 boundaries)."""
    ct = audio.cell_t
    G = len(ct) - 1
    fam_cell = np.stack([logsumexp(audio.E[:, nearest_family(f)], axis=1) for f in fams], axis=1)   # [G, F]
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
    """[K, J] cost of chart beat k (family column fam_idx[k]; -1 = unknown X) on beat j."""
    logp = S / TAU - logsumexp(S / TAU, axis=1, keepdims=True)          # [J, F]
    C = np.minimum(-logp[:, np.maximum(fam_idx, 0)].T, CAP)
    C[np.asarray(fam_idx) < 0] = X_COST
    return C


def posterior(S: np.ndarray) -> np.ndarray:
    """[J, F] posterior at temperature 1 over the chart's families (the ``fit`` measure)."""
    return np.exp(S - logsumexp(S, axis=1, keepdims=True))


def _families(labels) -> tuple[list[tuple], dict[tuple, int]]:
    fams = sorted({family(h) for h in labels if h != "X"} | {("N",)}, key=str)
    fi = {f: i for i, f in enumerate(fams)}
    fi[("X",)] = -1
    return fams, fi


# ----------------------------------------------------------------------------------------- DTW

def dtw(C: np.ndarray) -> list[tuple[int, int]]:
    """Path [(k, j)] through the [K, J] cost matrix. Steps: (k-1, j-1) free, (k, j-1) stretch,
    (k-1, j) squeeze. The chart may start after beat 0 and end before the last beat (LEAD per
    skipped beat)."""
    K, J = C.shape
    INF = 1e18
    D = np.full((K, J), INF)
    P = np.zeros((K, J), dtype=np.int8)                  # 0 start, 1 diag, 2 stretch, 3 squeeze

    def stretch(row: np.ndarray, how: np.ndarray, k: int) -> None:
        # row[j] = min(row[j], row[j-1] + PEN_STRETCH + C[k, j]) along j: a running minimum
        s = np.cumsum(PEN_STRETCH + C[k])
        m = np.minimum.accumulate(row - s)
        new = m + s
        use = new < row - 1e-12
        row[use] = new[use]
        how[use] = 2

    row0 = C[0] + LEAD * np.arange(J)
    how0 = np.zeros(J, dtype=np.int8)
    stretch(row0, how0, 0)
    D[0], P[0] = row0, how0
    for k in range(1, K):
        diag = np.full(J, INF)
        diag[1:] = D[k - 1, :-1]
        sq = D[k - 1, :] + PEN_SQUEEZE
        best = np.minimum(diag, sq)
        how = np.where(diag <= sq, 1, 3).astype(np.int8)
        row = best + C[k]
        stretch(row, how, k)
        D[k], P[k] = row, how
    end = D[K - 1] + LEAD * (J - 1 - np.arange(J))
    j = int(np.argmin(end))
    k = K - 1
    path = [(k, j)]
    while not (k == 0 and (P[k, j] == 0 or j == 0)):
        p = P[k, j]
        if p == 1:
            k, j = k - 1, j - 1
        elif p == 2:
            j -= 1
        elif p == 3:
            k -= 1
        else:
            break
        path.append((k, j))
    return path[::-1]


def chart_beat_times(path: list[tuple[int, int]], K: int, beats: np.ndarray) -> np.ndarray:
    """Start time of every chart beat (K+1 values, the last = end of the last chart beat)."""
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for k, j in path:
        first.setdefault(k, j)
        last[k] = j
    med = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    T = np.zeros(K + 1)
    k = 0
    while k < K:
        j = first.get(k, last.get(k - 1, 0))
        group = [kk for kk in range(k, K) if first.get(kk) == j]       # chart beats squeezed into beat j
        if not group:
            group = [k]
        n = len(group)
        t0 = beats[min(j, len(beats) - 1)]
        t1 = beats[j + 1] if j + 1 < len(beats) else t0 + med
        if t1 <= t0:
            t1 = t0 + med
        for i, kk in enumerate(group):
            T[kk] = t0 + (t1 - t0) * i / n
        k += n
    jl = last.get(K - 1, len(beats) - 2)
    T[K] = beats[jl + 1] if jl + 1 < len(beats) else beats[min(jl, len(beats) - 1)] + med
    return T


def at(T: np.ndarray, x: float) -> float:
    """Time at fractional chart beat x."""
    k = min(max(int(math.floor(x + 1e-9)), 0), len(T) - 2)
    return float(T[k] + (x - k) * (T[k + 1] - T[k]))


# ------------------------------------------------------------------------------------- explore

@dataclass
class Explored:
    form: list[tuple[str, int, int]]       # (template name | "", first beat j0, end beat j1); "" = idle
    state_of_beat: np.ndarray              # [J] chart state index per recording beat (−1 = idle)
    cost: float


EX_STRETCH = 0.5         # explore: a recording beat that repeats the previous chart beat (bars inferred
EX_SQUEEZE = 1.0         #          from lyrics are rough, so durations are cheap to bend) / a skipped chart beat


def explore(tb: dict[str, ChartBeats], st_fam: np.ndarray, S: np.ndarray, idle: float | None = None,
            pen_section: float | None = None) -> Explored:
    """Best concatenation of the templates (any order, repeats allowed, idle beats before /
    between / after) for the recording: a section-level Viterbi over per-beat family costs.

    ``tb``: templates; ``st_fam``: family column (−1 = X) of every chart state, states being
    (template, chart beat) in ``tb`` order; ``S``: [J, F] beat family scores. Inside a template
    the path moves one chart beat per recording beat, stays (EX_STRETCH) or skips one chart beat
    (EX_SQUEEZE); an unknown (X) chart beat costs as much as idling, so it never attracts."""
    idle = IDLE if idle is None else idle
    pen_section = PEN_SECTION if pen_section is None else pen_section
    names = list(tb)
    states = [(n, p) for n in names for p in range(tb[n].K)]
    n_s = len(states)
    J = S.shape[0]
    INF = 1e18
    pos = np.array([p for n, p in states])
    is_start = pos == 0
    ends = np.array([i for i, (n, p) in enumerate(states) if p == tb[n].K - 1])
    cost = cost_matrix(S, st_fam).T                                       # [J, n_s]
    cost[:, st_fam < 0] = idle
    prev_idx = np.maximum(np.arange(n_s) - 1, 0)
    prev2_idx = np.maximum(np.arange(n_s) - 2, 0)
    # two idle states: PRE (nothing played yet: a section may start for free) and IDLE (between /
    # after sections: starting the next section costs PEN_SECTION, like a direct section change,
    # so a one-beat gap cannot undercut it)
    PRE_S, IDLE_S = n_s, n_s + 1
    D = np.full(n_s + 2, INF)
    D[PRE_S] = idle
    D[:n_s][is_start] = cost[0, is_start]
    back = np.zeros((J, n_s + 2), dtype=np.int64)
    back[0, :] = -1
    for j in range(1, J):
        e = int(ends[np.argmin(D[ends])])
        # a new section may start after another section or after idling (PEN_SECTION), or after PRE
        opts = ((D[e] + pen_section, e), (D[IDLE_S] + pen_section, IDLE_S), (D[PRE_S], PRE_S))
        v_new, src_new = min(opts)
        diag = np.where(is_start, INF, D[prev_idx])
        stay = D[:n_s] + EX_STRETCH
        skip = np.where(pos >= 2, D[prev2_idx] + EX_SQUEEZE, INF)
        new = np.where(is_start, v_new, INF)
        stack = np.stack([diag, stay, new, skip])
        w = np.argmin(stack, axis=0)
        nd = stack[w, np.arange(n_s)] + cost[j]
        back[j, :n_s] = np.select([w == 0, w == 1, w == 2], [np.arange(n_s) - 1, np.arange(n_s), np.full(n_s, src_new)],
                                  np.arange(n_s) - 2)
        # idle: keep idling, or a section just ended
        src_idle, v_idle = (e, D[e]) if D[e] < D[IDLE_S] else (IDLE_S, D[IDLE_S])
        back[j, IDLE_S] = src_idle
        back[j, PRE_S] = PRE_S
        D = np.concatenate([nd, [D[PRE_S] + idle, v_idle + idle]])
    e = int(ends[np.argmin(D[ends])])
    s = e if D[e] <= D[IDLE_S] else IDLE_S
    total = float(min(D[e], D[IDLE_S], D[PRE_S]))
    if D[PRE_S] < min(D[e], D[IDLE_S]):
        s = PRE_S
    seq = np.zeros(J, dtype=np.int64)
    for j in range(J - 1, -1, -1):
        seq[j] = s
        if j > 0:
            s = int(back[j, s])
    state_of_beat = np.where(seq >= n_s, -1, seq)
    form: list[list] = []
    for j in range(J):
        s = int(seq[j])
        if s >= n_s:
            if form and form[-1][0] == "":
                form[-1][2] = j + 1
            else:
                form.append(["", j, j + 1])
            continue
        n, p = states[s]
        if p == 0 and (j == 0 or seq[j - 1] != s):
            form.append([n, j, j + 1])
        elif form and form[-1][0] == n:
            form[-1][2] = j + 1
        else:
            form.append([n, j, j + 1])
    return Explored([tuple(f) for f in form], state_of_beat, total)


# --------------------------------------------------------------------------------------- align

@dataclass
class Aligned:
    transpose: int                          # semitones added to the chart (−5..6)
    score: float                            # normalised match 0..1 (fit vs the shuffled baseline)
    fit: float                              # mean aligned-family posterior
    baseline: float                         # the same for the shuffled chart
    verdict: str                            # "good" | "maybe" | "different"
    chords: list[tuple[float, float, str]]  # (t0, t1, Harte in the recording's key)
    sections: list[tuple[float, str]]       # (start time, label)
    warnings: list[str] = field(default_factory=list)
    form: list[tuple[str, int, int]] = field(default_factory=list)
    coverage: float = 0.0                   # share of the recording's beats under a chart section
    bpb: int = 4
    fits: dict[int, float] = field(default_factory=dict)   # fit per transposition tried
    key_recording: int = 0
    beat_scale: float = 1.0                 # recording beats per chart beat (1, 2 or 0.5)
    regularity: float = 0.0                 # share of alignment steps with no stretch or squeeze

    def chart_text(self, text: str) -> str:
        """The input chart moved to the recording's key."""
        return transpose_chart_text(text, self.transpose, self.key_recording, self.bpb)


def _norm_t(t: int) -> int:
    t %= 12
    return t - 12 if t > 6 else t


def _half_beats(beats: np.ndarray) -> np.ndarray:
    """The beat list with a midpoint between every two beats (chart beat = half a recording beat)."""
    b = np.asarray(beats, dtype=float)
    if len(b) < 2:
        return b
    mids = 0.5 * (b[:-1] + b[1:])
    return np.sort(np.concatenate([b, mids]))


def _form_text(form: list[tuple[str, int, int]], templates: dict[str, list[str]], bpb: int, t: int) -> str:
    """The formed chart (templates in recording order, ``N?`` bars for idle stretches ≥ a bar)."""
    out: list[str] = []
    for name, j0, j1 in form:
        if name == "":
            n_bars = int(round((j1 - j0) / bpb))
            if n_bars >= 1:
                out.append("[Gap]")
                out.append("| " + " | ".join(["N?"] * n_bars) + " |")
            continue
        out.append(f"[{name}]")
        out += templates[name]
    text = "\n".join(out) + "\n"
    return transpose_chart_text(text, t, None, bpb) if t % 12 else text


def _shuffled_labels(cb: ChartBeats, rng: np.random.Generator) -> list[str]:
    """The chart's beat labels with the chord EVENTS shuffled (durations kept, order destroyed)."""
    runs: list[tuple[str, int]] = []
    for lab in cb.beat_label:
        if runs and runs[-1][0] == lab:
            runs[-1] = (lab, runs[-1][1] + 1)
        else:
            runs.append((lab, 1))
    chords = [r for r in runs if r[0] != "X"]
    if len(chords) < 3:
        return list(cb.beat_label)
    perm = rng.permutation(len(chords))
    labels = [chords[i][0] for i in perm]
    out: list[str] = []
    ci = 0
    for lab, n in runs:
        if lab == "X":
            out += ["X"] * n
        else:
            out += [labels[ci]] * n
            ci += 1
    return out


def _fit_of_path(path: list[tuple[int, int]], labels: list[str], P1: np.ndarray, fi: dict[tuple, int]) -> float:
    vals = []
    seen: set[int] = set()
    for k, j in path:
        if k in seen:
            continue
        seen.add(k)
        f = fi.get(family(labels[k]), -1)
        if f >= 0:
            vals.append(float(P1[min(j, P1.shape[0] - 1), f]))
    return float(np.mean(vals)) if vals else 0.0


def verdict_of(score: float, fit: float, coverage: float = 1.0) -> str:
    """"good" = this chart follows the recording; "maybe" = partly (another version / arrangement,
    or a chart covering only part of the song); "different" = a different song."""
    if score >= GOOD_SCORE and fit >= GOOD_FIT and coverage >= 0.5:
        return "good"
    if score >= MAYBE_SCORE and fit >= MAYBE_FIT:
        return "maybe"
    return "different"


def _subset(S: np.ndarray, fi: dict[tuple, int], fams: list[tuple]) -> tuple[np.ndarray, dict[tuple, int]]:
    """The columns of S for these families (+ N): the posterior is taken over the chart's OWN
    chords at one transposition, not over every transposition's chords at once."""
    cols = sorted({fi[f] for f in fams if f in fi} | {fi[("N",)]})
    remap = {c: i for i, c in enumerate(cols)}
    fi_t = {f: remap[c] for f, c in fi.items() if c in remap}
    fi_t[("X",)] = -1
    return S[:, cols], fi_t


@dataclass
class _Stage:
    """One alignment hypothesis (transposition + beat scale) taken through the DTW."""
    t: int
    scale: float
    ex: Explored
    beats: np.ndarray
    cb: ChartBeats
    path: list[tuple[int, int]]
    T: np.ndarray
    fit: float
    baseline: float
    score: float
    regularity: float = 0.0                # share of DTW steps that advance both chart and recording


def _regularity(path: list[tuple[int, int]]) -> float:
    steps = [(k1 - k0, j1 - j0) for (k0, j0), (k1, j1) in zip(path, path[1:])]
    return sum(1 for dk, dj in steps if dk == 1 and dj == 1) / max(len(steps), 1)


def _dtw_stage(ex: Explored, t: int, scale: float, beats: np.ndarray, S: np.ndarray, fi: dict, fams: list,
               templates: dict[str, list[str]], bpb: int, audio: Audio, seed: int) -> _Stage:
    """The formed chart (templates in the explored order, gaps as unknown bars) aligned beat by
    beat; ``fit`` along the path and the shuffled baseline, posteriors over this chart's own chords."""
    form_text = _form_text(ex.form, templates, bpb, t)
    cb = chart_beats(form_text, bpb)
    fam_cb = [family(h) for h in cb.beat_label]
    missing = {f for f in fam_cb if f != ("X",) and f not in fi}
    if missing:                                                      # a sub-beat chord not in the templates' beat labels
        fams = sorted(set(fams) | missing, key=str)
        fi = {f: i for i, f in enumerate(fams)}
        fi[("X",)] = -1
        S = beat_family_scores(audio, beats, fams)
    S_t, fi_t = _subset(S, fi, fam_cb)
    P1 = posterior(S_t)
    fam_idx = np.array([fi_t[f] for f in fam_cb], dtype=np.int64)
    path = dtw(cost_matrix(S_t, fam_idx))
    T = chart_beat_times(path, cb.K, beats)
    fit = _fit_of_path(path, cb.beat_label, P1, fi_t)
    rng = np.random.default_rng(seed)
    base_fits = []
    for _ in range(2):
        labs = _shuffled_labels(cb, rng)
        fidx = np.array([fi_t[family(h)] for h in labs], dtype=np.int64)
        pth = dtw(cost_matrix(S_t, fidx))
        base_fits.append(_fit_of_path(pth, labs, P1, fi_t))
    baseline = float(np.mean(base_fits))
    score = float(np.clip((fit - baseline) / max(1.0 - baseline, 1e-6), 0.0, 1.0))
    return _Stage(t, scale, ex, beats, cb, path, T, fit, baseline, score, _regularity(path))


def align_chart_text(text: str, audio: Audio, bpb: int = 4, written_key: int | None = None,
                     transposes: list[int] | None = None, seed: int = 7) -> Aligned:
    """Align a bar-text chart (sections = templates) to the recording; try the transpositions
    (all 12 by default, the one suggested by ``written_key`` vs the recording's key breaks ties)
    and three beat scales (the chart's beat = 1, 2 or ½ recording beats)."""
    templates = dict(split_templates(text))
    if not templates:
        raise ValueError("the chart has no bars")
    tb = {n: chart_beats(f"[{n}]\n" + "\n".join(ls) + "\n", bpb) for n, ls in templates.items()}
    tb = {n: cb for n, cb in tb.items() if any(h not in ("X", "N") for h in cb.beat_label)}
    if not tb:
        raise ValueError("the chart has no chords")
    # a section with a single chord (an "Outro | D |") carries no order information and only
    # lures the search: left out while the chart has sections with real progressions
    rich = {n: cb for n, cb in tb.items() if len({family(h) for h in cb.beat_label if h not in ("X", "N")}) >= 2}
    if rich:
        tb = rich
    templates = {n: templates[n] for n in tb}
    for n, cb in tb.items():
        if cb.K < MIN_SECTION_BEATS:                                 # a very short section: at least a bar
            cb.beat_label = cb.beat_label + [cb.beat_label[-1]] * (MIN_SECTION_BEATS - cb.K)
    base_labels = [lab for n in tb for lab in tb[n].beat_label]
    if transposes is None:
        transposes = list(range(12))
    hint = None
    if written_key is not None:
        hint = _norm_t(audio.key_index % 12 - written_key % 12)
        if (audio.key_index >= 12) != (written_key >= 12):      # relative keys: same chords
            hint2 = _norm_t(audio.key_index % 12 - (written_key % 12 + (3 if written_key >= 12 else -3)))
            hint = hint if hint in transposes else hint2
    all_labels = {transpose_harte(h, t) for h in base_labels for t in transposes}
    fams, fi = _families(all_labels)
    # beat scales: the chart's beat may be one recording beat, two (the chart counts half notes,
    # or our grid runs at double time) or half of one
    grids = {1.0: audio.beats, 2.0: audio.beats[::2] if len(audio.beats) > 4 else audio.beats,
             0.5: _half_beats(audio.beats)}
    S_by_scale: dict[float, np.ndarray] = {}

    def scores(scale: float) -> np.ndarray:
        if scale not in S_by_scale:
            S_by_scale[scale] = beat_family_scores(audio, grids[scale], fams)
        return S_by_scale[scale]

    fits: dict[int, float] = {}
    tried: list[tuple[float, int, float, Explored]] = []            # (−goodness, t, scale, explored)

    def try_one(t: int, scale: float) -> None:
        S_ = scores(scale)
        fam_t = [family(transpose_harte(lab, t)) for lab in base_labels]
        S_t, fi_t = _subset(S_, fi, fam_t)
        st_fam = np.array([fi_t[f] for f in fam_t], dtype=np.int64)
        ex = explore(tb, st_fam, S_t)
        P1 = posterior(S_t)
        vals = [float(P1[j, st_fam[s]]) for j, s in enumerate(ex.state_of_beat) if s >= 0 and st_fam[s] >= 0]
        fit_t = float(np.mean(vals)) if vals else 0.0
        cov_t = float(np.mean(ex.state_of_beat >= 0)) if len(ex.state_of_beat) else 0.0
        if scale == 1.0:
            fits[_norm_t(t)] = round(fit_t, 4)
        # goodness: the fit along the path, discounted when the path explains little of the song
        good = fit_t * min(1.0, cov_t / COVER_FULL) + (0.05 if hint is not None and _norm_t(t) == hint else 0.0)
        tried.append((-good, t, scale, ex))

    for t in transposes:
        try_one(t, 1.0)
    top = [t for _, t, _, _ in sorted(tried, key=lambda x: x[0])[:3]]
    for t in top:
        for scale in (2.0, 0.5):
            if len(grids[scale]) >= 8:
                try_one(t, scale)
    # the best explore per scale goes through the DTW; the scale whose path is the most regular
    # (chart beats and recording beats advancing together) wins, the plain beat on near-ties
    stages: list[_Stage] = []
    for scale in (1.0, 2.0, 0.5):
        cands = [x for x in tried if x[2] == scale]
        if not cands:
            continue
        c, t, _, ex = min(cands, key=lambda x: x[0])
        stages.append(_dtw_stage(ex, t, scale, grids[scale], scores(scale), fi, fams, templates, bpb, audio, seed))
    def _rank(s: _Stage) -> float:
        cov = float(np.mean(s.ex.state_of_beat >= 0)) if len(s.ex.state_of_beat) else 0.0
        if cov < 0.1:                                                # an all-idle explore
            return -1.0 + cov
        val = s.score * min(1.0, cov / COVER_FULL)
        if s.scale != 1.0:                                           # another beat scale must earn it clearly
            val -= 0.08
            if s.regularity < 0.8:
                val -= 0.2
        return val

    best = max(stages, key=_rank)
    t_best, scale, ex, beats, cb, path, T = best.t, best.scale, best.ex, best.beats, best.cb, best.path, best.T
    fit, baseline, score = best.fit, best.baseline, best.score
    J = len(beats) - 1
    # chords and sections in time
    chords: list[tuple[float, float, str]] = []
    for a in cb.events:
        if a.harte in ("X",):
            continue
        t0, t1 = at(T, a.beat0), at(T, a.beat1)
        if t1 - t0 < 0.02:
            continue
        r, q, b = H.parse_harte(a.harte, strict=False)
        h = "N" if r < 0 else H.to_harte_in_key(r, q, b, audio.key_at(t0))
        chords.append((round(t0, 3), round(t1, 3), h))
    chords.sort()
    sections: list[tuple[float, str]] = []
    first_bar: dict[int, int] = {}
    for bar, si in enumerate(cb.section_of_bar):
        first_bar.setdefault(si, bar)
    for si, lab in enumerate(cb.labels):
        if lab and lab != "Gap" and si in first_bar:
            sections.append((round(float(T[first_bar[si] * bpb]), 3), lab))
    covered = int(np.sum(ex.state_of_beat >= 0))
    coverage = covered / max(J, 1)
    warnings: list[str] = []
    from collections import Counter
    per_k = Counter(k for k, _ in path)
    per_j = Counter(j for _, j in path)
    n_stretch = sum(n - 1 for n in per_k.values() if n > 1)
    n_squeeze = sum(n - 1 for n in per_j.values() if n > 1)
    if n_stretch > 0.25 * cb.K:
        warnings.append(f"the chart's beats were stretched over {n_stretch} extra recording beats (slow passages, or bars missing from the chart)")
    if n_squeeze > 0.25 * cb.K:
        warnings.append(f"{n_squeeze} chart beats had to share a recording beat (the chart may have more bars than the recording)")
    if coverage < 0.6:
        warnings.append(f"the chart covers only {coverage:.0%} of the recording")
    gaps = [(j0, j1) for n, j0, j1 in ex.form if n == "" and j1 - j0 >= bpb]
    if gaps:
        warnings.append("stretches the chart does not cover (intro, interlude or tag): " +
                        ", ".join(f"{beats[j0]:.0f}–{beats[min(j1, len(beats) - 1)]:.0f} s" for j0, j1 in gaps[:4]))
    if scale != 1.0:
        warnings.append("the chart's beat is " + ("two of" if scale == 2.0 else "half of") + " the recording's beats")
    score_c = score * min(1.0, coverage / COVER_FULL)              # a chart explaining little of the song scores little
    return Aligned(_norm_t(t_best), round(score_c, 3), round(fit, 3), round(baseline, 3),
                   verdict_of(score_c, fit, coverage), chords, sections, warnings, ex.form, round(coverage, 3), bpb,
                   fits, audio.key_index, scale, round(best.regularity, 3))


def labels_for(harte: str, key: int) -> list[str]:
    """[simple, standard, full] labels of a Harte chord in a key (the web never spells)."""
    r, q, b = H.parse_harte(harte, strict=False)
    if r < 0 or q == "N":
        return ["N", "N", "N"]
    d = SM.labels(r, q, b, key)
    return [d["simple"], d["standard"], d["full"]]
