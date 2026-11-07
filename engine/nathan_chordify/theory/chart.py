"""Bar-text charts (DESIGN §0.1), used for Nathan's ground truth and the Chordify baseline.

    [Verse 1]
    | B . . B+7 | Emaj7 . A7 Em6 |

- One bar between ``|`` marks, one token per beat; ``.`` means "the same chord continues".
- A bar with fewer tokens than beats and no dots (and no brackets) is split evenly: ``| B E |``
  is two beats each. Such chords get ``pos=None``: the split is resolved against the real number
  of beats in that bar when the chart is aligned (``align``, M4).
- Brackets split one beat evenly: ``(A7 Em6)`` = two half-beats; ``.`` may be used inside a
  bracket (``(. Em6)`` = the change comes on the "and").
- A bar with dots or brackets but fewer tokens than beats lets its last chord run to the bar end.
- ``N`` = no chord, ``%`` = repeat the previous bar, a trailing ``?`` = unsure (``Em/G?``).
- Section labels are ``[Label]`` on their own line, or at the start of a line of bars.
  Text outside the outermost ``|`` marks of a line (e.g. a "(M–H)" note) is ignored.
- Lines starting with ``#`` are comments.

Ground truth (M4, DESIGN §8.4, §13.1) adds, on top of ``parse``/``format``:

- ``parse_annotated``: also reads a section's start time (``[Intro @2.9]``) and the per-line
  confidence notes ``(H)`` / ``(M)`` / ``(L)`` written after the last bar line;
- ``align_chart`` / ``align``: lay the bars on a beat list (Nathan's tapped beats, or the
  detected grid until he taps), honouring beat-dot positions, even splits and anchors;
- ``chart_from_times``: the reverse, a timed chord list placed on bars of a beat list (the
  ``<key>.chart.txt`` export).

Chord tokens are typed lead-sheet symbols (``harte.parse_symbol``) or Harte labels; they are
stored as canonical Harte with the written root spelling ("F#/A#" → "F#:maj/3").
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..types import Chart, ChartChord, ChartSection, RhythmGrid, Segment
from . import harte as H

_LABEL_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$")
_EPS = 1e-6


class ChartError(ValueError):
    """A chart line that cannot be read; the message says where."""


def _chord(tok: str, where: str) -> tuple[str, bool]:
    unsure = tok.endswith("?")
    text = tok[:-1] if unsure else tok
    try:
        return H.symbol_to_harte(text), unsure
    except ValueError as e:
        raise ChartError(f"{where}: {e}") from None


def _tokens(text: str, where: str) -> list[str]:
    """Split a bar into tokens. A token that starts with "(" is a bracketed beat; parentheses
    inside a chord symbol ("Cm(maj7)", "C(add9)") stay part of that chord."""
    toks: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j, depth = i, 0
        while j < n:
            ch = text[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    raise ChartError(f"{where}: unbalanced brackets")
                if depth == 0 and text[i] == "(":
                    j += 1
                    break
            elif ch.isspace() and depth == 0:
                break
            j += 1
        if depth != 0:
            raise ChartError(f"{where}: unbalanced brackets")
        toks.append(text[i:j])
        i = j
    return toks


def _parse_bar(text: str, beats: int, where: str) -> tuple[list[ChartChord], set[int]]:
    """One bar → (chords, indices of the chords written with "?")."""
    toks = _tokens(text, where)
    if not toks:
        raise ChartError(f"{where}: empty bar")
    per_beat = any(t == "." or t.startswith("(") for t in toks) or len(toks) == beats
    items: list[tuple[float | None, str, bool]] = []
    if not per_beat:
        if len(toks) > beats:
            raise ChartError(f"{where}: {len(toks)} chords in a {beats}-beat bar")
        items = [(None, *_chord(t, where)) for t in toks]
    else:
        if len(toks) > beats:
            raise ChartError(f"{where}: {len(toks)} beats in a {beats}-beat bar")
        for b, tok in enumerate(toks):
            if tok == ".":
                continue
            if tok.startswith("("):
                parts = _tokens(tok[1:-1], where)
                if not parts:
                    raise ChartError(f"{where}: empty brackets")
                for k, part in enumerate(parts):
                    if part != ".":
                        items.append((b + k / len(parts), *_chord(part, where)))
                continue
            items.append((float(b), *_chord(tok, where)))
    bar = [ChartChord(pos, harte) for pos, harte, _ in items]
    return bar, {i for i, (_, _, unsure) in enumerate(items) if unsure}


def parse(text: str, beats_per_bar: int = 4) -> Chart:
    """Parse bar text into a Chart: sections → bars → ``ChartChord(pos, harte)`` pairs, with the
    "?" flags in ``ChartSection.unsure`` as (bar index, chord index) pairs."""
    chart = Chart([], beats_per_bar)
    section: ChartSection | None = None
    prev: tuple[list[ChartChord], set[int]] | None = None
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):    # blank or comment line
            continue
        m = _LABEL_RE.match(line)
        if m:
            section = ChartSection(m.group(1).strip(), [])
            chart.sections.append(section)
            line = m.group(2).strip()
            if "|" not in line:                  # nothing, or a note such as "(M)" after the label
                continue
        if "|" not in line:
            raise ChartError(f"line {ln}: no bar lines '|' in {raw!r}")
        pieces = line.split("|")[1:-1]          # text outside the outer bar lines is ignored
        if section is None:
            section = ChartSection(None, [])
            chart.sections.append(section)
        for i, piece in enumerate(pieces):
            where = f"line {ln}, bar {i + 1}"
            piece = piece.strip()
            if not piece:
                continue                         # "||" double bar line
            if piece == "%":
                if prev is None:
                    raise ChartError(f"{where}: '%' with no previous bar")
                bar, unsure = list(prev[0]), set(prev[1])
            else:
                bar, unsure = _parse_bar(piece, beats_per_bar, where)
            bi = len(section.bars)
            section.bars.append(bar)
            section.unsure |= {(bi, ci) for ci in unsure}
            prev = (bar, unsure)
    return chart


# ---------------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------------

def _token(c: ChartChord, unsure: bool = False) -> str:
    sym = "N" if c.harte == "N" else H.harte_to_symbol(c.harte)
    return sym + ("?" if unsure else "")


def _format_bar(bar: list[ChartChord], beats: int, unsure: set[int] = frozenset()) -> str:
    if not bar:
        return "."                               # held bar: the previous chord continues
    if all(c.pos is None for c in bar):
        return " ".join(_token(c, i in unsure) for i, c in enumerate(bar))
    timed = [(i, c) for i, c in enumerate(bar) if c.pos is not None]
    n_beats = max(beats, int(np.floor(max(c.pos for _, c in timed) + _EPS)) + 1)
    toks = []
    for b in range(n_beats):
        ev = [(i, c) for i, c in timed if b - _EPS <= c.pos < b + 1 - _EPS]
        if not ev:
            toks.append(".")
        elif len(ev) == 1 and abs(ev[0][1].pos - b) < _EPS:
            toks.append(_token(ev[0][1], ev[0][0] in unsure))
        else:
            k = next((k for k in (2, 3, 4, 6, 8, 12)
                      if all(abs((c.pos - b) * k - round((c.pos - b) * k)) < 1e-4 for _, c in ev)), 12)
            parts = ["."] * k
            for i, c in ev:
                parts[min(k - 1, int(round((c.pos - b) * k)))] = _token(c, i in unsure)
            toks.append("(" + " ".join(parts) + ")")
    while len(toks) > 1 and toks[-1] == ".":
        toks.pop()                               # trailing dots are implied
    if len(toks) < beats and all(t != "." and not t.startswith("(") for t in toks):
        toks += ["."] * (beats - len(toks))      # keep it per-beat, not an even split
    return " ".join(toks)


def format_chart(chart: Chart, bars_per_line: int = 4, anchors: list[float | None] | None = None,
                 bar_conf: list[str | None] | None = None) -> str:
    """Chart → bar text (round-trips through ``parse``; with anchors / confidence notes it
    round-trips through ``parse_annotated``).

    ``anchors`` (one per section) writes ``[Label @12.34]``; ``bar_conf`` (one per bar of
    ``chart.bars``, "H" / "M" / "L") ends each line with the weakest note of its bars, ``(L)``."""
    lines: list[str] = []
    g = 0                                         # global bar index
    for si, s in enumerate(chart.sections):
        anchor = anchors[si] if anchors is not None and si < len(anchors) else None
        if anchor is not None:
            lines.append(f"[{s.label or 'Part'} @{anchor:.2f}]")
        elif s.label is not None:
            lines.append(f"[{s.label}]")
        for i in range(0, len(s.bars), bars_per_line):
            chunk = s.bars[i:i + bars_per_line]
            row = [_format_bar(bar, chart.beats_per_bar, {ci for bi, ci in s.unsure if bi == i + j})
                   for j, bar in enumerate(chunk)]
            line = "| " + " | ".join(row) + " |"
            if bar_conf is not None:
                flags = [bar_conf[g + j] for j in range(len(chunk)) if g + j < len(bar_conf) and bar_conf[g + j]]
                if flags:
                    line += f"   ({min(flags, key=_CONF_RANK.__getitem__)})"
            lines.append(line)
            g += len(chunk)
    return "\n".join(lines) + "\n"


def chart_from_segments(segments: list[Segment], grid: RhythmGrid, labels: list[str],
                        sections: list[dict] | None = None) -> Chart:
    """Build a Chart from decoded segments on a beat grid; ``labels[i]`` is the Harte label of
    ``segments[i]``. Bars come from ``grid.bars`` and onsets from ``grid.cell_beat``; a bar with
    no change is an empty bar (written ``| . |``). ``sections`` are the analysis-JSON section
    dicts ({"label", "start", ...}). Bars of other lengths than the most common one keep their
    chords but the chart records a single ``beats_per_bar``."""
    counts = Counter(n for _, n in grid.bars)
    beats_per_bar = counts.most_common(1)[0][0] if counts else 4
    starts = [float(grid.cell_beat[s.c0]) for s in segments]
    bounds = sorted((float(s.get("start", 0.0)), s.get("label")) for s in (sections or []))
    chart = Chart([], beats_per_bar)
    cur: ChartSection | None = None
    for first, n in grid.bars:
        t_bar = float(grid.beats[first]) if first < len(grid.beats) else float("inf")
        label = None
        for st, lab in bounds:
            if st <= t_bar + 1e-3:
                label = lab
        if cur is None or label != cur.label:
            cur = ChartSection(label, [])
            chart.sections.append(cur)
        bar = [ChartChord(round(max(st - first, 0.0), 4), lab)
               for st, lab in zip(starts, labels)
               if not np.isnan(st) and -_EPS <= st - first < n - _EPS]
        cur.bars.append(sorted(bar, key=lambda c: c.pos))
    return chart


def format(chart_or_segments, grid: RhythmGrid | None = None, labels: list[str] | None = None,
           sections: list[dict] | None = None, bars_per_line: int = 4) -> str:
    """Bar text from a Chart, or (DESIGN §4) from ``segments`` + ``grid`` with one Harte label
    per segment in ``labels``."""
    if isinstance(chart_or_segments, Chart):
        return format_chart(chart_or_segments, bars_per_line)
    if grid is None or labels is None:
        raise TypeError("format(segments, grid, labels) needs the grid and one label per segment")
    return format_chart(chart_from_segments(chart_or_segments, grid, labels, sections), bars_per_line)


# ---------------------------------------------------------------------------------------------
# Ground truth (M4): annotated charts, alignment to beats, and back
# ---------------------------------------------------------------------------------------------

_CONF_RANK = {"L": 0, "M": 1, "H": 2}
_ANCHOR_RE = re.compile(r"^(.*?)\s*@\s*(\d+(?:\.\d+)?)\s*s?\s*$")
_NOTE_RE = re.compile(r"\(\s*([HML])(?:\s*[–\-/]\s*([HML]))?\s*\)")


@dataclass
class AnnotatedChart:
    """A chart plus what ``parse`` ignores: section start times and per-bar confidence."""
    chart: Chart
    anchors: list[float | None]               # one per ``chart.sections``: "[Intro @2.9]" → 2.9
    bar_conf: list[str]                       # one per ``chart.bars``: "H" | "M" | "L"


def _line_bars(line: str) -> int:
    """Number of bars ``parse`` reads from a line of bars (``%`` and held bars count)."""
    if "|" not in line:
        return 0
    return sum(1 for piece in line.split("|")[1:-1] if piece.strip())


def parse_annotated(text: str, beats_per_bar: int = 4, default_conf: str = "H") -> AnnotatedChart:
    """``parse`` that also keeps the ``@ time`` of section labels (stripped from the label) and
    the ``(H)`` / ``(M)`` / ``(L)`` note at the end of each line (``(M–H)`` counts as M, the
    lower one). Bars without a note get ``default_conf``. Chords written with ``?`` stay in
    ``ChartSection.unsure``."""
    clean: list[str] = []
    anchors: list[float | None] = []
    bar_conf: list[str] = []
    sec_conf = default_conf                        # "[Chorus] (M)": the default for that section's lines

    def note(text_: str, fallback: str) -> str:
        nm = _NOTE_RE.search(text_)
        return min(nm.group(1), nm.group(2) or nm.group(1), key=_CONF_RANK.__getitem__) if nm else fallback

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LABEL_RE.match(line)
        if m:
            label, line = m.group(1).strip(), m.group(2).strip()
            am = _ANCHOR_RE.match(label)
            anchor = None
            if am:
                label, anchor = (am.group(1).strip() or "Part"), float(am.group(2))
            anchors.append(anchor)
            clean.append(f"[{label}]")
            sec_conf = note(line, default_conf) if "|" not in line else default_conf
            if "|" not in line:
                continue
        elif "|" in line and not anchors:
            anchors.append(None)                   # bars before the first label: an unlabeled section
        clean.append(line)
        n = _line_bars(line)
        if n:
            bar_conf.extend([note(line[line.rfind("|") + 1:], sec_conf)] * n)
    chart = parse("\n".join(clean), beats_per_bar)
    anchors = (anchors + [None] * len(chart.sections))[:len(chart.sections)]
    n_bars = len(chart.bars)
    bar_conf = (bar_conf + [default_conf] * n_bars)[:n_bars]
    return AnnotatedChart(chart, anchors, bar_conf)


@dataclass
class Aligned:
    """One chart chord laid on a beat list (``align_chart``)."""
    t0: float
    t1: float
    harte: str
    bar: int                                  # global chart bar index (``chart.bars``)
    pos: float                                # onset in beats from the bar start
    beat0: float                              # fractional index into the beat list
    beat1: float
    section: str | None
    unsure: bool                              # written with "?"
    conf: str                                 # the bar's note: "H" | "M" | "L" ("?" chords are "L")


@dataclass
class ChartLayout:
    """Where each chart bar sits on the beat list (``align_chart``)."""
    bar_start: list[float]                    # fractional beat index of each bar's first beat
    beats_per_bar: int
    section_of_bar: list[int]                 # section index per bar
    warnings: list[str] = field(default_factory=list)


def time_at(beats: np.ndarray, x: float | np.ndarray) -> np.ndarray | float:
    """Seconds at a fractional beat index, linear between beats and extrapolated with the
    first / last inter-beat interval outside the list."""
    b = np.asarray(beats, dtype=float)
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(b)
    if n < 2:
        raise ValueError("at least two beats are needed")
    out = np.interp(xs, np.arange(n, dtype=float), b)
    lo, hi = xs < 0, xs > n - 1
    out[lo] = b[0] + xs[lo] * (b[1] - b[0])
    out[hi] = b[-1] + (xs[hi] - (n - 1)) * (b[-1] - b[-2])
    return out if np.ndim(x) else float(out[0])


def beat_at(beats: np.ndarray, t: float | np.ndarray) -> np.ndarray | float:
    """Fractional beat index of a time (inverse of ``time_at``)."""
    b = np.asarray(beats, dtype=float)
    ts = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(b)
    if n < 2:
        raise ValueError("at least two beats are needed")
    out = np.interp(ts, b, np.arange(n, dtype=float))
    lo, hi = ts < b[0], ts > b[-1]
    out[lo] = (ts[lo] - b[0]) / (b[1] - b[0])
    out[hi] = n - 1 + (ts[hi] - b[-1]) / (b[-1] - b[-2])
    return out if np.ndim(t) else float(out[0])


def layout(chart: Chart, beats: np.ndarray, start_beat: float = 0.0,
           anchors: list[float | None] | None = None) -> ChartLayout:
    """Bar positions on the beat list: the first bar starts at beat ``start_beat``; a section
    with an anchor time starts at the beat nearest to it; other sections follow on."""
    beats = np.asarray(beats, dtype=float)
    bpb = chart.beats_per_bar
    starts: list[float] = []
    sec_of: list[int] = []
    warnings: list[str] = []
    cur = float(start_beat)
    for si, s in enumerate(chart.sections):
        a = anchors[si] if anchors is not None and si < len(anchors) else None
        if a is not None:
            x = float(round(beat_at(beats, a)))
            if starts and x < cur - _EPS:
                warnings.append(f"section {s.label or si + 1!s} @{a:g} s overlaps the previous one; "
                                f"it starts after it instead")
                x = cur
            cur = x
        for _ in s.bars:
            starts.append(cur)
            sec_of.append(si)
            cur += bpb
    if starts and cur - 1 > len(beats) - 1 + _EPS:
        over = cur - 1 - (len(beats) - 1)
        warnings.append(f"the chart runs {over:.0f} beat(s) past the last beat; those times are extrapolated")
    return ChartLayout(starts, bpb, sec_of, warnings)


def align_chart(chart: Chart, beats: np.ndarray, start_beat: float = 0.0,
                anchors: list[float | None] | None = None, bar_conf: list[str] | None = None,
                lay: ChartLayout | None = None) -> list[Aligned]:
    """Lay a chart on a beat list (DESIGN §8.4, §13.1 step 3).

    Positions written with dots or brackets are honoured; a bar written without dots is split
    evenly over its beats; a held bar (``| . |``) continues the previous chord; ``%`` repeats.
    Each chord ends where the next one starts, or at the end of its section when the next
    section is anchored later (a gap the chart doesn't cover), or at the end of the last bar."""
    beats = np.asarray(beats, dtype=float)
    lay = lay or layout(chart, beats, start_beat, anchors)
    bpb = chart.beats_per_bar
    onsets: list[tuple[float, int, float, str, bool, int]] = []      # (x, bar, pos, harte, unsure, section)
    g = 0
    for si, s in enumerate(chart.sections):
        for bi, bar in enumerate(s.bars):
            x0 = lay.bar_start[g]
            n = len(bar)
            for ci, c in enumerate(bar):
                pos = c.pos if c.pos is not None else bpb * ci / n
                onsets.append((x0 + pos, g, float(pos), c.harte, (bi, ci) in s.unsure, si))
            g += 1
    if not onsets:
        return []
    # section ends (fractional beat index), for chords that run into an uncovered gap
    sec_end: dict[int, float] = {}
    for gi, si in enumerate(lay.section_of_bar):
        sec_end[si] = lay.bar_start[gi] + bpb
    out: list[Aligned] = []
    for k, (x, gbar, pos, harte, unsure, si) in enumerate(onsets):
        if k + 1 < len(onsets):
            nx, nsi = onsets[k + 1][0], onsets[k + 1][5]
            end = min(nx, sec_end[si]) if nsi != si else nx
        else:
            end = sec_end[si]
        if end <= x + _EPS:
            continue                                   # a zero-length chord (two at one position)
        conf = "L" if unsure else (bar_conf[gbar] if bar_conf is not None and gbar < len(bar_conf) else "H")
        t0, t1 = time_at(beats, [x, end])
        out.append(Aligned(round(float(t0), 4), round(float(t1), 4), harte, gbar, pos,
                           x, end, chart.sections[si].label, unsure, conf))
    return out


def align(chart: Chart, beats: np.ndarray) -> list[Segment]:
    """DESIGN §4 entry point: chart bars → Segments on a beat list (the first bar starts at the
    first beat). ``c0``/``c1`` are beat indices (rounded outwards for sub-beat chords); ``cand``
    is the chord's vocabulary index (-1 when it is not in the vocabulary); ``conf`` is 1.0, or
    0.5 for chords written with "?"."""
    from . import vocab as VOC
    v = VOC.candidates()
    segs: list[Segment] = []
    for a in align_chart(chart, beats):
        r, q, b = H.parse_harte(a.harte, strict=False)
        i = VOC.index_of(r, q, b)
        cand = -1 if i is None else i
        cls = -1 if i is None else v.cands[i].cls
        c0 = int(np.floor(a.beat0 + 1e-6))
        c1 = max(c0 + 1, int(np.ceil(a.beat1 - 1e-6)))
        segs.append(Segment(c0=c0, c1=c1, t0=a.t0, t1=a.t1, cand=cand, cls=cls, layer="main",
                            conf=0.5 if a.unsure else 1.0, edited=True))
    return segs


def _quantize(x: float) -> float:
    """Snap a within-bar position to the nearest 1/4 or 1/3 of a beat (whichever is closer)."""
    q4 = round(x * 4) / 4
    q3 = round(x * 3) / 3
    return q3 if abs(q3 - x) < abs(q4 - x) - 1e-9 else q4


def chart_from_times(chords: list[tuple[float, float, str, bool]], beats: np.ndarray,
                     bar_starts: list[float], beats_per_bar: int = 4,
                     section_of_bar: list[int] | None = None,
                     section_labels: list[str | None] | None = None,
                     restate: bool = True) -> Chart:
    """Timed chords ``(t0, t1, harte, unsure)`` → a Chart on bars of a beat list.

    ``bar_starts`` are the fractional beat indices of the bars; each bar has ``beats_per_bar``
    beats. Onsets are quantized to ¼ or ⅓ beat. With ``restate`` the chord sounding at a bar's
    first beat is written there when nothing starts on it (lead-sheet style; the timeline is
    the same). Chords before the first bar or after the last are not written."""
    beats = np.asarray(beats, dtype=float)
    chords = sorted(chords, key=lambda c: c[0])
    xs = beat_at(beats, np.array([c[0] for c in chords], dtype=float)) if chords else np.zeros(0)
    xe = beat_at(beats, np.array([c[1] for c in chords], dtype=float)) if chords else np.zeros(0)
    section_of_bar = section_of_bar or [0] * len(bar_starts)
    labels = section_labels or [None]
    chart = Chart([], beats_per_bar)
    cur_si = None
    for g, x0 in enumerate(bar_starts):
        si = section_of_bar[g]
        if si != cur_si:
            chart.sections.append(ChartSection(labels[si] if si < len(labels) else None, []))
            cur_si = si
        sec = chart.sections[-1]
        x1 = x0 + beats_per_bar
        items: dict[float, tuple[str, bool, float]] = {}
        for k, (t0, t1, harte, unsure) in enumerate(chords):
            if x0 - 0.125 <= xs[k] < x1 - 0.125:
                pos = min(max(_quantize(float(xs[k] - x0)), 0.0), beats_per_bar - 0.25)
                dur = float(xe[k] - xs[k])
                if pos not in items or dur > items[pos][2]:
                    items[pos] = (harte, unsure, dur)
        if restate and not any(abs(p) < _EPS for p in items):
            sounding = [k for k in range(len(chords)) if xs[k] < x0 - 0.125 and xe[k] > x0 + 0.125]
            if sounding:
                k = sounding[-1]
                items[0.0] = (chords[k][2], chords[k][3], float(xe[k] - x0))
        bi = len(sec.bars)
        poss = sorted(items)
        n = len(poss)
        even = n >= 2 and all(abs(p - beats_per_bar * i / n) < 1e-6 for i, p in enumerate(poss))
        bar = []
        for ci, pos in enumerate(poss):
            harte, unsure, _ = items[pos]
            bar.append(ChartChord(None if even else pos, harte))   # "| B E |" rather than "| B . E . |"
            if unsure:
                sec.unsure.add((bi, ci))
        sec.bars.append(bar)
    return chart
