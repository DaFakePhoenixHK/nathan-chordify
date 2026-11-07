"""The printable chord sheet (user's request 2, 2026-11-04): the bar chart, and the chords over the lyrics.

    GET /api/tracks/{key}/export/sheet?format=txt|chordpro|json&transpose=0&level=simple|standard|full
                                       &slash=1&lyrics=1&acc=auto&sheet=<result id>

(a) The **bar chart**: title, key (after the transposition), BPM, meter; section labels; four bars per
    line; one token per beat, ``.`` = the same chord continues, ``N`` = no chord, ``(. B)`` = the change
    comes on the second half of the beat. The chords are the MAIN layer of the edited analysis at the
    chosen level, spelled by the engine's speller (``theory.simplify.label``) — never by the web.
(b) The **chords over the lyrics** when ``lyrics=1`` and a sheet with lyrics was fetched or pasted
    (``cache/<key>/sheets/<id>.lyrics.txt``, ChordPro-style ``[C]word``): the lyric text is the sheet's,
    the chords are OURS. Every inline chord of the sheet is matched to the sheet's aligned chord events
    (same order, same chords), which gives each lyric line a time span; our chord changes are then written
    at the nearest syllable. When the sheet can't be lined up that way, the sheet's own chords are kept
    (moved to the export key) and a header line says so.

``build`` returns a ``SheetExport``; ``to_text`` / ``to_chordpro`` / ``to_json`` render it.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .. import schema
from ..theory import harte as H
from ..theory import simplify as SM
from ..theory import spelling as S
from ..theory.display import key_index

LEVELS = SM.LEVELS
_EPS = 1e-3
_INLINE = re.compile(r"\[([^\[\]]+)\]")
_LABEL_WORDS = re.compile(r"^(intro|verse|chorus|bridge|pre[- ]?chorus|tag|outro|ending|interlude|coda|refrain|"
                          r"instrumental|solo|vamp|turnaround|part|section|主歌|副歌|導歌|前奏|間奏|尾奏|過門|橋段|結尾)",
                          re.IGNORECASE)


# ---------------------------------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------------------------------

def key_at(a: schema.Analysis, t: float) -> int:
    for s in a.key.segments:
        if s.start <= t < s.end:
            return key_index(s.tonic, s.mode)
    if a.key.segments and t >= a.key.segments[-1].end:
        s = a.key.segments[-1]
        return key_index(s.tonic, s.mode)
    return key_index(a.key.global_.tonic, a.key.global_.mode)


def main_chords(a: schema.Analysis) -> list[schema.ChordOut]:
    """The main layer in time order (what the chart and the lyrics use)."""
    return sorted((c for c in a.chords if c.layer == "main"), key=lambda c: c.start)


def label_main(a: schema.Analysis, chords: list[schema.ChordOut], transpose: int = 0, pref: str = "auto",
               level: str = "full", slash: bool = True) -> list[str]:
    """One spelled label per chord ("N" for no chord), exactly like display.json but with the Slash
    toggle applied by the speller (``slash=False`` hides every bass and the upper-structure form)."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    trip = [H.parse_harte(c.harte, strict=False) for c in chords]
    out: list[str] = []
    nxt: int | None = None
    labels_rev: list[str] = []
    for c, (r, q, b) in zip(reversed(chords), reversed(trip)):
        if r < 0 or q == "N":
            labels_rev.append("N")
        else:
            labels_rev.append(SM.label(r, q, b, key_at(a, c.start), transpose=transpose % 12, pref=pref,
                                       level=level, slash=None if slash else False, next_root=nxt))
            nxt = r
    out = labels_rev[::-1]
    return out


# ---------------------------------------------------------------------------------------------------
# the bar chart
# ---------------------------------------------------------------------------------------------------

def _sounding(chords: list[schema.ChordOut], labels: list[str], t: float) -> str | None:
    """Label of the main chord sounding at t (None when none)."""
    for c, lab in zip(chords, labels):
        if c.start - _EPS <= t < c.end - _EPS:
            return lab
    return None


def _section_labels(a: schema.Analysis) -> dict[int, str]:
    """Bar index → section label at the bar where a section starts."""
    out: dict[int, str] = {}
    prev: str | None = None
    for bar in a.bars:
        if bar.section and bar.section != prev:
            out[bar.i] = bar.section
        prev = bar.section
    if not out and a.sections:
        for s in a.sections:
            k = next((b.i for b in a.bars if b.start >= s.start - 0.05), None)
            if k is not None and k not in out:
                out[k] = s.label
    return out


def bar_chart(a: schema.Analysis, chords: list[schema.ChordOut], labels: list[str], bars_per_line: int = 4) -> str:
    """Bar text: four bars per line, one token per beat, "." continues, "N" = no chord, brackets for a
    change inside a beat. Reads back with ``theory.chart.parse``."""
    beats_of: dict[int, list[float]] = {}
    for b in a.beats:
        if b.bar >= 0:
            beats_of.setdefault(b.bar, []).append(b.t)
    sections = _section_labels(a)
    lines: list[str] = []
    row: list[str] = []
    last: str | None = None

    def flush() -> None:
        nonlocal row
        if row:
            lines.append("| " + " | ".join(row) + " |")
        row = []

    for bar in a.bars:
        if bar.i in sections:
            flush()
            lines.append(f"[{sections[bar.i]}]")
        beats = sorted(beats_of.get(bar.i, []))
        if not beats:
            beats = [bar.start]
        toks: list[str] = []
        for k, b0 in enumerate(beats):
            b1 = beats[k + 1] if k + 1 < len(beats) else bar.end
            at_start = _sounding(chords, labels, b0 + 0.02)
            inside = [((c.start - b0) / max(b1 - b0, 1e-6), lab) for c, lab in zip(chords, labels)
                      if b0 + 0.08 * (b1 - b0) <= c.start < b1 - _EPS]
            if at_start is None or at_start == "N":
                first = "N" if last not in (None, "N") else "."
                if last is None and at_start == "N":
                    first = "N"
            else:
                first = at_start if at_start != last else "."
            if at_start is not None:
                last = at_start
            if not inside:
                toks.append(first)
                continue
            parts_n = 2 if all(abs(p - 0.5) < 0.17 for p, _ in inside) else 4
            parts = ["."] * parts_n
            parts[0] = first
            for p, lab in inside:
                parts[min(parts_n - 1, max(1, round(p * parts_n)))] = lab
                last = lab
            toks.append("(" + " ".join(parts) + ")")
        row.append(" ".join(toks))
        if len(row) == bars_per_line:
            flush()
    flush()
    return "\n".join(lines) + ("\n" if lines else "")


def before_first_bar(a: schema.Analysis, chords: list[schema.ChordOut], labels: list[str]) -> list[str]:
    """Labels of the chords that sound before the first bar (a free-time intro), without repeats."""
    if not a.bars:
        return []
    t0 = a.bars[0].start
    out: list[str] = []
    for c, lab in zip(chords, labels):
        if c.start < t0 - 0.25 and lab != "N" and (not out or out[-1] != lab):
            out.append(lab)
    return out


# ---------------------------------------------------------------------------------------------------
# chords over the lyrics
# ---------------------------------------------------------------------------------------------------

@dataclass
class LyricLine:
    kind: str                       # "text" | "label" | "blank"
    text: str                       # the words without the inline chords ("" for blank / label)
    label: str | None = None
    markers: list[tuple[int, str | None]] = field(default_factory=list)   # (char index, Harte or None)
    raw: str = ""


def _is_label(content: str) -> bool:
    if _LABEL_WORDS.match(content.strip()):
        return True
    sym, _ = _norm_chord(content)
    return sym is None


def _norm_chord(tok: str) -> tuple[str | None, str | None]:
    try:
        from ..sheets.parsers import norm_chord
        return norm_chord(tok)
    except Exception:  # noqa: BLE001 - the parsers module is optional here
        try:
            return H.harte_to_symbol(H.symbol_to_harte(tok.strip())), None
        except ValueError:
            return None, None


def _marker_harte(tok: str) -> str | None:
    sym, _ = _norm_chord(tok)
    if sym is None or sym == "N":
        return None
    try:
        return H.symbol_to_harte(sym)
    except ValueError:
        return None


def parse_lyrics(text: str) -> list[LyricLine]:
    """ChordPro-style text → lines with the chord markers taken out and remembered by position."""
    out: list[LyricLine] = []
    for raw in text.splitlines():
        s = raw.rstrip()
        if not s.strip():
            out.append(LyricLine("blank", "", raw=raw))
            continue
        m = re.fullmatch(r"\s*\[([^\[\]]+)\]\s*", s)
        if m and _is_label(m.group(1)):
            out.append(LyricLine("label", "", label=m.group(1).strip(), raw=raw))
            continue
        if re.fullmatch(r"\s*\{[^}]*\}\s*", s):          # a ChordPro directive: keep as a label-ish line
            out.append(LyricLine("label", "", label=None, raw=raw))
            continue
        markers: list[tuple[int, str | None]] = []
        plain: list[str] = []
        pos = 0
        for mm in _INLINE.finditer(s):
            plain.append(s[pos:mm.start()])
            n = sum(len(p) for p in plain)
            markers.append((n, _marker_harte(mm.group(1))))
            pos = mm.end()
        plain.append(s[pos:])
        out.append(LyricLine("text", "".join(plain), markers=markers, raw=raw))
    return out


def _triple(h: str) -> tuple[int, str, int]:
    try:
        return H.parse_harte(h, strict=False)
    except ValueError:
        return (-1, "N", -1)


def _family(h: str) -> tuple:
    """(root pc, chord family) of a Harte label, the way the sheet aligner groups chords."""
    try:
        from ..sheets.align import family
        f = family(h)
        return f[:2] if len(f) >= 2 else f
    except Exception:  # noqa: BLE001
        r, q, _ = _triple(h)
        fam = "min" if q in ("min", "m7", "m6", "m9", "m11", "madd9", "mM7") else \
            "dim" if q in ("dim", "dim7", "m7b5") else q if q in ("N", "aug", "sus2", "sus4", "5") else "maj"
        return (r, fam)


def _lcs(xs: list, ys: list, gap_x: float = 0.5, gap_y: float = 0.5, mismatch: float = 1.0) -> list[tuple[int, int]]:
    """Monotone matching of two chord sequences as index pairs: a global alignment that scores a match
    +2 and charges every skipped element (``gap``) and every paired non-match (``mismatch``), so the two
    sequences move forward together instead of jumping ahead to the next identical chord (a plain
    longest-common-subsequence drifts on the repetitive chords of a song)."""
    n, m = len(xs), len(ys)
    if not n or not m:
        return []
    NEG = float("-inf")
    # score[i][j]: best score aligning xs[:i] with ys[:j]
    score = [[NEG] * (m + 1) for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]          # 1 = pair, 2 = skip x, 3 = skip y
    score[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, mv = NEG, 0
            if i > 0 and j > 0:
                s = score[i - 1][j - 1] + (2.0 if xs[i - 1] is not None and xs[i - 1] == ys[j - 1] else -mismatch)
                if s > best:
                    best, mv = s, 1
            if i > 0 and score[i - 1][j] - gap_x > best:
                best, mv = score[i - 1][j] - gap_x, 2
            if j > 0 and score[i][j - 1] - gap_y > best:
                best, mv = score[i][j - 1] - gap_y, 3
            score[i][j], move[i][j] = best, mv
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        mv = move[i][j]
        if mv == 1:
            if xs[i - 1] is not None and xs[i - 1] == ys[j - 1]:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif mv == 2:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def marker_times(lines: list[LyricLine], aligned: list[tuple[float, float, str]], gap_x: float = 0.5,
                 gap_y: float = 0.5) -> tuple[list[float | None], float]:
    """A time for every chord marker of the lyrics (in reading order), from timed chord events (the
    sheet's aligned chords, or our own detected chords). Returns (times, matched share). Markers that found no partner are interpolated between
    their neighbours; without any match every time is None."""
    marks = [(li, k, h) for li, ln in enumerate(lines) for k, (_, h) in enumerate(ln.markers)]
    # chords are compared by family (root + major / minor / dim / sus ...): a sheet's "Fm" and a detected
    # "Fm7" are the same chord here, and inversions don't count
    hs = [_family(h) if h else None for _, _, h in marks]
    al = [_family(h) for _, _, h in aligned]
    if not marks or not aligned:
        return [None] * len(marks), 0.0
    pairs = _lcs(hs, al, gap_x, gap_y)
    if len(pairs) < 0.5 * sum(1 for h in hs if h is not None):
        roots = [h[0] if h else None for h in hs]
        pairs2 = _lcs(roots, [t[0] for t in al], gap_x, gap_y)
        if len(pairs2) > len(pairs):
            pairs = pairs2
    times: list[float | None] = [None] * len(marks)
    for i, j in pairs:
        times[i] = float(aligned[j][0])
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        return times, 0.0
    # interpolate the rest by index (and extrapolate at the ends with the mean spacing)
    span = (times[known[-1]] - times[known[0]]) / max(1, known[-1] - known[0]) if len(known) > 1 else 2.0
    for i in range(len(times)):
        if times[i] is not None:
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        if prev is not None and nxt is not None:
            f = (i - prev) / (nxt - prev)
            times[i] = times[prev] + f * (times[nxt] - times[prev])
        elif prev is not None:
            times[i] = times[prev] + (i - prev) * span
        else:
            times[i] = max(0.0, times[nxt] - (nxt - i) * span)
    # keep the sequence monotone
    for i in range(1, len(times)):
        if times[i] < times[i - 1]:
            times[i] = times[i - 1]
    share = len(pairs) / max(1, sum(1 for h in hs if h is not None))
    return times, share


def _is_cjk(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def _candidates(text: str) -> list[int]:
    """Where a chord may be written in a lyric line: the start, after a space, and every CJK character."""
    out = [0] if text else [0]
    for i, ch in enumerate(text):
        if i == 0:
            continue
        if _is_cjk(ch) or (text[i - 1].isspace() and not ch.isspace()):
            out.append(i)
    return sorted(set(out))


@dataclass
class Placement:
    lines: list[LyricLine]
    inserts: list[dict[int, list[str]]]      # per line: char index → labels written there
    placed: bool                             # our chords were placed (else the sheet's own chords stay)
    share: float = 0.0
    n_placed: int = 0
    after: dict[int, list[str]] = field(default_factory=dict)   # line → chords of an interlude written after it


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def place_chords(lines: list[LyricLine], aligned: list[tuple[float, float, str]],
                 ours: list[tuple[float, float, str]], min_share: float = 0.3, gap_s: float = 2.5,
                 events: list[tuple[float, float, str]] | None = None) -> Placement:
    """Write OUR chord changes (t0, t1, label) over the sheet's lyric lines, using the sheet's aligned
    chord events for the timing.

    Every chord marker of the sheet gets a time (``marker_times``). Inside a line the time runs linearly
    between its markers; the words before the first marker and after the last one run at the song's
    typical pace (seconds per letter, from the marked spans). When the recording has a long stretch
    between the end of one line and the start of the next (an interlude the sheet doesn't write out),
    the chords in it go on a chord-only line between the two. Each chord goes to the nearest syllable."""
    # the timing comes from OUR detected chords when given (``events``: every marker is lined up with the
    # chord sequence we are about to write, skipping one of our many chords is cheap, skipping a sheet
    # chord is not); the sheet's own aligned events are the fallback
    if events:
        times, share = marker_times(lines, events, gap_x=0.8, gap_y=0.25)
    else:
        times, share = marker_times(lines, aligned)
    inserts: list[dict[int, list[str]]] = [dict() for _ in lines]
    if not times or share < min_share:
        return Placement(lines, inserts, False, share, 0)
    text_lines = [li for li, ln in enumerate(lines) if ln.kind == "text"]
    anchors_of: dict[int, list[tuple[int, float]]] = {li: [] for li in text_lines}
    k = 0
    for li, ln in enumerate(lines):
        for (ci, _) in ln.markers:
            anchors_of[li].append((ci, float(times[k])))
            k += 1
    if sum(len(anchors_of[li]) for li in text_lines) < 2:
        return Placement(lines, inserts, False, share, 0)
    # the song's pace: seconds per letter, from spans between two markers of one line
    rates = [(t2 - t1) / (c2 - c1) for li in text_lines
             for (c1, t1), (c2, t2) in zip(anchors_of[li], anchors_of[li][1:]) if c2 > c1 and t2 > t1]
    first_t = min(t for li in text_lines for _, t in anchors_of[li])
    last_t = max(t for li in text_lines for _, t in anchors_of[li])
    total_chars = sum(max(1, len(lines[li].text)) for li in text_lines)
    rate = _median(rates) if rates else (last_t - first_t) / max(1, total_chars)
    rate = min(max(rate, 0.05), 2.0)
    # one chord's typical length: the words before a line's first marker / after its last one get at most
    # about that much time, so a long stretch between two lines shows up as an interlude
    all_t = [t for li in text_lines for _, t in anchors_of[li]]
    spans = [t2 - t1 for t1, t2 in zip(all_t, all_t[1:]) if t2 > t1]
    reach = 1.5 * (_median(spans) if spans else 2.0)

    # control points per line: (char, time), starting at char 0 and ending at len(text)
    ctrl: dict[int, list[tuple[int, float]]] = {}
    anchored = [li for li in text_lines if anchors_of[li]]
    prev_end = 0.0
    for idx, li in enumerate(anchored):
        n = max(1, len(lines[li].text))
        a = anchors_of[li]
        local = [(t2 - t1) / (c2 - c1) for (c1, t1), (c2, t2) in zip(a, a[1:]) if c2 > c1 and t2 > t1]
        r = min(max(_median(local), 0.05), 2.0) if local else rate      # this line's own pace when it has one
        nxt = anchors_of[anchored[idx + 1]][0][1] if idx + 1 < len(anchored) else a[-1][1] + r * n
        t_start = max(prev_end, a[0][1] - min(r * a[0][0], reach))
        t_end = min(nxt, a[-1][1] + min(r * (n - a[-1][0]), reach))
        t_end = max(t_end, a[-1][1])
        pts = [(0, t_start)] + [(c, t) for c, t in a if 0 < c < n] + [(n, t_end)]
        ctrl[li] = pts
        prev_end = t_end
    # lines with no marker: a span at the song's pace after the previous line, squeezed to fit before the next
    for li in text_lines:
        if li in ctrl:
            continue
        before = [x for x in anchored if x < li]
        after = [x for x in anchored if x > li]
        t0 = ctrl[before[-1]][-1][1] if before else max(0.0, first_t - rate * sum(max(1, len(lines[x].text)) for x in text_lines if x < li) - rate)
        limit = ctrl[after[0]][0][1] if after else float("inf")
        n = max(1, len(lines[li].text))
        t1 = min(t0 + rate * n, limit)
        ctrl[li] = [(0, t0), (n, max(t0, t1))]
        # later marker-less lines start after this one
        anchored.append(li)
        anchored.sort()

    def time_at(li: int, ci: int) -> float:
        pts = ctrl[li]
        for (c0, t0), (c1, t1) in zip(pts, pts[1:]):
            if c0 <= ci <= c1:
                return t0 if c1 == c0 else t0 + (ci - c0) / (c1 - c0) * (t1 - t0)
        return pts[-1][1] if ci > pts[-1][0] else pts[0][1]

    # interludes: a long stretch between the end of one line and the start of the next
    windows: list[tuple[float, float, int]] = []       # (t_from, t_to, line the chords go after)
    order = sorted(text_lines)
    for la, lb in zip(order, order[1:]):
        ta, tb = ctrl[la][-1][1], ctrl[lb][0][1]
        if tb - ta > gap_s:
            windows.append((ta, tb, la))

    cands: list[tuple[float, int, int]] = []           # (time, line, char)
    for li in text_lines:
        for ci in _candidates(lines[li].text):
            cands.append((time_at(li, ci), li, ci))
    cands.sort()
    lead = max(2.0, gap_s)
    after: dict[int, list[str]] = {}
    n_placed = 0
    for t0, _t1, lab in ours:
        if lab == "N" or t0 < first_t - lead or t0 > ctrl[order[-1]][-1][1] + lead:
            continue
        w = next((w for w in windows if w[0] + 0.3 < t0 < w[1] - 0.3), None)
        if w is not None:
            after.setdefault(w[2], []).append(lab)
            n_placed += 1
            continue
        best = min(cands, key=lambda c: abs(c[0] - t0))
        _, li, ci = best
        inserts[li].setdefault(ci, []).append(lab)
        n_placed += 1
    return Placement(lines, inserts, True, share, n_placed, after)


def render_chordpro(p: Placement) -> str:
    """Inline ``[C]word`` text: our chords when placed, else the sheet's own markers."""
    out: list[str] = []
    for li, ln in enumerate(p.lines):
        if ln.kind == "blank":
            out.append("")
        elif ln.kind == "label":
            out.append(f"[{ln.label}]" if ln.label is not None else ln.raw.strip())
        elif not p.placed:
            out.append(ln.raw.strip())
        elif not ln.text.strip():                # a chord-only line of the sheet: our chords, or nothing
            ins = p.inserts[li]
            out.append(" ".join(f"[{x}]" for i in sorted(ins) for x in ins[i]))
        else:
            ins = p.inserts[li]
            s = []
            for i, ch in enumerate(ln.text):
                if i in ins:
                    s.append("".join(f"[{x}]" for x in ins[i]))
                s.append(ch)
            if len(ln.text) in ins:
                s.append("".join(f"[{x}]" for x in ins[len(ln.text)]))
            out.append("".join(s).rstrip())
        if p.placed and p.after.get(li):                 # an interlude the sheet doesn't write out
            out.append(" ".join(f"[{x}]" for x in p.after[li]))
    return "\n".join(out).rstrip("\n") + "\n"


def _width(s: str) -> int:
    return sum(2 if _is_cjk(ch) else 1 for ch in s)


def two_line(chordpro: str) -> str:
    """Chords on the line above the words (CJK letters count two columns, for a monospace font)."""
    out: list[str] = []
    for raw in chordpro.splitlines():
        s = raw.rstrip()
        if not s.strip():
            out.append("")
            continue
        m = re.fullmatch(r"\s*\[([^\[\]]+)\]\s*", s)
        if m and _is_label(m.group(1)):
            out.append(f"[{m.group(1).strip()}]")
            continue
        if "[" not in s:
            out.append(s)
            continue
        chord_line = ""
        words = ""
        pos = 0
        for mm in _INLINE.finditer(s):
            words += s[pos:mm.start()]
            col = _width(words)
            if len(chord_line) < col:
                chord_line += " " * (col - len(chord_line))
            elif chord_line and not chord_line.endswith(" "):
                chord_line += " "
            chord_line += mm.group(1)
            pos = mm.end()
        words += s[pos:]
        if words.strip():
            out.append(chord_line.rstrip())
            out.append(words.rstrip())
        else:
            out.append(chord_line.rstrip())
    return "\n".join(out).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------------------------------
# the lyrics source: the applied / chosen / best sheet with lyrics
# ---------------------------------------------------------------------------------------------------

def lyrics_source(song_dir: Path, sheet_id: str | None = None) -> dict | None:
    """The sheet result whose lyrics to use: the one asked for, else the applied one, else the best
    result that has lyrics. Returns ``{"id", "source", "lyrics_text", "aligned", "title"}`` or None."""
    try:
        from ..sheets import lookup as L
    except Exception:  # noqa: BLE001
        return None
    st = L.load_state(song_dir)
    results = [r for r in st.get("results") or [] if isinstance(r, dict)]
    order: list[dict] = []
    if sheet_id:
        order += [r for r in results if r.get("id") == sheet_id]
    ap = st.get("applied") or {}
    if ap.get("id"):
        order += [r for r in results if r.get("id") == ap["id"]]
    order += results
    seen: set[str] = set()
    for r in order:
        rid = str(r.get("id"))
        if rid in seen:
            continue
        seen.add(rid)
        text = r.get("lyrics_text")
        if not text:
            p = L.sheets_dir(song_dir) / f"{rid.replace(':', '-')}.lyrics.txt"
            if p.exists():
                try:
                    text = p.read_text("utf-8")
                except OSError:
                    text = None
        if text and "[" in text:
            return {"id": rid, "source": r.get("source"), "title": r.get("title"), "lyrics_text": text,
                    "aligned": [(float(x["t0"]), float(x["t1"]), str(x["harte"])) for x in r.get("aligned") or []]}
    return None


# ---------------------------------------------------------------------------------------------------
# the export
# ---------------------------------------------------------------------------------------------------

@dataclass
class SheetExport:
    title: str
    artist: str | None
    key: str                         # after the transposition
    key_original: str
    transpose: int
    bpm: int
    meter: str
    level: str
    slash: bool
    chart_text: str                  # the bar chart (sections, | bars |)
    key_changes: list[str] = field(default_factory=list)   # "bar 17: E"
    intro: list[str] = field(default_factory=list)         # chords before bar 1
    lyrics_chordpro: str | None = None                     # [C]word text with OUR chords (or the sheet's)
    lyrics_text: str | None = None                         # the two-line form of the same
    lyrics_ours: bool = False                              # True: our chords were placed over the words
    lyrics_source: str | None = None                       # site / "pasted"
    lyrics_sheet_id: str | None = None
    notes: list[str] = field(default_factory=list)
    made: str = ""

    def header_lines(self) -> list[str]:
        out = [f"# {self.title}" + (f" — {self.artist}" if self.artist else "")]
        key = f"# Key: {self.key}"
        if self.transpose % 12:
            key += f"  (original {self.key_original}, {self.transpose:+d})"
        out.append(key)
        out.append(f"# Tempo: {self.bpm} BPM · Meter: {self.meter} · Chords: {self.level}"
                   + ("" if self.slash else " · no slash chords"))
        for kc in self.key_changes:
            out.append(f"# Key change at {kc}")
        out.append('# One token per beat; "." = the same chord continues; N = no chord.')
        if self.intro:
            out.append("# Before bar 1: " + " ".join(self.intro))
        for n in self.notes:
            out.append(f"# {n}")
        out.append(f"# Nathan Chordify · {self.made}")
        return out

    def to_text(self) -> str:
        parts = ["\n".join(self.header_lines()) + "\n", self.chart_text]
        if self.lyrics_text:
            parts.append("\n" + ("# Chords over the words" if self.lyrics_ours else "# The sheet's own chords over the words") + "\n")
            parts.append(self.lyrics_text)
        return "\n".join(p.rstrip("\n") for p in parts if p) + "\n"

    def to_chordpro(self) -> str:
        out = [f"{{title: {self.title}}}"]
        if self.artist:
            out.append(f"{{subtitle: {self.artist}}}")
        out.append(f"{{key: {self.key}}}")
        out.append(f"{{tempo: {self.bpm}}}")
        out.append(f"{{time: {self.meter}}}")
        for n in self.notes:
            out.append(f"{{comment: {n}}}")
        out.append(f"{{comment: Nathan Chordify · {self.made}}}")
        out.append("")
        if self.lyrics_chordpro:
            for ln in self.lyrics_chordpro.splitlines():
                m = re.fullmatch(r"\s*\[([^\[\]]+)\]\s*", ln)
                out.append(f"{{comment: {m.group(1).strip()}}}" if m and _is_label(m.group(1)) else ln)
            out.append("")
        out.append("{comment: Bar chart}")
        out.append("{start_of_tab}")
        out.append(self.chart_text.rstrip("\n"))
        out.append("{end_of_tab}")
        return "\n".join(out) + "\n"

    def to_json(self) -> dict:
        return {
            "title": self.title, "artist": self.artist, "key": self.key, "key_original": self.key_original,
            "transpose": self.transpose, "bpm": self.bpm, "meter": self.meter, "level": self.level, "slash": self.slash,
            "chart_text": self.chart_text, "key_changes": self.key_changes, "intro": self.intro,
            "lyrics_chordpro": self.lyrics_chordpro, "lyrics_text": self.lyrics_text, "lyrics_ours": self.lyrics_ours,
            "lyrics_source": self.lyrics_source, "lyrics_sheet_id": self.lyrics_sheet_id, "notes": self.notes,
            "made": self.made, "text": self.to_text(), "chordpro": self.to_chordpro() if self.lyrics_chordpro else None,
        }


def build(a: schema.Analysis, song_dir: Path | None = None, *, transpose: int = 0, level: str = "full",
          slash: bool = True, lyrics: bool = True, pref: str = "auto", sheet_id: str | None = None,
          title: str | None = None, artist: str | None = None) -> SheetExport:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    if pref not in S.PREFS:
        raise ValueError(f"acc must be one of {S.PREFS}")
    t = transpose % 12
    chords = main_chords(a)
    labels = label_main(a, chords, t, pref, level, slash)
    k0 = key_index(a.key.global_.tonic, a.key.global_.mode)
    meter = f"{a.meter[0].num}/{a.meter[0].den}" if a.meter else "4/4"
    changes = []
    for kc in a.key.changes or []:
        seg = next((s for s in a.key.segments if abs(s.start - kc.t) < 0.06), None)
        name = S.key_name(key_index(seg.tonic, seg.mode), t, pref) if seg else kc.to
        changes.append(f"bar {kc.bar + 1}: {name}")
    ex = SheetExport(
        title=title or a.source.title, artist=artist if artist is not None else a.source.artist,
        key=S.key_name(k0, t, pref), key_original=S.key_name(k0, 0, pref), transpose=transpose,
        bpm=int(round(a.tempo.bpm)), meter=meter, level=level, slash=slash,
        chart_text=bar_chart(a, chords, labels), key_changes=changes,
        intro=before_first_bar(a, chords, labels),
        made=datetime.now().strftime("%Y-%m-%d"),
    )
    if lyrics and song_dir is not None:
        src = lyrics_source(Path(song_dir), sheet_id)
        if src is not None:
            _add_lyrics(ex, a, chords, labels, src, t, k0)
    return ex


def _add_lyrics(ex: SheetExport, a: schema.Analysis, chords: list[schema.ChordOut], labels: list[str],
                src: dict, t: int, k0: int) -> None:
    lines = parse_lyrics(src["lyrics_text"])
    ours = [(c.start, c.end, lab) for c, lab in zip(chords, labels)]
    # consecutive repeats of one label are one change
    changes: list[tuple[float, float, str]] = []
    for t0, t1, lab in ours:
        if changes and changes[-1][2] == lab:
            changes[-1] = (changes[-1][0], t1, lab)
        else:
            changes.append((t0, t1, lab))
    # timing from the sheet's own alignment to the recording first (an audio match); when the sheet has
    # none, or it lines up too few of the markers, match the markers against our detected chord sequence
    pl = place_chords(lines, src.get("aligned") or [], changes)
    if not pl.placed:
        events = [(c.start, c.end, c.harte) for c in chords if c.harte != "N"]
        pl = place_chords(lines, [], changes, events=events)
    ex.lyrics_source = src.get("source")
    ex.lyrics_sheet_id = src.get("id")
    if pl.placed:
        ex.lyrics_chordpro = render_chordpro(pl)
        ex.lyrics_ours = True
    else:
        from ..sheets.parsers import transpose_chordpro
        text = src["lyrics_text"]
        try:
            text = transpose_chordpro(text, t, (k0 + t) % 12 + (12 if k0 >= 12 else 0)) if t else text
        except Exception:  # noqa: BLE001
            pass
        ex.lyrics_chordpro = "\n".join(ln.rstrip() for ln in text.splitlines()).rstrip("\n") + "\n"
        ex.lyrics_ours = False
        ex.notes.append("The chords over the words are the sheet's own (moved to this key), not the detected ones: "
                        "the sheet could not be lined up with the recording closely enough.")
    ex.lyrics_text = two_line(ex.lyrics_chordpro)
