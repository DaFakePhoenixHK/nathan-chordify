"""`.lab` export and the flat chord timeline at a Detail level (DESIGN §5.11, §5.13 field rules)."""
from __future__ import annotations

from pathlib import Path

from .. import schema


def visible(c: schema.ChordOut, detail: float) -> bool:
    """A passing chord shows when s > 0 and s ≥ min_detail (§5.11); main chords always show."""
    if c.layer == "main":
        return True
    return detail > 0 and c.min_detail is not None and detail >= c.min_detail


def flatten(chords: list[schema.ChordOut], detail: float = 0.8) -> list[schema.ChordOut]:
    """Non-overlapping timeline: main chords with the visible passing chords carved in.

    Returns copies whose start/end are the visible spans (a main chord split by an overlay
    appears twice).
    """
    mains = sorted((c for c in chords if c.layer == "main"), key=lambda c: c.start)
    over = []
    for o in sorted((c for c in chords if c.layer == "passing" and visible(c, detail)), key=lambda c: c.start):
        if not over or o.start >= over[-1].end - 1e-6:
            over.append(o)
    out = list(over)
    for m in mains:
        cur = m.start
        for o in over:
            if o.end <= cur + 1e-6 or o.start >= m.end - 1e-6:
                continue
            if o.start > cur + 1e-6:
                out.append(m.model_copy(update={"start": cur, "end": o.start}))
            cur = max(cur, o.end)
        if cur < m.end - 1e-6:
            out.append(m.model_copy(update={"start": cur, "end": m.end}))
    out.sort(key=lambda c: c.start)
    merged: list[schema.ChordOut] = []
    for c in out:
        if merged and merged[-1].harte == c.harte and abs(merged[-1].end - c.start) < 1e-3:
            merged[-1] = merged[-1].model_copy(update={"end": c.end})
        else:
            merged.append(c)
    return merged


def lab_lines(analysis: schema.Analysis, detail: float = 0.8) -> list[str]:
    return [f"{c.start:.3f}\t{c.end:.3f}\t{c.harte}" for c in flatten(analysis.chords, detail)]


def write_lab(analysis: schema.Analysis, path: Path, detail: float = 0.8) -> Path:
    path = Path(path)
    path.write_text("\n".join(lab_lines(analysis, detail)) + "\n", "utf-8")
    return path
