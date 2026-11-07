"""WP7 repeats and automatic sections (DESIGN §5.15) — M5.

TODO(M5): self-similarity of per-beat [harmonic chroma (12) ‖ class posteriors reduced to
root + maj/min (25)], diagonal stripes at bar-multiple lags (≥ 4 bars, mean similarity ≥ 0.7),
grouping into A/B/C…, the per-cell consensus class (≥ 2/3 of the repeats agree) for the
repetition prior, and the Intro/Verse/Pre-chorus/Chorus/Bridge/Interlude/Outro labels from
position and vocal-stem energy.
"""
from __future__ import annotations

import numpy as np

from ..types import RhythmGrid, Stems


def find(grid: RhythmGrid, stems: Stems | None, post_cls: np.ndarray) -> tuple[list[dict], np.ndarray | None]:
    """Auto sections and the consensus class per cell. Stub until M5: no sections, rep = None."""
    return [], None
