"""Music theory core (WP6): vocabulary and equivalence classes, templates, Harte labels,
key-aware spelling, Nashville / Roman numbers, simplify levels, piano voicings, priors and
bar-text charts. Pure Python + NumPy; no audio. ``key.py`` (M1/M5) and ``display.py`` (needs
the output schema) come later.

Keys are integers 0..23 as in ``KeyTrack.per_beat`` (0–11 major tonic pc, 12–23 minor tonic
pc + 12). Chords are (root pc, internal quality, bass pc) triples or ``types.Candidate``; N is
(-1, "N", -1).
"""
from . import templates  # noqa: F401  (cheap; the other modules are imported on demand)
