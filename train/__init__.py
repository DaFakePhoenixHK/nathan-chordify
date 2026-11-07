"""Training-data generators (restart 2026-11-01).

- ``train.piano``: a small SFZ sampler that plays the Salamander Grand Piano (CC-BY 3.0, Alexander Holm)
  from ``models/piano/salamander``;
- ``train.chords``: every chord of the engine vocabulary × voicings across the keyboard, rendered with
  that piano and exact labels (``python -m train.chords --stats | --preview N``).
"""
