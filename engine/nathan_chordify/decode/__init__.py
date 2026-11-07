"""Fusion and decoding (WP7, DESIGN §5.8–5.12, §5.15).

- ``evidence``: calibrated cell evidence (``build``), candidate pruning (``prune``), the 11.6 ms
  timeline helpers and the source registry;
- ``calibrate``: temperatures by NLL (``fit``) and the break-even table;
- ``hsmm``: semi-Markov Viterbi + forward–backward (``decode``);
- ``layers``: main / passing layers, salience, class-level conf, min_detail (``build``);
- ``post``: bass events, phantoms, A–P–A demotion, merging, display members, alternatives
  (``finalize``);
- ``sections``: repeats and auto sections (M5 stub);
- ``params``: ``DecoderParams`` and the fixed constants.
"""
from . import calibrate, evidence, hsmm, layers, params, post, sections  # noqa: F401
from .params import DecoderParams, fast_rescan  # noqa: F401
