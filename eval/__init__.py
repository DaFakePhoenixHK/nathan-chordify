"""Evaluation package for Nathan_chordify (WP14, DESIGN §13).

Run modules from the repo root, e.g. ``python -m eval.run_eval --set eval/data/d2 --pred @oracle``
or ``python -m eval.synth.render --set mini-d2 --out eval/data/d2/``.

The metrics module is ``chord_metrics`` on purpose: a top-level module named ``metrics`` would
collide with consonance-ACE's ``metrics`` package (DESIGN §3).
"""
