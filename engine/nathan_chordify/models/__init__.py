"""Chord-model adapters (DESIGN §5.5): ``base.ChordModel`` and its subclasses.

Adapters import torch lazily, so importing this package is cheap.
"""
from __future__ import annotations

from .base import ChordModel, evidence_path, input_name, load_evidence, save_evidence

__all__ = ["ChordModel", "evidence_path", "input_name", "load_evidence", "save_evidence",
           "release_gpu"]


def release_gpu() -> None:
    """Unload every resident chord model (OOM ladder step 2, §5.14)."""
    import sys
    for name in ("lv_chordia", "ace"):
        mod = sys.modules.get(f"{__name__}.{name}")
        if mod is not None:
            mod.release()
