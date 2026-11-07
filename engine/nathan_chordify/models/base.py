"""Chord-model adapter interface and the evidence cache (DESIGN §4, §5.5, §5.14).

Each adapter turns one audio input into a ``ChordEvidence`` at the model's native hop.
Results are cached as ``cache/<key>/evidence/<name>.v<adapter_version>.npz`` (``name`` is the
evidence name, e.g. ``lv@harmonic``), so a new adapter version re-runs only that model.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..types import ChordEvidence

_OPTIONAL = ("root", "bass", "notes", "label_probs")


def input_name(wav: Path) -> str:
    """The input tag of an evidence name: ``audio.wav`` → "mix", ``stems/harmonic.flac`` → "harmonic"."""
    stem = Path(wav).name.split(".")[0]
    return "mix" if stem in ("audio", "mix") else stem


def evidence_path(key_dir: Path, name: str, adapter_version: int) -> Path:
    return Path(key_dir) / "evidence" / f"{name}.v{adapter_version}.npz"


def save_evidence(ev: ChordEvidence, path: Path) -> None:
    """Write ``ev`` to ``path`` atomically (no pickles: labels go in a JSON string)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"frame_t": np.asarray(ev.frame_t)}
    for k in _OPTIONAL:
        v = getattr(ev, k)
        if v is not None:
            arrays[k] = np.asarray(v)
    for k, v in ev.heads.items():
        arrays[f"head__{k}"] = np.asarray(v)
    meta = {"name": ev.name, "offset_s": float(ev.offset_s), "head_labels": ev.head_labels,
            "labels": ev.labels, "heads": list(ev.heads)}
    arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load_evidence(path: Path) -> ChordEvidence:
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        opt = {k: (z[k] if k in z.files else None) for k in _OPTIONAL}
        heads = {k: z[f"head__{k}"] for k in meta["heads"]}
        return ChordEvidence(name=meta["name"], frame_t=z["frame_t"], offset_s=meta["offset_s"],
                             heads=heads, head_labels=meta["head_labels"], labels=meta["labels"], **opt)


class ChordModel(ABC):
    """One chord-recognition network. Subclasses set ``name`` and ``adapter_version``."""

    name: str = ""                 # evidence prefix: "lv", "ace", "btc"
    adapter_version: int = 1       # bump when the adapter's output changes (invalidates the cache)

    @abstractmethod
    def run(self, wav: Path, device: str) -> ChordEvidence:
        """Run the model on one input (any soundfile-readable file) and return its evidence."""

    def evidence_name(self, wav: Path) -> str:
        return f"{self.name}@{input_name(wav)}"

    def run_cached(self, wav: Path, device: str, key_dir: Path) -> ChordEvidence:
        """``run`` with the npz cache under ``key_dir`` (= ``AudioAsset.dir``)."""
        path = evidence_path(key_dir, self.evidence_name(wav), self.adapter_version)
        if path.exists():
            return load_evidence(path)
        ev = self.run(wav, device)
        save_evidence(ev, path)
        return ev
