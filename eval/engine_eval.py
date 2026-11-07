"""Engine predictors for ``run_eval`` (DESIGN §13.4) and the cached per-item decode states.

    python -m eval.engine_eval prepare --set D2          # analyze every item once (GPU) → eval/data/cache
    python -m eval.engine_eval b1 --set D2               # baseline B1 (lv-chordia + its own CRF) once per item

Predictors (``run_eval --pred``):
- ``@engine`` / ``@engine:<detail>``: our full pipeline (always re-decoded from the cached decode state),
  the flat timeline at a Detail level (default
  0.8 = Detailed). With ``--params '{"lam_scale": 0.9}'`` the item is re-decoded from its cached
  decode state (``pipeline.redecode``), which takes well under a second for a D2 item;
- ``@engine-simple``: baseline B5, our main layer only (Detail 0);
- ``@lv-crf``: baseline B1, ``lv_chordia.chord_recognition(abs_path, "submission")``.

Everything an item produces lives in ``eval/data/cache/<cache key>/`` (``config.CACHE_DIR`` is
pointed there while the engine runs, so the 72 mini-D2 items never land in the app cache).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_CACHE = ROOT / "eval" / "data" / "cache"
B1_DIR = EVAL_CACHE / "_b1"


@contextlib.contextmanager
def eval_cache(path: Path = EVAL_CACHE):
    """Point the engine's cache at ``path`` for the duration of the block."""
    from nathan_chordify import config
    old = config.CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    config.CACHE_DIR = path
    try:
        yield path
    finally:
        config.CACHE_DIR = old


def cache_key(audio: Path) -> str:
    from nathan_chordify.ingest import resolve
    return resolve(str(Path(audio).resolve()))[0]


_KEYS: dict[str, str] = {}


def item_key(item) -> str:
    """Cache key of an item's audio (sha256 of the file; memoised in eval/data/cache/keys.json)."""
    kf = EVAL_CACHE / "keys.json"
    if not _KEYS and kf.exists():
        _KEYS.update(json.loads(kf.read_text("utf-8")))
    k = _KEYS.get(item.id)
    if k is None:
        k = _KEYS[item.id] = cache_key(item.audio)
        EVAL_CACHE.mkdir(parents=True, exist_ok=True)
        kf.write_text(json.dumps(_KEYS, indent=1), "utf-8")
    return k


def item_dir(item) -> Path:
    return EVAL_CACHE / item_key(item)


def prepare_item(item, force: bool = False, device: str = "cuda", trim: bool = True) -> Path:
    """Run the full pipeline on an item once; returns its cache folder (with decode_state)."""
    from nathan_chordify import pipeline
    d = item_dir(item)
    if force or not (d / pipeline.STATE_NAME).exists():
        if item.audio is None:
            raise FileNotFoundError(f"{item.id}: no audio")
        with eval_cache():
            pipeline.analyze(str(Path(item.audio).resolve()), device=device, write=True)
        if trim:                                           # the source copy is never read again
            for f in d.glob("source.*"):
                f.unlink()
    return d


def lab_from_analysis(analysis, detail: float) -> tuple[np.ndarray, list[str]]:
    from nathan_chordify.export import flatten
    flat = flatten(analysis.chords, detail)
    iv = np.array([[c.start, c.end] for c in flat], dtype=float).reshape(-1, 2)
    return iv, [c.harte for c in flat]


def engine_prediction(item, detail: float = 0.8, params: dict | None = None):
    """Always re-decoded from the item's decode state with the CURRENT decoder (the analysis.json
    written by ``prepare`` may predate decoder changes); well under a second per D2 item."""
    from nathan_chordify import pipeline
    d = prepare_item(item)
    return lab_from_analysis(pipeline.redecode(d, params or None), detail)


def b1_prediction(item) -> tuple[np.ndarray, list[str]]:
    """Baseline B1: lv-chordia with its own CRF/HMM decoder and the 'submission' dictionary."""
    B1_DIR.mkdir(parents=True, exist_ok=True)
    out = B1_DIR / f"{item.id}.json"
    if not out.exists():
        from lv_chordia.chord_recognition import chord_recognition
        res = chord_recognition(str(Path(item.audio).resolve()), "submission")
        out.write_text(json.dumps(res), "utf-8")
    res = json.loads(out.read_text("utf-8"))
    iv = np.array([[r["start_time"], r["end_time"]] for r in res], dtype=float).reshape(-1, 2)
    return iv, [r["chord"] for r in res]


# ---------------------------------------------------------------------------------------------
# In-memory states for calibration and tuning
# ---------------------------------------------------------------------------------------------

def load_states(items) -> dict:
    """{item id: DecodeState} for every prepared item."""
    from nathan_chordify import pipeline
    out = {}
    for it in items:
        d = item_dir(it)
        if (d / pipeline.STATE_NAME).exists():
            out[it.id] = pipeline.load_state(d)
    return out


def gt_cells(grid, intervals: np.ndarray, labels: list[str], *, min_share: float = 0.75) -> np.ndarray:
    """Per-cell ground truth as vocabulary indices (−1 = unlabelled, mixed or out of vocabulary).

    A cell gets the label covering ≥ ``min_share`` of it; N maps to candidate 0."""
    from nathan_chordify.theory.harte import parse_harte
    from nathan_chordify.theory.vocab import index_of
    ct = np.asarray(grid.cell_t, dtype=float)
    G = len(ct) - 1
    out = np.full(G, -1, dtype=np.int64)
    ids = []
    for lab in labels:
        try:
            r, q, b = parse_harte(lab, strict=False)
            ids.append(index_of(r, q, b) if q != "N" else 0)
        except ValueError:
            ids.append(None)
    for (t0, t1), i in zip(intervals, ids):
        if i is None:
            continue
        a = np.maximum(ct[:-1], t0)
        b = np.minimum(ct[1:], t1)
        share = np.clip(b - a, 0, None) / np.maximum(np.diff(ct), 1e-9)
        out[share >= min_share] = i
    return out


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def _items(set_name: str):
    from .run_eval import discover, resolve_set
    items = []
    for s in set_name.split(","):
        items += discover(resolve_set(s.strip()))
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.engine_eval")
    ap.add_argument("cmd", choices=["prepare", "b1"])
    ap.add_argument("--set", default="D2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    items = _items(a.set)
    t_all = time.perf_counter()
    for n, it in enumerate(items):
        t = time.perf_counter()
        if a.cmd == "prepare":
            prepare_item(it, a.force, a.device)
        else:
            b1_prediction(it)
        print(f"[{n + 1}/{len(items)}] {it.id} {time.perf_counter() - t:.1f}s", flush=True)
    print(f"done in {time.perf_counter() - t_all:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
