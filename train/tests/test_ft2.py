"""train/ft2.py: the two FT2 variants and the gate arms (no training here)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train import ft2  # noqa: E402


def test_variants_keep_the_ft1_safeguards():
    a, b = ft2.config("ft2-a"), ft2.config("ft2-b")
    assert a.bn_eval and b.bn_eval and a.l2sp == b.l2sp == 0.01 and a.batch == b.batch == 8
    assert (a.freeze_layers, a.freeze_notes) == (2, False)
    assert (b.freeze_layers, b.freeze_notes) == (6, True)
    assert ft2.POOL2.name == "pool2" and ft2.POOL1.name == "pool"


def test_arms_always_include_original_and_ft1b():
    arms = ft2.arms()
    assert arms["original"] == (None, None)
    w, t = arms["ft1-b"]
    assert w.name == "best.pt" and w.parent.name == "ft1-b" and t == ft2.FT1B_TEMP


def test_ram_pool_feeds_the_ft1_trainer():
    import pytest
    from train import piano as P
    if not P.available():
        pytest.skip("Salamander piano not downloaded")
    from train import finetune_ace as FT
    pool = ft2.RamPool("val", 2, workers=1, chunk=2, start=0)
    assert len(pool) == 2 and pool.hours > 0 and pool.nbytes() > 0
    assert [it["meta"]["id"] for it in pool.items] == ["val-7000002-0", "val-7000002-1"]
    x, r, b, n = FT.make_batch(pool, [0, 1], None)
    assert tuple(x.shape) == (2, 1, 144, 862)
