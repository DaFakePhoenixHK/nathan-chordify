"""train/finetune_ace.py + the ACE adapter's weights option (fast: tiny fake pool, CPU)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train import finetune_ace as FT  # noqa: E402
from train import song_mix as S  # noqa: E402

torch = pytest.importorskip("torch")


def _ace_ok() -> bool:
    try:
        from nathan_chordify.models import ace
        ace.check_checkpoint()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_ace = pytest.mark.skipif(not _ace_ok(), reason="consonance-ACE checkpoint missing")


def _fake_shard(dirpath: Path, split: str, n: int = 3, F: int = 900, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out = dirpath / split
    out.mkdir(parents=True, exist_ok=True)
    root = np.concatenate([np.full(F // 2, 0, np.int8), np.full(F - F // 2, 7, np.int8)])
    bass = root.copy()
    notes = np.where(root == 0, (1 << 0) | (1 << 4) | (1 << 7), (1 << 7) | (1 << 11) | (1 << 2)).astype(np.uint16)
    metas, cqts = [], []
    for i in range(n):
        cqts.append(rng.random((S.N_BINS, F)).astype(np.float16))
        metas.append({"id": f"{split}-0-{i}", "duration": 21.0, "source": "diatonic", "meter": "4/4", "layers": [],
                      "bpm": 120.0, "lead": 0.0, "end_t": 21.0, "frames": F,
                      "chords": [[0.0, 10.4, "C:maj", False], [10.4, 21.0, "G:maj", False]]})
    np.savez(out / "shard-0-0000.npz", cqt=np.concatenate(cqts, axis=1),
             env=np.full(n * F, 0.5, np.float16), root=np.tile(root, n), bass=np.tile(bass, n),
             notes=np.tile(notes, n), offsets=np.arange(n + 1) * F, meta=np.array(json.dumps(metas)))


def test_pooldata_and_crop(tmp_path):
    _fake_shard(tmp_path, "train")
    d = FT.PoolData("train", tmp_path)
    assert len(d) == 3 and d.hours == pytest.approx(3 * 21 / 3600)
    x, r, b, n = FT.crop(d.items[0], 10)
    assert x.shape == (S.N_BINS, S.WIN_FRAMES) and r.shape == (S.WIN_FRAMES,) and n.shape == (S.WIN_FRAMES, 12)
    np.testing.assert_allclose(x, d.items[0]["cqt"][:, 10:10 + S.WIN_FRAMES].astype(np.float32) / 0.5, rtol=1e-6)
    # a transposition moves the CQT by 2 bins per semitone and rotates every label
    xs, rs, bs, ns = FT.crop(d.items[0], 10, shift=2)
    np.testing.assert_array_equal(xs[4:], x[:-4])
    assert np.all(xs[:4] == 0)
    assert set(np.unique(rs)) == {2, 9}
    np.testing.assert_array_equal(ns, np.roll(n, 2, axis=1))
    xb, rb, bb, nb = FT.make_batch(d, [0, 1], np.random.default_rng(0), max_shift=1)
    assert tuple(xb.shape) == (2, 1, S.N_BINS, S.WIN_FRAMES) and rb.dtype == torch.int64


def test_shift_cqt_down():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = FT.shift_cqt(x, -1)
    np.testing.assert_array_equal(y[:4], x[2:])
    assert np.all(y[4:] == 0)


@needs_ace
def test_criterion_and_freeze():
    model = FT.build_model()
    names = FT.freeze(model, 2)
    assert not any(n.startswith("input_projection") for n in names)
    assert not any(n.startswith("conformer.conformer_layers.0.") or n.startswith("conformer.conformer_layers.1.")
                   for n in names)
    assert any(n.startswith("conformer.conformer_layers.5.") for n in names)
    assert any(n.startswith("output_root") for n in names)
    heads = FT.freeze(FT.build_model(), 6, notes_head=True)
    assert heads and all(n.split(".")[0] in ("root_fusion", "output_root", "bass_fusion", "output_bass") for n in heads)
    crit = FT.Criterion(model)
    lab = torch.tensor([[0, 12]])
    sm = crit.smoothed(lab)
    assert torch.allclose(sm.sum(-1), torch.ones(1, 2))
    assert int(sm[0, 0].argmax()) == 0 and int(sm[0, 1].argmax()) == 12
    assert torch.allclose(sm, model.loss.create_smoothed_labels_vectorized(lab))   # = ACE's own smoothing
    out = {"root": torch.zeros(1, 2, 13), "bass": torch.zeros(1, 2, 13), "onehot": torch.zeros(1, 2, 12)}
    p = crit.parts(out, lab, lab, torch.zeros(1, 2, 12))
    assert float(p["root"]) == pytest.approx(np.log(13), rel=1e-5)        # uniform logits → log 13
    assert float(p["notes"]) == pytest.approx(np.log(2), rel=1e-5)


@needs_ace
def test_train_tiny_saves_loadable_checkpoint_and_manifest(tmp_path, monkeypatch):
    _fake_shard(tmp_path, "train", n=2)
    _fake_shard(tmp_path, "val", n=1, seed=1)
    tr, va = FT.PoolData("train", tmp_path), FT.PoolData("val", tmp_path)
    cfg = FT.TrainConfig(run_id="t", max_steps=2, eval_every=1, batch=2, warmup=1, device="cpu", patience=5)
    out = FT.train(cfg, tr, va, out_root=tmp_path / "ft")
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    log = [json.loads(x) for x in (out / "log.jsonl").read_text("utf-8").splitlines()]
    assert [r["step"] for r in log] == [0, 1, 2]
    assert {"root_acc", "bass_acc", "notes_exact", "notes_f1", "loss"} <= set(log[-1])
    model, ft = FT.load_weights(out / "last.pt")
    assert ft["run_id"] == "t" and ft["step"] == 2
    # frozen layers did not move; the heads did
    base = FT.build_model()
    sd, sd0 = model.state_dict(), base.state_dict()
    assert torch.equal(sd["input_projection.weight"], sd0["input_projection.weight"])
    assert not torch.equal(sd["output_root.weight"], sd0["output_root.weight"])
    man = tmp_path / "MANIFEST.json"
    man.write_text(json.dumps({"other": [1]}), "utf-8")
    e = FT.update_manifest(out, {"note": "x"}, manifest=man)
    data = json.loads(man.read_text("utf-8"))
    assert data["other"] == [1] and data["ace_ft"][0]["run_id"] == "t" and e["note"] == "x"
    assert {f["path"].split("\\")[-1].split("/")[-1] for f in e["files"]} == {"best.pt", "last.pt"}
    FT.update_manifest(out, manifest=man)                                   # replaces, no duplicate
    assert len(json.loads(man.read_text("utf-8"))["ace_ft"]) == 1


@needs_ace
def test_adapter_weights_option(tmp_path, monkeypatch):
    from nathan_chordify.models import ace
    from nathan_chordify.models.base import evidence_path
    model = FT.build_model()
    with torch.no_grad():
        model.output_root.bias += 1.0                       # any change: a "fine-tuned" file
    w = tmp_path / "best.pt"
    FT.save_ckpt(w, model, FT.base_checkpoint()["hyper_parameters"], {"run_id": "x"})
    monkeypatch.delenv(ace.WEIGHTS_ENV, raising=False)
    assert ace.resolve_weights() is None and ace.Ace().weights is None
    assert ace.Ace().adapter_version == 1
    monkeypatch.setenv(ace.WEIGHTS_ENV, str(w))
    a = ace.Ace()
    assert a.weights == w.resolve() and a.weights_tag.startswith("ft") and len(a.weights_tag) == 10
    assert a.adapter_version == f"1-{a.weights_tag}"
    p = evidence_path(tmp_path, "ace@mix", a.adapter_version)
    assert p.name == f"ace@mix.v1-{a.weights_tag}.npz"
    assert not p.match("ace@*.v1.npz")                       # never picked up as original evidence
    assert ace.Ace(weights="original").weights is None
    run = Path(ace.config.MODELS_DIR) / "ace_ft" / ace.FT1_RUN / "best.pt"
    if run.is_file():                                        # a run id resolves to its best.pt
        assert ace.resolve_weights(ace.FT1_RUN) == run.resolve()
    monkeypatch.setenv(ace.WEIGHTS_ENV, "original")
    assert ace.Ace().weights is None
    with pytest.raises(FileNotFoundError):
        ace.resolve_weights(tmp_path / "nope.pt")
    try:
        m = ace.load("cpu", w)
        assert torch.allclose(m.output_root.bias, model.output_root.bias)
        assert ace.load("cpu", w) is m and ace.load("cpu") is not m
    finally:
        ace.release()


def test_gate_decision():
    b = {"overall": {"detailed": {"root": 0.89, "majmin_inv": 0.84, "sevenths_inv": 0.70, "short_P": 0.2,
                                  "short_R": 0.5, "short_F1": 0.3, "flicker_per_min": 2.0, "bass_acc": 0.87}}}
    a = json.loads(json.dumps(b))
    a["overall"]["detailed"].update(majmin_inv=0.85, sevenths_inv=0.72)
    assert FT.gate_decision(b, a)["pass"]
    a["overall"]["detailed"]["root"] = 0.88
    assert not FT.gate_decision(b, a)["pass"]
    a["overall"]["detailed"].update(root=0.9, short_P=0.1)
    assert not FT.gate_decision(b, a)["pass"]


def test_lr_schedule():
    c = FT.TrainConfig(run_id="x", lr=1e-4, warmup=10, max_steps=110)
    assert FT._lr_at(0, c) == pytest.approx(1e-5)
    assert FT._lr_at(9, c) == pytest.approx(1e-4)
    assert FT._lr_at(10, c) == pytest.approx(1e-4)
    assert FT._lr_at(110, c) == pytest.approx(2e-6)


@needs_ace
def test_heads_only_training_keeps_the_notes_head_exact(tmp_path):
    """--freeze-layers 6 --freeze-notes: BatchNorm statistics and the chord-tone output stay the original's."""
    _fake_shard(tmp_path, "train", n=2)
    _fake_shard(tmp_path, "val", n=1, seed=1)
    tr, va = FT.PoolData("train", tmp_path), FT.PoolData("val", tmp_path)
    cfg = FT.TrainConfig(run_id="h", max_steps=2, eval_every=2, batch=2, warmup=1, device="cpu",
                         freeze_layers=6, freeze_notes=True)
    out = FT.train(cfg, tr, va, out_root=tmp_path / "ft")
    model, _ = FT.load_weights(out / "last.pt")
    base = FT.build_model()
    sd, sd0 = model.state_dict(), base.state_dict()
    changed = {k.split(".")[0] for k in sd if not torch.equal(sd[k], sd0[k])}
    assert changed and changed <= {"root_fusion", "output_root", "bass_fusion", "output_bass"}
    x = FT.make_batch(va, [0], None)[0]
    model.eval(), base.eval()
    with torch.no_grad():
        assert torch.allclose(model(x)["onehot"], base(x)["onehot"])
