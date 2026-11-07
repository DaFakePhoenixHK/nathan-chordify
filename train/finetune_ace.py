"""Fine-tune consonance-ACE on rendered worship-piano songs (FT1) and gate it on real songs.

    python -m train.finetune_ace baseline                    # the original ACE on the synthetic val split
    python -m train.finetune_ace train --run-id ft1-a        # fine-tune → models/ace_ft/ft1-a/{best,last}.pt
    python -m train.finetune_ace calibrate --run-id ft1-a    # T_ace on the synthetic val split (+ original)
    python -m train.finetune_ace gate --run-id ft1-a         # D1 (5 real songs) + mini-D2, original vs fine-tuned

**Data**: the pool of ``train/song_mix.py`` (``data/train/pool/{train,val}``), loaded into RAM. A
training example is a random 862-frame crop (= ACE's 20 s window) of an item's stored |CQT|,
divided by the peak |audio| of its frames (ACE's per-window peak normalisation), optionally shifted
by whole semitones (2 CQT bins each, zero fill; labels rotated). Validation = the first 862 frames
of every val item (fixed).

**Model**: ``ConformerDecomposedModel`` built from the original checkpoint
(``third_party/consonance-ACE/ACE/checkpoints/conformer_decomposed_smooth.ckpt``, 5.26 M parameters:
input projection, 6 Conformer layers of width 144, the chord-tone head and the root / bass heads
conditioned on it).

**Objective** (per frame): root and bass = cross-entropy over the 13 classes with ACE's consonance
label smoothing (α = 0.075, as the checkpoint); notes = binary cross-entropy (weight 2, as the
checkpoint). Note: ACE's own ``ConsonanceDecomposedLoss`` applies its log-softmax over TIME (the
logits arrive as [B, 13, T] and it normalises the last axis), so the original root/bass heads were
never trained to be normalised across classes; we fine-tune with the per-frame class softmax that
our adapter actually uses (``models/ace.py``: softmax over the 13 classes). ACE's cardinality term
(weight 0.05, thresholded, no gradient) is dropped.

**Against forgetting** what ACE learned from real recordings: small learning rate (7e-5 peak, linear
warm-up, cosine decay), the input projection and the first ``--freeze`` Conformer layers frozen,
an L2-SP penalty λ·Σ(θ − θ₀)² pulling the trainable weights towards the original, frozen layers kept
in eval mode and (``bn_eval``, default) every BatchNorm's running statistics kept as the original's
(re-estimating them on synthetic audio shifts the features on real recordings; run ft1-a did that),
and early stopping on the synthetic validation loss (patience in evaluations). bf16 autocast on CUDA.
``--freeze-layers 6 --freeze-notes`` trains only the root and bass branches (run ft1-b).

**Outputs**: ``models/ace_ft/<run_id>/best.pt`` and ``last.pt`` (``{"state_dict",
"hyper_parameters", "ft": {...}}``, the same layout as the original checkpoint, loadable with
``weights_only=True``), ``config.json``, ``log.jsonl`` (loss and per-head accuracy per evaluation),
``val_best.json``; an entry under ``"ace_ft"`` in ``models/MANIFEST.json`` (config, data stats,
sha256).

**Using the weights**: ``ACE_WEIGHTS=<path to best.pt>`` (or ``models.ace.Ace(weights=...)``); the
evidence cache then gets its own file names (``ace@mix.v1-ft<sha8>.npz``). The default stays the
original checkpoint unless the FT1 gate (``eval/reports/FT1.md``) passes.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import song_mix as SM  # noqa: E402

FT_DIR = ROOT / "models" / "ace_ft"
MANIFEST = ROOT / "models" / "MANIFEST.json"
WIN = SM.WIN_FRAMES
N_CLASS = 13


# ------------------------------------------------------------------------------------------ data

class PoolData:
    """All items of one split in RAM (float16 CQT views into the shard arrays)."""

    def __init__(self, split: str, root: Path | None = None, limit: int | None = None):
        self.split = split
        self.items: list[dict] = []
        for f in SM.shard_files(split, root):
            z = SM.load_shard(f)
            off = z["offsets"]
            for k, meta in enumerate(z["meta"]):
                a, b = int(off[k]), int(off[k + 1])
                self.items.append({"cqt": z["cqt"][:, a:b], "env": z["env"][a:b], "root": z["root"][a:b],
                                   "bass": z["bass"][a:b], "notes": z["notes"][a:b], "meta": meta})
                if limit and len(self.items) >= limit:
                    return

    def __len__(self) -> int:
        return len(self.items)

    @property
    def hours(self) -> float:
        return sum(it["meta"]["duration"] for it in self.items) / 3600


def crop(item: dict, start: int, shift: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(x [144, WIN] float32 peak-normalised, root [WIN], bass [WIN], notes [WIN, 12]) of one item,
    transposed by ``shift`` semitones (CQT rolled by 2·shift bins with zero fill)."""
    a, b = start, start + WIN
    x = item["cqt"][:, a:b].astype(np.float32)
    peak = float(item["env"][a:b].max()) if b <= len(item["env"]) else float(item["env"][a:].max())
    x = x / max(peak, 1e-8)
    root = item["root"][a:b].astype(np.int64)
    bass = item["bass"][a:b].astype(np.int64)
    notes = SM.notes_matrix(item["notes"][a:b])
    if x.shape[1] < WIN:                                    # (never for pool items: they are ≥ 20 s)
        pad = WIN - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad)))
        root = np.pad(root, (0, pad), constant_values=12)
        bass = np.pad(bass, (0, pad), constant_values=12)
        notes = np.pad(notes, ((0, pad), (0, 0)))
    if shift:
        x = shift_cqt(x, shift)
        root = np.where(root < 12, (root + shift) % 12, root)
        bass = np.where(bass < 12, (bass + shift) % 12, bass)
        notes = np.roll(notes, shift, axis=1)
    return x, root, bass, notes


def shift_cqt(x: np.ndarray, semitones: int) -> np.ndarray:
    """Move a [144, T] CQT up (+) or down (−) by whole semitones (2 bins each), zero fill."""
    k = 2 * int(semitones)
    out = np.zeros_like(x)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def make_batch(data: PoolData, idx: list[int], rng: np.random.Generator | None, max_shift: int = 0,
               p_shift: float = 0.5):
    """Tensors for items ``idx``: random crops (rng given) or crops at frame 0 (rng None)."""
    import torch
    xs, rs, bs, ns = [], [], [], []
    for i in idx:
        it = data.items[i]
        F = it["cqt"].shape[1]
        if rng is None:
            start, shift = 0, 0
        else:
            start = int(rng.integers(0, max(1, F - WIN + 1)))
            shift = int(rng.integers(-max_shift, max_shift + 1)) if (max_shift and rng.random() < p_shift) else 0
        x, r, b, n = crop(it, start, shift)
        xs.append(x)
        rs.append(r)
        bs.append(b)
        ns.append(n)
    return (torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(rs)),
            torch.from_numpy(np.stack(bs)), torch.from_numpy(np.stack(ns)))


# ----------------------------------------------------------------------------------------- model

def base_checkpoint() -> dict:
    from nathan_chordify.models import ace
    return ace.load_checkpoint()


def build_model(state_dict: dict | None = None, hparams: dict | None = None):
    """A trainable ConformerDecomposedModel (CPU) with the original (or the given) weights."""
    import inspect
    from nathan_chordify.models import ace
    cls = ace.import_ace()["model"]
    if state_dict is None or hparams is None:
        ck = base_checkpoint()
        state_dict = state_dict or ck["state_dict"]
        hparams = hparams or ck["hyper_parameters"]
    sig = inspect.signature(cls.__init__).parameters
    model = cls(loss="consonance_decomposed", **{k: v for k, v in dict(hparams).items() if k in sig})
    model.load_state_dict(state_dict, strict=True)
    return model


def set_train_mode(model, n_frozen: int, bn_eval: bool = True) -> None:
    """``model.train()``, but frozen Conformer layers stay in eval mode (no dropout, and their
    BatchNorm running statistics are NOT re-estimated on synthetic audio), and with ``bn_eval``
    every BatchNorm keeps the original running statistics."""
    import torch
    model.train()
    for layer in model.conformer.conformer_layers[:max(0, n_frozen)]:
        layer.eval()
    if bn_eval:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()


def freeze(model, n_layers: int, notes_head: bool = False) -> list[str]:
    """Freeze the input projection and the first ``n_layers`` Conformer layers (n ≥ 1), and with
    ``notes_head`` also the chord-tone head (``output_chord``); returns the names of the trainable
    parameters. ``freeze(model, 6, True)`` leaves only the root and bass branches trainable."""
    if notes_head:
        for p in model.output_chord.parameters():
            p.requires_grad_(False)
    if n_layers > 0:
        for p in model.input_projection.parameters():
            p.requires_grad_(False)
        for layer in model.conformer.conformer_layers[:n_layers]:
            for p in layer.parameters():
                p.requires_grad_(False)
    return [n for n, p in model.named_parameters() if p.requires_grad]


class Criterion:
    """Per-frame losses: smoothed class CE for root and bass, BCE for the chord tones."""

    def __init__(self, model, w_root: float = 1.0, w_bass: float = 1.0, w_notes: float = 2.0):
        self.sim = model.loss.similarity_matrix          # [13, 13] consonance similarity (buffer)
        self.alpha = float(model.loss.smoothing_alpha)
        self.w = (w_root, w_bass, w_notes)

    def smoothed(self, labels):
        import torch
        sim = self.sim.to(labels.device)[labels] * self.alpha            # [B, T, 13]
        sim = sim + torch.nn.functional.one_hot(labels, N_CLASS) * (1.0 - self.alpha)
        return sim / sim.sum(-1, keepdim=True)

    def parts(self, out: dict, root, bass, notes) -> dict:
        import torch.nn.functional as F
        lr = F.log_softmax(out["root"].float(), dim=-1)
        lb = F.log_softmax(out["bass"].float(), dim=-1)
        l_root = -(self.smoothed(root) * lr).sum(-1).mean()
        l_bass = -(self.smoothed(bass) * lb).sum(-1).mean()
        l_notes = F.binary_cross_entropy_with_logits(out["onehot"].float(), notes)
        total = self.w[0] * l_root + self.w[1] * l_bass + self.w[2] * l_notes
        return {"loss": total, "root": l_root, "bass": l_bass, "notes": l_notes}


def evaluate(model, data: PoolData, device: str = "cuda", batch: int = 16, per_item: bool = False) -> dict:
    """Validation metrics on the fixed crops: loss parts, root / bass accuracy (all frames and
    chord frames), N recall, chord-tone exact match and micro F1, root NLL (class softmax)."""
    import torch
    crit = Criterion(model)
    model.eval()
    tot = {k: 0.0 for k in ("loss", "root", "bass", "notes")}
    nb = 0
    cnt = {k: 0 for k in ("frames", "root_ok", "bass_ok", "chord", "root_ok_chord", "bass_ok_chord", "n_frames",
                          "n_ok", "notes_exact", "tp", "fp", "fn", "all_ok")}
    nll = 0.0
    items = []
    with torch.inference_mode():
        for b0 in range(0, len(data), batch):
            idx = list(range(b0, min(len(data), b0 + batch)))
            x, r, bs, n = (t.to(device) for t in make_batch(data, idx, None))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                out = model(x)
            parts = crit.parts(out, r, bs, n)
            for k in tot:
                tot[k] += float(parts[k])
            nb += 1
            pr = out["root"].float().argmax(-1)
            pb = out["bass"].float().argmax(-1)
            pn = out["onehot"].float() > 0
            nll += float(-(torch.log_softmax(out["root"].float(), -1).gather(-1, r[..., None])).sum())
            chord = r < 12
            ok_r, ok_b = pr == r, pb == bs
            ok_n = (pn == (n > 0.5)).all(-1)
            cnt["frames"] += r.numel()
            cnt["root_ok"] += int(ok_r.sum())
            cnt["bass_ok"] += int(ok_b.sum())
            cnt["chord"] += int(chord.sum())
            cnt["root_ok_chord"] += int((ok_r & chord).sum())
            cnt["bass_ok_chord"] += int((ok_b & chord).sum())
            cnt["n_frames"] += int((~chord).sum())
            cnt["n_ok"] += int((ok_r & ~chord).sum())
            cnt["notes_exact"] += int(ok_n.sum())
            cnt["all_ok"] += int((ok_r & ok_b & ok_n).sum())
            t = n > 0.5
            cnt["tp"] += int((pn & t).sum())
            cnt["fp"] += int((pn & ~t).sum())
            cnt["fn"] += int((~pn & t).sum())
            if per_item:
                for j, i in enumerate(idx):
                    items.append({"id": data.items[i]["meta"]["id"], "root": float(ok_r[j].float().mean()),
                                  "notes": float(ok_n[j].float().mean())})
    f = max(cnt["frames"], 1)
    P = cnt["tp"] / max(cnt["tp"] + cnt["fp"], 1)
    R = cnt["tp"] / max(cnt["tp"] + cnt["fn"], 1)
    res = {"loss": tot["loss"] / max(nb, 1), "loss_root": tot["root"] / max(nb, 1),
           "loss_bass": tot["bass"] / max(nb, 1), "loss_notes": tot["notes"] / max(nb, 1),
           "root_acc": cnt["root_ok"] / f, "bass_acc": cnt["bass_ok"] / f,
           "root_acc_chord": cnt["root_ok_chord"] / max(cnt["chord"], 1),
           "bass_acc_chord": cnt["bass_ok_chord"] / max(cnt["chord"], 1),
           "n_recall": cnt["n_ok"] / max(cnt["n_frames"], 1), "notes_exact": cnt["notes_exact"] / f,
           "notes_f1": 2 * P * R / max(P + R, 1e-9), "all_heads_ok": cnt["all_ok"] / f, "root_nll": nll / f,
           "frames": cnt["frames"]}
    res = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}
    if per_item:
        res["items"] = items
    return res


def breakdown(data: PoolData, items: list[dict]) -> dict:
    """Mean per-item root / chord-tone accuracy by source, meter and band (piano only or not)."""
    by: dict[str, dict[str, list]] = {}
    meta = {it["meta"]["id"]: it["meta"] for it in data.items}
    for r in items:
        m = meta[r["id"]]
        for key in (f"source:{m['source']}", f"meter:{m['meter']}",
                    "band:piano only" if not m["layers"] else "band:with band"):
            d = by.setdefault(key, {"root": [], "notes": []})
            d["root"].append(r["root"])
            d["notes"].append(r["notes"])
    return {k: {"n": len(v["root"]), "root": round(float(np.mean(v["root"])), 4),
                "notes": round(float(np.mean(v["notes"])), 4)} for k, v in sorted(by.items())}


# --------------------------------------------------------------------------------------- training

@dataclass
class TrainConfig:
    run_id: str
    lr: float = 7e-5
    warmup: int = 300
    max_steps: int = 9000
    batch: int = 8                  # 16 needs ~2.9 GB VRAM: too much next to the other GPU users
    freeze_layers: int = 2
    freeze_notes: bool = False      # also freeze the chord-tone head (train only the root / bass branches)
    bn_eval: bool = True            # keep every BatchNorm's running statistics (ft1-a ran with False)
    l2sp: float = 0.01
    weight_decay: float = 0.0
    eval_every: int = 500
    patience: int = 6               # evaluations without a val-loss improvement
    min_delta: float = 1e-3
    max_shift: int = 2              # random transposition of the CQT (semitones)
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda"
    max_minutes: float = 150.0


def _lr_at(step: int, c: TrainConfig) -> float:
    if step < c.warmup:
        return c.lr * (step + 1) / c.warmup
    prog = min(1.0, (step - c.warmup) / max(1, c.max_steps - c.warmup))
    return c.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_ckpt(path: Path, model, hparams: dict, ft: dict) -> None:
    import torch
    sd = {k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
          for k, v in model.state_dict().items()}
    tmp = path.with_name(path.name + ".part")
    torch.save({"state_dict": sd, "hyper_parameters": dict(hparams), "ft": ft}, tmp)
    tmp.replace(path)


def train(c: TrainConfig, train_data: PoolData | None = None, val_data: PoolData | None = None,
          out_root: Path | None = None) -> Path:
    import torch
    out = Path(out_root or FT_DIR) / c.run_id
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(c.seed)
    rng = np.random.default_rng(c.seed)
    dev = c.device if (c.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    t_load = time.perf_counter()
    train_data = train_data or PoolData("train")
    val_data = val_data or PoolData("val")
    print(f"data: train {len(train_data)} items ({train_data.hours:.1f} h), val {len(val_data)} items "
          f"({val_data.hours:.1f} h), loaded in {time.perf_counter() - t_load:.0f} s", flush=True)
    ck = base_checkpoint()
    model = build_model(ck["state_dict"], ck["hyper_parameters"]).to(dev)
    names = freeze(model, c.freeze_layers, c.freeze_notes)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    theta0 = [p.detach().clone() for p in params]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(params, lr=c.lr, weight_decay=c.weight_decay)
    crit = Criterion(model)
    cfg = {**asdict(c), "device": dev, "trainable_params": n_train, "all_params": n_all,
           "base_checkpoint": str(SM_base_ckpt_path()), "base_sha256": sha256(SM_base_ckpt_path()),
           "data": {"train_items": len(train_data), "train_hours": round(train_data.hours, 2),
                    "val_items": len(val_data), "val_hours": round(val_data.hours, 2)},
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(cfg, indent=1), "utf-8")
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    v0 = evaluate(model, val_data, dev, c.batch)
    rec = {"step": 0, "phase": "val", "lr": 0.0, **v0}
    log.write(json.dumps(rec) + "\n")
    log.flush()
    print(f"step 0 (original): val loss {v0['loss']:.4f} root {v0['root_acc']:.4f} bass {v0['bass_acc']:.4f} "
          f"notes exact {v0['notes_exact']:.4f} F1 {v0['notes_f1']:.4f}", flush=True)
    best, best_step, bad = v0["loss"], 0, 0
    hp = ck["hyper_parameters"]
    ft_meta = {"run_id": c.run_id, "base_sha256": cfg["base_sha256"]}
    save_ckpt(out / "best.pt", model, hp, {**ft_meta, "step": 0, "val": v0})
    t0 = time.perf_counter()
    order = rng.permutation(len(train_data))
    pos = 0
    run = {"loss": 0.0, "root": 0.0, "bass": 0.0, "notes": 0.0, "l2sp": 0.0, "n": 0}
    stop = "max_steps"
    step = 0
    for step in range(1, c.max_steps + 1):
        set_train_mode(model, c.freeze_layers, c.bn_eval)
        if pos + c.batch > len(order):
            order, pos = rng.permutation(len(train_data)), 0
        idx = order[pos:pos + c.batch].tolist()
        pos += c.batch
        x, r, b, n = (t.to(dev, non_blocking=True) for t in make_batch(train_data, idx, rng, c.max_shift))
        for g in opt.param_groups:
            g["lr"] = _lr_at(step - 1, c)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(x)
        parts = crit.parts(o, r, b, n)
        reg = sum(((p - p0) ** 2).sum() for p, p0 in zip(params, theta0))
        loss = parts["loss"] + c.l2sp * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, c.grad_clip)
        opt.step()
        for k in ("loss", "root", "bass", "notes"):
            run[k] += float(parts[k].detach())
        run["l2sp"] += float(reg.detach())
        run["n"] += 1
        if step % c.eval_every == 0 or step == c.max_steps:
            tr = {f"train_{k}": round(v / max(run["n"], 1), 5) for k, v in run.items() if k != "n"}
            run = {k: 0.0 for k in run} | {"n": 0}
            v = evaluate(model, val_data, dev, c.batch)
            el = time.perf_counter() - t0
            rec = {"step": step, "phase": "val", "lr": round(_lr_at(step - 1, c), 8), "elapsed_s": round(el, 1),
                   "s_per_step": round(el / step, 4),
                   "epoch": round(step * c.batch / len(train_data), 3), **tr, **v}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            improved = v["loss"] < best - c.min_delta
            print(f"step {step} ep {rec['epoch']:.2f} {el / 60:.1f} min | train {tr['train_loss']:.4f} "
                  f"(l2sp {tr['train_l2sp']:.3f}) | val {v['loss']:.4f} root {v['root_acc']:.4f} bass {v['bass_acc']:.4f} "
                  f"notes exact {v['notes_exact']:.4f} F1 {v['notes_f1']:.4f}{' *' if improved else ''}", flush=True)
            if improved:
                best, best_step, bad = v["loss"], step, 0
                save_ckpt(out / "best.pt", model, hp, {**ft_meta, "step": step, "val": v})
            else:
                bad += 1
            if bad >= c.patience:
                stop = f"early stop (no val improvement for {c.patience} evaluations)"
                break
            if el / 60 > c.max_minutes:
                stop = "time limit"
                break
    save_ckpt(out / "last.pt", model, hp, {**ft_meta, "step": step})
    log.close()
    summary = {"best_step": best_step, "best_val_loss": best, "stopped": stop, "steps": step,
               "gpu_minutes": round((time.perf_counter() - t0) / 60, 1), "original_val": v0}
    cfg.update(summary=summary, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "config.json").write_text(json.dumps(cfg, indent=1), "utf-8")
    print(json.dumps(summary, indent=1), flush=True)
    return out


def SM_base_ckpt_path() -> Path:
    from nathan_chordify.models import ace
    return ace.checkpoint_path()


def load_weights(path: Path):
    """(model on CPU, ft meta) from a fine-tuned checkpoint."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return build_model(ck["state_dict"], ck["hyper_parameters"]), ck.get("ft", {})


# --------------------------------------------------------------------------------------- manifest

def update_manifest(run_dir: Path, extra: dict | None = None, manifest: Path | None = None) -> dict:
    """Add/replace the run's entry under "ace_ft" in models/MANIFEST.json."""
    manifest = Path(manifest or MANIFEST)
    data = json.loads(manifest.read_text("utf-8")) if manifest.exists() else {}
    cfg = json.loads((run_dir / "config.json").read_text("utf-8"))
    files = []
    for name in ("best.pt", "last.pt"):
        p = run_dir / name
        if p.exists():
            files.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)})
    entry = {"run_id": cfg["run_id"], "files": files, "config": {k: v for k, v in cfg.items() if k not in ("summary",)},
             "summary": cfg.get("summary"), **(extra or {})}
    runs = [r for r in data.get("ace_ft", []) if isinstance(r, dict) and r.get("run_id") != cfg["run_id"]]
    data["ace_ft"] = runs + [entry]
    manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
    return entry


# ------------------------------------------------------------------------------------ calibration

def fit_temperature(model, data: PoolData, device: str = "cuda", every: int = 1) -> dict:
    """T_ace by class-level NLL (decode/calibrate.fit) on synthetic val items: ACE evidence on the
    item audio, grid = the known beats, GT = the item's chords (cells ≥ 75 % covered)."""
    from nathan_chordify.decode import calibrate as CAL
    from nathan_chordify.decode import evidence as EV
    from nathan_chordify.decode.evidence import cell_beats
    from nathan_chordify.models import ace
    from nathan_chordify.theory.vocab import candidates
    from nathan_chordify.types import ChordEvidence, DecoderParams
    from eval import engine_eval as EE
    V = candidates()
    p = replace(DecoderParams(), temp={})
    evs, gts, ws = [], [], []
    for k, it in enumerate(data.items[::max(1, int(every))]):
        m = it["meta"]
        bpb, sub = SM.METERS[m["meter"]]
        spb = 60.0 / m["bpm"]
        end = min(m["duration"], m["end_t"])
        n_bars = int((end - m["lead"]) / (spb * bpb))
        if n_bars < 2:
            continue
        grid = CAL.synthetic_grid(m["bpm"], n_bars, bpb, 4 if sub == 2 else 3, m["lead"])
        root, bass, notes = ace_heads(model, it, device)
        n = root.shape[0]
        ev = ChordEvidence(name="ace@mix", frame_t=np.arange(n) * SM.HOP / SM.SR, offset_s=0.0, root=root,
                           bass=bass, notes=notes, heads={},
                           head_labels={k2: list(v) for k2, v in ace.HEAD_LABELS.items()}, label_probs=None,
                           labels=None)
        evs.append(EV.build_full(grid, [ev], None, None, V, p))
        iv = np.array([[c[0], c[1]] for c in m["chords"]], dtype=float)
        labs = [c[2] for c in m["chords"]]
        gts.append(EE.gt_cells(grid, iv, labs))
        ws.append(cell_beats(grid))
    temps = CAL.fit(evs, gts, vocab=V, sources=["ace"], weights=ws)
    T = temps["ace"]
    ccls = np.array([c.cls for c in V.cands])
    tot = {1.0: 0.0, 0.5801: 0.0, T: 0.0}
    wsum = 0.0
    for ev, g, w in zip(evs, gts, ws):
        raw = ev.parts["aux:raw:ace"]
        gcls = np.array([V.cands[c].cls if c >= 0 else -1 for c in g])
        mask = ccls[None, :] == gcls[:, None]
        ok = (g >= 0) & mask.any(axis=1)
        for tt in tot:
            tot[tt] += CAL._nll(raw[ok], mask[ok], w[ok], tt)
        wsum += float(w[ok].sum())
    return {"T": round(T, 4), "nll_T1": round(tot[1.0] / wsum, 4), "nll_T0.5801": round(tot[0.5801] / wsum, 4),
            "nll_fit": round(tot[T] / wsum, 4), "beats": round(wsum, 1), "items": len(evs)}


def ace_heads(model, item: dict, device: str = "cuda") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ACE outputs for a pool item over its whole length, with the adapter's 20 s / 5 s-overlap
    windows (from the stored CQT; per-window peak normalisation from ``env``)."""
    import torch
    from nathan_chordify.models import ace
    F = item["cqt"].shape[1]
    a = ace.Ace()
    _, win, crop_ = a.window_frames()
    ws = ace.windows(F, win, crop_)
    root = np.zeros((F, 13), np.float32)
    bass = np.zeros((F, 13), np.float32)
    notes = np.zeros((F, 12), np.float32)
    model.eval()
    feats = []
    for s, _, _, _ in ws:
        x = item["cqt"][:, s:s + win].astype(np.float32)
        x = x / max(float(item["env"][s:s + win].max()), 1e-8)
        if x.shape[1] < win:
            x = np.pad(x, ((0, 0), (0, win - x.shape[1])))
        feats.append(x)
    with torch.inference_mode():
        out = model(torch.from_numpy(np.stack(feats))[:, None].to(device))
        r = torch.softmax(out["root"].float(), -1).cpu().numpy()
        b = torch.softmax(out["bass"].float(), -1).cpu().numpy()
        n = torch.sigmoid(out["onehot"].float()).cpu().numpy()
    for k, (s, _, k0, k1) in enumerate(ws):
        root[k0:k1] = r[k, k0 - s:k1 - s]
        bass[k0:k1] = b[k, k0 - s:k1 - s]
        notes[k0:k1] = n[k, k0 - s:k1 - s]
    return root, bass, notes


# ------------------------------------------------------------------------------------------- gate

def ft_tag(weights: Path) -> str:
    from nathan_chordify.models import ace
    return ace.weights_tag(weights)


def ft_evidence_path(d: Path, name: str, tag: str) -> Path:
    from nathan_chordify.models.base import evidence_path
    from nathan_chordify.models.ace import Ace
    return evidence_path(d, name, f"{Ace.adapter_version}-{tag}")


def prepare_evidence(weights: Path, dirs: list[Path], device: str = "cuda") -> dict:
    """Fine-tuned ACE evidence for cache folders (mix + harmonic stem), cached next to the
    original evidence as ``evidence/ace@<input>.v1-<tag>.npz``."""
    from eval import ace_eval as AE
    from nathan_chordify.models import ace
    m = ace.Ace(weights=weights)
    t = time.perf_counter()
    n = 0
    for d in dirs:
        for w in AE.inputs(d):
            m.run_cached(w, device, d)
            n += 1
    return {"inputs": n, "seconds": round(time.perf_counter() - t, 1), "tag": m.weights_tag}


@contextlib.contextmanager
def ace_swap(weights: Path | None, temp_ace: float | None):
    """While active, decode states carry the fine-tuned ACE evidence (instead of the original)
    and the decoder uses ``temp_ace``. ``weights=None``: the original evidence, default T."""
    from nathan_chordify import pipeline
    from nathan_chordify.models.base import load_evidence
    orig_load, orig_params = pipeline.load_state, pipeline.params_from
    tag = ft_tag(weights) if weights else None

    def load_state(target):
        st = orig_load(target)
        if tag is None:
            return st
        ft = []
        for e in st.evid:
            if e.name.startswith("ace@"):
                p = ft_evidence_path(st.dir, e.name, tag)
                if not p.exists():
                    raise FileNotFoundError(f"{p} missing: run prepare_evidence first")
                ft.append(load_evidence(p))
        st.evid = [e for e in st.evid if not e.name.startswith("ace@")] + ft
        return st

    def params_from(overrides):
        p = orig_params(overrides)
        if temp_ace is None:
            return p
        return replace(p, temp={**p.temp, "ace": float(temp_ace)})

    pipeline.load_state, pipeline.params_from = load_state, params_from
    try:
        yield
    finally:
        pipeline.load_state, pipeline.params_from = orig_load, orig_params


def d1_dirs() -> list[Path]:
    from eval import d1_eval as D1
    from nathan_chordify import config
    return [config.CACHE_DIR / s["key"] for s in D1.songs()]


def d2_dirs() -> list[Path]:
    from eval import ace_eval as AE
    from eval import engine_eval as EE
    return [EE.item_dir(it) for it in AE.d2_items()]


def score_d2(weights: Path | None, temp_ace: float | None) -> dict:
    """mini-D2 at Detailed, with the original ACE (weights None) or the fine-tuned one."""
    from eval import ace_eval as AE
    from eval import chord_metrics as cm
    from eval import engine_eval as EE
    from eval import run_eval as RE
    if weights is None:
        return AE.score_d2(None, True)[0]
    results = {}
    with ace_swap(weights, temp_ace):
        for it in AE.d2_items():
            est = EE.engine_prediction(it, AE.DETAIL, None)
            ref_iv, ref_lab = cm.load_lab(it.lab)
            results[it.id] = cm.evaluate_all(ref_iv, ref_lab, est[0], est[1], RE.item_beats(it, ref_iv),
                                             RE.ref_tags(it, ref_iv, ref_lab))
    return AE.summary(RE.aggregate(results))


def run_d1(weights: Path | None, temp_ace: float | None, out: str) -> dict:
    from eval import d1_eval as D1
    with ace_swap(weights, temp_ace):
        D1.main(["--out", out])
    return json.loads((ROOT / f"{out}.json").read_text("utf-8"))


def ace_alone_d1(weights: Path | None) -> dict:
    """Diagnostic: ACE's own frame outputs on the D1 mixes (no decoder) against the draft GT:
    root / bass argmax accuracy and chord-tone exact match (notes > 0.5 = the GT chord's pcs),
    over frames whose GT is a chord (X and N left out), pooled over the 5 songs."""
    from eval import chord_metrics as cm
    from eval import d1_eval as D1
    from nathan_chordify import config
    from nathan_chordify.models.base import evidence_path, load_evidence
    tot = {"frames": 0, "root": 0, "bass": 0, "notes": 0}
    per = {}
    for s in D1.songs():
        d = config.CACHE_DIR / s["key"]
        p = ft_evidence_path(d, "ace@mix", ft_tag(weights)) if weights else evidence_path(d, "ace@mix", 1)
        ev = load_evidence(p)
        iv, lab = cm.load_lab(ROOT / s["lab"])
        t = np.asarray(ev.frame_t)
        k = np.searchsorted(iv[:, 0], t, side="right") - 1
        ok = (k >= 0) & (t < iv[np.clip(k, 0, None), 1])
        n = {"frames": 0, "root": 0, "bass": 0, "notes": 0}
        for j in np.flatnonzero(ok):
            spec = SM.spec_of(lab[k[j]])
            if spec is None:
                continue
            n["frames"] += 1
            n["root"] += int(np.argmax(ev.root[j]) == spec.root)
            n["bass"] += int(np.argmax(ev.bass[j]) == spec.bass)
            pcs = set(np.flatnonzero(ev.notes[j] > 0.5).tolist())
            n["notes"] += int(pcs == set(spec.pcs))
        per[s["key"]] = {k2: round(v / max(n["frames"], 1), 4) for k2, v in n.items() if k2 != "frames"}
        for k2 in tot:
            tot[k2] += n[k2]
    return {"overall": {k2: round(v / max(tot["frames"], 1), 4) for k2, v in tot.items() if k2 != "frames"},
            "frames": tot["frames"], "songs": per}


def gate_decision(before: dict, after: dict) -> dict:
    """FT1 gate on D1 overall (Detailed): keep the fine-tuned model only if majmin_inv AND
    sevenths_inv improve, root does not drop and short-chord precision does not drop."""
    b, a = before["overall"]["detailed"], after["overall"]["detailed"]
    d = {k: round(a[k] - b[k], 4) for k in ("root", "majmin_inv", "sevenths_inv", "short_P", "short_R",
                                              "short_F1", "flicker_per_min", "bass_acc")}
    ok = d["majmin_inv"] > 0 and d["sevenths_inv"] > 0 and d["root"] >= 0 and d["short_P"] >= 0
    return {"delta": d, "pass": bool(ok)}


# -------------------------------------------------------------------------------------------- CLI

def _run_dir(run_id: str) -> Path:
    return FT_DIR / run_id


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m train.finetune_ace", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline", help="the original ACE on the synthetic val split")
    t = sub.add_parser("train", help="fine-tune")
    for f in TrainConfig.__dataclass_fields__.values():
        if f.name == "run_id":
            t.add_argument("--run-id", required=True)
        elif isinstance(f.default, bool):
            t.add_argument("--" + f.name.replace("_", "-"), action="store_true", default=f.default)
        else:
            t.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    t.add_argument("--train-limit", type=int, default=None)
    c = sub.add_parser("calibrate", help="T_ace on the synthetic val split (fine-tuned + original)")
    c.add_argument("--run-id", required=True)
    c.add_argument("--every", type=int, default=2, help="use every N-th val item (memory)")
    g = sub.add_parser("gate", help="D1 + mini-D2, original vs fine-tuned")
    g.add_argument("--run-id", required=True)
    g.add_argument("--temp-ace", type=float, default=None, help="default: the run's calibrated T")
    g.add_argument("--skip-d2", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "baseline":
        val = PoolData("val")
        model = build_model().to("cuda")
        res = evaluate(model, val, "cuda", per_item=True)
        res["breakdown"] = breakdown(val, res.pop("items"))
        print(json.dumps(res, indent=1))
        (FT_DIR / "baseline_val.json").parent.mkdir(parents=True, exist_ok=True)
        (FT_DIR / "baseline_val.json").write_text(json.dumps(res, indent=1), "utf-8")
    elif a.cmd == "train":
        kw = {k: getattr(a, k) for k in TrainConfig.__dataclass_fields__}
        cfg = TrainConfig(**kw)
        tr = PoolData("train", limit=a.train_limit)
        out = train(cfg, tr, PoolData("val"))
        val = PoolData("val")
        model, ft = load_weights(out / "best.pt")
        res = evaluate(model.to(cfg.device), val, cfg.device, per_item=True)
        res["breakdown"] = breakdown(val, res.pop("items"))
        (out / "val_best.json").write_text(json.dumps(res, indent=1), "utf-8")
        print(json.dumps(res["breakdown"], indent=1))
    elif a.cmd == "calibrate":
        out = _run_dir(a.run_id)
        val = PoolData("val")
        ft_model, _ = load_weights(out / "best.pt")
        res = {"fine_tuned": fit_temperature(ft_model.to("cuda"), val, "cuda", a.every),
               "original": fit_temperature(build_model().to("cuda"), val, "cuda", a.every)}
        (out / "calibration.json").write_text(json.dumps(res, indent=1), "utf-8")
        print(json.dumps(res, indent=1))
    elif a.cmd == "gate":
        out = _run_dir(a.run_id)
        w = out / "best.pt"
        T = a.temp_ace
        if T is None:
            T = json.loads((out / "calibration.json").read_text("utf-8"))["fine_tuned"]["T"]
        dirs = d1_dirs() + ([] if a.skip_d2 else d2_dirs())
        print("evidence:", prepare_evidence(w, dirs), flush=True)
        res = {"temp_ace_ft": T}
        res["d1_original"] = run_d1(None, None, "eval/reports/FT1-d1-original")["overall"]
        res["d1_ft"] = run_d1(w, T, f"eval/reports/FT1-d1-{a.run_id}")["overall"]
        res["gate"] = gate_decision({"overall": res["d1_original"]}, {"overall": res["d1_ft"]})
        res["ace_alone_d1_original"] = ace_alone_d1(None)
        res["ace_alone_d1_ft"] = ace_alone_d1(w)
        if not a.skip_d2:
            res["d2_original"] = score_d2(None, None)
            res["d2_ft"] = score_d2(w, T)
        (out / "gate.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), "utf-8")
        print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
