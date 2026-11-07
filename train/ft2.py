"""FT2 driver: fine-tune consonance-ACE on the plain-chord, near-middle-C pool of ``train/song_mix_v2.py``
and gate it like FT1 (the user's recipe, 2026-11-01). New file: ``train/finetune_ace.py`` is used as a
library, unchanged (its ``train`` / ``evaluate`` / ``fit_temperature`` / gate helpers read any pool
folder in the shard format).

    python -m train.ft2 train --run ft2-a       # "retrain most of the model" (as ft1-a, BatchNorm statistics kept)
    python -m train.ft2 train --run ft2-b       # root + bass branches only (as ft1-b)
    python -m train.ft2 calibrate --run ft2-a   # T_ace on the FT2 val split (also the original's)
    python -m train.ft2 compare-val             # original / ft1-b / ft2-a / ft2-b on the FT2 AND the FT1 val sets
    python -m train.ft2 gate                    # D1 (Detailed + Simple) + mini-D2 for all four, gate vs original and ft1-b
    python -m train.ft2 all-in-ram              # pool rendered into RAM (no disk) → train a + b → calibrate → compare

Runs land in ``models/ace_ft/<run>/`` (best.pt, last.pt, config.json, log.jsonl, ft2.json,
val_best.json, calibration.json); comparisons in ``models/ace_ft/ft2_val_compare.json`` and
``models/ace_ft/ft2_gate.json``; D1 tables in ``eval/reports/FT2-d1-<arm>.json``. The app default is
NOT changed (``models/ace.py`` ``DEFAULT_WEIGHTS`` stays None); every fine-tuned model is opt-in with
``ACE_WEIGHTS=<run id>``.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import finetune_ace as FT  # noqa: E402
from train import song_mix_v2 as V2  # noqa: E402

POOL2 = V2.POOL_DIR
POOL1 = ROOT / "data" / "train" / "pool"
FT_DIR = FT.FT_DIR
FT1B = FT_DIR / "ft1-b"
FT1B_TEMP = 0.6267

#: the two FT2 variants (same safeguards as FT1; bn_eval keeps every BatchNorm's statistics)
RUNS = {
    "ft2-a": dict(freeze_layers=2, freeze_notes=False, lr=7e-5, max_steps=9000, patience=6),
    "ft2-b": dict(freeze_layers=6, freeze_notes=True, lr=1e-4, max_steps=6000, patience=4),
}


class RamPool:
    """The FT2 pool kept in memory (``FT.PoolData`` duck type: ``items``, ``hours``, ``len``).

    Used because C: had fallen below the coordinator's 20 GB floor (an antivirus scan and a grown
    page file, not our files): items are rendered by worker processes (``song_mix_v2.render_items``,
    deterministic by seed) and never written to disk."""

    KEYS = ("cqt", "env", "root", "bass", "notes", "meta")

    def __init__(self, split: str, n_items: int, workers: int = 4, chunk: int = 20, start: int = 0):
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from train import song_mix as SM
        self.split = split
        seed = V2.SEEDS[split]
        jobs = [list(range(a, min(start + n_items, a + chunk))) for a in range(start, start + n_items, chunk)]
        got: dict[int, dict] = {}
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, initializer=SM._init_worker) as ex:
            futs = [ex.submit(V2.render_items, split, seed, job) for job in jobs]
            for k, f in enumerate(as_completed(futs), 1):
                for it in f.result():
                    got[int(it["meta"]["id"].rsplit("-", 1)[1])] = {key: it[key] for key in self.KEYS}
                if k % 25 == 0 or k == len(jobs):
                    print(f"[{split}] {len(got)}/{n_items} items in RAM, {time.perf_counter() - t0:.0f} s", flush=True)
        self.items = [got[i] for i in sorted(got)]
        self.seconds = time.perf_counter() - t0

    def __len__(self) -> int:
        return len(self.items)

    @property
    def hours(self) -> float:
        return sum(it["meta"]["duration"] for it in self.items) / 3600

    def nbytes(self) -> int:
        return sum(it["cqt"].nbytes + it["env"].nbytes + it["root"].nbytes * 2 + it["notes"].nbytes
                   for it in self.items)


def config(run: str) -> FT.TrainConfig:
    return FT.TrainConfig(run_id=run, batch=8, warmup=300, l2sp=0.01, eval_every=500, bn_eval=True, **RUNS[run])


def train(run: str, tr=None, va=None) -> Path:
    t0 = time.perf_counter()
    tr = tr if tr is not None else FT.PoolData("train", POOL2)
    va = va if va is not None else FT.PoolData("val", POOL2)
    out = FT.train(config(run), tr, va)
    model, _ = FT.load_weights(out / "best.pt")
    model = model.to("cuda")
    res = FT.evaluate(model, va, "cuda", per_item=True)
    res["breakdown"] = breakdown(va, res.pop("items"))
    (out / "val_best.json").write_text(json.dumps(res, indent=1), "utf-8")
    side = {"recipe": "ft2 (train/song_mix_v2.py)",
            "pool": "in RAM (song_mix_v2.render_items)" if isinstance(tr, RamPool) else str(POOL2),
            "train_items": len(tr),
            "train_hours": round(tr.hours, 2), "val_items": len(va), "val_hours": round(va.hours, 2),
            "wall_minutes": round((time.perf_counter() - t0) / 60, 1)}
    (out / "ft2.json").write_text(json.dumps(side, indent=1), "utf-8")
    return out


def breakdown(data: FT.PoolData, items: list[dict]) -> dict:
    """FT1's breakdown plus the FT2 arrangement (solo / soft / band)."""
    import numpy as np
    res = FT.breakdown(data, items)
    meta = {it["meta"]["id"]: it["meta"] for it in data.items}
    by: dict[str, list] = {}
    for r in items:
        by.setdefault(f"arrangement:{meta[r['id']].get('arrangement', '?')}", []).append(r)
    for k, rs in sorted(by.items()):
        res[k] = {"n": len(rs), "root": round(float(np.mean([r["root"] for r in rs])), 4),
                  "notes": round(float(np.mean([r["notes"] for r in rs])), 4)}
    return res


def calibrate(run: str, every: int = 2, va=None) -> dict:
    out = FT_DIR / run
    va = va if va is not None else FT.PoolData("val", POOL2)
    model, _ = FT.load_weights(out / "best.pt")
    res = {"fine_tuned": FT.fit_temperature(model.to("cuda"), va, "cuda", every),
           "original": FT.fit_temperature(FT.build_model().to("cuda"), va, "cuda", every), "val_set": "FT2"}
    (out / "calibration.json").write_text(json.dumps(res, indent=1), "utf-8")
    return res


def arms() -> dict[str, tuple[Path | None, float | None]]:
    """name → (weights, T_ace): the original, ft1-b and the FT2 runs that have a calibration."""
    out: dict[str, tuple[Path | None, float | None]] = {"original": (None, None), "ft1-b": (FT1B / "best.pt", FT1B_TEMP)}
    for run in RUNS:
        cal = FT_DIR / run / "calibration.json"
        if cal.exists():
            out[run] = (FT_DIR / run / "best.pt", json.loads(cal.read_text("utf-8"))["fine_tuned"]["T"])
    return out


def compare_val(va2=None) -> dict:
    """Synthetic val metrics of every arm on the FT2 val set and on the FT1 val set."""
    sets = {"FT2 val": va2 if va2 is not None else FT.PoolData("val", POOL2), "FT1 val": FT.PoolData("val", POOL1)}
    res: dict = {}
    for name, (w, _) in arms().items():
        model = (FT.load_weights(w)[0] if w else FT.build_model()).to("cuda")
        res[name] = {}
        for sname, data in sets.items():
            r = FT.evaluate(model, data, "cuda", per_item=True)
            bd = breakdown(data, r.pop("items")) if sname == "FT2 val" else FT.breakdown(data, r.pop("items"))
            res[name][sname] = {**{k: r[k] for k in ("root_acc", "bass_acc", "notes_exact", "notes_f1", "n_recall",
                                                      "root_nll", "loss")}, "breakdown": bd}
        print(name, {s: {k: v for k, v in x.items() if k != "breakdown"} for s, x in res[name].items()}, flush=True)
    (FT_DIR / "ft2_val_compare.json").write_text(json.dumps(res, indent=1), "utf-8")
    return res


def run_all_in_ram(n_train: int = 6000, n_val: int = 700, workers: int = 4) -> None:
    """Everything that needs the FT2 pool, in one process with the pool in RAM: render val + train,
    pool stats, train ft2-a and ft2-b, calibrate both, compare on the FT2 and FT1 val sets."""
    va = RamPool("val", n_val, workers)
    tr = RamPool("train", n_train, workers)
    st = {"note": "pool kept in RAM (C: below the 20 GB floor); items are deterministic by seed",
          "train": V2.stats_metas([it["meta"] for it in tr.items]), "val": V2.stats_metas([it["meta"] for it in va.items]),
          "ram_bytes": {"train": tr.nbytes(), "val": va.nbytes()},
          "render_minutes": {"train": round(tr.seconds / 60, 1), "val": round(va.seconds / 60, 1)}}
    (FT_DIR / "ft2_pool_stats.json").write_text(json.dumps(st, indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps({k: {kk: st[k][kk] for kk in ("items", "hours", "plain_share", "slash_share", "N_frame_share")}
                      for k in ("train", "val")}), flush=True)
    for run in RUNS:
        print("trained", train(run, tr, va), flush=True)
    for run in RUNS:
        print(run, json.dumps(calibrate(run, 2, va)), flush=True)
    compare_val(va)


def gate(skip_d2: bool = False) -> dict:
    """D1 (third use; nothing is tuned on it) + mini-D2 for the original, ft1-b, ft2-a, ft2-b, with
    FT1's gate rule against the original and against ft1-b."""
    A = arms()
    dirs = FT.d1_dirs() + ([] if skip_d2 else FT.d2_dirs())
    res: dict = {"temps": {k: v[1] for k, v in A.items()}, "d1": {}, "ace_alone_d1": {}, "d2": {}, "gate": {}}
    for name, (w, _) in A.items():
        if w is not None:
            print(name, "evidence:", FT.prepare_evidence(w, dirs), flush=True)
    for name, (w, T) in A.items():
        res["d1"][name] = FT.run_d1(w, T, f"eval/reports/FT2-d1-{name}")["overall"]
        res["ace_alone_d1"][name] = FT.ace_alone_d1(w)
        if not skip_d2:
            res["d2"][name] = FT.score_d2(w, T)
    for name in A:
        if name.startswith("ft2"):
            res["gate"][name] = {
                "vs_original": FT.gate_decision({"overall": res["d1"]["original"]}, {"overall": res["d1"][name]}),
                "vs_ft1-b": FT.gate_decision({"overall": res["d1"]["ft1-b"]}, {"overall": res["d1"][name]}),
                "simple_vs_original": {k: round(res["d1"][name]["simple"][k] - res["d1"]["original"]["simple"][k], 4)
                                       for k in ("root", "majmin_inv", "sevenths_inv", "short_P", "bass_acc")},
            }
    (FT_DIR / "ft2_gate.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), "utf-8")
    return res


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m train.ft2", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run", choices=list(RUNS), required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--run", choices=list(RUNS), required=True)
    c.add_argument("--every", type=int, default=2)
    sub.add_parser("compare-val")
    r = sub.add_parser("all-in-ram", help="render the pool into RAM, train ft2-a + ft2-b, calibrate, compare")
    r.add_argument("--train-items", type=int, default=6000)
    r.add_argument("--val-items", type=int, default=700)
    r.add_argument("--workers", type=int, default=4)
    g = sub.add_parser("gate")
    g.add_argument("--skip-d2", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        print(train(a.run))
    elif a.cmd == "calibrate":
        print(json.dumps(calibrate(a.run, a.every), indent=1))
    elif a.cmd == "compare-val":
        compare_val()
    elif a.cmd == "all-in-ram":
        run_all_in_ram(a.train_items, a.val_items, a.workers)
    elif a.cmd == "gate":
        r = gate(a.skip_d2)
        print(json.dumps({k: r[k] for k in ("temps", "gate")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
