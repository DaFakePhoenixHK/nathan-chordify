"""Run a separate copy of the app with fine-tuned ACE weights, next to the normal one.

    .venv\\Scripts\\python scripts\\try_ace_weights.py                # version A (ft1-a) on port 8766
    .venv\\Scripts\\python scripts\\try_ace_weights.py --run ft1-b --port 8767

Nothing of the normal app is touched: the copy keeps its songs, library and ground truth under
``data/variants/<run>/`` and listens on its own port. On the first start it copies the audio, stems
and cached evidence of the songs in ``cache/`` into its own cache and analyses them there (about a
minute per song with the models warm), so the songs you already have open straight away.

The fine-tuned weights are switched on with ``ACE_WEIGHTS=<run>`` and decoded with the temperature
fitted for them (``models/ace_ft/<run>/calibration.json``). ``DecoderParams`` has no setting for
that, so this script changes its default ``temp["ace"]`` in this process and in the spawned GPU
worker (which re-imports this file as ``__mp_main__``).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMP_ENV = "NCHORD_TRY_ACE_TEMP"


def use_temp() -> None:
    """Make every new DecoderParams decode ACE at $NCHORD_TRY_ACE_TEMP (unless ``temp`` is passed)."""
    t = os.environ.get(TEMP_ENV)
    if not t:
        return
    from nathan_chordify.types import DecoderParams
    if getattr(DecoderParams.__init__, "_try_ace_temp", False):
        return
    orig = DecoderParams.__init__

    def __init__(self, *a, **k):
        orig(self, *a, **k)
        if "temp" not in k:
            self.temp = {**self.temp, "ace": float(t)}

    __init__._try_ace_temp = True
    DecoderParams.__init__ = __init__


use_temp()          # module level on purpose: also runs in the spawned worker


def seed_songs(src: Path, dst: Path, tag: str) -> list[str]:
    """Copy each analysed song's audio, stems, rhythm cache, lv evidence and this run's ACE evidence."""
    keys = []
    for d in sorted(src.iterdir()):
        if not (d.is_dir() and (d / "analysis.json").exists() and (d / "audio.wav").exists()):
            continue
        out = dst / d.name
        keys.append(d.name)
        if (out / "meta.json").exists():
            continue
        (out / "evidence").mkdir(parents=True, exist_ok=True)
        for name in ("audio.wav", "play.m4a", "meta.json", "rhythm_raw.json"):
            if (d / name).exists():
                shutil.copy2(d / name, out / name)
        if (d / "stems").is_dir():
            shutil.copytree(d / "stems", out / "stems", dirs_exist_ok=True)
        for f in (d / "evidence").glob("lv@*.npz"):
            shutil.copy2(f, out / "evidence" / f.name)
        for f in (d / "evidence").glob(f"ace@*.v*-{tag}.npz"):
            shutil.copy2(f, out / "evidence" / f.name)
    return keys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="try_ace_weights", description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="ft1-a", help="fine-tuned run under models/ace_ft/ (default: ft1-a = version A)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-seed", action="store_true", help="don't copy and analyse the songs from cache/")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args(argv)

    cal = json.loads((ROOT / "models" / "ace_ft" / args.run / "calibration.json").read_text("utf-8"))
    os.environ["ACE_WEIGHTS"] = args.run
    os.environ[TEMP_ENV] = str(cal["fine_tuned"]["T"])
    use_temp()

    from nathan_chordify import config
    from nathan_chordify.models import ace
    base = ROOT / "data" / "variants" / args.run
    cache = base / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tag = ace.weights_tag(args.run)
    print(f"ACE weights {args.run} ({tag}), temperature {os.environ[TEMP_ENV]}; songs in {cache}", flush=True)

    if not args.no_seed:
        todo = [k for k in seed_songs(config.CACHE_DIR, cache, tag) if not (cache / k / "analysis.json").exists()]
        if todo:
            from nathan_chordify import pipeline
            config.CACHE_DIR = cache
            for i, k in enumerate(todo, 1):
                meta = json.loads((cache / k / "meta.json").read_text("utf-8"))
                print(f"[{i}/{len(todo)}] analysing {meta.get('title') or k} ...", flush=True)
                t = time.perf_counter()
                pipeline.analyze(meta.get("source_url") or meta.get("path"))
                print(f"      done in {time.perf_counter() - t:.0f} s", flush=True)
            from nathan_chordify.models import release_gpu
            release_gpu()

    if not args.no_open:
        url = f"http://127.0.0.1:{args.port}"
        threading.Timer(6.0, webbrowser.open, (url,)).start()
    from nathan_chordify.server.__main__ import main as serve
    return serve(["--port", str(args.port), "--cache-dir", str(cache), "--data-dir", str(base / "data"),
                  "--eval-dir", str(base / "eval")])


if __name__ == "__main__":
    sys.exit(main())
