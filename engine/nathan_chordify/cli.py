"""`nchord` command line (WP8, M1).

    nchord analyze <URL or file> [--out a.json] [--lab a.lab] [--chart] [--detail 0.8] [--level full]
    nchord chart <cache key or analysis.json> [--detail 0.8] [--level full] [--transpose N]
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

from . import config


def _progress(stage: str, pct: float, msg: str) -> None:
    print(f"  [{stage:<10}] {pct * 100:5.1f}%  {msg}", flush=True)


def _label(c, level: str, display=None, transpose: int = 0, pref: str = "auto") -> str:
    if display is not None and transpose:
        lv = {"simple": 0, "standard": 1, "full": 2}[level]
        return display.variants[f"{transpose % 12}:{pref}"].chords[c.id][lv]
    return getattr(c.label, level)


def chart_text(analysis, detail: float = 0.8, level: str = "full", display=None, transpose: int = 0,
               bars_per_line: int = 4) -> str:
    """Bar-text chart in the §0.1 format: one token per beat, '.' = no change on that beat.
    Chords that start between beats are joined to their beat's token with '~' (e.g. 'B ~B+7')."""
    from .export import flatten
    flat = flatten(analysis.chords, detail)
    bars = analysis.bars
    lines, row = [], []
    j = 0
    for bar in bars:
        tokens = []
        for k in range(bar.beats):
            b0 = bar.start + (bar.end - bar.start) * k / bar.beats
            b1 = bar.start + (bar.end - bar.start) * (k + 1) / bar.beats
            starting = []
            while j < len(flat) and flat[j].start < b1 - 1e-3:
                if flat[j].start >= b0 - 0.08 or (k == 0 and not tokens and flat[j].end > b0 + 0.05):
                    starting.append(flat[j])
                j += 1
            if not starting:
                tokens.append(".")
            else:
                names = [_label(c, level, display, transpose) for c in starting]
                on_beat = abs(starting[0].start - b0) < 0.08 or (k == 0 and starting[0].start < b0)
                tokens.append(" ~".join(names) if on_beat else "~" + " ~".join(names))
        row.append(" ".join(tokens))
        if len(row) == bars_per_line:
            lines.append("| " + " | ".join(row) + " |")
            row = []
    if row:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _summary(a) -> str:
    k = a.key.global_
    passing = sum(c.layer == "passing" for c in a.chords)
    rt = ", ".join(f"{n} {v:.1f}s" for n, v in a.stats.runtime_s.items())
    return (f"{a.source.title} — key {k.tonic} {k.mode} (conf {k.conf:.2f}), {a.tempo.bpm:.1f} BPM, "
            f"{len(a.bars)} bars, {len(a.chords) - passing} main + {passing} passing chords\n  time: {rt}")


def cmd_analyze(args) -> int:
    from . import pipeline
    from .export import write_lab
    from .ingest import IngestError
    t0 = time.perf_counter()
    try:
        a = pipeline.analyze(args.source, mode=args.mode, device=args.device,
                             progress=None if args.quiet else _progress)
    except IngestError as e:
        print(f"error: {e.message}\nhint: {e.hint}", file=sys.stderr)
        return 2
    print(_summary(a), f"\n  total {time.perf_counter() - t0:.1f}s")
    if args.out:
        Path(args.out).write_text(a.model_dump_json(by_alias=True, indent=1), "utf-8")
        print("wrote", args.out)
    if args.lab:
        write_lab(a, Path(args.lab), args.detail)
        print("wrote", args.lab)
    if args.chart:
        print()
        print(chart_text(a, args.detail, args.level))
    return 0


def _load(target: str):
    from . import schema
    p = Path(target)
    if not p.exists():
        p = config.CACHE_DIR / target / "analysis.json"
    a = schema.Analysis.model_validate_json(p.read_text("utf-8"))
    dp = p.parent / "display.json"
    d = schema.Display.model_validate_json(dp.read_text("utf-8")) if dp.exists() else None
    return a, d


def cmd_chart(args) -> int:
    a, d = _load(args.target)
    print(_summary(a))
    print(chart_text(a, args.detail, args.level, d, args.transpose))
    return 0


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="nchord", description="Chord recognition that keeps passing chords.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyze", help="analyse a YouTube link or an audio file")
    an.add_argument("source")
    an.add_argument("--mode", choices=["full", "fast"], default="full")
    an.add_argument("--device", default="cuda")
    an.add_argument("--out")
    an.add_argument("--lab")
    an.add_argument("--chart", action="store_true")
    an.add_argument("--detail", type=float, default=0.8, help="0 Simple … 0.5 Standard … 0.8 Detailed … 1 Everything")
    an.add_argument("--level", choices=["simple", "standard", "full"], default="full")
    an.add_argument("--quiet", action="store_true")
    an.set_defaults(fn=cmd_analyze)
    ch = sub.add_parser("chart", help="print the chart of an analysed song")
    ch.add_argument("target", help="cache key (e.g. yt-Y7boUFbM76k) or a path to analysis.json")
    ch.add_argument("--detail", type=float, default=0.8)
    ch.add_argument("--level", choices=["simple", "standard", "full"], default="full")
    ch.add_argument("--transpose", type=int, default=0)
    ch.set_defaults(fn=cmd_chart)
    sv = sub.add_parser("serve", help="run the local server + web UI (options as for python -m nathan_chordify.server)")
    sv.add_argument("server_args", nargs=argparse.REMAINDER)
    sv.set_defaults(fn=lambda a: __import__("nathan_chordify.server.__main__", fromlist=["main"]).main(a.server_args))
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
