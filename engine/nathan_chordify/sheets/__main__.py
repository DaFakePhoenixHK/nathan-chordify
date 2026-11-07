"""Find a cached song's chord sheet online and line it up with the recording (nothing is applied).

    python -m nathan_chordify.sheets <cache key> [--query TEXT] [--sources cantonhymn,simplyworship]
                                     [--url LINK] [--chart FILE] [--offline] [--refine] [--json]

Prints every result's verdict (good / maybe / different), the transposition, the match numbers and
the first bars of the chart in the recording's key; results and chart files go to
cache/<key>/sheets/ like the server's search. ``--refine`` additionally runs the chart-guided
re-decode for the best result and prints how many chords would change (still writes nothing to
analysis.json or the edit log).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from .. import config


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m nathan_chordify.sheets", description=__doc__.splitlines()[0])
    ap.add_argument("key", help="cache key, e.g. yt-WLtqi4s29LQ")
    ap.add_argument("--query", help="search text instead of the video title")
    ap.add_argument("--sources", help="comma-separated sources to try, in order")
    ap.add_argument("--url", help="a sheet link from a supported site (no search)")
    ap.add_argument("--chart", help="a chart file (bar text or [C]words) to align (no search)")
    ap.add_argument("--offline", action="store_true", help="only pages already in the page cache")
    ap.add_argument("--refine", action="store_true", help="also run the guided re-decode for the best result")
    ap.add_argument("--json", action="store_true", help="print the state as JSON")
    ap.add_argument("--cache-dir", help="song cache (default: <repo>/cache)")
    a = ap.parse_args(argv)
    from . import align as A
    from . import guide as G
    from . import lookup as L
    from .fetch import Fetcher
    cache = Path(a.cache_dir) if a.cache_dir else config.CACHE_DIR
    d = cache / a.key
    if not (d / "analysis.json").exists():
        print(f"no analysis for {a.key} in {cache}")
        return 1
    fetcher = Fetcher(offline=a.offline)
    analysis = L.load_analysis(d)
    audio = A.load_audio(d, analysis)
    print(f"{a.key}: evidence from {audio.source}; recording key {L.S.key_name(audio.key_index)}")
    if a.chart:
        st = L.add_pasted(d, Path(a.chart).read_text("utf-8"), analysis=analysis, audio=audio, title=Path(a.chart).stem)
    elif a.url:
        st = L.add_url(d, a.url, fetcher, analysis=analysis, audio=audio)
    else:
        sources = tuple(s.strip() for s in a.sources.split(",")) if a.sources else None
        q = L.song_query(d, analysis, a.query)
        print(f"search terms: {q.terms} (artist {q.artist!r}, {q.language})")
        st = L.run_search(d, fetcher, query=a.query, analysis=analysis, audio=audio, sources=sources)
    if a.json:
        print(json.dumps(st, ensure_ascii=False, indent=1))
        return 0
    print(f"status: {st['status']}" + (f"  error: {st['error']}" if st.get("error") else "") +
          f"  requests: {fetcher.requests}")
    for n in st.get("notes") or []:
        print(f"  note: {n}")
    for r in st["results"]:
        m = r["match"]
        print(f"\n[{r['verdict']:<9}] {r['source']:<15} {r['title']} — {r.get('artist') or '?'}"
              f"{' (' + str(r['year']) + ')' if r.get('year') else ''}{'  ★ same video' if r.get('video_match') else ''}")
        print(f"  {r['url']}")
        print(f"  written {r['key_written']} → recording {r['key_recording']} (transpose {r['transpose']:+d}); "
              f"score {m['score']:.2f}, fit {m['align']:.2f}, title {m['title']:.2f}, coverage {m.get('coverage', 0):.2f}; "
              f"{r['n_chords']} chords, meter {r.get('meter')}")
        for w in r.get("warnings") or []:
            print(f"  ! {w}")
        for extra in r.get("also") or []:
            print(f"  = same chords: {extra}")
        body = [ln for ln in r["chart_text"].splitlines() if ln.strip()]
        for ln in body[:8]:
            print(f"    {ln}")
        if len(body) > 8:
            print(f"    … {len(body) - 8} more lines")
        if r.get("lyrics_text"):
            print(f"  lyrics with chords: {len(r['lyrics_text'].splitlines())} lines (cache/{a.key}/sheets/)")
    if a.refine and st["results"]:
        from .. import pipeline
        best = st["results"][0]
        chords = [(x["t0"], x["t1"], x["harte"]) for x in best["aligned"]]
        if (d / pipeline.STATE_NAME).exists():
            ref = G.refine_state(pipeline.load_state(d), chords, analysis=analysis)
        else:
            ref = G.refine_from_analysis(analysis, chords)
        ops, n_changed, n_kept = G.diff_ops(G.main_timeline(analysis), ref.main, "cli", "cli")
        print(f"\nrefine with [{best['verdict']}] {best['title']} ({ref.method}): {n_changed} chord stretches would change, "
              f"{n_kept} kept; {len(ref.flagged)} of {ref.n_sheet} sheet chords flagged (audio disagrees)")
        for f in ref.flagged[:8]:
            print(f"  flag {f['t0']:7.2f}s  detected {f['detected']:<14} sheet {f['sheet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
