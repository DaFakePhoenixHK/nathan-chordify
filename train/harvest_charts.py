"""Harvest real worship chord progressions from free, no-login chord charts -> data/charts/.

What is kept: the CHORD SEQUENCE of each candidate song (data/songs_candidates.yaml), plus section
labels, key, meter and tempo when the page gives them, written in the repo's bar-text chart format
(engine/nathan_chordify/theory/chart.py, one token per beat, "." = the chord continues):

    [Chorus]
    | G . A . | Bm | G . A . | Dsus4 . D . |

What is never kept: lyrics (they are read in memory only, to estimate how long each chord lasts,
and dropped before anything is written) and audio. Raw downloaded pages hold lyrics, so they are
cached OUTSIDE the repo (default %TEMP%/nathan_chordify_chart_cache) and only used to avoid asking
the sites twice. YouTube is never used.

Sources (all readable without an account):
  - WorshipTogether.com song pages (English): chords over lyrics, original key, BPM.
  - CantonHymn.net (Cantonese / Mandarin): the site's own song-detail JSON ("[C]" inline chords),
    original key, year.
  - Simply Worship, simplyworship.hk (Cantonese / Mandarin): inline chords, key, time signature,
    tempo. Saved as a second chart in data/charts/alt/ when CantonHymn also has the song.
  - Ultimate Guitar public "Chords" pages, only for English candidates without a WorshipTogether
    chart (the most-voted public chord version of the song).
Politeness: one request at a time, >= 1.2 s apart (CantonHymn >= 2.5 s: it starts failing after
~45 fast calls), back-off 30 s then 120 s on errors, a host that keeps failing is paused for this
run; re-running retries only what is missing (pages already fetched come from the cache).

The five test songs are EXCLUDED so training never sees the test answers:
立志擺上, 活出愛, Here I Bow, 10,000 Reasons, 回家 (Eternity Girls). (回家 by 李健達 is a different
song and is kept.)

How bars are inferred ("# bars: inferred") when a chart has chords over lyrics but no bar lines:
  1. Syllables are counted from each line's first chord to its end (a Chinese character = 1,
     English words by vowel groups, a pause such as "，" or a wide space = 1, the held last note +
     breath = 3). Words before the first chord are a pickup and ignored; a sung line without chords
     makes the line before it longer (its last chord goes on).
  2. rate = the song's typical syllables per chord ("chords per line ~ bars": a typical chord lasts
     about a bar), blended with a tempo guess when the page gives a BPM (~80/BPM syllables per
     beat), kept between 4.5 (Chinese) / 3.5 (English) and 8 syllables per 4/4 bar. A line
     lasts syllables / rate bars, rounded to 1, 2, 4, 6 or 8 (at least one bar per two chords) -
     so every line starts on a bar line.
  3. Inside the line each chord goes to the bar line nearest to where its syllable falls, in order,
     one chord per position; only a line with more chords than bars also uses half-bar positions
     (bar lines slightly preferred).
  4. Chord-only lines (intros, interludes) use the chart's own rhythm when given: "C // G //"
     (WorshipTogether: one slash = one beat), "| F | C G |" bar lines; otherwise one bar per chord.
These durations are approximate; the chord ORDER is what the source says.

CLI (from the repo root, with the project venv):
  python -m train.harvest_charts fetch [--only ID,ID] [--refresh]  polite download + charts + index + stats
  python -m train.harvest_charts build                            rebuild charts from the cache (no network)
  python -m train.harvest_charts validate                         parse every chart, exit 1 on a failure
  python -m train.harvest_charts index                            rebuild index.csv + eval/reports/charts_stats.md
  python -m train.harvest_charts purge-cache                      delete the raw page cache (it holds lyrics)
The cache is deleted after each finished harvest (purge-cache); "build" then needs a new "fetch"
(about 190 polite requests, ~5 minutes).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "engine"))

from nathan_chordify.theory import chart as CH  # noqa: E402
from nathan_chordify.theory import harte as H  # noqa: E402
from nathan_chordify.theory import numbers as NUM  # noqa: E402

CANDIDATES = ROOT / "data" / "songs_candidates.yaml"
OUT_DIR = ROOT / "data" / "charts"
ALT_DIR = OUT_DIR / "alt"
INDEX = OUT_DIR / "index.csv"
LOG = OUT_DIR / "harvest_log.csv"
STATS = ROOT / "eval" / "reports" / "charts_stats.md"

# The five test songs (ids in data/songs_candidates.yaml): never harvested.
EXCLUDE_IDS = frozenset({"lizhi-baishang", "huo-chu-ai", "here-i-bow", "10000-reasons", "ca-回家"})
EXCLUDE_TITLES = ("立志擺上", "活出愛", "here i bow", "10,000 reasons", "10000 reasons")

from nathan_chordify.sheets.fetch import (  # noqa: E402,F401  (moved there 2026-11-03; re-exported for the tests)
    BACKOFF, DEFAULT_CACHE, HOST_GAP, MIN_GAP, UA, Fetcher, today)
from nathan_chordify.sheets.parsers import (  # noqa: E402,F401
    LINE_END, NICE_BARS, PAUSE, RATE_MAX, RATE_MIN, TEMPO_SYLLABLES, Block, Line, Raw, Section, Seg, Tok,
    _BAR_LINE, _CJK, _LABEL_RE, _LABELS, _REPEAT, _RHYTHM, _SYM, _Q_FALLBACK, _Q_VOCAB, _MINORISH,
    _bar_groups, _clean_token, _rhythm_seq, _slash_counts, _span_end, _split_bar, _strip_tags, _texts,
    _ug_store, bars_text, beats_for_meter, build_chart_text, ch_api, chart_body, check_chart_text,
    chord_events, chordpro_line, clean_header, eng_syllables, estimate_key, expand, instrumental_items,
    is_instrumental, key_name, line_bars, match_label, norm_chord, parse_cantonhymn, parse_key,
    parse_simplyworship, parse_ultimateguitar, parse_worshiptogether, place_block, read_header,
    section_blocks, sectionize, slash_meter, song_rate, syllables, to_bar_text, ug_pick)


# =============================================================================================
# Candidate songs and their charts
# =============================================================================================

@dataclass
class Song:
    id: str
    title: str
    artist: str
    language: str
    year: object = None
    key: str | None = None
    tempo: float | None = None
    meter: str | None = None
    note: str = ""
    wt: str | None = None
    ch: str | None = None
    sw: str | None = None
    ug_search: str | None = None


def load_candidates(path: Path = CANDIDATES) -> list[Song]:
    import yaml
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = []
    for s in d["songs"]:
        if s["id"] in EXCLUDE_IDS:
            continue
        song = Song(s["id"], s["title"], s.get("artist") or "", s.get("language") or "", s.get("year"),
                    s.get("key"), s.get("tempo_bpm"), s.get("meter"), s.get("note") or "")
        if song.title.strip().lower() in EXCLUDE_TITLES:
            continue
        for c in s.get("charts") or []:
            site, url = c.get("site") or "", c.get("url")
            if not url:
                continue
            if site == "WorshipTogether":
                song.wt = url
            elif site == "CantonHymn":
                song.ch = url
            elif site == "Simply Worship":
                song.sw = url
            elif site.startswith("Ultimate Guitar") and "search.php" in url:
                song.ug_search = url
        out.append(song)
    return out


def yaml_meter(song: Song) -> str | None:
    if song.meter and re.fullmatch(r"\d+/\d+", str(song.meter)):
        return str(song.meter)
    if song.note.strip() == "3/4":
        return "3/4"
    return None



def build_chart(song: Song, raw: Raw, source: str, url: str, retrieved: str) -> tuple[str | None, dict]:
    """Raw source text -> (chart file text or None, info); ``sheets.parsers.build_chart_text`` with
    the candidate list's meter as the fallback time signature."""
    meter_hint = yaml_meter(song)
    tempo = song.tempo if source == "WorshipTogether" else None
    text, info = build_chart_text(raw, source=source, url=url, retrieved=retrieved, title=song.title,
                                  artist=song.artist or raw.meta.get("artist"), language=song.language,
                                  meter_hint=meter_hint, tempo_hint=tempo,
                                  extra_header=[f"# year: {clean_header(song.year)}"] if song.year and not raw.meta.get("year") else None)
    if text is not None and meter_hint and info["meter"] == meter_hint and not raw.meta.get("meter"):
        text = text.replace(f"# meter: {meter_hint} (from the recording)", f"# meter: {meter_hint} (from data/songs_candidates.yaml)")
    return text, info


# =============================================================================================
# Fetch per song
# =============================================================================================

def harvest_song(song: Song, fetcher: Fetcher, log: list[dict]) -> list[tuple[str, str, str]]:
    """-> [(variant, file name, chart text)]; variant "primary" or "alt-<source>"."""
    results: list[tuple[str, str, str]] = []

    def attempt(source: str, url: str, fetch_url: str, parser) -> str | None:
        text, status = fetcher.get(fetch_url)
        row = {"slug": song.id, "source": source, "url": url, "status": status, "note": ""}
        log.append(row)
        if text is None:
            row["note"] = "not fetched"
            return None
        raw = parser(text)
        if raw is None:
            row["note"] = "no chart on the page"
            return None
        chart, info = build_chart(song, raw, source, url, fetcher.cached_date(fetch_url) or today())
        if chart is None:
            row["note"] = info.get("error", "failed")
            return None
        row["note"] = f"ok ({info['n_events']} chord changes, key {info['key']}, {info['meter']})"
        return chart

    if song.language == "english":
        chart = attempt("WorshipTogether", song.wt, song.wt, parse_worshiptogether) if song.wt else None
        if chart is None and song.ug_search:
            page, status = fetcher.get(song.ug_search)
            log.append({"slug": song.id, "source": "UltimateGuitar search", "url": song.ug_search,
                        "status": status, "note": ""})
            pick = ug_pick(page, song.title, song.artist) if page else None
            if pick:
                chart = attempt("UltimateGuitar", pick["tab_url"], pick["tab_url"], parse_ultimateguitar)
            elif page:
                log[-1]["note"] = "no public Chords version found"
        if chart:
            results.append(("primary", song.id, chart))
        return results
    ch_chart = attempt("CantonHymn", song.ch, ch_api(song.ch), parse_cantonhymn) if song.ch else None
    sw_chart = attempt("SimplyWorship", song.sw, song.sw, parse_simplyworship) if song.sw else None
    if ch_chart and sw_chart:
        # CantonHymn has no time signature: borrow Simply Worship's when it differs from the 4/4 guess
        sw_meter = re.search(r"^# meter: (\S+) \(from the page\)", sw_chart, re.M)
        if sw_meter and sw_meter.group(1) != "4/4":
            ch_chart = None
            log.append({"slug": song.id, "source": "CantonHymn", "url": song.ch, "status": "-",
                        "note": f"skipped: Simply Worship gives {sw_meter.group(1)}, CantonHymn gives no meter"})
    if ch_chart:
        results.append(("primary", song.id, ch_chart))
        if sw_chart:
            results.append(("alt", f"{song.id}.simplyworship", sw_chart))
    elif sw_chart:
        results.append(("primary", song.id, sw_chart))
    return results


def write_log(rows: list[dict]) -> None:
    with LOG.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "source", "url", "status", "note"])
        w.writeheader()
        w.writerows(rows)


def run_harvest(songs: list[Song], fetcher: Fetcher) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ALT_DIR.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    for n, song in enumerate(songs, 1):
        res = harvest_song(song, fetcher, log)
        for variant, name, text in res:
            folder = OUT_DIR if variant == "primary" else ALT_DIR
            (folder / f"{name}.chart.txt").write_text(text, encoding="utf-8")
        got = ", ".join(v for v, _, _ in res) or "NOTHING"
        print(f"[{n}/{len(songs)}] {song.id}: {got}  (requests so far {fetcher.requests})", flush=True)
    return log


# =============================================================================================
# Validation, index, stats
# =============================================================================================


# =============================================================================================
# Validation, index, stats
# =============================================================================================

def chart_files() -> list[tuple[str, Path]]:
    out = [("primary", p) for p in sorted(OUT_DIR.glob("*.chart.txt"))]
    out += [("alt", p) for p in sorted(ALT_DIR.glob("*.chart.txt"))]
    return out


def validate() -> int:
    bad = 0
    files = chart_files()
    for variant, p in files:
        probs = check_chart_text(p.read_text(encoding="utf-8"))
        if probs:
            bad += 1
            print(f"FAIL {p.relative_to(ROOT)}: {'; '.join(probs[:3])}")
    print(f"{len(files) - bad}/{len(files)} chart files parse cleanly")
    return 1 if bad else 0


def analyse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    hdr = read_header(text)
    probs = check_chart_text(text)
    row = {
        "slug": path.name[:-len(".chart.txt")], "title": hdr.get("title", ""), "artist": hdr.get("artist", ""),
        "language": hdr.get("language", ""), "key": hdr.get("key", "").split(" ")[0],
        "meter": hdr.get("meter", "").split(" ")[0], "tempo": hdr.get("tempo", "").replace(" BPM", ""),
        "source": hdr.get("source", "").split(" ")[0], "url": " ".join(hdr.get("source", "").split(" ")[1:]),
        "retrieved": hdr.get("retrieved", ""), "bars_inferred": "inferred" in hdr.get("bars", ""),
        "parse_ok": not probs, "problems": "; ".join(probs[:2]),
        "n_bars": 0, "n_chords": 0, "distinct_qualities": "", "n_distinct_chords": 0, "slash_share": 0.0,
        "_events": [], "_key": None,
    }
    if probs:
        return row
    m = int(hdr.get("beats_per_bar", "4"))
    chart = CH.parse(text, m)
    events = chord_events(chart)
    quals = Counter(H.parse_harte_full(h).quality for h, _ in events if h != "N")
    row.update(n_bars=sum(len(s.bars) for s in chart.sections), n_chords=len(events),
               n_distinct_chords=len({h for h, _ in events}),
               distinct_qualities=" ".join(q for q, _ in quals.most_common()),
               slash_share=round(sum(1 for h, _ in events if "/" in h) / max(1, len(events)), 3),
               _events=events, _key=parse_key(row["key"]))
    return row


INDEX_FIELDS = ["slug", "title", "artist", "language", "key", "meter", "n_bars", "n_chords",
                "distinct_qualities", "source", "parse_ok", "variant", "file", "tempo", "n_distinct_chords",
                "slash_share", "bars_inferred", "url", "retrieved", "problems"]


def build_index() -> list[dict]:
    rows = []
    for variant, p in chart_files():
        r = analyse(p)
        r["variant"] = variant
        r["file"] = str(p.relative_to(ROOT)).replace("\\", "/")
        rows.append(r)
    with INDEX.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def _nash(h: str, k: int, level: str) -> str:
    r, q, b = H.parse_harte(h)
    if q == "N":
        return "N"
    return NUM.nashville(r, q, b, k, level=level)


def write_stats(rows: list[dict], path: Path = STATS, log: list[dict] | None = None) -> str:
    prim = [r for r in rows if r["variant"] == "primary"]
    ok = [r for r in prim if r["parse_ok"]]
    alt = [r for r in rows if r["variant"] == "alt"]
    langs = Counter(r["language"] for r in prim)
    srcs = Counter(r["source"] for r in prim)
    ev_all = [(h, d, r["_key"] or 0) for r in ok for h, d in r["_events"] if h != "N"]
    n_ev = len(ev_all) or 1
    total_beats = sum(d for _, d, _ in ev_all) or 1.0
    q_count = Counter(H.parse_harte_full(h).quality for h, _, _ in ev_all)
    sym = Counter(H.harte_to_symbol(h) for h, _, _ in ev_all)
    nash_simple = Counter(_nash(h, k, "simple") for h, _, k in ev_all)
    nash_full = Counter(_nash(h, k, "full") for h, _, k in ev_all)
    slash = Counter(_nash(h, k, "full") for h, _, k in ev_all if "/" in h)
    bigr_s, bigr_f, four = Counter(), Counter(), Counter()
    for r in ok:
        k = r["_key"] or 0
        seq_s = [_nash(h, k, "simple") for h, _ in r["_events"] if h != "N"]
        seq_f = [_nash(h, k, "full") for h, _ in r["_events"] if h != "N"]
        dedup = [a for i, a in enumerate(seq_s) if i == 0 or a != seq_s[i - 1]]
        bigr_s.update(zip(dedup, dedup[1:]))
        bigr_f.update((a, b) for a, b in zip(seq_f, seq_f[1:]))
        four.update(tuple(dedup[i:i + 4]) for i in range(len(dedup) - 3))

    def share(pred) -> tuple[str, str]:
        ev = sum(1 for h, _, _ in ev_all if pred(h))
        songs = sum(1 for r in ok if any(pred(h) for h, _ in r["_events"] if h != "N"))
        return f"{100 * ev / n_ev:.1f}%", f"{songs}/{len(ok)}"

    qual = lambda h: H.parse_harte_full(h).quality  # noqa: E731
    groups = [
        ("sus (sus2, sus4, 7sus4, 9sus4, 13sus4)", lambda h: qual(h) in ("sus2", "sus4", "7sus4", "9sus4", "13sus4")),
        ("add9 / madd9 / 6/9", lambda h: qual(h) in ("add9", "madd9", "6/9")),
        ("maj7 / maj9", lambda h: qual(h) in ("maj7", "maj9", "maj7#11")),
        ("minor 7th family (m7, m9, m11)", lambda h: qual(h) in ("min7", "m7", "m9", "m11")),
        ("dominant 7th family (7, 9, 13, 7b9 ...)", lambda h: qual(h) in ("7", "9", "13", "7b9", "7#9", "7b13", "+7")),
        ("slash chords (any bass other than the root)", lambda h: "/" in h),
        ("plain major / minor triads", lambda h: qual(h) in ("maj", "min")),
    ]
    L: list[str] = []
    L.append("# Chord-chart harvest: statistics")
    L.append("")
    L.append(f"Generated {today()} by `train/harvest_charts.py index` from `data/charts/*.chart.txt` "
             "(primary chart per song; the second Simply Worship charts in `data/charts/alt/` are not counted below).")
    L.append("Chords only: no lyrics were stored. Test songs excluded (立志擺上, 活出愛, Here I Bow, 10,000 Reasons, 回家 Eternity Girls).")
    L.append("Bar lengths are **inferred** from the syllables under each chord (charts have no bar lines), so durations and")
    L.append("\"changes per bar\" are approximate; chord order, chord types and progressions are as the charts give them.")
    L.append("Numbers (1, 4, 5, 6m, 5/7 ...) are Nashville numbers / 級數 relative to each chart's key, so songs in different keys add up.")
    L.append("")
    L.append("## Coverage")
    L.append("")
    L.append(f"- Songs with a chart: **{len(prim)}** of 123 candidates (test songs excluded); parse OK: **{len(ok)}/{len(prim)}**"
             f" ({100 * len(ok) / max(1, len(prim)):.0f}%). Extra Simply Worship second charts: {len(alt)} "
             f"(parse OK {sum(1 for r in alt if r['parse_ok'])}).")
    L.append("- Per language: " + ", ".join(f"{k} {v}" for k, v in langs.most_common()))
    L.append("- Per source: " + ", ".join(f"{k} {v}" for k, v in srcs.most_common()))
    keys = Counter(r["key"] for r in ok)
    meters = Counter(r["meter"] for r in ok)
    L.append("- Keys: " + ", ".join(f"{k} {v}" for k, v in keys.most_common()))
    L.append("- Meters: " + ", ".join(f"{k} {v}" for k, v in meters.most_common()) +
             " (4/4 is assumed when a page gives no time signature)")
    nb = sum(r["n_bars"] for r in ok)
    L.append(f"- Total: {nb} bars, {n_ev} chord changes, {len(sym)} distinct chord symbols; "
             f"{n_ev / max(1, nb):.2f} changes per bar (inferred).")
    if log:
        net = [r for r in log if r["status"] not in ("-",)]
        bad = Counter(f"{r['source']} {r['status']}" for r in net if r["status"] not in ("200", "cache"))
        L.append(f"- Fetch log (`data/charts/harvest_log.csv`): {len(net)} page requests; "
                 + ("all answered HTTP 200, no source blocked us." if not bad else
                    "problems: " + ", ".join(f"{k} x{v}" for k, v in bad.items())))
        empty = Counter(r["source"] for r in log if "no chords" in r.get("note", ""))
        if empty:
            L.append("- Pages with no chords on them (another source was used): "
                     + ", ".join(f"{k} {v}" for k, v in empty.items()))
    L.append("")
    L.append("## How often each kind of chord appears (share of chord changes / songs that use it)")
    L.append("")
    L.append("| Kind | Share of chord changes | Songs using it |")
    L.append("|---|---:|---:|")
    for name, pred in groups:
        a, b = share(pred)
        L.append(f"| {name} | {a} | {b} |")
    L.append("")
    L.append("## How long chords last (inferred, so approximate)")
    L.append("")
    lens = [d for _, d, _ in ev_all]
    bins = [("half a bar or less", lambda d: d <= 2), ("up to one bar", lambda d: 2 < d <= 4),
            ("up to two bars", lambda d: 4 < d <= 8), ("longer", lambda d: d > 8)]
    L.append("| Length (beats; 4/4 bar = 4) | Share of chord changes |")
    L.append("|---|---:|")
    for name, pred in bins:
        L.append(f"| {name} | {100 * sum(1 for d in lens if pred(d)) / max(1, len(lens)):.1f}% |")
    L.append("")
    L.append("## Chord qualities (Harte names)")
    L.append("")
    L.append("| Quality | Changes | Share | Share of time |")
    L.append("|---|---:|---:|---:|")
    q_time = Counter()
    for h, d, _ in ev_all:
        q_time[qual(h)] += d
    for q, c in q_count.most_common():
        L.append(f"| {q} | {c} | {100 * c / n_ev:.1f}% | {100 * q_time[q] / total_beats:.1f}% |")
    L.append("")
    L.append("## Most common chords as numbers (full detail)")
    L.append("")
    L.append(" ".join(f"`{k}` {v}," for k, v in nash_full.most_common(30)).rstrip(","))
    L.append("")
    L.append("Simplified to triads (simple level): " + " ".join(f"`{k}` {v}," for k, v in nash_simple.most_common(15)).rstrip(","))
    L.append("")
    L.append("## Most common slash chords (as numbers)")
    L.append("")
    L.append(" ".join(f"`{k}` {v}," for k, v in slash.most_common(20)).rstrip(",") or "(none)")
    L.append("")
    L.append("## Most common chord symbols (absolute, as written)")
    L.append("")
    L.append(" ".join(f"`{k}` {v}," for k, v in sym.most_common(30)).rstrip(","))
    L.append("")
    L.append("## Progressions: most common chord-to-chord moves (bigrams)")
    L.append("")
    L.append("Simple level (triads, repeats merged):")
    L.append("")
    L.append("| Move | Count |")
    L.append("|---|---:|")
    for (a, b), c in bigr_s.most_common(25):
        L.append(f"| {a} → {b} | {c} |")
    L.append("")
    L.append("Full detail (as written, with sus / 7ths / slash basses):")
    L.append("")
    L.append("| Move | Count |")
    L.append("|---|---:|")
    for (a, b), c in bigr_f.most_common(25):
        L.append(f"| {a} → {b} | {c} |")
    L.append("")
    L.append("## Four-chord patterns (simple level)")
    L.append("")
    L.append("| Pattern | Count |")
    L.append("|---|---:|")
    for pat, c in four.most_common(15):
        L.append(f"| {' – '.join(pat)} | {c} |")
    L.append("")
    L.append("## Per language")
    L.append("")
    L.append("| Language | Songs | Changes | sus | add9 | maj7 | slash |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for lang, _ in langs.most_common():
        evs = [h for r in ok if r["language"] == lang for h, _ in r["_events"] if h != "N"]
        n = len(evs) or 1
        f = lambda pred: f"{100 * sum(1 for h in evs if pred(h)) / n:.1f}%"  # noqa: E731
        L.append(f"| {lang} | {sum(1 for r in ok if r['language'] == lang)} | {len(evs)} | {f(groups[0][1])} | "
                 f"{f(groups[1][1])} | {f(groups[2][1])} | {f(groups[5][1])} |")
    L.append("")
    L.append("## Suggested training weights (from these counts)")
    L.append("")
    L.append("- Sample chord *qualities* roughly in proportion to the quality table above (plain triads dominate; "
             "sus, add9, 7ths and slash chords are the next tier), but keep a floor for rare ones so the model still sees them.")
    L.append("- Sample *progressions* from these charts (transposed to all 12 keys) rather than random chord pairs: "
             "the bigram tables show the moves real worship songs use (1–5–6m–4 family, 4→5, 5/7 and 1/3 passing basses).")
    L.append("- Slash chords are common enough that the bass note matters: render the written bass in the left hand.")
    L.append("- \"2\" chords (C2, G2) are read as sus2, except a 2 chord over its own third (D2/F#), read as add9.")
    L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(L) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def read_log() -> list[dict] | None:
    if not LOG.exists():
        return None
    with LOG.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# =============================================================================================
# CLI
# =============================================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["fetch", "build", "validate", "index", "purge-cache"])
    ap.add_argument("--only", default="", help="comma-separated candidate ids")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and download again")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="raw page cache (outside the repo)")
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        return validate()
    if a.cmd == "purge-cache":
        import shutil
        cache = Path(a.cache)
        if cache.exists() and cache.resolve() != ROOT.resolve() and ROOT.resolve() not in cache.resolve().parents:
            shutil.rmtree(cache)
            print(f"deleted {cache}")
        else:
            print(f"nothing to delete at {cache}")
        return 0
    if a.cmd == "index":
        rows = build_index()
        write_stats(rows, log=read_log())
        print(f"index: {len(rows)} chart files -> {INDEX.relative_to(ROOT)}; stats -> {STATS.relative_to(ROOT)}")
        return 0
    songs = load_candidates()
    if a.only:
        wanted = set(a.only.split(","))
        songs = [s for s in songs if s.id in wanted]
    fetcher = Fetcher(Path(a.cache), offline=(a.cmd == "build"), refresh=a.refresh)
    if not a.only:      # a full run rewrites every chart: clear old ones first
        for _, p in chart_files():
            p.unlink()
    log = run_harvest(songs, fetcher)
    if a.cmd == "fetch" and not a.only:     # "build" keeps the network log of the last fetch
        write_log(log)
    rows = build_index()
    write_stats(rows, log=read_log())
    print(f"done: {fetcher.requests} network requests; {sum(1 for r in rows if r['variant'] == 'primary')} songs "
          f"with a chart; log -> {LOG.relative_to(ROOT)}")
    return validate()


if __name__ == "__main__":
    sys.exit(main())
