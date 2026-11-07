"""The whole flow for one song: meta → search terms → candidates → charts → alignment →
``SheetsState`` (cache/<key>/sheets/state.json) with ranked ``SheetResult`` entries.

Rules (user, 2026-11-03): sheets are a REFERENCE, not the truth; one source per language first,
another only when nothing "good" was found; same-title songs are all fetched and the alignment
decides; identical chord sequences are one result (``also`` lists the other URLs); a page whose
YouTube ids include the recording's id ranks first (``video_match``). Chord-only chart texts go to
``cache/<key>/sheets/<id>.chart.txt``; chords-over-lyrics (user's own songs only) to
``cache/<key>/sheets/<id>.lyrics.txt`` — never under data/ or in logs.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from .. import schema
from ..theory import harte as H
from ..theory import simplify as SM
from ..theory import spelling as S
from . import align as A
from . import parsers as P
from . import query as Q
from . import sources as SRC
from .fetch import Fetcher

log = logging.getLogger("nathan_chordify.sheets")

SHEETS_DIR = "sheets"
STATE_FILE = "state.json"
VERDICT_RANK = {"good": 0, "maybe": 1, "different": 2}
MAX_RESULTS = 12

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sheets_dir(song_dir: Path) -> Path:
    return Path(song_dir) / SHEETS_DIR


def state_lock(song_dir: Path) -> threading.Lock:
    k = str(Path(song_dir).resolve()).lower()
    with _locks_guard:
        if k not in _locks:
            _locks[k] = threading.Lock()
        return _locks[k]


def empty_state() -> dict:
    return {"status": "idle", "error": None, "query": {"title": "", "artist": None, "terms": []},
            "results": [], "applied": None, "ts": None}


def load_state(song_dir: Path) -> dict:
    p = sheets_dir(song_dir) / STATE_FILE
    st = empty_state()
    if p.exists():
        try:
            st.update(json.loads(p.read_text("utf-8")))
        except (OSError, ValueError) as e:
            log.warning("%s unreadable: %s", p, e)
    return st


def save_state(song_dir: Path, st: dict) -> dict:
    d = sheets_dir(song_dir)
    d.mkdir(parents=True, exist_ok=True)
    st["ts"] = now_iso()
    tmp = d / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(d / STATE_FILE)
    return st


def read_meta(song_dir: Path) -> dict:
    p = Path(song_dir) / "meta.json"
    try:
        return json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def load_analysis(song_dir: Path) -> schema.Analysis | None:
    p = Path(song_dir) / "analysis.json"
    if not p.exists():
        return None
    return schema.Analysis.model_validate_json(p.read_text("utf-8"))


def song_query(song_dir: Path, analysis: schema.Analysis | None = None, override: str | None = None) -> Q.Query:
    """The search terms of a song (``override``: the user's own search text)."""
    if override and override.strip():
        q = Q.parse_title(override.strip(), None, "upload")
        q.terms = [override.strip()] + [t for t in q.terms if t != override.strip()]
        return q
    meta = read_meta(song_dir)
    if not meta.get("title") and analysis is not None:
        meta = {**meta, "title": analysis.source.title, "artist": analysis.source.artist,
                "source_type": analysis.source.type}
    if not meta.get("title"):
        meta = {**meta, "title": Path(song_dir).name}
    return Q.from_meta(meta)


def video_id_of(song_dir: Path, meta: dict | None = None) -> str | None:
    key = Path(song_dir).name
    if key.startswith("yt-"):
        return key[3:]
    meta = meta or read_meta(song_dir)
    return meta.get("id") if (meta.get("source_type") == "youtube") else None


# ------------------------------------------------------------------------------------- results

def _aligned_entries(chords: list[tuple[float, float, str]], audio: A.Audio) -> list[dict]:
    """[{t0, t1, harte, labels}] spelled exactly like display.variants["0:auto"] (same speller,
    same levels, next chord's root for secondary dominants)."""
    out = []
    trip = [H.parse_harte(h, strict=False) for _, _, h in chords]
    for i, (t0, t1, h) in enumerate(chords):
        r, q, b = trip[i]
        nxt = next((trip[j][0] for j in range(i + 1, len(chords)) if trip[j][0] >= 0), None)
        key = audio.key_at(t0)
        if r < 0 or q == "N":
            labels = ["N", "N", "N"]
        else:
            labels = [SM.label(r, q, b, key, level=lv, next_root=nxt) for lv in SM.LEVELS]
        out.append({"t0": t0, "t1": t1, "harte": h, "labels": labels})
    return out


def make_result(cand: SRC.Candidate | None, doc: SRC.ChartDoc, audio: A.Audio, *, source: str | None = None,
                url: str | None = None, rid: str | None = None, title: str | None = None,
                query_title: str | None = None, written_key: int | None = None) -> dict:
    """One ``SheetResult`` from a chart document aligned to the recording."""
    hdr = P.read_header(doc.text)
    bpb = int(hdr.get("beats_per_bar") or doc.meta.get("beats_per_bar") or 4)
    key_written = doc.meta.get("key_written")
    wk = written_key if written_key is not None else doc.meta.get("key_index")
    try:
        al = A.align_chart_text(doc.body, audio, bpb, wk)
    except ValueError as e:
        raise ValueError(f"chart not usable: {e}") from e
    src = source or (cand.source if cand else "pasted")
    u = url if url is not None else (cand.url if cand else None)
    t_title = title or doc.meta.get("title") or (cand.title if cand else "") or ""
    vids = list(doc.meta.get("youtube_ids") or [])
    video_match = bool(audio.key.startswith("yt-") and audio.key[3:] in vids)
    chart_text = al.chart_text(doc.body)
    lyrics = doc.lyrics
    if lyrics:
        try:
            lyrics = P.transpose_chordpro(lyrics, al.transpose, audio.key_index)
        except Exception:  # noqa: BLE001 - lyrics are a bonus; never fail the result on them
            pass
    warnings = list(al.warnings)
    if doc.meta.get("capo"):
        warnings.append(f"the page is a guitar chart with capo {doc.meta['capo']}; the chords were moved to the recording's key")
    match_title = SRC.title_similarity(query_title or t_title, t_title) if query_title else (cand.title_sim if cand else 1.0)
    kw = key_written
    if kw is not None and doc.meta.get("key_index") is not None:
        kw = S.key_name(int(doc.meta["key_index"]))
    return {
        "id": rid or (cand.id if cand else SRC.result_id("pasted", doc.body)),
        "source": src, "url": u, "title": t_title, "artist": doc.meta.get("artist") or (cand.artist if cand else None),
        "year": doc.meta.get("year"), "key_written": kw, "key_recording": S.key_name(audio.key_index),
        "transpose": int(al.transpose), "meter": doc.meta.get("meter") or hdr.get("meter", "").split(" ")[0] or None,
        "n_chords": int(doc.meta.get("n_events") or len(al.chords)),
        "match": {"title": round(float(match_title), 3), "align": al.fit, "score": al.score,
                  "coverage": al.coverage, "beat_scale": al.beat_scale},
        "verdict": al.verdict, "video_match": video_match, "also": [],
        "chart_text": chart_text, "aligned": _aligned_entries(al.chords, audio),
        "sections": [{"t": t, "label": lab} for t, lab in al.sections],
        "warnings": warnings, "lyrics_text": lyrics, "ts": now_iso(),
    }


def rank_key(r: dict) -> tuple:
    return (0 if r.get("video_match") else 1, VERDICT_RANK.get(r.get("verdict"), 3), -float(r["match"]["score"]),
            -float(r["match"]["title"]))


def dedupe(results: list[dict]) -> list[dict]:
    """Results with the same chord sequence (in the recording's key) are one result; the others'
    URLs go to ``also``. The better-ranked one is kept."""
    out: list[dict] = []
    sigs: dict[tuple, dict] = {}
    for r in sorted(results, key=rank_key):
        sig = SRC.chord_signature(r["chart_text"], int(P.read_header(r["chart_text"]).get("beats_per_bar") or 4))
        if sig in sigs and sig:
            keep = sigs[sig]
            if r.get("url") and r["url"] not in keep["also"] and r["url"] != keep.get("url"):
                keep["also"].append(r["url"])
            continue
        sigs[sig] = r
        out.append(r)
    return out


def write_result_files(song_dir: Path, r: dict, doc: SRC.ChartDoc | None = None) -> None:
    d = sheets_dir(song_dir)
    d.mkdir(parents=True, exist_ok=True)
    safe = r["id"].replace(":", "-")
    (d / f"{safe}.chart.txt").write_text((doc.text if doc else r["chart_text"]), "utf-8")
    if r.get("lyrics_text"):
        (d / f"{safe}.lyrics.txt").write_text(r["lyrics_text"], "utf-8")


# -------------------------------------------------------------------------------------- search

def run_search(song_dir: Path, fetcher: Fetcher, *, query: str | None = None, analysis: schema.Analysis | None = None,
               audio: A.Audio | None = None, sources: tuple[str, ...] | None = None,
               on_state=None, max_per_source: int = SRC.MAX_PER_SOURCE) -> dict:
    """Search → fetch → align for one song; writes and returns the state. ``on_state(st)`` is
    called with each intermediate state (results so far) for pollers."""
    song_dir = Path(song_dir)
    analysis = analysis or load_analysis(song_dir)
    if analysis is None:
        raise FileNotFoundError(f"no analysis in {song_dir}")
    q = song_query(song_dir, analysis, query)
    st = load_state(song_dir)
    st.update({"status": "searching", "error": None, "query": q.to_dict(), "results": []})
    save_state(song_dir, st)
    if on_state:
        on_state(st)
    audio = audio or A.load_audio(song_dir, analysis)
    vid = video_id_of(song_dir)
    results: list[dict] = []
    docs: dict[str, SRC.ChartDoc] = {}
    offline: list[str] = []
    seen_urls: set[str] = set()
    notes: list[str] = []
    order = SRC.source_order(q.language, sources, q.terms)
    for source in order:
        if results and any(r["verdict"] == "good" or r["video_match"] for r in results):
            break                                          # one source per language is enough when it found the song
        try:
            cands = SRC.search_source(source, q.terms, q.artist, fetcher, vid)
        except SRC.SourceOffline:
            offline.append(source)
            continue
        queue = [c for c in cands if c.has_chart or c.video_match][:max_per_source]
        fetched = 0
        while queue and fetched < max_per_source:
            cand = queue.pop(0)
            if cand.url in seen_urls:
                continue
            seen_urls.add(cand.url)
            try:
                doc = SRC.fetch_chart(cand, fetcher)
            except SRC.SourceOffline:
                offline.append(source)
                break
            fetched += 1
            if doc is None:
                notes.append(f"{cand.url}: no readable chords")
                continue
            for rel in doc.related:
                if rel.url not in seen_urls and rel.url not in {c.url for c in queue}:
                    queue.append(rel)
            try:
                r = make_result(cand, doc, audio, query_title=q.title)
            except ValueError as e:
                notes.append(f"{cand.url}: {e}")
                continue
            if cand.video_match or (vid and vid in (doc.meta.get("youtube_ids") or [])):
                r["video_match"] = True
            results.append(r)
            docs[r["id"]] = doc
            st["results"] = sorted(results, key=rank_key)
            save_state(song_dir, st)
            if on_state:
                on_state(st)
    results = dedupe(results)[:MAX_RESULTS]
    for r in results:
        write_result_files(song_dir, r, docs.get(r["id"]))
    st = load_state(song_dir)                              # "applied" may have changed meanwhile
    st["query"] = q.to_dict()
    st["results"] = results
    st["notes"] = notes[:20]
    if not results and offline and len(set(offline)) >= len(set(order)):
        st["status"] = "error"
        st["error"] = {"code": "sheets_unreachable", "msg": "The chord-sheet websites could not be reached.",
                       "hint": "Check the internet connection and try again."}
    else:
        st["status"] = "done"
        st["error"] = None
        if offline:
            st["notes"] = [f"could not reach: {', '.join(sorted(set(offline)))}"] + st["notes"]
    save_state(song_dir, st)
    if on_state:
        on_state(st)
    return st


def add_pasted(song_dir: Path, text: str, *, analysis: schema.Analysis | None = None,
               audio: A.Audio | None = None, title: str | None = None) -> dict:
    """A chart the user pasted (bar text, or ChordPro-style "[C]word" lines) → a SheetResult added
    to the state (the old pasted result with the same text is replaced)."""
    song_dir = Path(song_dir)
    analysis = analysis or load_analysis(song_dir)
    if analysis is None:
        raise FileNotFoundError(f"no analysis in {song_dir}")
    body = text.strip()
    if not body:
        raise ValueError("Paste a chart first.")
    lyrics = None
    if "|" not in body or "[" in body and "]" in body and len(P._INLINE.findall(body)) >= 4 and body.count("|") < 4:
        raw = P.Raw([P.chordpro_line(ln) for ln in body.splitlines()], {})
        chart, info = P.build_chart_text(raw, source="pasted", url="", retrieved=P.today() if hasattr(P, "today") else "",
                                         title=title or "pasted chart", min_events=2)
        if chart is None:
            raise ValueError("No chords could be read from the pasted text.")
        lyrics = P.raw_to_chordpro(raw)
        doc = SRC.ChartDoc(chart, P.chart_body(chart), lyrics, {"key_written": None, "key_index": info["key_index"],
                                                                "meter": info["meter"], "tempo": None, "year": None,
                                                                "artist": None, "title": title or "pasted chart",
                                                                "youtube_ids": [], "capo": None, "n_events": info["n_events"],
                                                                "beats_per_bar": info["beats_per_bar"]})
    else:
        hdr = P.read_header(body)
        bpb = int(hdr.get("beats_per_bar") or 4)
        from ..theory import chart as CH
        try:
            chart = CH.parse(body, bpb)
        except CH.ChartError as e:
            raise ValueError(f"The chart could not be read: {e}") from e
        events = P.chord_events(chart)
        if len(events) < 2:
            raise ValueError("The pasted chart has fewer than two chords.")
        k, _ = P.estimate_key(events, P.parse_key(hdr.get("key")))
        text_full = ("" if hdr else f"# title: {title or 'pasted chart'}\n# key: {P.key_name(k)} (estimated)\n# meter: 4/4\n"
                     f"# beats_per_bar: {bpb}\n# source: pasted\n# retrieved: {P.today() if hasattr(P, 'today') else ''}\n") + body + "\n"
        doc = SRC.ChartDoc(text_full, P.chart_body(body), None, {"key_written": hdr.get("key"), "key_index": k,
                                                                 "meter": hdr.get("meter", "").split(" ")[0] or None,
                                                                 "tempo": None, "year": None, "artist": None,
                                                                 "title": title or hdr.get("title") or "pasted chart",
                                                                 "youtube_ids": [], "capo": None, "n_events": len(events),
                                                                 "beats_per_bar": bpb})
    audio = audio or A.load_audio(song_dir, analysis)
    rid = SRC.result_id("pasted", body)
    r = make_result(None, doc, audio, source="pasted", url=None, rid=rid, title=doc.meta["title"])
    return _add_result(song_dir, r, doc)


def add_url(song_dir: Path, url: str, fetcher: Fetcher, *, analysis: schema.Analysis | None = None,
            audio: A.Audio | None = None) -> dict:
    """A sheet LINK from a supported site → fetched, parsed, aligned, added."""
    song_dir = Path(song_dir)
    analysis = analysis or load_analysis(song_dir)
    if analysis is None:
        raise FileNotFoundError(f"no analysis in {song_dir}")
    cand = SRC.candidate_from_url(url)
    if cand is None:
        raise LookupError("That link is not from a chord-sheet site this app can read (CantonHymn, Simply Worship, "
                          "WorshipTogether, Ultimate Guitar).")
    doc = SRC.fetch_chart(cand, fetcher)
    if doc is None:
        raise LookupError("No chords could be read from that page.")
    audio = audio or A.load_audio(song_dir, analysis)
    r = make_result(cand, doc, audio)
    vid = video_id_of(song_dir)
    if vid and vid in (doc.meta.get("youtube_ids") or []):
        r["video_match"] = True
    return _add_result(song_dir, r, doc)


def _add_result(song_dir: Path, r: dict, doc: SRC.ChartDoc) -> dict:
    with state_lock(song_dir):
        st = load_state(song_dir)
        st["results"] = [x for x in st.get("results") or [] if x.get("id") != r["id"]] + [r]
        st["results"] = sorted(dedupe(st["results"]), key=rank_key)[:MAX_RESULTS]
        if st.get("status") != "searching":
            st["status"] = "done"
            st["error"] = None
        write_result_files(song_dir, r, doc)
        save_state(song_dir, st)
    return st
