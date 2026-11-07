"""Ground-truth mode (DESIGN §8.4, §11, §13.1; review E1/E3/W2).

Per song, ``cache/<key>/gt.json`` keeps Nathan's annotation state:

- ``beats``: his **tapped** beats (Space along with playback at 70%), independent of our tracker
  (review E1: GT aligned to our own grid would be circular);
- ``chart``: the bar-text chart he pasted (DESIGN §0.1 format, ``[Section @time]`` anchors and
  ``(H)/(M)/(L)`` line notes allowed), how it was laid on the beats, and its unsure spots;
- ``reviewed``: ticked once he has checked every bar by ear.

Importing a chart parses it (``theory.chart.parse_annotated``), lays it on the tapped beats (or the
detected grid until he taps; ``theory.chart.align_chart``) and appends one ``set_chord`` range op
per chord plus ``set_section`` ops, all in one op *group*, to the song's edit log: the GT draft is
an ordinary edit layer, so the chord editor corrects it, undo works, and desktop and phone merge.
Re-importing reverts the previous import's group. Chords from ``(L)`` lines or written with ``?``
get conf 0.35 (drawn dashed), ``(M)`` 0.75, everything else 1.0.

The GT chord sequence is the edited analysis' main layer plus the passing chords Nathan made or
accepted (our unedited passing guesses are left out, so they can't anchor the annotation; review
E3). Export writes ``eval/gt/<key>.lab``, ``<key>.beats`` (time, beat-in-bar) and
``<key>.chart.txt`` (bar text with section anchors and line notes, readable by
``eval/checklist.py``) and adds / updates the song in ``eval/songs.yaml`` (``reviewed``).

Routes:

    GET    /api/tracks/{key}/gt                      the GT state (+ the draft chart's availability)
    PUT    /api/tracks/{key}/gt/beats   {beats, t0?, t1?}   store tapped beats (t0/t1: replace a stretch)
    POST   /api/tracks/{key}/gt/beats   (same)
    DELETE /api/tracks/{key}/gt/beats
    POST   /api/tracks/{key}/gt/chart   {text, start?, beats?: auto|tapped|detected, import_id?, replace?}
    POST   /api/tracks/{key}/gt/import  {format: chart|lab, text, …}    (DESIGN §11 name)
    POST   /api/tracks/{key}/gt/review  {reviewed}
    POST   /api/tracks/{key}/gt/export  {reviewed?}  → {files, songs_yaml, warnings}
    GET    /api/tracks/{key}/gt/export?fmt=lab|beats|chart&reviewed=   writes the files, returns one
    GET    /api/tracks/{key}/gt/draft-chart          eval/gt/<key>.chart.txt (to start from the draft)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import threading
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .. import schema
from ..export import flatten
from ..theory import chart as CH
from ..theory import harte as H
from . import edits as ED
from .common import ApiError, ctx, now_iso

log = logging.getLogger("nathan_chordify.server.gt")

GT_FILE = "gt.json"
EXPORT_MARK = "# nathan_chordify ground-truth export"
CONF_OF_FLAG = {"L": 0.35, "M": 0.75, "H": 1.0}
MIN_BEAT_GAP = 0.08                   # taps closer than this are one tap
MAX_CHART_CHARS = 100_000

_yaml_lock = threading.Lock()


# ---- state --------------------------------------------------------------------------------------

def _default() -> dict:
    return {"beats": None, "beats_ts": None, "chart": None, "reviewed": False, "reviewed_ts": None,
            "exported": None, "exported_ts": None}


def load_state(d: Path) -> dict:
    p = d / GT_FILE
    st = _default()
    if p.exists():
        try:
            st.update(json.loads(p.read_text("utf-8")))
        except (OSError, ValueError) as e:
            log.warning("%s unreadable: %s", p, e)
    return st


def save_state(d: Path, st: dict) -> None:
    tmp = d / (GT_FILE + ".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(d / GT_FILE)


def _beats_sig(beats) -> str:
    return hashlib.sha1(np.round(np.asarray(beats, dtype=float), 3).tobytes()).hexdigest()[:12]


def status_of(st: dict) -> str:
    if st.get("reviewed"):
        return "reviewed"
    return "draft" if (st.get("chart") or st.get("beats") or st.get("exported")) else "none"


def gt_info(st: dict) -> dict:
    ch = st.get("chart") or {}
    return {"status": status_of(st), "tapped_beats": bool(st.get("beats")), "chart": ch.get("text")}


def decorate(c, key: str, analysis: dict) -> dict:
    """The edited analysis as served: library renames and the GT status (a copy; the cached
    analysis is never mutated)."""
    out = dict(analysis)
    row = c.library.get(key)
    if row and row.get("renamed"):
        out["source"] = {**analysis["source"], "title": row["title"], "artist": row["artist"]}
    st = load_state(c.cache_dir / key)
    out["gt"] = gt_info(st)
    return out


def public_state(c, key: str, st: dict, a: schema.Analysis | None = None) -> dict:
    """The GT state for the client (beats, chart info, unsure spots, files)."""
    ch = st.get("chart")
    eval_gt = Path(c.eval_dir) / "gt"
    out = {"status": status_of(st), "reviewed": bool(st.get("reviewed")), "reviewed_ts": st.get("reviewed_ts"),
           "beats": st.get("beats") or [], "beats_ts": st.get("beats_ts"), "chart": None,
           "exported": st.get("exported"), "exported_ts": st.get("exported_ts"),
           "draft_chart": (eval_gt / f"{key}.chart.txt").exists()}
    if ch:
        stale = bool(ch.get("source") == "tapped" and st.get("beats") and
                     ch.get("beats_sig") != _beats_sig(st["beats"]))
        out["chart"] = {k: ch.get(k) for k in ("text", "import_id", "group", "source", "start", "start_beat",
                                               "n_chords", "unsure", "warnings", "ts", "format")}
        out["chart"]["stale"] = stale
    return out


# ---- beats --------------------------------------------------------------------------------------

def clean_beats(beats: list, duration: float) -> list[float]:
    out = []
    for b in beats:
        try:
            x = float(b)
        except (TypeError, ValueError):
            raise ApiError(400, "bad_beats", "Beats must be numbers (seconds).") from None
        if not math.isfinite(x) or x < 0 or x > duration + 5:
            raise ApiError(400, "bad_beats", f"A beat at {x} s is outside the song.")
        out.append(x)
    out.sort()
    merged: list[float] = []
    for x in out:
        if merged and x - merged[-1] < MIN_BEAT_GAP:
            continue
        merged.append(round(x, 4))
    return merged


def beat_list(st: dict, a: schema.Analysis, source: str = "auto") -> tuple[np.ndarray, str]:
    """The beats to lay a chart on: Nathan's tapped beats, else the detected metered beats."""
    tapped = st.get("beats") or []
    if source in ("auto", "tapped") and len(tapped) >= 2:
        return np.asarray(tapped, dtype=float), "tapped"
    if source == "tapped":
        raise ApiError(409, "no_tapped_beats", "No tapped beats yet.", "Tap the beats first (GT mode → Tap beats).")
    det = sorted(b.t for b in a.beats if b.bar >= 0)
    if len(det) < 2:
        raise ApiError(409, "no_beats", "This song has no beat grid.", "Tap the beats first.")
    return np.asarray(det, dtype=float), "detected"


def default_start(a: schema.Analysis, beats: np.ndarray, source: str) -> int:
    if source == "tapped":
        return 0
    first = next((b.t for b in sorted(a.beats, key=lambda b: b.t) if b.bar >= 0 and b.beat == 1), None)
    if first is None:
        return 0
    return int(np.argmin(np.abs(beats - first)))


# ---- chart / lab import -------------------------------------------------------------------------

def parse_lab(text: str) -> list[tuple[float, float, str]]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            raise ApiError(400, "bad_lab", f"line {n}: expected 'start end chord'")
        try:
            t0, t1 = float(parts[0]), float(parts[1])
            harte = H.symbol_to_harte(" ".join(parts[2:]))
        except ValueError as e:
            raise ApiError(400, "bad_lab", f"line {n}: {e}") from None
        if t1 > t0 >= 0:
            out.append((t0, t1, harte))
    if not out:
        raise ApiError(400, "bad_lab", "No chords in this .lab text.")
    return out


def import_chart(c, key: str, text: str, fmt: str = "chart", start: float | None = None,
                 source: str = "auto", import_id: str | None = None, device: str | None = None,
                 replace: bool = True, beats_per_bar: int = 4) -> dict:
    d = ED.song_dir(c, key)
    if not text or not text.strip():
        raise ApiError(400, "empty_chart", "Paste a chart first.")
    if len(text) > MAX_CHART_CHARS:
        raise ApiError(413, "chart_too_long", "That chart is too long.")
    if import_id is not None and not ED.OP_ID_RE.match(import_id):
        raise ApiError(400, "bad_import_id", "Bad import id.")
    import_id = import_id or uuid.uuid4().hex[:12]
    group = f"gt-{import_id}"
    e = ED.edited(d)
    a = e.model
    st = load_state(d)
    warnings: list[str] = []
    sections: list[tuple[float, str]] = []
    if fmt == "lab":
        chords = [(t0, t1, h, "H", False) for t0, t1, h in parse_lab(text)]
        src, start_beat, beats_used = "lab", None, None
    elif fmt == "chart":
        try:
            ac = CH.parse_annotated(text, beats_per_bar)
        except CH.ChartError as err:
            raise ApiError(400, "bad_chart", str(err), "Format: | B . . B+7 | Emaj7 . A7 Em6 | (see the help).") from None
        if not ac.chart.bars:
            raise ApiError(400, "bad_chart", "No bars in this chart.", "Bars go between | marks.")
        beats, src = beat_list(st, a, source)
        if start is not None:
            start_beat = int(np.argmin(np.abs(beats - float(start))))
        else:
            start_beat = default_start(a, beats, src)
        lay = CH.layout(ac.chart, beats, start_beat, ac.anchors)
        warnings += lay.warnings
        al = CH.align_chart(ac.chart, beats, lay=lay, bar_conf=ac.bar_conf)
        chords = [(x.t0, x.t1, x.harte, x.conf, x.unsure) for x in al]
        seen_sec = set()
        for g, si in enumerate(lay.section_of_bar):
            label = ac.chart.sections[si].label
            if label and si not in seen_sec:
                seen_sec.add(si)
                sections.append((float(CH.time_at(beats, lay.bar_start[g])), label))
        beats_used = beats
    else:
        raise ApiError(400, "bad_format", "format must be 'chart' or 'lab'.")
    dur = a.source.duration
    chords = [(max(0.0, t0), min(t1, dur), h, f, u) for t0, t1, h, f, u in chords if t0 < dur - 0.01]
    if not chords:
        raise ApiError(400, "chart_outside", "The chart doesn't overlap the song.", "Check where it starts.")
    ops: list[dict] = []
    earlier = [g for g in st.get("groups") or [] if g != group]
    if replace:
        # revert every earlier import (not just the last: reverting the last one's group would bring
        # back the one it replaced). The reverts belong to the new group, so undoing it restores them.
        for n, g in enumerate(earlier):
            ops.append({"id": f"{import_id}.rv{n}", "kind": "revert", "target": {"group": g}, "value": {"group": group}})
    for n, (t0, t1, h, flag, unsure) in enumerate(chords):
        ops.append({"id": f"{import_id}.{n:04d}", "kind": "set_chord",
                    "target": {"t0": round(t0, 4), "t1": round(t1, 4), "mode": "range", "layer": "main"},
                    "value": {"harte": h, "group": group, "conf": CONF_OF_FLAG.get(flag, 1.0)}})
    for n, (t, label) in enumerate(sections):
        ops.append({"id": f"{import_id}.s{n:03d}", "kind": "set_section", "target": {"t0": round(t, 4)},
                    "value": {"label": label, "group": group}})
    appended, dups, e2 = ED.post_ops(d, ops, device or "gt-import")
    unsure = [{"t0": round(t0, 3), "t1": round(t1, 3), "harte": h, "flag": flag}
              for t0, t1, h, flag, u in chords if flag == "L" or u]
    st = load_state(d)
    st["chart"] = {"text": text, "format": fmt, "import_id": import_id, "group": group, "source": src,
                   "start": None if beats_used is None or start_beat is None else round(float(beats_used[start_beat]), 4),
                   "start_beat": start_beat, "beats_per_bar": beats_per_bar, "n_chords": len(chords),
                   "unsure": unsure, "warnings": warnings, "ts": now_iso(),
                   "beats_sig": None if beats_used is None else _beats_sig(beats_used)}
    st["groups"] = earlier + [group]
    st["reviewed"] = False
    save_state(d, st)
    return {"gt": public_state(c, key, st), "analysis": decorate(c, key, e2.analysis), "display": e2.display,
            "rev": e2.model.edits.rev, "n_chords": len(chords), "unsure": unsure, "warnings": warnings,
            "appended": len(appended), "duplicates": len(dups)}


# ---- export -------------------------------------------------------------------------------------

def gt_chords(a: schema.Analysis) -> list[schema.ChordOut]:
    """The GT sequence: main layer + passing chords Nathan made or accepted (no suggestions)."""
    keep = [ch for ch in a.chords if not ch.suggestion and (ch.layer == "main" or ch.edited)]
    return flatten(keep, 1.0)


def detected_beats(a: schema.Analysis) -> np.ndarray:
    return np.asarray(sorted(b.t for b in a.beats if b.bar >= 0), dtype=float)


def _bar_grid(st: dict, a: schema.Analysis):
    """The bars of the chart export: (beat list, bar_starts, beats_per_bar, section_of_bar, labels,
    chart_bar_conf). The imported chart's own bars on the beats it was laid on, while those are
    unchanged; otherwise the analysis' bars on the detected beats."""
    ch = st.get("chart") or {}
    if ch.get("format") == "chart":
        src = ch.get("source")
        beats = np.asarray(st.get("beats") or [], dtype=float) if src == "tapped" else detected_beats(a)
        if len(beats) >= 2 and ch.get("beats_sig") == _beats_sig(beats):
            try:
                ac = CH.parse_annotated(ch["text"], ch.get("beats_per_bar") or 4)
                lay = CH.layout(ac.chart, beats, ch.get("start_beat") or 0, ac.anchors)
                return (beats, lay.bar_start, lay.beats_per_bar, lay.section_of_bar,
                        [s.label for s in ac.chart.sections], ac.bar_conf)
            except CH.ChartError:
                pass
    beats = detected_beats(a)
    bars = [br for br in sorted(a.bars, key=lambda b: b.i) if not br.free]
    if not bars or len(beats) < 2:
        return beats, [], 4, [], [None], None
    bpb = max(set(br.beats for br in bars), key=[br.beats for br in bars].count)
    starts = [float(CH.beat_at(beats, br.start)) for br in bars]
    sec_of, labels, cur = [], [], object()
    for br in bars:
        if br.section != cur:
            labels.append(br.section)
            cur = br.section
        sec_of.append(len(labels) - 1)
    return beats, starts, bpb, sec_of, labels, None


def beat_numbers(times: np.ndarray, grid_beats: np.ndarray, bar_starts: list[float], bpb: int) -> list[int]:
    """Beat-in-bar of each time on the chart's bars (1..bpb; 0 outside them or between beats)."""
    if len(grid_beats) < 2 or not bar_starts:
        return [0] * len(times)
    x = np.atleast_1d(CH.beat_at(grid_beats, np.asarray(times, dtype=float)))
    starts = np.asarray(bar_starts, dtype=float)
    out = []
    for xi in x:
        k = int(np.searchsorted(starts, xi + 0.25, side="right")) - 1
        if k < 0 or xi - starts[k] >= bpb - 0.25:
            out.append(0)
            continue
        off = xi - starts[k]
        out.append(int(round(off)) + 1 if abs(off - round(off)) <= 0.25 else 0)
    return out


def build_export(c, key: str, reviewed: bool | None = None) -> tuple[dict[str, str], dict, list[str], dict]:
    """(file texts by kind, songs.yaml entry, warnings, state)."""
    d = ED.song_dir(c, key)
    e = ED.edited(d)
    a = e.model
    st = load_state(d)
    if reviewed is not None:
        st["reviewed"] = bool(reviewed)
        st["reviewed_ts"] = now_iso() if reviewed else None
    warnings: list[str] = []
    chords = gt_chords(a)
    lab = "".join(f"{ch.start:.3f}\t{ch.end:.3f}\t{ch.harte}\n" for ch in chords)

    # .beats: Nathan's tapped beats (else ours, flagged), with the beat in the bar
    beats, src = beat_list(st, a, "auto")
    if src != "tapped":
        warnings.append("No tapped beats yet: the .beats file holds our detected beats (not independent GT).")
    elif chords:
        covered = (beats[-1] - beats[0]) / max(1e-6, chords[-1].end - chords[0].start)
        if covered < 0.8:
            warnings.append(f"The tapped beats cover only {100 * covered:.0f}% of the song.")
    grid_beats, starts, bpb, sec_of, labels, chart_conf = _bar_grid(st, a)
    pos = beat_numbers(beats, grid_beats, starts, bpb)
    head = (f"{EXPORT_MARK} (tapped beats) {now_iso()}\n"
            if src == "tapped" else f"{EXPORT_MARK} — DETECTED beats, not tapped {now_iso()}\n")
    beats_txt = head + "".join(f"{t:.4f}\t{p}\n" for t, p in zip(beats, pos))

    # .chart.txt: the GT chords on the bars, section anchors, line notes
    timed = [(ch.start, ch.end, ch.harte, ch.conf < 0.5) for ch in chords]
    chart = CH.chart_from_times(timed, grid_beats, starts, bpb, sec_of, labels) if starts else CH.Chart([], bpb)
    bar_conf = []
    g = 0
    for s_ in chart.sections:
        for bi, bar in enumerate(s_.bars):
            unsure = any(bb == bi for bb, _ in s_.unsure)
            if unsure:
                bar_conf.append("L")
            elif st.get("reviewed"):
                bar_conf.append("H")
            else:
                bar_conf.append(chart_conf[g] if chart_conf and g < len(chart_conf) else "M")
            g += 1
    anchors = []
    for si in range(len(chart.sections)):
        g0 = sec_of.index(si) if si in sec_of else None
        anchors.append(None if g0 is None else float(CH.time_at(grid_beats, starts[g0])))
    title = a.source.title
    status = "REVIEWED" if st.get("reviewed") else "DRAFT (not reviewed yet)"
    key_name = f"{a.key.global_.tonic} {a.key.global_.mode}"
    chart_txt = (f"{EXPORT_MARK} — {now_iso()}\n# {title} — {key} — {status}\n"
                 f"# Key {key_name}, ~{a.tempo.bpm:.1f} BPM; GT beats: {src}. Format: DESIGN §0.1 bar text;\n"
                 f"# [Section @start-seconds]; (H)/(M)/(L) = confidence of the line; '?' = unsure chord.\n\n"
                 + CH.format_chart(chart, 4, anchors, bar_conf))
    if not chords:
        warnings.append("No chords to export.")
    rel = f"eval/gt/{key}"
    entry = {"key": key, "title": title, "artist": a.source.artist, "set": "D1",
             "reviewed": bool(st.get("reviewed")), "lab": f"{rel}.lab", "beats": f"{rel}.beats",
             "chart": f"{rel}.chart.txt", "tapped_beats": src == "tapped", "source_url": a.source.url,
             "key_sig": key_name, "bpm": round(a.tempo.bpm, 1), "n_chords": len(chords), "updated": now_iso()}
    return {"lab": lab, "beats": beats_txt, "chart": chart_txt}, entry, warnings, st


def update_songs_yaml(eval_dir: Path, entry: dict) -> Path:
    """Add or update one song in eval/songs.yaml (created when missing; other songs and unknown
    fields are kept)."""
    import yaml
    p = Path(eval_dir) / "songs.yaml"
    with _yaml_lock:
        data = None
        if p.exists():
            try:
                data = yaml.safe_load(p.read_text("utf-8"))
            except (OSError, yaml.YAMLError) as e:
                bak = p.with_suffix(".yaml.bak")
                shutil.copyfile(p, bak)
                log.warning("songs.yaml unreadable (%s); kept a copy at %s", e, bak)
        if data is None:
            data = {"songs": []}
        if isinstance(data, list):
            data = {"songs": data}
        if not isinstance(data, dict):
            data = {"songs": []}
        if "songs" in data and isinstance(data["songs"], list):
            songs = data["songs"]
            for s in songs:
                if isinstance(s, dict) and s.get("key") == entry["key"]:
                    s.update(entry)
                    break
            else:
                songs.append(dict(entry))
        else:                                          # a mapping keyed by song key
            cur = data.get(entry["key"]) if isinstance(data.get(entry["key"]), dict) else {}
            data[entry["key"]] = {**cur, **{k: v for k, v in entry.items() if k != "key"}}
        head = ("# Evaluation songs (DESIGN §13.3). D1 = Nathan's annotated songs; `reviewed: true` once he has\n"
                "# checked the GT by ear in Ground-truth mode. Written by the server (GT export); edits are kept.\n")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".yaml.tmp")
        tmp.write_text(head + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), "utf-8")
        tmp.replace(p)
    return p


def write_export(c, key: str, reviewed: bool | None = None) -> dict:
    files, entry, warnings, st = build_export(c, key, reviewed)
    gt_dir = Path(c.eval_dir) / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    chart_p = gt_dir / f"{key}.chart.txt"
    if chart_p.exists():                               # keep the hand-made draft chart once
        old = chart_p.read_text("utf-8")
        draft = gt_dir / f"{key}.chart.draft.txt"
        if not old.startswith(EXPORT_MARK) and not draft.exists():
            draft.write_text(old, "utf-8")
            warnings.append(f"The previous chart was kept as {draft.name}.")
    paths = {"lab": gt_dir / f"{key}.lab", "beats": gt_dir / f"{key}.beats", "chart": chart_p}
    for k, p in paths.items():
        p.write_text(files[k], "utf-8")
    yaml_p = update_songs_yaml(Path(c.eval_dir), entry)
    d = c.cache_dir / key
    st["exported"] = {k: str(p.relative_to(Path(c.eval_dir).parent)).replace("\\", "/")
                      if p.is_relative_to(Path(c.eval_dir).parent) else str(p) for k, p in paths.items()}
    st["exported_ts"] = now_iso()
    save_state(d, st)
    return {"files": st["exported"], "songs_yaml": str(yaml_p), "reviewed": bool(st.get("reviewed")),
            "warnings": warnings, "n_chords": entry["n_chords"], "gt": public_state(c, key, st), "entry": entry}


# ---- routes -------------------------------------------------------------------------------------

router = APIRouter()


@router.get("/api/tracks/{key}/gt")
def get_gt(key: str, c=Depends(ctx)) -> JSONResponse:
    d = ED.song_dir(c, key)
    return JSONResponse(public_state(c, key, load_state(d)), headers={"Cache-Control": "no-cache"})


class BeatsIn(BaseModel):
    beats: list[float]
    t0: float | None = None             # replace only the stored beats in [t0, t1] (a partial re-tap)
    t1: float | None = None


@router.api_route("/api/tracks/{key}/gt/beats", methods=["PUT", "POST"])
def put_beats(key: str, body: BeatsIn, c=Depends(ctx)) -> dict:
    d = ED.song_dir(c, key)
    a = ED.edited(d).model
    new = clean_beats(body.beats, a.source.duration)
    with ED.song_lock(d):
        st = load_state(d)
        old = st.get("beats") or []
        if body.t0 is not None and body.t1 is not None and old:
            lo, hi = sorted((body.t0, body.t1))
            new = clean_beats([b for b in old if b < lo - 1e-6 or b > hi + 1e-6] + new, a.source.duration)
        if len(new) < 2:
            raise ApiError(400, "bad_beats", "Tap at least two beats.")
        st["beats"] = new
        st["beats_ts"] = now_iso()
        save_state(d, st)
    return public_state(c, key, st)


@router.delete("/api/tracks/{key}/gt/beats")
def delete_beats(key: str, c=Depends(ctx)) -> dict:
    d = ED.song_dir(c, key)
    with ED.song_lock(d):
        st = load_state(d)
        st["beats"], st["beats_ts"] = None, now_iso()
        save_state(d, st)
    return public_state(c, key, st)


class ChartIn(BaseModel):
    text: str
    format: str = "chart"
    start: float | None = None          # seconds: the chart's first bar starts at the beat nearest this
    beats: str = "auto"                 # auto | tapped | detected
    import_id: str | None = None        # client-made: a retried import appends nothing twice
    device: str | None = None
    replace: bool = True                # revert the previous import first
    beats_per_bar: int = 4


@router.post("/api/tracks/{key}/gt/chart")
def post_chart(key: str, body: ChartIn, c=Depends(ctx)) -> JSONResponse:
    if body.beats not in ("auto", "tapped", "detected"):
        raise ApiError(400, "bad_request", "beats must be auto, tapped or detected.")
    if not 2 <= body.beats_per_bar <= 12:
        raise ApiError(400, "bad_request", "beats_per_bar must be 2–12.")
    return JSONResponse(import_chart(c, key, body.text, body.format, body.start, body.beats, body.import_id,
                                     body.device, body.replace, body.beats_per_bar))


@router.post("/api/tracks/{key}/gt/import")
def post_import(key: str, body: ChartIn, c=Depends(ctx)) -> JSONResponse:
    return post_chart(key, body, c)


class ReviewIn(BaseModel):
    reviewed: bool = True
    export: bool = True                 # reviewing also (re)writes the eval/gt files


@router.post("/api/tracks/{key}/gt/review")
def post_review(key: str, body: ReviewIn, c=Depends(ctx)) -> dict:
    d = ED.song_dir(c, key)
    if body.export and body.reviewed:
        return write_export(c, key, True)
    with ED.song_lock(d):
        st = load_state(d)
        st["reviewed"] = body.reviewed
        st["reviewed_ts"] = now_iso() if body.reviewed else None
        save_state(d, st)
    if (Path(c.eval_dir) / "songs.yaml").exists() and st.get("exported"):
        update_songs_yaml(Path(c.eval_dir), {"key": key, "reviewed": body.reviewed, "updated": now_iso()})
    return {"gt": public_state(c, key, st), "reviewed": body.reviewed}


class ExportIn(BaseModel):
    reviewed: bool | None = None


@router.post("/api/tracks/{key}/gt/export")
def post_export(key: str, body: ExportIn, c=Depends(ctx)) -> dict:
    return write_export(c, key, body.reviewed)


@router.get("/api/tracks/{key}/gt/export")
def get_export(key: str, fmt: str = "lab", reviewed: bool | None = None, c=Depends(ctx)) -> Response:
    if fmt not in ("lab", "beats", "chart"):
        raise ApiError(400, "bad_format", "fmt must be lab, beats or chart.")
    write_export(c, key, reviewed)
    name = {"lab": f"{key}.lab", "beats": f"{key}.beats", "chart": f"{key}.chart.txt"}[fmt]
    text = (Path(c.eval_dir) / "gt" / name).read_text("utf-8")
    return Response(text, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/api/tracks/{key}/gt/draft-chart")
def get_draft_chart(key: str, c=Depends(ctx)) -> dict:
    ED.song_dir(c, key)
    gt_dir = Path(c.eval_dir) / "gt"
    for name, kind in ((f"{key}.chart.draft.txt", "draft"), (f"{key}.chart.txt", "current")):
        p = gt_dir / name
        if p.exists():
            return {"text": p.read_text("utf-8"), "kind": kind, "file": f"eval/gt/{name}"}
    raise ApiError(404, "no_draft", "No chart for this song in eval/gt yet.")

