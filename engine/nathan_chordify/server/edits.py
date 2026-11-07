"""Chord edits (DESIGN §4 ``EditOp``, §5.13 field rules, §8.4, §11; review W3/W4).

Edits are an append-only op log, ``cache/<key>/edits.jsonl`` (one JSON ``EditOp`` per line):

- the client makes each op's ``id`` (a uuid): a retried POST appends nothing twice;
- the server assigns ``seq`` (1, 2, 3 …) and ``ts`` on append, under a per-song lock, so two
  devices that edit at the same time interleave instead of overwriting each other;
- ``apply`` replays every op that isn't reverted, in ``seq`` order, on the stored analysis; on
  the same target the later op wins. Undo is a ``revert`` op (``{"op_id": …}``, or
  ``{"group": …}`` for a whole chart import); reverting a revert is a redo. Nothing is rewritten.

Op kinds (``target`` / ``value``):

=================== ==================================================== ==========================
kind                target                                               value
=================== ==================================================== ==========================
set_chord           {chord_id, t0, t1, layer} = that chord (in place);   {harte | text [+ shift]},
                    {t0, t1} = a time range (the range gets the chord,   layer?, conf?, group?
                    everything else in it is cut away)
insert_rest         as set_chord                                         {} (the chord is "N")
delete_change       {chord_id, t0, t1, layer}: the previous chord runs   {}
                    on (a passing chord just goes)
accept_suggestion   {chord_id, t0, t1}                                   {harte, min_detail?}
reject_suggestion   {chord_id, t0, t1}                                   {harte?}
half_double_tempo   {}                                                   {factor: 0.5 | 2}
set_section         {t0} or {bar}                                        {label} (null removes it)
set_key             {} (whole song) or {t0} (from there on)              {key: "B:major"}
revert              {op_id} or {group}                                   {}
=================== ==================================================== ==========================

``set_meter``, ``set_subdiv``, ``set_free_time``, ``shift_downbeat``, ``move_beat`` and
``tap_beats`` arrive with the beat-grid editor (M7) and are refused with 501 until then.

A chord target is resolved by ``chord_id`` first (ids never change: new chords get ``u<seq>``,
split-off pieces ``<id>~<seq>``); when that id is gone (re-analysis, a reverted op) the chord of
the same layer that best matches ``t0``/``t1`` is used, and failing that the op acts on the time
range. Every changed chord is re-spelled and re-voiced (harte, labels, degrees, equivalents,
voicing, bass line) and ``display.json`` is rebuilt (``build_display``, memoised per chord).

Re-scan suggestions (``POST …/redecode`` with a region) are kept in ``suggestions.json`` and
shown as passing chords with ``suggestion: true`` until they are accepted or rejected.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import schema
from ..theory import harte as H
from ..theory import numbers as NUM
from ..theory import simplify as SM
from ..theory import spelling as S
from ..theory import templates as T
from ..theory import vocab as VOC
from ..theory import voicing as V
from .common import ApiError, check_key, ctx, not_implemented, now_iso

log = logging.getLogger("nathan_chordify.server.edits")

EDITS_FILE = "edits.jsonl"
SUGGESTIONS_FILE = "suggestions.json"

KINDS = ("set_chord", "delete_change", "insert_rest", "accept_suggestion", "reject_suggestion",
         "half_double_tempo", "set_section", "set_key", "revert")
KINDS_LATER = ("set_meter", "set_subdiv", "set_free_time", "shift_downbeat", "move_beat", "tap_beats")

OP_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
DEVICE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
MAX_OPS_PER_POST = 2000
EPS = 1e-3
SUGGESTION_MIN_DETAIL = 0.05          # accepted suggestions show from Standard (same as the re-scan)


# =============================================================================================
# Op validation (on append; a bad op never enters the log)
# =============================================================================================

def _num(v, what: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ApiError(400, "bad_op", f"{what} must be a number.") from None
    if not math.isfinite(x):
        raise ApiError(400, "bad_op", f"{what} must be a finite number.")
    return x


def _chord_target(t: dict, need: bool = True) -> dict:
    out: dict = {}
    if t.get("chord_id") is not None:
        cid = str(t["chord_id"])
        if not OP_ID_RE.match(cid.replace("~", "-")):
            raise ApiError(400, "bad_op", "Bad chord id.")
        out["chord_id"] = cid
    if t.get("t0") is not None or t.get("t1") is not None:
        t0, t1 = _num(t.get("t0"), "target.t0"), _num(t.get("t1"), "target.t1")
        if not 0 <= t0 < t1:
            raise ApiError(400, "bad_op", "The target range must have 0 ≤ t0 < t1.")
        out["t0"], out["t1"] = round(t0, 4), round(t1, 4)
    if t.get("layer") is not None:
        if t["layer"] not in ("main", "passing"):
            raise ApiError(400, "bad_op", "target.layer must be 'main' or 'passing'.")
        out["layer"] = t["layer"]
    if t.get("mode") == "range":
        out["mode"] = "range"
    if need and "chord_id" not in out and "t0" not in out:
        raise ApiError(400, "bad_op", "A chord op needs target.chord_id or target.t0/t1.")
    if out.get("mode") == "range" and "t0" not in out:
        raise ApiError(400, "bad_op", "A range op needs target.t0 and target.t1.")
    return out


def _canon_harte(value: dict) -> str:
    """value.harte, or value.text (a typed chord, read at transposition value.shift) → Harte."""
    if value.get("harte") is not None:
        try:
            r, q, b = H.parse_harte(str(value["harte"]), strict=False)
        except ValueError as e:
            raise ApiError(400, "bad_chord", str(e)) from None
        return H.to_harte(r, q, b) if q != T.N else "N"
    if value.get("text") is not None:
        p = parse_chord_text(str(value["text"]))
        if p is None:
            raise ApiError(400, "bad_chord", f"Can't read the chord {value['text']!r}.", PARSE_HINT)
        r, q, b = p
        if q == T.N:
            return "N"
        sh = int(value.get("shift") or 0)
        return H.to_harte((r - sh) % 12, q, (b - sh) % 12)
    raise ApiError(400, "bad_op", "set_chord needs value.harte or value.text.")


def normalize_op(raw: dict, device: str | None = None) -> dict:
    """Validate one client op and bring it to the stored shape (without seq/ts)."""
    if not isinstance(raw, dict):
        raise ApiError(400, "bad_op", "Each op must be an object.")
    oid = raw.get("id")
    if not isinstance(oid, str) or not OP_ID_RE.match(oid):
        raise ApiError(400, "bad_op", "Each op needs an id (letters, digits, - _ . :; up to 80).")
    kind = raw.get("kind")
    if kind in KINDS_LATER:
        raise not_implemented(f"The {kind} edit", "M7")
    if kind not in KINDS:
        raise ApiError(400, "bad_op", f"Unknown op kind {kind!r}.")
    target = raw.get("target") or {}
    value = raw.get("value") or {}
    if not isinstance(target, dict) or not isinstance(value, dict):
        raise ApiError(400, "bad_op", "target and value must be objects.")
    dev = str(raw.get("device") or device or "unknown")
    if not DEVICE_RE.match(dev):
        dev = "unknown"
    v: dict = {}
    if isinstance(value.get("group"), str) and OP_ID_RE.match(value["group"]):
        v["group"] = value["group"]
    if kind in ("set_chord", "insert_rest"):
        t = _chord_target(target)
        v["harte"] = "N" if kind == "insert_rest" else _canon_harte(value)
        if value.get("text") is not None:
            v["text"] = str(value["text"])[:40]
        if value.get("layer") is not None:
            if value["layer"] not in ("main", "passing"):
                raise ApiError(400, "bad_op", "value.layer must be 'main' or 'passing'.")
            v["layer"] = value["layer"]
        for k in ("conf", "min_detail"):
            if value.get(k) is not None:
                x = _num(value[k], f"value.{k}")
                if not 0 <= x <= 1:
                    raise ApiError(400, "bad_op", f"value.{k} must be between 0 and 1.")
                v[k] = round(x, 3)
    elif kind == "delete_change":
        t = _chord_target(target)
    elif kind in ("accept_suggestion", "reject_suggestion"):
        t = _chord_target(target)
        if value.get("harte") is not None or value.get("text") is not None:
            v["harte"] = _canon_harte(value)
        elif kind == "accept_suggestion":
            raise ApiError(400, "bad_op", "accept_suggestion needs value.harte.")
        if kind == "accept_suggestion" and "t0" not in t:
            raise ApiError(400, "bad_op", "accept_suggestion needs target.t0/t1.")
        if value.get("min_detail") is not None:
            v["min_detail"] = round(min(max(_num(value["min_detail"], "value.min_detail"), 0.0), 1.0), 3)
    elif kind == "half_double_tempo":
        t = {}
        f = _num(value.get("factor"), "value.factor")
        if f not in (0.5, 2.0):
            raise ApiError(400, "bad_op", "value.factor must be 0.5 (half tempo) or 2 (double tempo).")
        v["factor"] = f
    elif kind == "set_section":
        t = {}
        if target.get("t0") is not None:
            t["t0"] = round(_num(target["t0"], "target.t0"), 4)
        if target.get("bar") is not None:
            t["bar"] = int(_num(target["bar"], "target.bar"))
        if not t:
            raise ApiError(400, "bad_op", "set_section needs target.t0 or target.bar.")
        label = value.get("label")
        if label is not None:
            label = str(label).strip()[:40] or None
        v["label"] = label
    elif kind == "set_key":
        t = {}
        if target.get("t0") is not None:
            t["t0"] = round(max(0.0, _num(target["t0"], "target.t0")), 4)
        text = str(value.get("key") or value.get("tonic") or "")
        k = parse_key(text, value.get("mode"))
        if k is None:
            raise ApiError(400, "bad_op", "value.key must be a key such as 'B:major' or 'G#m'.")
        tonic = _KEY_RE.match(text).group(1)
        tonic = tonic[0].upper() + tonic[1:].replace("♯", "#").replace("♭", "b")
        v["key"] = f"{tonic}:{'minor' if k >= 12 else 'major'}"          # as written (C# stays C#)
    else:                                              # revert
        t = {}
        if isinstance(target.get("op_id"), str) and OP_ID_RE.match(target["op_id"]):
            t["op_id"] = target["op_id"]
        elif isinstance(target.get("group"), str) and OP_ID_RE.match(target["group"]):
            t["group"] = target["group"]
        else:
            raise ApiError(400, "bad_op", "revert needs target.op_id or target.group.")
    return {"id": oid, "kind": kind, "device": dev, "target": t, "value": v}


# =============================================================================================
# Storage: the op log and the re-scan suggestions
# =============================================================================================

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def song_lock(d: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(str(d.resolve()).lower(), threading.Lock())


def read_ops(d: Path) -> list[dict]:
    """Every op in the log (a torn or garbled line is skipped with a warning)."""
    p = d / EDITS_FILE
    if not p.exists():
        return []
    out = []
    for n, line in enumerate(p.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            op = json.loads(line)
            if isinstance(op, dict) and isinstance(op.get("id"), str) and isinstance(op.get("seq"), int):
                out.append(op)
                continue
        except ValueError:
            pass
        log.warning("%s line %d: unreadable edit op skipped", p, n)
    out.sort(key=lambda o: o["seq"])
    return out


def append_ops(d: Path, ops: list[dict]) -> tuple[list[dict], list[dict]]:
    """Append normalized ops, skipping ids already in the log (idempotent retries).

    Returns (appended, duplicates), each with the op's seq."""
    with song_lock(d):
        existing = read_ops(d)
        seen = {o["id"]: o for o in existing}
        seq = existing[-1]["seq"] if existing else 0
        appended, dups, lines = [], [], []
        for op in ops:
            if op["id"] in seen:
                dups.append(seen[op["id"]])
                continue
            seq += 1
            stored = {"id": op["id"], "seq": seq, "ts": now_iso(), "device": op["device"], "kind": op["kind"],
                      "target": op["target"], "value": op["value"]}
            seen[op["id"]] = stored
            appended.append(stored)
            lines.append(json.dumps(stored, ensure_ascii=False, separators=(",", ":")))
        if lines:
            p = d / EDITS_FILE
            d.mkdir(parents=True, exist_ok=True)
            with open(p, "ab") as f:
                if f.tell() > 0:
                    with open(p, "rb") as r:            # a crash may have left a torn last line
                        r.seek(-1, 2)
                        if r.read(1) != b"\n":
                            f.write(b"\n")
                f.write(("\n".join(lines) + "\n").encode("utf-8"))
                f.flush()
    return appended, dups


def active_ops(ops: list[dict]) -> tuple[list[dict], set[str]]:
    """(ops still in effect, in seq order, without the reverts; ids of reverted ops).

    Walks the log backwards: a revert that is itself reverted does nothing (redo)."""
    reverted: set[str] = set()
    groups: set[str] = set()
    by_seq = sorted(ops, key=lambda o: o["seq"])
    for op in reversed(by_seq):
        if op["id"] in reverted or op.get("value", {}).get("group") in groups:
            reverted.add(op["id"])                      # (a reverted revert does nothing: redo)
            continue
        if op["kind"] == "revert":
            t = op.get("target", {})
            if "op_id" in t:
                reverted.add(t["op_id"])
            elif "group" in t:
                groups.add(t["group"])
    for op in by_seq:                                   # a group revert covers the whole group
        if op.get("value", {}).get("group") in groups:
            reverted.add(op["id"])
    active = [o for o in by_seq if o["id"] not in reverted and o["kind"] != "revert"]
    return active, reverted


def suggestion_id(start: float, end: float, harte: str) -> str:
    return "sg-" + hashlib.sha1(f"{start:.3f}|{end:.3f}|{harte}".encode()).hexdigest()[:10]


def base_tag(a: schema.Analysis) -> str:
    return f"{a.created_at}|{a.params_hash}|{a.engine_version}"


def store_suggestions(d: Path, base: schema.Analysis, result: schema.Analysis,
                      region: tuple[float, float]) -> list[schema.ChordOut]:
    """Keep the re-scan's new chords (``suggestion: true``) as pending suggestions of this song;
    earlier suggestions inside the re-scanned region are replaced."""
    new = []
    for c in result.chords:
        if not c.suggestion:
            continue
        sid = suggestion_id(c.start, c.end, c.harte)
        new.append(c.model_copy(update={"id": sid, "layer": "passing", "edited": False,
                                        "min_detail": c.min_detail if c.min_detail is not None else SUGGESTION_MIN_DETAIL}))
    with song_lock(d):
        ops = read_ops(d)
        rev = ops[-1]["seq"] if ops else 0              # edits up to here are older than the re-scan
        old, revs = load_suggestions(d, base, with_revs=True)
        lo, hi = region
        keep = [c for c in old if c.end <= lo + EPS or c.start >= hi - EPS]
        ids = {c.id for c in keep}
        merged = keep + [c for c in new if c.id not in ids]
        merged.sort(key=lambda c: c.start)
        revs = {c.id: revs.get(c.id, 0) for c in keep} | {c.id: rev for c in new if c.id not in ids}
        data = {"base": base_tag(base), "chords": [c.model_dump(mode="json", by_alias=True) for c in merged],
                "revs": revs}
        (d / SUGGESTIONS_FILE).write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    return new


def load_suggestions(d: Path, base: schema.Analysis, with_revs: bool = False):
    """Pending suggestions made against this exact analysis (a re-analysis drops them); with
    ``with_revs`` also {suggestion id: op-log rev when the re-scan ran}."""
    p = d / SUGGESTIONS_FILE
    out: list[schema.ChordOut] = []
    revs: dict[str, int] = {}
    if p.exists():
        try:
            data = json.loads(p.read_text("utf-8"))
            if data.get("base") == base_tag(base):
                out = [schema.ChordOut.model_validate(c) for c in data.get("chords", [])]
                revs = {str(k): int(v) for k, v in (data.get("revs") or {}).items()}
        except (OSError, ValueError) as e:
            log.warning("%s unreadable: %s", p, e)
    return (out, revs) if with_revs else out


# =============================================================================================
# Keys and typed chords
# =============================================================================================

PARSE_HINT = "Try C, Cm7, F#/A#, B+7, Esus4, E/F#, C#m7b5, Bbmaj7 or N (no chord)."
_KEY_RE = re.compile(r"^\s*([A-Ga-g](?:##|bb|[#b♯♭])?)\s*(?::|\s)?\s*(major|minor|maj|min|m|M|)\s*$")


def parse_key(text: str, mode: str | None = None) -> int | None:
    """"B:major" / "B" / "G#m" / "G#:minor" / "Bb minor" → key index (0–11 major, 12–23 minor)."""
    m = _KEY_RE.match(text or "")
    if not m:
        return None
    try:
        pc = S.parse_note(m.group(1)).pc
    except ValueError:
        return None
    md = (mode or m.group(2) or "major").lower()
    minor = md in ("minor", "min", "m") and m.group(2) != "M"
    return pc + (12 if minor else 0)


def key_string(k: int) -> str:
    name = S.key_name(k)
    minor = S.key_is_minor(k)
    return f"{name[:-1] if minor else name}:{'minor' if minor else 'major'}"


def key_index(tonic: str, mode: str) -> int:
    return S.parse_note(tonic).pc + (12 if mode == "minor" else 0)


def key_at(a: schema.Analysis, t: float) -> int:
    """The local key at time t (the same rule as theory.display)."""
    segs = a.key.segments
    for seg in segs:
        if seg.start <= t < seg.end:
            return key_index(seg.tonic, seg.mode)
    if segs:
        return key_index(segs[-1].tonic, segs[-1].mode)
    return key_index(a.key.global_.tonic, a.key.global_.mode)


def parse_chord_text(text: str) -> tuple[int, str, int] | None:
    try:
        p = H.parse_any(text)
    except ValueError:
        return None
    return p.triple


# =============================================================================================
# apply: the op log → the edited analysis
# =============================================================================================

def _triple(c: schema.ChordOut) -> tuple[int, str, int]:
    try:
        return H.parse_harte(c.harte, strict=False)
    except ValueError:
        return H.NO_CHORD


def _overlap(c: schema.ChordOut, t0: float, t1: float) -> float:
    return max(0.0, min(c.end, t1) - max(c.start, t0))


def _blank_chord(cid: str, start: float, end: float, harte: str, layer: str, conf: float,
                 min_detail: float | None) -> schema.ChordOut:
    return schema.ChordOut(
        id=cid, start=round(start, 4), end=round(end, 4), start_beat=None, end_beat=None, bar=0,
        beat_in_bar=None, dur_beats=None, harte=harte, root=None, quality="N", bass=None,
        label=schema.Labels(simple="", standard="", full=""),
        degree=schema.Degree(root=None, bass=None, nashville="", roman=""),
        layer=layer, min_detail=None if layer == "main" else min_detail, conf=conf, edited=True)


@dataclass
class _Work:
    a: schema.Analysis
    chords: list[schema.ChordOut]
    dirty: set[str] = field(default_factory=set)        # re-spell
    moved: set[str] = field(default_factory=set)        # recompute beat fields
    orig: dict[str, tuple[str, float]] = field(default_factory=dict)   # id → (harte, conf) before editing
    tempo_factor: float = 1.0
    section_ops: list[tuple[dict, str | None]] = field(default_factory=list)
    key_ops: list[tuple[dict, str]] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    sugg_rev: dict[str, int] = field(default_factory=dict)   # suggestion id → op-log rev of its re-scan

    def get(self, cid: str | None) -> schema.ChordOut | None:
        if cid is None:
            return None
        for c in self.chords:
            if c.id == cid:
                return c
        return None

    def remove(self, c: schema.ChordOut) -> None:
        self.chords = [x for x in self.chords if x is not c]

    def touch(self, c: schema.ChordOut, respell: bool = False) -> None:
        self.moved.add(c.id)
        if respell:
            self.dirty.add(c.id)

    def resolve(self, t: dict, pool=None) -> schema.ChordOut | None:
        """The chord an op means (see the module doc), or None = act on the time range."""
        if t.get("mode") == "range":
            return None
        pool = self.chords if pool is None else pool
        t0, t1 = t.get("t0"), t.get("t1")
        cid = t.get("chord_id")
        if cid is not None:
            c = next((x for x in pool if x.id == cid), None)
            if c is not None and (t0 is None or _overlap(c, t0, t1) > 0 or abs(c.start - t0) < 0.05):
                return c
        if t0 is None or cid is None:
            return None
        layer = t.get("layer")
        best, ov = None, 0.0
        for c in pool:
            if layer is not None and c.layer != layer:
                continue
            o = _overlap(c, t0, t1)
            if o > ov + 1e-9:
                best, ov = c, o
        if best is None:
            return None
        if ov >= 0.6 * (t1 - t0) and ov >= 0.6 * (best.end - best.start):
            return best
        return None

    def carve(self, t0: float, t1: float, layers: tuple[str, ...], seq: int) -> None:
        """Cut [t0, t1) out of the chords of these layers (splitting a chord that spans it).
        A pending suggestion survives the edits older than its re-scan."""
        out: list[schema.ChordOut] = []
        for c in self.chords:
            if c.layer not in layers or c.end <= t0 + EPS or c.start >= t1 - EPS or \
                    (c.suggestion and seq <= self.sugg_rev.get(c.id, 0)):
                out.append(c)
                continue
            if c.start >= t0 - EPS and c.end <= t1 + EPS:
                continue                                      # wholly inside: gone
            if c.start < t0 - EPS and c.end > t1 + EPS:      # spans the range: split in two
                right = c.model_copy(deep=True, update={"id": f"{c.id}~{seq}", "start": round(t1, 4)})
                c.end = round(t0, 4)
                self.touch(c)
                self.touch(right)
                if right.id not in self.dirty and c.id in self.dirty:
                    self.dirty.add(right.id)
                out += [c, right]
                continue
            if c.start < t0 - EPS:
                c.end = round(t0, 4)
            else:
                c.start = round(t1, 4)
            self.touch(c)
            out.append(c)
        self.chords = out

    def relabel(self, c: schema.ChordOut, harte: str, conf: float | None) -> None:
        if c.id not in self.orig:
            self.orig[c.id] = (c.harte, c.conf)
        c.harte = harte
        c.conf = 1.0 if conf is None else conf
        c.edited = True
        c.suggestion = False
        c.sources = {}
        self.touch(c, respell=True)


def _op_set_chord(w: _Work, op: dict) -> None:
    t, v, seq = op["target"], op["value"], op["seq"]
    harte = v["harte"]
    c = w.resolve(t)
    if c is not None:
        w.relabel(c, harte, v.get("conf"))
        if v.get("layer") and v["layer"] != c.layer:
            c.layer = v["layer"]
            c.min_detail = None if c.layer == "main" else v.get("min_detail", SUGGESTION_MIN_DETAIL)
        return
    if "t0" not in t:
        w.notes.append({"op": op["id"], "note": "chord not found"})
        return
    layer = v.get("layer") or t.get("layer") or "main"
    if t.get("chord_id") is not None and t.get("mode") != "range":
        layer = t.get("layer") or layer
    t0, t1 = t["t0"], t["t1"]
    # what was there (our detection, or an earlier edit) stays one click away as an alternative
    want = H.parse_harte(harte, strict=False)
    under = sorted((c for c in w.chords if c.layer == layer and _overlap(c, t0, t1) > EPS),
                   key=lambda c: -_overlap(c, t0, t1))
    alts, seen = [], {want}
    for c in under:
        tr = _triple(c)
        if tr not in seen:
            seen.add(tr)
            alts.append(schema.Alternative(harte=c.harte, conf=round(c.conf, 3)))
    w.carve(t0, t1, ("main", "passing") if layer == "main" else ("passing",), seq)
    new = _blank_chord(f"u{seq}", t0, t1, harte, layer, v.get("conf", 1.0), v.get("min_detail", SUGGESTION_MIN_DETAIL))
    new.alternatives = alts[:3]
    w.chords.append(new)
    w.touch(new, respell=True)


def _op_delete(w: _Work, op: dict) -> None:
    c = w.resolve(op["target"])
    if c is None:
        w.notes.append({"op": op["id"], "note": "chord not found"})
        return
    w.remove(c)
    if c.layer == "passing":
        return
    mains = sorted((x for x in w.chords if x.layer == "main"), key=lambda x: x.start)
    prev = next((x for x in reversed(mains) if abs(x.end - c.start) < 0.05), None)
    nxt = next((x for x in mains if abs(x.start - c.end) < 0.05), None)
    if prev is not None:
        prev.end = c.end
        prev.edited = True
        w.touch(prev)
    elif nxt is not None:
        nxt.start = c.start
        nxt.edited = True
        w.touch(nxt)


def _suggestion_match(w: _Work, t: dict, harte: str | None) -> schema.ChordOut | None:
    sugg = [c for c in w.chords if c.suggestion]
    if t.get("chord_id"):
        c = next((x for x in sugg if x.id == t["chord_id"]), None)
        if c is not None:
            return c
    if "t0" not in t:
        return None
    want = H.parse_harte(harte, strict=False) if harte else None
    for c in sugg:
        if want is not None and _triple(c) != want:
            continue
        if _overlap(c, t["t0"], t["t1"]) >= 0.5 * min(c.end - c.start, t["t1"] - t["t0"]):
            return c
    return None


def _op_accept(w: _Work, op: dict) -> None:
    t, v = op["target"], op["value"]
    c = _suggestion_match(w, t, v.get("harte"))
    if c is not None:
        c.suggestion = False
        c.edited = True
        c.conf = 1.0                                      # confirmed by the user
        if v.get("min_detail") is not None:
            c.min_detail = v["min_detail"]
        w.touch(c)
        return
    w.carve(t["t0"], t["t1"], ("passing",), op["seq"])
    new = _blank_chord(f"a{op['seq']}", t["t0"], t["t1"], v["harte"], "passing", 1.0,
                       v.get("min_detail", SUGGESTION_MIN_DETAIL))
    w.chords.append(new)
    w.touch(new, respell=True)


def _op_reject(w: _Work, op: dict) -> None:
    c = _suggestion_match(w, op["target"], op["value"].get("harte"))
    if c is not None:
        w.remove(c)


def _halve(beats: list[list], bars: list[dict]) -> tuple[list[list], list[dict]]:
    """Half tempo: keep every other beat, two bars become one (free-time bars stay)."""
    nb, nbar = [], []
    i = 0
    while i < len(bars):
        a = bars[i]
        if a["free"]:
            nbar.append({**a, "beats_l": a["beats_l"]})
            i += 1
            continue
        group = [a]
        if i + 1 < len(bars) and not bars[i + 1]["free"]:
            group.append(bars[i + 1])
        allb = [b for g in group for b in g["beats_l"]]
        kept = allb[0::2]
        nbar.append({"start": a["start"], "end": group[-1]["end"], "section": a["section"], "free": False,
                     "beats_l": kept})
        i += len(group)
    return nb, nbar


def _double(beats: list[list], bars: list[dict]) -> tuple[list[list], list[dict]]:
    """Double tempo: a beat in the middle of every beat, each bar becomes two."""
    nbar = []
    for a in bars:
        bl = a["beats_l"]
        if a["free"] or not bl:
            nbar.append(a)
            continue
        dense = []
        for j, (t, conf) in enumerate(bl):
            nxt = bl[j + 1][0] if j + 1 < len(bl) else a["end"]
            dense += [(t, conf), ((t + nxt) / 2, conf * 0.5)]
        n = len(bl)
        first, second = dense[:n], dense[n:]
        mid = second[0][0] if second else a["end"]
        nbar.append({"start": a["start"], "end": mid, "section": a["section"], "free": False, "beats_l": first})
        if second:
            nbar.append({"start": mid, "end": a["end"], "section": a["section"], "free": False, "beats_l": second})
    return [], nbar


def _regrid(a: schema.Analysis, factor: float) -> None:
    """Rebuild beats and bars at half / double tempo (DESIGN §5.3 user override). Chord times
    don't move; their beat positions are recomputed by the caller."""
    steps = int(round(math.log2(factor)))
    if steps == 0:
        return
    by_bar: dict[int, list[tuple[float, float]]] = {}
    loose = []                                          # beats outside the bars (free time, pickups)
    for b in sorted(a.beats, key=lambda b: b.t):
        if b.bar >= 0:
            by_bar.setdefault(b.bar, []).append((b.t, b.conf))
        else:
            loose.append(b)
    bars = [{"start": br.start, "end": br.end, "section": br.section, "free": br.free,
             "beats_l": by_bar.get(br.i, [])} for br in sorted(a.bars, key=lambda x: x.i)]
    old_n = len(bars)
    for _ in range(abs(steps)):
        _, bars = (_halve if steps < 0 else _double)([], bars)
    beats_out: list[schema.BeatOut] = []
    bars_out: list[schema.BarOut] = []
    for i, br in enumerate(bars):
        for j, (t, conf) in enumerate(br["beats_l"]):
            beats_out.append(schema.BeatOut(i=0, t=round(t, 4), bar=i, beat=0 if br["free"] else j + 1,
                                            conf=round(conf, 3)))
        bars_out.append(schema.BarOut(i=i, start=round(br["start"], 4), end=round(br["end"], 4),
                                      beats=len(br["beats_l"]), section=br["section"], free=br["free"]))
    beats_out += [b.model_copy() for b in loose]
    beats_out.sort(key=lambda b: b.t)
    for i, b in enumerate(beats_out):
        b.i = i
    a.beats, a.bars = beats_out, bars_out
    ratio = len(bars_out) / old_n if old_n else 1.0
    half = {2: 4, 3: 6, 4: 4, 6: 6}
    dbl = {4: 2, 6: 3, 2: 2, 3: 3}
    subdiv = []
    for s in a.tempo.subdivision:
        cpb = s.cells_per_beat
        for _ in range(abs(steps)):
            cpb = (half if steps < 0 else dbl)[cpb]
        fb = int(round(s.from_bar * ratio))
        if subdiv and subdiv[-1].from_bar == fb:
            subdiv[-1] = schema.SubdivisionOut(from_bar=fb, cells_per_beat=cpb)
        else:
            subdiv.append(schema.SubdivisionOut(from_bar=fb, cells_per_beat=cpb))
    a.tempo = a.tempo.model_copy(update={"bpm": round(a.tempo.bpm * factor, 2), "subdivision": subdiv})
    a.meter = [m.model_copy(update={"from_bar": int(round(m.from_bar * ratio))}) for m in a.meter]
    a.key.changes = [ch.model_copy(update={"bar": _bar_at(a, ch.t)}) for ch in a.key.changes]


def _bar_at(a: schema.Analysis, t: float) -> int:
    best = 0
    for br in a.bars:
        if br.start <= t + 0.05:
            best = br.i
    return best


def _nearest_bar(a: schema.Analysis, t: float) -> int:
    if not a.bars:
        return 0
    return min(a.bars, key=lambda br: abs(br.start - t)).i


def _apply_sections(a: schema.Analysis, ops: list[tuple[dict, str | None]]) -> None:
    bars = sorted(a.bars, key=lambda b: b.i)
    if not bars:
        return
    starts: dict[int, tuple[str, str | None, str]] = {}
    for s in a.sections:
        starts[_nearest_bar(a, s.start)] = (s.label, s.group, s.source)
    for t, label in ops:
        bi = _nearest_bar(a, t["t0"]) if "t0" in t else min(max(int(t["bar"]), 0), bars[-1].i)
        if label:
            starts[bi] = (label, None, "user")
        else:
            starts.pop(bi, None)
    order = sorted(starts)
    secs = []
    for k, bi in enumerate(order):
        label, group, source = starts[bi]
        end = bars[order[k + 1]].start if k + 1 < len(order) else bars[-1].end
        secs.append(schema.SectionOut(label=label, group=group, start=bars[bi].start, end=end, source=source))
    a.sections = secs
    cur = None
    for br in bars:
        if br.i in starts:
            cur = starts[br.i][0]
        br.section = cur


def _apply_keys(a: schema.Analysis, ops: list[tuple[dict, str]]) -> None:
    dur = a.source.duration
    for t, kstr in ops:
        k = parse_key(kstr)
        tonic, mode = kstr.split(":")
        t0 = t.get("t0")
        if t0 is None or t0 <= 0.05:
            a.key.global_ = schema.KeyGlobal(tonic=tonic, mode=mode, conf=1.0)
            a.key.segments = [schema.KeySegment(start=0.0, end=dur, tonic=tonic, mode=mode)]
            a.key.changes = []
            continue
        keep = []
        for s in a.key.segments:
            if s.start >= t0 - 0.05:
                continue
            keep.append(s.model_copy(update={"end": min(s.end, t0)}))
        prev = keep[-1] if keep else None
        keep.append(schema.KeySegment(start=t0, end=dur, tonic=tonic, mode=mode))
        a.key.segments = keep
        changes = [ch for ch in a.key.changes if ch.t < t0 - 0.05]
        if prev is not None and (prev.tonic, prev.mode) != (tonic, mode):
            pk = key_index(prev.tonic, prev.mode)
            diff = (k % 12 - pk % 12) % 12
            if prev.mode == mode and diff in (1, 2):
                kind = "step_up"
            elif prev.mode != mode and diff == (3 if mode == "major" else 9):
                kind = "relative"
            elif prev.mode != mode and diff == 0:
                kind = "parallel"
            else:
                kind = "modulation"
            changes.append(schema.KeyChangeOut(t=round(t0, 3), bar=_bar_at(a, t0), from_=f"{prev.tonic}:{prev.mode}",
                                               to=f"{tonic}:{mode}", kind=kind))
        a.key.changes = changes


def _next_roots(chords: list[schema.ChordOut]) -> dict[str, int | None]:
    trip = [_triple(c) for c in chords]
    out: dict[str, int | None] = {}
    nxt = None
    for c, (r, _, _) in zip(reversed(chords), reversed(trip)):
        out[c.id] = nxt
        if r >= 0:
            nxt = r
    return out


def _sort_key(c: schema.ChordOut):
    return (c.start, c.layer != "main")


def respell(c: schema.ChordOut, k: int, next_root: int | None, orig: tuple[str, float] | None = None) -> None:
    """Re-derive every spelled field of a chord from its Harte label (§5.12 steps 7–9)."""
    r, q, b = _triple(c)
    c.quality = q
    if q == T.N or r < 0:
        c.harte, c.root, c.bass = "N", None, None
        c.label = schema.Labels(simple="N", standard="N", full="N")
        c.degree = schema.Degree(**NUM.degrees(r, q, b, k))
        c.equivalents = []
    else:
        c.harte = H.to_harte_in_key(r, q, b, k, next_root=next_root)
        sp = S.spell(r, q, b, k, next_root=next_root)
        c.root, c.bass = sp.root, sp.bass
        c.label = schema.Labels(**SM.labels(r, q, b, k, next_root=next_root))
        c.degree = schema.Degree(**NUM.degrees(r, q, b, k))
        c.equivalents = [H.to_harte_in_key(*e, k, next_root=next_root) for e in VOC.equivalents(r, q, b)]
    if orig is not None:
        alts = [schema.Alternative(harte=orig[0], conf=round(orig[1], 3))] + list(c.alternatives)
        seen, out = {(r, q, b)}, []
        for al in alts:
            try:
                tr = H.parse_harte(al.harte, strict=False)
            except ValueError:
                continue
            if tr in seen:
                continue
            seen.add(tr)
            out.append(al)
        c.alternatives = out[:6]


def _beat_fields(a: schema.Analysis, chords: list[schema.ChordOut], ids: set[str] | None) -> None:
    beats = sorted(a.beats, key=lambda b: b.t)
    if len(beats) < 2:
        return
    bt = np.array([b.t for b in beats])
    free = [(r.start, r.end) for r in a.tempo.free_time]

    def idx(t: float) -> float:
        n = len(bt)
        if t < bt[0]:
            return (t - bt[0]) / (bt[1] - bt[0])
        if t > bt[-1]:
            return n - 1 + (t - bt[-1]) / (bt[-1] - bt[-2])
        return float(np.interp(t, bt, np.arange(n)))

    def in_free(t: float, strict_end: bool) -> bool:
        return any(s - 1e-6 <= t < e - 1e-6 if not strict_end else s + 1e-6 < t < e - 1e-6 for s, e in free)

    for c in chords:
        if ids is not None and c.id not in ids:
            continue
        sb = None if in_free(c.start, False) else round(idx(c.start), 4)
        eb = None if in_free(c.end, True) else round(idx(c.end), 4)
        bi = max(int(np.searchsorted(bt, c.start + 1e-6)) - 1, 0)
        c.bar = int(beats[bi].bar)
        c.start_beat, c.end_beat = sb, eb
        c.beat_in_bar = None if sb is None or beats[bi].beat == 0 else round(beats[bi].beat + (sb - math.floor(sb)), 4)
        c.dur_beats = None if sb is None or eb is None else round(eb - sb, 4)


def apply_ops(analysis: schema.Analysis, ops: list[dict], suggestions: list[schema.ChordOut] | None = None,
              suggestion_revs: dict[str, int] | None = None) -> tuple[schema.Analysis, list[dict]]:
    """The edited analysis (without the display file) and per-op notes ("chord not found")."""
    a = analysis.model_copy(deep=True)
    active, _ = active_ops(ops)
    sugg = [s.model_copy(deep=True, update={"suggestion": True}) for s in (suggestions or [])]
    if not active and not sugg:
        a.edits = schema.EditsInfo(rev=max((o["seq"] for o in ops), default=0), n_ops=len(ops))
        return a, []
    orig_next = _next_roots(sorted(a.chords, key=_sort_key))
    orig_bass = {(round(b.start, 3), round(b.end, 3)): b for b in a.bass_line}
    orig_span = {c.id: (c.start, c.end, c.harte) for c in a.chords}
    w = _Work(a, list(a.chords) + sugg, sugg_rev=dict(suggestion_revs or {}))
    for s in sugg:
        w.touch(s)                                       # suggestions come with their own spelling
    for op in active:
        kind = op["kind"]
        if kind in ("set_chord", "insert_rest"):
            _op_set_chord(w, op)
        elif kind == "delete_change":
            _op_delete(w, op)
        elif kind == "accept_suggestion":
            _op_accept(w, op)
        elif kind == "reject_suggestion":
            _op_reject(w, op)
        elif kind == "half_double_tempo":
            w.tempo_factor *= op["value"]["factor"]
        elif kind == "set_section":
            w.section_ops.append((op["target"], op["value"].get("label")))
        elif kind == "set_key":
            w.key_ops.append((op["target"], op["value"]["key"]))

    # grid, sections, key
    factor = min(max(w.tempo_factor, 0.25), 4.0)
    regrid = abs(factor - 1.0) > 1e-9
    if regrid:
        _regrid(a, factor)
    if w.section_ops or (regrid and a.sections):
        _apply_sections(a, w.section_ops)
    if w.key_ops:
        _apply_keys(a, w.key_ops)

    # chords: order, spelling, voicing, beat positions, bass line
    chords = sorted((c for c in w.chords if c.end > c.start + EPS), key=_sort_key)
    nxt = _next_roots(chords)
    for c in chords:
        if w.key_ops or c.id in w.dirty or (c.id in orig_next and orig_next[c.id] != nxt[c.id] and not c.suggestion):
            respell(c, key_at(a, c.start), nxt[c.id], w.orig.get(c.id))
    triples = [_triple(c) for c in chords]
    voicings = V.voice_sequence(triples)
    for c, v in zip(chords, voicings):
        c.voicing = schema.Voicing(lh=list(v.lh), rh=list(v.rh))
    _beat_fields(a, chords, None if regrid else w.moved)
    bass_line = []
    for c, v, (r, _, _) in zip(chords, voicings, triples):
        if r < 0 or not v.lh:
            continue
        ob = orig_bass.get((round(c.start, 3), round(c.end, 3)))
        unchanged = orig_span.get(c.id) == (c.start, c.end, c.harte) and c.id not in w.dirty
        role = ob.role if ob is not None and unchanged else "chord_bass"
        bass_line.append(schema.BassNoteOut(
            start=c.start, end=c.end, note=c.bass or "", midi=int(v.lh[0]), conf=c.conf, role=role,
            min_detail=c.min_detail if c.layer == "passing" else None))
    a.chords = chords
    a.bass_line = bass_line
    a.stats = a.stats.model_copy(update={"n_main": sum(c.layer == "main" for c in chords),
                                         "n_passing": sum(c.layer == "passing" for c in chords)})
    a.edits = schema.EditsInfo(rev=max((o["seq"] for o in ops), default=0), n_ops=len(ops))
    return a, w.notes


# =============================================================================================
# display.json, memoised (theory.display.build re-labels every chord: ~1 s per song)
# =============================================================================================

@lru_cache(maxsize=500_000)
def _label(r: int, q: str, b: int, k: int, t: int, pref: str, lv: str, nxt: int | None) -> str:
    return SM.label(r, q, b, k, transpose=t, pref=pref, level=lv, upper_structure=True, next_root=nxt)


@lru_cache(maxsize=50_000)
def _bass_name(pc: int, k: int, t: int, pref: str) -> str:
    return S.spell(pc, "maj", pc, k, t, pref).root


@lru_cache(maxsize=2_000)
def _key_name(k: int, t: int, pref: str) -> str:
    return S.key_name(k, t, pref)


def build_display(analysis: schema.Analysis) -> schema.Display:
    """Exactly ``theory.display.build(analysis)`` (a test checks it), with per-chord memoised labels
    so an edit only spells the chords it changed."""
    from ..theory import display as DISP
    chords = analysis.chords
    triples = DISP.chord_triples(analysis)
    keys = [key_at(analysis, c.start) for c in chords]
    order = sorted(range(len(chords)), key=lambda i: (chords[i].start, chords[i].layer != "main"))
    next_root: dict[int, int | None] = {}
    for x, y in zip(order, order[1:] + [None]):
        next_root[x] = triples[y][0] if y is not None and triples[y][0] >= 0 else None
    seg_keys = [key_index(s.tonic, s.mode) for s in analysis.key.segments] or \
               [key_index(analysis.key.global_.tonic, analysis.key.global_.mode)]
    bass_keys = [key_at(analysis, b.start) for b in analysis.bass_line]
    variants: dict[str, schema.DisplayVariant] = {}
    for t in range(12):
        for pref in DISP.PREFS:
            labels = {}
            for i, c in enumerate(chords):
                r, q, b = triples[i]
                labels[c.id] = [_label(r, q, b, keys[i], t, pref, lv, next_root[i]) for lv in SM.LEVELS]
            variants[f"{t}:{pref}"] = schema.DisplayVariant(
                key_names=[_key_name(k, t, pref) for k in seg_keys], chords=labels,
                bass_line=[_bass_name(bn.midi % 12, k, t, pref) for bn, k in zip(analysis.bass_line, bass_keys)])
    base = V.voice_sequence(triples)
    voicings = {}
    for t in range(12):
        moved = V.transpose_voicings(base, t)
        voicings[str(t)] = {c.id: schema.Voicing(lh=list(v.lh), rh=list(v.rh)) for c, v in zip(chords, moved)}
    return schema.Display(variants=variants, voicings=voicings)


def apply(analysis: schema.Analysis, ops: list[dict],
          suggestions: list[schema.ChordOut] | None = None) -> tuple[schema.Analysis, schema.Display]:
    """DESIGN §4 ``server.edits.apply``: the edited analysis and its display file."""
    a, _ = apply_ops(analysis, ops, suggestions)
    return a, build_display(a)


# =============================================================================================
# The edited song, cached per (analysis, log, suggestions) version
# =============================================================================================

@dataclass
class Edited:
    model: schema.Analysis
    analysis: dict                 # JSON form of model
    display: dict
    ops: list[dict]
    notes: list[dict]


_cache: OrderedDict[str, tuple[tuple, Edited]] = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_SIZE = 6


def _sig(d: Path) -> tuple:
    out = []
    for name in ("analysis.json", "display.json", EDITS_FILE, SUGGESTIONS_FILE):
        try:
            st = (d / name).stat()
            out.append((st.st_mtime_ns, st.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def load_base(d: Path) -> schema.Analysis:
    return schema.Analysis.model_validate_json((d / "analysis.json").read_text("utf-8"))


def edited(d: Path, base: schema.Analysis | None = None) -> Edited:
    """The song with every edit applied (memoised until a file of the song changes)."""
    key = str(d.resolve()).lower()
    sig = _sig(d)
    if base is None:
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None and hit[0] == sig:
                _cache.move_to_end(key)
                return hit[1]
    from_disk = base is None
    base = base or load_base(d)
    ops = read_ops(d)
    sugg, revs = load_suggestions(d, base, with_revs=True)
    model, notes = apply_ops(base, ops, sugg, revs)
    active, _ = active_ops(ops)
    if not active and not sugg and from_disk and (d / "display.json").exists():
        display = json.loads((d / "display.json").read_text("utf-8"))
    else:
        display = build_display(model).model_dump(mode="json", by_alias=True)
    e = Edited(model, model.model_dump(mode="json", by_alias=True), display, ops, notes)
    if from_disk:
        with _cache_lock:
            _cache[key] = (sig, e)
            _cache.move_to_end(key)
            while len(_cache) > _CACHE_SIZE:
                _cache.popitem(last=False)
    return e


def forget(d: Path) -> None:
    with _cache_lock:
        _cache.pop(str(d.resolve()).lower(), None)


def post_ops(d: Path, raw_ops: list, device: str | None = None) -> tuple[list[dict], list[dict], Edited]:
    """Validate the whole batch, append it (idempotently) and return the edited song."""
    if len(raw_ops) > MAX_OPS_PER_POST:
        raise ApiError(413, "too_many_ops", f"At most {MAX_OPS_PER_POST} ops per request.")
    norm = [normalize_op(o, device) for o in raw_ops]
    ids = [o["id"] for o in norm]
    if len(set(ids)) != len(ids):
        raise ApiError(400, "bad_op", "Two ops in one request have the same id.")
    appended, dups = append_ops(d, norm)
    return appended, dups, edited(d)


def op_summaries(ops: list[dict]) -> list[dict]:
    """The log for the history list: each op plus whether it is still in effect."""
    _, reverted = active_ops(ops)
    return [{**o, "active": o["id"] not in reverted} for o in ops]


# =============================================================================================
# Typed chord entry and the picker vocabulary (POST /api/theory/parse, GET /api/theory/vocab)
# =============================================================================================

def _shown_labels(r: int, q: str, b: int, k: int, shift: int, pref: str, nxt: int | None = None) -> list[str]:
    return [_label(r, q, b, k, shift % 12, pref, lv, nxt) for lv in SM.LEVELS]


def describe(triple: tuple[int, str, int], k: int, shift: int = 0, pref: str = "auto",
             next_root: int | None = None) -> dict:
    """Everything the chord picker shows for a concert-pitch chord, at the display shift."""
    r, q, b = triple
    if q == T.N or r < 0:
        return {"ok": True, "harte": "N", "root": None, "quality": "N", "bass": None,
                "label": {"simple": "N", "standard": "N", "full": "N"}, "shown": ["N.C.", "N.C.", "N.C."],
                "degree": NUM.degrees(r, q, b, k), "voicing": {"lh": [], "rh": []}}
    sp = S.spell(r, q, b, k, next_root=next_root)
    v = V.transpose_voicings([V.voice(r, q, b)], shift % 12)[0]
    return {"ok": True, "harte": H.to_harte_in_key(r, q, b, k, next_root=next_root), "root": sp.root,
            "quality": q, "bass": sp.bass, "label": SM.labels(r, q, b, k, next_root=next_root),
            "shown": _shown_labels(r, q, b, k, shift, pref, next_root), "degree": NUM.degrees(r, q, b, k),
            "voicing": {"lh": list(v.lh), "rh": list(v.rh)}}


def parse_for_picker(text: str, k: int, shift: int = 0, pref: str = "auto") -> dict:
    """A typed chord (read at the display shift) → concert Harte, labels, voicing; or an error."""
    s = (text or "").strip()
    if not s:
        return {"ok": False, "input": text, "error": "Type a chord.", "hint": PARSE_HINT}
    try:
        p = H.parse_any(s)
    except ValueError as e:
        msg = str(e)
        hint = PARSE_HINT
        if "unknown chord quality" in msg:
            hint = "Qualities: m 7 maj7 m7 6 m6 9 add9 sus2 sus4 7sus4 ° °7 ø7 + +7 m(maj7) 6/9 11 13 7b9 7#9. " + hint
        return {"ok": False, "input": text, "error": msg[:1].upper() + msg[1:] + ".", "hint": hint}
    if p.quality == T.N:
        return {**describe(H.NO_CHORD, k, shift, pref), "input": text}
    sh = shift % 12
    tr = ((p.root - sh) % 12, p.quality, (p.bass - sh) % 12)
    return {**describe(tr, k, shift, pref), "input": text}


def picker_vocab(k: int, shift: int = 0, pref: str = "auto") -> dict:
    """Root names (spelled in the shown key) and quality suffixes for the root/quality/bass pickers.
    A picked chord is sent back as text ("F#" + "m7" + "/A") to ``parse_for_picker``."""
    ctx_ = S.key_context(k, shift % 12, pref)
    roots = []
    for pc in range(12):
        n = S.display(S.root_note(pc, "maj", ctx_))
        roots.append({"pc": pc, "name": str(n)})
    quals = []
    for q in T.QUALITIES:
        suffix = SM.quality_symbol(q)
        try:
            ok = H.parse_symbol("C" + suffix) == (0, q, 0)
        except ValueError:
            ok = False
        if not ok:
            suffix = next(a for a, qq in H._Q_ALIASES.items() if qq == q and H.parse_symbol("C" + a) == (0, q, 0))
        quals.append({"q": q, "suffix": suffix, "label": SM.quality_symbol(q) or "maj"})
    return {"key": key_string(k), "shown_key": S.key_name(k, shift % 12, pref), "roots": roots, "qualities": quals}


# =============================================================================================
# Routes
# =============================================================================================

router = APIRouter()


def song_dir(c, key: str) -> Path:
    d = c.cache_dir / check_key(key)
    if not (d / "analysis.json").exists():
        if key in c.jobs.active_keys():
            raise ApiError(404, "not_ready", "This song is still being analysed.")
        raise ApiError(404, "not_found", "No analysis for this song.")
    return d


@router.get("/api/tracks/{key}/edits")
def get_edits(key: str, since: int = 0, c=Depends(ctx)) -> JSONResponse:
    """The op log (ops with seq > since) and the current rev, for the history list and for
    other devices to notice new edits."""
    d = song_dir(c, key)
    ops = read_ops(d)
    rev = ops[-1]["seq"] if ops else 0
    return JSONResponse({"rev": rev, "n_ops": len(ops),
                         "ops": [o for o in op_summaries(ops) if o["seq"] > since]},
                        headers={"Cache-Control": "no-cache"})


class OpsIn(BaseModel):
    ops: list[dict]
    device: str | None = None


@router.post("/api/tracks/{key}/edits")
def post_edits(key: str, body: OpsIn, c=Depends(ctx)) -> JSONResponse:
    """Append ops (idempotent by op id) → the merged analysis, its display file and the new rev."""
    d = song_dir(c, key)
    appended, dups, e = post_ops(d, body.ops, body.device)
    mine = {o["id"] for o in appended}
    from . import gt as GT
    return JSONResponse({
        "rev": e.model.edits.rev, "n_ops": e.model.edits.n_ops,
        "appended": [{"id": o["id"], "seq": o["seq"]} for o in appended],
        "duplicates": [{"id": o["id"], "seq": o["seq"]} for o in dups],
        "notes": [n for n in e.notes if n["op"] in mine],
        "analysis": GT.decorate(c, key, e.analysis), "display": e.display})


class ParseIn(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    hartes: list[str] | None = None
    key: str | None = None
    shift: int = 0
    pref: str = "auto"
    next: str | None = None


@router.post("/api/theory/parse")
def theory_parse(body: ParseIn) -> dict:
    """Typed chord entry (§8.4): ``text`` (read at transposition ``shift``) → concert Harte,
    labels (concert and shown) and a voicing, or ``{ok: false, error, hint}``. ``texts`` does the
    same for a list; ``hartes`` labels concert Harte strings at the shift (the picker's
    alternatives and equivalents)."""
    k = parse_key(body.key or "C:major")
    if k is None:
        raise ApiError(400, "bad_key", f"Not a key: {body.key!r}", "Use e.g. 'B:major' or 'G#m'.")
    pref = body.pref if body.pref in S.PREFS else "auto"
    out: dict = {"key": key_string(k), "shift": body.shift % 12}
    if body.text is not None:
        out.update(parse_for_picker(body.text, k, body.shift, pref))
    if body.texts is not None:
        out["results"] = [parse_for_picker(t, k, body.shift, pref) for t in body.texts[:200]]
    if body.hartes is not None:
        labels = {}
        for h in body.hartes[:200]:
            try:
                tr = H.parse_harte(h, strict=False)
            except ValueError:
                continue
            labels[h] = describe(tr, k, body.shift, pref)["shown"]
        out["labels"] = labels
    if body.text is None and body.texts is None and body.hartes is None:
        raise ApiError(400, "bad_request", "Send text, texts or hartes.")
    return out


@router.get("/api/theory/vocab")
def theory_vocab(key: str = "C:major", shift: int = 0, pref: str = "auto") -> dict:
    k = parse_key(key)
    if k is None:
        raise ApiError(400, "bad_key", f"Not a key: {key!r}")
    return picker_vocab(k, shift, pref if pref in S.PREFS else "auto")
