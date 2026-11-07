"""Chord sheets found online, lined up with the recording, used to refine the chords.

    GET    /api/tracks/{key}/sheets                 SheetsState (cache/<key>/sheets/state.json)
    POST   /api/tracks/{key}/sheets/search          {query?, force?} → state "searching" (poll GET every 1.5 s)
    POST   /api/tracks/{key}/sheets/add             {text?, url?}   → a pasted chart / a sheet link, aligned and added
    POST   /api/tracks/{key}/sheets/{id}/apply      {mode: refine|sheet} → {analysis, display, rev, applied, summary}
    DELETE /api/tracks/{key}/sheets/applied         → {analysis, display, rev, applied: null}
    GET    /api/settings                            {sheets_auto_lookup}
    PATCH  /api/settings                            partial → the full settings

Errors: 409 not_analysed, 502 sheets_unreachable, 404 no_sheet, 400 bad_chart.

"refine" (default) is a chart-guided re-decode from the cached decode state (``sheets.guide``):
the sheet is a prior, the audio can overrule it, disagreements are flagged; "sheet" applies the
aligned chords as written. Either is ONE edit group ``sheet-<id>`` through ``edits.post_ops``
(undo = ``revert {group}``); applying another sheet reverts the previous sheet group first.
A search runs in a thread per song (network + CPU alignment, no GPU). With the setting
``sheets_auto_lookup`` (default on) the search starts by itself when a full analysis finishes.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import pipeline, schema
from ..sheets import align as A
from ..sheets import guide as G
from ..sheets import lookup as L
from ..sheets import sources as SRC
from ..sheets.fetch import Fetcher
from ..theory.display import key_index as _key_index
from . import edits as ED
from . import gt as GT
from .common import ApiError, check_key, ctx, now_iso

log = logging.getLogger("nathan_chordify.server.sheets")

router = APIRouter()

SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {"sheets_auto_lookup": True}
CONF_BY_VERDICT = {"good": 1.0, "maybe": 0.75, "different": 0.5}
MAX_PASTE_CHARS = 100_000


# ---- settings -------------------------------------------------------------------------------------

def load_settings(data_dir: Path) -> dict:
    out = dict(DEFAULT_SETTINGS)
    p = Path(data_dir) / SETTINGS_FILE
    if p.exists():
        try:
            saved = json.loads(p.read_text("utf-8"))
            if isinstance(saved, dict):
                out.update({k: v for k, v in saved.items() if k in DEFAULT_SETTINGS})
        except (OSError, ValueError) as e:
            log.warning("%s unreadable: %s", p, e)
    return out


def save_settings(data_dir: Path, settings: dict) -> None:
    p = Path(data_dir) / SETTINGS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(p)


def user_settings(c) -> dict:
    st = getattr(c, "user_settings", None)
    if st is None:
        st = c.user_settings = load_settings(c.data_dir if hasattr(c, "data_dir") else c.library.path.parent)
    return st


@router.get("/api/settings")
def get_settings(c=Depends(ctx)) -> dict:
    return dict(user_settings(c))


@router.patch("/api/settings")
def patch_settings(body: dict, c=Depends(ctx)) -> dict:
    st = user_settings(c)
    unknown = sorted(set(body) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ApiError(400, "bad_setting", f"Unknown setting(s): {', '.join(unknown)}")
    for k, v in body.items():
        if not isinstance(v, bool):
            raise ApiError(400, "bad_setting", f"{k} must be true or false.")
        st[k] = v
    save_settings(c.data_dir if hasattr(c, "data_dir") else c.library.path.parent, st)
    return dict(st)


# ---- helpers ---------------------------------------------------------------------------------------

def fetcher_of(c) -> Fetcher:
    f = getattr(c.settings, "sheets_fetcher", None)
    if f is None:
        f = getattr(c, "_sheets_fetcher", None)
        if f is None:
            f = c._sheets_fetcher = Fetcher()
    return f


def song_dir(c, key: str) -> Path:
    d = c.cache_dir / check_key(key)
    if not (d / "analysis.json").exists():
        raise ApiError(409, "not_analysed", "This song hasn't been analysed yet.",
                       "Analyse it first; the sheet search starts by itself afterwards.")
    return d


def _threads(c) -> dict[str, threading.Thread]:
    t = getattr(c, "sheet_threads", None)
    if t is None:
        t = c.sheet_threads = {}
    return t


def _group_active(d: Path, group: str) -> bool:
    ops = ED.read_ops(d)
    active, _ = ED.active_ops(ops)
    return any((o.get("value") or {}).get("group") == group for o in active)


def public_state(c, key: str, d: Path) -> dict:
    st = L.load_state(d)
    th = _threads(c).get(key)
    changed = False
    if st.get("status") == "searching" and (th is None or not th.is_alive()):
        st["status"] = "idle" if not st.get("results") else "done"     # a search interrupted by a restart
        changed = True
    ap = st.get("applied")
    if ap and not _group_active(d, ap.get("group", "")):
        st["applied"] = None                                              # undone from the edit bar
        changed = True
    if changed:
        L.save_state(d, st)
    q = st.get("query") or {}
    st["query"] = {"title": q.get("title") or "", "artist": q.get("artist") or "", "terms": list(q.get("terms") or [])}
    st.pop("notes", None)
    st.pop("applied_groups", None)
    return st


def start_search(c, key: str, query: str | None = None, force: bool = False) -> dict:
    """Start the search thread for a song (no-op while one is running); returns the state."""
    d = song_dir(c, key)
    threads = _threads(c)
    th = threads.get(key)
    if th is not None and th.is_alive():
        return public_state(c, key, d)
    st = L.load_state(d)
    if st.get("status") == "done" and st.get("results") and not force and not query:
        return public_state(c, key, d)
    with L.state_lock(d):
        st["status"], st["error"] = "searching", None
        if force or query:
            st["results"] = []
        L.save_state(d, st)
    fetcher = fetcher_of(c)

    def run() -> None:
        try:
            L.run_search(d, fetcher, query=query)
        except Exception as e:  # noqa: BLE001 - the state must never stay "searching"
            log.exception("sheet search failed for %s", key)
            with L.state_lock(d):
                st2 = L.load_state(d)
                st2["status"] = "error"
                st2["error"] = {"code": "search_failed", "msg": f"Looking for the sheet failed ({type(e).__name__}).",
                                "hint": "Try again, or paste the chart yourself."}
                L.save_state(d, st2)

    th = threading.Thread(target=run, name=f"nchord-sheets-{key}", daemon=True)
    threads[key] = th
    th.start()
    return public_state(c, key, d)


def maybe_auto_search(c, key: str | None, mode: str = "full") -> None:
    """Called when an analysis job finishes: start the lookup when the setting is on."""
    if not key or mode != "full" or c.settings.is_stub or not user_settings(c).get("sheets_auto_lookup", True):
        return
    try:
        start_search(c, key)
    except ApiError:
        pass
    except Exception:  # noqa: BLE001
        log.exception("auto sheet lookup failed for %s", key)


# ---- routes ----------------------------------------------------------------------------------------

@router.get("/api/tracks/{key}/sheets")
def get_sheets(key: str, c=Depends(ctx)) -> JSONResponse:
    d = song_dir(c, key)
    return JSONResponse(public_state(c, key, d), headers={"Cache-Control": "no-cache"})


class SearchIn(BaseModel):
    query: str | None = None
    force: bool = False


@router.post("/api/tracks/{key}/sheets/search")
def post_search(key: str, body: SearchIn, c=Depends(ctx)) -> JSONResponse:
    if body.query is not None and len(body.query) > 200:
        raise ApiError(400, "bad_request", "The search text is too long.")
    return JSONResponse(start_search(c, key, body.query, body.force))


class AddIn(BaseModel):
    text: str | None = None
    url: str | None = None


@router.post("/api/tracks/{key}/sheets/add")
def post_add(key: str, body: AddIn, c=Depends(ctx)) -> JSONResponse:
    d = song_dir(c, key)
    if (body.text or "").strip():
        if len(body.text) > MAX_PASTE_CHARS:
            raise ApiError(413, "chart_too_long", "That chart is too long.")
        try:
            L.add_pasted(d, body.text)
        except ValueError as e:
            raise ApiError(400, "bad_chart", str(e), "Format: | B . . B+7 | Emaj7 . A7 Em6 | or chords in [brackets] over words.") from None
    elif (body.url or "").strip():
        try:
            L.add_url(d, body.url.strip(), fetcher_of(c))
        except LookupError as e:
            raise ApiError(404, "no_sheet", str(e)) from None
        except SRC.SourceOffline:
            raise ApiError(502, "sheets_unreachable", "The chord-sheet website could not be reached.",
                           "Check the internet connection and try again.") from None
        except ValueError as e:
            raise ApiError(400, "bad_chart", f"The chart on that page could not be used: {e}") from None
    else:
        raise ApiError(400, "bad_chart", "Paste a chart or a link to one first.")
    return JSONResponse(public_state(c, key, d))


class ApplyIn(BaseModel):
    mode: str = "refine"


def _key_at(a: schema.Analysis, t: float) -> int:
    for s in a.key.segments:
        if s.start <= t < s.end:
            return _key_index(s.tonic, s.mode)
    return _key_index(a.key.global_.tonic, a.key.global_.mode)


def _find_result(st: dict, rid: str) -> dict:
    for r in st.get("results") or []:
        if r.get("id") == rid:
            return r
    raise ApiError(404, "no_sheet", "That sheet is no longer in the list.", "Search again, then pick one.")


@router.post("/api/tracks/{key}/sheets/{rid}/apply")
def post_apply(key: str, rid: str, body: ApplyIn, c=Depends(ctx)) -> JSONResponse:
    if body.mode not in ("refine", "sheet"):
        raise ApiError(400, "bad_request", "mode must be 'refine' or 'sheet'.")
    d = song_dir(c, key)
    st = L.load_state(d)
    r = _find_result(st, rid)
    chords = [(float(x["t0"]), float(x["t1"]), x["harte"]) for x in r.get("aligned") or []]
    sections = [(float(s["t"]), s["label"]) for s in r.get("sections") or []]
    if not chords:
        raise ApiError(400, "bad_chart", "This sheet has no chords lined up with the recording.")
    e = ED.edited(d)
    a = e.model
    conf = CONF_BY_VERDICT.get(r.get("verdict"), 0.75)
    uid = uuid.uuid4().hex[:8]
    safe = rid.replace(":", "-")
    group = f"sheet-{safe}-{uid}"
    prefix = f"sheet.{safe}.{uid}"
    summary: dict = {"n_changed": 0, "n_kept": 0, "flagged": [], "mode": body.mode}
    if body.mode == "sheet":
        ops = G.sheet_ops(chords, group, prefix, conf)
    else:
        current = G.main_timeline(a)
        if (d / pipeline.STATE_NAME).exists():
            try:
                with c.redecode_lock:
                    dst = pipeline.load_state(d)
                    ref = G.refine_state(dst, chords, analysis=a)
            except Exception as ex:  # noqa: BLE001 - fall back to the chord-level refine
                log.warning("guided re-decode failed for %s (%s); using the analysis-only refine", key, ex)
                ref = G.refine_from_analysis(a, chords)
        else:
            ref = G.refine_from_analysis(a, chords)
        ops, n_changed, n_kept = G.diff_ops(current, ref.main, group, prefix, 0.9)
        summary.update(n_changed=n_changed, n_kept=n_kept, method=ref.method, n_sheet=ref.n_sheet, n_agree=ref.n_agree)
        trip = [G._triple(h) for _, _, h in chords]
        flagged = []
        for f in ref.flagged:
            k = _key_at(a, f["t0"])
            i = next((i for i, (t0, _, h) in enumerate(chords) if abs(t0 - f["t0"]) < 1e-3), None)
            nxt = next((trip[j][0] for j in range(i + 1, len(trip)) if trip[j][0] >= 0), None) if i is not None else None
            flagged.append({"t0": f["t0"], "t1": f["t1"], "detected": G.full_label(f["detected"], k),
                            "sheet": G.full_label(f["sheet"], k, nxt)})
        summary["flagged"] = flagged
    if body.mode == "sheet":
        summary.update(n_changed=len(ops), n_kept=0)
    ops += G.section_ops(sections, group, prefix)
    earlier = [g for g in st.get("applied_groups") or [] if g != group]
    if st.get("applied") and st["applied"].get("group") not in earlier:
        earlier.append(st["applied"]["group"])
    reverts = [{"id": f"{prefix}.rv{n}", "kind": "revert", "target": {"group": g}, "value": {"group": group}}
               for n, g in enumerate(earlier) if _group_active(d, g)]
    if not ops and not reverts:
        applied = {"id": rid, "group": group, "mode": body.mode, "ts": now_iso()}
        with L.state_lock(d):
            st = L.load_state(d)
            st["applied"] = applied
            L.save_state(d, st)
        return JSONResponse({"analysis": GT.decorate(c, key, e.analysis), "display": e.display, "rev": e.model.edits.rev,
                             "applied": applied, "summary": summary})
    _, _, e2 = ED.post_ops(d, reverts + ops, "sheet")
    applied = {"id": rid, "group": group, "mode": body.mode, "ts": now_iso()}
    with L.state_lock(d):
        st = L.load_state(d)
        st["applied"] = applied
        st["applied_groups"] = [g for g in earlier] + [group]
        L.save_state(d, st)
    return JSONResponse({"analysis": GT.decorate(c, key, e2.analysis), "display": e2.display,
                         "rev": e2.model.edits.rev, "applied": applied, "summary": summary})


@router.delete("/api/tracks/{key}/sheets/applied")
def delete_applied(key: str, c=Depends(ctx)) -> JSONResponse:
    d = song_dir(c, key)
    st = L.load_state(d)
    ap = st.get("applied")
    # every sheet group still in effect goes (a re-apply reverted the previous sheet inside its own
    # group, so reverting only the last group would bring that previous sheet back)
    groups = list(st.get("applied_groups") or [])
    if ap and ap.get("group") not in groups:
        groups.append(ap["group"])
    # all of them, latest first (reverting the last group alone would redo the revert it holds of
    # the one before; a revert of an already reverted group is harmless)
    if groups and any(_group_active(d, g) for g in groups):
        uid = uuid.uuid4().hex[:8]
        ops = [{"id": f"sheet.undo.{uid}.{n}", "kind": "revert", "target": {"group": g}, "value": {}}
               for n, g in enumerate(reversed(groups))]
        _, _, e = ED.post_ops(d, ops, "sheet")
    else:
        e = ED.edited(d)
    with L.state_lock(d):
        st = L.load_state(d)
        st["applied"] = None
        L.save_state(d, st)
    return JSONResponse({"analysis": GT.decorate(c, key, e.analysis), "display": e.display,
                         "rev": e.model.edits.rev, "applied": None})


def audio_for(c, key: str) -> A.Audio:
    """(CLI / tests) the alignment evidence of a song in this server's cache."""
    return A.load_audio(song_dir(c, key))
