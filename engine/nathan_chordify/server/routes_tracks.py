"""Library and track routes (DESIGN §11).

    GET    /api/tracks?q=                  {tracks: [...]}  (chords_subset filter: M7)
    GET    /api/tracks/{key}               analysis with the edits applied (?original=1: as analysed);
                                           user renames and the GT status applied; marks the song opened
    GET    /api/tracks/{key}/display       display.json of the edited analysis (?original=1: as analysed)
    GET    /api/tracks/{key}/audio         play.m4a, audio/mp4, HTTP Range; ?mix=original|novocals|harmony
                                           &transpose=-6..6 (the mix moved to another key, same speed)
    GET    /api/tracks/{key}/mixes         which practice mixes exist / can be made (server/mixes.py)
                                           + which shifted copies are already made
    PATCH  /api/tracks/{key}               {title?, artist?}
    DELETE /api/tracks/{key}               the song and its whole cache folder
    POST   /api/tracks/{key}/redecode      {params?, region?, write?} → {analysis, display}; with a region
                                           (a re-scan) the new chords are kept as pending suggestions and
                                           the edited song (with them) comes back
    POST   /api/tracks/{key}/reanalyze     full re-run from the cached audio → {job_id, key}
"""
from __future__ import annotations

import json
import shutil
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import edits as ED
from . import gt as GT
from . import mixes as MIX
from . import storage
from .common import ApiError, check_key, ctx, not_implemented, track_urls
from .routes_jobs import validate_params

router = APIRouter()


def _analysis_path(c, key: str):
    p = c.cache_dir / check_key(key) / "analysis.json"
    if not p.exists():
        if key in c.jobs.active_keys():
            raise ApiError(404, "not_ready", "This song is still being analysed.")
        raise ApiError(404, "not_found", "No analysis for this song.")
    return p


@router.get("/api/tracks")
def list_tracks(q: str | None = None, chords_subset: str | None = None, level: str | None = None,
                c=Depends(ctx)) -> dict:
    # TODO(M7): chords_subset / level filter (songs whose chords ⊆ the given set, §11).
    active = c.jobs.active_jobs_by_key()
    rows = c.library.list(q)
    for r in rows:
        r.update(track_urls(r["key"]))
        r["job_id"] = active.get(r["key"])
    return {"tracks": rows}


@router.get("/api/tracks/{key}")
def get_track(key: str, original: bool = False, c=Depends(ctx)) -> JSONResponse:
    p = _analysis_path(c, key)
    if c.library.get(key) is None:
        c.sync_track(key)
    data = json.loads(p.read_text("utf-8")) if original else ED.edited(p.parent).analysis
    data = GT.decorate(c, key, data)
    c.library.touch(key)
    return JSONResponse(data, headers={"Cache-Control": "no-cache"})


@router.get("/api/tracks/{key}/display")
def get_display(key: str, original: bool = False, c=Depends(ctx)):
    p = c.cache_dir / check_key(key) / "display.json"
    if not p.exists():
        raise ApiError(404, "not_found", "No display file for this song.")
    if original or not (p.parent / "analysis.json").exists():
        return FileResponse(p, media_type="application/json", headers={"Cache-Control": "no-cache"})
    return JSONResponse(ED.edited(p.parent).display, headers={"Cache-Control": "no-cache"})


def _parse_transpose(value: str) -> int:
    try:
        n = int(str(value).strip() or "0")
    except ValueError:
        n = None
    if n is None or not MIX.TRANSPOSE_MIN <= n <= MIX.TRANSPOSE_MAX:
        raise ApiError(400, "bad_transpose", f"Transpose must be a whole number from {MIX.TRANSPOSE_MIN} to "
                       f"{MIX.TRANSPOSE_MAX} semitones, not {value!r}.",
                       "A bigger move is the same key an octave away: use the small number.")
    return n


@router.api_route("/api/tracks/{key}/audio", methods=["GET", "HEAD"])
def get_audio(key: str, mix: str = "original", transpose: str = "0", c=Depends(ctx)) -> FileResponse:
    """The player file with full Range support (Starlette's FileResponse: 206, multi-range, 416).

    ``mix=original|novocals|harmony`` (server/mixes.py): the practice mixes are made from the cached
    stems on the first request (a few seconds; HEAD waits for them too) and cached as
    ``play_<mix>.m4a``. 409 ``no_stems`` when the song was analysed without separation.

    ``transpose=-6..6`` (0 = as recorded): the mix moved by that many semitones at the same speed,
    made on the first request (~15 s for a 5-minute song; HEAD waits) and cached as
    ``play[_<mix>]_t<±n>.m4a``; a song keeps at most 12 of them. 400 ``bad_transpose`` outside the
    range, 500 ``shift_failed`` when ffmpeg could not do it.
    """
    if mix not in MIX.MIXES:
        raise ApiError(400, "bad_mix", f"Unknown mix {mix!r}.", "Use original, novocals or harmony.")
    n = _parse_transpose(transpose)
    d = c.cache_dir / check_key(key)
    if not (d / "play.m4a").exists():
        raise ApiError(404, "not_found", "No audio for this song yet.")
    try:
        p = MIX.ensure(d, mix, n)
    except MIX.NoStems:
        raise ApiError(409, "no_stems", MIX.NO_STEMS_MSG, MIX.NO_STEMS_HINT) from None
    except MIX.MixError as e:
        raise ApiError(500, "mix_failed", str(e), "Try again; if it keeps failing, re-analyse the song.") from None
    except MIX.ShiftError as e:
        raise ApiError(500, "shift_failed", f"The audio could not be moved to the new key. {e}",
                       "Try again, or switch the Audio choice back to Original.") from None
    return FileResponse(p, media_type="audio/mp4", headers={"Cache-Control": "no-cache"})


@router.get("/api/tracks/{key}/mixes")
def get_mixes(key: str, c=Depends(ctx)) -> dict:
    """{mixes: {original|novocals|harmony: ready|makeable|no_stems|missing}, transposes_ready: {mix: [n, …]}}
    for the Sound selector and the Audio (original / transposed) choice."""
    d = c.cache_dir / check_key(key)
    if not d.exists():
        raise ApiError(404, "not_found", "No such song.")
    return {"mixes": MIX.status(d), "transposes_ready": MIX.transposes_ready(d)}


class TrackPatch(BaseModel):
    title: str | None = None
    artist: str | None = None


@router.patch("/api/tracks/{key}")
def patch_track(key: str, body: TrackPatch, c=Depends(ctx)) -> dict:
    check_key(key)
    if c.library.get(key) is None:
        raise ApiError(404, "not_found", "No such song in the library.")
    title = body.title.strip() if body.title is not None else None
    if title == "":
        raise ApiError(400, "bad_title", "The title can't be empty.")
    artist = body.artist.strip() if body.artist is not None else None
    row = c.library.rename(key, title, artist)
    return {**row, **track_urls(key)}


@router.delete("/api/tracks/{key}")
def delete_track(key: str, c=Depends(ctx)) -> dict:
    """Move the folder to cache/_trash first: on Windows that fails cleanly if a file is in use."""
    check_key(key)
    if key in c.jobs.active_keys():
        raise ApiError(409, "busy", "This song is being analysed.", "Cancel the analysis first.")
    d = c.cache_dir / key
    if d.exists():
        trash = c.cache_dir / storage.TRASH
        trash.mkdir(exist_ok=True)
        target = trash / f"{key}-{uuid.uuid4().hex[:8]}"
        try:
            d.rename(target)
        except OSError as e:
            raise ApiError(409, "in_use", "The song's files are in use.", "Stop playback, then try again.") from e
        shutil.rmtree(target, ignore_errors=True)
    elif c.library.get(key) is None:
        raise ApiError(404, "not_found", "No such song.")
    c.library.delete(key)
    return {"deleted": key}


class RedecodeIn(BaseModel):
    params: dict | None = None            # DecoderParams overrides
    region: tuple[float, float] | None = None   # (t0, t1) seconds: "re-scan this section (more sensitive)"
    write: bool = False                   # False = preview; True replaces analysis.json / display.json


@router.post("/api/tracks/{key}/redecode")
def redecode(key: str, body: RedecodeIn, c=Depends(ctx)) -> JSONResponse:
    """Fast re-decode from the cached decode state (§5.11) → {analysis, display}.

    CPU-only NumPy, so it runs in the server's thread pool instead of queueing behind a GPU job.
    Chord ids change on a re-decode, so the matching display (labels for every transposition)
    comes with it.
    """
    _analysis_path(c, key)
    from .. import pipeline
    fn = getattr(pipeline, "redecode", None)
    if fn is None:
        raise not_implemented("Re-decode", "pipeline.redecode (WP8)")
    params = validate_params(body.params)
    if body.region is not None and not 0 <= body.region[0] < body.region[1]:
        raise ApiError(400, "bad_region", "The region must be (start, end) in seconds with start < end.")
    try:
        with c.redecode_lock:
            result = fn(key, params, body.region, **({"write": True} if body.write else {}))
    except FileNotFoundError as e:
        raise ApiError(409, "no_decode_state", "This song was analysed before re-scanning was possible.",
                       "Re-analyse it once (the cached stems are reused), then try again.") from e
    d = c.cache_dir / key
    if body.region is not None and not body.write:
        # "Re-scan this section": keep the new chords as pending suggestions (accept / reject are
        # edit ops) and answer with the edited song, so earlier edits stay in view.
        base = ED.load_base(d)
        new = ED.store_suggestions(d, base, result, (float(body.region[0]), float(body.region[1])))
        e = ED.edited(d)
        return JSONResponse({"analysis": GT.decorate(c, key, e.analysis), "display": e.display,
                             "rev": e.model.edits.rev, "n_suggestions": len(new)})
    if body.write:
        c.sync_track(key)
        e = ED.edited(d)
        return JSONResponse({"analysis": GT.decorate(c, key, e.analysis), "display": e.display, "rev": e.model.edits.rev})
    e = ED.edited(d, base=result)                  # a preview with other params: edits still apply
    return JSONResponse({"analysis": GT.decorate(c, key, e.analysis), "display": e.display, "rev": e.model.edits.rev})


@router.post("/api/tracks/{key}/reanalyze")
def reanalyze(key: str, c=Depends(ctx)) -> dict:
    """Full re-run from the cached audio (no cache shortcut)."""
    check_key(key)
    d = c.cache_dir / key
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise ApiError(404, "not_found", "No cached audio for this song.")
    meta = json.loads(meta_path.read_text("utf-8"))
    if meta.get("source_type") == "upload":
        src = next((p for p in d.glob("source.*")), None)
        if src is None:
            raise ApiError(404, "not_found", "The uploaded file is no longer in the cache.", "Upload it again.")
        source = str(src)
    else:
        source = meta.get("source_url") or ""
    c.check_disk()
    job = c.jobs.submit(source, key=key, mode="full", title=c.track_title(key))
    c.library.ensure_track(key, c.track_title(key) or key, status="analysing")
    c.library.save_job(job.snapshot())
    return {"job_id": job.id, "key": key, "state": job.state, "cached": False}
