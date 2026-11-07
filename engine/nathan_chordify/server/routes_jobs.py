"""Uploads, jobs and SSE progress (DESIGN §11).

    POST   /api/uploads               multipart field "file" → {upload_id, key, filename, size, title, cached}
    POST   /api/jobs                  {source | upload_id, mode, params, force} → {job_id, key, state, cached}
    GET    /api/jobs                  jobs of this server run
    GET    /api/jobs/{id}             one job's snapshot
    GET    /api/jobs/{id}/events      SSE: progress / done / error (the stream ends after done or error)
    DELETE /api/jobs/{id}             cancel
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import uuid
from dataclasses import fields
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from python_multipart.multipart import MultipartParser, parse_options_header

from .. import ingest
from ..ingest.errors import IngestError
from .common import ApiError, ctx

router = APIRouter()

UPLOAD_MAX_BYTES = 1_000_000_000          # same limit as downloads (ingest.youtube.MAX_BYTES)
KEEPALIVE_S = 15.0
UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{16}$")
UPLOAD_INFO = ".upload.json"              # sidecar next to the file: {upload_id, key, filename, size, title, path}
_LINK_RE = re.compile(r"^(https?://|www\.|(m\.|music\.)?youtube\.com/|youtu\.be/)", re.I)
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


# ---- helpers ------------------------------------------------------------------------------------

def safe_filename(name: str) -> str:
    """A Windows-safe file name that keeps the user's title (ingest uses the stem as the title)."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    base = re.sub(r'[<>:"|?*\x00-\x1f]', "_", base).strip(" .") or "upload"
    stem, suffix = os.path.splitext(base)
    stem = stem[:100] or "upload"
    if stem.upper() in _RESERVED:
        stem = "_" + stem
    return stem + suffix[:10].lower()


def validate_params(params: dict | None) -> dict | None:
    """Only `DecoderParams` fields may be overridden; {} counts as the defaults."""
    if not params:
        return None
    from ..types import DecoderParams
    allowed = {f.name for f in fields(DecoderParams)}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ApiError(400, "bad_params", f"Unknown decoder parameter(s): {', '.join(unknown)}")
    return params


def classify_source(source: str) -> tuple[str, str | None]:
    """A link from the browser → (source for the worker, key if known without network).

    Local paths are refused (files come through /api/uploads). Non-YouTube links are resolved by
    the worker, so the server never imports yt-dlp (its .pyd files must stay replaceable).
    """
    s = source.strip().strip('"')
    if not _LINK_RE.match(s):
        raise ApiError(400, "bad_source", "Paste a YouTube (or other site) link, or upload an audio file.",
                       "Files on this computer go through Upload.")
    if not s.lower().startswith(("http://", "https://")):
        s = "https://" + s
    vid = ingest.youtube_id(s)
    if vid:
        return f"https://www.youtube.com/watch?v={vid}", f"yt-{vid}"
    if "deezer." in (urlparse(s).hostname or "").lower():
        raise ApiError(400, "drm", "Deezer links can't be analysed (the audio is DRM-protected).",
                       "Upload the audio file instead.")
    return s, None


class _UploadSink:
    """python-multipart callbacks that stream the "file" part straight to disk."""

    def __init__(self, folder: Path, max_bytes: int):
        self.folder, self.max_bytes = folder, max_bytes
        self.path: Path | None = None
        self.filename: str | None = None
        self.size = 0
        self.too_large = False
        self._fh = None
        self._name = self._value = b""
        self._headers: dict[bytes, bytes] = {}

    def callbacks(self) -> dict:
        return {"on_part_begin": self._begin, "on_header_field": self._field, "on_header_value": self._val,
                "on_header_end": self._header_end, "on_headers_finished": self._headers_done,
                "on_part_data": self._data, "on_part_end": self._end}

    def _begin(self) -> None:
        self._headers = {}

    def _field(self, data: bytes, start: int, end: int) -> None:
        self._name += data[start:end]

    def _val(self, data: bytes, start: int, end: int) -> None:
        self._value += data[start:end]

    def _header_end(self) -> None:
        self._headers[self._name.strip().lower()] = self._value.strip()
        self._name = self._value = b""

    def _headers_done(self) -> None:
        _, opts = parse_options_header(self._headers.get(b"content-disposition", b""))
        fname = opts.get(b"filename")
        if opts.get(b"name") == b"file" and fname is not None and self.path is None:
            self.filename = fname.decode("utf-8", "replace")
            self.path = self.folder / safe_filename(self.filename)
            self._fh = open(self.path, "wb")

    def _data(self, data: bytes, start: int, end: int) -> None:
        if self._fh is None or self.too_large:
            return
        self.size += end - start
        if self.size > self.max_bytes:
            self.too_large = True
            return
        self._fh.write(data[start:end])

    def _end(self) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ---- uploads ------------------------------------------------------------------------------------

@router.post("/api/uploads")
async def upload(request: Request, c=Depends(ctx)) -> dict:
    """Stream a multipart upload to cache/_uploads/<upload_id>/<name>; the key is its sha256."""
    c.check_disk()
    ctype, opts = parse_options_header(request.headers.get("content-type", ""))
    if ctype != b"multipart/form-data" or not opts.get(b"boundary"):
        raise ApiError(415, "bad_upload", "Send the file as multipart/form-data in a field named \"file\".")
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > UPLOAD_MAX_BYTES + 1_000_000:
        raise ApiError(413, "too_large", "The file is larger than 1 GB.", "Use a shorter or more compressed file.")
    upload_id = uuid.uuid4().hex[:16]
    folder = c.uploads_dir / upload_id
    folder.mkdir(parents=True, exist_ok=True)
    sink = _UploadSink(folder, UPLOAD_MAX_BYTES)
    parser = MultipartParser(opts[b"boundary"], sink.callbacks())
    try:
        async for chunk in request.stream():
            parser.write(chunk)
            if sink.too_large:
                raise ApiError(413, "too_large", "The file is larger than 1 GB.",
                               "Use a shorter or more compressed file.")
        parser.finalize()
        if sink.path is None or sink.size == 0:
            raise ApiError(400, "bad_upload", "The upload has no file in a field named \"file\".")
    except ApiError:
        sink.close()
        shutil.rmtree(folder, ignore_errors=True)
        raise
    except Exception as e:  # noqa: BLE001 - malformed multipart, client gone, disk error
        sink.close()
        shutil.rmtree(folder, ignore_errors=True)
        raise ApiError(400, "bad_upload", f"The upload could not be read ({type(e).__name__}).") from e
    finally:
        sink.close()
    try:
        key, meta = await run_in_threadpool(ingest.resolve, str(sink.path))
    except IngestError as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise ApiError(400, e.code, e.message, e.hint) from e
    info = {"upload_id": upload_id, "key": key, "filename": sink.filename, "size": sink.size,
            "title": meta.get("title")}
    (folder / UPLOAD_INFO).write_text(json.dumps({**info, "path": str(sink.path)}, ensure_ascii=False), "utf-8")
    return {**info, "cached": c.has_fresh_analysis(key)}


# ---- jobs ---------------------------------------------------------------------------------------

class JobIn(BaseModel):
    source: str | None = None             # a link (YouTube or another site)
    upload_id: str | None = None          # or a file sent to /api/uploads
    mode: Literal["full", "fast"] = "full"
    params: dict | None = None            # DecoderParams overrides; non-empty → always re-analysed
    force: bool = False                   # ignore a cached analysis


@router.post("/api/jobs")
def create_job(body: JobIn, c=Depends(ctx)) -> dict:
    params = validate_params(body.params)
    title = None
    if body.upload_id:
        source, key, title = c.upload_source(body.upload_id)
    elif body.source:
        source, key = classify_source(body.source)
    else:
        raise ApiError(400, "bad_source", "Send a link (\"source\") or an \"upload_id\".")
    if key and not body.force and params is None and c.has_fresh_analysis(key, body.mode):
        job = c.jobs.add_finished(source, key, title=c.track_title(key), upload_id=body.upload_id)
        c.sync_track(key)
        c.library.save_job(job.snapshot())
        if body.upload_id:
            c.drop_upload(body.upload_id)
        return {"job_id": job.id, "key": key, "state": job.state, "cached": True}
    c.check_disk()
    job = c.jobs.submit(source, key=key, mode=body.mode, params=params, upload_id=body.upload_id, title=title)
    if key:
        c.library.ensure_track(key, title or key, status="analysing",
                               source_type="upload" if body.upload_id else "youtube",
                               source_url=None if body.upload_id else source)
    c.library.save_job(job.snapshot())
    return {"job_id": job.id, "key": job.key, "state": job.state, "cached": False}


@router.get("/api/jobs")
def list_jobs(c=Depends(ctx)) -> dict:
    return {"jobs": c.jobs.list(), "worker": c.jobs.worker_state}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, c=Depends(ctx)) -> dict:
    job = c.jobs.get(job_id)
    if job is None:
        raise ApiError(404, "no_job", "No such job (finished jobs are kept for an hour).")
    return job.snapshot()


@router.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str, c=Depends(ctx)) -> dict:
    job = c.jobs.cancel(job_id)
    if job is None:
        raise ApiError(404, "no_job", "No such job.")
    return job.snapshot()


def _sse(event: tuple[str, dict]) -> str:
    name, data = event
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, c=Depends(ctx)) -> StreamingResponse:
    """Server-sent events. The first event is the job's current state; the stream closes after
    `done` or `error`, so a reconnect simply replays the latest state."""
    got = c.jobs.subscribe(job_id, asyncio.get_running_loop())
    if got is None:
        raise ApiError(404, "no_job", "No such job (finished jobs are kept for an hour).")
    sub, first = got

    async def stream():
        try:
            yield "retry: 3000\n\n"
            event = first
            while True:
                yield _sse(event)
                if event[0] in ("done", "error"):
                    return
                while True:
                    try:
                        event = await asyncio.wait_for(sub.queue.get(), KEEPALIVE_S)
                        break
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            return
                        yield ": keep-alive\n\n"
        finally:
            c.jobs.unsubscribe(job_id, sub)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
