"""FastAPI app factory (DESIGN §2, §10, §11).

    from nathan_chordify.server.app import create_app
    app = create_app()                       # real GPU worker, pre-warmed in the lifespan

The lifespan opens data/library.sqlite, re-syncs it with cache/, spawns the GPU worker (which
pre-warms in the background; jobs wait for it) and runs the storage clean-up. web/dist is served
at "/" with an SPA fallback whenever it exists (checked per request, so a later build just works).
"""
from __future__ import annotations

import json
import logging
import mimetypes
import shutil
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .. import ENGINE_VERSION, config
from . import (edits, gt, routes_export, routes_jobs, routes_later, routes_sheets, routes_system, routes_tracks,
               storage)
from .common import ApiError, KEY_RE, now_iso
from .db import Library
from .jobs import Job, JobManager
from .worker import DEFAULT_ANALYZE, ProcessWorker, ThreadWorker, load_target

log = logging.getLogger("nathan_chordify.server")

STUB_ANALYZE = "nathan_chordify.server.stub:analyze"

# §10, plus inline styles (React style props, injected CSS) and data: fonts.
CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; "
       "media-src 'self' blob: data:; img-src 'self' blob: data:; connect-src 'self'; "
       "style-src 'self' 'unsafe-inline'; font-src 'self' data:; manifest-src 'self'; "
       "object-src 'none'; base-uri 'self'; frame-ancestors 'self'")

# Windows takes MIME types from the registry, where .js is sometimes text/plain (module scripts then fail).
for _ext, _type in ((".js", "text/javascript"), (".mjs", "text/javascript"), (".css", "text/css"),
                    (".svg", "image/svg+xml"), (".wasm", "application/wasm"), (".json", "application/json"),
                    (".webmanifest", "application/manifest+json"), (".m4a", "audio/mp4")):
    mimetypes.add_type(_type, _ext)


@dataclass
class ServerSettings:
    worker: str = "process"                       # "process" (spawned GPU worker) | "thread" (tests)
    analyze: str = DEFAULT_ANALYZE                # "module:function" run by the worker
    analyze_fn: Callable | None = None            # thread worker only
    prewarm: bool = True
    device: str = "cuda"
    web_dist: Path = field(default_factory=lambda: config.ROOT / "web" / "dist")
    policy: storage.Policy = field(default_factory=storage.Policy)
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "[::1]", "*.ts.net")
    pip_cmd: list[str] | None = None              # tests replace the real pip call
    storage_on_start: bool = True
    sheets_auto_lookup: bool = True               # default of the user setting (data/settings.json overrides)
    sheets_fetcher: object | None = None          # tests pass a fake sheets.fetch.Fetcher (no network)

    @property
    def is_stub(self) -> bool:
        return self.analyze == STUB_ANALYZE or self.analyze_fn is not None


class ServerContext:
    """Everything the routes share: library, job manager, storage, settings."""

    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.cache_dir: Path = config.CACHE_DIR
        self.eval_dir: Path = config.EVAL_DIR             # ground-truth export (eval/gt, eval/songs.yaml)
        self.data_dir: Path = config.DATA_DIR
        self.uploads_dir = self.cache_dir / storage.UPLOADS
        self.library = Library(config.DATA_DIR / "library.sqlite")
        # user settings (GET/PATCH /api/settings): defaults from ServerSettings, saved in data/settings.json
        self.user_settings: dict = {**routes_sheets.load_settings(self.data_dir)}
        if not (self.data_dir / routes_sheets.SETTINGS_FILE).exists():
            self.user_settings["sheets_auto_lookup"] = bool(settings.sheets_auto_lookup)
        self.sheet_threads: dict[str, threading.Thread] = {}   # one sheet search per song at a time
        self.jobs = JobManager(self._make_worker, on_finished=self.on_job_finished)
        self.update_lock = threading.Lock()
        self.redecode_lock = threading.Lock()
        self._storage_lock = threading.Lock()
        self._storage_thread: threading.Thread | None = None
        self.last_cleanup: dict | None = None

    def _make_worker(self):
        s = self.settings
        if s.worker == "thread":
            return ThreadWorker(s.analyze_fn or load_target(s.analyze))
        return ProcessWorker({"analyze": s.analyze, "prewarm": s.prewarm, "device": s.device,
                              "cache_dir": str(self.cache_dir), "logs_dir": str(config.LOGS_DIR)})

    # ---- lifecycle ------------------------------------------------------------------------------
    def start(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.library.mark_interrupted_jobs()
        self.reconcile()
        self.jobs.start()
        if self.settings.storage_on_start:
            self.schedule_storage()

    def stop(self) -> None:
        self.jobs.close()
        if self._storage_thread is not None:
            self._storage_thread.join(30)
        self.library.close()

    # ---- library <-> cache ----------------------------------------------------------------------
    def reconcile(self) -> None:
        """Add analysed songs missing from the library; drop rows whose folder is gone."""
        rows = {r["key"]: r for r in self.library.list()}
        for d in storage.song_dirs(self.cache_dir):
            if not KEY_RE.match(d.name):
                continue
            has = (d / "analysis.json").exists()
            row = rows.get(d.name)
            if has and (row is None or row["status"] != "ready"):
                self.sync_track(d.name)
            elif not has and row is not None and row["status"] == "analysing":
                self.library.set_status(d.name, "failed")
        for key in rows:
            if not (self.cache_dir / key).exists():
                self.library.delete(key)

    def sync_track(self, key: str, status: str = "ready") -> None:
        p = self.cache_dir / key / "analysis.json"
        try:
            src = json.loads(p.read_text("utf-8"))["source"]
            created = now_iso(p.stat().st_mtime)
        except (OSError, ValueError, KeyError):
            return
        self.library.upsert_track(key, title=src.get("title") or key, artist=src.get("artist"),
                                  duration=src.get("duration"), source_type=src.get("type"),
                                  source_url=src.get("url"), status=status, created=created)

    def track_title(self, key: str) -> str | None:
        row = self.library.get(key)
        return row["title"] if row else None

    def has_fresh_analysis(self, key: str, mode: str = "full") -> bool:
        """analysis.json exists, from this engine version and in this mode (default params only)."""
        p = self.cache_dir / key / "analysis.json"
        if not p.exists():
            return False
        try:
            a = json.loads(p.read_text("utf-8"))
        except (OSError, ValueError):
            return False
        return a.get("engine_version") == ENGINE_VERSION and a.get("mode") == mode

    # ---- uploads --------------------------------------------------------------------------------
    def upload_source(self, upload_id: str) -> tuple[str, str, str]:
        """(path for the worker, key, title) of a finished upload."""
        if not routes_jobs.UPLOAD_ID_RE.match(upload_id or ""):
            raise ApiError(400, "bad_upload", "Not an upload id.")
        try:
            info = json.loads((self.uploads_dir / upload_id / routes_jobs.UPLOAD_INFO).read_text("utf-8"))
            path = Path(info["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
        except (OSError, ValueError, KeyError) as e:
            raise ApiError(404, "no_upload", "This upload is gone (uploads are kept for a day).",
                           "Upload the file again.") from e
        return str(path), info["key"], info.get("title") or path.stem

    def drop_upload(self, upload_id: str) -> None:
        if routes_jobs.UPLOAD_ID_RE.match(upload_id or ""):
            shutil.rmtree(self.uploads_dir / upload_id, ignore_errors=True)

    # ---- jobs -----------------------------------------------------------------------------------
    def on_job_finished(self, job: Job) -> None:
        """Pump-thread callback: library status, job history, upload clean-up, storage limits."""
        self.library.save_job(job.snapshot())
        if job.key:
            if job.state == "done":
                self.sync_track(job.key)
            elif (self.cache_dir / job.key / "analysis.json").exists():
                self.sync_track(job.key)
            elif job.state == "cancelled":
                self.library.delete(job.key)
            else:
                self.library.set_status(job.key, "failed")
        if job.upload_id and job.state == "done":
            self.drop_upload(job.upload_id)
        self.schedule_storage()
        if job.state == "done" and not job.cached:
            routes_sheets.maybe_auto_search(self, job.key, job.mode)   # "Find sheet" by itself (setting)

    # ---- storage --------------------------------------------------------------------------------
    def check_disk(self) -> None:
        d = storage.disk_status(self.cache_dir, self.settings.policy)
        if d["refuse"]:
            raise ApiError(507, "disk_full",
                           f"Only {d['free_bytes'] / storage.GB:.1f} GB free on the cache drive; a new analysis "
                           f"needs at least {d['refuse_below_bytes'] / storage.GB:.0f} GB.",
                           "Free some space (Settings > Storage > Clean up), then retry.")

    def run_storage(self) -> dict:
        with self._storage_lock:
            rep = storage.enforce(self.cache_dir, self.library.last_opened(), self.settings.policy,
                                  protect=self.jobs.active_keys(), protect_uploads=self.jobs.active_uploads())
            self.last_cleanup = {**rep.to_dict(), "at": now_iso()}
        if rep.freed:
            log.info("storage: freed %.2f GB (%d items)", rep.freed / storage.GB, len(rep.removed))
        return self.last_cleanup

    def schedule_storage(self) -> None:
        """Run the clean-up in the background (startup, after each job); skipped if one is running."""
        if self._storage_thread is not None and self._storage_thread.is_alive():
            return

        def run():
            try:
                self.run_storage()
            except Exception:  # noqa: BLE001
                log.exception("storage clean-up failed")
        self._storage_thread = threading.Thread(target=run, name="nchord-storage", daemon=True)
        self._storage_thread.start()


class SecurityHeaders:
    """Pure ASGI middleware (doesn't buffer SSE or audio): CSP and a few safe defaults."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                h = MutableHeaders(scope=message)
                h.setdefault("Content-Security-Policy", CSP)
                h.setdefault("X-Content-Type-Options", "nosniff")
                h.setdefault("Referrer-Policy", "no-referrer")
            await send(message)

        await self.app(scope, receive, send_with_headers)


NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>Nathan Chordify</title>
<body style="font-family:system-ui;margin:2rem"><h1>Nathan Chordify server is running</h1>
<p>The web UI isn't built yet: run <code>npm run build</code> in <code>web\\</code>.
The API is at <a href="/api/docs">/api/docs</a>.</p></body>"""


def _mount_web(app: FastAPI, settings: ServerSettings) -> None:
    @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def spa(path: str):
        if path == "api" or path.startswith("api/"):
            raise ApiError(404, "not_found", f"No API route /{path}")
        dist = settings.web_dist.resolve()
        index = dist / "index.html"
        if not index.exists():
            return HTMLResponse(NOT_BUILT)
        if path:
            f = (dist / path).resolve()
            if f.is_file() and f.is_relative_to(dist):
                immutable = path.startswith("assets/")       # Vite's content-hashed files
                cache = "public, max-age=31536000, immutable" if immutable else "no-cache"
                return FileResponse(f, headers={"Cache-Control": cache})
            if "." in path.rsplit("/", 1)[-1]:
                raise ApiError(404, "not_found", f"No file /{path}")
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = ServerContext(settings)
        app.state.ctx = c
        c.start()
        log.info("server ready (worker: %s, cache: %s)", settings.analyze, c.cache_dir)
        try:
            yield
        finally:
            c.stop()

    app = FastAPI(title="Nathan Chordify", version=ENGINE_VERSION, lifespan=lifespan,
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.state.settings = settings

    @app.exception_handler(ApiError)
    async def api_error(request: Request, e: ApiError):
        return JSONResponse(e.to_dict(), status_code=e.status)

    for r in (routes_system, routes_jobs, routes_tracks, routes_export, edits, gt, routes_sheets, routes_later):
        app.include_router(r.router)
    _mount_web(app, settings)
    app.add_middleware(SecurityHeaders)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    return app
