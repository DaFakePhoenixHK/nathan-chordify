"""Job queue (DESIGN §2, §11): FIFO, one GPU worker slot, cancel, crash retry, SSE fan-out.

A pump thread reads the worker's messages, updates the jobs and pushes events to every SSE
subscriber of that job (each subscriber is an asyncio queue on its request's event loop).
`on_finished(job)` (library + job history) runs *before* the terminal event goes out, so a
client that reacts to `done` already sees the updated library.

Events (name, payload):
    progress  {job_id, state, stage, pct, overall, msg, eta_s, position, attempt}
    done      {job_id, key, analysis_url, display_url, audio_url, cached, title}
    error     {job_id, code, msg, hint}            code "cancelled" after DELETE /api/jobs/{id}
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .common import now_iso, track_urls

log = logging.getLogger("nathan_chordify.server.jobs")

ACTIVE = ("queued", "running")
TERMINAL = ("done", "error", "cancelled")
MAX_ATTEMPTS = 2               # a crashed or OOM job is retried once
CANCEL_GRACE_S = 20.0          # then the worker is killed and restarted
KEEP_FINISHED_S = 3600.0       # finished jobs stay queryable for an hour
CRASH_WINDOW_S = 120.0
MAX_CRASHES = 3                # crashes within the window → stop restarting ("failed")

# (start, share) of the whole job per pipeline phase, roughly the §5.14 timing table.
PHASES = {"ingest": (0.00, 0.10), "audio": (0.10, 0.04), "separate": (0.14, 0.46), "rhythm": (0.60, 0.06),
          "evidence": (0.66, 0.20), "decode": (0.86, 0.10), "output": (0.96, 0.04)}

CANCELLED = {"code": "cancelled", "msg": "The analysis was cancelled.", "hint": ""}
WORKER_FAILED = {"code": "worker_failed", "msg": "The analysis engine keeps crashing.",
                 "hint": "See logs\\worker.log, then restart the app."}
SERVER_STOPPED = {"code": "server_stopped", "msg": "The server stopped before this analysis finished.",
                  "hint": "Start the app again and resubmit the song."}
WORKER_CRASHED = {"code": "worker_crashed", "msg": "The analysis engine crashed twice on this song.",
                  "hint": "See logs\\worker.log. Try again, or try a shorter file."}


def phase_of(stage: str, seen: set[str]) -> str | None:
    """Map a pipeline stage to a phase. Ingest's audio decoding and chord decoding share the name "decode"."""
    if stage in ("download", "import"):
        return "ingest"
    if stage == "decode":
        return "decode" if seen & {"separate", "rhythm", "evidence"} else "audio"
    return stage if stage in PHASES else None


class WorkerHandle(Protocol):
    def start(self) -> None: ...
    def send(self, job: dict) -> None: ...
    def recv(self, timeout: float) -> dict | None: ...
    def cancel(self, seq: int) -> None: ...
    def is_alive(self) -> bool: ...
    @property
    def exitcode(self) -> int | None: ...
    def stop(self, timeout: float = 5.0) -> None: ...
    def kill(self) -> None: ...


@dataclass
class Job:
    id: str
    seq: int
    source: str
    mode: str = "full"
    params: dict | None = None
    key: str | None = None
    upload_id: str | None = None
    state: str = "queued"
    stage: str = "queued"
    pct: float = 0.0
    overall: float = 0.0
    msg: str = ""
    eta_s: int | None = None
    position: int = 0
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    error: dict | None = None
    attempts: int = 0
    low_mem: bool = False
    cached: bool = False
    title: str | None = None
    cancel_at: float | None = None
    t0: float = 0.0                       # monotonic start of the current attempt
    seen: set = field(default_factory=set)

    def public_source(self) -> str:
        return f"upload:{self.upload_id}" if self.upload_id else self.source

    def snapshot(self) -> dict:
        d = {"job_id": self.id, "key": self.key, "source": self.public_source(), "mode": self.mode,
             "params": self.params, "state": self.state, "stage": self.stage, "pct": round(self.pct, 3),
             "overall": round(self.overall, 3), "msg": self.msg, "eta_s": self.eta_s, "position": self.position,
             "created": now_iso(self.created), "started": now_iso(self.started) if self.started else None,
             "finished": now_iso(self.finished) if self.finished else None, "error": self.error,
             "attempts": self.attempts, "cached": self.cached, "title": self.title}
        if self.state == "done" and self.key:
            d.update(track_urls(self.key))
        return d

    def progress_event(self) -> tuple[str, dict]:
        return "progress", {"job_id": self.id, "state": self.state, "stage": self.stage, "pct": round(self.pct, 3),
                            "overall": round(self.overall, 3), "msg": self.msg, "eta_s": self.eta_s,
                            "position": self.position, "attempt": self.attempts}

    def current_event(self) -> tuple[str, dict]:
        if self.state == "done":
            return "done", {"job_id": self.id, "key": self.key, **track_urls(self.key or ""), "cached": self.cached,
                            "title": self.title}
        if self.state in ("error", "cancelled"):
            return "error", {"job_id": self.id, **(self.error or CANCELLED)}
        return self.progress_event()


class Subscriber:
    """One SSE connection: events are handed over to its event loop thread-safely."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue()

    def push(self, event: tuple[str, dict]) -> None:
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, event)
        except RuntimeError:          # the request's loop is gone
            pass


class JobManager:
    """Owns the job table, the FIFO queue and the worker; one job runs at a time."""

    def __init__(self, make_worker: Callable[[], WorkerHandle], on_finished: Callable[[Job], None] | None = None):
        self._make_worker = make_worker
        self._on_finished = on_finished or (lambda job: None)
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._running: str | None = None
        self._subs: dict[str, list[Subscriber]] = {}
        self._seq = itertools.count(1)
        self._worker: WorkerHandle | None = None
        self._crashes: deque[float] = deque()
        self._hold = False
        self._halt = threading.Event()
        self._pump_thread: threading.Thread | None = None
        self.worker_state = "stopped"     # stopped | starting | warming | ready | busy | updating | failed
        self.worker_info: dict = {}

    # ---- lifecycle ------------------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            self._spawn_worker()
        self._pump_thread = threading.Thread(target=self._pump, name="nchord-job-pump", daemon=True)
        self._pump_thread.start()

    def close(self) -> None:
        """Stop the pump and the worker; open SSE streams of unfinished jobs get `server_stopped`."""
        self._halt.set()
        if self._pump_thread:
            self._pump_thread.join(5)
        with self._lock:
            w, self._worker = self._worker, None
            self.worker_state = "stopped"
            for job in self._jobs.values():
                if job.state in ACTIVE:
                    self._emit(job, ("error", {"job_id": job.id, **SERVER_STOPPED}))
        if w is not None:
            w.stop()

    def stop_worker(self) -> bool:
        """Stop the worker and hold the queue (update-downloader). False if a job is running."""
        with self._lock:
            if self._running:
                return False
            self._hold = True
            w, self._worker = self._worker, None
            self.worker_state = "updating"
            self._announce_queue()
        if w is not None:
            w.stop()
        return True

    def start_worker(self) -> None:
        """(Re)start the worker after `stop_worker()` or a "failed" state; queued jobs run once it is ready."""
        with self._lock:
            self._hold = False
            self._crashes.clear()
            if self._worker is None:
                self._spawn_worker()
            self._announce_queue()

    def _spawn_worker(self) -> None:
        self._worker = self._make_worker()
        self.worker_state = "starting"
        self._worker.start()

    # ---- public API -----------------------------------------------------------------------------
    def submit(self, source: str, *, key: str | None = None, mode: str = "full", params: dict | None = None,
               upload_id: str | None = None, title: str | None = None) -> Job:
        """Queue a job; an identical active job for the same key is returned instead of a duplicate."""
        with self._lock:
            self._prune()
            if key:
                for j in self._jobs.values():
                    if j.key == key and j.state in ACTIVE and j.mode == mode and j.params == params:
                        return j
            job = Job(id=uuid.uuid4().hex[:12], seq=next(self._seq), source=source, mode=mode, params=params,
                      key=key, upload_id=upload_id, title=title)
            self._jobs[job.id] = job
            self._queue.append(job.id)
            if self.worker_state == "failed":
                self._fail(job, WORKER_FAILED)
            else:
                self._announce_queue()
                self._dispatch()
            return job

    def add_finished(self, source: str, key: str, title: str | None = None, upload_id: str | None = None) -> Job:
        """A job that is already done (the song's analysis is cached)."""
        with self._lock:
            self._prune()
            now = time.time()
            job = Job(id=uuid.uuid4().hex[:12], seq=next(self._seq), source=source, key=key, upload_id=upload_id,
                      state="done", stage="done", pct=1.0, overall=1.0, msg="Already analysed", cached=True,
                      title=title, started=now, finished=now)
            self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            return [j.snapshot() for j in sorted(self._jobs.values(), key=lambda j: j.created)]

    def active_keys(self) -> set[str]:
        return set(self.active_jobs_by_key())

    def active_jobs_by_key(self) -> dict[str, str]:
        with self._lock:
            return {j.key: j.id for j in self._jobs.values() if j.state in ACTIVE and j.key}

    def active_uploads(self) -> set[str]:
        with self._lock:
            return {j.upload_id for j in self._jobs.values() if j.state in ACTIVE and j.upload_id}

    def queue_length(self) -> int:
        with self._lock:
            return len(self._queue) + (1 if self._running else 0)

    def cancel(self, job_id: str) -> Job | None:
        """A queued job ends at once; a running one stops at its next progress call (or the worker
        is restarted after CANCEL_GRACE_S)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in TERMINAL:
                return job
            if job.state == "queued":
                if job_id in self._queue:
                    self._queue.remove(job_id)
                job.state = "cancelled"
                self._finish(job, CANCELLED)
                self._announce_queue()
            elif job.cancel_at is None:
                job.cancel_at = time.monotonic()
                job.msg = "Cancelling"
                if self._worker is not None:
                    self._worker.cancel(job.seq)
                self._emit(job, job.progress_event())
            return job

    def subscribe(self, job_id: str, loop: asyncio.AbstractEventLoop) -> tuple[Subscriber, tuple[str, dict]] | None:
        """Register an SSE subscriber; returns it with the job's current event (sent first)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            sub = Subscriber(loop)
            self._subs.setdefault(job_id, []).append(sub)
            return sub, job.current_event()

    def unsubscribe(self, job_id: str, sub: Subscriber) -> None:
        with self._lock:
            subs = self._subs.get(job_id, [])
            if sub in subs:
                subs.remove(sub)
            if not subs:
                self._subs.pop(job_id, None)

    # ---- pump -----------------------------------------------------------------------------------
    def _pump(self) -> None:
        while not self._halt.is_set():
            w = self._worker
            if w is None:
                self._halt.wait(0.2)
                continue
            msg = w.recv(0.2)
            with self._lock:
                if w is not self._worker:
                    continue                      # replaced meanwhile (cancel timeout, update, crash)
                try:
                    if msg is not None:
                        self._handle(msg)
                    elif not w.is_alive():
                        self._on_crash(w)
                    else:
                        self._check_cancel_timeout()
                    self._dispatch()
                except Exception:  # noqa: BLE001 - the pump must survive anything
                    log.exception("job pump error")

    def _handle(self, msg: dict) -> None:
        t = msg.get("t")
        if t == "state":
            self.worker_state = msg.get("state", self.worker_state)
            self._announce_queue()
            return
        if t == "ready":
            self.worker_state = "ready"
            self.worker_info = msg.get("info") or {}
            self._crashes.clear()
            log.info("GPU worker ready: %s", self.worker_info)
            return
        job = self._jobs.get(msg.get("job", ""))
        if job is None or job.state in TERMINAL:
            return
        if t == "started":
            job.state, job.stage, job.msg = "running", "starting", "Starting"
            job.started = job.started or time.time()
            job.t0 = time.monotonic()
            self._emit(job, job.progress_event())
        elif t == "progress":
            self._apply_progress(job, msg.get("stage", ""), float(msg.get("pct", 0.0)), msg.get("msg", ""))
            self._emit(job, job.progress_event())
        elif t == "done":
            job.key = msg.get("key") or job.key
            job.title = msg.get("title") or job.title
            job.state, job.stage, job.pct, job.overall, job.eta_s = "done", "done", 1.0, 1.0, 0
            job.msg = f"{msg.get('n_chords', 0)} chords"
            self._finish(job, None)
        elif t == "cancelled":
            job.state = "cancelled"
            self._finish(job, CANCELLED)
        elif t == "error":
            err = {"code": msg.get("code", "internal"), "msg": msg.get("msg", "The analysis failed."),
                   "hint": msg.get("hint", "")}
            if msg.get("retry") and job.attempts < MAX_ATTEMPTS and job.cancel_at is None:
                log.warning("job %s: %s; restarting the worker and retrying", job.id, err["code"])
                self._running = None
                self._replace_worker()
                self._requeue(job, "The GPU ran out of memory; retrying with a lower-memory setting")
            else:
                job.state = "error"
                self._finish(job, err)

    def _apply_progress(self, job: Job, stage: str, pct: float, msg: str) -> None:
        ph = phase_of(stage, job.seen)
        job.seen.add(stage)
        if ph:
            start, share = PHASES[ph]
            job.overall = max(job.overall, min(0.99, start + share * min(max(pct, 0.0), 1.0)))
        job.stage, job.pct, job.msg = stage, pct, msg
        elapsed = time.monotonic() - job.t0
        ov = job.overall
        job.eta_s = int(round(elapsed * (1 - ov) / ov)) if ov >= 0.05 and elapsed >= 2.0 else None

    def _finish(self, job: Job, error: dict | None) -> None:
        job.finished = time.time()
        if error is not None:
            job.error = dict(error)
        if self._running == job.id:
            self._running = None
            if self._worker is not None and self.worker_state == "busy":
                self.worker_state = "ready"
        try:
            self._on_finished(job)
        except Exception:  # noqa: BLE001 - bookkeeping must not block the event
            log.exception("on_finished failed for job %s", job.id)
        self._emit(job, job.current_event())

    def _fail(self, job: Job, error: dict) -> None:
        if job.id in self._queue:
            self._queue.remove(job.id)
        job.state = "error"
        self._finish(job, error)

    def _requeue(self, job: Job, msg: str) -> None:
        job.state, job.stage, job.msg, job.low_mem = "queued", "queued", msg, True
        job.pct = job.overall = 0.0
        job.eta_s = None
        job.seen = set()
        self._queue.appendleft(job.id)
        self._emit(job, job.progress_event())

    def _replace_worker(self) -> None:
        old, self._worker = self._worker, None
        if old is not None:
            old.kill()
        self._spawn_worker()

    def _on_crash(self, w: WorkerHandle) -> None:
        log.error("GPU worker exited unexpectedly (exit code %s)", w.exitcode)
        now = time.monotonic()
        self._crashes.append(now)
        while self._crashes and now - self._crashes[0] > CRASH_WINDOW_S:
            self._crashes.popleft()
        job = self._jobs.get(self._running) if self._running else None
        self._running = None
        if len(self._crashes) >= MAX_CRASHES:
            self._worker = None
            w.kill()
            self.worker_state = "failed"
            if job is not None:
                self._fail(job, WORKER_FAILED)
            for jid in list(self._queue):
                self._fail(self._jobs[jid], WORKER_FAILED)
            return
        self._replace_worker()
        if job is None:
            return
        if job.cancel_at is not None:
            job.state = "cancelled"
            self._finish(job, CANCELLED)
        elif job.attempts < MAX_ATTEMPTS:
            self._requeue(job, "The analysis engine stopped unexpectedly; retrying once")
        else:
            job.state = "error"
            self._finish(job, WORKER_CRASHED)

    def _check_cancel_timeout(self) -> None:
        job = self._jobs.get(self._running) if self._running else None
        if job is None or job.cancel_at is None or time.monotonic() - job.cancel_at < CANCEL_GRACE_S:
            return
        log.warning("job %s did not stop within %.0f s; restarting the worker", job.id, CANCEL_GRACE_S)
        self._running = None
        self._replace_worker()
        job.state = "cancelled"
        self._finish(job, CANCELLED)

    def _dispatch(self) -> None:
        if self.worker_state == "failed":
            for jid in list(self._queue):
                self._fail(self._jobs[jid], WORKER_FAILED)
            return
        if self._hold or self._running or self._worker is None or self.worker_state != "ready":
            return
        while self._queue:
            job = self._jobs.get(self._queue.popleft())
            if job is None or job.state != "queued":
                continue
            job.attempts += 1
            job.stage, job.msg, job.position = "starting", "Starting", 0
            self._running = job.id
            self.worker_state = "busy"
            self._worker.send({"job": job.id, "seq": job.seq, "source": job.source, "mode": job.mode,
                               "params": job.params, "low_mem": job.low_mem})
            self._emit(job, job.progress_event())
            self._announce_queue()
            return

    def _announce_queue(self) -> None:
        """Tell queued jobs why they wait (engine warming up, update, or jobs ahead)."""
        ahead = 1 if self._running else 0
        for i, jid in enumerate(self._queue):
            job = self._jobs.get(jid)
            if job is None or job.state != "queued":
                continue
            pos = ahead + i
            if self.worker_state in ("starting", "warming"):
                stage, msg = "warming", "Warming up the analysis engine"
            elif self.worker_state == "updating":
                stage, msg = "queued", "Waiting for the downloader update"
            else:
                stage, msg = "queued", (f"Waiting for {pos} job{'s' if pos != 1 else ''} ahead" if pos else "Queued")
            if (job.stage, job.msg, job.position) != (stage, msg, pos):
                job.stage, job.msg, job.position = stage, msg, pos
                self._emit(job, job.progress_event())

    def _emit(self, job: Job, event: tuple[str, dict]) -> None:
        for sub in list(self._subs.get(job.id, [])):
            sub.push(event)

    def _prune(self) -> None:
        now = time.time()
        old = [jid for jid, j in self._jobs.items()
               if j.state in TERMINAL and j.finished and now - j.finished > KEEP_FINISHED_S and jid not in self._subs]
        for jid in old:
            del self._jobs[jid]
