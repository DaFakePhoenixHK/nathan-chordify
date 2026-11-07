"""The GPU worker (DESIGN §2, §9): one spawned process, pre-warmed, one job at a time.

Child side: `main()` is the `multiprocessing` spawn target. It imports torch and the models once
(pre-warm), then runs jobs from the job queue through `pipeline.analyze` and reports back:

    {"t": "state", "state": "warming"}                 pre-warm started
    {"t": "ready", "info": {torch, cuda, gpu, ...}}    waiting for jobs
    {"t": "started", "job": id}
    {"t": "progress", "job": id, "stage": s, "pct": 0..1, "msg": text}
    {"t": "done", "job": id, "key": k, "title": ..., "artist": ..., "duration": ..., "n_chords": n, "runtime_s": {...}}
    {"t": "error", "job": id, "code": c, "msg": m, "hint": h, "retry": bool}    retry: CUDA OOM
    {"t": "cancelled", "job": id}

Server side: `ProcessWorker` (the real process) and `ThreadWorker` (tests: same protocol in a
thread) share the handle interface used by `jobs.JobManager`.

This module must stay cheap to import: the child imports it before anything else.
"""
from __future__ import annotations

import atexit
import importlib
import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger("nathan_chordify.server.worker")

DEFAULT_ANALYZE = "nathan_chordify.pipeline:analyze"
PROGRESS_EVERY_S = 0.25


class JobCancelled(BaseException):
    """Raised from the progress callback; BaseException so `except Exception` in the engine can't eat it."""


# ---------------------------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------------------------

def main(jobs, msgs, cancel, opts: dict) -> None:
    """Spawn target. `opts`: analyze ("module:function"), prewarm (bool), device, cache_dir, logs_dir."""
    from .. import config
    if opts.get("cache_dir"):
        config.CACHE_DIR = Path(opts["cache_dir"])
    if opts.get("logs_dir"):
        config.LOGS_DIR = Path(opts["logs_dir"])
    config.ensure_stdio("worker.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s worker %(levelname)s %(name)s: %(message)s")
    try:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)      # Ctrl+C reaches every console process; the server stops us
    except (ValueError, OSError):
        pass
    _watch_parent()
    analyze = load_target(opts.get("analyze") or DEFAULT_ANALYZE)
    serve(jobs, msgs, cancel, analyze, prewarm=prewarm if opts.get("prewarm", True) else None,
          device=opts.get("device") or "cuda")


def load_target(spec: str) -> Callable:
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def _watch_parent() -> None:
    """Exit if the server dies without stopping us, so no orphan keeps holding VRAM."""
    parent = mp.parent_process()
    if parent is None:
        return

    def watch():
        while True:
            time.sleep(2.0)
            if not parent.is_alive():
                os._exit(0)
    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def serve(jobs, msgs, cancel, analyze: Callable, *, prewarm: Callable | None = None, device: str = "cuda") -> None:
    """The worker loop (shared by the process and the test thread)."""
    info: dict = {}
    if prewarm is not None:
        msgs.put({"t": "state", "state": "warming"})
        try:
            info = prewarm(device)
        except Exception as e:  # noqa: BLE001 - a failed pre-warm only costs time on the first job
            log.exception("pre-warm failed")
            info = {"prewarm_error": f"{type(e).__name__}: {e}"}
    msgs.put({"t": "ready", "info": info})
    while True:
        job = jobs.get()
        if job is None:
            return
        run_job(job, msgs, cancel, analyze, device)


def run_job(job: dict, msgs, cancel, analyze: Callable, device: str) -> None:
    from ..ingest.errors import IngestError
    jid, seq = job["job"], job["seq"]
    last = {"t": 0.0, "stage": None}

    def check_cancel() -> None:
        if cancel.value == seq:
            raise JobCancelled()

    def progress(stage: str, pct: float, msg: str) -> None:
        check_cancel()
        now = time.monotonic()
        pct = float(pct)
        if stage != last["stage"] or pct >= 1.0 or now - last["t"] >= PROGRESS_EVERY_S:
            last.update(t=now, stage=stage)
            msgs.put({"t": "progress", "job": jid, "stage": stage, "pct": pct, "msg": str(msg)})

    msgs.put({"t": "started", "job": jid})
    _keep_awake(True)
    restore = _low_memory() if job.get("low_mem") else None
    t0 = time.perf_counter()
    try:
        check_cancel()
        a = analyze(job["source"], mode=job.get("mode") or "full", params=job.get("params") or None,
                    progress=progress, device=device, write=True)
        msgs.put({"t": "done", "job": jid, "key": a.source.key, "title": a.source.title, "artist": a.source.artist,
                  "duration": a.source.duration, "n_chords": len(a.chords), "runtime_s": dict(a.stats.runtime_s)})
        log.info("job %s done in %.1f s (%s)", jid, time.perf_counter() - t0, a.source.key)
    except JobCancelled:
        msgs.put({"t": "cancelled", "job": jid})
    except IngestError as e:
        msgs.put({"t": "error", "job": jid, **e.to_dict(), "retry": False})
    except Exception as e:  # noqa: BLE001 - every failure must reach the UI
        log.exception("job %s failed", jid)
        if _is_oom(e):
            msgs.put({"t": "error", "job": jid, "code": "oom", "msg": "The GPU ran out of memory.",
                      "hint": "Close other apps that use the GPU, then try again.", "retry": True})
        else:
            msgs.put({"t": "error", "job": jid, "code": "internal",
                      "msg": f"The analysis failed: {type(e).__name__}: {str(e)[:300]}",
                      "hint": "Details are in the worker log (logs\\worker.log). Try again or use another file.",
                      "retry": False})
    finally:
        if restore:
            restore()
        _keep_awake(False)
        _free_vram()


def prewarm(device: str = "cuda") -> dict:
    """Import torch and load lv-chordia, consonance-ACE, htdemucs_ft and Beat This! once (30–50 s cold)."""
    t0 = time.perf_counter()
    import torch
    info: dict = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": None, "device": device,
                  "prewarm_errors": []}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": p.name, "vram_total_mb": int(p.total_memory // 2**20)}
    else:
        device = info["device"] = "cpu"

    def lv():
        from ..models import lv_chordia
        lv_chordia.load(device)

    def ace():
        from ..models import ace as ace_mod        # importing ACE + its checkpoint costs 12–20 s cold
        ace_mod.load(device)

    def demucs():
        from ..separation import demucs_sep
        demucs_sep._separator(device, demucs_sep.SEGMENT, None)

    def beat_this():
        from ..rhythm import beats
        beats._model(device)

    for name, fn in (("lv-chordia", lv), ("consonance-ACE", ace), ("demucs", demucs), ("beat-this", beat_this)):
        t = time.perf_counter()
        try:
            fn()
            log.info("pre-warm: %s loaded in %.1f s", name, time.perf_counter() - t)
        except Exception as e:  # noqa: BLE001 - the job loads it again (and reports a real error) if needed
            log.exception("pre-warm: %s failed", name)
            info["prewarm_errors"].append(f"{name}: {type(e).__name__}: {e}")
    info["prewarm_s"] = round(time.perf_counter() - t0, 1)
    return info


def _low_memory() -> Callable[[], None] | None:
    """Retry after a crash or OOM (§5.14): drop resident models and use smaller demucs segments."""
    try:
        from .. import models
        from ..separation import demucs_sep
        models.release_gpu()                          # only touches models that are loaded
        if getattr(demucs_sep, "_sep", None) is not None:
            demucs_sep.release()
        old = demucs_sep.SEGMENT
        demucs_sep.SEGMENT = demucs_sep.SEGMENT_OOM
        return lambda: setattr(demucs_sep, "SEGMENT", old)
    except Exception:  # noqa: BLE001
        log.exception("low-memory mode unavailable")
        return None


def _is_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def _free_vram() -> None:
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _keep_awake(on: bool) -> None:
    """Stop Windows from sleeping mid-analysis (SetThreadExecutionState, §9)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        flags = es_continuous | (es_system_required if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------------------------
# Server side: worker handles
# ---------------------------------------------------------------------------------------------

class ProcessWorker:
    """Handle of the spawned GPU worker process."""

    def __init__(self, opts: dict):
        ctx = mp.get_context("spawn")
        self.opts = opts
        self._jobs = ctx.Queue()
        self._msgs = ctx.Queue()
        self._cancel = ctx.Value("q", 0, lock=False)
        self._proc = ctx.Process(target=main, args=(self._jobs, self._msgs, self._cancel, opts),
                                 name="nchord-gpu-worker", daemon=False)
        self._stopped = False

    def start(self) -> None:
        self._proc.start()
        atexit.register(self.stop)
        log.info("GPU worker started (pid %s)", self._proc.pid)

    def send(self, job: dict) -> None:
        self._jobs.put(job)

    def recv(self, timeout: float) -> dict | None:
        try:
            return self._msgs.get(timeout=timeout)
        except (queue.Empty, EOFError, OSError, ValueError):
            return None

    def cancel(self, seq: int) -> None:
        self._cancel.value = seq

    def is_alive(self) -> bool:
        return self._proc.is_alive()

    @property
    def exitcode(self) -> int | None:
        return self._proc.exitcode

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the worker to finish, then terminate it if it doesn't (e.g. mid pre-warm)."""
        if self._stopped:
            return
        self._stopped = True
        if self._proc.is_alive():
            try:
                self._jobs.put(None)
            except (OSError, ValueError):
                pass
            self._proc.join(timeout)
        self.kill()

    def kill(self) -> None:
        self._stopped = True
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(3)
        for q in (self._jobs, self._msgs):
            q.cancel_join_thread()
            q.close()
        try:
            atexit.unregister(self.stop)
        except Exception:  # noqa: BLE001
            pass


class _Flag:
    value = 0


class ThreadWorker:
    """In-process stand-in for tests: the same loop and protocol, in a daemon thread.

    A `SystemExit` raised by the analyze function ends the thread, which the manager sees
    exactly like a crashed worker process.
    """

    def __init__(self, analyze: Callable, prewarm_fn: Callable | None = None):
        self._jobs: queue.Queue = queue.Queue()
        self._msgs: queue.Queue = queue.Queue()
        self._cancel = _Flag()
        self._killed = False
        self._thread = threading.Thread(target=self._run, args=(analyze, prewarm_fn), name="nchord-test-worker",
                                        daemon=True)

    def _run(self, analyze, prewarm_fn) -> None:
        try:
            serve(self._jobs, self._msgs, self._cancel, analyze, prewarm=prewarm_fn, device="cpu")
        except SystemExit as e:                                  # the simulated crash: the thread just ends
            log.info("test worker exited (%s)", e.code)

    def start(self) -> None:
        self._thread.start()

    def send(self, job: dict) -> None:
        self._jobs.put(job)

    def recv(self, timeout: float) -> dict | None:
        try:
            return self._msgs.get(timeout=timeout)
        except queue.Empty:
            return None

    def cancel(self, seq: int) -> None:
        self._cancel.value = seq

    def is_alive(self) -> bool:
        return not self._killed and self._thread.is_alive()

    @property
    def exitcode(self) -> int | None:
        return None if self.is_alive() else 1

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread.is_alive():
            self._jobs.put(None)
            self._thread.join(timeout)
        self._killed = True

    def kill(self) -> None:
        self._killed = True         # a thread can't be killed; it is abandoned instead
