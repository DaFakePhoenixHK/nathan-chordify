"""System routes (DESIGN §11): health, versions, storage, downloader update.

    GET  /health                            {status, worker, queue, engine_version}
    GET  /api/system/info                   versions, GPU, ffmpeg/deno/yt-dlp, disk, worker
    GET  /api/system/storage                cache size per song, cap, retention, free disk
    POST /api/system/storage/cleanup        apply retention + cap now → report
    POST /api/system/update-downloader      stop worker → pip install -U --pre "yt-dlp[default]" → restart

The server process never imports torch or yt-dlp: the GPU comes from the worker's pre-warm
report (or nvidia-smi before that), yt-dlp's version from its package metadata.
"""
from __future__ import annotations

import functools
import platform
import subprocess
import sys
import time
from importlib import metadata

from fastapi import APIRouter, Depends

from .. import ENGINE_VERSION, config, schema
from . import storage
from .common import ApiError, ctx

router = APIRouter()


def pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _first_line(args: list[str], timeout: float = 10.0) -> str | None:
    try:
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, creationflags=config.NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    out = (r.stdout or r.stderr or "").strip().splitlines()
    return out[0].strip() if r.returncode == 0 and out else None


@functools.lru_cache(maxsize=1)
def ffmpeg_version() -> str | None:
    line = _first_line([str(config.FFMPEG), "-hide_banner", "-version"])
    return line.split(" Copyright")[0].removeprefix("ffmpeg version ").strip() if line else None


@functools.lru_cache(maxsize=1)
def deno_version() -> str | None:
    try:
        import deno
        line = _first_line([deno.find_deno_bin(), "--version"])
    except Exception:  # noqa: BLE001
        return None
    return line.removeprefix("deno ").strip() if line else None


@functools.lru_cache(maxsize=1)
def nvidia_smi() -> dict | None:
    line = _first_line(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    try:
        return {"name": parts[0], "vram_total_mb": int(float(parts[1])), "driver": parts[2]}
    except (IndexError, ValueError):
        return None


@router.get("/health")
def health(c=Depends(ctx)) -> dict:
    return {"status": "ok", "worker": c.jobs.worker_state, "queue": c.jobs.queue_length(),
            "engine_version": ENGINE_VERSION}


@router.get("/api/system/info")
def info(c=Depends(ctx)) -> dict:
    w = c.jobs.worker_info
    smi = nvidia_smi()
    gpu = dict(w["gpu"]) if w.get("gpu") else (dict(smi) if smi else None)
    if gpu and smi and "driver" not in gpu:
        gpu["driver"] = smi.get("driver")
    disk = storage.disk_status(c.cache_dir, c.settings.policy)
    return {
        "engine_version": ENGINE_VERSION, "schema_version": schema.SCHEMA_VERSION,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {n: pkg_version(n) for n in ("fastapi", "uvicorn", "starlette", "pydantic")},
        "torch": w.get("torch"), "cuda": w.get("cuda"), "gpu": gpu,
        "ffmpeg": ffmpeg_version(), "deno": deno_version(), "yt_dlp": pkg_version("yt-dlp"),
        "disk": disk, "worker": {"state": c.jobs.worker_state, "prewarm_s": w.get("prewarm_s"),
                                 "prewarm_errors": w.get("prewarm_errors") or [], "device": w.get("device")},
        "stub": c.settings.is_stub, "phone_url": None,           # TODO(M6): Tailscale serve URL
    }


@router.get("/api/system/storage")
def storage_info(c=Depends(ctx)) -> dict:
    out = storage.summary(c.cache_dir, c.library.last_opened(), c.settings.policy)
    out["last_cleanup"] = c.last_cleanup
    return out


@router.post("/api/system/storage/cleanup")
def storage_cleanup(c=Depends(ctx)) -> dict:
    return c.run_storage()


@router.post("/api/system/update-downloader")
def update_downloader(c=Depends(ctx)) -> dict:
    """The worker holds yt-dlp's .pyd files open, so it is stopped first (Windows can't replace them)."""
    if not c.update_lock.acquire(blocking=False):
        raise ApiError(409, "busy", "An update is already running.")
    try:
        before = pkg_version("yt-dlp")
        if not c.jobs.stop_worker():
            raise ApiError(409, "busy", "An analysis is running.", "Update the downloader when it has finished.")
        t0 = time.perf_counter()
        try:
            cmd = c.settings.pip_cmd or [sys.executable, "-m", "pip", "install", "-U", "--pre", "yt-dlp[default]"]
            constraints = config.ENGINE_DIR / "constraints.txt"
            if c.settings.pip_cmd is None and constraints.exists():
                cmd += ["-c", str(constraints)]                   # keeps torch/numpy pinned
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   timeout=600, creationflags=config.NO_WINDOW)
                ok, log = r.returncode == 0, (r.stdout or "") + (r.stderr or "")
            except (OSError, subprocess.SubprocessError) as e:
                ok, log = False, f"{type(e).__name__}: {e}"
        finally:
            c.jobs.start_worker()
        return {"ok": ok, "before": before, "after": pkg_version("yt-dlp"), "seconds": round(time.perf_counter() - t0, 1),
                "log": log[-3000:]}
    finally:
        c.update_lock.release()
