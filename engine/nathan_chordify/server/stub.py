"""A stand-in for `pipeline.analyze` with no GPU: it writes the 回家 fixture, re-keyed (tests, UI work).

    python -m nathan_chordify.server --stub

Uploads go through the real ingest (ffmpeg → audio.wav + play.m4a); YouTube links work only for
songs already in the cache (no network). A real analysis.json in the cache is never overwritten.
Env: NCHORD_STUB_DELAY (seconds per stage, default 0.02); NCHORD_STUB_CRASH_ONCE=<marker file>
makes the first call kill the process (crash-recovery tests).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import ENGINE_VERSION, config, ingest, schema
from ..ingest.errors import IngestError

STUB_VERSION = f"{ENGINE_VERSION}+stub"
STAGES = (("separate", "Separating stems"), ("rhythm", "Tracking beats"), ("evidence", "Reading chords"),
          ("decode", "Decoding chords"), ("output", "Spelling and voicing"))


def fixture_paths() -> tuple[Path, Path]:
    d = config.SHARED_DIR / "fixtures"
    return d / "huijia.analysis.json", d / "huijia.display.json"


def _noop(stage: str, pct: float, msg: str) -> None:
    pass


def _crash_once() -> None:
    marker = os.environ.get("NCHORD_STUB_CRASH_ONCE")
    if marker and not Path(marker).exists():
        Path(marker).write_text("crashed", "utf-8")
        os._exit(3)


def analyze(source: str, mode: str = "full", params: dict | None = None, progress=None, device: str = "cuda",
            write: bool = True) -> schema.Analysis:
    progress = progress or _noop
    _crash_once()
    delay = float(os.environ.get("NCHORD_STUB_DELAY", "0.02"))
    key, meta = ingest.resolve(source)
    d = ingest.cache_dir(key)
    existing = d / "analysis.json"
    if existing.exists():
        a = schema.Analysis.model_validate_json(existing.read_text("utf-8"))
        if a.engine_version != STUB_VERSION:            # a real analysis: keep it
            progress("output", 1.0, "Stub mode: using the cached analysis")
            return a
    if meta["source_type"] != "upload" and not (d / "play.m4a").exists():
        raise IngestError("unavailable", "Stub mode only analyses uploads and songs already in the cache.",
                          "Start the server without --stub to download new songs.")

    if config.FFMPEG.exists():
        asset = ingest.fetch(source, progress)
        title, artist, duration, url = asset.title, asset.artist, asset.duration_s, asset.source_url
    else:                                                # no ffmpeg: serve the uploaded bytes as-is
        progress("import", 0.0, "Copying file")
        d.mkdir(parents=True, exist_ok=True)
        src = Path(meta["path"])
        shutil.copyfile(src, d / f"source{src.suffix.lower()}")
        shutil.copyfile(src, d / "play.m4a")
        title, artist, duration, url = meta.get("title") or key, None, 0.0, None

    for stage, msg in STAGES:
        progress(stage, 0.0, msg)
        time.sleep(delay)
        progress(stage, 1.0, msg)

    fa, fd = fixture_paths()
    a = schema.Analysis.model_validate_json(fa.read_text("utf-8"))
    a.engine_version = STUB_VERSION
    a.mode = mode
    a.params_hash = hashlib.sha1(json.dumps(params or {}, sort_keys=True).encode()).hexdigest()[:8]
    a.created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    a.source = schema.SourceInfo(key=key, type=meta["source_type"], url=url, title=title, artist=artist,
                                 duration=round(float(duration or a.source.duration), 3))
    a.audio = schema.AudioInfo(play=f"/api/tracks/{key}/audio", stems=[])
    a.display = f"/api/tracks/{key}/display"
    if write:
        d.mkdir(parents=True, exist_ok=True)
        (d / "analysis.json").write_text(a.model_dump_json(by_alias=True, indent=1), "utf-8")
        shutil.copyfile(fd, d / "display.json")
    progress("done", 1.0, f"{len(a.chords)} chords")
    return a
