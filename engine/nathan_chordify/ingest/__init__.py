"""WP1 ingest: source → cache/<key>/{source.*, audio.wav, play.m4a, meta.json} (DESIGN §5.1)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .. import config
from ..types import AudioAsset, Progress
from . import decode, youtube
from .errors import IngestError
from .resolve import resolve, youtube_id

__all__ = ["resolve", "fetch", "youtube_id", "IngestError", "cache_dir"]


def _noop(stage: str, pct: float, msg: str) -> None:
    pass


def cache_dir(key: str) -> Path:
    return config.CACHE_DIR / key


def _asset(key: str, d: Path, meta: dict) -> AudioAsset:
    return AudioAsset(key=key, dir=d, wav=d / "audio.wav", play=d / "play.m4a",
                      duration_s=float(meta.get("duration_s") or 0.0), title=meta.get("title") or key,
                      artist=meta.get("uploader") or meta.get("artist"),
                      source_type=meta["source_type"], source_url=meta.get("source_url"))


def fetch(source: str, progress: Progress | None = None, max_len_s: int = youtube.MAX_LEN_S,
          force: bool = False) -> AudioAsset:
    """Download or import `source` once; later calls are served from the cache."""
    progress = progress or _noop
    key, meta = resolve(source)
    d = cache_dir(key)
    meta_path = d / "meta.json"
    if meta_path.exists():
        meta = {**meta, **json.loads(meta_path.read_text("utf-8"))}
    wav, m4a = d / "audio.wav", d / "play.m4a"
    if wav.exists() and m4a.exists() and meta.get("duration_s"):
        return _asset(key, d, meta)

    d.mkdir(parents=True, exist_ok=True)
    src = next((p for p in d.glob("source.*") if p.suffix not in (".part", ".ytdl")), None)
    if src is None:
        if meta["source_type"] == "upload":
            progress("import", 0.0, "Copying file")
            orig = Path(meta["path"])
            src = d / f"source{orig.suffix.lower() or '.bin'}"
            shutil.copyfile(orig, src)
        else:
            progress("download", 0.0, "Downloading audio")
            src, dl_meta = youtube.download(meta["source_url"], d, progress, max_len_s=max_len_s, force=force)
            meta.update({k: v for k, v in dl_meta.items() if v is not None})
    progress("decode", 0.0, "Decoding audio")
    duration = decode.probe_duration(src)
    if duration > max_len_s:
        raise IngestError("too_long", f"The audio is {duration / 60:.0f} min; the limit is {max_len_s // 60} min.",
                          "Raise the limit in Settings, or use a shorter excerpt.")
    if not wav.exists():
        decode.to_wav(src, wav)
    if not m4a.exists():
        decode.to_play_m4a(wav, m4a)
    meta["duration_s"] = decode.probe_duration(wav)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    progress("decode", 1.0, "Audio ready")
    return _asset(key, d, meta)
