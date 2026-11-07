"""Cache retention, size cap and free-space checks (DESIGN §5.14).

- Retention: stems and evidence of songs not opened for `retention_days` (30) are deleted.
- Size cap (10 GB for cache/): the least recently opened songs lose their stems first, then
  their evidence, then audio.wav (re-decoded from source.* on demand).
- Always kept: analysis*.json, display.json, edits.jsonl, play.m4a, source.*, meta.json, so a
  song stays playable and re-analysable.
- New jobs are refused below 5 GB free on the cache drive; the UI warns below 15 GB.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

GB = 1_000_000_000
UPLOADS = "_uploads"
TRASH = "_trash"

# Evictable parts of a song folder, in eviction order.
TIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("stems", ("stems",)),
    ("evidence", ("evidence", "notes.npz")),
    ("audio", ("audio.wav",)),
)
RETENTION_TIERS = ("stems", "evidence")


@dataclass
class Policy:
    retention_days: float = 30.0
    cap_bytes: int = 10 * GB
    refuse_free_bytes: int = 5 * GB
    warn_free_bytes: int = 15 * GB
    upload_max_age_s: float = 24 * 3600.0


@dataclass
class Report:
    total_before: int = 0
    total_after: int = 0
    freed: int = 0
    removed: list[dict] = field(default_factory=list)     # {key, tier, bytes, reason}

    def to_dict(self) -> dict:
        return {"total_before": self.total_before, "total_after": self.total_after, "freed": self.freed,
                "removed": self.removed}


def path_size(p: Path) -> int:
    """Bytes of a file or a whole folder (files that vanish meanwhile count as 0)."""
    try:
        if p.is_file():
            return p.stat().st_size
    except OSError:
        return 0
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def song_dirs(cache: Path) -> list[Path]:
    if not cache.is_dir():
        return []
    return [d for d in cache.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]


def tier_paths(song: Path, tier: str) -> list[Path]:
    names = dict(TIERS)[tier]
    return [song / n for n in names if (song / n).exists()]


def _remove(p: Path) -> int:
    """Delete a file or folder; returns the bytes freed (0 if Windows refused, e.g. a file in use)."""
    size = path_size(p)
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    except OSError:
        return max(0, size - path_size(p)) if p.exists() else size
    return size


def _fallback_time(song: Path) -> float:
    for name in ("analysis.json", "meta.json"):
        try:
            return (song / name).stat().st_mtime
        except OSError:
            pass
    try:
        return song.stat().st_mtime
    except OSError:
        return time.time()


def enforce(cache: Path, last_opened: dict[str, float], policy: Policy, protect: set[str] | frozenset = frozenset(),
            protect_uploads: set[str] | frozenset = frozenset(), now: float | None = None) -> Report:
    """Apply retention, the size cap and the upload clean-up. `protect` keys (active jobs) are skipped."""
    now = time.time() if now is None else now
    rep = Report(total_before=path_size(cache))
    songs = [d for d in song_dirs(cache) if d.name not in protect]
    opened = {d.name: last_opened.get(d.name) or _fallback_time(d) for d in songs}

    def drop(song: Path, tier: str, reason: str) -> int:
        freed = 0
        for p in tier_paths(song, tier):
            freed += _remove(p)
        if freed:
            rep.removed.append({"key": song.name, "tier": tier, "bytes": freed, "reason": reason})
            rep.freed += freed
        return freed

    # 1) retention
    max_age = policy.retention_days * 86400.0
    for song in songs:
        if now - opened[song.name] > max_age:
            for tier in RETENTION_TIERS:
                drop(song, tier, "retention")

    # 2) deleted songs whose files were still locked, and leftover uploads (the job copies the
    #    file into cache/<key>/source.*)
    trash = cache / TRASH
    if trash.is_dir():
        for d in trash.iterdir():
            freed = _remove(d)
            rep.freed += freed
            rep.removed.append({"key": f"{TRASH}/{d.name}", "tier": "trash", "bytes": freed, "reason": "deleted"})
    up = cache / UPLOADS
    if up.is_dir():
        for d in up.iterdir():
            if d.name in protect_uploads:
                continue
            try:
                age = now - d.stat().st_mtime
            except OSError:
                continue
            if age > policy.upload_max_age_s:
                freed = _remove(d)
                rep.freed += freed
                rep.removed.append({"key": f"{UPLOADS}/{d.name}", "tier": "upload", "bytes": freed, "reason": "age"})

    # 3) size cap: least recently opened first, one tier at a time across all songs
    total = rep.total_before - rep.freed
    if total > policy.cap_bytes:
        lru = sorted(songs, key=lambda d: opened[d.name])
        for tier, _ in TIERS:
            for song in lru:
                if total <= policy.cap_bytes:
                    break
                total -= drop(song, tier, "cap")
            if total <= policy.cap_bytes:
                break
    rep.total_after = path_size(cache)
    return rep


def disk_status(path: Path, policy: Policy) -> dict:
    """Free space on the drive that holds `path` (the nearest existing parent)."""
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    usage = shutil.disk_usage(p)
    return {"free_bytes": usage.free, "total_bytes": usage.total,
            "warn": usage.free < policy.warn_free_bytes, "refuse": usage.free < policy.refuse_free_bytes,
            "warn_below_bytes": policy.warn_free_bytes, "refuse_below_bytes": policy.refuse_free_bytes}


def summary(cache: Path, last_opened: dict[str, float], policy: Policy) -> dict:
    """Sizes per song and per tier, for Settings → Storage."""
    tracks = []
    for song in song_dirs(cache):
        tiers = {tier: sum(path_size(p) for p in tier_paths(song, tier)) for tier, _ in TIERS}
        t = last_opened.get(song.name) or _fallback_time(song)
        tracks.append({"key": song.name, "bytes": path_size(song), "evictable": tiers,
                       "last_opened": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))})
    tracks.sort(key=lambda r: r["bytes"], reverse=True)
    return {"cache_bytes": path_size(cache), "cap_bytes": policy.cap_bytes,
            "retention_days": policy.retention_days, "disk": disk_status(cache, policy), "tracks": tracks}
