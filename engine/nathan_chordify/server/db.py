"""SQLite library at data/library.sqlite: the tracks list and the job history (DESIGN §2, §11).

The cache folder stays the source of truth for analyses; this file only adds what the cache
can't hold cheaply (renames, last-opened times, job history). `reconcile()` re-syncs the two.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .common import now_iso, parse_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    key          TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    artist       TEXT,
    duration     REAL,
    source_type  TEXT,
    source_url   TEXT,
    created      TEXT NOT NULL,
    last_opened  TEXT,
    status       TEXT NOT NULL DEFAULT 'ready',   -- analysing | ready | failed
    renamed      INTEGER NOT NULL DEFAULT 0       -- 1 once the user edited title/artist
);
CREATE TABLE IF NOT EXISTS jobs (
    id        TEXT PRIMARY KEY,
    key       TEXT,
    source    TEXT NOT NULL,
    mode      TEXT NOT NULL,
    params    TEXT,
    state     TEXT NOT NULL,                      -- queued | running | done | error | cancelled
    created   TEXT NOT NULL,
    started   TEXT,
    finished  TEXT,
    error     TEXT,
    attempts  INTEGER NOT NULL DEFAULT 0,
    cached    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_key ON jobs(key);
PRAGMA user_version = 1;
"""


def _track(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["renamed"] = bool(d["renamed"])
    return d


class Library:
    """Thread-safe wrapper around one SQLite connection (the server is a single process)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _rows(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _exec(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            return self._db.execute(sql, args).rowcount

    # ---- tracks ---------------------------------------------------------------------------------
    def ensure_track(self, key: str, title: str, status: str = "analysing", **fields) -> None:
        """Insert a placeholder row if the key is new; otherwise only set its status."""
        self._exec("INSERT OR IGNORE INTO tracks(key, title, artist, duration, source_type, source_url, created, status)"
                   " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                   (key, title, fields.get("artist"), fields.get("duration"), fields.get("source_type"),
                    fields.get("source_url"), now_iso(), status))
        self.set_status(key, status)

    def upsert_track(self, key: str, *, title: str, artist: str | None = None, duration: float | None = None,
                     source_type: str | None = None, source_url: str | None = None, status: str = "ready",
                     created: str | None = None) -> None:
        """Insert or refresh a track from its analysis; a user rename is kept."""
        self._exec(
            "INSERT INTO tracks(key, title, artist, duration, source_type, source_url, created, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET"
            "  title = CASE WHEN tracks.renamed THEN tracks.title ELSE excluded.title END,"
            "  artist = CASE WHEN tracks.renamed THEN tracks.artist ELSE COALESCE(excluded.artist, tracks.artist) END,"
            "  duration = COALESCE(excluded.duration, tracks.duration),"
            "  source_type = COALESCE(excluded.source_type, tracks.source_type),"
            "  source_url = COALESCE(excluded.source_url, tracks.source_url),"
            "  status = excluded.status",
            (key, title, artist, duration, source_type, source_url, created or now_iso(), status))

    def set_status(self, key: str, status: str) -> None:
        self._exec("UPDATE tracks SET status = ? WHERE key = ?", (status, key))

    def touch(self, key: str) -> None:
        self._exec("UPDATE tracks SET last_opened = ? WHERE key = ?", (now_iso(), key))

    def get(self, key: str) -> dict | None:
        rows = self._rows("SELECT * FROM tracks WHERE key = ?", (key,))
        return _track(rows[0]) if rows else None

    def list(self, q: str | None = None) -> list[dict]:
        sql = "SELECT * FROM tracks"
        args: tuple = ()
        if q:
            like = f"%{q.strip()}%"
            sql += " WHERE title LIKE ? OR artist LIKE ? OR key LIKE ?"
            args = (like, like, like)
        sql += " ORDER BY COALESCE(last_opened, created) DESC"
        return [_track(r) for r in self._rows(sql, args)]

    def rename(self, key: str, title: str | None = None, artist: str | None = None) -> dict | None:
        if title is not None:
            self._exec("UPDATE tracks SET title = ?, renamed = 1 WHERE key = ?", (title, key))
        if artist is not None:
            self._exec("UPDATE tracks SET artist = ?, renamed = 1 WHERE key = ?", (artist or None, key))
        return self.get(key)

    def delete(self, key: str) -> None:
        self._exec("DELETE FROM tracks WHERE key = ?", (key,))

    def keys(self) -> set[str]:
        return {r["key"] for r in self._rows("SELECT key FROM tracks")}

    def last_opened(self) -> dict[str, float]:
        """key → epoch seconds of the last open (the creation time if never opened)."""
        out = {}
        for r in self._rows("SELECT key, created, last_opened FROM tracks"):
            t = parse_iso(r["last_opened"]) or parse_iso(r["created"])
            if t is not None:
                out[r["key"]] = t
        return out

    # ---- jobs -----------------------------------------------------------------------------------
    def save_job(self, snap: dict) -> None:
        self._exec(
            "INSERT OR REPLACE INTO jobs(id, key, source, mode, params, state, created, started, finished, error,"
            " attempts, cached) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (snap["job_id"], snap.get("key"), snap["source"], snap["mode"],
             json.dumps(snap.get("params")) if snap.get("params") else None, snap["state"], snap["created"],
             snap.get("started"), snap.get("finished"),
             json.dumps(snap["error"], ensure_ascii=False) if snap.get("error") else None,
             snap.get("attempts", 0), int(bool(snap.get("cached")))))

    def jobs(self, key: str | None = None, limit: int = 50) -> list[dict]:
        if key:
            rows = self._rows("SELECT * FROM jobs WHERE key = ? ORDER BY created DESC LIMIT ?", (key, limit))
        else:
            rows = self._rows("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d["params"] = json.loads(d["params"]) if d["params"] else None
            d["error"] = json.loads(d["error"]) if d["error"] else None
            d["cached"] = bool(d["cached"])
            out.append(d)
        return out

    def mark_interrupted_jobs(self) -> int:
        """Jobs left queued/running by a previous server run can't resume; record them as errors."""
        err = json.dumps({"code": "server_restarted", "msg": "The server stopped before this job finished.",
                          "hint": "Submit the song again."})
        return self._exec("UPDATE jobs SET state = 'error', error = ?, finished = ? WHERE state IN ('queued', 'running')",
                          (err, now_iso()))
