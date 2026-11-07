"""Polite page fetching with a cache OUTSIDE the repo (pages hold lyrics; the repo never does).

One request at a time, >= MIN_GAP s apart (CantonHymn >= 2.5 s: it starts failing after ~45 fast
calls), back-off 30 s then 120 s on errors, a host that keeps failing is paused for this run,
never logs in. Cached pages live under %TEMP%/nathan_chordify_chart_cache (``CHART_CACHE`` env)
and are re-fetched after ``max_age_s`` (callers pass 24 h for pages, 7 days for search results).

Moved here from train/harvest_charts.py (2026-11-03) so the server's sheet lookup and the
training harvester share one fetcher.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
import time
import urllib.parse
from collections import Counter
from pathlib import Path

DEFAULT_CACHE = Path(os.environ.get("CHART_CACHE", Path(tempfile.gettempdir()) / "nathan_chordify_chart_cache"))

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/130.0 Safari/537.36")
MIN_GAP = 1.2                        # seconds between any two requests
HOST_GAP = {"cantonhymn.net": 2.5}   # gentler for CantonHymn
BACKOFF = (30, 120)                  # seconds before the 2nd and 3rd attempt
PAGE_MAX_AGE_S = 24 * 3600           # raw pages (they hold lyrics) are kept at most a day
SEARCH_MAX_AGE_S = 7 * 24 * 3600     # search results per cleaned title: a week


def today() -> str:
    return dt.date.today().isoformat()


class Fetcher:
    """One request at a time, spaced out, cached on disk, with back-off; never logs in."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE, offline: bool = False, refresh: bool = False,
                 sleep=time.sleep):
        self.cache = Path(cache_dir)
        self.offline, self.refresh, self.sleep = offline, refresh, sleep
        self.last_any = 0.0
        self.last_host: dict[str, float] = {}
        self.fail_streak: Counter = Counter()
        self.paused: set[str] = set()
        self.requests = 0
        self.events: list[tuple[str, str]] = []     # (url, status) of network attempts
        self._session = None

    def _path(self, url: str) -> Path:
        host = urllib.parse.urlparse(url).netloc
        return self.cache / host / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".txt")

    def cached_date(self, url: str) -> str | None:
        p = self._path(url)
        return dt.date.fromtimestamp(p.stat().st_mtime).isoformat() if p.exists() else None

    def cache_age_s(self, url: str) -> float | None:
        p = self._path(url)
        return time.time() - p.stat().st_mtime if p.exists() else None

    def get(self, url: str, max_age_s: float | None = None) -> tuple[str | None, str]:
        """(text or None, status) - status "cache", "200", "404", "blocked", "error ...".

        ``max_age_s``: a cached copy older than this is fetched again (None = any age)."""
        p = self._path(url)
        if p.exists() and not self.refresh:
            if max_age_s is None or time.time() - p.stat().st_mtime <= max_age_s:
                return p.read_text(encoding="utf-8"), "cache"
        if self.offline:
            return None, "offline"
        host = urllib.parse.urlparse(url).netloc
        if host in self.paused:
            return None, "paused"
        import requests
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9,zh-HK;q=0.8"})
        status = "error"
        for attempt in range(1 + len(BACKOFF)):
            if attempt:
                self.sleep(BACKOFF[attempt - 1])
            gap = max(MIN_GAP - (time.time() - self.last_any),
                      HOST_GAP.get(host, MIN_GAP) - (time.time() - self.last_host.get(host, 0.0)))
            if gap > 0:
                self.sleep(gap)
            self.last_any = self.last_host[host] = time.time()
            self.requests += 1
            try:
                r = self._session.get(url, timeout=25)
            except Exception as e:  # network error: back off and retry
                status = f"error {type(e).__name__}"
                self.events.append((url, status))
                continue
            status = str(r.status_code)
            self.events.append((url, status))
            if r.status_code == 200:
                self.fail_streak[host] = 0
                p.parent.mkdir(parents=True, exist_ok=True)
                r.encoding = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else "utf-8"
                p.write_text(r.text, encoding="utf-8")
                return r.text, "200"
            if r.status_code in (404, 410):
                return None, status
            if r.status_code in (401, 403):   # a block or a login wall: do not push
                self.fail_streak[host] += 3
                break
        self.fail_streak[host] += 1
        if self.fail_streak[host] >= 3:
            self.paused.add(host)
            status += " (host paused for this run)"
        return None, status



    def get_json(self, url: str, max_age_s: float | None = None) -> tuple[dict | list | None, str]:
        """``get`` + ``json.loads`` (None, "bad json" when the body is not JSON)."""
        text, status = self.get(url, max_age_s)
        if text is None:
            return None, status
        try:
            return json.loads(text), status
        except ValueError:
            return None, "bad json"

    def forget(self, url: str) -> None:
        """Drop one cached page (e.g. a search result the user wants refreshed)."""
        p = self._path(url)
        if p.exists():
            p.unlink()


def is_online_error(status: str) -> bool:
    """True for a fetch status that means "could not reach the site" (not a 404)."""
    return status in ("offline", "paused") or status.startswith("error") or status in ("500", "502", "503", "504")
