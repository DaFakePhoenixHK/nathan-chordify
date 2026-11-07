"""Shared helpers for the routes: API errors, cache-key checks, timestamps, the app context."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from .app import ServerContext

# "yt-<id>" | "url-<extractor>-<id>" | "sha256-<hex64>" (ingest.resolve). No dots or slashes,
# so a key can never walk out of cache/.
KEY_RE = re.compile(r"^(yt|url|sha256)-[A-Za-z0-9_-]{1,128}$")


class ApiError(Exception):
    """An error answered as JSON `{code, msg, hint}` (the same shape as the SSE `error` event)."""

    def __init__(self, status: int, code: str, msg: str, hint: str = ""):
        super().__init__(msg)
        self.status, self.code, self.msg, self.hint = status, code, msg, hint

    def to_dict(self) -> dict:
        return {"code": self.code, "msg": self.msg, "hint": self.hint}


def not_implemented(feature: str, milestone: str) -> ApiError:
    return ApiError(501, "not_implemented", f"{feature} is not built yet (planned for {milestone}).")


def check_key(key: str) -> str:
    if not KEY_RE.match(key or ""):
        raise ApiError(400, "bad_key", f"Not a track key: {key!r}")
    return key


def now_iso(t: float | None = None) -> str:
    dt = datetime.now(timezone.utc) if t is None else datetime.fromtimestamp(t, timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def track_urls(key: str) -> dict:
    base = f"/api/tracks/{key}"
    return {"analysis_url": base, "display_url": f"{base}/display", "audio_url": f"{base}/audio"}


def ctx(request: Request) -> "ServerContext":
    """FastAPI dependency: the per-app context made by the lifespan."""
    return request.app.state.ctx
