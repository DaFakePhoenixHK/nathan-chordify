"""Letting a phone or tablet reach this server safely.

The chord models need this computer's GPU, so an iPhone or iPad is a remote screen for it. Whatever
route the phone comes in by (a public HTTPS link, a private network), a request that does not come
from this computer must show the secret phone code. The code is created once, kept in
``data/phone_token.txt`` (never committed) and handed to the phone as part of a pairing link.

A tunnel connects to this server from 127.0.0.1, so the client address alone proves nothing: a
request only counts as local when its address is loopback, it carries no forwarding header, its
Host names this computer, and no foreign web page sent it.
"""
from __future__ import annotations

import secrets
import time
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse, Response

TOKEN_FILE = "phone_token.txt"
APP_URL = "https://dafakephoenixhk.github.io/nathan-chordify/"

FORWARD_HEADERS = ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-real-ip", "forwarded",
                   "tailscale-user-login")
LOCAL_NAMES = {"127.0.0.1", "localhost", "::1"}

# Never answered to a phone, even with the code: machine administration and the code itself.
LOCAL_ONLY = ("/api/phone", "/api/system", "/api/docs", "/api/openapi.json")
# <audio> and EventSource cannot send headers, so these GET routes also accept ?token=
QUERY_TOKEN_SUFFIXES = ("/audio", "/events", "/export/sheet")

ALLOW_METHODS = "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS"
ALLOW_HEADERS = "authorization, content-type, accept, range"
EXPOSE_HEADERS = "content-range, accept-ranges, content-length, content-disposition"


def phone_token(data_dir: Path) -> str:
    """The secret phone code, created on first use."""
    p = Path(data_dir) / TOKEN_FILE
    try:
        t = p.read_text("ascii").strip()
    except (OSError, UnicodeError):
        t = ""
    if len(t) >= 20:
        return t
    return reset_token(data_dir)


def reset_token(data_dir: Path) -> str:
    """Make a new code: every paired phone has to scan again."""
    t = secrets.token_urlsafe(18)
    p = Path(data_dir) / TOKEN_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(t, "ascii")
    return t


def pair_link(app_url: str | None, public_url: str, token: str) -> str:
    """The link a phone opens or scans. The part after # never reaches the web host's logs.

    Without a published app page, the computer serves the app itself at its public address."""
    if not app_url:
        return f"{public_url.rstrip('/')}/#/pair?t={token}"
    return f"{app_url}#/pair?s={quote(public_url.rstrip('/'), safe='')}&t={token}"


_app_seen = {"at": -1e9, "ok": False}


def app_online(timeout: float = 4.0) -> bool:
    """Is the phone app published at APP_URL? Asked again after 10 minutes (1 while it isn't)."""
    if time.monotonic() - _app_seen["at"] < (600 if _app_seen["ok"] else 60):
        return bool(_app_seen["ok"])
    try:
        req = urllib.request.Request(APP_URL + "manifest.webmanifest", method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ok = r.status == 200
    except OSError:
        ok = False
    _app_seen.update(at=time.monotonic(), ok=ok)
    return ok


def _hostname(netloc: str) -> str:
    """A Host header or URL authority, as the bare lower-case name ("[::1]:80" gives "::1")."""
    netloc = netloc.rsplit("@", 1)[-1].lower()
    if netloc.startswith("["):
        return netloc[1:].split("]", 1)[0]
    return netloc.split(":", 1)[0]


def is_local(scope, headers: Headers) -> bool:
    client = (scope.get("client") or ("", 0))[0]
    if client not in ("127.0.0.1", "::1", "testclient"):
        return False
    if any(h in headers for h in FORWARD_HEADERS):
        return False
    host = _hostname(headers.get("host", ""))
    if host not in LOCAL_NAMES and not (client == "testclient" and host == "testserver"):
        return False
    origin = headers.get("origin")
    return not origin or _hostname(urlsplit(origin).netloc) in LOCAL_NAMES


def _given_token(scope, headers: Headers, path: str, method: str) -> str:
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if method in ("GET", "HEAD") and path.endswith(QUERY_TOKEN_SUFFIXES):
        q = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        return (q.get("token") or [""])[0]
    return ""


def _problem(status: int, code: str, msg: str, hint: str, extra: dict) -> JSONResponse:
    return JSONResponse({"code": code, "msg": msg, "hint": hint}, status_code=status, headers=extra)


class RemoteGuard:
    """Pure ASGI middleware (doesn't buffer SSE or audio): the phone code, plus CORS for the phone app."""

    def __init__(self, app, settings, data_dir):
        self.app = app
        self.settings = settings
        self.data_dir = data_dir                # a callable, so tests can move the data folder

    def _allowed_origin(self, headers: Headers, scope) -> str | None:
        origin = headers.get("origin")
        if not origin:
            return None
        if origin in self.settings.phone_origins:
            return origin
        same = f"{scope.get('scheme', 'http')}://{headers.get('host', '')}"
        return origin if origin.lower() == same.lower() else None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        path, method = scope.get("path", ""), scope.get("method", "GET")
        origin = self._allowed_origin(headers, scope)
        cors = {}
        if origin:
            cors = {"Access-Control-Allow-Origin": origin, "Vary": "Origin",
                    "Access-Control-Expose-Headers": EXPOSE_HEADERS,
                    "Cross-Origin-Resource-Policy": "cross-origin"}

        if method == "OPTIONS" and "access-control-request-method" in headers:
            if origin:
                cors.update({"Access-Control-Allow-Methods": ALLOW_METHODS, "Access-Control-Allow-Headers": ALLOW_HEADERS,
                             "Access-Control-Max-Age": "600"})
            return await Response(status_code=204, headers=cors)(scope, receive, send)

        if path.startswith("/api") and not is_local(scope, headers):
            answer = None
            if path.startswith(LOCAL_ONLY):
                answer = _problem(403, "local_only", "This can only be done on the computer itself.", "", cors)
            else:
                given, token = _given_token(scope, headers, path, method), phone_token(self.data_dir())
                if not (given and secrets.compare_digest(given.encode(), token.encode())):
                    answer = _problem(401, "need_code", "This phone isn't connected to the computer yet.",
                                      "Scan the code shown on the computer again.", cors)
            if answer is not None:
                return await answer(scope, receive, send)

        if not cors:
            return await self.app(scope, receive, send)

        async def send_with_cors(message):
            if message["type"] == "http.response.start":
                h = MutableHeaders(scope=message)
                for k, v in cors.items():
                    h[k] = v
            await send(message)

        await self.app(scope, receive, send_with_cors)
