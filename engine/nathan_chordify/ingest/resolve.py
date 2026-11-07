"""Classify a source (YouTube link, other URL, local file) into a cache key (DESIGN §5.1 step 1).

YouTube IDs are parsed from the link itself, with no network call, so the same song never
touches YouTube twice (rate limits on VPN IPs are common; see CLAUDE.md).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .errors import IngestError

_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com",
             "www.youtube-nocookie.com"}


def youtube_id(url: str) -> str | None:
    """Return the 11-character video ID for any YouTube link form, else None.

    Handles watch?v=, youtu.be/, shorts/, embed/, live/, v/ and music.youtube.com. Playlist, Mix and
    search parameters (&list=, &start_radio=, &pp=, &t=) are ignored.
    """
    try:
        u = urlparse(url.strip())
    except ValueError:
        return None
    host = (u.hostname or "").lower()
    if host in ("youtu.be", "www.youtu.be"):
        cand = u.path.lstrip("/").split("/")[0]
        return cand if _YT_ID.match(cand) else None
    if host in _YT_HOSTS:
        if u.path == "/watch":
            cand = parse_qs(u.query).get("v", [""])[0]
            return cand if _YT_ID.match(cand) else None
        parts = [p for p in u.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            return parts[1] if _YT_ID.match(parts[1]) else None
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(source: str) -> tuple[str, dict]:
    """Return (cache key, metadata) for a source without downloading anything.

    Keys: "yt-<id>" | "url-<extractor>-<id>" | "sha256-<hex64>".
    """
    s = source.strip().strip('"')
    p = Path(s)
    if not s.lower().startswith(("http://", "https://")) and p.exists():
        if not p.is_file():
            raise IngestError("not_a_file", f"{p} is not a file", "Choose an audio or video file.")
        return f"sha256-{sha256_file(p)}", {"source_type": "upload", "path": str(p.resolve()),
                                            "title": p.stem, "source_url": None}
    if not s.lower().startswith(("http://", "https://")):
        if "youtu" in s.lower() or s.startswith("www."):
            s = "https://" + s
        else:
            raise IngestError("bad_source", f"Not a link or an existing file: {source!r}",
                              "Paste a YouTube link, or choose an audio file.")
    vid = youtube_id(s)
    if vid:
        return f"yt-{vid}", {"source_type": "youtube", "id": vid,
                             "source_url": f"https://www.youtube.com/watch?v={vid}"}
    host = (urlparse(s).hostname or "").lower()
    if "deezer." in host:
        raise IngestError("drm", "Deezer links can't be analysed (the audio is DRM-protected).",
                          "Upload the audio file instead.")
    # Other sites: ask yt-dlp for the extractor and ID (one metadata request, no download).
    from .youtube import probe
    info = probe(s)
    extractor = re.sub(r"[^a-z0-9]+", "", (info.get("extractor_key") or "site").lower())
    ident = re.sub(r"[^A-Za-z0-9_-]+", "_", str(info.get("id") or hashlib.sha1(s.encode()).hexdigest()[:16]))
    return f"url-{extractor}-{ident}", {"source_type": "url", "id": ident, "source_url": s,
                                        "title": info.get("title"), "uploader": info.get("uploader"),
                                        "duration": info.get("duration")}
