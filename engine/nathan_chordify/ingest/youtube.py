"""yt-dlp wrapper: one download per song, error codes for the fallback ladder, and rate-limit back-off.

Standing rule (CLAUDE.md): never suggest pausing the VPN. On HTTP 429 / bot checks we stop retrying
for BACKOFF_S, because every extra request extends YouTube's block.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .. import config
from .errors import IngestError

BACKOFF_S = 30 * 60
MAX_LEN_S = 30 * 60
MAX_BYTES = 1_000_000_000
_BACKOFF_FILE = config.CACHE_DIR / ".ytdlp_backoff.json"


def _base_opts() -> dict:
    import deno
    return {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True,
            "js_runtimes": {"deno": {"path": deno.find_deno_bin()}, "node": {}},
            "retries": 3, "socket_timeout": 20}


def backoff_remaining() -> float:
    try:
        until = json.loads(_BACKOFF_FILE.read_text("utf-8"))["until"]
    except (OSError, ValueError, KeyError):
        return 0.0
    return max(0.0, until - time.time())


def _set_backoff() -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _BACKOFF_FILE.write_text(json.dumps({"until": time.time() + BACKOFF_S}), "utf-8")


def _check_backoff(force: bool) -> None:
    left = backoff_remaining()
    if left > 0 and not force:
        raise IngestError("backoff", f"YouTube is rate-limiting this connection. Try again in about "
                                     f"{int(left // 60) + 1} min.",
                          "Waiting avoids extending the block. You can also upload the audio file instead.")


def _classify(err: Exception) -> IngestError:
    msg = str(err)
    low = msg.lower()
    if "429" in msg or "not a bot" in low or "sign in to confirm" in low:
        _set_backoff()
        return IngestError("rate_limited", "YouTube is rate-limiting this connection (bot check).",
                           "Try again in 30-60 min, or upload the audio file instead.")
    if "403" in msg:
        return IngestError("forbidden", "YouTube refused the audio stream (HTTP 403).",
                           "Update the downloader (Settings > Update downloader), then retry.")
    if any(k in low for k in ("video unavailable", "private video", "removed", "copyright", "members-only")):
        return IngestError("unavailable", "This video isn't available.", "Check the link, or upload the file.")
    if any(k in low for k in ("timed out", "getaddrinfo", "connection", "network is unreachable")):
        return IngestError("network", "Network error while contacting the site.", "Check the connection and retry.")
    return IngestError("download_failed", f"Download failed: {msg.splitlines()[-1][:300]}",
                       "Update the downloader, or upload the audio file instead.")


def probe(url: str, force: bool = False) -> dict:
    """Metadata only (one request, no download)."""
    import yt_dlp
    _check_backoff(force)
    try:
        with yt_dlp.YoutubeDL({**_base_opts(), "skip_download": True}) as ydl:
            return ydl.extract_info(url, download=False) or {}
    except yt_dlp.utils.DownloadError as e:
        raise _classify(e) from e


def download(url: str, dest_dir: Path, progress: Callable[[str, float, str], None] | None = None,
             max_len_s: int = MAX_LEN_S, force: bool = False) -> tuple[Path, dict]:
    """Download best audio to dest_dir/source.<ext>. Returns (path, metadata)."""
    import yt_dlp
    _check_backoff(force)
    dest_dir.mkdir(parents=True, exist_ok=True)

    def hook(d: dict) -> None:
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress("download", min(1.0, d.get("downloaded_bytes", 0) / total), "Downloading audio")

    opts = {**_base_opts(), "format": "bestaudio/best", "outtmpl": str(dest_dir / "source.%(ext)s"),
            "ffmpeg_location": str(config.FFMPEG_BIN_DIR), "max_filesize": MAX_BYTES,
            "match_filter": yt_dlp.utils.match_filter_func(f"duration <= {max_len_s}"),
            "progress_hooks": [hook]}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True) or {}
    except yt_dlp.utils.DownloadError as e:
        raise _classify(e) from e
    files = [p for p in dest_dir.glob("source.*") if p.suffix not in (".part", ".ytdl", ".json")]
    if not files:
        dur = info.get("duration")
        if dur and dur > max_len_s:
            raise IngestError("too_long", f"The video is {dur / 60:.0f} min; the limit is {max_len_s // 60} min.",
                              "Raise the limit in Settings, or upload a shorter excerpt.")
        raise IngestError("download_failed", "The download finished without an audio file.",
                          "Update the downloader, or upload the audio file instead.")
    meta = {k: info.get(k) for k in ("id", "title", "uploader", "duration", "thumbnail", "upload_date",
                                     "extractor_key", "webpage_url")}
    return files[0], meta
