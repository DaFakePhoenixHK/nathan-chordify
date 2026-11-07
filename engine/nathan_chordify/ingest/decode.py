"""ffmpeg decoding (DESIGN §5.1 step 5). Every call hides its console window under pythonw."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .. import config
from .errors import IngestError


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          creationflags=config.NO_WINDOW)


def probe_duration(path: Path) -> float:
    r = _run([str(config.FFPROBE), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)])
    if r.returncode != 0:
        raise IngestError("decode_failed", f"ffprobe could not read {path.name}.", "Is it an audio or video file?")
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except (KeyError, ValueError, TypeError) as e:
        raise IngestError("decode_failed", f"No duration found in {path.name}.", "Is it an audio or video file?") from e


def to_wav(src: Path, wav: Path) -> None:
    """Any ffmpeg-readable input → 44.1 kHz stereo PCM16 WAV (the analysis input)."""
    tmp = wav.with_suffix(".tmp.wav")
    r = _run([str(config.FFMPEG), "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-vn",
              "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(tmp)])
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise IngestError("decode_failed", f"ffmpeg could not decode {src.name}: {r.stderr.strip()[-200:]}",
                          "Try another file format (mp3, m4a, wav, flac).")
    tmp.replace(wav)


def to_play_m4a(wav: Path, m4a: Path) -> None:
    """AAC 192k with faststart, for the web player (HTTP range requests)."""
    tmp = m4a.with_suffix(".tmp.m4a")
    r = _run([str(config.FFMPEG), "-nostdin", "-y", "-loglevel", "error", "-i", str(wav),
              "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)])
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise IngestError("decode_failed", f"ffmpeg could not encode the player audio: {r.stderr.strip()[-200:]}")
    tmp.replace(m4a)
