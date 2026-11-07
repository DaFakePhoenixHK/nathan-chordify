"""Practice mixes of the cached Demucs stems (user's request 1, 2026-11-04: "play without the singer")
and pitch-shifted copies of them (user's request, 2026-11-04: "select between the original audio and the
transposed audio").

    play.m4a                original (made by the pipeline)
    play_novocals.m4a       bass + drums + other   = everything but the singer
    play_harmony.m4a        bass + other           = no singer, no drums ("band only")
    play_t-3.m4a            the original moved down 3 semitones (same speed)
    play_novocals_t+2.m4a   the "no singer" mix moved up 2 semitones, …

A mix is made on its first request with ONE ffmpeg call (``amix`` of the FLAC stems → AAC 192k with
faststart, like ``ingest.decode.to_play_m4a``) and cached next to play.m4a. A shifted copy is made from
the mix file with ffmpeg's ``rubberband`` filter (pitch only, speed unchanged; ~15 s for a 5-minute song)
and cached the same way; a song keeps at most ``MAX_SHIFTED`` shifted files (the least recently used go).
A per-file lock stops two requests from making the same file at once; the second request waits and gets
the finished file. Songs analysed without separation (fast mode) have no stems → ``NoStems``.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
from pathlib import Path

from .. import config
from ..separation import available

MIXES: tuple[str, ...] = ("original", "novocals", "harmony")
FILES: dict[str, str] = {"original": "play.m4a", "novocals": "play_novocals.m4a", "harmony": "play_harmony.m4a"}
STEMS: dict[str, tuple[str, ...]] = {"novocals": ("bass", "drums", "other"), "harmony": ("bass", "other")}
#: a single stem file that already IS the mix (harmonic.flac = bass + other), used when present
SHORTCUT: dict[str, str] = {"harmony": "harmonic"}

NO_STEMS_MSG = "This song was analysed without separating the voice."
NO_STEMS_HINT = "Re-analyse it to use this (the download is reused; only the separation runs)."

#: pitch shift of the player audio, in semitones (a wider move is the same key an octave away)
TRANSPOSE_MIN, TRANSPOSE_MAX = -6, 6
#: shifted files kept per song; beyond this the least recently used one is deleted
MAX_SHIFTED = 12
SHIFT_RE = re.compile(r"^play(?:_(novocals|harmony))?_t([+-]\d+)\.m4a$")


class NoStems(Exception):
    """The song has no separated stems (fast mode, or stems removed by the storage clean-up)."""


class MixError(Exception):
    """ffmpeg could not write the mix; the message holds the tail of its stderr."""


class ShiftError(Exception):
    """ffmpeg could not write the pitch-shifted copy; the message holds the tail of its stderr."""


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(path: Path) -> threading.Lock:
    k = str(path.resolve()).lower()
    with _locks_guard:
        if k not in _locks:
            _locks[k] = threading.Lock()
        return _locks[k]


def mix_path(song_dir: Path, mix: str) -> Path:
    return Path(song_dir) / FILES[mix]


def shift_path(song_dir: Path, mix: str, transpose: int) -> Path:
    """``play_t-3.m4a`` / ``play_novocals_t+2.m4a`` …; ``transpose`` 0 is the mix file itself."""
    if transpose == 0:
        return mix_path(song_dir, mix)
    stem = FILES[mix][: -len(".m4a")]
    return Path(song_dir) / f"{stem}_t{transpose:+d}.m4a"


def parse_shift_name(name: str) -> tuple[str, int] | None:
    """(mix, transpose) of a shifted file's name, or None."""
    m = SHIFT_RE.match(name)
    if not m:
        return None
    return (m.group(1) or "original"), int(m.group(2))


def stem_inputs(song_dir: Path, mix: str) -> list[Path] | None:
    """The stem files to mix for ``mix``, or None when one is missing (``available`` also rejects
    the ``.absent`` markers of fast mode)."""
    d = Path(song_dir) / "stems"
    short = SHORTCUT.get(mix)
    if short and available(d / f"{short}.flac"):
        return [d / f"{short}.flac"]
    paths = [d / f"{s}.flac" for s in STEMS[mix]]
    return paths if all(available(p) for p in paths) else None


def encode(inputs: list[Path], out: Path) -> None:
    """One ffmpeg call: sum the stems (no per-input attenuation: the stems add up to the mix) → AAC m4a.
    Module-level so tests can replace it."""
    tmp = out.with_suffix(".tmp.m4a")
    args = [str(config.FFMPEG), "-nostdin", "-y", "-loglevel", "error"]
    for p in inputs:
        args += ["-i", str(p)]
    if len(inputs) > 1:
        chain = "".join(f"[{i}:a]" for i in range(len(inputs)))
        args += ["-filter_complex", f"{chain}amix=inputs={len(inputs)}:duration=longest:normalize=0[m]", "-map", "[m]"]
    args += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=config.NO_WINDOW)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise MixError(f"ffmpeg could not make the mix: {r.stderr.strip()[-300:]}")
    tmp.replace(out)


def _ffmpeg_shift(src: Path, tmp: Path, filt: str) -> subprocess.CompletedProcess:
    args = [str(config.FFMPEG), "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-af", filt,
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)]
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          creationflags=config.NO_WINDOW)


def shift_filters(transpose: int) -> list[str]:
    """ffmpeg audio filter chains to try in order for a move of ``transpose`` semitones: ``rubberband``
    in its pitch-quality mode, plain ``rubberband`` (a build without the option), then ``asetrate`` +
    ``aresample`` + ``atempo`` (a build without the filter: coarser, but the right key and speed)."""
    f = 2.0 ** (transpose / 12.0)
    return [
        f"rubberband=pitch={f:.10f}:pitchq=quality",
        f"rubberband=pitch={f:.10f}",
        f"aresample=48000,asetrate={48000 * f:.3f},aresample=48000,atempo={1 / f:.10f}",
    ]


def shift(src: Path, out: Path, transpose: int) -> None:
    """Move the pitch of ``src`` by ``transpose`` semitones (speed unchanged) → AAC m4a at ``out``.
    Module-level so tests can replace it."""
    tmp = out.with_suffix(".tmp.m4a")
    err = ""
    for filt in shift_filters(transpose):
        r = _ffmpeg_shift(src, tmp, filt)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(out)
            return
        err = r.stderr.strip()[-300:]
        tmp.unlink(missing_ok=True)
    raise ShiftError(f"ffmpeg could not shift the audio: {err}")


def ensure(song_dir: Path, mix: str, transpose: int = 0) -> Path:
    """The player file for ``mix`` moved by ``transpose`` semitones: made on the first call, then cached.
    ``original`` at 0 is never made here (the pipeline writes play.m4a)."""
    if mix not in MIXES:
        raise ValueError(f"unknown mix {mix!r}")
    if not TRANSPOSE_MIN <= transpose <= TRANSPOSE_MAX:
        raise ValueError(f"transpose {transpose} outside {TRANSPOSE_MIN}..{TRANSPOSE_MAX}")
    out = mix_path(song_dir, mix)
    if mix != "original" and not out.exists():
        with _lock(out):
            if not out.exists():               # else: made by the request we waited for
                inputs = stem_inputs(song_dir, mix)
                if inputs is None:
                    raise NoStems(NO_STEMS_MSG)
                encode(inputs, out)
    if transpose == 0:
        return out
    shifted = shift_path(song_dir, mix, transpose)
    if shifted.exists():
        _touch(shifted)
        return shifted
    with _lock(shifted):
        if shifted.exists():
            _touch(shifted)
            return shifted
        shift(out, shifted, transpose)
    prune_shifted(song_dir, keep=shifted)
    return shifted


def _touch(p: Path) -> None:
    """Bump the modification time: the least recently USED file is the one pruned (Windows does not
    keep access times reliably)."""
    try:
        os.utime(p, None)
    except OSError:
        pass


def shifted_files(song_dir: Path) -> list[Path]:
    return [p for p in Path(song_dir).glob("play*_t*.m4a") if parse_shift_name(p.name)]


def prune_shifted(song_dir: Path, keep: Path | None = None, limit: int = MAX_SHIFTED) -> list[Path]:
    """Delete the least recently used shifted files beyond ``limit`` (never ``keep``; a file in use on
    Windows cannot be deleted and is left for next time). Returns what was removed."""
    files = shifted_files(song_dir)
    if len(files) <= limit:
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    removed: list[Path] = []
    for p in files[limit:]:
        if keep is not None and p.resolve() == keep.resolve():
            continue
        try:
            p.unlink()
            removed.append(p)
        except OSError:
            pass
    return removed


def status(song_dir: Path) -> dict:
    """{mix: "ready" | "makeable" | "no_stems"} for the UI."""
    out = {}
    for m in MIXES:
        if mix_path(song_dir, m).exists():
            out[m] = "ready"
        elif m != "original" and stem_inputs(song_dir, m) is not None:
            out[m] = "makeable"
        else:
            out[m] = "no_stems" if m != "original" else "missing"
    return out


def transposes_ready(song_dir: Path) -> dict[str, list[int]]:
    """{mix: [semitones, …]} of the shifted copies already on disk (sorted)."""
    out: dict[str, list[int]] = {}
    for p in shifted_files(song_dir):
        parsed = parse_shift_name(p.name)
        if parsed:
            out.setdefault(parsed[0], []).append(parsed[1])
    return {m: sorted(v) for m, v in sorted(out.items())}
