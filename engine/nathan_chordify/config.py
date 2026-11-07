"""Paths, environment and process helpers.

Import this (via the package) before torch / huggingface_hub so the model caches land in
the project's models/ folder instead of the user profile.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # ...\Nathan_chordify
ENGINE_DIR = ROOT / "engine"
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "cache"
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
TOOLS_DIR = ROOT / "tools"
THIRD_PARTY_DIR = ROOT / "third_party"
SHARED_DIR = ROOT / "shared"
EVAL_DIR = ROOT / "eval"

FFMPEG_BIN_DIR = TOOLS_DIR / "ffmpeg" / "bin"
FFMPEG = FFMPEG_BIN_DIR / "ffmpeg.exe"
FFPROBE = FFMPEG_BIN_DIR / "ffprobe.exe"

# Hide console windows of child processes when the parent is pythonw.exe (DESIGN §9).
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

_env_done = False


def setup_env() -> None:
    """Set cache locations and quiet progress bars. Idempotent; call before importing torch."""
    global _env_done
    if _env_done:
        return
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
    os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Once the demucs weights are on disk, stop the HF Hub round-trip demucs makes on every load.
    if (MODELS_DIR / "hf" / "hub" / "models--adefossez--HTDemucs-ft").is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if FFMPEG_BIN_DIR.is_dir():
        path = os.environ.get("PATH", "")
        if str(FFMPEG_BIN_DIR) not in path.split(os.pathsep):
            os.environ["PATH"] = str(FFMPEG_BIN_DIR) + os.pathsep + path
    _env_done = True


def ensure_stdio(log_name: str) -> None:
    """Under pythonw.exe sys.stdout/stderr are None and uvicorn's logging crashes (review F1).

    Redirect both to a line-buffered log file in logs/.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stream = open(LOGS_DIR / log_name, "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
