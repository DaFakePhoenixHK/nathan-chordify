"""The phone link: a free Cloudflare "quick tunnel" that gives this server a public HTTPS address.

No account is needed. The address is random and changes every time the link starts, so the phone
scans a new code then. Every request through it must carry the phone code (remote.RemoteGuard).
cloudflared only connects outwards over HTTPS (port 443): it works with the VPN on and needs no
firewall change. On Windows it is tied to this process, so it can't outlive the server.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from .. import config

log = logging.getLogger("nathan_chordify.server.tunnel")

EXE = config.TOOLS_DIR / "cloudflared" / "cloudflared.exe"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
READY_MARK = "Registered tunnel connection"
STATE_FILE = "phone_link.txt"              # "on" when the link should start with the server


def wanted(data_dir: Path) -> bool:
    try:
        return (Path(data_dir) / STATE_FILE).read_text("ascii").strip() == "on"
    except (OSError, UnicodeError):
        return False


def remember(data_dir: Path, on: bool) -> None:
    p = Path(data_dir) / STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("on" if on else "off", "ascii")


def _kill_with_us(proc: subprocess.Popen) -> object | None:
    """Windows: put the child in a job that closes (and kills it) when this process ends."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)

        class Basic(ctypes.Structure):
            _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                        ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", wintypes.DWORD),
                        ("f", ctypes.c_size_t), ("g", wintypes.DWORD), ("h", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", ctypes.c_uint64 * 6), ("p1", ctypes.c_size_t),
                        ("p2", ctypes.c_size_t), ("p3", ctypes.c_size_t), ("p4", ctypes.c_size_t)]

        info = Extended()
        info.basic.LimitFlags = 0x2000                      # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
        k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(proc._handle))
        return job
    except Exception:                                       # best effort: stop() still ends it
        log.warning("couldn't tie cloudflared to the server process", exc_info=True)
        return None


class Tunnel:
    """Starts cloudflared, reads its public address, restarts it if it stops."""

    def __init__(self, port: int, exe: Path = EXE, popen: Callable = subprocess.Popen, retry_s: float = 3.0):
        self.port, self.exe, self.popen, self.retry_s = port, Path(exe), popen, retry_s
        self.state = "off"                                  # off | starting | on | error
        self.url: str | None = None
        self.error: str | None = None
        self._proc = None
        self._job = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def status(self) -> dict:
        return {"state": self.state, "url": self.url if self.state == "on" else None, "error": self.error}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self.exe.exists():
            self.state, self.error = "error", "The phone link program is missing (tools/cloudflared/cloudflared.exe)."
            return
        self._stop.clear()
        self.state, self.error = "starting", None
        self._thread = threading.Thread(target=self._run, name="nchord-tunnel", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._thread:
            self._thread.join(8)
        self.state, self.url, self.error = "off", None, None

    def _run(self) -> None:
        wait = self.retry_s
        while not self._stop.is_set():
            self.state, self.url = "starting", None
            cmd = [str(self.exe), "tunnel", "--no-autoupdate", "--protocol", "http2",
                   "--url", f"http://127.0.0.1:{self.port}"]
            try:
                self._proc = self.popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                        creationflags=config.NO_WINDOW)
            except OSError as e:
                self.state, self.error = "error", f"The phone link couldn't start: {e}"
                return
            self._job = _kill_with_us(self._proc)
            found = None
            for line in self._proc.stderr:
                m = URL_RE.search(line)
                if m and not found:
                    found = m.group(0)
                if found and READY_MARK in line and self.state != "on":
                    self.url, self.state, self.error = found, "on", None
                    wait = self.retry_s
                    log.info("phone link on: %s", found)
            self._proc.wait()
            if self._stop.is_set():
                break
            self.state, self.url = "error", None
            self.error = "The phone link stopped. Trying again…"
            log.warning("cloudflared ended (code %s); retrying in %.0f s", self._proc.returncode, wait)
            if self._stop.wait(wait):
                break
            wait = min(60.0, wait * 2)
