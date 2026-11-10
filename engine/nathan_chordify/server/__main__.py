"""`python -m nathan_chordify.server [--host 127.0.0.1] [--port 8765] [--stub] [--no-prewarm]
[--cache-dir DIR] [--data-dir DIR] [--eval-dir DIR]`

The directory options run a sandboxed server (e.g. a UI test on a copy of the cache, so edits and
ground-truth exports don't touch the real song cache, library or eval/ files).

Runs the API + web UI with one spawned GPU worker. Under pythonw.exe stdout/stderr go to
logs\\server.log (uvicorn's logging crashes on a None stdout, DESIGN §9). All start-up code stays
under the `__main__` guard because the worker is started with multiprocessing "spawn".
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request


def _already_running(host: str, port: int) -> bool:
    probe = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        with urllib.request.urlopen(f"http://{probe}:{port}/health", timeout=0.5) as r:
            return json.load(r).get("status") == "ok"
    except (OSError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    from .. import config
    config.ensure_stdio("server.log")
    ap = argparse.ArgumentParser(prog="python -m nathan_chordify.server", description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--stub", action="store_true", help="no GPU: answer every job with the 回家 fixture (UI work)")
    ap.add_argument("--no-prewarm", action="store_true", help="load the models on the first job instead")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", help="song cache (default: <repo>/cache)")
    ap.add_argument("--data-dir", help="library database folder (default: <repo>/data)")
    ap.add_argument("--eval-dir", help="ground-truth export root with gt/ and songs.yaml (default: <repo>/eval)")
    ap.add_argument("--public-url", help="the HTTPS address phones use to reach this server (shown as a QR code)")
    args = ap.parse_args(argv)
    from pathlib import Path
    for opt, attr in ((args.cache_dir, "CACHE_DIR"), (args.data_dir, "DATA_DIR"), (args.eval_dir, "EVAL_DIR")):
        if opt:
            setattr(config, attr, Path(opt).resolve())

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if _already_running(args.host, args.port):
        print(f"A Nathan Chordify server already answers on port {args.port}.", flush=True)
        return 0

    import uvicorn

    from .app import STUB_ANALYZE, ServerSettings, create_app
    settings = ServerSettings(prewarm=not (args.no_prewarm or args.stub), device=args.device, public_url=args.public_url,
                              port=args.port)
    if args.stub:
        settings.analyze = STUB_ANALYZE
    if args.host not in ("127.0.0.1", "localhost", "0.0.0.0", "::"):
        settings.allowed_hosts = (*settings.allowed_hosts, args.host)
    print(f"Nathan Chordify on http://{args.host}:{args.port}  (stub={args.stub})", flush=True)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info",
                timeout_graceful_shutdown=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
