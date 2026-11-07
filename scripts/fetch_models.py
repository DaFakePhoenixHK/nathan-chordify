"""Download / verify model weights into Nathan_chordify\\models and record them in MANIFEST.json.

Usage:
    .venv\\Scripts\\python scripts\\fetch_models.py            # models needed for M0-M4
    .venv\\Scripts\\python scripts\\fetch_models.py --all      # also the M5 models (piano transcription)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from nathan_chordify import config  # noqa: E402  (sets HF_HOME / TORCH_HOME before torch loads)

PIANO_URL = ("https://zenodo.org/record/4034264/files/"
             "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1")
PIANO_FILE = config.MODELS_DIR / "piano_transcription" / "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
PIANO_MD5 = "22b961b77c1878239fec963362097045"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_demucs() -> list[Path]:
    from demucs.pretrained import get_model
    get_model("htdemucs_ft")
    hub = Path(config.MODELS_DIR / "hf")
    return sorted(p for p in hub.rglob("*") if p.is_file() and "HTDemucs" in str(p) and p.suffix in {".safetensors", ".yaml", ".th"})


def fetch_beat_this() -> list[Path]:
    from beat_this.inference import load_model
    load_model("final0", device="cpu")
    root = Path(config.MODELS_DIR / "torch")
    return sorted(p for p in root.rglob("*.ckpt") if p.is_file())


def check_lv_chordia() -> list[Path]:
    cache = Path(sys.prefix) / "share" / "lv-chordia" / "cache_data"
    files = sorted(cache.glob("*.sdict"))
    if len(files) < 5:
        raise SystemExit(f"lv-chordia weights missing in {cache} (found {len(files)}, need 5). "
                         "Without them it silently runs with RANDOM weights. Reinstall lv-chordia.")
    return files


def fetch_piano() -> list[Path]:
    PIANO_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not PIANO_FILE.exists() or md5(PIANO_FILE) != PIANO_MD5:
        tmp = PIANO_FILE.with_suffix(".part")
        urllib.request.urlretrieve(PIANO_URL, tmp)
        if md5(tmp) != PIANO_MD5:
            tmp.unlink()
            raise SystemExit("piano transcription checkpoint: md5 mismatch")
        tmp.replace(PIANO_FILE)
    return [PIANO_FILE]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also fetch the M5 models")
    args = ap.parse_args()

    steps = [("htdemucs_ft", fetch_demucs), ("beat_this_final0", fetch_beat_this), ("lv_chordia", check_lv_chordia)]
    if args.all:
        steps.append(("piano_transcription", fetch_piano))

    manifest_path = config.MODELS_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    for name, fn in steps:
        print(f"[{name}] ...", flush=True)
        files = fn()
        if not files:
            raise SystemExit(f"[{name}] no weight files found after download")
        manifest[name] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
        print(f"[{name}] OK: {len(files)} file(s), {sum(p.stat().st_size for p in files) / 1e6:.1f} MB", flush=True)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), "utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
