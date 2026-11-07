"""Environment smoke test (DESIGN §12). Every check prints PASS/FAIL with a fix hint.

Usage:  .venv\\Scripts\\python scripts\\smoke_test.py [--quick] [--no-net]
  --quick   skip the GPU model runs (demucs, Beat This!, lv-chordia)
  --no-net  skip the yt-dlp check
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from nathan_chordify import config  # noqa: E402

TEST_URL = "https://www.youtube.com/watch?v=Y7boUFbM76k&list=RDY7boUFbM76k&start_radio=1"
results: list[tuple[str, bool, str]] = []


def check(name: str, hint: str = ""):
    def deco(fn):
        def run():
            t0 = time.perf_counter()
            try:
                msg = fn() or ""
                results.append((name, True, msg))
                print(f"PASS  {name}  ({time.perf_counter() - t0:.1f}s) {msg}", flush=True)
            except Exception as e:  # noqa: BLE001
                results.append((name, False, str(e)))
                print(f"FAIL  {name}: {e}\n      hint: {hint}", flush=True)
                traceback.print_exc(limit=2)
        run.check_name = fn.__name__
        return run
    return deco


@check("disk space >= 15 GB free", "free space on C: (delete old cache/ songs or other files)")
def disk():
    free = shutil.disk_usage(config.ROOT).free / 1e9
    assert free >= 15, f"only {free:.1f} GB free"
    return f"{free:.1f} GB free"


@check("torchaudio MelSpectrogram on CUDA", "fallback: pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130")
def torchaudio_cuda():
    import torch
    import torchaudio
    y = torchaudio.transforms.MelSpectrogram().cuda()(torch.randn(1, 22050, device="cuda"))
    assert y.is_cuda
    return f"torch {torch.__version__}, torchaudio {torchaudio.__version__}"


@check("CUDA device", "update the NVIDIA driver; check torch.version.cuda")
def cuda():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    name = torch.cuda.get_device_name(0)
    assert "4070" in name, name
    return f"{name}, CUDA {torch.version.cuda}"


@check("ffmpeg + ffprobe (tools\\ffmpeg\\bin)", "re-run the ffmpeg step of the install")
def ffmpeg():
    for exe in (config.FFMPEG, config.FFPROBE):
        out = subprocess.run([str(exe), "-hide_banner", "-version"], capture_output=True, text=True,
                             creationflags=config.NO_WINDOW, check=True).stdout.splitlines()[0]
    return out.split(" Copyright")[0]


@check("deno (JS runtime for yt-dlp)", "pip install deno==2.9.7")
def deno_rt():
    import deno
    out = subprocess.run([deno.find_deno_bin(), "--version"], capture_output=True, text=True,
                         creationflags=config.NO_WINDOW, check=True).stdout.splitlines()[0]
    return out


@check("imports of pinned packages", "pip install -r engine\\requirements.txt -c engine\\constraints.txt")
def imports():
    mods = ["numpy", "scipy", "numba", "soundfile", "soxr", "librosa", "mir_eval", "pretty_midi", "mido",
            "music21", "einops", "demucs", "huggingface_hub", "safetensors", "beat_this", "lv_chordia",
            "piano_transcription_inference", "torchlibrosa", "lightning", "torchmetrics", "gin", "jams",
            "pumpp", "fastapi", "uvicorn", "multipart", "pydantic", "webview", "yt_dlp"]
    bad = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{m} ({type(e).__name__}: {e})")
    assert not bad, "; ".join(bad)
    return f"{len(mods)} modules"


@check("librosa CQT with numpy 2.5 / scipy 1.18", "pin librosa / scipy per review-feasibility.md")
def librosa_cqt():
    import librosa
    import numpy as np
    y = np.random.default_rng(0).standard_normal(22050 * 5).astype(np.float32) * 0.1
    C = librosa.cqt(y, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12)
    assert C.shape[0] == 84


def drum_track(bpm: float, seconds: float, sr: int = 22050):
    """Kick on 1 and 3, snare on 2 and 4, closed hat on 8ths, plus a bass note per bar (a pop groove)."""
    import numpy as np
    rng = np.random.default_rng(1)
    y = np.zeros(int(seconds * sr), np.float32)
    beat = 60.0 / bpm

    def add(t, sig):
        s = int(t * sr)
        e = min(len(y), s + len(sig))
        if s < len(y):
            y[s:e] += sig[:e - s]

    tk = np.arange(int(0.25 * sr)) / sr
    kick = (np.sin(2 * np.pi * (50 + 80 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 12)).astype(np.float32)
    ns = int(0.15 * sr)
    snare = (rng.standard_normal(ns) * np.exp(-np.arange(ns) / sr * 25)).astype(np.float32) * 0.5
    nh = int(0.04 * sr)
    hat = (np.diff(rng.standard_normal(nh + 1)) * np.exp(-np.arange(nh) / sr * 90)).astype(np.float32) * 0.15
    tb = np.arange(int(4 * beat * sr)) / sr
    bass_notes = [55.0, 41.2, 49.0, 36.7]
    i, t = 0, 0.0
    while t < seconds - beat:
        if i % 4 == 0:
            f0 = bass_notes[(i // 4) % 4]
            add(t, (0.3 * np.sin(2 * np.pi * f0 * tb) * np.exp(-tb * 0.8)).astype(np.float32))
        add(t, kick if i % 2 == 0 else snare)
        add(t, hat)
        add(t + beat / 2, hat)
        t += beat
        i += 1
    return y / max(1e-6, float(np.abs(y).max())) * 0.8, sr


@check("Beat This! on a 100 BPM drum groove", "check beat_this install and the final0 checkpoint")
def beat_this_check():
    import numpy as np
    from beat_this.inference import Audio2Beats
    y, sr = drum_track(100.0, 30.0)
    a2b = Audio2Beats(checkpoint_path="final0", device="cuda", dbn=False)
    beats, downbeats = a2b(y, sr)
    bpm = 60.0 / float(np.median(np.diff(beats)))
    ok = any(abs(bpm - b) <= 0.04 * b for b in (50, 100, 200))   # tempo octave is resolved later in rhythm.py
    assert ok, f"bpm {bpm:.1f}"
    return f"{len(beats)} beats, {len(downbeats)} downbeats, {bpm:.1f} BPM"


@check("demucs htdemucs_ft on 10 s (CUDA)", "run scripts\\fetch_models.py; check VRAM")
def demucs_check():
    import numpy as np
    import torch
    from demucs.api import Separator
    sep = Separator(model="htdemucs_ft", device="cuda")
    sr = sep.samplerate
    t = np.arange(sr * 10) / sr
    x = (0.2 * np.sin(2 * np.pi * 110 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    mix = torch.from_numpy(np.stack([x, x]))          # stays on CPU (review F10)
    torch.cuda.reset_peak_memory_stats()
    _, stems = sep.separate_tensor(mix, sr)
    peak = torch.cuda.max_memory_allocated() / 1e9
    assert set(stems) >= {"vocals", "drums", "bass", "other"}, list(stems)
    return f"stems {sorted(stems)}, peak VRAM {peak:.2f} GB"


@check("lv-chordia weights present", "reinstall lv-chordia (it silently uses RANDOM weights if missing)")
def lv_weights():
    files = list((Path(sys.prefix) / "share" / "lv-chordia" / "cache_data").glob("*.sdict"))
    assert len(files) >= 5, f"found {len(files)} .sdict files"
    return f"{len(files)} checkpoints"


@check("yt-dlp resolves the test song (Mix link -> single video)", "Update downloader; pause the VPN; or upload the file")
def ytdlp():
    import deno
    import yt_dlp
    opts = {"quiet": True, "noplaylist": True, "skip_download": True,
            "js_runtimes": {"deno": {"path": deno.find_deno_bin()}, "node": {}}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(TEST_URL, download=False)
    assert info.get("id") == "Y7boUFbM76k", info.get("id")
    n_audio = sum(1 for f in info.get("formats", []) if f.get("vcodec") == "none")
    return f"'{info.get('title')}', {info.get('duration')} s, {n_audio} audio formats"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-net", action="store_true")
    ap.add_argument("--only", help="run only the check functions named here (comma-separated)")
    a = ap.parse_args()
    checks = [disk, torchaudio_cuda, cuda, ffmpeg, deno_rt, imports, librosa_cqt, lv_weights]
    if not a.quick:
        checks += [beat_this_check, demucs_check]
    if not a.no_net:
        checks.append(ytdlp)
    if a.only:
        everything = {f.check_name: f for f in
                      [disk, torchaudio_cuda, cuda, ffmpeg, deno_rt, imports, librosa_cqt, lv_weights,
                       beat_this_check, demucs_check, ytdlp]}
        checks = [everything[w] for w in a.only.split(",")]
    for c in checks:
        c()
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
