"""M0 spikes (DESIGN §14.1): (a) download 回家, (b) lv-chordia per-head probabilities, (c) demucs + Beat This!.

Writes everything under cache/yt-Y7boUFbM76k/ and prints a timing / VRAM report.
Usage:  .venv\\Scripts\\python scripts\\spikes\\m0_spike.py [URL]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
from nathan_chordify import config  # noqa: E402

import numpy as np  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=Y7boUFbM76k&list=RDY7boUFbM76k&start_radio=1"
report: dict = {"timings_s": {}, "vram_peak_gb": {}}
PCS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
TRIADS = ["maj", "min", "sus4", "sus2", "dim", "aug"]


def timed(name):
    def deco(fn):
        def run(*a, **k):
            import torch
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            report["timings_s"][name] = round(time.perf_counter() - t0, 1)
            report["vram_peak_gb"][name] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            print(f"[{name}] {report['timings_s'][name]} s, peak VRAM {report['vram_peak_gb'][name]} GB", flush=True)
            return out
        return run
    return deco


@timed("download+decode")
def download() -> tuple[Path, dict]:
    from nathan_chordify import ingest
    asset = ingest.fetch(URL)                      # cache-first: no network when the song is cached
    return asset.wav, {"key": asset.key, "title": asset.title, "artist": asset.artist,
                       "duration": asset.duration_s}


@timed("demucs htdemucs_ft")
def separate(wav: Path) -> dict[str, Path]:
    import soundfile as sf
    import torch
    from demucs.api import Separator
    out_dir = wav.parent / "stems"
    names = ["vocals", "drums", "bass", "other", "harmonic"]
    paths = {n: out_dir / f"{n}.flac" for n in names}
    if all(p.exists() for p in paths.values()):
        return paths
    out_dir.mkdir(exist_ok=True)
    x, sr = sf.read(wav, dtype="float32", always_2d=True)           # [N, 2]
    sep = Separator(model="htdemucs_ft", device="cuda")
    mix = torch.from_numpy(x.T.copy())                                # CPU tensor (review F10)
    _, stems = sep.separate_tensor(mix, sr)
    for n, t in stems.items():
        sf.write(paths[n], t.numpy().T, sr, subtype="PCM_16")
    harm = (stems["bass"] + stems["other"]).numpy().T
    sf.write(paths["harmonic"], np.clip(harm, -1, 1), sr, subtype="PCM_16")
    return paths


@timed("beat_this")
def beats(wav: Path) -> dict:
    import soundfile as sf
    from beat_this.inference import Audio2Beats
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    b, db = Audio2Beats(checkpoint_path="final0", device="cuda", dbn=False)(x.mean(axis=1), sr)
    ibi = float(np.median(np.diff(b)))
    bar_len = float(np.median(np.diff(db))) if len(db) > 1 else float("nan")
    bpb = np.round(np.diff(db) / ibi).astype(int) if len(db) > 1 else np.array([])
    vals, counts = np.unique(bpb, return_counts=True)
    return {"beats": b.tolist(), "downbeats": db.tolist(), "bpm": round(60 / ibi, 2),
            "median_bar_s": round(bar_len, 3),
            "beats_per_bar_hist": {int(v): int(c) for v, c in zip(vals, counts)}}


def lv_cqt(path: Path) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=22050, mono=True)
    C = librosa.hybrid_cqt(y, sr=22050, bins_per_octave=36, fmin=librosa.note_to_hz("F#0"),
                           n_bins=288, tuning=None, hop_length=512).T
    return np.abs(C).astype(np.float32)


@timed("lv-chordia heads x2")
def lv_heads(paths: dict[str, Path]) -> dict[str, list[np.ndarray]]:
    from lv_chordia.chord_recognition import MODEL_NAMES
    from lv_chordia.chordnet_ismir_naive import ChordNet
    from lv_chordia.mir.nn.train import NetworkInterface
    nets = [NetworkInterface(ChordNet(None), n, load_checkpoint=False) for n in MODEL_NAMES]
    out = {}
    for name, p in paths.items():
        cqt = lv_cqt(p)
        per_model = [net.inference(cqt) for net in nets]
        out[name] = [np.mean([m[i] for m in per_model], axis=0) for i in range(6)]
    return out


@timed("lv-chordia baseline B1 (own CRF/HMM)")
def lv_baseline(wav: Path) -> list[dict]:
    from lv_chordia.chord_recognition import chord_recognition
    return chord_recognition(str(wav.resolve()), "submission")


def frame_label(triad_p: np.ndarray, bass_p: np.ndarray, sev_p: np.ndarray) -> str:
    i = int(triad_p.argmax())
    if i == 0:
        return "N"
    q, r = divmod(i - 1, 12)
    lab = PCS[r] + ("" if TRIADS[q] == "maj" else "m" if TRIADS[q] == "min" else TRIADS[q])
    s = int(sev_p.argmax())
    lab += {0: "", 1: "maj7", 2: "7", 3: "°7"}[s] if not (TRIADS[q] == "min" and s == 2) else "7"
    b = int(bass_p.argmax())
    if b > 0 and b - 1 != r:
        lab += "/" + PCS[b - 1]
    return lab


def main():
    import torch
    wav, meta = download()
    report["meta"] = meta
    stems = separate(wav)
    bt = beats(wav)
    report["rhythm"] = {k: v for k, v in bt.items() if k not in ("beats", "downbeats")}
    heads = lv_heads({"mix": wav, "harmonic": stems["harmonic"]})
    report["lv_heads_shapes"] = {k: [list(h.shape) for h in v] for k, v in heads.items()}
    d = wav.parent / "spike"
    d.mkdir(exist_ok=True)
    for k, v in heads.items():
        np.savez_compressed(d / f"lv_{k}.npz", *v)
    (d / "rhythm.json").write_text(json.dumps(bt), "utf-8")
    b1 = lv_baseline(wav)
    (d / "b1_lv_crf.json").write_text(json.dumps(b1, indent=1), "utf-8")
    report["b1_segments"] = len(b1)

    # Per-beat argmax label on the harmonic stem, printed bar by bar (eyeball check vs the chart)
    tri, bas, sev = heads["harmonic"][0], heads["harmonic"][1], heads["harmonic"][2]
    ft = (np.arange(tri.shape[0]) * 512 + 256) / 22050
    b = np.array(bt["beats"])
    db = set(np.round(bt["downbeats"], 3))
    lines, cur = [], []
    for i in range(len(b) - 1):
        m = (ft >= b[i]) & (ft < b[i + 1])
        lab = frame_label(tri[m].mean(0), bas[m].mean(0), sev[m].mean(0)) if m.any() else "?"
        if round(b[i], 3) in db and cur:
            lines.append(" ".join(cur))
            cur = []
        cur.append(lab)
    lines.append(" ".join(cur))
    (d / "beat_labels.txt").write_text("\n".join(f"{i:3d} | {l}" for i, l in enumerate(lines)), "utf-8")
    report["gpu"] = torch.cuda.get_device_name(0)
    (d / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nFirst 24 bars (per-beat argmax on the harmonic stem):")
    for i, l in enumerate(lines[:24]):
        print(f"{i:3d} | {l}")


if __name__ == "__main__":
    main()
