"""lv-chordia adapter (WP4a, DESIGN §5.5): per-head softmax evidence, no CRF/HMM decoding.

We call the package's internals directly (no fork): the 5-network ensemble
(``ChordNet`` + ``NetworkInterface``) on the same hybrid CQT the package computes, averaged
per head. Output heads and their classes (``HEAD_LABELS``):

- ``triad`` [T,73]: 0 = N; ``1 + (type−1)·12 + root`` for types maj, min, sus4, sus2, dim, aug
  and roots 0..11 from C;
- ``bass`` [T,13]: 0 = no bass; ``1 + pc`` (absolute pitch class);
- ``seventh`` [T,4]: none, 7, b7, bb7;  ``ninth`` [T,4]: none, 9, #9, b9;
  ``eleventh`` [T,3]: none, 11, #11;  ``thirteenth`` [T,3]: none, 13, b13.

Convenience arrays use the ``ChordEvidence`` convention **12 pcs first, N last**:
``root[t, pc]`` = Σ over triad types of the triad head, ``root[t, 12]`` = P(N);
``bass[t, pc]`` = bass head ``1 + pc``, ``bass[t, 12]`` = P(no bass).

Time: librosa's CQT frames are centred (verified with a click: the peak is symmetric about
frame k = sample / 512), so frame k is centred at ``k·512/22050`` s, and
``frame_t = k·512/22050 − offset_s``. ``offset_s`` is the network latency; it is 0 until it is
measured on D2 in M2 (TODO(M2): cross-correlate label changes with the D2 boundaries).

Chunking: the CQT is computed once for the whole song on the CPU, then the ensemble runs on
120 s frame windows with 10 s overlap, keeping all but 5 s at each interior edge, so VRAM stays
flat for any song length. (The design text writes each window to a temp WAV; in-memory CQT
windows are equivalent and simpler.)

Device: lv-chordia moves itself to CUDA whenever CUDA is visible (``use_gpu`` is set in
``NetworkBehavior.__init__``). Setting ``use_gpu`` on each net before it is wrapped gives a
CPU ensemble in-process, which is the CPU/OOM fallback (instead of a subprocess with
``CUDA_VISIBLE_DEVICES=""``).
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ..types import ChordEvidence
from .base import ChordModel

log = logging.getLogger(__name__)

SR = 22050
HOP = 512
N_BINS = 288
BINS_PER_OCTAVE = 36
FMIN_NOTE = "F#0"

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TRIAD_TYPES = ("maj", "min", "sus4", "sus2", "dim", "aug")
HEADS = ("triad", "bass", "seventh", "ninth", "eleventh", "thirteenth")
HEAD_LABELS: dict[str, list[str]] = {
    "triad": ["N"] + [f"{pc}:{t}" for t in TRIAD_TYPES for pc in PC_NAMES],
    "bass": ["N"] + PC_NAMES,
    "seventh": ["none", "7", "b7", "bb7"],
    "ninth": ["none", "9", "#9", "b9"],
    "eleventh": ["none", "11", "#11"],
    "thirteenth": ["none", "13", "b13"],
}

_nets: dict[str, list] = {}          # "cuda" | "cpu" → the 5 loaded NetworkInterfaces
_lock = threading.Lock()


def weights_dir() -> Path:
    return Path(sys.prefix) / "share" / "lv-chordia" / "cache_data"


def check_weights() -> list[Path]:
    """Assert the 5 ``.sdict`` files exist (lv-chordia silently uses random weights otherwise)."""
    from lv_chordia.chord_recognition import MODEL_NAMES
    paths = [weights_dir() / f"{n}.sdict" for n in MODEL_NAMES]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"lv-chordia weights missing in {weights_dir()}: {missing}. "
                                "Reinstall lv-chordia==1.1.0 into the venv.")
    from lv_chordia.mir import common
    if Path(common.CACHE_DATA_PATH).resolve() != weights_dir().resolve():
        raise RuntimeError(f"lv-chordia would load weights from {common.CACHE_DATA_PATH}, "
                           f"not {weights_dir()}")
    return paths


def _norm_device(device: str) -> str:
    import torch
    return "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"


def load(device: str = "cuda") -> list:
    """The 5 ensemble nets on ``device`` (lazy module-level singleton, kept between runs)."""
    dev = _norm_device(device)
    with _lock:
        if dev not in _nets:
            check_weights()
            from lv_chordia.chord_recognition import MODEL_NAMES
            from lv_chordia.chordnet_ismir_naive import ChordNet
            from lv_chordia.mir.nn.train import NetworkInterface
            nets = []
            for name in MODEL_NAMES:
                net = ChordNet(None)
                net.use_gpu = dev == "cuda"
                iface = NetworkInterface(net, name, load_checkpoint=False)
                if not iface.finalized:
                    raise RuntimeError(f"lv-chordia did not load {name}.sdict (random weights)")
                nets.append(iface)
            _nets[dev] = nets
        return _nets[dev]


def release() -> None:
    """Drop the loaded nets (frees VRAM, e.g. before retrying separation after an OOM)."""
    with _lock:
        _nets.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_audio(wav: Path) -> np.ndarray:
    """Mono float32 at 22.05 kHz, exactly as lv-chordia's own loader (librosa.load, soxr_hq)."""
    import librosa
    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    return y


def cqt_features(y: np.ndarray) -> np.ndarray:
    """|hybrid CQT| [T, 288] float32, the package's CQTV2 features (frames centred at k·HOP)."""
    import librosa
    c = librosa.hybrid_cqt(y, sr=SR, bins_per_octave=BINS_PER_OCTAVE, fmin=librosa.note_to_hz(FMIN_NOTE),
                           n_bins=N_BINS, tuning=None, hop_length=HOP)
    return np.abs(c).T.astype(np.float32)


def windows(n_frames: int, win: int, crop: int) -> list[tuple[int, int, int, int]]:
    """(start, end, keep_start, keep_end) frame windows: ``win`` frames, overlap ``2·crop``.

    Interior edges drop ``crop`` frames; the kept ranges tile [0, n_frames) exactly.
    """
    if n_frames <= win:
        return [(0, n_frames, 0, n_frames)]
    step = win - 2 * crop
    out, s = [], 0
    while True:
        e = min(s + win, n_frames)
        last = e == n_frames
        out.append((s, e, s + crop if s > 0 else 0, e if last else e - crop))
        if last:
            return out
        s += step


def triad_to_root(triad: np.ndarray) -> np.ndarray:
    """[T,73] triad head → [T,13] root marginal (12 pcs, then N)."""
    t = triad.shape[0]
    root = triad[:, 1:].reshape(t, len(TRIAD_TYPES), 12).sum(axis=1)
    return np.concatenate([root, triad[:, :1]], axis=1)


def bass_to_pc(bass: np.ndarray) -> np.ndarray:
    """[T,13] bass head (0 = none, 1 + pc) → [T,13] (12 pcs, then N)."""
    return np.concatenate([bass[:, 1:], bass[:, :1]], axis=1)


class LvChordia(ChordModel):
    """lv-chordia ensemble → ``ChordEvidence`` ("lv@mix", "lv@harmonic")."""

    name = "lv"
    adapter_version = 1
    offset_s = 0.0          # TODO(M2): measured latency on D2; acceptance |residual| < 20 ms
    chunk_s = 120.0
    overlap_s = 10.0        # crop = overlap / 2 at each interior edge

    def __init__(self) -> None:
        self.stats: dict[str, float] = {}

    def run(self, wav: Path, device: str = "cuda") -> ChordEvidence:
        t0 = time.perf_counter()
        y = load_audio(wav)
        if y.size < HOP:
            raise ValueError(f"{wav} is too short for chord recognition")
        cqt = cqt_features(y)
        t1 = time.perf_counter()
        heads = self.infer(cqt, device)
        t2 = time.perf_counter()
        n = cqt.shape[0]
        self.stats = {"audio_s": y.size / SR, "cqt_s": t1 - t0, "infer_s": t2 - t1, "frames": n}
        log.info("lv-chordia %s: %.0f s audio, CQT %.1f s, inference %.1f s",
                 Path(wav).name, y.size / SR, t1 - t0, t2 - t1)
        return ChordEvidence(
            name=self.evidence_name(wav),
            frame_t=np.arange(n) * HOP / SR - self.offset_s,
            offset_s=self.offset_s,
            root=triad_to_root(heads["triad"]),
            bass=bass_to_pc(heads["bass"]),
            notes=None,
            heads=heads,
            head_labels={k: list(v) for k, v in HEAD_LABELS.items()},
            label_probs=None,
            labels=None,
        )

    def infer(self, cqt: np.ndarray, device: str = "cuda") -> dict[str, np.ndarray]:
        """Ensemble-mean softmax per head over the whole CQT, in chunks; CPU fallback on OOM."""
        import torch
        dev = _norm_device(device)
        try:
            return self._infer(cqt, load(dev))
        except torch.cuda.OutOfMemoryError:
            if dev != "cuda":
                raise
            log.warning("lv-chordia: CUDA out of memory; retrying on the CPU")
            torch.cuda.empty_cache()
            return self._infer(cqt, load("cpu"))

    def _infer(self, cqt: np.ndarray, nets: list) -> dict[str, np.ndarray]:
        fps = SR / HOP
        crop = int(round(self.overlap_s / 2 * fps))
        win = max(int(round(self.chunk_s * fps)), 4 * crop + 1)
        n = cqt.shape[0]
        out = {h: np.zeros((n, len(HEAD_LABELS[h])), np.float32) for h in HEADS}
        for s, e, k0, k1 in windows(n, win, crop):
            chunk = np.ascontiguousarray(cqt[s:e])
            acc = None
            for net in nets:
                res = net.inference(chunk)
                acc = list(res) if acc is None else [a + r for a, r in zip(acc, res)]
            for h, a in zip(HEADS, acc):
                out[h][k0:k1] = a[k0 - s:k1 - s] / len(nets)
        return out
