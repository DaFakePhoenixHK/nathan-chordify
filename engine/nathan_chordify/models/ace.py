"""consonance-ACE adapter (WP4b, DESIGN §5.5): root / bass softmax and note sigmoids, no decoding.

The network is ``ConformerDecomposedModel`` from ``third_party/consonance-ACE`` (MIT, pinned in
``third_party/VERSIONS.txt``) with the repo's own checkpoint
``ACE/checkpoints/conformer_decomposed_smooth.ckpt``. The repo is **never pip-installed** (it pins
torch 2.6 / lightning 2.5.1 / beat-this 0.1); its folder is appended to ``sys.path`` and
``ACE.mir_evaluation`` is imported first (review-feasibility issue 2: it puts ``ACE/preprocess`` on
``sys.path``, which ``ACE.preprocess.audio_processor``'s bare ``from transforms import`` needs).
ACE's metrics code imports a top-level ``metrics`` package through a ``sys.path`` hack; ours is
``eval/chord_metrics.py``, so nothing shadows it (issue 3).

Outputs (``ChordEvidence``, the convention of evidence.py's ``ace`` source, §5.9):

- ``root`` [T,13]: softmax of the root head, pcs C..B then N (ACE's own class order:
  ``NoteEncoder(i + 1)`` for i < 12, 12 = N);
- ``bass`` [T,13]: softmax of the bass head, same order;
- ``notes`` [T,12]: sigmoid of the chord-tone ("onehot") head, absolute pcs from C.

Features, exactly as ACE's ``inference.py`` / ``AudioChunkProcessor``: the file is read at its own
rate, resampled to 22.05 kHz with torchaudio's default sinc resampler, averaged to mono; each
20 s window (441,000 samples, right-padded with zeros) is peak-normalised and turned into
|CQT| with ACE's ``CQTransform`` (librosa.cqt, hop 512, 24 bins/octave, 6 octaves from C1 →
[144, 862]); the model sees [B, 1, 144, 862].

Time: librosa's CQT frames are centred, so frame k is centred at k·512/22050 s (23.2 ms hop), and
``frame_t = k·512/22050 − offset_s``. ``offset_s`` is the network latency measured on mini-D2
(``eval/ace_eval.py offset``; DESIGN §5.5 acceptance |residual| < 20 ms).

Chunking (§5.5): ACE was trained on 20 s segments and its own inference runs 20 s chunks without
overlap (with a sinusoidal positional encoding, so a window's length is part of the input
distribution). We run 20 s windows with 5 s overlap and keep each window's frames up to the
middle of its overlaps (2.5 s cropped at every interior edge); the last window is aligned to the
end of the file instead of being zero-padded. Windows go to the GPU in fixed-size batches, so
VRAM is flat for any song length. The weights stay resident between runs (``release()`` frees
them; ``models.release_gpu()`` calls it).

Fine-tuned weights (FT1, ``train/finetune_ace.py``): ``Ace(weights=path)``, else the environment
variable ``ACE_WEIGHTS`` (a path, absolute or relative to the project root, or a run id such as
``ft1-b`` = ``models/ace_ft/ft1-b/best.pt``; empty / "original" = the original checkpoint), else
``DEFAULT_WEIGHTS`` (None = the original checkpoint; see eval/reports/FT1.md for why it is still
None after ft1-b's narrow gate pass). A fine-tuned file has the original checkpoint's
layout (``{"state_dict", "hyper_parameters", "ft"}``, loaded with ``weights_only=True``). Its
evidence is cached under its own name, ``evidence/ace@<input>.v1-ft<sha256[:8]>.npz``, so it never
mixes with the original's ``ace@<input>.v1.npz``. Decode with the temperature fitted for those
weights (``DecoderParams.temp["ace"]``; the fit is in ``models/ace_ft/<run>/calibration.json``).
"""
from __future__ import annotations

import hashlib
import inspect
import logging
import os
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config
from ..types import ChordEvidence
from .base import ChordModel

log = logging.getLogger(__name__)

SR = 22050
HOP = 512
BINS_PER_OCTAVE = 24
N_OCTAVES = 6
START_NOTE = "C1"
N_BINS = BINS_PER_OCTAVE * N_OCTAVES          # 144 = the checkpoint's input_dim

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
HEAD_LABELS: dict[str, list[str]] = {"root": PC_NAMES + ["N"], "bass": PC_NAMES + ["N"], "notes": list(PC_NAMES)}

CKPT_REL = Path("ACE") / "checkpoints" / "conformer_decomposed_smooth.ckpt"
CKPT_BYTES = 57_624_994                        # ordinary git blob at the pinned commit (VERSIONS.txt)
WEIGHTS_ENV = "ACE_WEIGHTS"
DEFAULT_WEIGHTS: str | None = None             # None = the original checkpoint (FT1 gate: eval/reports/FT1.md)
FT1_RUN, FT1_TEMP = "ft1-b", 0.6267            # opt-in: ACE_WEIGHTS=ft1-b + DecoderParams.temp["ace"] = 0.6267

_models: dict[object, object] = {}             # "cuda" | "cpu" (original) or (device, weights path) → the network
_lock = threading.Lock()
_ace: dict[str, object] = {}                   # the imported ACE classes


def repo_dir() -> Path:
    return config.THIRD_PARTY_DIR / "consonance-ACE"


def checkpoint_path() -> Path:
    return repo_dir() / CKPT_REL


def check_checkpoint() -> Path:
    """The checkpoint exists and has the pinned size (catches a missing clone or an LFS pointer)."""
    p = checkpoint_path()
    if not p.is_file():
        raise FileNotFoundError(f"consonance-ACE checkpoint missing: {p}. Clone the pinned commit "
                                "(third_party/VERSIONS.txt) into third_party/consonance-ACE.")
    if p.stat().st_size != CKPT_BYTES:
        raise RuntimeError(f"{p} is {p.stat().st_size} bytes, expected {CKPT_BYTES} (wrong commit?)")
    return p


def import_ace(audio_processor: bool = False) -> dict[str, object]:
    """Import the ACE classes from ``third_party/consonance-ACE`` in the safe order.

    ``audio_processor=True`` also imports ``ACE.preprocess.audio_processor`` (only the tests need
    it, as the reference pipeline); its module-level ``logging.basicConfig`` is undone."""
    with _lock:
        if "model" not in _ace or (audio_processor and "chunker" not in _ace):
            root = str(repo_dir())
            if root not in sys.path:
                sys.path.append(root)          # appended: it can never shadow our own packages
            import ACE                          # noqa: F401
            got = Path(ACE.__file__).resolve()
            if repo_dir().resolve() not in got.parents:
                raise ImportError(f"'ACE' resolves to {got}, not to {repo_dir()} (a pip-installed copy?)")
            import ACE.mir_evaluation           # noqa: F401  (FIRST: puts ACE/preprocess on sys.path)
            from ACE.models.conformer_decomposed import ConformerDecomposedModel
            from ACE.preprocess.transforms import CQTransform
            _ace.update(model=ConformerDecomposedModel, cqt=CQTransform)
            if audio_processor:
                rl = logging.getLogger()
                handlers, level = list(rl.handlers), rl.level
                from ACE.preprocess.audio_processor import AudioChunkProcessor
                for h in rl.handlers[:]:
                    if h not in handlers:
                        rl.removeHandler(h)
                rl.setLevel(level)
                _ace["chunker"] = AudioChunkProcessor
        return _ace


def resolve_weights(weights: str | Path | None = None) -> Path | None:
    """The fine-tuned weights file to use (None = the original checkpoint): ``weights``, else
    ``$ACE_WEIGHTS``, else ``DEFAULT_WEIGHTS``; "", "original" and "none" mean the original."""
    w = weights if weights is not None else (os.environ.get(WEIGHTS_ENV) or DEFAULT_WEIGHTS)
    if w is None or str(w).strip().lower() in ("", "original", "none"):
        return None
    p = Path(str(w).strip())
    if not p.is_absolute():
        run = config.MODELS_DIR / "ace_ft" / str(w).strip() / "best.pt"     # a run id: "ft1-b"
        p = run if run.is_file() else config.ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"ACE weights not found: {p} (set {WEIGHTS_ENV} to a fine-tuned .pt or 'original')")
    return p.resolve()


@lru_cache(maxsize=16)
def _sha8(path: str, size: int, mtime: float) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def weights_tag(weights: str | Path | None) -> str:
    """"original", or "ft" + the first 8 hex digits of the weights file's sha256."""
    p = resolve_weights(weights) if weights is not None else None
    if p is None:
        return "original"
    st = p.stat()
    return "ft" + _sha8(str(p), st.st_size, st.st_mtime)


def _norm_device(device: str) -> str:
    import torch
    return "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"


def load_checkpoint() -> dict:
    """The checkpoint dict, loaded with ``weights_only=True`` (DESIGN §5.5) on the CPU."""
    import torch
    return torch.load(check_checkpoint(), map_location="cpu", weights_only=True)


def load(device: str = "cuda", weights: str | Path | None = None):
    """The ACE network on ``device`` in eval mode (lazy module-level singleton, kept between runs);
    ``weights``: a fine-tuned file (see the module doc; default: $ACE_WEIGHTS, else the original)."""
    dev = _norm_device(device)
    w = resolve_weights(weights)
    key = dev if w is None else (dev, str(w))
    if key in _models:
        return _models[key]
    cls = import_ace()["model"]
    with _lock:
        if key not in _models:
            if w is None:
                ck = load_checkpoint()
            else:
                import torch
                ck = torch.load(w, map_location="cpu", weights_only=True)
            hp = dict(ck["hyper_parameters"])
            sig = inspect.signature(cls.__init__).parameters
            kwargs = {k: v for k, v in hp.items() if k in sig}
            model = cls(loss="consonance_decomposed", **kwargs)
            model.load_state_dict(ck["state_dict"], strict=True)
            model.eval().requires_grad_(False)
            _models[key] = model.to(dev)
        return _models[key]


def release() -> None:
    """Drop the loaded networks (frees VRAM, e.g. before retrying separation after an OOM)."""
    with _lock:
        _models.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_audio(wav: Path) -> np.ndarray:
    """Mono float32 at 22.05 kHz, as ACE's AudioChunkProcessor: native-rate read, torchaudio
    sinc resampling (its defaults), then the channel mean."""
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    x = torch.from_numpy(np.ascontiguousarray(y.T))
    if int(sr) != SR:
        x = AF.resample(x, int(sr), SR)
    return x.mean(dim=0).numpy()


def n_frames(n_samples: int) -> int:
    """Frames of a centred CQT (librosa): 1 + n // hop."""
    return 1 + int(n_samples) // HOP


def windows(n: int, win: int, crop: int) -> list[tuple[int, int, int, int]]:
    """(start, end, keep_start, keep_end) frame windows of exactly ``win`` frames.

    Starts step by ``win − 2·crop``; the last window is aligned to the end (start = n − win), so
    no window is shorter than ``win`` unless the whole file is (then: one window, padded). Each
    window keeps its frames up to the middle of the overlap with its neighbours, so interior
    edges lose ≥ ``crop`` frames and the kept ranges tile [0, n) exactly."""
    if n <= win:
        return [(0, win, 0, n)]
    step = win - 2 * crop
    if step <= 0:
        raise ValueError("crop must be < win / 2")
    starts, s = [], 0
    while s + win < n:
        starts.append(s)
        s += step
    starts.append(n - win)
    bounds = [0] + [(a + b + win) // 2 for a, b in zip(starts, starts[1:])] + [n]
    return [(s, s + win, bounds[i], bounds[i + 1]) for i, s in enumerate(starts)]


class Ace(ChordModel):
    """consonance-ACE → ``ChordEvidence`` ("ace@mix", "ace@harmonic")."""

    name = "ace"
    adapter_version = 1
    offset_s = 0.0          # measured on mini-D2 (eval/ace_eval.py offset): within ±20 ms, not applied
    chunk_s = 20.0          # ACE's training segment length; do not change
    overlap_s = 5.0         # crop = overlap / 2 at each interior edge
    batch = 4               # windows per forward pass (VRAM is flat in the song length)

    def __init__(self, weights: str | Path | None = None) -> None:
        self.stats: dict[str, float] = {}
        self.weights = resolve_weights(weights)            # None = the original checkpoint
        self.weights_tag = weights_tag(self.weights) if self.weights else "original"
        if self.weights is not None:                      # its own evidence cache files (".v1-ft1a2b3c4d.npz")
            self.adapter_version = f"{type(self).adapter_version}-{self.weights_tag}"

    def run(self, wav: Path, device: str = "cuda") -> ChordEvidence:
        t0 = time.perf_counter()
        y = load_audio(wav)
        if y.size < HOP:
            raise ValueError(f"{wav} is too short for chord recognition")
        t1 = time.perf_counter()
        root, bass, notes = self.infer(y, device)
        t2 = time.perf_counter()
        n = root.shape[0]
        self.stats.update(audio_s=y.size / SR, load_s=t1 - t0, total_s=t2 - t0, frames=n)
        log.info("consonance-ACE %s: %.0f s audio, %d windows, CQT %.1f s, inference %.1f s",
                 Path(wav).name, y.size / SR, self.stats.get("windows", 0), self.stats.get("cqt_s", 0.0),
                 self.stats.get("infer_s", 0.0))
        return ChordEvidence(
            name=self.evidence_name(wav),
            frame_t=np.arange(n) * HOP / SR - self.offset_s,
            offset_s=self.offset_s,
            root=root, bass=bass, notes=notes,
            heads={},
            head_labels={k: list(v) for k, v in HEAD_LABELS.items()},
            label_probs=None, labels=None,
        )

    def infer(self, y: np.ndarray, device: str = "cuda") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(root [T,13], bass [T,13], notes [T,12]) for mono 22.05 kHz audio; CPU fallback on OOM."""
        import torch
        dev = _norm_device(device)
        try:
            return self._infer(y, dev)
        except torch.cuda.OutOfMemoryError:
            if dev != "cuda":
                raise
            log.warning("consonance-ACE: CUDA out of memory; retrying on the CPU")
            torch.cuda.empty_cache()
            return self._infer(y, "cpu")

    def window_frames(self) -> tuple[int, int, int]:
        """(window samples, window frames, crop frames)."""
        L = int(round(self.chunk_s * SR))
        crop = int(round(self.overlap_s / 2 * SR / HOP))
        return L, n_frames(L), crop

    def features(self, y: np.ndarray, start_frame: int, L: int) -> np.ndarray:
        """[144, F] |CQT| of the window starting at frame ``start_frame`` (zero-padded to L samples,
        peak-normalised), exactly ACE's chunk pipeline."""
        import torch
        cqt = import_ace()["cqt"](SR, HOP)       # stateless; librosa caches the filters
        s = start_frame * HOP
        seg = np.zeros(L, np.float32)
        part = y[s:s + L]
        seg[:part.size] = part
        peak = float(np.abs(seg).max())
        if peak > 0:
            seg /= peak
        return cqt(torch.from_numpy(seg)[None]).numpy()[0].astype(np.float32, copy=False)

    def _infer(self, y: np.ndarray, dev: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch
        model = load(dev, self.weights if self.weights is not None else "original")
        L, win, crop = self.window_frames()
        n = n_frames(y.size)
        ws = windows(n, win, crop)
        root = np.zeros((n, 13), np.float32)
        bass = np.zeros((n, 13), np.float32)
        notes = np.zeros((n, 12), np.float32)
        t_cqt = t_inf = 0.0
        for b0 in range(0, len(ws), max(1, int(self.batch))):
            group = ws[b0:b0 + max(1, int(self.batch))]
            t = time.perf_counter()
            feats = np.stack([self.features(y, s, L) for s, _, _, _ in group])     # [B, 144, win]
            t_cqt += time.perf_counter() - t
            t = time.perf_counter()
            with torch.inference_mode():
                out = model(torch.from_numpy(feats)[:, None].to(dev))              # [B, 1, 144, win]
                r = torch.softmax(out["root"].float(), dim=-1).cpu().numpy()
                bs = torch.softmax(out["bass"].float(), dim=-1).cpu().numpy()
                nt = torch.sigmoid(out["onehot"].float()).cpu().numpy()
            t_inf += time.perf_counter() - t
            for k, (s, _, k0, k1) in enumerate(group):
                root[k0:k1] = r[k, k0 - s:k1 - s]
                bass[k0:k1] = bs[k, k0 - s:k1 - s]
                notes[k0:k1] = nt[k, k0 - s:k1 - s]
        self.stats = {"cqt_s": t_cqt, "infer_s": t_inf, "windows": len(ws), "device": dev}
        return root, bass, notes
