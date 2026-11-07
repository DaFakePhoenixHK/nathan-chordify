"""A small, fast SFZ sampler for the Salamander Grand Piano (no external synth needed).

The piano: Salamander Grand Piano V3 (Yamaha C5, 16 velocity layers, sampled every minor third,
48 kHz / 24-bit FLAC) by Alexander Holm, CC-BY 3.0 — see ``models/piano/salamander/ATTRIBUTION.txt``.
The default SFZ is the "Retuned" one (per-sample ``tune=`` corrections by Markus Fiedler, close to
equal temperament; the plain SFZ keeps the recorded stretch tuning, about +60 cents at C8).

How a note is played (``Piano.render``):

- **SFZ mapping**: ``parse_sfz`` reads ``<control>/<global>/<master>/<group>/<region>`` headers,
  ``#define``/``#include``, key names ("c4" = 60) and the opcodes the Salamander bank uses
  (lokey/hikey/key, lovel/hivel, pitch_keycenter, tune, transpose, volume, amp_veltrack,
  ampeg_release, trigger=release, rt_decay, pitch_keytrack, on_locc64/on_hicc64, lorand/hirand,
  group/off_by). Every other opcode is kept in ``Region.opcodes`` but ignored.
- **Pitch**: the region that covers the key (= the nearest sample) is resampled with soxr from
  ``sample_rate × 2^(semitones/12 + cents/1200)`` to 44.1 kHz, so the rate change and the pitch
  shift are one resampling step.
- **Velocity**: ``velocity_mode="pick"`` (default) plays the layer whose lovel..hivel holds the
  velocity; ``"xfade"`` blends the two nearest layers (equal power). The gain follows the SFZ
  velocity tracking (sfizz rule): ``g = (1 − t) + t·(v/127)²`` with ``t = amp_veltrack/100``.
- **Note-off / pedal**: on note-off the sound decays exponentially to −60 dB in ``ampeg_release``
  seconds (1 s for damped keys, 5 s for the undamped top keys 89–108). While the sustain pedal is
  down a released key keeps ringing until the pedal goes up. Re-striking a key that still sounds
  chokes the old voice in ~15 ms (one string set per key).
- **Release samples** (optional, on by default): at the effective release the bank's
  ``trigger=release`` regions (string resonance + hammer/damper noise) play, quieter the longer the
  key was held (``rt_decay`` dB per second). **Pedal noise** (optional) plays the pedal-down /
  pedal-up mechanics samples (pedal-down noise is cut by pedal-up: ``off_by``).
- **Effects** (optional): ``room()`` = a synthetic stereo room reverb (pre-delay, early
  reflections, exponential noise tail that darkens), ``master()`` = gain + a look-ahead peak limiter.

- **Detune** (``tune_cents``): the dry mix is rendered on a timeline stretched by the pitch ratio
  and resampled once, so note times stay exact and the per-key caches below stay valid.

Caches: samples are decoded lazily on first use and kept in RAM (float16, cut to ``max_seconds``,
default 12 s; the whole bank is ≈ 1 GB that way; ``preload()`` decodes it in parallel in ~2 s), and
the first ``pitch_cache_seconds`` (5 s) of every (sample, key) already resampled to its pitch are
kept in an LRU of ``pitch_cache_mb`` (1000 MB). After warm-up a 2 s four-note chord renders in
~15–20 ms (``tests/test_piano.py`` checks < 50 ms); each process holds its own caches (≈ 1.7 GB
once warm), so lower ``pitch_cache_mb`` / ``max_seconds`` for many parallel workers.
"""
from __future__ import annotations

import math
import re
import threading
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from scipy.ndimage import maximum_filter1d
from scipy.signal import fftconvolve, lfilter

SR = 44100
A0, C8 = 21, 108
ROOT = Path(__file__).resolve().parents[1]
PIANO_DIR = ROOT / "models" / "piano" / "salamander"
DEFAULT_SFZ = "SalamanderGrandPianoRetuned-V3+20200602.sfz"
#: Extra tuning (cents) by sample-name prefix for the Salamander bank. Measured 2026-11-01 (energy
#: centroid of the fundamental): after the Retuned SFZ's own corrections every sampled key from F#6
#: to A7 sits within ±5 cents, but the C8 samples (keys 107–108) are still +56 cents sharp.
SALAMANDER_TUNE_FIX = {"C8v": -56.0}


# ---------------------------------------------------------------------------------------- SFZ parser

_LETTER = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_KEY_RE = re.compile(r"^([a-gA-G])([#b]?)(-?\d+)$")
_TOKEN_RE = re.compile(r"<(\w+)>|([A-Za-z0-9_$]+)=")
_KEY_OPCODES = ("lokey", "hikey", "key", "pitch_keycenter")


def key_number(text: str) -> int:
    """SFZ key value → MIDI number: "60", "c4" (= 60), "c#4", "db4", "-1" (= no key)."""
    s = str(text).strip()
    try:
        return int(s)
    except ValueError:
        pass
    m = _KEY_RE.match(s)
    if not m:
        raise ValueError(f"bad SFZ key {text!r}")
    return 12 * (int(m[3]) + 1) + _LETTER[m[1].lower()] + {"#": 1, "b": -1, "": 0}[m[2]]


@dataclass(frozen=True)
class Region:
    """One ``<region>`` with its inherited opcodes (only the ones the sampler uses are typed)."""
    sample: Path
    lokey: int = 0
    hikey: int = 127
    lovel: int = 1
    hivel: int = 127
    keycenter: int = 60
    tune: float = 0.0              # cents
    transpose: int = 0             # semitones
    volume: float = 0.0            # dB
    amp_veltrack: float = 100.0    # %
    ampeg_release: float = 0.001   # s
    trigger: str = "attack"
    rt_decay: float = 0.0          # dB per second held (release triggers)
    pitch_keytrack: float = 100.0  # cents per key
    lorand: float = 0.0
    hirand: float = 1.0
    group: int = 0
    off_by: int | None = None
    on_locc64: int | None = None
    on_hicc64: int | None = None
    opcodes: tuple[tuple[str, str], ...] = ()

    def covers(self, key: int, vel: int) -> bool:
        return self.lokey <= key <= self.hikey and self.lovel <= vel <= self.hivel


def _read_sfz_text(path: Path, defines: dict[str, str], depth: int = 0) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#define"):
            parts = s.split(None, 2)
            if len(parts) == 3:
                defines[parts[1]] = parts[2].strip()
            continue
        if s.startswith("#include"):
            m = re.search(r'"([^"]+)"', s)
            if m and depth < 8:
                out.append(_read_sfz_text(path.parent / m[1].replace("\\", "/"), defines, depth + 1))
            continue
        for name in sorted(defines, key=len, reverse=True):
            line = line.replace(name, defines[name])
        out.append(line)
    return "\n".join(out)


def _opt_int(v: str | None) -> int | None:
    return None if v is None else int(float(v))


def _make_region(op: dict[str, str], base_dir: Path, octave_offset: int, note_offset: int) -> Region:
    def key(name: str, default: int) -> int:
        if name not in op:
            return default
        k = key_number(op[name])
        return k if k < 0 else k + 12 * octave_offset + note_offset

    lokey, hikey, center = key("lokey", 0), key("hikey", 127), key("pitch_keycenter", 60)
    if "key" in op:
        k = key("key", 60)
        lokey = k if "lokey" not in op else lokey
        hikey = k if "hikey" not in op else hikey
        center = k if "pitch_keycenter" not in op else center
    f = lambda name, d: float(op[name]) if name in op else d  # noqa: E731
    sample = (op.get("default_path", "") + op["sample"]).replace("\\", "/")
    return Region(
        sample=(base_dir / sample).resolve(), lokey=lokey, hikey=hikey,
        lovel=int(f("lovel", 1)), hivel=int(f("hivel", 127)), keycenter=center,
        tune=f("tune", 0.0), transpose=int(f("transpose", 0)), volume=f("volume", 0.0),
        amp_veltrack=f("amp_veltrack", 100.0), ampeg_release=f("ampeg_release", 0.001),
        trigger=op.get("trigger", "attack"), rt_decay=f("rt_decay", 0.0),
        pitch_keytrack=f("pitch_keytrack", 100.0), lorand=f("lorand", 0.0), hirand=f("hirand", 1.0),
        group=int(f("group", 0)), off_by=_opt_int(op.get("off_by")),
        on_locc64=_opt_int(op.get("on_locc64")), on_hicc64=_opt_int(op.get("on_hicc64")),
        opcodes=tuple(sorted(op.items())))


def parse_sfz(path) -> list[Region]:
    """Parse an SFZ file into regions (headers inherit: control → global → master → group → region)."""
    path = Path(path)
    text = _read_sfz_text(path, {})
    toks = list(_TOKEN_RE.finditer(text))
    control: dict[str, str] = {}
    scopes: dict[str, dict[str, str]] = {"global": {}, "master": {}, "group": {}}
    regions: list[Region] = []
    header, cur = None, {}

    def close():
        if header == "region" and "sample" in cur:
            op = {**scopes["global"], **scopes["master"], **scopes["group"], **cur}
            if "default_path" in control:
                op.setdefault("default_path", control["default_path"])
            regions.append(_make_region(op, path.parent, int(control.get("octave_offset", 0)),
                                        int(control.get("note_offset", 0))))

    for i, m in enumerate(toks):
        if m[1]:                                   # a header
            close()
            header = m[1].lower()
            if header == "control":
                control = {}
                cur = control
            elif header == "global":
                scopes.update({"global": {}, "master": {}, "group": {}})
                cur = scopes["global"]
            elif header == "master":
                scopes["master"], scopes["group"] = {}, {}
                cur = scopes["master"]
            elif header == "group":
                scopes["group"] = {}
                cur = scopes["group"]
            elif header == "region":
                cur = {}
            else:                                  # <curve>, <effect>, <midi>, ...: ignored
                cur = {}
            continue
        end = toks[i + 1].start() if i + 1 < len(toks) else len(text)
        value = text[m.end():end].strip()
        name = m[2]
        if name not in ("sample", "default_path"):
            value = value.split()[0] if value.split() else ""
        cur[name] = value
    close()
    return regions


def find_sfz(directory: Path | str | None = None) -> Path:
    """The Salamander SFZ under models/piano/salamander (the Retuned one when present)."""
    d = Path(directory) if directory else PIANO_DIR
    hits = sorted(d.rglob(DEFAULT_SFZ)) or sorted(d.rglob("*.sfz"))
    if not hits:
        raise FileNotFoundError(f"no .sfz found under {d} (download the Salamander Grand Piano first)")
    return hits[0]


def available(directory: Path | str | None = None) -> bool:
    try:
        find_sfz(directory)
        return True
    except FileNotFoundError:
        return False


# ------------------------------------------------------------------------------------------ notes

@dataclass(frozen=True)
class PianoNote:
    pitch: int        # MIDI 21..108
    velocity: int     # 1..127
    onset: float      # s
    duration: float   # s, key down → key up (the pedal may sustain it longer)


def as_note(n) -> PianoNote:
    if isinstance(n, PianoNote):
        return n
    if isinstance(n, dict):
        return PianoNote(int(n["pitch"]), int(n["velocity"]), float(n["onset"]), float(n["duration"]))
    p, v, t, d = n
    return PianoNote(int(p), int(v), float(t), float(d))


def pedal_intervals(pedal, notes: Sequence[PianoNote]) -> list[tuple[float, float]]:
    """None/False → no pedal; True → down just before the first note, up at the last key-up;
    otherwise a list of (down, up) times (merged when they overlap)."""
    if pedal is None or pedal is False:
        return []
    if pedal is True:
        if not notes:
            return []
        return [(max(0.0, min(n.onset for n in notes) - 0.05), max(n.onset + n.duration for n in notes))]
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in pedal):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _release_time(t_off: float, pedal: list[tuple[float, float]]) -> float:
    for a, b in pedal:
        if a <= t_off < b:
            return b
    return t_off


# ---------------------------------------------------------------------------------------- effects

@lru_cache(maxsize=32)
def room_ir(rt60: float, seed: int = 0, sr: int = SR) -> np.ndarray:
    """Stereo room impulse response (2, n): 10–25 ms pre-delay, a few early reflections and an
    exponential noise tail reaching −60 dB at ``rt60`` that darkens over time; unit energy."""
    rng = np.random.default_rng([int(seed), int(round(rt60 * 1000))])
    n = int(min(4.0, 1.1 * rt60) * sr)
    t = np.arange(n) / sr
    env = 10 ** (-3 * t / rt60)
    darken = np.exp(-t / (0.25 * rt60))
    pre = int(rng.uniform(0.010, 0.025) * sr)
    irs = []
    for _ in range(2):
        bright = rng.standard_normal(n)
        dark = lfilter([0.3], [1, -0.7], rng.standard_normal(n)) * 2.0
        ir = np.zeros(n + pre)
        ir[pre:] = env * (darken * bright + (1 - darken) * dark)
        for d, g in zip(rng.uniform(0.004, 0.04, 6), rng.uniform(0.3, 0.7, 6)):
            ir[int(d * sr)] += g * rng.choice((-1, 1)) * 4
        irs.append(ir / np.sqrt(np.sum(ir ** 2)))
    out = np.vstack(irs).astype(np.float32)
    out.setflags(write=False)
    return out


def room(x: np.ndarray, rt60: float = 1.2, wet: float = 0.25, seed: int = 0, sr: int = SR) -> np.ndarray:
    """Add a synthetic room reverb to stereo audio (n, 2); the length is unchanged."""
    if rt60 <= 0 or wet <= 0 or len(x) == 0:
        return x
    ir = room_ir(round(float(rt60), 3), int(seed), sr)
    y = x.astype(np.float32, copy=True)
    for ch in range(2):
        y[:, ch] += wet * fftconvolve(x[:, ch], ir[ch])[: len(x)].astype(np.float32)
    return y


def limiter(x: np.ndarray, ceiling_db: float = -1.0, hold_s: float = 0.005, release_s: float = 0.08,
            sr: int = SR) -> np.ndarray:
    """Look-ahead peak limiter: a gain envelope that dips ``hold_s`` before each over-ceiling
    peak and recovers with ``release_s``; a final clip guarantees the ceiling."""
    ceiling = 10 ** (ceiling_db / 20)
    peak = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    if len(peak) == 0 or peak.max() <= ceiling:
        return x
    red = 1.0 - np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))      # needed reduction 0..1
    h = max(1, int(hold_s * sr))
    red = maximum_filter1d(red, size=2 * h + 1)
    a = 1.0 - math.exp(-1.0 / max(1.0, release_s * sr))
    smooth = lfilter([a], [1, a - 1], red)
    red = np.maximum(red, smooth)                                        # instant attack, slow release
    g = (1.0 - red).astype(np.float32)
    y = x * (g[:, None] if x.ndim == 2 else g)
    return np.clip(y, -ceiling, ceiling)


def master(x: np.ndarray, gain_db: float = 0.0, limit: bool = True, ceiling_db: float = -1.0,
           sr: int = SR) -> np.ndarray:
    y = x * np.float32(10 ** (gain_db / 20))
    return limiter(y, ceiling_db, sr=sr) if limit else y


def normalize(x: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    p = float(np.max(np.abs(x))) if len(x) else 0.0
    return x if p == 0 else (x * np.float32(10 ** (peak_db / 20) / p)).astype(np.float32)


# ---------------------------------------------------------------------------------------- sampler

@dataclass
class _Voice:
    regions: list[tuple[Region, float]]   # (region, weight)
    note: PianoNote
    release: float                        # effective release time (s, absolute)
    choke: float | None                   # next strike of the same key (s, absolute)


def _vel_gain(r: Region, vel: float) -> float:
    t = abs(r.amp_veltrack) / 100.0
    c = (max(0.0, min(127.0, vel)) / 127.0) ** 2
    if r.amp_veltrack < 0:
        c = 1.0 - c
    return ((1.0 - t) + t * c) * 10 ** (r.volume / 20)


class Piano:
    """SFZ sampler; see the module docstring. ``render(notes, pedal=...)`` → (n, 2) float32 at 44.1 kHz."""

    def __init__(self, sfz: Path | str | None = None, sr: int = SR, max_seconds: float | None = 12.0,
                 quality: str = "HQ", release_samples: bool = True, pedal_noise: bool = True,
                 velocity_mode: str = "pick", cache_dtype=np.float16, tune_fix: dict[str, float] | None = None,
                 pitch_cache_seconds: float = 5.0, pitch_cache_mb: float = 1000.0):
        self.sfz = Path(sfz) if sfz else find_sfz()
        self.pitch_cache_seconds = pitch_cache_seconds
        self.pitch_cache_bytes = int(pitch_cache_mb * 1e6)
        self.sr = int(sr)
        self.max_seconds = max_seconds
        self.quality = quality
        self.release_samples = release_samples
        self.pedal_noise = pedal_noise
        self.velocity_mode = velocity_mode
        self.cache_dtype = cache_dtype
        fix = tune_fix if tune_fix is not None else (
            SALAMANDER_TUNE_FIX if "salamander" in self.sfz.name.lower() else {})
        self.regions = [next((replace(r, tune=r.tune + c) for pre, c in fix.items()
                              if r.sample.name.startswith(pre)), r) for r in parse_sfz(self.sfz)]
        self.note_regions = [r for r in self.regions if r.trigger in ("attack", "first", "legato")
                             and r.on_locc64 is None and r.lokey >= 0]
        self.release_regions = [r for r in self.regions if r.trigger == "release"
                                and r.on_locc64 is None and r.lokey >= 0]
        self.pedal_down_regions = [r for r in self.regions if r.on_locc64 is not None and r.on_locc64 >= 64]
        self.pedal_up_regions = [r for r in self.regions if r.on_hicc64 is not None and r.on_hicc64 < 64]
        if not self.note_regions:
            raise ValueError(f"{self.sfz} has no playable note regions")
        self._layers: dict[int, list[Region]] = {}
        for k in range(128):
            rs = [r for r in self.note_regions if r.lokey <= k <= r.hikey]
            if rs:
                self._layers[k] = sorted(rs, key=lambda r: (r.lovel, r.hivel))
        self.keys = sorted(self._layers)
        self._cache: dict[Path, tuple[np.ndarray, int, bool]] = {}
        # samples already resampled to their key's pitch at self.sr (LRU, <= pitch_cache_bytes):
        # (path, ratio, whole sample?) -> frames x 2 in cache_dtype
        self._pitched: OrderedDict[tuple[Path, float, bool], np.ndarray] = OrderedDict()
        self._pitched_bytes = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ samples
    @property
    def cache_bytes(self) -> int:
        return sum(a.nbytes for a, _, _ in self._cache.values()) + self._pitched_bytes

    def load(self, path: Path) -> tuple[np.ndarray, int, bool]:
        """(frames × 2 array in ``cache_dtype``, sample rate, cut-short flag), cached."""
        hit = self._cache.get(path)
        if hit is not None:
            return hit
        info = sf.info(str(path))
        n = info.frames
        if self.max_seconds is not None:
            n = min(n, int(self.max_seconds * info.samplerate))
        x, sr = sf.read(str(path), frames=n, dtype="float32", always_2d=True)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        elif x.shape[1] > 2:
            x = x[:, :2]
        entry = (np.ascontiguousarray(x.astype(self.cache_dtype)), int(sr), n < info.frames)
        with self._lock:
            self._cache.setdefault(path, entry)
        return self._cache[path]

    def preload(self, keys: Iterable[int] | None = None, workers: int = 8, releases: bool = True) -> int:
        """Decode the samples for these keys (default: all) in parallel; returns bytes cached."""
        ks = set(self.keys if keys is None else keys)
        paths = {r.sample for r in self.note_regions if any(r.lokey <= k <= r.hikey for k in ks)}
        if releases and self.release_samples:
            paths |= {r.sample for r in self.release_regions if any(r.lokey <= k <= r.hikey for k in ks)}
        if self.pedal_noise:
            paths |= {r.sample for r in self.pedal_down_regions + self.pedal_up_regions}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(self.load, sorted(paths)))
        return self.cache_bytes

    def clear_cache(self) -> None:
        self._cache.clear()
        self._pitched.clear()
        self._pitched_bytes = 0

    # ------------------------------------------------------------------ region choice
    def layers(self, key: int) -> list[Region]:
        if key in self._layers:
            return self._layers[key]
        nearest = min(self.keys, key=lambda k: (abs(k - key), k))
        return self._layers[nearest]

    def pick(self, key: int, vel: int, mode: str | None = None) -> list[tuple[Region, float]]:
        """Velocity layer(s) and their mix weights for a key."""
        layers = self.layers(key)
        mode = mode or self.velocity_mode
        if mode == "pick" or len(layers) == 1:
            for r in layers:
                if r.lovel <= vel <= r.hivel:
                    return [(r, 1.0)]
            return [(min(layers, key=lambda r: min(abs(vel - r.lovel), abs(vel - r.hivel))), 1.0)]
        if mode != "xfade":
            raise ValueError("velocity_mode must be 'pick' or 'xfade'")
        centres = [(r.lovel + r.hivel) / 2 for r in layers]
        if vel <= centres[0]:
            return [(layers[0], 1.0)]
        if vel >= centres[-1]:
            return [(layers[-1], 1.0)]
        i = max(j for j, c in enumerate(centres) if c <= vel)
        w = (vel - centres[i]) / (centres[i + 1] - centres[i])
        return [(layers[i], math.cos(w * math.pi / 2)), (layers[i + 1], math.sin(w * math.pi / 2))]

    @staticmethod
    def _ratio(r: Region, key: int) -> float:
        semis = (key - r.keycenter) * r.pitch_keytrack / 100.0 + r.transpose
        return 2.0 ** (semis / 12.0 + r.tune / 1200.0)

    # ------------------------------------------------------------------ voices
    def pitched(self, r: Region, key: int, whole: bool = False) -> np.ndarray:
        """Region ``r`` resampled to ``key``'s pitch at ``self.sr``: the first
        ``pitch_cache_seconds`` (or the whole cached sample), kept in an LRU cache."""
        ratio = self._ratio(r, key)
        ck = (r.sample, round(ratio, 10), whole)
        hit = self._pitched.get(ck)
        if hit is not None:
            self._pitched.move_to_end(ck)
            return hit
        data, sr, _ = self.load(r.sample)
        if whole:
            x, n_out = data, None
        else:
            n_out = int(self.pitch_cache_seconds * self.sr)
            x = data[: int(math.ceil(n_out * ratio * sr / self.sr)) + 4096]
        y = soxr.resample(x.astype(np.float32), sr * ratio, self.sr, quality=self.quality)
        if n_out is not None:
            y = np.ascontiguousarray(y[:n_out].astype(self.cache_dtype))
        else:                       # whole (short) samples: float32, inaudible tail (< -100 dBFS) cut
            loud = np.flatnonzero(np.abs(y).max(axis=1) > 1e-5)
            y = np.ascontiguousarray(y[: (loud[-1] + 1) if len(loud) else 1])
        with self._lock:
            if ck not in self._pitched:
                self._pitched[ck] = y
                self._pitched_bytes += y.nbytes
            while self._pitched_bytes > self.pitch_cache_bytes and len(self._pitched) > 1:
                _, old = self._pitched.popitem(last=False)
                self._pitched_bytes -= old.nbytes
        return y

    def _voice_audio(self, r: Region, key: int, n_out: int) -> np.ndarray:
        """n_out frames of region r at key's pitch (float32): from the pitched cache when it is
        long enough, else resampled directly (samples cut at max_seconds fade out at the end)."""
        pre = self.pitched(r, key)
        if n_out <= len(pre) or len(pre) < int(self.pitch_cache_seconds * self.sr):
            y = np.zeros((n_out, 2), np.float32)
            m = min(n_out, len(pre))
            y[:m] = pre[:m]
            return y
        data, sr, cut = self.load(r.sample)
        ratio = self._ratio(r, key)
        need = int(math.ceil(n_out * ratio * sr / self.sr)) + 64
        x = data[:need].astype(np.float32)
        if cut and need > len(data):
            f = min(len(x), int(0.25 * sr))
            x[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)[:, None]
        y = soxr.resample(x, sr * ratio, self.sr, quality=self.quality)[:n_out]
        if len(y) < n_out:
            y = np.concatenate([y, np.zeros((n_out - len(y), 2), np.float32)])
        return y

    def _play_voice(self, out: np.ndarray, v: _Voice) -> None:
        n = v.note
        start = int(round(n.onset * self.sr))
        r0 = v.regions[0][0]
        tail = max(0.01, r0.ampeg_release)
        rel = max(0.0, v.release - n.onset)
        end = rel + tail
        if v.choke is not None:
            end = min(end, max(0.0, v.choke - n.onset) + 0.08)
        n_out = min(int(math.ceil(end * self.sr)), len(out) - start)
        if n_out <= 0:
            return
        y = None
        for r, w in v.regions:
            z = self._voice_audio(r, n.pitch, n_out)
            z *= np.float32(w * _vel_gain(r, n.velocity))
            y = z if y is None else y + z
        k_rel = int(round(rel * self.sr))
        if k_rel < n_out:                              # damper: -60 dB after ``tail`` seconds
            k = np.arange(n_out - k_rel, dtype=np.float32)
            y[k_rel:] *= np.exp(k * np.float32(-6.9078 / (tail * self.sr)))[:, None]
        if v.choke is not None:
            k_c = int(round((v.choke - n.onset) * self.sr))
            if k_c < n_out:
                k = np.arange(n_out - k_c, dtype=np.float32)
                y[k_c:] *= np.exp(k * np.float32(-1.0 / (0.015 * self.sr)))[:, None]
        f = min(n_out, int(0.004 * self.sr))           # click-free end
        y[n_out - f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)[:, None]
        out[start:start + n_out] += y

    def _add_at(self, out: np.ndarray, x: np.ndarray, t: float, gain: float, cut_at: float | None = None) -> None:
        start = int(round(t * self.sr))
        if start >= len(out) or gain == 0.0:
            return
        x = x[: len(out) - start]
        if cut_at is not None:                         # off_by: fade the rest away in 20 ms
            k = int(round((cut_at - t) * self.sr))
            if k < len(x):
                f = min(len(x) - max(k, 0), int(0.02 * self.sr))
                x = x[: max(k, 0) + f].copy()
                x[max(k, 0):] *= np.linspace(1.0, 0.0, f, dtype=np.float32)[:, None]
        out[start:start + len(x)] += x * np.float32(gain)

    def voices(self, notes: Sequence[PianoNote], pedal: list[tuple[float, float]],
               velocity_mode: str | None = None) -> list[_Voice]:
        by_key: dict[int, list[PianoNote]] = {}
        for nt in notes:
            by_key.setdefault(nt.pitch, []).append(nt)
        out = []
        for key, ns in by_key.items():
            ns.sort(key=lambda x: x.onset)
            for i, nt in enumerate(ns):
                choke = ns[i + 1].onset if i + 1 < len(ns) else None
                rel = _release_time(nt.onset + max(0.0, nt.duration), pedal)
                if choke is not None and choke >= rel + max(0.01, self.layers(key)[0].ampeg_release):
                    choke = None
                out.append(_Voice(self.pick(key, nt.velocity, velocity_mode), nt, rel, choke))
        return out

    # ------------------------------------------------------------------ render
    def render(self, notes: Iterable, pedal=None, duration: float | None = None, *,
               tune_cents: float = 0.0, seed: int = 0, velocity_mode: str | None = None,
               release_samples: bool | None = None, pedal_noise: bool | None = None,
               reverb: tuple[float, float] | None = None, gain_db: float = 0.0,
               limit: bool = False) -> np.ndarray:
        """Render notes ((pitch, velocity, onset, duration) tuples, dicts or ``PianoNote``) to
        stereo float32 (n, 2) at ``self.sr``.

        ``pedal``: None/False, True (held over all the notes) or a list of (down, up) times.
        ``duration``: output length in s (default: until the last sound dies, + reverb tail).
        ``tune_cents``: global detune (the dry mix is rendered on a stretched timeline and
        resampled once, so the note times stay exact). ``reverb``: (rt60 s, wet 0..1) or None.
        ``gain_db`` / ``limit``: master gain and the peak limiter (ceiling -1 dBFS).
        """
        ns = [as_note(n) for n in notes]
        for nt in ns:
            if not (0 <= nt.pitch <= 127 and 1 <= nt.velocity <= 127):
                raise ValueError(f"bad note {nt}")
        ped = pedal_intervals(pedal, ns)
        use_rel = self.release_samples if release_samples is None else release_samples
        use_ped = self.pedal_noise if pedal_noise is None else pedal_noise
        if tune_cents:
            s = 2.0 ** (float(tune_cents) / 1200.0)
            ns2 = [PianoNote(n.pitch, n.velocity, n.onset * s, n.duration * s) for n in ns]
            ped2 = [(a * s, b * s) for a, b in ped]
            dry = self._render_dry(ns2, ped2, None if duration is None else duration * s,
                                   seed, velocity_mode, use_rel, use_ped)
            out = soxr.resample(dry, self.sr * s, self.sr, quality=self.quality)
            n = int(math.ceil(duration * self.sr)) if duration is not None else len(out)
            out = out[:n] if len(out) >= n else np.concatenate([out, np.zeros((n - len(out), 2), np.float32)])
        else:
            out = self._render_dry(ns, ped, duration, seed, velocity_mode, use_rel, use_ped)
        if duration is None and reverb:
            out = np.concatenate([out, np.zeros((int(0.5 * float(reverb[0]) * self.sr), 2), np.float32)])
        if reverb:
            out = room(out, float(reverb[0]), float(reverb[1]), seed, self.sr)
        if gain_db or limit:
            out = master(out, gain_db, limit, sr=self.sr)
        return out

    def _render_dry(self, ns: list[PianoNote], ped: list[tuple[float, float]], duration: float | None,
                    seed: int, velocity_mode: str | None, use_rel: bool, use_ped: bool) -> np.ndarray:
        vs = self.voices(ns, ped, velocity_mode)
        rng = np.random.default_rng([int(seed), 7919])
        # release-trigger samples and pedal noises: (region, key, time, gain, cut_at)
        extras: list[tuple[Region, int, float, float, float | None]] = []
        if use_rel:
            for v in vs:
                if v.choke is not None:
                    continue
                held = max(0.0, v.release - v.note.onset)
                u = float(rng.random())
                for r in self.release_regions:
                    if r.covers(v.note.pitch, v.note.velocity) and r.lorand <= u < r.hirand + 1e-9:
                        g = _vel_gain(r, v.note.velocity) * 10 ** (-r.rt_decay * held / 20)
                        if g > 1e-4:
                            extras.append((r, v.note.pitch, v.release, g, None))
        if use_ped:
            for a, b in ped:
                for regs, t, cut in ((self.pedal_down_regions, a, b), (self.pedal_up_regions, b, None)):
                    u = float(rng.random())
                    for r in regs:
                        if r.lorand <= u < r.hirand + 1e-9:
                            extras.append((r, r.keycenter, t, _vel_gain(r, 127), cut if r.off_by else None))
        if duration is None:
            end = 0.0
            for v in vs:
                e = v.release + max(0.01, v.regions[0][0].ampeg_release)
                if v.choke is not None:
                    e = min(e, v.choke + 0.08)
                end = max(end, e)
            for r, key, t, g, cut in extras:
                d = len(self.pitched(r, key, whole=True)) / self.sr
                end = max(end, t + (d if cut is None else min(d, cut - t + 0.02)))
            duration = end + 0.01
        out = np.zeros((int(math.ceil(duration * self.sr)), 2), np.float32)
        for v in vs:
            self._play_voice(out, v)
        for r, key, t, g, cut in extras:
            self._add_at(out, self.pitched(r, key, whole=True), t, g, cut)
        return out


@lru_cache(maxsize=4)
def get_piano(sfz: str | None = None, **kw) -> Piano:
    """A shared Piano (one sample cache per process)."""
    return Piano(sfz, **kw)
