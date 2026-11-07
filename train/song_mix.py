"""Training "songs" for fine-tuning consonance-ACE (FT1): rendered chord-progression excerpts with
exact frame labels for ACE's three heads.

One item = a 15–30 s excerpt of a made-up worship song:

- **chords** from four sources: real progressions from free charts (``data/charts/*.chart.txt`` and
  ``data/charts/alt/``, 123 worship songs + 41 alternative versions; the 5 D1 test songs are
  excluded there) transposed to a random key, with NEW chord lengths 70 % of the time when the
  chart's bar lengths were inferred from the lyrics (the chord order is exact, the lengths are not);
  the 32 idiom progressions of ``train.chords.progression_library`` (passing chords, walk-ups, IV/5,
  ...); a worship-style diatonic generator (I–V–vi–IV with add9/maj7/sus colours, slash basses,
  secondary dominants, passing °7, sus resolutions, borrowed iv, some minor keys); and random chords
  of the full 1,668-chord vocabulary (``train.chords.sample_chord``);
- **meter / tempo**: 4/4 (55–140 BPM), 3/4 (55–140) or 12/8 (dotted quarter 40–75, three eighths
  per beat); chords change on beats (passing chords on half beats);
- **piano** (always there): the Salamander Grand Piano (``train.piano``) playing ``train.chords``
  voicings (left hand + right hand, two-hand blocks, one hand; close / open / spread), voice-led
  from chord to chord, in a performance style per section (block, rolled, arpeggio, pulse,
  bass_chord), legato pedal that is sometimes lifted a little late (the old chord bleeds into the
  new one, as with real pianists), soft passing chords;
- **band** (each optional, per section; ``--piano-focus`` items are mostly piano alone or piano +
  voice): a synthetic bass on the chord's bass note (slash basses
  included), a string pad that holds the previous MAIN chord across passing chords, noise drums,
  a vocal-like melody (chord tones plus some non-chord neighbour notes), using the layer synths of
  ``eval/synth/render.py``; 1–3 arrangement sections (e.g. piano alone, then the band comes in);
- **recording**: random layer levels, reverb (RT60 0.3–3 s), EQ tilt / high-pass / low-pass,
  background noise, a limiter.

Labels, per ACE frame (hop 512 at 22.05 kHz, frame k at k·512/22050 s): ``root`` pc 0–11 or 12 = N,
``bass`` pc or 12, ``notes`` 12-bit mask of the chord's pitch classes (template tones ∪ bass,
``ChordSpec.pcs``: what the chord IS, not only what this voicing plays). N before the first chord
and after the last chord's sound has ended (lead-in, ending ring-out, zero padding).

Features, exactly ACE's (``nathan_chordify.models.ace``): mono 22.05 kHz, |CQT| hop 512, 24 bins /
octave, 6 octaves from C1 → [144, F]. They are stored UNNORMALISED for the whole excerpt (padded
with zeros to ≥ 20 s) as float16 together with the per-frame peak of |audio| (``env``): a training
crop of 862 frames (= ACE's 20 s window) is divided by the peak of its own frames, which equals
ACE's per-window peak normalisation (the CQT is linear).

The pool (``data/train/pool/``): ``<split>/shard-<seed>-<k>.npz`` (``cqt`` [144, ΣF] float16,
``env``, ``root``, ``bass`` int8, ``notes`` uint16, ``offsets`` [n+1], ``meta`` JSON per item) and
``pool.json`` (stats). Item ``i`` of a split is deterministic: ``rng = default_rng([seed, i])``.
The validation split uses its own seed and its own chart songs (1 in 8, ``is_val_slug``).

CLI::

    python -m train.song_mix build --split train --items 12000 --workers 7
    python -m train.song_mix build --split val --items 600 --workers 7
    python -m train.song_mix stats
    python -m train.song_mix preview --n 6 --out data/train/preview_songs
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POOL_DIR = ROOT / "data" / "train" / "pool"
CHART_DIR = ROOT / "data" / "charts"
POOL_CAP_BYTES = 8 * 1024 ** 3

SR = 22050                        # ACE rate = the render rate (the CQT stops at C7 ≈ 2.1 kHz)
HOP = 512
N_BINS = 144
WIN_S = 20.0
WIN_SAMPLES = int(round(WIN_S * SR))          # 441,000 (ACE chunk)
WIN_FRAMES = 1 + WIN_SAMPLES // HOP           # 862
N_CLASS = 12                                  # root / bass index of "no chord"

SEEDS = {"train": 2026, "val": 7_000_001}
METERS = {"4/4": (4, 2), "3/4": (3, 2), "12/8": (4, 3)}      # beats per bar, subdivisions per beat
METER_P = {"4/4": 0.6, "3/4": 0.22, "12/8": 0.18}
SOURCE_P_CHARTS = {"chart": 0.45, "idiom": 0.15, "diatonic": 0.3, "random": 0.1}
P_RERHYTHM = 0.7                  # charts whose bar lengths were inferred: new chord lengths 70 % of the time
SOURCE_P_NO_CHARTS = {"idiom": 0.3, "diatonic": 0.55, "random": 0.15}
STYLES = ("block", "rolled", "arpeggio", "pulse", "bass_chord")
STYLE_P = (0.22, 0.13, 0.27, 0.2, 0.18)
LAYERS = ("bass", "pad", "drums", "vocal")
NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


# ------------------------------------------------------------------------------------------ chords

def _C():
    from train import chords
    return chords


def spec_of(label: str):
    """A Harte label / typed symbol → the vocabulary ChordSpec (out-of-vocabulary chords are
    simplified: a foreign bass is dropped, then extensions; None when nothing fits; N → None)."""
    from nathan_chordify.theory import harte as H
    from nathan_chordify.theory import templates as T
    C = _C()
    if label in ("N", "X", ""):
        return None
    try:
        p = H.parse_any(label)
        root, q, bass = p.root, p.quality, p.bass
    except (ValueError, KeyError):
        try:
            root, q, bass = H.parse_harte(label, strict=False)
        except (ValueError, KeyError):
            return None
    if q == T.N:
        return None
    for qq, bb in ((q, bass), (q, root)):
        try:
            return C.lookup(root, qq, bb)
        except KeyError:
            continue
    return None


# ----------------------------------------------------------------------------------- progressions

@dataclass
class PChord:
    spec: object                 # train.chords.ChordSpec
    beats: float
    passing: bool = False


def _merge(chords: list[PChord]) -> list[PChord]:
    out: list[PChord] = []
    for c in chords:
        if c.beats <= 1e-9:
            continue
        if out and out[-1].spec == c.spec:
            out[-1].beats += c.beats
            out[-1].passing = out[-1].passing and c.passing
        else:
            out.append(PChord(c.spec, c.beats, c.passing))
    return out


def _transpose(chords: list[PChord], k: int) -> list[PChord]:
    return [PChord(c.spec.transpose(k), c.beats, c.passing) for c in chords]


def _refit_meter(chords: list[PChord], src_bpb: int, bpb: int) -> list[PChord]:
    """Re-lay a progression written in ``src_bpb``-beat bars into ``bpb``-beat bars (each source
    bar becomes one bar: chord onsets scale by bpb/src_bpb and snap to half beats)."""
    if src_bpb == bpb:
        return chords
    out: list[PChord] = []
    pos = 0.0
    events = []
    for c in chords:
        events.append((pos, c))
        pos += c.beats
    total = pos
    n_bars = int(np.ceil(total / src_bpb - 1e-9))
    scale = bpb / src_bpb
    new = []
    for p0, c in events:
        q = np.round(p0 * scale * 2) / 2
        if new and abs(new[-1][0] - q) < 1e-9:
            new[-1] = (q, c)                       # a collision: the later chord wins
        else:
            new.append((q, c))
    end = n_bars * bpb
    for i, (q, c) in enumerate(new):
        q1 = new[i + 1][0] if i + 1 < len(new) else end
        if q1 > q:
            out.append(PChord(c.spec, q1 - q, c.passing and (q1 - q) <= 1.0))
    return _merge(out)


def _chart_meter(text: str) -> int:
    m = re.search(r"(?im)^#.*\b(?:meter|time)\b[^0-9]*(\d+)\s*/\s*(\d+)", text)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 8 and num % 3 == 0:
            return num // 3
        if num in (2, 3, 4, 5, 6):
            return num
    return 4


def _parse_chart_text(text: str, bpb: int) -> list[PChord]:
    """Bar text → chords with beat lengths (sections in order; unreadable lines are skipped)."""
    from nathan_chordify.theory import chart as CH
    try:
        bars = CH.parse(text, bpb).bars
    except CH.ChartError:
        bars = []
        for line in text.splitlines():
            if "|" not in line or line.strip().startswith("#"):
                continue
            try:
                bars += CH.parse(line, bpb).bars
            except CH.ChartError:
                continue
    out: list[PChord] = []
    for bar in bars:
        if not bar:
            continue
        if any(c.pos is None for c in bar):
            onsets = [i * bpb / len(bar) for i in range(len(bar))]
        else:
            onsets = [float(c.pos) for c in bar]
        for i, c in enumerate(bar):
            b = (onsets[i + 1] if i + 1 < len(bar) else bpb) - onsets[i]
            spec = spec_of(c.harte)
            if spec is None:
                if out:
                    out[-1].beats += b
                continue
            if i == 0 and onsets[0] > 0 and out:
                out[-1].beats += onsets[0]
            out.append(PChord(spec, b, b <= 1.0 and len(bar) > 1))
    return _merge(out)


_CHART_CACHE: dict[str, tuple[float, tuple]] = {}


def _slug(path: Path) -> str:
    return path.name.split(".")[0]


def chart_files(split: str | None = None) -> list[Path]:
    """Harvested chart files (``data/charts/*.chart.txt`` + the alternative versions in
    ``data/charts/alt/``); split "val" = 1 song in 8 (``is_val_slug``, with its alternative
    versions), "train" = the rest."""
    if not CHART_DIR.is_dir():
        return []
    files = sorted(CHART_DIR.glob("*.chart.txt")) + sorted((CHART_DIR / "alt").glob("*.chart.txt"))
    if split == "val":
        return [f for f in files if is_val_slug(_slug(f))]
    if split == "train":
        return [f for f in files if not is_val_slug(_slug(f))]
    return files


def is_val_slug(slug: str) -> bool:
    """A stable 1-in-8 song split (crc32 of the song's file slug)."""
    import zlib
    return zlib.crc32(slug.encode("utf-8")) % 8 == 0


def load_chart(path: Path) -> tuple[int, list[PChord], bool]:
    """(beats per bar, chords in the written key, bar lengths inferred?) of a chart file
    (cached per mtime). "Inferred" = the header says ``# bars: inferred`` (chord order exact,
    lengths estimated from the lyrics)."""
    key = str(path)
    mt = path.stat().st_mtime
    hit = _CHART_CACHE.get(key)
    if hit and hit[0] == mt:
        return hit[1]
    text = path.read_text("utf-8", errors="replace")
    bpb = _chart_meter(text)
    inferred = bool(re.search(r"(?im)^#\s*bars\s*:\s*inferred", text))
    res = (bpb, _parse_chart_text(text, bpb), inferred)
    _CHART_CACHE[key] = (mt, res)
    return res


def rerhythm(chords: list[PChord], bpb: int, rng: np.random.Generator) -> list[PChord]:
    """Keep a chart's chord ORDER, draw new lengths: mostly whole and half bars, some 2-bar
    holds and 1-beat changes; a chord that starts mid-bar usually fills the bar."""
    seq = _merge([PChord(c.spec, c.beats, c.passing) for c in chords])
    if bpb == 3:
        lens, p = [1.0, 2.0, 3.0, 6.0], [0.1, 0.15, 0.6, 0.15]
    else:
        lens, p = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0], [0.07, 0.33, 0.04, 0.38, 0.04, 0.14]
    p = np.array(p) / sum(p)
    out, pos = [], 0.0
    for c in seq:
        inbar = pos % bpb
        if inbar and rng.random() < 0.8:
            L = bpb - inbar
        else:
            L = float(lens[int(rng.choice(len(lens), p=p))])
        out.append(PChord(c.spec, L, L <= 1.0))
        pos += L
    return _merge(out)


# worship-style diatonic generator --------------------------------------------------------------

_MAJ_DEG = {1: (0, "maj"), 2: (2, "min"), 3: (4, "min"), 4: (5, "maj"), 5: (7, "maj"), 6: (9, "min")}
_MIN_DEG = {1: (0, "min"), 3: (3, "maj"), 4: (5, "min"), 5: (7, "min"), 6: (8, "maj"), 7: (10, "maj")}
_MAJ_NEXT = {1: {4: 3, 5: 3, 6: 2.5, 2: 1, 3: 0.6}, 2: {5: 4, 4: 1, 1: 0.7, 3: 0.5},
             3: {6: 3, 4: 2, 2: 0.6}, 4: {1: 3, 5: 3.5, 6: 1, 2: 1},
             5: {1: 3, 6: 3, 4: 2}, 6: {4: 4, 5: 2, 2: 1.5, 3: 0.5, 1: 0.7}}
_MIN_NEXT = {1: {6: 3, 4: 2.5, 7: 2, 3: 1.5, 5: 1.5}, 3: {7: 2, 6: 2, 4: 1.5}, 4: {1: 2, 5: 2, 7: 2, 6: 1},
             5: {1: 3, 6: 1.5}, 6: {7: 3, 3: 1.5, 4: 1, 5: 1}, 7: {3: 3, 1: 2, 6: 1}}
_COLOURS = {
    "maj": (("maj", 0.45), ("add9", 0.15), ("maj7", 0.1), ("sus2", 0.08), ("6", 0.05), ("maj9", 0.05),
            ("6/9", 0.04), ("sus4", 0.04), ("5", 0.02), ("maj7#11", 0.02)),
    "min": (("min", 0.48), ("m7", 0.32), ("madd9", 0.07), ("m9", 0.07), ("m11", 0.04), ("m6", 0.02)),
}
_DOM_COLOURS = (("maj", 0.38), ("7", 0.18), ("sus4", 0.13), ("7sus4", 0.1), ("9sus4", 0.08), ("13sus4", 0.03),
                ("9", 0.04), ("13", 0.03), ("add9", 0.03))


def _pick(rng, pairs):
    names = [p[0] for p in pairs]
    w = np.array([p[1] for p in pairs], dtype=float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def _lookup(root: int, q: str, bass: int | None = None):
    C = _C()
    try:
        return C.lookup(root, q, root if bass is None else bass)
    except KeyError:
        return C.lookup(root, q, root)


def diatonic_progression(rng: np.random.Generator, key: int, bpb: int, n_bars: int) -> list[PChord]:
    """A made-up worship progression: Markov over scale degrees, colours, slash basses, passing
    chords (secondary dominants, °7, walk-ups, sus resolutions, borrowed iv)."""
    minor = rng.random() < 0.18
    degs, nxt = (_MIN_DEG, _MIN_NEXT) if minor else (_MAJ_DEG, _MAJ_NEXT)
    deg = 1 if rng.random() < 0.6 else int(rng.choice(list(degs)))
    out: list[PChord] = []
    rhythm = rng.choice(["bar", "half", "mixed"], p=[0.4, 0.25, 0.35])
    for bar in range(n_bars):
        if rhythm == "bar" or (rhythm == "mixed" and rng.random() < 0.55):
            lens = [float(bpb)]
        elif bpb == 3:
            lens = [2.0, 1.0] if rng.random() < 0.7 else [1.0, 2.0]
        else:
            lens = [2.0, 2.0] if rng.random() < 0.8 else [3.0, 1.0]
        for L in lens:
            iv, base_q = degs[deg]
            root = (key + iv) % 12
            if deg == 5 and not minor:
                q = _pick(rng, _DOM_COLOURS)
            else:
                q = _pick(rng, _COLOURS[base_q])
            bass = None
            u = rng.random()
            if deg == 1 and u < 0.15:
                bass = (root + (4 if base_q == "maj" else 3)) % 12          # I/3
            elif deg == 1 and u < 0.22:
                bass = (root + 7) % 12                                     # I/5
            elif deg == 5 and u < 0.25:
                bass = (root + 4) % 12                                     # V/3 (G/B)
            elif deg == 4 and u < 0.1:
                bass = (root + 7) % 12 if rng.random() < 0.5 else root     # IV/5 or IV
            elif deg == 6 and u < 0.12:
                bass, q = (root + 10) % 12, "m7"                           # vi7/5 (Am/G)
            elif deg == 2 and u < 0.12 and not minor:
                bass = (root + 3) % 12                                     # ii/4 (Dm/F)
            spec = _lookup(root, q, bass)
            out.append(PChord(spec, L))
            new = int(rng.choice(list(nxt[deg]), p=np.array(list(nxt[deg].values())) / sum(nxt[deg].values())))
            # a passing chord into the next chord, taking the last beat (or half beat)
            if L >= 2 and rng.random() < 0.28:
                p_len = 1.0 if rng.random() < 0.6 else 0.5
                t_iv, t_q = degs[new]
                target = (key + t_iv) % 12
                kind = rng.choice(["secdom", "dim", "sus", "walk", "borrowed"], p=[0.3, 0.2, 0.2, 0.2, 0.1])
                if kind == "secdom":
                    ps = _lookup((target + 7) % 12, "7", (target + 11) % 12 if rng.random() < 0.5 else None)
                elif kind == "dim":
                    ps = _lookup((target + 11) % 12, "dim7")
                elif kind == "sus":
                    ps = _lookup(target, "sus4")
                elif kind == "walk":
                    step = (target - 2) % 12 if rng.random() < 0.5 else (target + 2) % 12
                    ps = _lookup(step, "min" if ((step - key) % 12) in (2, 4, 9) else "maj")
                else:
                    ps = _lookup((key + 5) % 12, "m6" if rng.random() < 0.5 else "min")
                out[-1].beats -= p_len
                out.append(PChord(ps, p_len, True))
            deg = new
    return _merge(out)


def random_chords(rng: np.random.Generator, total_beats: float) -> list[PChord]:
    """Random chords of the whole vocabulary (balanced roots / qualities / inversions), 1–4 beats."""
    C = _C()
    out, b = [], 0.0
    while b < total_beats - 1e-9:
        L = float(rng.choice([1.0, 2.0, 2.0, 3.0, 4.0, 4.0]))
        out.append(PChord(C.sample_chord(rng, "balanced"), L, L <= 1.0))
        b += L
    return _merge(out)


def _loop_to(chords: list[PChord], total: float, start: float = 0.0) -> list[PChord]:
    """Cycle a progression (starting ``start`` beats in) until ``total`` beats are filled."""
    seq = []
    if not chords:
        return seq
    length = sum(c.beats for c in chords)
    pos, b = start % length, 0.0
    i = 0
    acc = 0.0
    while acc + chords[i].beats <= pos + 1e-9:          # find the chord at `pos`
        acc += chords[i].beats
        i = (i + 1) % len(chords)
    first = acc + chords[i].beats - pos
    while b < total - 1e-9:
        c = chords[i]
        L = first if first is not None else c.beats
        first = None
        L = min(L, total - b)
        seq.append(PChord(c.spec, L, c.passing))
        b += L
        i = (i + 1) % len(chords)
    return _merge(seq)


def make_progression(rng: np.random.Generator, source: str, bpb: int, total_beats: float,
                     split: str = "train") -> tuple[dict, list[PChord]]:
    """(info, chords filling ``total_beats``) from one of the sources."""
    C = _C()
    key = int(rng.integers(12))
    info: dict = {"source": source, "key": NAMES[key]}
    if source == "chart":
        files = chart_files(split)
        for _ in range(8):
            if not files:
                break
            f = files[int(rng.integers(len(files)))]
            try:
                src_bpb, chords, inferred = load_chart(f)
            except Exception:          # noqa: BLE001 - another worker writes these files right now
                continue
            if sum(c.beats for c in chords) < 8:
                continue
            re_r = bool(inferred and rng.random() < P_RERHYTHM)
            chords = rerhythm(chords, src_bpb, rng) if re_r else chords
            chords = _refit_meter(chords, src_bpb, bpb)
            k = int(rng.integers(12))
            chords = _transpose(chords, k)
            length = sum(c.beats for c in chords)
            start = float(int(rng.integers(0, max(1, int(length // bpb)))) * bpb)
            info.update(chart=f.name, transpose=k, src_meter=src_bpb, rerhythm=re_r)
            info.pop("key")
            return info, _loop_to(chords, total_beats, start)
        source = "diatonic"
        info["source"] = "diatonic"
    if source == "idiom":
        name, key, chords = C.random_progression(rng, n_chords=99)
        chords = [PChord(p.chord, p.beats, p.passing) for p in chords]
        chords = _refit_meter(_merge(chords), 4, bpb)
        info.update(progression=name, key=NAMES[key])
        return info, _loop_to(chords, total_beats)
    if source == "diatonic":
        n_bars = int(np.ceil(total_beats / bpb)) + 1
        return info, _loop_to(diatonic_progression(rng, key, bpb, min(n_bars, 8)), total_beats)
    if source == "random":
        info.pop("key")
        return info, random_chords(rng, total_beats)
    raise ValueError(f"unknown source {source!r}")


# ------------------------------------------------------------------------------------- rendering

@dataclass
class Seg:
    b0: float
    b1: float
    t0: float
    t1: float
    spec: object
    passing: bool


@dataclass
class SongPlan:
    meter: str
    bpm: float
    lead: float
    segs: list[Seg]
    end_t: float                 # last chord's release
    sections: list[tuple[float, dict]] = field(default_factory=list)   # (start time, layer gains dB or None)
    info: dict = field(default_factory=dict)


def _beat_time(b, lead: float, spb: float, swing: float, sub: int):
    b = np.asarray(b, dtype=float)
    n = np.floor(b + 1e-9)
    f = np.clip(b - n, 0.0, None)
    if swing != 0.5 and sub == 2:
        f = np.where(f < 0.5, f * 2 * swing, swing + (f - 0.5) * 2 * (1 - swing))
    return lead + (n + f) * spb


def _layer_plan(rng, n_sections: int) -> list[dict]:
    """Per section: level (dB re the piano) of each band layer, or None = off. Later sections
    tend to be fuller (worship songs build up)."""
    out = []
    full = rng.random()
    for k in range(n_sections):
        build = (k + 1) / n_sections
        sec = {}
        for name, p_on, lo, hi in (("bass", 0.55, -9, 2), ("pad", 0.45, -18, -3),
                                   ("drums", 0.45, -16, -3), ("vocal", 0.55, -9, 3)):
            on = rng.random() < p_on * (0.5 + build) * (0.6 + 0.8 * full)
            sec[name] = round(float(rng.uniform(lo, hi)), 1) if on else None
        sec["piano"] = round(float(rng.uniform(-6, 0)) if any(v is not None for v in sec.values()) else 0.0, 1)
        out.append(sec)
    return out


def _arrange(plan: list[dict], arrangement: str, rng) -> list[dict]:
    """Piano-focused arrangements: "solo" = piano alone; "vocal" = piano + voice (sometimes a
    soft pad or bass); "band" = the plan as drawn."""
    if arrangement == "band":
        return plan
    out = []
    for sec in plan:
        sec = {k: None for k in LAYERS} | {"piano": 0.0}
        if arrangement == "vocal":
            sec["vocal"] = round(float(rng.uniform(-9, 3)), 1)
            if rng.random() < 0.3:
                sec["pad"] = round(float(rng.uniform(-20, -8)), 1)
            if rng.random() < 0.2:
                sec["bass"] = round(float(rng.uniform(-12, -3)), 1)
        out.append(sec)
    return out


def plan_song(rng: np.random.Generator, split: str = "train", charts_ok: bool | None = None,
              piano_focus: bool = False, focus_rng: np.random.Generator | None = None) -> SongPlan:
    """One song plan. ``piano_focus``: the arrangement is "solo" (45 %), "vocal" (35 %) or the
    usual band plan (20 %), drawn from ``focus_rng`` (so the rest of the plan is unchanged)."""
    meter = str(rng.choice(list(METER_P), p=np.array(list(METER_P.values()))))
    bpb, sub = METERS[meter]
    bpm = float(rng.uniform(40, 75)) if meter == "12/8" else float(rng.uniform(55, 140))
    spb = 60.0 / bpm
    dur = float(rng.uniform(15, 20)) if rng.random() < 0.2 else float(rng.uniform(20, 30))
    lead = float(rng.uniform(0.0, 1.2)) if rng.random() < 0.35 else 0.0
    swing = 0.5
    if meter == "4/4" and bpm < 100 and rng.random() < 0.12:
        swing = float(rng.uniform(0.58, 0.67))
    ending = rng.random() < 0.25
    music_s = dur - lead - (float(rng.uniform(1.0, 2.5)) if ending else 0.0)
    total_beats = max(2.0, np.floor(music_s / spb * 2) / 2) if ending else np.ceil(music_s / spb) + 1
    if charts_ok is None:
        charts_ok = bool(chart_files(split))
    probs = SOURCE_P_CHARTS if charts_ok else SOURCE_P_NO_CHARTS
    source = str(rng.choice(list(probs), p=np.array(list(probs.values()))))
    info, chords = make_progression(rng, source, bpb, float(total_beats), split)
    segs, b = [], 0.0
    for c in chords:
        t0, t1 = _beat_time([b, b + c.beats], lead, spb, swing, sub)
        segs.append(Seg(b, b + c.beats, float(t0), float(t1), c.spec, c.passing))
        b += c.beats
    end_t = float(segs[-1].t1) + 0.3        # an ending: the last chord rings a little past the release
    if not ending:                          # the excerpt is cut in the middle of the music
        end_t = max(float(segs[-1].t1), dur + 0.5)
    n_sec = int(rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15]))
    bars = int((dur - lead) / spb // bpb)
    cuts = sorted(rng.choice(np.arange(1, max(2, bars)), size=min(n_sec - 1, max(0, bars - 1)), replace=False)) \
        if n_sec > 1 and bars > 2 else []
    starts = [0.0] + [float(_beat_time(c * bpb, lead, spb, swing, sub)) for c in cuts]
    plan = _layer_plan(rng, len(starts))
    arrangement = "band"
    if piano_focus:
        frng = focus_rng if focus_rng is not None else np.random.default_rng(0)
        arrangement = str(frng.choice(["solo", "vocal", "band"], p=[0.45, 0.35, 0.2]))
        plan = _arrange(plan, arrangement, frng)
    info.update(meter=meter, bpm=round(bpm, 2), swing=round(swing, 3), duration=round(dur, 3),
                ending=bool(ending), lead=round(lead, 3), arrangement=arrangement if piano_focus else "mixed")
    return SongPlan(meter, bpm, lead, segs, end_t, list(zip(starts, plan)), info)


def _movement(prev, new) -> float:
    if not prev:
        return 0.0
    a = sum(min(abs(m - p) for p in prev) for m in new)
    b = sum(min(abs(m - p) for m in new) for p in prev)
    return (a + b) / 2 + 0.5 * abs(new[-1] - prev[-1])


def plan_piano(plan: SongPlan, rng) -> tuple[list, list[tuple[float, float]], dict]:
    """Piano notes + pedal intervals for the whole song (voices from ``train.chords``)."""
    C = _C()
    bpb, sub = METERS[plan.meter]
    spb = 60.0 / plan.bpm
    layout = str(rng.choice(["lh_rh", "two_hand_block", "one_hand"], p=[0.62, 0.26, 0.12]))
    sec_styles = [str(rng.choice(STYLES, p=STYLE_P)) for _ in plan.sections]
    pedal_on = rng.random() < 0.7
    late_lift = float(rng.uniform(0.0, 0.12)) if rng.random() < 0.35 else 0.0
    base_vel = float(rng.uniform(45, 95))
    notes, pedal = [], []
    prev_rh, prev_bass = (), None
    for s in plan.segs:
        k = max(i for i, (t, _) in enumerate(plan.sections) if t <= s.t0 + 1e-6)
        style = sec_styles[k]
        lay = layout
        if lay == "one_hand" and s.spec.inversion == "non_chord":
            lay = "lh_rh"
        cands = C.voicings(s.spec, rng, 10, layout=lay) or C.voicings(s.spec, rng, 4)
        if not cands:
            continue
        cost = np.array([_movement(prev_rh, v.rh) + (0.3 * abs(v.notes[0] - prev_bass) if prev_bass else 0.0)
                         for v in cands])
        w = np.exp(-(cost - cost.min()) / 2.0)
        v = cands[int(rng.choice(len(cands), p=w / w.sum()))]
        prev_rh, prev_bass = v.rh, v.notes[0]
        st = "block" if (s.passing or s.b1 - s.b0 <= 1.0) and style in ("arpeggio", "pulse", "bass_chord") else style
        vel = base_vel * (0.82 if s.passing else 1.0) + float(rng.normal(0, 4))
        notes += C.perform(v, st, s.t0, s.t1, rng, vel, spb, pedal_on)
        if pedal_on:
            pedal.append((s.t0 + float(rng.uniform(0.02, 0.08)), s.t1 + late_lift))
    info = {"layout": layout, "styles": sec_styles, "pedal": bool(pedal_on), "late_lift": round(late_lift, 3)}
    return notes, pedal, info


@contextlib.contextmanager
def _synth_rate(sr: int):
    """Run eval/synth/render.py's layer synths at ``sr`` (they read its module-level SR)."""
    from eval.synth import render as R
    old = R.SR
    R.SR = sr
    try:
        yield R
    finally:
        R.SR = old


def _render_band(plan: SongPlan, n: int, rng_seed: int) -> dict[str, np.ndarray]:
    """Bass / pad / drums / vocal buses (2, n) at 22.05 kHz with the synths of eval/synth/render.py."""
    with _synth_rate(SR) as R:
        return _render_band_r(R, plan, n, rng_seed)


def _render_band_r(R, plan: SongPlan, n: int, rng_seed: int) -> dict[str, np.ndarray]:
    bpb, sub = METERS[plan.meter]
    meter = {"4/4": "4/4", "3/4": "3/4", "12/8": "6/8"}[plan.meter]
    rng = lambda k: np.random.default_rng([rng_seed, k])  # noqa: E731
    used = {name for _, sec in plan.sections for name in LAYERS if sec.get(name) is not None}
    p = R.RenderParams(bpm=plan.bpm, meter=meter, swing=plan.info.get("swing", 0.5),
                       bass_pattern="eighths" if rng(0).random() < 0.35 else "hold",
                       humanize_ms=float(rng(0).uniform(2, 8)), lead_in=plan.lead, seed=rng_seed)
    segs = [R._Seg(s.b0, s.b1, s.spec.label, s.spec.symbol, s.passing, (), s.t0, s.t1) for s in plan.segs]
    total = plan.segs[-1].b1
    end_t = plan.segs[-1].t1
    buses = {k: np.zeros((2, n)) for k in LAYERS}
    if "bass" in used:
        for nt in R._plan_bass(segs, p, rng(2)):
            R._add(buses["bass"], R._bass_note(nt.midi, max(0.02, nt.t1 - nt.t0)), nt.t0, 0.0, nt.vel)
    if "pad" in used:
        for nt in R._plan_pad(segs, end_t):
            release = 1.2 if nt.t1 < end_t - 1e-6 else 0.25
            left, right = R._pad_note(nt.midi, max(0.02, nt.t1 - nt.t0), release)
            R._add(buses["pad"], left, nt.t0, -1.0, nt.vel * 0.707)
            R._add(buses["pad"], right, nt.t0, 1.0, nt.vel * 0.707)
    if "vocal" in used:
        vnotes = _vocal_notes(R._plan_vocal(segs, total, p, rng(3)), plan, rng(8))
        R._add(buses["vocal"], R._vocal(vnotes, n, rng(5)), 0.0, 0.0)
    if "drums" in used:
        kit = R._drum_kit(rng(6))
        pans = {"kick": 0.0, "snare": -0.1, "hat": 0.3}
        for kind, t0, vel in plan_drums(plan, total, rng(4)):
            R._add(buses["drums"], kit[kind], t0, pans[kind], vel)
    return buses


def _vocal_notes(notes, plan: SongPlan, rng):
    """Some sung notes start on a neighbour (non-chord) tone and resolve: appoggiaturas."""
    from eval.synth import render as R
    out = []
    for nt in notes:
        if nt.t1 - nt.t0 > 0.5 and rng.random() < 0.2:
            mid = nt.t0 + (nt.t1 - nt.t0) * float(rng.uniform(0.3, 0.5))
            out.append(R.Note("vocal", nt.midi + int(rng.choice([-2, -1, 1, 2])), nt.t0, mid, nt.vel))
            out.append(R.Note("vocal", nt.midi, mid, nt.t1, nt.vel))
        else:
            out.append(nt)
    return out


def plan_drums(plan: SongPlan, total_beats: float, rng) -> list[tuple[str, float, float]]:
    bpb, sub = METERS[plan.meter]
    spb = 60.0 / plan.bpm
    swing = plan.info.get("swing", 0.5)
    groove = str(rng.choice(["backbeat", "half", "light"], p=[0.5, 0.25, 0.25]))
    hits = []
    for k in range(int(np.ceil(total_beats - 1e-9))):
        pos = k % bpb
        if pos == 0 or (groove == "backbeat" and bpb == 4 and pos == 2 and plan.meter != "12/8"):
            hits.append(("kick", k, 1.0))
        if groove == "half":
            if (bpb == 4 and pos == 2) or (bpb == 3 and pos == 2):
                hits.append(("snare", k, 0.9))
        elif groove == "backbeat":
            if (bpb == 4 and pos in (1, 3)) or (bpb == 3 and pos in (1, 2)):
                hits.append(("snare", k, 0.85))
        if groove != "light" or pos == 0:
            hits += [("hat", k + j / sub, 0.55 if j == 0 else 0.35) for j in range(sub)]
        if groove == "backbeat" and rng.random() < 0.15:
            hits.append(("kick", k + (sub - 1) / sub, 0.6))
    jit = lambda: float(np.clip(rng.normal(0, 0.004), -0.01, 0.01))  # noqa: E731
    return [(kind, float(_beat_time(b, plan.lead, spb, swing, sub)) + jit(), vel)
            for kind, b, vel in hits if b < total_beats - 1e-9]


def _gain_curve(plan: SongPlan, name: str, n: int, sr: int) -> np.ndarray | None:
    """Linear gain per sample for a layer over the arrangement sections (20 ms ramps)."""
    vals = []
    for t, sec in plan.sections:
        db = sec.get(name)
        vals.append((t, 0.0 if db is None else 10 ** (db / 20)))
    if all(v == 0.0 for _, v in vals):
        return None
    xs, ys = [0.0], [vals[0][1]]
    for t, v in vals[1:]:
        xs += [t - 0.01, t + 0.01]
        ys += [ys[-1], v]
    return np.interp(np.arange(n) / sr, xs, ys)


def _eq(y: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    """Random recording colour: tilt (low/high shelf), high-pass, low-pass; y (2, n) at 44.1 kHz."""
    from scipy.signal import butter, lfilter
    info = {}
    if rng.random() < 0.5:
        f = float(rng.uniform(40, 180))
        b, a = butter(2, f / (SR / 2), btype="high")
        y = lfilter(b, a, y, axis=1)
        info["hp"] = round(f, 1)
    if rng.random() < 0.35:
        f = float(rng.uniform(2500, 9000))
        b, a = butter(2, f / (SR / 2), btype="low")
        y = lfilter(b, a, y, axis=1)
        info["lp"] = round(f, 1)
    if rng.random() < 0.5:                                  # tilt: blend with a one-pole low-pass
        g = float(rng.uniform(-6, 6))
        k = np.exp(-2 * np.pi * 800 / SR)
        low = lfilter([1 - k], [1, -k], y, axis=1)
        high = y - low
        y = low * 10 ** (-g / 40) + high * 10 ** (g / 40)
        info["tilt_db"] = round(g, 2)
    return y, info


def render_song(plan: SongPlan, seed: int, piano=None) -> tuple[np.ndarray, dict]:
    """Mono float32 audio at 22.05 kHz (the plan's ``duration`` long) and render info."""
    from scipy.signal import lfilter
    from train import piano as P
    rng = np.random.default_rng([seed, 99])
    piano = piano or get_piano()
    if piano.sr != SR:
        raise ValueError(f"the piano must render at {SR} Hz (song_mix.get_piano())")
    dur = plan.info["duration"]
    n = int(round(dur * SR))
    notes, pedal, pinfo = plan_piano(plan, rng)
    tune = float(np.clip(rng.normal(0, 6), -20, 20)) if rng.random() < 0.5 else 0.0
    pn = piano.render(notes, pedal=pedal or None, duration=dur, tune_cents=tune, seed=int(rng.integers(2 ** 31)),
                      pedal_noise=bool(rng.random() < 0.5))
    buses = {"piano": pn[:n].T.astype(np.float64)}
    if buses["piano"].shape[1] < n:
        buses["piano"] = np.pad(buses["piano"], ((0, 0), (0, n - buses["piano"].shape[1])))
    buses.update(_render_band(plan, n, int(rng.integers(2 ** 31))))
    i0, i1 = int(plan.lead * SR), max(int(plan.lead * SR) + 1, min(n, int(plan.segs[-1].t1 * SR)))
    mix, send = np.zeros((2, n)), np.zeros(n)
    sends = {"piano": 1.0, "bass": 0.1, "pad": 1.0, "drums": 0.35, "vocal": 1.0}
    levels = {}
    for name, bus in buses.items():
        rms = float(np.sqrt(np.mean(bus[:, i0:i1] ** 2))) if i1 > i0 else 0.0
        if rms == 0.0:
            continue
        g = _gain_curve(plan, name, n, SR)
        if g is None:
            continue
        bus = bus * (0.1 / rms) * g[None, :]
        mix += bus
        send += sends[name] * bus.mean(axis=0)
        levels[name] = True
    rt60 = float(rng.uniform(0.3, 3.0))
    wet = float(rng.uniform(0.05, 0.45)) if rng.random() < 0.85 else 0.0
    if wet > 0:
        from scipy.signal import fftconvolve
        with _synth_rate(SR) as R:
            ir = R._reverb_ir(rt60, rng)
        for ch in range(2):
            mix[ch] += wet * fftconvolve(send, ir[ch])[:n]
    mix, eq = _eq(mix, rng)
    noise_db = None
    if rng.random() < 0.45:
        noise_db = float(rng.uniform(-65, -30))
        white = rng.standard_normal((2, n))
        nz = white if rng.random() < 0.5 else lfilter([1.0], [1.0, -0.97], white, axis=1)
        nz /= np.sqrt(np.mean(nz ** 2)) + 1e-12
        ref = float(np.sqrt(np.mean(mix ** 2))) + 1e-12
        mix += nz * ref * 10 ** (noise_db / 20)
    peak = float(np.max(np.abs(mix))) + 1e-12
    mix = mix / peak * 10 ** (float(rng.uniform(-14, -1)) / 20)
    if rng.random() < 0.4:                                  # loud master: push into the limiter
        mix = P.limiter((mix * 10 ** (float(rng.uniform(2, 8)) / 20)).T.astype(np.float32), -1.0, sr=SR).T
    y = mix.mean(axis=0).astype(np.float32)
    info = {**pinfo, "layers": sorted(k for k in levels if k != "piano"), "rt60": round(rt60, 2),
            "wet": round(wet, 3), "eq": eq, "noise_db": None if noise_db is None else round(noise_db, 1),
            "tune_cents": round(tune, 2), "sections": [[round(t, 3), s] for t, s in plan.sections]}
    return y, info


# ----------------------------------------------------------------------------------------- labels

def frame_times(n_frames: int) -> np.ndarray:
    return np.arange(n_frames) * HOP / SR


def n_frames_of(n_samples: int) -> int:
    return 1 + int(n_samples) // HOP


def frame_labels(plan: SongPlan, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(root, bass, notes-bitmask) per ACE frame from the chord timeline (N outside it)."""
    t = frame_times(n_frames)
    root = np.full(n_frames, N_CLASS, np.int8)
    bass = np.full(n_frames, N_CLASS, np.int8)
    notes = np.zeros(n_frames, np.uint16)
    stop = min(plan.end_t, float(plan.info.get("duration", np.inf)))
    for i, s in enumerate(plan.segs):
        t1 = min(s.t1 if i + 1 < len(plan.segs) else plan.end_t, stop)
        m = (t >= s.t0) & (t < t1)
        root[m] = s.spec.root
        bass[m] = s.spec.bass
        notes[m] = sum(1 << p for p in s.spec.pcs)
    return root, bass, notes


def notes_matrix(mask: np.ndarray) -> np.ndarray:
    """uint16 bitmask [F] → float32 [F, 12]."""
    return ((mask[:, None].astype(np.int32) >> np.arange(12)[None, :]) & 1).astype(np.float32)


def features(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(|CQT| [144, F] float32 of the zero-padded audio, per-frame peak |y| [F])."""
    from nathan_chordify.models import ace
    L = max(len(y), WIN_SAMPLES)
    seg = np.zeros(L, np.float32)
    seg[:len(y)] = y
    import torch
    cqt = ace.import_ace()["cqt"](SR, HOP)(torch.from_numpy(seg)[None]).numpy()[0].astype(np.float32)
    F = cqt.shape[1]
    padded = np.concatenate([np.zeros(HOP // 2, np.float32), np.abs(seg), np.zeros(HOP, np.float32)])
    env = padded[:HOP * F].reshape(F, HOP).max(axis=1)
    return cqt, env.astype(np.float32)


def make_item(i: int, seed: int, split: str = "train", piano=None, charts_ok: bool | None = None,
              with_audio: bool = False, piano_focus: bool = False) -> dict:
    """Item ``i`` of a split: plan → audio → features + frame labels (deterministic)."""
    rng = np.random.default_rng([int(seed), int(i)])
    plan = plan_song(rng, split, charts_ok, piano_focus, np.random.default_rng([int(seed), int(i), 7]))
    y, rinfo = render_song(plan, int(rng.integers(2 ** 31)), piano)
    cqt, env = features(y)
    F = cqt.shape[1]
    root, bass, notes = frame_labels(plan, F)
    chords = [[round(s.t0, 4), round(s.t1, 4), s.spec.label, bool(s.passing)] for s in plan.segs]
    meta = {"id": f"{split}-{seed}-{i}", **plan.info, **rinfo, "frames": int(F), "chords": chords,
            "end_t": round(plan.end_t, 4)}
    out = {"cqt": cqt, "env": env, "root": root, "bass": bass, "notes": notes, "meta": meta}
    if with_audio:
        out["audio"] = y
    return out


# ------------------------------------------------------------------------------------------- pool

def _init_worker(threads: int = 1) -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass


def get_piano(pitch_cache_mb: float = 600.0):
    """The Salamander piano rendering at 22.05 kHz (one per process)."""
    from train import piano as P
    return P.get_piano(sr=SR, pitch_cache_mb=pitch_cache_mb, max_seconds=10.0)


def _worker_piano():
    return get_piano()


def write_shard(split: str, seed: int, shard: int, items: list[int], out_dir: Path | None = None,
                piano_focus: bool = False) -> dict:
    """Render ``items`` and write one shard file (skipped when it already exists)."""
    out_dir = Path(out_dir or POOL_DIR) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard-{seed}-{shard:04d}.npz"
    if path.exists():
        return {"path": str(path), "skipped": True}
    _init_worker(1)
    piano = _worker_piano()
    charts_ok = bool(chart_files(split))
    t = time.perf_counter()
    parts = [make_item(i, seed, split, piano, charts_ok, piano_focus=piano_focus) for i in items]
    offsets = np.cumsum([0] + [p["cqt"].shape[1] for p in parts])
    arrays = {
        "cqt": np.concatenate([p["cqt"] for p in parts], axis=1).astype(np.float16),
        "env": np.concatenate([p["env"] for p in parts]).astype(np.float16),
        "root": np.concatenate([p["root"] for p in parts]),
        "bass": np.concatenate([p["bass"] for p in parts]),
        "notes": np.concatenate([p["notes"] for p in parts]),
        "offsets": offsets.astype(np.int64),
        "meta": np.array(json.dumps([p["meta"] for p in parts], ensure_ascii=False)),
    }
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
    return {"path": str(path), "items": len(items), "seconds": round(time.perf_counter() - t, 1),
            "bytes": path.stat().st_size, "charts": charts_ok}


def pool_bytes(root: Path | None = None) -> int:
    root = Path(root or POOL_DIR)
    return sum(f.stat().st_size for f in root.rglob("*.npz")) if root.exists() else 0


def build(split: str, n_items: int, workers: int = 6, shard_size: int = 100, start: int = 0,
          out_dir: Path | None = None, seed: int | None = None, piano_focus: bool = False) -> list[dict]:
    """Render items [start, start + n_items) of a split into shards with a process pool."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    seed = SEEDS[split] if seed is None else seed
    out_dir = Path(out_dir or POOL_DIR)
    jobs = []
    for k in range(start // shard_size, int(np.ceil((start + n_items) / shard_size))):
        items = [i for i in range(k * shard_size, (k + 1) * shard_size) if start <= i < start + n_items]
        if items:
            jobs.append((k, items))
    res = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futs = {ex.submit(write_shard, split, seed, k, items, out_dir, piano_focus): k for k, items in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            res.append(r)
            size = pool_bytes(out_dir)
            print(f"[{split} {n}/{len(jobs)}] {Path(r['path']).name} {r.get('seconds', 'skip')} s, "
                  f"pool {size / 1e9:.2f} GB, {time.perf_counter() - t0:.0f} s elapsed", flush=True)
            if size > POOL_CAP_BYTES:
                print("pool cap (8 GB) reached: stopping", flush=True)
                for g in futs:
                    g.cancel()
                break
    return res


# ----------------------------------------------------------------------------------- reading/stats

def shard_files(split: str, root: Path | None = None) -> list[Path]:
    return sorted((Path(root or POOL_DIR) / split).glob("shard-*.npz"))


def load_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def stats(root: Path | None = None) -> dict:
    """Pool composition (items, hours, sources, meters, layers, styles, qualities, inversions)."""
    from nathan_chordify.theory import harte as H
    C = _C()
    out = {}
    for split in ("train", "val"):
        files = shard_files(split, root)
        if not files:
            continue
        c = {k: Counter() for k in ("source", "meter", "layers", "style", "quality", "inversion", "layout")}
        n = frames = 0
        secs = 0.0
        n_chords = n_passing = 0
        charts = set()
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                metas = json.loads(str(z["meta"]))
            for m in metas:
                n += 1
                frames += m["frames"]
                secs += m["duration"]
                c["source"][m["source"]] += 1
                c["meter"][m["meter"]] += 1
                c["layout"][m["layout"]] += 1
                for st in m["styles"]:
                    c["style"][st] += 1
                c["layers"][",".join(m["layers"]) or "piano only"] += 1
                if m.get("chart"):
                    charts.add(m["chart"])
                for _, _, lab, passing in m["chords"]:
                    r, q, b = H.parse_harte(lab)
                    spec = C.lookup(r, q, b)
                    c["quality"][q] += 1
                    c["inversion"][spec.inversion] += 1
                    n_chords += 1
                    n_passing += bool(passing)
        out[split] = {"items": n, "hours": round(secs / 3600, 2), "frames": frames, "chords": n_chords,
                      "passing": n_passing, "chart_files": len(charts),
                      "bytes": sum(f.stat().st_size for f in files),
                      **{k: dict(v.most_common()) for k, v in c.items()}}
    return out


def write_preview(out_dir: Path, n: int = 6, seed: int = 4242) -> list[dict]:
    """A few rendered items as FLAC + their chord lists (for listening)."""
    import soundfile as sf
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    infos = []
    for i in range(n):
        it = make_item(i, seed, "val", with_audio=True)
        name = f"{i:02d}_{it['meta']['source']}_{it['meta']['meter'].replace('/', '-')}.flac"
        sf.write(out_dir / name, it["audio"], SR, subtype="PCM_16", format="FLAC")
        infos.append({"file": name, **it["meta"]})
    (out_dir / "labels.json").write_text(json.dumps(infos, indent=1, ensure_ascii=False), "utf-8")
    return infos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m train.song_mix", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="render a split into data/train/pool/<split>/")
    b.add_argument("--split", choices=list(SEEDS), required=True)
    b.add_argument("--items", type=int, required=True)
    b.add_argument("--start", type=int, default=0)
    b.add_argument("--workers", type=int, default=6)
    b.add_argument("--shard-size", type=int, default=100)
    b.add_argument("--piano-focus", action="store_true", help="mostly piano alone / piano + voice")
    sub.add_parser("stats", help="print pool composition and write data/train/pool/pool.json")
    pv = sub.add_parser("preview", help="write a few items as FLAC for listening")
    pv.add_argument("--n", type=int, default=6)
    pv.add_argument("--out", default=str(ROOT / "data" / "train" / "preview_songs"))
    a = ap.parse_args(argv)
    from train import song_mix as SM          # pickled by module path for the worker processes
    if a.cmd == "build":
        SM.build(a.split, a.items, a.workers, a.shard_size, a.start, piano_focus=a.piano_focus)
    elif a.cmd == "stats":
        st = stats()
        st["total_bytes"] = pool_bytes()
        (POOL_DIR / "pool.json").write_text(json.dumps(st, indent=1, ensure_ascii=False), "utf-8")
        print(json.dumps(st, indent=1, ensure_ascii=False))
    elif a.cmd == "preview":
        for info in write_preview(Path(a.out), a.n):
            print(info["file"], info["meter"], info["bpm"], info["layers"], len(info["chords"]), "chords")
    return 0


if __name__ == "__main__":
    sys.exit(main())
