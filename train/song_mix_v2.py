"""FT2 training songs: plain chords, piano near middle C, simple mixes (the user's recipe, 2026-11-01).

The user listened to the FT1 preview and asked for: mostly plain major / minor chords, only add / sus /
dim / 7 / 9 colours on top, the piano "as close to middle C as possible", and simpler recordings. This
module is a NEW builder (``train/song_mix.py`` and the FT1 pool are left untouched); it reuses the FT1
building blocks read-only (chart parsing helpers, CQT features, the Salamander piano, the playing
styles of ``train/chords.py``, the band synths of ``eval/synth/render.py``) and writes the same shard
format, so ``train/finetune_ace.py`` trains on it unchanged.

What is different from FT1 (exact numbers; ``stats()`` reports what came out):

- **Chords** (``ALLOWED``): major, minor and only these colours — add9, madd9 (add); sus2, sus4, 7sus4
  (sus); dim, dim7 (dim); 7, maj7, m7 (7); 9, maj9, m9 (9). Anything else is mapped to the nearest
  allowed chord (``QUALITY_MAP``: 6→maj, m6→m, 11 (= 9sus4)→7sus4, 13sus4→7sus4, 13→9, aug→maj, +7→7,
  +maj7→maj7, m7b5→dim, mMaj7→m, 6/9→add9, m11→m7, 7b9/7#9/7b13→7, maj7#11→maj7, 5→maj), keeping
  the bass when the new chord allows it. Sources: harvested charts 58 %, a worship diatonic generator
  35 % (colour table tuned to ≈ 80 % plain chords overall, slash chords ≈ 12 %: I/3, I/5, V/3, IV/5,
  vi/5, ii/4), the idiom progressions (mapped) 4 %, random allowed chords 3 %.
- **Register** (MIDI, middle C = 60): right hand 53–81 (F3–A5), the inversion / octave drawn with weight
  exp(−|mean pitch − target| / 2) × (0.25 if no C3–B3 bass fits under it), target ~ N(61, 3); left hand = the bass note, 48–59 (C3–B3) preferred (weight 1) and 43–47 (G2–B2) rarely
  (weight 0.12), never lower, no octave doublings (a second left-hand note, when there is one, is a
  different chord tone); one-hand voicings put the bass at 50–62. Synth bass 40–55 (E2–G3), −14…−6 dB.
- **Recordings** (``ARRANGEMENT_P``): piano alone 45 %; piano + soft pad and/or voice 30 %; band
  (drums + bass, maybe pad / voice) 25 %, sometimes with a piano-only first section.
- **No-chord stretches** (labelled N): a silent lead-in in 60 % of items (0.3–2.5 s); a rest inside the
  music in 35 % of items (1–2 rests of 1 beat … 1 bar: keys and pedal up, pad / bass / voice stop,
  drums may go on); a proper ending in 35 % (the last chord is labelled to its release + 0.3 s, then N).
  A chord before a rest is labelled to the rest start + 0.25 s (the decay), then N.
- Meters and tempi as FT1 (4/4 60 %, 3/4 22 %, 12/8 18 %; 55–140 BPM, 12/8 dotted quarter 40–75).

Pool: ``data/train/pool2/{train,val}/shard-<seed>-<k>.npz`` + ``pool.json``; seeds differ from FT1's.
FT2 itself kept its pool in RAM instead (``render_items`` in worker processes, ``train/ft2.py
all-in-ram``) because C: was below the 20 GB free-space floor on 2026-11-01 (an antivirus scan and a
grown page file); the items are the same, deterministic by seed.

CLI::

    python -m train.song_mix_v2 preview --n 20             # data/train/preview2/ (FLAC + labels.csv)
    python -m train.song_mix_v2 build --split val --items 700 --workers 6
    python -m train.song_mix_v2 build --split train --items 12000 --workers 6
    python -m train.song_mix_v2 stats
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import song_mix as SM  # noqa: E402  (read-only reuse of the FT1 building blocks)

POOL_DIR = ROOT / "data" / "train" / "pool2"
PREVIEW_DIR = ROOT / "data" / "train" / "preview2"
POOL_CAP_BYTES = 8 * 1024 ** 3
MIN_FREE_BYTES = 20 * 1024 ** 3
SR, HOP = SM.SR, SM.HOP
SEEDS = {"train": 2027, "val": 7_000_002}
METERS = SM.METERS
METER_P = SM.METER_P

ALLOWED = ("maj", "min", "add9", "madd9", "sus2", "sus4", "7sus4", "dim", "dim7", "7", "maj7", "m7",
           "9", "maj9", "m9")
PLAIN = ("maj", "min")
QUALITY_MAP = {"6": "maj", "m6": "min", "9sus4": "7sus4", "13sus4": "7sus4", "13": "9", "aug": "maj",
               "+7": "7", "+maj7": "maj7", "m7b5": "dim", "mM7": "min", "6/9": "add9", "m11": "m7",
               "7b9": "7", "7#9": "7", "7b13": "7", "maj7#11": "maj7", "5": "maj"}
SOURCE_P = {"chart": 0.58, "diatonic": 0.35, "idiom": 0.04, "random": 0.03}
SOURCE_P_NO_CHARTS = {"diatonic": 0.8, "idiom": 0.12, "random": 0.08}
ARRANGEMENT_P = {"solo": 0.45, "soft": 0.30, "band": 0.25}
STYLES = ("block", "rolled", "arpeggio", "pulse", "bass_chord")
STYLE_P = (0.3, 0.15, 0.25, 0.15, 0.15)
LAYERS = SM.LAYERS

RH_RANGE = (53, 81)              # F3..A5
RH_MEAN = (61.0, 3.0)            # target mean pitch of the right hand ~ N(61, 3)
LH_MIN = 43                      # G2: never lower
LH_PREF = (48, 59)               # C3..B3
LH_LOW_W = 0.12                  # weight of a bass note in G2..B2
ONE_HAND_BASS = (50, 62)
SYNTH_BASS = (40, 55)            # E2..G3
P_LEAD, LEAD_S = 0.6, (0.3, 2.5)
P_REST, P_ENDING = 0.35, 0.35


def _C():
    from train import chords
    return chords


# ------------------------------------------------------------------------------------------ chords

def to_allowed(spec):
    """The nearest allowed chord (same root; bass kept when the new quality allows it)."""
    if spec is None:
        return None
    C = _C()
    q = spec.quality if spec.quality in ALLOWED else QUALITY_MAP[spec.quality]
    if q == spec.quality:
        return spec
    try:
        return C.lookup(spec.root, q, spec.bass)
    except KeyError:
        return C.lookup(spec.root, q, spec.root)


def allowed_spec(label: str):
    return to_allowed(SM.spec_of(label))


def _parse_chart_text(text: str, bpb: int) -> list:
    """``SM._parse_chart_text`` with every chord mapped to the allowed list."""
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
    out = []
    for bar in bars:
        if not bar:
            continue
        onsets = ([i * bpb / len(bar) for i in range(len(bar))] if any(c.pos is None for c in bar)
                  else [float(c.pos) for c in bar])
        for i, c in enumerate(bar):
            b = (onsets[i + 1] if i + 1 < len(bar) else bpb) - onsets[i]
            spec = allowed_spec(c.harte)
            if spec is None:
                if out:
                    out[-1].beats += b
                continue
            if i == 0 and onsets[0] > 0 and out:
                out[-1].beats += onsets[0]
            out.append(SM.PChord(spec, b, b <= 1.0 and len(bar) > 1))
    return SM._merge(out)


_CHART_CACHE: dict[str, tuple] = {}


def load_chart(path: Path) -> tuple[int, list, bool]:
    key, mt = str(path), path.stat().st_mtime
    hit = _CHART_CACHE.get(key)
    if hit and hit[0] == mt:
        return hit[1]
    import re
    text = path.read_text("utf-8", errors="replace")
    bpb = SM._chart_meter(text)
    res = (bpb, _parse_chart_text(text, bpb), bool(re.search(r"(?im)^#\s*bars\s*:\s*inferred", text)))
    _CHART_CACHE[key] = (mt, res)
    return res


_MAJ_DEG = {1: (0, "maj"), 2: (2, "min"), 3: (4, "min"), 4: (5, "maj"), 5: (7, "maj"), 6: (9, "min")}
_MIN_DEG = {1: (0, "min"), 3: (3, "maj"), 4: (5, "min"), 5: (7, "min"), 6: (8, "maj"), 7: (10, "maj")}
_MAJ_NEXT = SM._MAJ_NEXT
_MIN_NEXT = SM._MIN_NEXT
#: colour tables: plain ≈ 89 % before passing chords (≈ 85 % with them; ≈ 80 % over all sources)
COLOURS = {
    "maj": (("maj", 0.90), ("add9", 0.025), ("sus2", 0.025), ("maj7", 0.025), ("sus4", 0.015), ("maj9", 0.01)),
    "min": (("min", 0.86), ("m7", 0.11), ("madd9", 0.015), ("m9", 0.015)),
    "dom": (("maj", 0.84), ("7", 0.06), ("sus4", 0.045), ("7sus4", 0.03), ("9", 0.0125), ("sus2", 0.0125)),
}


def diatonic_progression(rng: np.random.Generator, key: int, bpb: int, n_bars: int) -> list:
    """A plain worship progression (Markov over scale degrees, few colours, ≈ 12 % slash chords,
    a passing chord into ≈ 10 % of the changes: secondary dominant, dim7, sus4, walk-up, borrowed iv)."""
    minor = rng.random() < 0.15
    degs, nxt = (_MIN_DEG, _MIN_NEXT) if minor else (_MAJ_DEG, _MAJ_NEXT)
    deg = 1 if rng.random() < 0.6 else int(rng.choice(list(degs)))
    out = []
    rhythm = rng.choice(["bar", "half", "mixed"], p=[0.45, 0.25, 0.3])
    for _ in range(n_bars):
        if rhythm == "bar" or (rhythm == "mixed" and rng.random() < 0.55):
            lens = [float(bpb)]
        elif bpb == 3:
            lens = [2.0, 1.0] if rng.random() < 0.7 else [1.0, 2.0]
        else:
            lens = [2.0, 2.0] if rng.random() < 0.85 else [3.0, 1.0]
        for L in lens:
            iv, base_q = degs[deg]
            root = (key + iv) % 12
            table = COLOURS["dom"] if (deg == 5 and not minor) else COLOURS[base_q]
            q = SM._pick(rng, table)
            bass = None
            u = rng.random()
            if deg == 1 and u < 0.08:
                bass = (root + (4 if base_q == "maj" else 3)) % 12            # I/3
            elif deg == 1 and u < 0.11:
                bass = (root + 7) % 12                                         # I/5
            elif deg == 5 and u < 0.18:
                bass = (root + 4) % 12                                         # V/3 (G/B)
            elif deg == 4 and u < 0.1:
                bass = (root + 2) % 12                                         # IV/5 (F/G)
            elif deg == 6 and u < 0.08:
                bass = (root + 10) % 12                                        # vi/5 (Am/G)
            elif deg == 2 and u < 0.08 and not minor:
                bass = (root + 3) % 12                                         # ii/4 (Dm/F)
            out.append(SM.PChord(_lookup(root, q, bass), L))
            w = np.array(list(nxt[deg].values()), dtype=float)
            new = int(rng.choice(list(nxt[deg]), p=w / w.sum()))
            if L >= 2 and rng.random() < 0.10:
                p_len = 1.0 if rng.random() < 0.7 else 0.5
                t_iv, _ = degs[new]
                target = (key + t_iv) % 12
                kind = rng.choice(["secdom", "dim", "sus", "walk", "borrowed"], p=[0.3, 0.15, 0.25, 0.2, 0.1])
                if kind == "secdom":
                    ps = _lookup((target + 7) % 12, "7" if rng.random() < 0.5 else "maj",
                                 (target + 11) % 12 if rng.random() < 0.4 else None)
                elif kind == "dim":
                    ps = _lookup((target + 11) % 12, "dim7" if rng.random() < 0.6 else "dim")
                elif kind == "sus":
                    ps = _lookup(target, "sus4")
                elif kind == "walk":
                    step = (target - 2) % 12 if rng.random() < 0.5 else (target + 2) % 12
                    ps = _lookup(step, "min" if ((step - key) % 12) in (2, 4, 9) else "maj")
                else:
                    ps = _lookup((key + 5) % 12, "min")
                out[-1].beats -= p_len
                out.append(SM.PChord(ps, p_len, True))
            deg = new
    return SM._merge(out)


def _lookup(root: int, q: str, bass: int | None = None):
    C = _C()
    try:
        return C.lookup(root, q, root if bass is None else bass)
    except KeyError:
        return C.lookup(root, q, root)


def random_chords(rng: np.random.Generator, total_beats: float) -> list:
    """Random chords from the allowed list only (uniform quality, 25 % with an allowed slash bass)."""
    from nathan_chordify.theory import templates as T
    out, b = [], 0.0
    while b < total_beats - 1e-9:
        L = float(rng.choice([1.0, 2.0, 2.0, 3.0, 4.0, 4.0]))
        root, q = int(rng.integers(12)), str(rng.choice(ALLOWED))
        ivs = [i for i in T.allowed_bass_intervals(q) if i]
        bass = (root + int(rng.choice(ivs))) % 12 if ivs and rng.random() < 0.25 else None
        out.append(SM.PChord(_lookup(root, q, bass), L, L <= 1.0))
        b += L
    return SM._merge(out)


def make_progression(rng: np.random.Generator, source: str, bpb: int, total_beats: float, split: str):
    C = _C()
    key = int(rng.integers(12))
    info: dict = {"source": source, "key": SM.NAMES[key]}
    if source == "chart":
        files = SM.chart_files(split)
        for _ in range(8):
            if not files:
                break
            f = files[int(rng.integers(len(files)))]
            src_bpb, chords, inferred = load_chart(f)
            if sum(c.beats for c in chords) < 8:
                continue
            re_r = bool(inferred and rng.random() < SM.P_RERHYTHM)
            chords = SM.rerhythm(chords, src_bpb, rng) if re_r else chords
            chords = SM._refit_meter(chords, src_bpb, bpb)
            k = int(rng.integers(12))
            chords = SM._transpose(chords, k)
            length = sum(c.beats for c in chords)
            start = float(int(rng.integers(0, max(1, int(length // bpb)))) * bpb)
            info.update(chart=f.name, transpose=k, src_meter=src_bpb, rerhythm=re_r)
            info.pop("key")
            return info, SM._loop_to(chords, total_beats, start)
        source = info["source"] = "diatonic"
    if source == "idiom":
        name, key, chords = C.random_progression(rng, n_chords=99)
        chords = SM._merge([SM.PChord(to_allowed(p.chord), p.beats, p.passing) for p in chords])
        chords = SM._refit_meter(chords, 4, bpb)
        info.update(progression=name, key=SM.NAMES[key])
        return info, SM._loop_to(chords, total_beats)
    if source == "diatonic":
        n_bars = int(np.ceil(total_beats / bpb)) + 1
        return info, SM._loop_to(diatonic_progression(rng, key, bpb, min(n_bars, 8)), total_beats)
    if source == "random":
        info.pop("key")
        return info, random_chords(rng, total_beats)
    raise ValueError(f"unknown source {source!r}")


# -------------------------------------------------------------------------------------------- plan

def _insert_rests(chords: list, bpb: int, rng) -> tuple[list, int]:
    """Replace the end of 1–2 chords (≥ 2 beats long, not the first) by a rest (spec None) of
    1 beat … 1 bar, so the next chord starts on time after a silence."""
    n_rests = 0
    if rng.random() >= P_REST:
        return chords, 0
    want = 1 if rng.random() < 0.7 else 2
    out = [SM.PChord(c.spec, c.beats, c.passing) for c in chords]
    cand = [i for i in range(1, len(out) - 1) if out[i].beats >= 2.0 and out[i].spec is not None]
    rng.shuffle(cand)
    for i in sorted(cand[:want], reverse=True):
        c = out[i]
        r = float(min(c.beats - 1.0, rng.choice([1.0, 2.0, float(bpb)])))
        if r < 1.0:
            continue
        c.beats -= r
        out.insert(i + 1, SM.PChord(None, r, False))
        n_rests += 1
    return out, n_rests


def _sections(rng, arrangement: str, starts: list[float]) -> list[dict]:
    def off() -> dict:
        return {k: None for k in LAYERS} | {"piano": 0.0}
    secs = []
    for k in range(len(starts)):
        s = off()
        if arrangement == "soft":
            pad, voice = rng.random() < 0.6, rng.random() < 0.7
            if not (pad or voice):
                voice = True
            if pad:
                s["pad"] = round(float(rng.uniform(-20, -9)), 1)
            if voice:
                s["vocal"] = round(float(rng.uniform(-10, 0)), 1)
        elif arrangement == "band":
            if k == 0 and len(starts) > 1 and rng.random() < 0.5:
                secs.append(s)                       # a piano-only first section, then the band
                continue
            s["drums"] = round(float(rng.uniform(-16, -6)), 1)
            s["bass"] = round(float(rng.uniform(-14, -6)), 1)
            if rng.random() < 0.5:
                s["pad"] = round(float(rng.uniform(-18, -8)), 1)
            if rng.random() < 0.7:
                s["vocal"] = round(float(rng.uniform(-9, 2)), 1)
            s["piano"] = round(float(rng.uniform(-4, 0)), 1)
        secs.append(s)
    return secs


def plan_song(rng: np.random.Generator, split: str = "train", charts_ok: bool | None = None):
    """A song plan (``SM.SongPlan``; rest segments have ``spec None``)."""
    meter = str(rng.choice(list(METER_P), p=np.array(list(METER_P.values()))))
    bpb, sub = METERS[meter]
    bpm = float(rng.uniform(40, 75)) if meter == "12/8" else float(rng.uniform(55, 140))
    spb = 60.0 / bpm
    dur = float(rng.uniform(15, 20)) if rng.random() < 0.2 else float(rng.uniform(20, 30))
    lead = float(rng.uniform(*LEAD_S)) if rng.random() < P_LEAD else 0.0
    swing = float(rng.uniform(0.58, 0.67)) if (meter == "4/4" and bpm < 100 and rng.random() < 0.1) else 0.5
    ending = rng.random() < P_ENDING
    music_s = dur - lead - (float(rng.uniform(1.0, 3.0)) if ending else 0.0)
    total_beats = max(2.0, np.floor(music_s / spb * 2) / 2) if ending else np.ceil(music_s / spb) + 1
    if charts_ok is None:
        charts_ok = bool(SM.chart_files(split))
    probs = SOURCE_P if charts_ok else SOURCE_P_NO_CHARTS
    source = str(rng.choice(list(probs), p=np.array(list(probs.values()))))
    info, chords = make_progression(rng, source, bpb, float(total_beats), split)
    chords, n_rests = _insert_rests(chords, bpb, rng)
    segs, b = [], 0.0
    for c in chords:
        t0, t1 = SM._beat_time([b, b + c.beats], lead, spb, swing, sub)
        segs.append(SM.Seg(b, b + c.beats, float(t0), float(t1), c.spec, c.passing))
        b += c.beats
    while segs and segs[-1].spec is None:                  # never end on a rest
        segs.pop()
    end_t = float(segs[-1].t1) + 0.3
    if not ending:
        end_t = max(float(segs[-1].t1), dur + 0.5)
    arrangement = str(rng.choice(list(ARRANGEMENT_P), p=np.array(list(ARRANGEMENT_P.values()))))
    n_sec = int(rng.choice([1, 2], p=[0.6, 0.4]))
    bars = int((dur - lead) / spb // bpb)
    starts = [0.0]
    if n_sec == 2 and bars > 2:
        starts.append(float(SM._beat_time(int(rng.integers(1, bars)) * bpb, lead, spb, swing, sub)))
    sections = _sections(rng, arrangement, starts)
    info.update(meter=meter, bpm=round(bpm, 2), swing=round(swing, 3), duration=round(dur, 3), ending=bool(ending),
                lead=round(lead, 3), arrangement=arrangement, rests=n_rests,
                drums_in_rests=bool(rng.random() < 0.4))
    return SM.SongPlan(meter, bpm, lead, segs, end_t, list(zip(starts, sections)), info)


# ---------------------------------------------------------------------------------------- voicings

def _stack(pcs: list[int], start: int) -> list[int]:
    notes = [start]
    for pc in pcs[1:]:
        notes.append(notes[-1] + ((pc - notes[-1]) % 12 or 12))
    return notes


def _rh_pcs(spec, rng, lh_pcs: set[int]) -> list[int]:
    """Right-hand pitch classes (distinct): the required upper tones, optional ones about half the
    time, the bass tone usually left to the left hand."""
    upper = set(spec.upper)
    pcs = set(spec.required) & upper
    for o in sorted(spec.optional):
        if rng.random() < 0.55:
            pcs.add(o)
    if spec.inversion == "chord_tone" and spec.bass in pcs and len(pcs) > 3 and rng.random() < 0.7:
        pcs.discard(spec.bass)
    if spec.inversion == "root" and spec.root in lh_pcs and len(pcs) > 3 and rng.random() < 0.4:
        pcs.discard(spec.root)                                   # rootless right hand over the bass
    return sorted(pcs)


def _rh_shapes(spec, pcs: list[int], open_pos: bool) -> list[list[int]]:
    """Every inversion of the right hand (close, or open = 2nd note up an octave), at every octave
    inside RH_RANGE; with fewer than 3 tones the root (else the bottom note) is doubled on top."""
    out = []
    for rot in range(len(pcs)):
        order = pcs[rot:] + pcs[:rot]
        base = _stack(order, 36 + order[0])
        if len(base) < 3:
            dbl = spec.root if spec.root in pcs else order[0]
            base.append(base[-1] + ((dbl - base[-1]) % 12 or 12))
        if open_pos and len(base) >= 3:
            base[1] += 12
            base.sort()
        if len(set(base)) < len(base):
            continue
        for sh in range(0, 61, 12):
            notes = [m + sh for m in base]
            if notes[0] >= RH_RANGE[0] and notes[-1] <= RH_RANGE[1]:
                out.append(notes)
    return out


def voicing(spec, rng: np.random.Generator, layout: str = "lh_rh", target: float | None = None):
    """One ``train.chords.Voicing`` near middle C (see the module doc), or None. The right hand is
    drawn among all its inversions / octaves with weight exp(−|mean pitch − target| / 2)."""
    C = _C()
    target = float(rng.normal(*RH_MEAN)) if target is None else target
    if layout == "one_hand" and spec.inversion != "non_chord":
        lows = [m for m in range(ONE_HAND_BASS[0], ONE_HAND_BASS[1] + 1) if m % 12 == spec.bass]
        b = lows[int(rng.integers(len(lows)))]
        rest = [p for p in _rh_pcs(spec, rng, {spec.bass}) if p != spec.bass]
        rest = sorted(set(rest), key=lambda p: (p - spec.bass) % 12)
        notes = _stack([spec.bass] + rest, b)
        if len(notes) < 3 or notes[-1] - notes[0] > 14 or notes[-1] > RH_RANGE[1]:
            return None
        v = C.Voicing((), tuple(notes), "one_hand", "close")
        return v if not check_voicing(spec, v) else None
    pcs = _rh_pcs(spec, rng, {spec.bass})
    open_pos = len(pcs) >= 3 and rng.random() < 0.35
    shapes = [sh for sh in _rh_shapes(spec, pcs, open_pos)
              if any(LH_MIN <= m < sh[0] and m % 12 == spec.bass for m in range(LH_MIN, sh[0]))]
    if not shapes:
        return None
    w = np.exp(-np.abs(np.array([np.mean(sh) for sh in shapes]) - target) / 2.0)
    pref = np.array([any(m % 12 == spec.bass for m in range(LH_PREF[0], min(sh[0], LH_PREF[1] + 1)))
                     for sh in shapes])
    w = w * np.where(pref, 1.0, 0.25)                     # leave room for a C3–B3 bass when possible
    rh = tuple(shapes[int(rng.choice(len(shapes), p=w / w.sum()))])
    bass_cands = [m for m in range(LH_MIN, rh[0]) if m % 12 == spec.bass]
    w = np.array([1.0 if LH_PREF[0] <= m <= LH_PREF[1] else LH_LOW_W for m in bass_cands])
    b = bass_cands[int(rng.choice(len(bass_cands), p=w / w.sum()))]
    lh = [b]
    if layout == "two_hand":
        extra = [m for m in range(b + 3, min(rh[0], b + 17)) if m % 12 in spec.lh_tones and m % 12 != spec.bass]
        if extra:
            lh.append(extra[int(rng.integers(len(extra)))])
    v = C.Voicing(tuple(lh), rh, layout, "open" if open_pos else "close")
    return v if not check_voicing(spec, v) else None


def check_voicing(spec, v) -> list[str]:
    errs = []
    notes = v.notes
    if notes[0] < LH_MIN:
        errs.append("below G2")
    if notes[0] % 12 != spec.bass:
        errs.append("lowest note is not the bass")
    if not v.pcs <= spec.pcs:
        errs.append("foreign tones")
    if not (spec.required | {spec.bass}) <= v.pcs:
        errs.append("missing tones")
    if v.rh and (v.rh[0] < (RH_RANGE[0] if v.lh else ONE_HAND_BASS[0]) or v.rh[-1] > RH_RANGE[1]):
        errs.append("right hand out of range")
    if v.lh and v.rh and v.lh[-1] >= v.rh[0]:
        errs.append("hands overlap")
    if len({m % 12 for m in v.lh}) < len(v.lh):
        errs.append("left-hand octave doubling")
    if len(set(notes)) < len(notes):
        errs.append("duplicate notes")
    return errs


def voicings(spec, rng, n: int = 8, layout: str = "lh_rh") -> list:
    out, seen = [], set()
    for _ in range(6 * n):
        if len(out) >= n:
            break
        v = voicing(spec, rng, layout)
        if v is None and layout != "lh_rh":
            v = voicing(spec, rng, "lh_rh")
        if v is None or (v.lh, v.rh) in seen:
            continue
        seen.add((v.lh, v.rh))
        out.append(v)
    return out


def plan_piano(plan, rng):
    """Piano notes, pedal intervals, info and the voicing of every chord segment (None for rests)."""
    C = _C()
    spb = 60.0 / plan.bpm
    layout = str(rng.choice(["lh_rh", "two_hand", "one_hand"], p=[0.6, 0.2, 0.2]))
    sec_styles = [str(rng.choice(STYLES, p=STYLE_P)) for _ in plan.sections]
    pedal_on = rng.random() < 0.7
    late_lift = float(rng.uniform(0.0, 0.12)) if rng.random() < 0.35 else 0.0
    base_vel = float(rng.uniform(45, 90))
    notes, pedal, used = [], [], []
    prev_rh, prev_bass = (), None
    segs = plan.segs
    for i, s in enumerate(segs):
        if s.spec is None:
            used.append(None)
            continue
        k = max(j for j, (t, _) in enumerate(plan.sections) if t <= s.t0 + 1e-6)
        style = sec_styles[k]
        cands = voicings(s.spec, rng, 8, layout)
        if not cands:
            used.append(None)
            continue
        cost = np.array([SM._movement(prev_rh, v.rh) + (0.3 * abs(v.notes[0] - prev_bass) if prev_bass else 0.0)
                         for v in cands])
        w = np.exp(-(cost - cost.min()) / 2.0)
        v = cands[int(rng.choice(len(cands), p=w / w.sum()))]
        prev_rh, prev_bass = v.rh, v.notes[0]
        used.append(v)
        st = "block" if (s.passing or s.b1 - s.b0 <= 1.0) and style in ("arpeggio", "pulse", "bass_chord") else style
        vel = base_vel * (0.82 if s.passing else 1.0) + float(rng.normal(0, 4))
        notes += C.perform(v, st, s.t0, s.t1, rng, vel, spb, pedal_on)
        if pedal_on:
            rest_next = i + 1 < len(segs) and segs[i + 1].spec is None
            pedal.append((s.t0 + float(rng.uniform(0.02, 0.08)), s.t1 + (0.0 if rest_next else late_lift)))
    lh = [m for v in used if v for m in v.lh] or [m for v in used if v for m in v.rh[:1]]
    rh = [m for v in used if v for m in (v.rh if v.lh else v.rh[1:])]
    info = {"layout": layout, "styles": sec_styles, "pedal": bool(pedal_on), "late_lift": round(late_lift, 3),
            "register": {"bass_min": int(min(v.notes[0] for v in used if v)),
                         "bass_max": int(max(v.notes[0] for v in used if v)),
                         "lh_max": int(max(lh)) if lh else None,
                         "rh_min": int(min(rh)) if rh else None, "rh_max": int(max(rh)) if rh else None,
                         "rh_mean": round(float(np.mean(rh)), 2) if rh else None}}
    return notes, pedal, info, used


# ------------------------------------------------------------------------------------------- band

def _plan_bass(R, segs, p, rng):
    """``render._plan_bass`` with the bass kept in E2..G3 (40–55)."""
    sub = 1.0 / R._METERS[p.meter][1]
    notes, prev = [], 45
    for s in segs:
        prev = R._nearest(R.chord_tones(s.label)[2], prev, *SYNTH_BASS)
        if p.bass_pattern == "eighths" and not s.passing:
            b = s.b0
            while b < s.b1 - 1e-9:
                b1 = min(b + sub, s.b1)
                t0, t1 = R._beat_time([b, b1], p)
                notes.append(R.Note("bass", prev, float(t0) + R._jitter(rng, p), float(t1) - 0.03,
                                    0.85 if b == s.b0 else 0.65))
                b = b1
        else:
            notes.append(R.Note("bass", prev, s.t0 + R._jitter(rng, p), s.t1 - 0.02, 0.85))
    return notes


def _plan_pad(R, segs_all, end_t: float):
    """Pad held over the main chords (as ``render._plan_pad``) but stopped by rests."""
    spans: list[list] = []
    for s in segs_all:
        if s.spec is None:
            if spans and spans[-1][2] is None:
                spans[-1][2] = s.t0
            continue
        if s.passing:
            continue
        if spans and spans[-1][2] is None and spans[-1][0] == s.spec.label:
            continue
        if spans and spans[-1][2] is None:
            spans[-1][2] = s.t0
        spans.append([s.spec.label, s.t0, None])
    notes, prev = [], None
    for label, t0, t1 in spans:
        t1 = end_t if t1 is None else t1
        v = R._voicing(R._upper_structure(label), prev, 52, 76)
        prev = v
        notes += [R.Note("pad", m, t0, t1, 0.6) for m in v]
    return notes


def render_band(plan, n: int, rng_seed: int) -> tuple[dict, dict]:
    with SM._synth_rate(SR) as R:
        meter = {"4/4": "4/4", "3/4": "3/4", "12/8": "6/8"}[plan.meter]
        rng = lambda k: np.random.default_rng([rng_seed, k])  # noqa: E731
        used = {name for _, sec in plan.sections for name in LAYERS if sec.get(name) is not None}
        p = R.RenderParams(bpm=plan.bpm, meter=meter, swing=plan.info.get("swing", 0.5),
                           bass_pattern="eighths" if rng(0).random() < 0.3 else "hold",
                           humanize_ms=float(rng(0).uniform(2, 8)), lead_in=plan.lead, seed=rng_seed)
        chords = [R._Seg(s.b0, s.b1, s.spec.label, s.spec.symbol, s.passing, (), s.t0, s.t1)
                  for s in plan.segs if s.spec is not None]
        rests = [(s.t0, s.t1) for s in plan.segs if s.spec is None]
        total = plan.segs[-1].b1
        end_t = plan.segs[-1].t1
        buses = {k: np.zeros((2, n)) for k in LAYERS}
        info: dict = {}
        if "bass" in used:
            bn = _plan_bass(R, chords, p, rng(2))
            info["synth_bass"] = [min(x.midi for x in bn), max(x.midi for x in bn)] if bn else None
            for nt in bn:
                R._add(buses["bass"], R._bass_note(nt.midi, max(0.02, nt.t1 - nt.t0)), nt.t0, 0.0, nt.vel)
        if "pad" in used:
            for nt in _plan_pad(R, plan.segs, end_t):
                release = 1.2 if nt.t1 < end_t - 1e-6 else 0.25
                left, right = R._pad_note(nt.midi, max(0.02, nt.t1 - nt.t0), release)
                R._add(buses["pad"], left, nt.t0, -1.0, nt.vel * 0.707)
                R._add(buses["pad"], right, nt.t0, 1.0, nt.vel * 0.707)
        if "vocal" in used:
            vn = SM._vocal_notes(R._plan_vocal(chords, total, p, rng(3)), plan, rng(8))
            vn = [x for x in vn if not any(a < x.t1 and x.t0 < b for a, b in rests)]
            if vn:
                R._add(buses["vocal"], R._vocal(vn, n, rng(5)), 0.0, 0.0)
        if "drums" in used:
            kit = R._drum_kit(rng(6))
            pans = {"kick": 0.0, "snare": -0.1, "hat": 0.3}
            keep = plan.info.get("drums_in_rests", False)
            for kind, t0, vel in SM.plan_drums(plan, total, rng(4)):
                if not keep and any(a <= t0 < b for a, b in rests):
                    continue
                R._add(buses["drums"], kit[kind], t0, pans[kind], vel)
    return buses, info


def render_song(plan, seed: int, piano=None):
    """Mono float32 at 22.05 kHz + info (FT1's ``render_song`` with the v2 piano and band)."""
    from scipy.signal import fftconvolve, lfilter
    from train import piano as P
    rng = np.random.default_rng([seed, 99])
    piano = piano or SM.get_piano()
    dur = plan.info["duration"]
    n = int(round(dur * SR))
    notes, pedal, pinfo, used = plan_piano(plan, rng)
    tune = float(np.clip(rng.normal(0, 5), -15, 15)) if rng.random() < 0.4 else 0.0
    pn = piano.render(notes, pedal=pedal or None, duration=dur, tune_cents=tune, seed=int(rng.integers(2 ** 31)),
                      pedal_noise=bool(rng.random() < 0.5))
    pbus = pn[:n].T.astype(np.float64)
    if pbus.shape[1] < n:
        pbus = np.pad(pbus, ((0, 0), (0, n - pbus.shape[1])))
    bands, binfo = render_band(plan, n, int(rng.integers(2 ** 31)))
    buses = {"piano": pbus, **bands}
    i0 = int(plan.lead * SR)
    i1 = max(i0 + 1, min(n, int(plan.segs[-1].t1 * SR)))
    mix, send = np.zeros((2, n)), np.zeros(n)
    sends = {"piano": 1.0, "bass": 0.1, "pad": 1.0, "drums": 0.35, "vocal": 1.0}
    levels = []
    for name, bus in buses.items():
        rms = float(np.sqrt(np.mean(bus[:, i0:i1] ** 2))) if i1 > i0 else 0.0
        g = SM._gain_curve(plan, name, n, SR) if rms > 0 else None
        if g is None:
            continue
        bus = bus * (0.1 / rms) * g[None, :]
        mix += bus
        send += sends[name] * bus.mean(axis=0)
        levels.append(name)
    rt60 = float(rng.uniform(0.3, 2.5))
    wet = float(rng.uniform(0.05, 0.35)) if rng.random() < 0.85 else 0.0
    if wet > 0:
        with SM._synth_rate(SR) as R:
            ir = R._reverb_ir(rt60, rng)
        for ch in range(2):
            mix[ch] += wet * fftconvolve(send, ir[ch])[:n]
    mix, eq = SM._eq(mix, rng)
    noise_db = None
    if rng.random() < 0.4:
        noise_db = float(rng.uniform(-70, -40))
        white = rng.standard_normal((2, n))
        nz = white if rng.random() < 0.5 else lfilter([1.0], [1.0, -0.97], white, axis=1)
        nz /= np.sqrt(np.mean(nz ** 2)) + 1e-12
        mix += nz * (float(np.sqrt(np.mean(mix ** 2))) + 1e-12) * 10 ** (noise_db / 20)
    mix = mix / (float(np.max(np.abs(mix))) + 1e-12) * 10 ** (float(rng.uniform(-14, -1)) / 20)
    if rng.random() < 0.25:
        mix = P.limiter((mix * 10 ** (float(rng.uniform(2, 6)) / 20)).T.astype(np.float32), -1.0, sr=SR).T
    y = mix.mean(axis=0).astype(np.float32)
    info = {**pinfo, **binfo, "layers": sorted(k for k in levels if k != "piano"), "rt60": round(rt60, 2),
            "wet": round(wet, 3), "eq": eq, "noise_db": None if noise_db is None else round(noise_db, 1),
            "tune_cents": round(tune, 2), "sections": [[round(t, 3), s] for t, s in plan.sections]}
    return y, info, used


# ----------------------------------------------------------------------------------------- labels

REST_DECAY_S = 0.25


def frame_labels(plan, n_frames: int):
    """(root, bass, notes-bitmask) per ACE frame; rests, lead-in and the end are N. A chord before a
    rest keeps its label for ``REST_DECAY_S`` into the rest."""
    t = SM.frame_times(n_frames)
    root = np.full(n_frames, SM.N_CLASS, np.int8)
    bass = np.full(n_frames, SM.N_CLASS, np.int8)
    notes = np.zeros(n_frames, np.uint16)
    stop = min(plan.end_t, float(plan.info.get("duration", np.inf)))
    segs = plan.segs
    for i, s in enumerate(segs):
        if s.spec is None:
            continue
        if i + 1 == len(segs):
            t1 = plan.end_t
        elif segs[i + 1].spec is None:
            t1 = min(s.t1 + REST_DECAY_S, segs[i + 1].t1)
        else:
            t1 = s.t1
        m = (t >= s.t0) & (t < min(t1, stop))
        root[m] = s.spec.root
        bass[m] = s.spec.bass
        notes[m] = sum(1 << p for p in s.spec.pcs)
    return root, bass, notes


def make_item(i: int, seed: int, split: str = "train", piano=None, charts_ok: bool | None = None,
              with_audio: bool = False) -> dict:
    rng = np.random.default_rng([int(seed), int(i)])
    plan = plan_song(rng, split, charts_ok)
    y, rinfo, used = render_song(plan, int(rng.integers(2 ** 31)), piano)
    cqt, env = SM.features(y)
    F = cqt.shape[1]
    root, bass, notes = frame_labels(plan, F)
    chords = [[round(s.t0, 4), round(s.t1, 4), s.spec.label if s.spec else "N", bool(s.passing)] for s in plan.segs]
    meta = {"id": f"{split}-{seed}-{i}", "recipe": "ft2", **plan.info, **rinfo, "frames": int(F), "chords": chords,
            "end_t": round(plan.end_t, 4), "n_frames_N": int((root == SM.N_CLASS).sum())}
    out = {"cqt": cqt, "env": env, "root": root, "bass": bass, "notes": notes, "meta": meta}
    if with_audio:
        out["audio"], out["voicings"], out["plan"] = y, used, plan
    return out


# ------------------------------------------------------------------------------------------- pool

def write_shard(split: str, seed: int, shard: int, items: list[int], out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or POOL_DIR) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard-{seed}-{shard:04d}.npz"
    if path.exists():
        return {"path": str(path), "skipped": True}
    SM._init_worker(1)
    piano = SM.get_piano()
    charts_ok = bool(SM.chart_files(split))
    t = time.perf_counter()
    parts = [make_item(i, seed, split, piano, charts_ok) for i in items]
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
            "bytes": path.stat().st_size}


_LOWMEM_PIANO = None


def lowmem_piano():
    """The 22.05 kHz Salamander with smaller caches (≈ 0.5–0.8 GB per process instead of ≈ 1.6)."""
    global _LOWMEM_PIANO
    if _LOWMEM_PIANO is None:
        from train import piano as P
        _LOWMEM_PIANO = P.Piano(sr=SR, pitch_cache_mb=250.0, max_seconds=8.0)
    return _LOWMEM_PIANO


def render_items(split: str, seed: int, items: list[int]) -> list[dict]:
    """Items rendered in a worker process and returned (float16 CQT) — for an in-RAM pool."""
    SM._init_worker(1)
    piano = lowmem_piano()
    charts_ok = bool(SM.chart_files(split))
    out = []
    for i in items:
        it = make_item(i, seed, split, piano, charts_ok)
        it["cqt"] = it["cqt"].astype(np.float16)
        it["env"] = it["env"].astype(np.float16)
        out.append(it)
    return out


def disk_free(path: Path = ROOT) -> int:
    import shutil
    return shutil.disk_usage(path).free


def build(split: str, n_items: int, workers: int = 6, shard_size: int = 100, start: int = 0,
          out_dir: Path | None = None, seed: int | None = None) -> list[dict]:
    """Render items [start, start + n_items) into shards; stops at the 8 GB pool cap or when C:
    would drop below 20 GB free."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    seed = SEEDS[split] if seed is None else seed
    out_dir = Path(out_dir or POOL_DIR)
    jobs = []
    for k in range(start // shard_size, int(np.ceil((start + n_items) / shard_size))):
        items = [i for i in range(k * shard_size, (k + 1) * shard_size) if start <= i < start + n_items]
        if items:
            jobs.append((k, items))
    res, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=SM._init_worker) as ex:
        futs = {ex.submit(write_shard, split, seed, k, items, out_dir): k for k, items in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            res.append(r)
            size, free = SM.pool_bytes(out_dir), disk_free()
            print(f"[{split} {n}/{len(jobs)}] {Path(r['path']).name} {r.get('seconds', 'skip')} s, "
                  f"pool {size / 1e9:.2f} GB, C: free {free / 1e9:.1f} GB, {time.perf_counter() - t0:.0f} s",
                  flush=True)
            if size > POOL_CAP_BYTES or free < MIN_FREE_BYTES:
                print("pool cap / disk floor reached: stopping", flush=True)
                for g in futs:
                    g.cancel()
                break
    return res


# ----------------------------------------------------------------------------------------- stats

def stats(root: Path | None = None) -> dict:
    """Composition of a v2 pool on disk (see ``stats_metas``)."""
    out = {}
    for split in ("train", "val"):
        files = SM.shard_files(split, root or POOL_DIR)
        if not files:
            continue
        metas = []
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                metas += json.loads(str(z["meta"]))
        out[split] = stats_metas(metas)
        out[split]["bytes"] = sum(f.stat().st_size for f in files)
    return out


def stats_metas(metas: list[dict]) -> dict:
    """Chord qualities (plain share), slash chords by bass interval, sources, meters, arrangements,
    N share, rests, and the MIDI ranges actually played ([min, 5 %, median, 95 %, max] over items)."""
    from nathan_chordify.theory import harte as H
    C = _C()
    c = {k: Counter() for k in ("source", "meter", "arrangement", "layout", "quality", "slash", "style")}
    n = frames = n_frames_n = chords = slash = rests = 0
    secs = 0.0
    reg = {"bass_min": [], "bass_max": [], "lh_max": [], "rh_min": [], "rh_max": [], "rh_mean": [],
           "synth_bass_min": [], "synth_bass_max": []}
    for m in metas:
        n += 1
        frames += m["frames"]
        n_frames_n += m.get("n_frames_N", 0)
        secs += m["duration"]
        rests += m.get("rests", 0)
        for k in ("source", "meter", "arrangement", "layout"):
            c[k][m[k]] += 1
        for st in m["styles"]:
            c["style"][st] += 1
        for k, v in m["register"].items():
            if v is not None:
                reg[k].append(v)
        if m.get("synth_bass"):
            reg["synth_bass_min"].append(m["synth_bass"][0])
            reg["synth_bass_max"].append(m["synth_bass"][1])
        for _, _, lab, _ in m["chords"]:
            if lab == "N":
                continue
            r, q, b = H.parse_harte(lab)
            spec = C.lookup(r, q, b)
            c["quality"][q] += 1
            chords += 1
            if spec.bass != spec.root:
                slash += 1
                c["slash"][f"bass +{spec.bass_interval} ({spec.inversion})"] += 1
    q = c["quality"]
    return {
        "items": n, "hours": round(secs / 3600, 2), "frames": frames, "N_frame_share": round(n_frames_n / frames, 4),
        "rests": rests, "chords": chords, "plain_share": round((q["maj"] + q["min"]) / max(chords, 1), 4),
        "slash_share": round(slash / max(chords, 1), 4),
        "quality_share": {k: round(v / max(chords, 1), 4) for k, v in q.most_common()},
        "outside_allowed": sorted(set(q) - set(ALLOWED)),
        "register": {k: ([int(min(v)), int(np.percentile(v, 5)), int(np.median(v)), int(np.percentile(v, 95)),
                          int(max(v))] if v else None) for k, v in reg.items()},
        **{k: dict(v.most_common()) for k, v in c.items() if k != "quality"},
    }


# --------------------------------------------------------------------------------------- preview

def _note(m: int) -> str:
    return f"{SM.NAMES[m % 12]}{m // 12 - 1}"


def write_preview(out_dir: Path = PREVIEW_DIR, n: int = 20, seed: int = 5150) -> list[dict]:
    """``n`` short FLAC files + labels.csv (one row per chord or rest) for listening."""
    import soundfile as sf
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    piano = SM.get_piano()
    rows, infos = [], []
    for i in range(n):
        it = make_item(i, seed, "val", piano, with_audio=True)
        m = it["meta"]
        name = f"{i + 1:02d}_{m['arrangement']}_{m['meter'].replace('/', '-')}_{int(round(m['bpm']))}bpm_{m['source']}.flac"
        sf.write(out_dir / name, it["audio"], SR, subtype="PCM_16", format="FLAC")
        for j, (s, v) in enumerate(zip(it["plan"].segs, it["voicings"])):
            spec = s.spec
            rows.append({
                "file": name, "segment": j + 1, "start_s": f"{s.t0:.3f}", "end_s": f"{s.t1:.3f}",
                "chord": spec.symbol if spec else "N (rest)", "harte": spec.label if spec else "N",
                "quality": spec.quality if spec else "", "bass": SM.NAMES[spec.bass] if spec else "",
                "inversion": spec.inversion if spec else "", "passing": "yes" if s.passing else "",
                "left_hand": " ".join(_note(x) for x in v.lh) if v else "",
                "right_hand": " ".join(_note(x) for x in v.rh) if v else "",
                "arrangement": m["arrangement"], "layers": "+".join(m["layers"]) or "piano only",
                "meter": m["meter"], "bpm": m["bpm"], "source": m["source"],
            })
        infos.append({"file": name, **m})
        print(f"[{i + 1}/{n}] {name}  {len(m['chords'])} segments, register {m['register']}", flush=True)
    with open(out_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "labels.json").write_text(json.dumps(infos, indent=1, ensure_ascii=False), "utf-8")
    return infos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m train.song_mix_v2", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", choices=list(SEEDS), required=True)
    b.add_argument("--items", type=int, required=True)
    b.add_argument("--start", type=int, default=0)
    b.add_argument("--workers", type=int, default=6)
    b.add_argument("--shard-size", type=int, default=100)
    sub.add_parser("stats")
    pv = sub.add_parser("preview")
    pv.add_argument("--n", type=int, default=20)
    pv.add_argument("--out", default=str(PREVIEW_DIR))
    a = ap.parse_args(argv)
    from train import song_mix_v2 as V2          # pickled by module path for the worker processes
    if a.cmd == "build":
        V2.build(a.split, a.items, a.workers, a.shard_size, a.start)
    elif a.cmd == "stats":
        st = V2.stats()
        st["total_bytes"] = SM.pool_bytes(POOL_DIR)
        (POOL_DIR / "pool.json").write_text(json.dumps(st, indent=1, ensure_ascii=False), "utf-8")
        print(json.dumps(st, indent=1, ensure_ascii=False))
    elif a.cmd == "preview":
        V2.write_preview(Path(a.out), a.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
