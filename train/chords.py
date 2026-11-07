"""Generated piano chords with exact labels (restart idea 2).

Every chord of the engine vocabulary (``theory.vocab``: 12 roots × 32 qualities × the allowed
basses = 1,668 chords, 1,188 sound-alike classes) is voiced the way a pianist would play it, across
the keyboard, and rendered with the Salamander Grand Piano (``train.piano``). Labels are exact
because we choose the notes.

Voicings (``voicings(chord, rng, n)``), every one checked by ``check_voicing``:

- **layouts**: ``lh_rh`` (left hand = the bass note, alone or doubled at the octave / with a 5th /
  a 10th, MIDI 28–55; right hand = 3–5 chord tones, MIDI 48–88), ``two_hand_block`` (a fuller left
  hand of 2–3 notes plus a right-hand chord sitting just above it: the "big hymn chord") and
  ``one_hand`` (only the right hand; its lowest note is the bass, so inversions still show);
- **right-hand position**: ``close`` (span ≤ 12 semitones), ``open`` (≤ 19: open triads, drop-2)
  or ``spread`` (≤ 24); the bottom/top tone (the right-hand inversion) varies, the optional 5th is
  left out sometimes, the root is doubled sometimes, and over a root bass the right hand may be
  rootless;
- every note stays in A0–C8 (21–108); the lowest note is always the bass; the notes use only the
  chord's tones (+ the bass) and contain every required tone.

Performance styles (``render_example(chord, voicing, style, rng)``): ``block`` (struck together),
``rolled`` (a quick spread from the bottom), ``arpeggio`` (broken chord: up / down / up-down /
bass-then-up / Alberti, with pedal), ``pulse`` (left hand held, right hand re-struck on every beat or
eighth) and ``bass_chord`` (bass on the beat, chord after it: "boom-chuck"). Velocities 30–110, the
sustain pedal sometimes, a little detuning (± cents), sometimes a room reverb, a random level.

Progressions (``render_progression(chords, bpm, rng)``): 4–8 chords from the worship idioms of
``eval/synth/render.py`` (augmented passing, borrowed iv, descending canon, walk-ups, rising m7,
passing °7, secondary dominants, sus resolutions, IV/5, pedal neighbours, royal road, 回家's
pre-chorus) plus ``PROGRESSIONS`` below (I–V/7–vi–IV, line cliché, 2-5-1 with extensions, ...), in
any key, voice-led from chord to chord, with legato pedalling; short passing chords are struck as
blocks.

``iter_dataset(n, seed)`` renders examples on the fly (deterministic by seed: example i uses
``np.random.default_rng([seed, i])``), so nothing big is stored on disk.

CLI::

    python -m train.chords --stats
    python -m train.chords --preview 60 --out data\\train\\preview\\
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np

from nathan_chordify.theory import harte as H
from nathan_chordify.theory import spelling as S
from nathan_chordify.theory import templates as T
from nathan_chordify.theory import vocab as V

from . import piano as P

SR = P.SR
KEY_RANGE = (21, 108)          # A0..C8
LH_RANGE = (28, 55)            # E1..G3
RH_RANGE = (48, 88)            # C3..E6
VEL_RANGE = (30, 110)
RH_SIZES = (3, 4, 5)
SPAN = {"close": 12, "open": 19, "spread": 24}
LH_MAX_SPAN = 16
ONE_HAND_LOW = (48, 64)        # lowest note of a one-hand voicing
LAYOUTS = ("lh_rh", "two_hand_block", "one_hand")
POSITIONS = ("close", "open", "spread")
STYLES = ("block", "rolled", "arpeggio", "pulse", "bass_chord")
NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def note_name(m: int) -> str:
    return f"{NAMES[m % 12]}{m // 12 - 1}"


def spelled_note(c: ChordSpec, m: int) -> str:
    """A note named as a tone of chord ``c`` (Db in Eb7, C# in A7), with its octave number."""
    root = S.parse_note(H.parse_harte_full(c.label).root_name)
    name = S.name_pc(root, c.quality, m % 12)
    octave = m // 12 - 1
    if name.startswith("Cb"):
        octave += 1
    elif name.startswith("B#"):
        octave -= 1
    return f"{name}{octave}"


# ------------------------------------------------------------------------------------------ chords

@dataclass(frozen=True)
class ChordSpec:
    root: int                  # pc 0..11
    quality: str               # one of templates.QUALITIES
    bass: int                  # pc 0..11
    label: str                 # canonical Harte ("C#:min7/b3")
    pcs: frozenset[int]        # template tones ∪ bass
    cls: int                   # sound-alike class (theory.vocab)
    index: int                 # index in vocab.candidates().cands

    @property
    def bass_interval(self) -> int:
        return (self.bass - self.root) % 12

    @property
    def inversion(self) -> str:
        """"root", "chord_tone" (an inversion: E/G#) or "non_chord" (a foreign bass: Em/A)."""
        if self.bass == self.root:
            return "root"
        return "chord_tone" if self.bass_interval in T.tones(self.quality) else "non_chord"

    @property
    def symbol(self) -> str:
        return H.harte_to_symbol(self.label)

    @property
    def required(self) -> frozenset[int]:
        return frozenset((self.root + i) % 12 for i in T.required(self.quality))

    @property
    def optional(self) -> frozenset[int]:
        return frozenset((self.root + i) % 12 for i in T.optional(self.quality))

    @property
    def upper(self) -> frozenset[int]:
        """Template tones (what the right hand may play)."""
        return frozenset((self.root + i) % 12 for i in T.tones(self.quality))

    @property
    def lh_tones(self) -> frozenset[int]:
        """Tones the left hand may add above the bass: chord tones that are not 9/11/13ths."""
        return frozenset((self.root + i) % 12 for i in T.chord_bass_intervals(self.quality))

    def transpose(self, k: int) -> ChordSpec:
        return lookup(self.root + k, self.quality, self.bass + k)

    def as_dict(self) -> dict:
        return {"label": self.label, "symbol": self.symbol, "root": self.root, "quality": self.quality,
                "bass": self.bass, "bass_interval": self.bass_interval, "inversion": self.inversion,
                "pcs": sorted(self.pcs), "class": self.cls}


@lru_cache(maxsize=1)
def enumerate_chords() -> tuple[ChordSpec, ...]:
    """Every chord of the vocabulary (no N): 12 roots × 32 qualities × allowed basses = 1,668."""
    v = V.candidates()
    out = []
    for i, c in enumerate(v.cands):
        if c.quality == T.N:
            continue
        out.append(ChordSpec(c.root, c.quality, c.bass, H.to_harte(c.root, c.quality, c.bass),
                             frozenset(c.pcs), c.cls, i))
    return tuple(out)


@lru_cache(maxsize=1)
def _by_triple() -> dict[tuple[int, str, int], ChordSpec]:
    return {(c.root, c.quality, c.bass): c for c in enumerate_chords()}


def lookup(root: int, quality: str, bass: int) -> ChordSpec:
    try:
        return _by_triple()[(root % 12, quality, bass % 12)]
    except KeyError:
        raise KeyError(f"({root}, {quality}, {bass}) is not in the vocabulary") from None


def chord(text: str) -> ChordSpec:
    """A chord from a Harte label ("C:maj7/3") or a typed symbol ("Cmaj7/E")."""
    p = H.parse_any(text)
    if p.quality == T.N:
        raise ValueError("N (no chord) has no voicings")
    return lookup(p.root, p.quality, p.bass)


def equivalents(c: ChordSpec) -> list[str]:
    """Other labels that sound the same (same notes and bass): C6 ~ Am7/C."""
    return [H.to_harte(*t) for t in V.equivalents(c.root, c.quality, c.bass)]


def sample_chord(rng: np.random.Generator, mode: str = "balanced") -> ChordSpec:
    """``"uniform"``: every one of the 1,668 chords equally; ``"balanced"``: root and quality
    uniform, then root position half the time and the other allowed basses share the rest."""
    chords = enumerate_chords()
    if mode == "uniform":
        return chords[int(rng.integers(len(chords)))]
    root = int(rng.integers(12))
    q = T.QUALITIES[int(rng.integers(len(T.QUALITIES)))]
    ivs = T.allowed_bass_intervals(q)
    iv = 0 if (len(ivs) == 1 or rng.random() < 0.5) else int(rng.choice(ivs[1:]))
    return lookup(root, q, root + iv)


# ---------------------------------------------------------------------------------------- voicings

@dataclass(frozen=True)
class Voicing:
    lh: tuple[int, ...]        # MIDI, ascending (empty for one_hand)
    rh: tuple[int, ...]        # MIDI, ascending
    layout: str
    position: str

    @property
    def notes(self) -> tuple[int, ...]:
        return tuple(sorted(self.lh + self.rh))

    @property
    def pcs(self) -> frozenset[int]:
        return frozenset(m % 12 for m in self.notes)

    def as_dict(self) -> dict:
        return {"lh": list(self.lh), "rh": list(self.rh), "layout": self.layout, "position": self.position,
                "names": [note_name(m) for m in self.notes]}


def rh_position(rh: Sequence[int]) -> str | None:
    span = rh[-1] - rh[0]
    for pos in POSITIONS:
        if span <= SPAN[pos]:
            return pos
    return None


def _lh_ok(c: ChordSpec, lh: Sequence[int]) -> bool:
    if not lh:
        return True
    if lh[0] % 12 != c.bass or not (LH_RANGE[0] <= lh[0] and lh[-1] <= LH_RANGE[1]):
        return False
    if len(lh) > 3 or lh[-1] - lh[0] > LH_MAX_SPAN or len(set(lh)) < len(lh):
        return False
    allowed = c.lh_tones | {c.bass}
    if any(m % 12 not in allowed for m in lh[1:]):
        return False
    for a, b in zip(lh, lh[1:]):                       # no muddy close intervals down low
        if b - a < 7 and a < 45:
            return False
        if b - a < 3:
            return False
    return True


def check_voicing(c: ChordSpec, v: Voicing) -> list[str]:
    """Problems with a voicing (empty list = valid)."""
    errs = []
    notes = v.notes
    if not notes:
        return ["no notes"]
    if notes[0] < KEY_RANGE[0] or notes[-1] > KEY_RANGE[1]:
        errs.append("outside A0–C8")
    if len(set(notes)) < len(notes):
        errs.append("duplicate notes")
    if notes[0] % 12 != c.bass:
        errs.append("lowest note is not the bass")
    if not v.pcs <= c.pcs:
        errs.append(f"foreign tones {sorted(v.pcs - c.pcs)}")
    if not (c.required | {c.bass}) <= v.pcs:
        errs.append(f"missing tones {sorted((c.required | {c.bass}) - v.pcs)}")
    rh = v.rh
    if not RH_SIZES[0] <= len(rh) <= RH_SIZES[-1]:
        errs.append("right hand needs 3–5 notes")
    elif tuple(sorted(rh)) != rh:
        errs.append("right hand not sorted")
    if rh and any(m % 12 not in c.upper for m in rh):
        errs.append("right hand plays a non-chord bass tone")
    if v.layout == "one_hand":
        if v.lh:
            errs.append("one_hand has a left hand")
        if rh and not (ONE_HAND_LOW[0] <= rh[0] <= ONE_HAND_LOW[1] and rh[-1] <= RH_RANGE[1]):
            errs.append("one_hand out of range")
        if rh and rh[-1] - rh[0] > SPAN["close"]:
            errs.append("one_hand span > 12")
        if c.inversion == "non_chord":
            errs.append("one_hand cannot play a non-chord bass")
    else:
        if not v.lh:
            errs.append("missing left hand")
        elif not _lh_ok(c, v.lh):
            errs.append("bad left hand")
        if rh and (rh[0] < RH_RANGE[0] or rh[-1] > RH_RANGE[1]):
            errs.append("right hand out of 48–88")
        if rh and v.lh and rh[0] <= v.lh[-1]:
            errs.append("hands overlap")
    if rh and rh_position(rh) is None:
        errs.append("right hand span > 24")
    elif rh and v.position in SPAN and rh[-1] - rh[0] > SPAN[v.position]:
        errs.append("span too wide for the position")
    return errs


@lru_cache(maxsize=4096)
def lh_candidates(c: ChordSpec) -> tuple[tuple[tuple[int, ...], str], ...]:
    """Every allowed left-hand note set, with its pattern name: single, octave, fifth,
    fifth_octave, tenth, or chord (bass + chord tones, for two_hand_block)."""
    out = []
    lo, hi = LH_RANGE
    for b in range(lo, hi + 1):
        if b % 12 != c.bass:
            continue
        above = [m for m in range(b + 1, min(hi, b + LH_MAX_SPAN) + 1) if m % 12 in (c.lh_tones | {c.bass})]
        for k in (0, 1, 2):
            for extra in combinations(above, k):
                lh = (b, *extra)
                if not _lh_ok(c, lh):
                    continue
                ivs = tuple(m - b for m in extra)
                kind = {(): "single", (12,): "octave", (7,): "fifth", (7, 12): "fifth_octave",
                        (15,): "tenth", (16,): "tenth"}.get(ivs, "chord")
                out.append((lh, kind))
    return tuple(out)


_LH_WEIGHTS = {
    "lh_rh": {"single": 0.35, "octave": 0.35, "fifth": 0.15, "fifth_octave": 0.1, "tenth": 0.05},
    "two_hand_block": {"octave": 0.25, "fifth_octave": 0.3, "chord": 0.45, "fifth": 0.0},
}


def _pick_lh(c: ChordSpec, layout: str, position: str, rng) -> tuple[int, ...] | None:
    cands = lh_candidates(c)
    weights = dict(_LH_WEIGHTS[layout])
    if layout == "lh_rh" and position != "spread":
        weights["tenth"] = 0.0
    kinds = [k for k in weights if weights[k] > 0 and any(kk == k for _, kk in cands)]
    if not kinds:
        return None
    w = np.array([weights[k] for k in kinds])
    kind = kinds[int(rng.choice(len(kinds), p=w / w.sum()))]
    opts = [lh for lh, kk in cands if kk == kind]
    # registers: the bass mostly around F2 (41), wider for two-hand blocks
    centre, sd = (41.0, 6.0) if layout == "lh_rh" else (40.0, 5.0)
    pw = np.array([math.exp(-0.5 * ((lh[0] - centre) / sd) ** 2) + 1e-3 for lh in opts])
    return opts[int(rng.choice(len(opts), p=pw / pw.sum()))]


def _rh_pcs(c: ChordSpec, lh: tuple[int, ...], rng) -> list[int] | None:
    """Distinct right-hand pitch classes (before doubling)."""
    lh_pcs = {m % 12 for m in lh}
    pcs = set(c.required)
    if c.inversion == "root":
        if lh and rng.random() < 0.5 and len(c.upper - {c.root}) >= 2:
            pcs.discard(c.root)                             # rootless right hand over the bass
    elif c.inversion == "chord_tone":
        if lh and rng.random() < 0.75:
            pcs.discard(c.bass)                             # the bass is in the left hand
    for o in sorted(c.optional):
        p_keep = 0.6 if (o - c.root) % 12 == 7 else 0.5
        if o not in pcs and rng.random() < p_keep:
            pcs.add(o)
    need = set(c.required) - lh_pcs if lh else set(c.required) | {c.bass}
    pcs |= need
    drop_order = sorted(c.optional & pcs) + sorted((pcs & lh_pcs) - need)
    while len(pcs) > RH_SIZES[-1] and drop_order:
        pcs.discard(drop_order.pop(0))
    if len(pcs) > RH_SIZES[-1]:
        return None
    return sorted(pcs)


def _stack(order: list[int], start: int) -> list[int]:
    notes = [start]
    for pc in order[1:]:
        notes.append(notes[-1] + ((pc - notes[-1]) % 12 or 12))
    return notes


def _arrange(c: ChordSpec, pcs: list[int], position: str, rng, bottom: int | None = None) -> list[int] | None:
    """Right-hand notes (relative register, fixed later) in close / open / spread position."""
    order = list(pcs)
    rng.shuffle(order)
    if bottom is not None:
        order = [bottom] + sorted((p for p in order if p != bottom), key=lambda p: (p - bottom) % 12)
    else:
        b0 = order[0]
        order = [b0] + sorted(order[1:], key=lambda p: (p - b0) % 12)   # a rotation = RH inversion
    notes = _stack(order, 60 + order[0])
    want = int(rng.choice(RH_SIZES, p=[0.45, 0.4, 0.15]))
    want = min(max(want, len(notes), 3), RH_SIZES[-1])
    # doubling (each tone at most once): the root first (60 %), else a random tone, placed where
    # it widens the hand least (above the bass when the bottom note is fixed)
    doubled: set[int] = set()
    while len(notes) < want:
        free = [p for p in pcs if p not in doubled]
        if not free:
            return None
        dbl = c.root if (c.root in free and rng.random() < 0.6) else int(rng.choice(free))
        doubled.add(dbl)
        lo = notes[0] + 1 if bottom is not None else notes[0] - 12
        opts = [m for m in range(lo, notes[-1] + 13) if m % 12 == dbl and m not in notes]
        if not opts:
            return None
        notes.append(min(opts, key=lambda m: (max(notes[-1], m) - min(notes[0], m), -m)))
        notes.sort()
    if position == "open" and len(notes) >= 3:
        if len(notes) == 3:
            notes[1] += 12                                   # open triad: C G E
        else:
            notes[-2] -= 12                                  # drop-2
        notes.sort()
    elif position == "spread" and len(notes) >= 3:
        k = len(notes)
        for i in range(1, k):
            if rng.random() < 0.5:
                notes[i] += 12
        notes.sort()
    if len(set(notes)) < len(notes):
        return None
    if bottom is not None and notes[0] % 12 != bottom:
        return None
    return notes


def _place(notes: list[int], lo: int, hi: int, rng, target_top: float, sd: float = 6.0) -> tuple[int, ...] | None:
    shifts = [s for s in range(-60, 61, 12) if notes[0] + s >= lo and notes[-1] + s <= hi]
    if not shifts:
        return None
    w = np.array([math.exp(-0.5 * ((notes[-1] + s - target_top) / sd) ** 2) + 1e-4 for s in shifts])
    s = shifts[int(rng.choice(len(shifts), p=w / w.sum()))]
    return tuple(m + s for m in notes)


def _one_voicing(c: ChordSpec, layout: str, position: str, rng) -> Voicing | None:
    if layout == "one_hand":
        if c.inversion == "non_chord":
            return None
        pcs = sorted(set(c.required) | {c.bass} | {o for o in c.optional if rng.random() < 0.5})
        if len(pcs) > RH_SIZES[-1]:
            pcs = sorted(set(c.required) | {c.bass})
        notes = _arrange(c, pcs, "close", rng, bottom=c.bass)
        if notes is None:
            return None
        rh = _place(notes, ONE_HAND_LOW[0], RH_RANGE[1], rng, target_top=float(rng.uniform(62, 72)))
        if rh is None or rh[0] > ONE_HAND_LOW[1]:
            return None
        return Voicing((), rh, layout, "close")
    lh = _pick_lh(c, layout, position, rng)
    if lh is None:
        return None
    pcs = _rh_pcs(c, lh, rng)
    if pcs is None:
        return None
    notes = _arrange(c, pcs, position, rng)
    if notes is None:
        return None
    lo = max(RH_RANGE[0], lh[-1] + 1)
    if layout == "two_hand_block":
        target = lh[-1] + 3 + (notes[-1] - notes[0]) + float(rng.uniform(0, 6))
        rh = _place(notes, lo, RH_RANGE[1], rng, target_top=target, sd=4.0)
    else:
        rh = _place(notes, lo, RH_RANGE[1], rng, target_top=float(rng.normal(74, 6)), sd=7.0)
    if rh is None:
        return None
    pos = rh_position(rh)
    if pos is None:
        return None
    return Voicing(lh, rh, layout, pos)


_LAYOUT_P = {"lh_rh": 0.6, "two_hand_block": 0.25, "one_hand": 0.15}
_POSITION_P = {"close": 0.55, "open": 0.3, "spread": 0.15}


def voicings(c: ChordSpec, rng: np.random.Generator, n: int = 1, layout: str | None = None,
             position: str | None = None, max_tries: int | None = None) -> list[Voicing]:
    """Up to ``n`` distinct valid voicings of a chord (random layouts/positions unless given)."""
    out: list[Voicing] = []
    seen = set()
    tries = max_tries or 40 * n + 40
    layouts = [layout] if layout else [k for k in LAYOUTS if not (k == "one_hand" and c.inversion == "non_chord")]
    lp = np.array([_LAYOUT_P[k] for k in layouts])
    positions = [position] if position else list(POSITIONS)
    pp = np.array([_POSITION_P[k] for k in positions])
    for _ in range(tries):
        if len(out) >= n:
            break
        lay = layouts[int(rng.choice(len(layouts), p=lp / lp.sum()))]
        pos = positions[int(rng.choice(len(positions), p=pp / pp.sum()))]
        v = _one_voicing(c, lay, pos, rng)
        if v is None or check_voicing(c, v):
            continue
        if position and v.position != position:
            continue
        key = (v.lh, v.rh)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


# --------------------------------------------------------------------------------- voicing counts

@lru_cache(maxsize=None)
def _combos(n: int, k: int) -> np.ndarray:
    return np.array(list(combinations(range(n), k)), dtype=np.int16).reshape(-1, k)


def count_voicings(c: ChordSpec) -> dict[str, int]:
    """How many distinct note sets the voicing rules allow for one chord (by layout).

    Two-hand sets: every left-hand candidate × every right-hand set of 3–5 notes in 48–88, above
    the left hand, span ≤ 24, using template tones and covering the required tones the left
    hand does not play. One-hand sets: 3–5 notes, lowest = the bass (48–64), span ≤ 12."""
    out = {"two_hand": 0, "one_hand": 0}
    notes = np.array([m for m in range(RH_RANGE[0], RH_RANGE[1] + 1) if m % 12 in c.upper], dtype=np.int32)
    bits = (1 << (notes % 12)).astype(np.int32)
    n = len(notes)
    tables = []
    for k in RH_SIZES:
        if n < k:
            continue
        idx = _combos(n, k)
        sel = notes[idx]
        mask = np.bitwise_or.reduce(bits[idx], axis=1)
        tables.append((sel[:, 0], sel[:, -1] - sel[:, 0], mask))
    lows = np.concatenate([t[0] for t in tables])
    spans = np.concatenate([t[1] for t in tables])
    masks = np.concatenate([t[2] for t in tables])
    req_all = sum(1 << p for p in c.required)
    sorted_lows: dict[int, np.ndarray] = {}
    for lh, _ in lh_candidates(c):
        need = req_all & ~sum(1 << (m % 12) for m in lh)
        if need not in sorted_lows:
            ok = (spans <= SPAN["spread"]) & ((masks & need) == need)
            sorted_lows[need] = np.sort(lows[ok])
        sl = sorted_lows[need]
        out["two_hand"] += int(len(sl) - np.searchsorted(sl, lh[-1], side="right"))
    if c.inversion != "non_chord":
        need = req_all | (1 << c.bass)
        ok = ((lows % 12) == c.bass) & (lows <= ONE_HAND_LOW[1]) & (spans <= SPAN["close"]) & ((masks & need) == need)
        out["one_hand"] = int(ok.sum())
    out["total"] = out["two_hand"] + out["one_hand"]
    return out


def voicing_stats(chords: Sequence[ChordSpec] | None = None) -> dict:
    chords = list(chords or enumerate_chords())
    per_q: Counter = Counter()
    total = Counter()
    for c in chords:
        cnt = count_voicings(c)
        per_q[c.quality] += cnt["total"]
        total.update(cnt)
    return {"chords": len(chords), "total": total["total"], "two_hand": total["two_hand"],
            "one_hand": total["one_hand"], "per_quality": dict(per_q),
            "lh_patterns": sum(len(lh_candidates(c)) for c in chords)}


# ------------------------------------------------------------------------------------ performance

def _velocities(v: Voicing, rng, base: float) -> dict[int, int]:
    out = {}
    top = v.rh[-1] if v.rh else None
    bass = v.notes[0]
    for m in v.notes:
        x = base + rng.normal(0, 5)
        if m == top:
            x += 6                                         # the melody note sings out a little
        if m == bass:
            x += 4
        out[m] = int(np.clip(round(x), *VEL_RANGE))
    return out


def _clamp_v(x: float) -> int:
    return int(np.clip(round(x), *VEL_RANGE))


def perform(v: Voicing, style: str, t0: float, t1: float, rng, base_vel: float, beat: float = 0.75,
            pedal: bool = False) -> list[P.PianoNote]:
    """Notes that play voicing ``v`` from t0 to t1 (key-up at t1) in a performance style."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    vel = _velocities(v, rng, base_vel)
    dur = t1 - t0
    notes: list[P.PianoNote] = []
    jit = lambda s=0.004: float(np.clip(rng.normal(0, s), -2.5 * s, 2.5 * s))  # noqa: E731
    if style == "block" or (style != "rolled" and dur < 0.45):
        lead = float(rng.uniform(0.0, 0.02)) if v.lh else 0.0     # the left hand often lands first
        for m in v.notes:
            on = t0 + max(0.0, jit() + (0.0 if m in v.lh else lead))
            notes.append(P.PianoNote(m, vel[m], on, max(0.05, t1 - on)))
    elif style == "rolled":
        gap = float(rng.uniform(0.012, 0.045))
        for i, m in enumerate(v.notes):
            on = t0 + i * gap + max(0.0, jit(0.002))
            notes.append(P.PianoNote(m, vel[m], on, max(0.05, t1 - on)))
    elif style == "arpeggio":
        seq_kind = str(rng.choice(["up", "down", "updown", "bass_up", "alberti"]))
        ns = list(v.notes)
        if seq_kind in ("bass_up", "alberti") and v.lh:
            for m in v.lh:                                     # the left hand holds the bass
                notes.append(P.PianoNote(m, vel[m], t0 + max(0.0, jit()), max(0.05, dur)))
            ns = list(v.rh)
        if seq_kind == "down":
            seq = ns[::-1]
        elif seq_kind == "updown":
            seq = ns + ns[-2:0:-1]
        elif seq_kind == "alberti" and len(ns) == 3:           # low, high, middle, high
            seq = [ns[0], ns[2], ns[1], ns[2]]
        else:
            seq = ns
        step = min(beat / float(rng.choice([2, 3, 4])), max(0.06, (dur - 0.1) / len(seq)))
        k = max(len(seq), int(dur / step))
        for i in range(k):
            on = t0 + i * step + max(0.0, jit(0.006))
            if on >= t1 - 0.05:
                break
            m = seq[i % len(seq)]
            off = t1 if pedal else min(t1, on + step * float(rng.uniform(1.0, 3.0)))
            accent = 6 if i % len(seq) == 0 else 0
            notes.append(P.PianoNote(m, _clamp_v(vel[m] - 6 + accent), on, max(0.05, off - on)))
    elif style == "pulse":
        for m in v.lh:
            notes.append(P.PianoNote(m, vel[m], t0 + max(0.0, jit()), max(0.05, dur)))
        step = beat / float(rng.choice([1, 2]))
        i = 0
        while t0 + i * step < t1 - 0.05:
            on = t0 + i * step + max(0.0, jit())
            length = min(t1, on + step * float(rng.uniform(0.55, 0.95))) - on
            acc = 0 if i == 0 else -float(rng.uniform(4, 12))
            for m in v.rh:
                notes.append(P.PianoNote(m, _clamp_v(vel[m] + acc), on, max(0.05, length)))
            i += 1
    elif style == "bass_chord":
        offset = min(beat * float(rng.choice([0.5, 1.0])), 0.5 * dur)
        lows = v.lh if v.lh else v.rh[:1]
        highs = v.rh if v.lh else v.rh[1:]
        for m in lows:
            notes.append(P.PianoNote(m, vel[m], t0 + max(0.0, jit()), max(0.05, dur)))
        t = t0 + offset
        while t < t1 - 0.1:
            on = t + max(0.0, jit())
            length = min(t1, on + offset * float(rng.uniform(0.8, 1.6))) - on
            for m in highs:
                notes.append(P.PianoNote(m, _clamp_v(vel[m] - 4), on, max(0.05, length)))
            t += 2 * offset
    return notes


def _style_pedal_p(style: str) -> float:
    return {"block": 0.5, "rolled": 0.6, "arpeggio": 0.85, "pulse": 0.2, "bass_chord": 0.4}[style]


def _fx(rng) -> dict:
    """Random recording conditions: room reverb, detuning and level."""
    reverb = None
    if rng.random() < 0.6:
        reverb = [round(float(rng.uniform(0.4, 2.5)), 3), round(float(rng.uniform(0.08, 0.35)), 3)]
    tune = float(np.clip(rng.normal(0.0, 6.0), -20.0, 20.0)) if rng.random() < 0.5 else 0.0
    return {"reverb": reverb, "tune_cents": round(tune, 2), "peak_db": round(float(rng.uniform(-12.0, -1.0)), 2)}


def _finish(piano: P.Piano, notes, pedal, duration: float, fx: dict, seed: int) -> np.ndarray:
    y = piano.render(notes, pedal=pedal, duration=duration, tune_cents=fx["tune_cents"], seed=seed,
                     reverb=tuple(fx["reverb"]) if fx["reverb"] else None)
    y = P.normalize(y, fx["peak_db"])
    return P.limiter(y, -0.5)


def _notes_dicts(notes: Sequence[P.PianoNote]) -> list[dict]:
    return [{"pitch": n.pitch, "velocity": n.velocity, "onset": round(n.onset, 5), "duration": round(n.duration, 5)}
            for n in sorted(notes, key=lambda n: (n.onset, n.pitch))]


def render_example(c: ChordSpec, voicing: Voicing | None = None, style: str | None = None,
                   rng: np.random.Generator | None = None, *, duration: float | None = None,
                   piano: P.Piano | None = None, fx: dict | None = None) -> tuple[np.ndarray, dict]:
    """Render one chord. Returns (audio (n, 2) float32 at 44.1 kHz, label dict).

    The chord sounds from ``t0`` (first key) to ``t1`` (keys released; with the pedal the
    strings ring a little longer), with silence before and a short ring-out after."""
    rng = rng if rng is not None else np.random.default_rng()
    piano = piano or P.get_piano()
    if voicing is None:
        vs = voicings(c, rng, 1)
        if not vs:
            raise RuntimeError(f"no voicing for {c.label}")
        voicing = vs[0]
    probs = check_voicing(c, voicing)
    if probs:
        raise ValueError(f"invalid voicing for {c.label}: {probs}")
    style = style or str(rng.choice(STYLES, p=[0.3, 0.15, 0.25, 0.15, 0.15]))
    dur = float(duration if duration is not None else rng.uniform(1.5, 3.5))
    lead = float(rng.uniform(0.05, 0.3))
    tail = float(rng.uniform(0.5, 1.0))
    bpm = float(rng.uniform(60, 120))
    base = float(rng.uniform(VEL_RANGE[0] + 10, VEL_RANGE[1] - 10))
    use_pedal = bool(rng.random() < _style_pedal_p(style))
    t0, t1 = lead, lead + dur
    notes = perform(voicing, style, t0, t1, rng, base, 60.0 / bpm, use_pedal)
    pedal = [(max(0.0, t0 - 0.03), t1)] if use_pedal else None
    fx = fx if fx is not None else _fx(rng)
    seed = int(rng.integers(2 ** 31))
    audio = _finish(piano, notes, pedal, t1 + tail, fx, seed)
    sounding = sorted({n.pitch % 12 for n in notes})
    label = {
        "kind": "chord", **c.as_dict(), "equivalents": equivalents(c), "sounding_pcs": sounding,
        "t0": round(t0, 5), "t1": round(t1, 5), "duration": round(len(audio) / SR, 5), "sr": SR,
        "voicing": voicing.as_dict(), "style": style, "bpm": round(bpm, 2), "base_velocity": round(base, 1),
        "pedal": [list(map(lambda x: round(x, 5), p)) for p in (pedal or [])], "fx": fx,
        "notes": _notes_dicts(notes),
    }
    return audio, label


# ---------------------------------------------------------------------------------- progressions

#: Extra worship / pop progressions (bar text in C, the eval/synth DSL: one token per beat,
#: "." holds, "(A B)" splits a beat, "*" marks a passing chord). Together with
#: eval.synth.render.IDIOMS these cover all 32 qualities.
PROGRESSIONS: dict[str, str] = {
    "axis": "| C . . . | G . . . | Am . . . | F . . . |",
    "axis_v7": "| C . . . | G/B . . . | Am . . . | F . . . |",
    "axis_minor_start": "| Am . . . | F . . . | C . . . | G . . . |",
    "canon_desc": "| C . *G/B . | Am . *Am/G . | F . *C/E . | Dm7 . G . |",
    "walk_up_ii": "| C . *Dm7 *C/E | F . *G . | Am . *G/B . | C . . . |",
    "four_over_five": "| F . . . | F/G . . . | C . . . | Csus4 . C . |",
    "sus_resolve": "| C . . . | Gsus4 . G . | Am . F . | D7sus4 . D7 . |",
    "passing_dim": "| C . *C#dim7 . | Dm7 . *D#dim . | C/E . *Fm6 . | C/G . G7 . |",
    "secondary_dom": "| C . *E7/G# . | Am . *C7/G . | F . *D7/F# . | G . G7 . |",
    "two_five_one": "| Dm9 . G13 . | Cmaj9 . *A7b9 . | Dm11 . G7#9 . | C6/9 . . . |",
    "line_cliche": "| Am . . . | Am(maj7) . . . | Am7 . . . | Am6 . . . |",
    "aug_cliche": "| C . . . | C+ . . . | C6 . . . | C7 . . . |",
    "jazz_colours": "| Fmaj7#11 . . . | Em7 . *A7b13 . | Dm7 . *Abmaj7#5 . | Cmaj7 . . . |",
    "worship_sus2": "| C2 . . . | Am7 . . . | Fmaj7 . . . | G6 . *G+7 . |",
    "add9_pad": "| Cadd9 . . . | G/B . . . | Am(add9) . . . | Fadd9 . . . |",
    "minor_circle": "| Am . Dm7 . | G7 . Cmaj7 . | Fmaj7 . Bm7b5 . | E7 . Am . |",
    "suspended_13": "| Dm11 . . . | G13sus4 . G9sus4 . | C . *Bb/C . | F/C . C . |",
    "power_rock": "| C5 . . . | Ab5 . . . | Bb5 . . . | C5 . . . |",
    "pedal_point": "| C . *F/C . | C . *G/C . | C . *Bb/C . | C . . . |",
    "gospel_turn": "| C . *C7/E . | F . *Fm6 . | C/G . *A7 . | Dm7 . G9 G7 |",
    "dim_resolve": "| C . *Bdim/D . | C/E . *Ebdim7 . | Dm7/F . G7sus4 . | C . . . |",
}


@dataclass
class ProgChord:
    chord: ChordSpec
    beats: float
    passing: bool = False
    tags: tuple[str, ...] = ()


def _from_render_chords(chords) -> list[ProgChord]:
    out = []
    for ch in chords:
        r, q, b = H.parse_harte(ch.label)
        out.append(ProgChord(lookup(r, q, b), float(ch.beats), bool(ch.passing), tuple(ch.tags)))
    return out


@lru_cache(maxsize=1)
def progression_library() -> dict[str, list[ProgChord]]:
    """All progressions in C (``eval.synth.render.IDIOMS`` are transposed to C)."""
    from eval.synth import render as R
    lib: dict[str, list[ProgChord]] = {}
    for idiom in R.IDIOMS:
        lib[idiom.name] = _from_render_chords(idiom.chords("C"))
    for name, chart in PROGRESSIONS.items():
        lib[name] = _from_render_chords(R.parse_chart(chart, 4, default_tag=name))
    return lib


def random_progression(rng: np.random.Generator, name: str | None = None, key: int | None = None,
                       n_chords: int | None = None) -> tuple[str, int, list[ProgChord]]:
    """(name, key pc, 4–8 consecutive chords of a library progression transposed to the key)."""
    lib = progression_library()
    names = sorted(lib)
    name = name or names[int(rng.integers(len(names)))]
    key = int(rng.integers(12)) if key is None else int(key) % 12
    full = lib[name]
    k = n_chords or int(rng.integers(4, 9))
    k = min(k, len(full))
    start = int(rng.integers(0, len(full) - k + 1))
    chords = [ProgChord(pc.chord.transpose(key), pc.beats, pc.passing, pc.tags) for pc in full[start:start + k]]
    return name, key, chords


def _movement(prev: Sequence[int], new: Sequence[int]) -> float:
    if not prev:
        return 0.0
    a = sum(min(abs(m - p) for p in prev) for m in new)
    b = sum(min(abs(m - p) for m in new) for p in prev)
    return (a + b) / 2 + 0.5 * abs(new[-1] - prev[-1])


def _as_prog(items) -> list[ProgChord]:
    out = []
    for it in items:
        if isinstance(it, ProgChord):
            out.append(it)
        elif isinstance(it, ChordSpec):
            out.append(ProgChord(it, 4.0))
        elif isinstance(it, str):
            out.append(ProgChord(chord(it), 4.0))
        else:
            c, beats, *rest = it
            c = chord(c) if isinstance(c, str) else c
            out.append(ProgChord(c, float(beats), bool(rest[0]) if rest else False))
    return out


def render_progression(chords, bpm: float, rng: np.random.Generator, *, style: str | None = None,
                       layout: str | None = None, piano: P.Piano | None = None, fx: dict | None = None,
                       pedal: bool | None = None) -> tuple[np.ndarray, dict]:
    """Render a short progression: ``chords`` = ProgChord / ChordSpec / symbol strings /
    (chord, beats[, passing]) tuples. Returns (audio, info with exact ``segments``)."""
    piano = piano or P.get_piano()
    items = _as_prog(chords)
    if not items:
        raise ValueError("empty progression")
    spb = 60.0 / float(bpm)
    style = style or str(rng.choice(STYLES, p=[0.2, 0.15, 0.2, 0.25, 0.2]))
    layout = layout or str(rng.choice(["lh_rh", "two_hand_block"], p=[0.75, 0.25]))
    use_pedal = bool(rng.random() < _style_pedal_p(style)) if pedal is None else bool(pedal)
    base = float(rng.uniform(VEL_RANGE[0] + 10, VEL_RANGE[1] - 15))
    lead = float(rng.uniform(0.1, 0.4))
    t = lead
    prev_rh: tuple[int, ...] = ()
    prev_bass: int | None = None
    notes: list[P.PianoNote] = []
    segs, pedal_iv = [], []
    for i, pc in enumerate(items):
        t0, t1 = t, t + pc.beats * spb
        cands = voicings(pc.chord, rng, 12, layout=layout if (layout != "one_hand" or pc.chord.inversion != "non_chord") else "lh_rh")
        if not cands:
            cands = voicings(pc.chord, rng, 4)
        if not cands:
            raise RuntimeError(f"no voicing for {pc.chord.label}")
        cost = np.array([_movement(prev_rh, v.rh) + (0.3 * abs(v.notes[0] - prev_bass) if prev_bass else 0.0)
                         for v in cands])
        w = np.exp(-(cost - cost.min()) / 2.0)
        v = cands[int(rng.choice(len(cands), p=w / w.sum()))]
        prev_rh, prev_bass = v.rh, v.notes[0]
        st = "block" if (pc.passing or pc.beats <= 1.0) and style in ("arpeggio", "pulse", "bass_chord") else style
        accent = base * (0.85 if pc.passing else 1.0) + float(rng.normal(0, 3))
        notes += perform(v, st, t0, t1, rng, accent, spb, use_pedal)
        if use_pedal:
            pedal_iv.append((t0 + 0.03, t1))                     # legato pedal: up at the change
        segs.append({"t0": round(t0, 5), "t1": round(t1, 5), **pc.chord.as_dict(), "beats": pc.beats,
                     "passing": pc.passing, "tags": list(pc.tags), "voicing": v.as_dict(), "style": st})
        t = t1
    fx = fx if fx is not None else _fx(rng)
    tail = float(rng.uniform(0.6, 1.2))
    seed = int(rng.integers(2 ** 31))
    audio = _finish(piano, notes, pedal_iv or None, t + tail, fx, seed)
    info = {"kind": "progression", "bpm": round(float(bpm), 3), "style": style, "layout": layout,
            "segments": segs, "t0": segs[0]["t0"], "t1": segs[-1]["t1"], "duration": round(len(audio) / SR, 5),
            "sr": SR, "pedal": [[round(a, 5), round(b, 5)] for a, b in pedal_iv], "fx": fx,
            "notes": _notes_dicts(notes)}
    return audio, info


# --------------------------------------------------------------------------------------- dataset

def make_example(i: int, seed: int, progression_share: float = 0.25, chord_mode: str = "balanced",
                 piano: P.Piano | None = None) -> dict:
    """Example ``i`` of the dataset with this seed (same inputs → same audio and labels)."""
    rng = np.random.default_rng([int(seed), int(i)])
    if rng.random() < progression_share:
        name, key, chords = random_progression(rng)
        bpm = float(rng.uniform(60, 120))
        audio, info = render_progression(chords, bpm, rng, piano=piano)
        info.update(progression=name, key=key)
    else:
        c = sample_chord(rng, chord_mode)
        audio, info = render_example(c, None, None, rng, piano=piano)
    return {"id": f"{seed}-{i}", "audio": audio, "sr": SR, **info}


def iter_dataset(n: int, seed: int = 0, progression_share: float = 0.25, chord_mode: str = "balanced",
                 piano: P.Piano | None = None, start: int = 0) -> Iterator[dict]:
    """Render ``n`` examples on the fly (nothing is written to disk)."""
    for i in range(start, start + n):
        yield make_example(i, seed, progression_share, chord_mode, piano)


# ------------------------------------------------------------------------------------------- CLI

def _safe(text: str) -> str:
    return (text.replace("6/9", "69").replace("#", "s").replace("/", "_over_").replace("+", "aug").replace("°", "dim")
            .replace("ø", "hdim").replace("(", "").replace(")", "").replace(" ", ""))


def _spelled_pcs(seg: dict) -> list[str]:
    c = lookup(seg["root"], seg["quality"], seg["bass"])
    root = S.parse_note(H.parse_harte_full(c.label).root_name)
    order = sorted(seg["pcs"], key=lambda p: (p - c.root) % 12)
    return [S.name_pc(root, c.quality, p) for p in order]


def preview_plan(n: int) -> list[tuple[str, int]]:
    """Every third item a progression; the single chords walk through all 32 qualities."""
    return [("progression" if k % 3 == 2 else "chord", k) for k in range(n)]


def write_preview(out_dir, n: int = 60, seed: int = 2026, piano: P.Piano | None = None,
                  verbose: bool = True) -> list[dict]:
    """A small listening set: ``n`` short FLAC files + labels.csv (one row per chord) + labels.json."""
    import soundfile as sf
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    piano = piano or P.get_piano()
    lib = sorted(progression_library())
    rows, infos = [], []
    n_chord = n_prog = 0
    for kind, k in preview_plan(n):
        rng = np.random.default_rng([int(seed), k])
        t = time.perf_counter()
        if kind == "chord":
            q = T.QUALITIES[n_chord % len(T.QUALITIES)]
            ivs = T.allowed_bass_intervals(q)
            iv = ivs[(n_chord // len(T.QUALITIES) + n_chord) % len(ivs)] if n_chord % 2 else 0
            root = int(rng.integers(12))
            c = lookup(root, q, root + iv)
            style = STYLES[n_chord % len(STYLES)]
            audio, info = render_example(c, None, style, rng, piano=piano)
            name = f"{k + 1:03d}_chord_{_safe(c.symbol)}_{style}"
            segs = [dict(info, t0=info["t0"], t1=info["t1"])]
            n_chord += 1
        else:
            pname = lib[n_prog % len(lib)]
            name_, key, chords = random_progression(rng, name=pname)
            style = STYLES[(n_prog * 2 + 1) % len(STYLES)]
            audio, info = render_progression(chords, float(rng.uniform(64, 100)), rng, style=style, piano=piano)
            info.update(progression=pname, key=key)
            name = f"{k + 1:03d}_prog_{pname}_in_{_safe(NAMES[key])}_{style}"
            segs = info["segments"]
            n_prog += 1
        fname = f"{name}.flac"
        sf.write(out_dir / fname, audio, SR, subtype="PCM_16", format="FLAC")
        for j, s in enumerate(segs):
            rows.append({
                "file": fname, "kind": kind, "segment": j + 1, "start_s": f"{s['t0']:.3f}", "end_s": f"{s['t1']:.3f}",
                "chord": s["symbol"], "harte": s["label"], "root": NAMES[s["root"]], "quality": s["quality"],
                "bass": NAMES[s["bass"]], "inversion": s["inversion"],
                "pitch_classes": " ".join(_spelled_pcs(s)),
                "notes": " ".join(spelled_note(lookup(s["root"], s["quality"], s["bass"]), m)
                                  for m in sorted(s["voicing"]["lh"] + s["voicing"]["rh"])),
                "layout": s["voicing"]["layout"],
                "position": s["voicing"]["position"], "style": s.get("style", info.get("style")),
                "pedal": "yes" if info["pedal"] else "no", "passing": "yes" if s.get("passing") else "",
                "progression": info.get("progression", ""), "bpm": info.get("bpm", ""),
                "reverb_rt60": info["fx"]["reverb"][0] if info["fx"]["reverb"] else "",
                "tune_cents": info["fx"]["tune_cents"],
            })
        info = {k2: v for k2, v in info.items() if k2 != "audio"}
        infos.append({"file": fname, **info})
        if verbose:
            print(f"[{k + 1:3d}/{n}] {fname}  {len(audio) / SR:4.1f} s  ({(time.perf_counter() - t) * 1000:.0f} ms)",
                  flush=True)
    with open(out_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "labels.json").write_text(json.dumps(infos, indent=1, ensure_ascii=False), encoding="utf-8")
    return infos


def print_stats(quick: bool = False) -> dict:
    chords = enumerate_chords()
    v = V.candidates()
    inv = Counter(c.inversion for c in chords)
    print(f"Chords in the vocabulary: {len(chords)} (12 roots x {len(T.QUALITIES)} qualities x allowed basses)")
    print(f"  root position {inv['root']}, inversions (chord-tone bass) {inv['chord_tone']}, "
          f"foreign bass (Em/A, F/G, ...) {inv['non_chord']}")
    print(f"  sound-alike classes (same notes + same bass, e.g. C6 = Am7/C): {len(v.classes) - 1}")
    print(f"  progression library: {len(progression_library())} progressions "
          f"(IDIOMS of eval/synth/render.py + PROGRESSIONS)")
    sample = chords if not quick else [c for c in chords if c.root == 0]
    t = time.perf_counter()
    st = voicing_stats(sample)
    scale = 1 if not quick else 12
    print(f"Voicings (distinct note sets the rules allow{'; root C only x 12' if quick else ''}): "
          f"{st['total'] * scale:,}  (two hands {st['two_hand'] * scale:,}, one hand {st['one_hand'] * scale:,}); "
          f"left-hand shapes {st['lh_patterns'] * scale:,}  [{time.perf_counter() - t:.1f} s]")
    print(f"  per chord: mean {st['total'] / len(sample):,.0f}")
    top = sorted(st["per_quality"].items(), key=lambda kv: -kv[1])
    print("  per quality: " + ", ".join(f"{q} {n * scale:,}" for q, n in top))
    print(f"Layouts {LAYOUTS}; right-hand positions {POSITIONS}; styles {STYLES}")
    print("  x velocities 30-110 x pedal on/off x timing x reverb/detune/level -> every example is new")
    return st


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m train.chords", description=__doc__.split("\n\n")[0])
    ap.add_argument("--stats", action="store_true", help="count chords and voicings")
    ap.add_argument("--quick", action="store_true", help="with --stats: count root C only (x12)")
    ap.add_argument("--preview", type=int, metavar="N", help="write N short FLAC files + labels.csv")
    ap.add_argument("--out", default=str(P.ROOT / "data" / "train" / "preview"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)
    if not a.stats and not a.preview:
        ap.error("give --stats and/or --preview N")
    if a.stats:
        print_stats(a.quick)
    if a.preview:
        t = time.perf_counter()
        piano = P.get_piano()
        infos = write_preview(a.out, a.preview, a.seed, piano)
        size = sum(f.stat().st_size for f in Path(a.out).glob("*.flac"))
        print(f"wrote {len(infos)} files ({size / 1e6:.1f} MB) + labels.csv to {a.out} "
              f"in {time.perf_counter() - t:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
