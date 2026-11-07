"""Piano voicings (DESIGN §5.12 step 10 v1.1, review A17). One implementation, on the server.

- **LH**: the bass note in MIDI 36–52, the one nearest the previous LH note (ties → lower).
- **RH**: 3–4 notes in MIDI 55–79 that always keep the 3rd (or sus tone), the 7th and the 9th:
  - *root position*: the template tones; above 4 notes the optional 5th goes first, then the
    root (the LH already plays it), then the optional 9th. ``rootless=True`` (Settings, off by
    default) drops the root whenever 3 notes remain;
  - *inversion* (bass is a chord tone): the RH keeps the root and drops the bass pc, doubling the
    root when fewer than 3 notes remain (B/D#: LH D#, RH B–F#–B, not D#–F#, which sounds like
    D#m; D#7/A#: RH D#–G–C#);
  - *non-chord bass* (Em/A, F#/B, E/F#): the RH plays the full upper chord;
  - *upper structures* (root-position 9sus4 / 13sus4 shown as F/G, Emaj7/F#): the RH plays the
    upper triad / maj7 (the template without root and 5th).
- **Voice leading**: among all RH placements (span ≤ 14 semitones) the one with the lowest
  cost = total semitone movement from the previous RH voicing (voices matched optimally)
  − 2 when the top note moves by ≤ 2 semitones (a *bonus*: the A17 sign fix) + 3 per minor 2nd
  whose lower note is below MIDI 60. With no previous voicing the cost is the distance of the
  voicing's centre from MIDI 66.
- ``transpose_voicings`` shifts every note by t, then the whole song by −12, 0 or +12 to
  minimise the notes outside the ranges (§5.12 step 12).
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass

from . import templates as T

LH_RANGE = (36, 52)
RH_RANGE = (55, 79)
MAX_SPAN = 14
LH_START = 44
RH_CENTRE = 66


@dataclass(frozen=True)
class Voicing:
    lh: tuple[int, ...]
    rh: tuple[int, ...]

    def as_dict(self) -> dict[str, list[int]]:
        return {"lh": list(self.lh), "rh": list(self.rh)}


EMPTY = Voicing((), ())


def _reduce(ivs: list[int], quality: str, keep_root: bool) -> list[int]:
    """Trim to 4 tones: optional 5th first, then (if allowed) the root, then the optional 9th."""
    opt = T.optional(quality)
    for drop in [7] + ([] if keep_root else [0]) + [2]:
        if len(ivs) <= 4:
            break
        if drop in ivs and (drop == 0 or drop in opt):
            ivs.remove(drop)
    assert len(ivs) <= 4, (quality, ivs)      # holds for all 32 qualities
    return ivs


def rh_pitch_classes(root: int, quality: str, bass: int, upper_structure: bool = True,
                     rootless: bool = False) -> list[int]:
    """RH pitch classes (a doubled root appears twice), lowest-priority-dropped first."""
    if quality == T.N or root < 0:
        return []
    tones = sorted(T.tones(quality))
    iv_bass = (bass - root) % 12
    if iv_bass == 0:
        if upper_structure and quality in ("9sus4", "13sus4"):
            ivs = [i for i in tones if i not in (0, 7)]
        else:
            ivs = _reduce(list(tones), quality, keep_root=False)
            if rootless and 0 in ivs and len(ivs) - 1 >= 3:
                ivs.remove(0)
    elif iv_bass in tones:
        ivs = _reduce([i for i in tones if i != iv_bass], quality, keep_root=True)
    else:
        ivs = _reduce(list(tones), quality, keep_root=False)
    if len(ivs) < 3:
        ivs = ivs + [0]                       # double (or add) the root
    return [(root + i) % 12 for i in ivs]


def _options(pc: int, lo: int, hi: int) -> list[int]:
    first = lo + ((pc - lo) % 12)
    return list(range(first, hi + 1, 12))


def _movement(prev: tuple[int, ...], new: tuple[int, ...]) -> int:
    """Total semitone movement with an optimal voice matching (extra voices: nearest note)."""
    a, b = (prev, new) if len(prev) <= len(new) else (new, prev)
    best = None
    for perm in itertools.permutations(b, len(a)):
        cost = sum(abs(x - y) for x, y in zip(a, perm))
        rest = [y for y in b if y not in perm] if len(b) > len(a) else []
        cost += sum(min(abs(y - x) for x in a) for y in rest)
        if best is None or cost < best:
            best = cost
    return best or 0


def _m2_penalty(notes: tuple[int, ...]) -> int:
    return sum(3 for x, y in zip(notes, notes[1:]) if y - x == 1 and x < 60)


def rh_cost(notes: tuple[int, ...], prev: tuple[int, ...] | None) -> float:
    if not prev:
        return abs((notes[0] + notes[-1]) / 2 - RH_CENTRE) + _m2_penalty(notes)
    cost = float(_movement(prev, notes))
    if abs(notes[-1] - prev[-1]) <= 2:
        cost -= 2.0
    return cost + _m2_penalty(notes)


def rh_candidates(pcs: list[int]) -> list[tuple[int, ...]]:
    """Every placement of the RH pitch classes in 55–79 with distinct notes and span ≤ 14."""
    lo, hi = RH_RANGE
    opts = [_options(pc, lo, hi) for pc in pcs]
    out = set()
    for combo in itertools.product(*opts):
        if len(set(combo)) < len(combo):
            continue
        notes = tuple(sorted(combo))
        if notes[-1] - notes[0] <= MAX_SPAN:
            out.add(notes)
    return sorted(out)


def choose_lh(bass: int, prev_lh: int | None) -> int:
    lo, hi = LH_RANGE
    ref = LH_START if prev_lh is None else prev_lh
    return min(_options(bass % 12, lo, hi), key=lambda m: (abs(m - ref), m))


def voice(root: int, quality: str, bass: int, prev: Voicing | None = None,
          upper_structure: bool = True, rootless: bool = False) -> Voicing:
    """Voice one chord, voice-led from ``prev`` (None or EMPTY = no previous chord)."""
    if quality == T.N or root < 0:
        return EMPTY
    pcs = rh_pitch_classes(root, quality, bass, upper_structure, rootless)
    prev_rh = prev.rh if prev and prev.rh else None
    prev_lh = prev.lh[0] if prev and prev.lh else None
    cands = rh_candidates(pcs)
    if not cands:                       # cannot happen with ≤ 4 pcs in a 25-semitone range
        raise RuntimeError(f"no RH voicing for {pcs}")
    rh = min(cands, key=lambda n: (rh_cost(n, prev_rh), n[-1] - n[0], n))
    return Voicing((choose_lh(bass, prev_lh),), rh)


def voice_sequence(chords: Iterable[tuple[int, str, int]], upper_structure: bool = True,
                   rootless: bool = False) -> list[Voicing]:
    """Voice a song's chords in order; N chords get EMPTY and do not reset the voice leading."""
    out: list[Voicing] = []
    prev: Voicing | None = None
    for root, quality, bass in chords:
        v = voice(root, quality, bass, prev, upper_structure, rootless)
        out.append(v)
        if v.rh:
            prev = v
    return out


def _outside(voicings: list[Voicing], shift: int) -> int:
    n = 0
    for v in voicings:
        n += sum(not (LH_RANGE[0] <= x + shift <= LH_RANGE[1]) for x in v.lh)
        n += sum(not (RH_RANGE[0] <= x + shift <= RH_RANGE[1]) for x in v.rh)
    return n


def transpose_voicings(voicings: list[Voicing], t: int) -> list[Voicing]:
    """Shift by t semitones, then the whole song by −12, 0 or +12 (fewest notes out of range;
    ties prefer 0, then −12)."""
    moved = [Voicing(tuple(x + t for x in v.lh), tuple(x + t for x in v.rh)) for v in voicings]
    shift = min((0, -12, 12), key=lambda s: (_outside(moved, s), abs(s), s))
    return [Voicing(tuple(x + shift for x in v.lh), tuple(x + shift for x in v.rh)) for v in moved]
