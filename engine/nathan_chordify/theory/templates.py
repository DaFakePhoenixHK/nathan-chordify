"""Chord-quality templates (DESIGN §5.8 v1.1): 32 internal qualities plus N.

Each quality is a list of tones given as Harte-style degrees relative to the root. A tone is
*required*, *optional* or (when it is not listed) *absent*:

- the perfect 5th is optional except in dim, aug, m7b5, dim7, +7, +maj7 and 5 (where the
  fifth is altered or the whole point of the chord);
- tones marked "(2 optional)" in §5.8 (the 9th of m11, 13sus4 and 13) are optional;
- every other listed tone is required.

Degrees ≥ 9 (9, b9, #9, 11, #11, 13, b13) are *extensions*: they are never allowed as the bass.
The sus tones (2 of sus2, 4 of sus4 / 7sus4 / 9sus4 / 13sus4) are chord tones, not extensions,
so Gsus4/C is a legal candidate (it equals Csus2 and shares its class).
"""
from __future__ import annotations

from dataclasses import dataclass

N = "N"

#: The 32 internal qualities, in the order of DESIGN §5.8.
QUALITIES: tuple[str, ...] = (
    "maj", "min", "dim", "aug", "sus2", "sus4", "5", "6", "m6",
    "7", "maj7", "m7", "m7b5", "dim7", "mM7", "+maj7",
    "7sus4", "+7", "add9", "madd9", "6/9", "9",
    "maj9", "m9", "m11", "9sus4", "13sus4",
    "13", "7b9", "7#9", "7b13", "maj7#11",
)

# degree list per quality; "?" = optional tone
_SPEC: dict[str, str] = {
    "maj": "1 3 5?",
    "min": "1 b3 5?",
    "dim": "1 b3 b5",
    "aug": "1 3 #5",
    "sus2": "1 2 5?",
    "sus4": "1 4 5?",
    "5": "1 5",
    "6": "1 3 5? 6",
    "m6": "1 b3 5? 6",
    "7": "1 3 5? b7",
    "maj7": "1 3 5? 7",
    "m7": "1 b3 5? b7",
    "m7b5": "1 b3 b5 b7",
    "dim7": "1 b3 b5 bb7",
    "mM7": "1 b3 5? 7",
    "+maj7": "1 3 #5 7",
    "7sus4": "1 4 5? b7",
    "+7": "1 3 #5 b7",
    "add9": "1 9 3 5?",
    "madd9": "1 9 b3 5?",
    "6/9": "1 9 3 5? 6",
    "9": "1 9 3 5? b7",
    "maj9": "1 9 3 5? 7",
    "m9": "1 9 b3 5? b7",
    "m11": "1 9? b3 11 5? b7",
    "9sus4": "1 9 4 5? b7",
    "13sus4": "1 9? 4 5? 13 b7",
    "13": "1 9? 3 5? 13 b7",
    "7b9": "1 b9 3 5? b7",
    "7#9": "1 #9 3 5? b7",
    "7b13": "1 3 5? b13 b7",
    "maj7#11": "1 3 #11 5? 7",
}

_BASE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}


def degree_semitones(degree: str) -> int:
    """Semitones (0..11) of a Harte degree such as "b3", "#5", "bb7", "9", "#11", "b13"."""
    d = degree.strip()
    acc = 0
    while d and d[0] in "b#":
        acc += -1 if d[0] == "b" else 1
        d = d[1:]
    if not d.isdigit():
        raise ValueError(f"bad degree {degree!r}")
    n = int(d)
    if n < 1 or n > 15:
        raise ValueError(f"bad degree {degree!r}")
    base = _BASE_SEMITONES[(n - 1) % 7 + 1]
    return (base + acc) % 12


def degree_step(degree: str) -> int:
    """Letter steps (0..6) above the root for a degree: "b3" → 2, "9" → 1, "#11" → 3, "13" → 5."""
    n = int(degree.lstrip("b#"))
    return (n - 1) % 7


def is_extension(degree: str) -> bool:
    """9ths, 11ths and 13ths (any alteration) are extensions."""
    return int(degree.lstrip("b#")) >= 8


@dataclass(frozen=True)
class Tone:
    interval: int     # semitones above the root, 0..11
    degree: str       # Harte degree name: "1", "b3", "#5", "bb7", "9", "#11", "b13", ...
    optional: bool    # optional template tone (τ = 0.5); otherwise required (τ = 0.85)

    @property
    def ext(self) -> bool:
        return is_extension(self.degree)


def _build() -> dict[str, tuple[Tone, ...]]:
    out: dict[str, tuple[Tone, ...]] = {}
    for q, spec in _SPEC.items():
        tones = []
        for tok in spec.split():
            opt = tok.endswith("?")
            deg = tok.rstrip("?")
            tones.append(Tone(degree_semitones(deg), deg, opt))
        ivs = [t.interval for t in tones]
        assert len(set(ivs)) == len(ivs), q
        out[q] = tuple(sorted(tones, key=lambda t: t.interval))
    return out


TEMPLATE: dict[str, tuple[Tone, ...]] = _build()
assert tuple(TEMPLATE) == QUALITIES

#: Generic degree names for intervals that are not chord tones (used for non-chord basses).
GENERIC_DEGREE = {0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "#4",
                  7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7"}

#: Non-chord bass intervals (+2, +5, +10) and the qualities that allow them (§5.8 v1.1).
NONCHORD_BASS_INTERVALS: tuple[int, ...] = (2, 5, 10)
NONCHORD_BASS_QUALITIES = frozenset({"maj", "min", "sus2", "sus4", "add9", "maj7", "m7", "6"})


def tones(q: str) -> frozenset[int]:
    """All template tones (required + optional), as intervals from the root."""
    return frozenset(t.interval for t in TEMPLATE[q])


def required(q: str) -> frozenset[int]:
    return frozenset(t.interval for t in TEMPLATE[q] if not t.optional)


def optional(q: str) -> frozenset[int]:
    return frozenset(t.interval for t in TEMPLATE[q] if t.optional)


def absent(q: str) -> frozenset[int]:
    return frozenset(range(12)) - tones(q)


def role(q: str, interval: int) -> str:
    """"required", "optional" or "absent" for an interval (semitones above the root)."""
    interval %= 12
    for t in TEMPLATE[q]:
        if t.interval == interval:
            return "optional" if t.optional else "required"
    return "absent"


def tone(q: str, interval: int) -> Tone | None:
    interval %= 12
    for t in TEMPLATE[q]:
        if t.interval == interval:
            return t
    return None


def chord_bass_intervals(q: str) -> tuple[int, ...]:
    """Chord tones that may be the bass: every template tone except 9ths, 11ths and 13ths."""
    return tuple(t.interval for t in TEMPLATE[q] if not t.ext)


def allowed_bass_intervals(q: str) -> tuple[int, ...]:
    """Root first, then the other chord-tone basses, then the non-chord basses (§5.8)."""
    out = list(chord_bass_intervals(q))
    if q in NONCHORD_BASS_QUALITIES:
        out += [i for i in NONCHORD_BASS_INTERVALS if i not in out]
    return tuple(out)


def bass_degree(q: str, interval: int) -> str:
    """Harte degree of a bass interval: the chord tone's own degree name when the bass is a
    template tone (so aug/#5, dim7/bb7), otherwise the generic name (2, 4, b7, ...)."""
    interval %= 12
    t = tone(q, interval)
    if t is not None and not t.ext:
        return t.degree
    return GENERIC_DEGREE[interval]


# --- quality families used by priors, spelling and numbers ---------------------------------

#: §5.11 core quality: strips only the 9/11/13 extensions and alterations.
CORE: dict[str, str] = {q: q for q in QUALITIES}
CORE.update({"add9": "maj", "madd9": "min", "6/9": "6", "9": "7", "13": "7", "7b9": "7",
             "7#9": "7", "7b13": "7", "maj9": "maj7", "maj7#11": "maj7", "m9": "m7", "m11": "m7"})

DOMINANT = frozenset({"7", "9", "13", "7b9", "7#9", "7b13", "+7"})
DIM_FAMILY = frozenset({"dim", "dim7", "m7b5"})
AUG_FAMILY = frozenset({"aug", "+7", "+maj7"})
SUS4_FAMILY = frozenset({"sus4", "7sus4", "9sus4", "13sus4"})
MINOR_FAMILY = frozenset({"min", "m6", "m7", "mM7", "madd9", "m9", "m11"})
#: Qualities whose third is minor (or that are diminished): lowercase Roman numerals.
LOWER_CASE = MINOR_FAMILY | DIM_FAMILY

#: Passing-type qualities (§5.8); a chord whose bass ≠ root is also passing-type.
PT_QUALITIES = frozenset({"dim", "dim7", "m7b5", "aug", "+7", "+maj7", "sus2", "sus4",
                          "7sus4", "9sus4", "13sus4"})
