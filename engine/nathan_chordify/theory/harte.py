"""Harte chord labels and typed lead-sheet symbols ↔ (root pc, internal quality, bass pc).

Canonical Harte strings for the 32 internal qualities (DESIGN §5.8) use the standard shorthands
where one exists and a shorthand plus added degrees otherwise, so ``mir_eval`` reads every label:

    maj min dim aug sus2 sus4 5 maj6 min6 7 maj7 min7 hdim7 dim7 minmaj7 aug(7)
    sus4(b7) aug(b7) maj(9) min(9) maj6(9) 9 maj9 min9 min11 sus4(b7,9) sus4(b7,9,13)
    13 7(b9) 7(#9) 7(b13) maj7(#11)

The bass is a degree relative to the root ("C#:min7/b3", "E:maj/2", "B:aug/#5"). N is
``(-1, "N", -1)``; "X" (unknown) also parses as N.

``parse_symbol`` reads what a user types in the chord editor or a bar-text chart: "F#/A#",
"C#m7", "Bsus4", "E/F#", "Em6", "B+7", "Bb", "Cmaj7", "CΔ7", "C°7", "Cø", "Cm(maj7)", "C6/9",
"F#11" ... Conventions chosen where usage differs:

- "11" (no "m") means 9sus4, the worship/gospel usage (F#11 = E/F#), not the dominant 11 with
  a 3rd; "C2" means Csus2 and "Cadd2" means Cadd9;
- "Cø" / "Cø7" / "Cm7b5" are all m7b5; "C°" is dim and "C°7" is dim7;
- "sus" alone is sus4; "7#5" / "aug7" are +7; "maj7#5" is +maj7.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import templates as T
from .spelling import parse_note, simple

# --- Harte <-> internal quality ---------------------------------------------------------------

HARTE: dict[str, str] = {
    "maj": "maj", "min": "min", "dim": "dim", "aug": "aug", "sus2": "sus2", "sus4": "sus4",
    "5": "5", "6": "maj6", "m6": "min6", "7": "7", "maj7": "maj7", "m7": "min7",
    "m7b5": "hdim7", "dim7": "dim7", "mM7": "minmaj7", "+maj7": "aug(7)", "7sus4": "sus4(b7)",
    "+7": "aug(b7)", "add9": "maj(9)", "madd9": "min(9)", "6/9": "maj6(9)", "9": "9",
    "maj9": "maj9", "m9": "min9", "m11": "min11", "9sus4": "sus4(b7,9)",
    "13sus4": "sus4(b7,9,13)", "13": "13", "7b9": "7(b9)", "7#9": "7(#9)", "7b13": "7(b13)",
    "maj7#11": "maj7(#11)",
}
assert set(HARTE) == set(T.QUALITIES)

#: Harte / mir_eval shorthand → intervals ("13" follows mir_eval: no 11th).
SHORTHAND: dict[str, frozenset[int]] = {k: frozenset(v) for k, v in {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "7": (0, 4, 7, 10), "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10), "minmaj7": (0, 3, 7, 11), "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9),
    "9": (0, 2, 4, 7, 10), "maj9": (0, 2, 4, 7, 11), "min9": (0, 2, 3, 7, 10),
    "sus4": (0, 5, 7), "sus2": (0, 2, 7), "11": (0, 2, 4, 5, 7, 10),
    "min11": (0, 2, 3, 5, 7, 10), "13": (0, 2, 4, 7, 9, 10), "maj13": (0, 2, 4, 7, 9, 11),
    "min13": (0, 2, 3, 7, 9, 10), "1": (0,), "5": (0, 7),
}.items()}

_BY_SET: dict[frozenset[int], str] = {T.tones(q): q for q in T.QUALITIES}
assert len(_BY_SET) == len(T.QUALITIES)  # every quality has a unique tone set
_BY_HARTE = {v: k for k, v in HARTE.items()}

NO_CHORD = (-1, T.N, -1)

# Common default names when only a pitch class is known (no key context).
DEFAULT_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


@dataclass(frozen=True)
class ParsedChord:
    root: int                 # pc, -1 for N
    quality: str
    bass: int                 # pc, -1 for N
    root_name: str            # as written ("F#"), "N" for N
    bass_interval: int = 0    # semitones above the root

    @property
    def triple(self) -> tuple[int, str, int]:
        return (self.root, self.quality, self.bass)


def quality_from_intervals(ivs: frozenset[int] | set[int], strict: bool = True) -> str:
    """Internal quality for a set of intervals (0 = root must be present).

    Exact template match first; then a template whose required tones ⊆ set ⊆ all tones (an
    omitted optional 5th or 9th); then (``strict=False``) the closest template.
    """
    s = frozenset(i % 12 for i in ivs) | {0}
    if s in _BY_SET:
        return _BY_SET[s]
    fits = [q for q in T.QUALITIES if T.required(q) <= s <= T.tones(q)]
    if fits:
        return min(fits, key=lambda q: (len(T.tones(q)), T.QUALITIES.index(q)))
    if strict:
        raise ValueError(f"no chord quality for intervals {sorted(s)}")

    def score(q: str) -> tuple[int, int]:
        tq = T.tones(q)
        return (len(s & tq) - len(s - tq) - len(T.required(q) - s), -T.QUALITIES.index(q))
    return max(T.QUALITIES, key=score)


def harte_degree_list(text: str) -> tuple[set[int], set[int]]:
    """Parse "b7,9,*5" → (added intervals, omitted intervals)."""
    add: set[int] = set()
    omit: set[int] = set()
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("*"):
            omit.add(T.degree_semitones(tok[1:]))
        else:
            add.add(T.degree_semitones(tok))
    return add, omit


_HARTE_RE = re.compile(r"^(?P<root>[A-G][#b]*)(?::(?P<short>[^(/]*)(?:\((?P<deg>[^)]*)\))?)?(?:/(?P<bass>[#b]*\d+))?$")


def parse_harte_full(label: str, strict: bool = True) -> ParsedChord:
    s = label.strip()
    if s in ("N", "X", ""):
        return ParsedChord(-1, T.N, -1, "N")
    m = _HARTE_RE.match(s)
    if not m:
        raise ValueError(f"bad Harte label {label!r}")
    root = parse_note(m["root"])
    short = (m["short"] or "").strip()
    deg = m["deg"]
    if short == "" and deg is None:
        short = "maj"
    if short == "":
        ivs: set[int] = {0}
    elif short in SHORTHAND:
        ivs = set(SHORTHAND[short])
    else:
        raise ValueError(f"unknown Harte shorthand {short!r} in {label!r}")
    if deg is not None:
        add, omit = harte_degree_list(deg)
        ivs = (ivs | add) - omit
    canon = short + (f"({deg})" if deg is not None else "")
    if strict and canon in _BY_HARTE:
        q = _BY_HARTE[canon]
    else:
        q = quality_from_intervals(ivs, strict)
    b_iv = T.degree_semitones(m["bass"]) if m["bass"] else 0
    return ParsedChord(root.pc, q, (root.pc + b_iv) % 12, str(root), b_iv)


def parse_harte(label: str, strict: bool = True) -> tuple[int, str, int]:
    """"B:aug(b7)" → (11, "+7", 11); "C#:min7/b3" → (1, "m7", 4); "N" → (-1, "N", -1)."""
    return parse_harte_full(label, strict).triple


def to_harte(root: int, quality: str, bass: int, root_name: str | None = None) -> str:
    """(root pc, quality, bass pc) → canonical Harte. ``root_name`` sets the spelling of the root
    (defaults to C C# D Eb E F F# G Ab A Bb B)."""
    if quality == T.N or root < 0:
        return "N"
    name = root_name if root_name is not None else DEFAULT_NAMES[root % 12]
    if parse_note(name).pc != root % 12:
        raise ValueError(f"root name {name!r} is not pitch class {root}")
    out = f"{name}:{HARTE[quality]}"
    iv = (bass - root) % 12
    if iv:
        out += "/" + T.bass_degree(quality, iv)
    return out


def to_harte_in_key(root: int, quality: str, bass: int, key: int, transpose: int = 0,
                    pref: str = "auto", theoretical: bool = False, next_root: int | None = None) -> str:
    """Canonical Harte with the root spelled in the local key (the JSON ``harte`` field):
    D#7/A# in B → "D#:7/5", #1°7 in B → "C:dim7"."""
    from .spelling import spell
    if quality == T.N or root < 0:
        return "N"
    sp = spell(root, quality, bass, key, transpose, pref, theoretical, next_root)
    return to_harte((root + transpose) % 12, quality, (bass + transpose) % 12, sp.root)


# --- typed lead-sheet symbols -----------------------------------------------------------------

_Q_ALIASES: dict[str, str] = {}


def _alias(q: str, *names: str) -> None:
    for n in names:
        _Q_ALIASES[n] = q


_alias("maj", "", "maj", "M", "major", "Maj")
_alias("min", "m", "min", "mi", "-", "minor")
_alias("dim", "dim", "°", "o")
_alias("aug", "aug", "+", "#5", "+5", "(#5)")
_alias("sus2", "sus2", "2", "sus9")
_alias("sus4", "sus4", "sus")
_alias("5", "5", "(no3)", "no3")
_alias("6", "6", "maj6", "M6", "add6")
_alias("m6", "m6", "min6", "mi6", "-6")
_alias("7", "7", "dom7")
_alias("maj7", "maj7", "M7", "Δ7", "Δ", "ma7", "j7", "Maj7")
_alias("m7", "m7", "min7", "mi7", "-7")
_alias("m7b5", "m7b5", "ø", "ø7", "m7-5", "min7b5", "-7b5", "mi7b5", "m7(b5)")
_alias("dim7", "dim7", "°7", "o7")
_alias("mM7", "mM7", "mmaj7", "m(maj7)", "minmaj7", "m/maj7", "mΔ7", "mΔ", "m(Δ7)", "-Δ7",
       "m(M7)", "-maj7", "mima7")
_alias("+maj7", "+maj7", "augmaj7", "maj7#5", "maj7+5", "+M7", "+Δ7", "Δ#5", "Δ7#5",
       "maj7(#5)", "M7#5", "M7+5", "aug(maj7)")
_alias("7sus4", "7sus4", "7sus", "sus7", "7(sus4)")
_alias("+7", "+7", "aug7", "7#5", "7+5", "7+", "7(#5)", "7aug")
_alias("add9", "add9", "add2", "(add9)", "(add2)", "(9)")
_alias("madd9", "madd9", "m(add9)", "madd2", "m(add2)", "-add9", "m(9)", "min(add9)")
_alias("6/9", "6/9", "69", "6add9", "6(add9)", "6(9)")
_alias("9", "9", "dom9", "7(9)", "7add9")
_alias("maj9", "maj9", "M9", "Δ9", "maj7(9)", "Maj9")
_alias("m9", "m9", "min9", "-9", "m7(9)", "mi9")
_alias("m11", "m11", "min11", "-11", "m7(11)", "m7add11", "m9(11)", "mi11")
_alias("9sus4", "9sus4", "9sus", "11", "7sus4(9)", "sus4(9)7")
_alias("13sus4", "13sus4", "13sus")
_alias("13", "13", "7(13)", "9(13)", "dom13")
_alias("7b9", "7b9", "7(b9)", "7-9")
_alias("7#9", "7#9", "7(#9)", "7+9")
_alias("7b13", "7b13", "7(b13)", "7-13", "7b6")
_alias("maj7#11", "maj7#11", "maj7(#11)", "M7#11", "Δ7#11", "Δ#11", "maj7+11", "Maj7#11")

_SYMBOL_RE = re.compile(r"^(?P<root>[A-Ga-g](?:##|bb|[#b♯♭x𝄪𝄫])?)(?P<q>.*?)(?:/(?P<bass>[A-Ga-g](?:##|bb|[#b♯♭x𝄪𝄫])?))?$")


def _norm_quality(q: str) -> str:
    return (q.strip().replace(" ", "").replace("♯", "#").replace("♭", "b").replace("−", "-")
            .replace("º", "°").replace("˚", "°").replace("Ø", "ø"))


def parse_symbol_full(text: str) -> ParsedChord:
    """Parse a typed lead-sheet symbol ("F#/A#", "C#m7", "E/F#", "Cø", "N") into a ParsedChord."""
    s = text.strip()
    if s.upper() in ("N", "NC", "N.C.", "N.C", "X"):
        return ParsedChord(-1, T.N, -1, "N")
    m = _SYMBOL_RE.match(s)
    if not m:
        raise ValueError(f"cannot read chord {text!r}")
    root = parse_note(m["root"])
    qtext = _norm_quality(m["q"])
    if qtext not in _Q_ALIASES:
        raise ValueError(f"unknown chord quality {m['q']!r} in {text!r}")
    q = _Q_ALIASES[qtext]
    b_iv = 0
    if m["bass"]:
        b_iv = (parse_note(m["bass"]).pc - root.pc) % 12
    return ParsedChord(root.pc, q, (root.pc + b_iv) % 12, str(root), b_iv)


def parse_symbol(text: str) -> tuple[int, str, int]:
    """"F#/A#" → (6, "maj", 10); "C°7" → (0, "dim7", 0); "N" → (-1, "N", -1)."""
    return parse_symbol_full(text).triple


def parse_any(text: str) -> ParsedChord:
    """Harte if the text contains ':' (or is exactly a Harte root such as "C"), else a symbol."""
    s = text.strip()
    if ":" in s:
        return parse_harte_full(s)
    return parse_symbol_full(s)


def symbol_to_harte(text: str) -> str:
    """Typed symbol → canonical Harte, keeping the typed root spelling ("F#/A#" → "F#:maj/3")."""
    p = parse_any(text)
    if p.quality == T.N:
        return "N"
    return to_harte(p.root, p.quality, p.bass, p.root_name)


def bass_name(root_name: str, quality: str, bass_interval: int) -> str:
    """Spell a bass by interval from a written root (double accidentals simplified)."""
    root = parse_note(root_name)
    return str(simple(root.up(T.bass_degree(quality, bass_interval))))


def harte_to_symbol(label: str, notation=None) -> str:
    """Harte → lead-sheet symbol keeping the Harte root spelling ("F#:maj/3" → "F#/A#").

    Uses the Full level with every slash shown; ``notation`` is a ``simplify.Notation``.
    """
    from .simplify import Notation, quality_symbol  # local import: simplify imports spelling only
    p = parse_harte_full(label)
    if p.quality == T.N:
        return "N"
    out = p.root_name + quality_symbol(p.quality, notation or Notation())
    if p.bass_interval:
        out += "/" + bass_name(p.root_name, p.quality, p.bass_interval)
    return out


__all__ = [
    "HARTE", "SHORTHAND", "ParsedChord", "NO_CHORD", "parse_harte", "parse_harte_full", "to_harte",
    "parse_symbol", "parse_symbol_full", "parse_any", "symbol_to_harte", "harte_to_symbol",
    "quality_from_intervals", "bass_name", "to_harte_in_key",
]
