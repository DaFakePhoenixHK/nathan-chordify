"""Simplify levels, slash visibility and chord-label strings (DESIGN §6.1, §6.3, §5.12 steps 5, 9).

``display_chord`` turns an internal chord into the chord to *show* at a level:

- the §6.1 quality table (Simple / Standard / Full);
- **upper-structure display** (piano default ON): 9sus4 → the major triad on the ♭7 over the
  root (G9sus4 → F/G, F#9sus4 → E/F#), 13sus4 → the maj7 on the ♭7 over the root
  (F#13sus4 → Emaj7/F#). It applies to root-position 9sus4/13sus4 at Standard and Full only
  (an inverted G9sus4/C would lose its root as "F/C", so it keeps its own name). At Simple the
  chord shows as a major triad on its own root (F/G → G). With the toggle OFF (or
  ``slash=False``) the chords keep their names: Full 9sus4 / 13sus4, Standard 7sus4;
- **7th only in the bass** (review A9, §5.12 step 5): when the bass is the chord's 7th and the
  upper evidence for it is weak (``seventh_in_upper=False``), Standard and Full show the triad
  over the bass (Cmaj7/B → C/B, AmM7/G# → Am/G#; 9 → add9, 13 → 6/9);
- **bass per level**: Simple never shows a bass; Standard shows inversions (the bass is a
  non-extension chord tone) and upper structures; Full shows every slash, including non-chord
  basses (Em/A, E/F#). ``slash=True`` shows every slash at every level, ``slash=False`` hides
  them all (and with them the upper-structure form).

``label`` adds key-aware spelling (spelling.py) and the notation toggles of §6.3.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import spelling as S
from . import templates as T

LEVELS = ("simple", "standard", "full")

_SIMPLE = {
    "maj": "maj", "min": "min", "dim": "dim", "aug": "maj", "sus2": "maj", "sus4": "maj",
    "5": "maj", "6": "maj", "m6": "min", "7": "maj", "maj7": "maj", "m7": "min",
    "m7b5": "dim", "dim7": "dim", "mM7": "min", "+maj7": "maj", "7sus4": "maj", "+7": "maj",
    "add9": "maj", "madd9": "min", "6/9": "maj", "9": "maj", "maj9": "maj", "m9": "min",
    "m11": "min", "9sus4": "maj", "13sus4": "maj", "13": "maj", "7b9": "maj", "7#9": "maj",
    "7b13": "maj", "maj7#11": "maj",
}
_STANDARD = {q: q for q in T.QUALITIES}
_STANDARD.update({
    "+maj7": "aug", "+7": "aug", "6/9": "6", "9": "7", "maj9": "maj7", "m9": "m7", "m11": "m7",
    "maj7#11": "maj7", "13": "7", "7b9": "7", "7#9": "7", "7b13": "7",
    "9sus4": "7sus4", "13sus4": "7sus4",          # only when not shown as an upper structure
})
_FULL = {q: q for q in T.QUALITIES}

#: level → internal quality → displayed quality (§6.1).
LEVEL_MAP: dict[str, dict[str, str]] = {"simple": _SIMPLE, "standard": _STANDARD, "full": _FULL}
for _m in LEVEL_MAP.values():
    assert set(_m) == set(T.QUALITIES)

#: §5.12 step 5: the chord to show when its 7th sounds only in the bass.
SEVENTH_ONLY_IN_BASS: dict[str, str] = {
    "7": "maj", "maj7": "maj", "m7": "min", "mM7": "min", "m7b5": "dim", "+7": "aug",
    "+maj7": "aug", "7sus4": "sus4", "9": "add9", "maj9": "add9", "m9": "madd9",
    "m11": "madd9", "13": "6/9", "7b9": "maj", "7#9": "maj", "7b13": "maj", "maj7#11": "maj",
    "9sus4": "sus4", "13sus4": "sus4",
}

UPPER_STRUCTURE: dict[str, str] = {"9sus4": "maj", "13sus4": "maj7"}   # upper chord on the ♭7


@dataclass(frozen=True)
class Notation:
    """§6.3 notation toggles."""
    maj7: str = "maj7"        # "maj7" | "Δ7" | "M7"
    symbols: bool = True      # True: ° and ø; False: dim and m7b5

    def __post_init__(self) -> None:
        if self.maj7 not in ("maj7", "Δ7", "M7"):
            raise ValueError("maj7 notation must be 'maj7', 'Δ7' or 'M7'")


DEFAULT_NOTATION = Notation()


def quality_symbol(q: str, notation: Notation = DEFAULT_NOTATION) -> str:
    """Lead-sheet suffix for an internal (or displayed) quality: "m7", "°7", "maj7", "+7" ..."""
    M = notation.maj7
    M9 = M.replace("7", "9")
    sym = notation.symbols
    table = {
        "maj": "", "min": "m", "dim": "°" if sym else "dim", "aug": "+", "sus2": "sus2",
        "sus4": "sus4", "5": "5", "6": "6", "m6": "m6", "7": "7", "maj7": M, "m7": "m7",
        "m7b5": "ø7" if sym else "m7b5", "dim7": "°7" if sym else "dim7", "mM7": f"m({M})",
        "+maj7": f"+{M}", "7sus4": "7sus4", "+7": "+7", "add9": "add9", "madd9": "m(add9)",
        "6/9": "6/9", "9": "9", "maj9": M9, "m9": "m9", "m11": "m11", "9sus4": "9sus4",
        "13sus4": "13sus4", "13": "13", "7b9": "7b9", "7#9": "7#9", "7b13": "7b13",
        "maj7#11": f"{M}#11", T.N: "",
    }
    return table[q]


@dataclass(frozen=True)
class DisplayChord:
    root: int                 # pc of the displayed root (the ♭7 for an upper structure)
    quality: str              # displayed quality
    bass: int | None          # pc, or None when no slash is shown
    kind: str                 # "root" | "inversion" | "nonchord" | "upper" | "none"
    root_degree: str = "1"    # displayed root relative to the chord root ("b7" for upper structures)


def bass_kind(quality: str, interval: int) -> str:
    """"root", "inversion" (non-extension chord tone) or "nonchord" (+2/+5/+10, or an extension)."""
    interval %= 12
    if interval == 0:
        return "root"
    return "inversion" if interval in T.chord_bass_intervals(quality) else "nonchord"


def seventh_interval(quality: str) -> int | None:
    """Interval of the chord's 7th (b7 or 7; not the bb7 of dim7), or None."""
    for t in T.TEMPLATE[quality]:
        if t.degree in ("7", "b7"):
            return t.interval
    return None


def display_chord(root: int, quality: str, bass: int, level: str = "full", slash: bool | None = None,
                  upper_structure: bool = True, seventh_in_upper: bool = True) -> DisplayChord:
    """The chord to show for an internal (root, quality, bass) at a level (see module doc)."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    if quality == T.N or root < 0:
        return DisplayChord(-1, T.N, None, "none")
    iv = (bass - root) % 12
    if (upper_structure and slash is not False and level != "simple"
            and quality in UPPER_STRUCTURE and iv == 0):
        return DisplayChord((root + 10) % 12, UPPER_STRUCTURE[quality], root, "upper", "b7")
    kind = bass_kind(quality, iv)
    q = quality
    if (not seventh_in_upper and level != "simple" and iv != 0
            and iv == seventh_interval(quality)):
        q = SEVENTH_ONLY_IN_BASS[quality]
    dq = LEVEL_MAP[level][q]
    if slash is True:
        show = kind != "root"
    elif slash is False:
        show = False
    else:
        show = {"simple": False, "standard": kind == "inversion", "full": kind != "root"}[level]
    return DisplayChord(root, dq, bass if show else None, kind)


def label(root: int, quality: str, bass: int, key: int, transpose: int = 0, pref: str = "auto",
          level: str = "full", slash: bool | None = None, upper_structure: bool = True,
          seventh_in_upper: bool = True, notation: Notation = DEFAULT_NOTATION,
          theoretical: bool = False, next_root: int | None = None) -> str:
    """Spelled label at a level: "F#/A#", "B+7", "C°7", "E/F#" (for F#9sus4), "N"."""
    dc = display_chord(root, quality, bass, level, slash, upper_structure, seventh_in_upper)
    if dc.quality == T.N:
        return "N"
    sp = S.spell(root, quality, bass, key, transpose, pref, theoretical, next_root)
    assert sp.root_note is not None
    if dc.kind == "upper":
        root_name = str(S.display(sp.root_note.up(dc.root_degree), theoretical))
        bass_name = sp.root
    else:
        root_name, bass_name = sp.root, sp.bass
    out = root_name + quality_symbol(dc.quality, notation)
    if dc.bass is not None and bass_name != root_name:
        out += "/" + bass_name
    return out


def labels(root: int, quality: str, bass: int, key: int, **kw) -> dict[str, str]:
    """{"simple": ..., "standard": ..., "full": ...} as in the analysis JSON ``label`` field."""
    return {lv: label(root, quality, bass, key, level=lv, **kw) for lv in LEVELS}


def label_candidate(c, key: int, **kw) -> str:
    """``label`` for a ``types.Candidate``."""
    return label(c.root, c.quality, c.bass, key, **kw)
