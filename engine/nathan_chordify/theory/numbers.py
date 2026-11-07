"""Nashville (級數) numbers and Roman numerals (DESIGN §5.12 step 8).

Both are computed from the *functional* spelling (spelling.py) of the *displayed* chord
(simplify.py), relative to the local key; numbering restarts at every key change simply because
the caller passes the local key of each chord. The spelling is done in the key moved to C major /
A minor, so degrees never change with transposition (§5.12 step 12).

**Nashville / 級數** (examples from §5.12): "1", "5/7", "6m", "4maj7", "b7(7)", "3(7)/7",
"#1°7", "1+(7)". The suffix is the lead-sheet suffix; when it starts with a digit it is put in
parentheses so it cannot be read as part of the degree ("b7(7)", "1(6)", "5(7sus4)"), and after
"+" likewise ("1+(7)", "1+(maj7)"). The bass is the key degree of the bass note ("5/7").
Minor keys are numbered from their **relative major** (A minor: Am = "6m", E7 = "3(7)",
G#°7 = "#5°7"), as in jianpu / 簡譜 practice and §6.4.

**Roman**: case follows the quality (lowercase for minor and diminished chords), degrees are
measured against the major scale of the local tonic (minor keys use their own tonic: i, bIII,
bVI, bVII, V7). Inversions of triads and seventh chords use figured bass (triads 6, 6/4; seventh
chords 6/5, 4/3, 4/2, replacing the 7: "V6/5", "IVmaj4/2", "viiø6/5"). Every other slash
(non-chord bass, upper structures, inversions of sus/6/add9/... chords) is written with the
bass's scale degree: "IV/5̂". Added-sixth chords are written "add6" ("Iadd6", "ivadd6") so they
cannot be confused with a first-inversion "I6".
"""
from __future__ import annotations

from . import simplify as SM
from . import spelling as S
from . import templates as T
from .spelling import Note

MAJOR = (0, 2, 4, 5, 7, 9, 11)
ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII")
HAT = "̂"   # combining circumflex: scale degree 5̂


def degree_str(note: Note, ref: Note) -> str:
    """Degree of a spelled note relative to a major-scale tonic: "1", "b7", "#1", "#5"."""
    num = (note.letter - ref.letter) % 7
    expected = (ref.pc + MAJOR[num]) % 12
    diff = ((note.pc - expected + 6) % 12) - 6
    acc = "#" * diff if diff > 0 else "b" * -diff
    return f"{acc}{num + 1}"


def nashville_suffix(q: str, notation: SM.Notation = SM.DEFAULT_NOTATION) -> str:
    sym = SM.quality_symbol(q, notation)
    if sym.startswith("+") and len(sym) > 1:
        return "+(" + sym[1:] + ")"
    if sym[:1].isdigit():
        return "(" + sym + ")"
    return sym


def _functional(root: int, quality: str, bass: int, key: int, level: str, slash: bool | None,
                upper_structure: bool, seventh_in_upper: bool, next_root: int | None):
    """Functional spelling of the displayed chord in the key moved to C major / A minor.

    Degrees must not change with transposition (§5.12 step 12), so the chord is spelled in the
    same-mode key without accidentals; the spelling rules are all relative to the tonic, and this
    fixes the only absolute choice (the "no rule" tie-break between e.g. #4 and b5).
    """
    dc = SM.display_chord(root, quality, bass, level, slash, upper_structure, seventh_in_upper)
    if dc.quality == T.N:
        return dc, None, None, None
    ref_key = 21 if S.key_is_minor(key) else 0
    shift = (ref_key - key) % 12
    root, bass = (root + shift) % 12, (bass + shift) % 12
    next_root = None if next_root is None or next_root < 0 else (next_root + shift) % 12
    ctx = S.key_context(ref_key)
    froot = S.root_note(root, quality, ctx, next_root)
    droot = froot.up(dc.root_degree)
    fbass = None
    if dc.bass is not None:
        fbass = froot if dc.kind == "upper" else froot.up(T.bass_degree(quality, (bass - root) % 12))
    return dc, ctx, droot, fbass


def nashville(root: int, quality: str, bass: int, key: int, level: str = "full",
              slash: bool | None = None, upper_structure: bool = True, seventh_in_upper: bool = True,
              notation: SM.Notation = SM.DEFAULT_NOTATION, next_root: int | None = None) -> str:
    """Nashville / 級數 number of a chord in a local key (0..23)."""
    dc, ctx, droot, fbass = _functional(root, quality, bass, key, level, slash, upper_structure,
                                        seventh_in_upper, next_root)
    if ctx is None:
        return "N"
    ref = ctx.tonic.up("b3") if ctx.minor else ctx.tonic
    out = degree_str(droot, ref) + nashville_suffix(dc.quality, notation)
    if fbass is not None:
        out += "/" + degree_str(fbass, ref)
    return out


def _roman_suffix(q: str, notation: SM.Notation) -> str:
    M = notation.maj7
    sym = notation.symbols
    table = {
        "maj": "", "min": "", "dim": "°" if sym else "dim", "aug": "+", "sus2": "sus2",
        "sus4": "sus4", "5": "5", "6": "add6", "m6": "add6", "7": "7", "maj7": M, "m7": "7",
        "m7b5": "ø7" if sym else "m7b5", "dim7": "°7" if sym else "dim7", "mM7": f"({M})",
        "+maj7": f"+{M}", "7sus4": "7sus4", "+7": "+7", "add9": "add9", "madd9": "add9",
        "6/9": "6/9", "9": "9", "maj9": M.replace("7", "9"), "m9": "9", "m11": "11",
        "9sus4": "9sus4", "13sus4": "13sus4", "13": "13", "7b9": "7b9", "7#9": "7#9",
        "7b13": "7b13", "maj7#11": f"{M}#11",
    }
    return table[q]


_TRIAD_MARK = frozenset({"maj", "min", "dim", "aug"})
_SEVENTHS = frozenset({"7", "maj7", "m7", "m7b5", "dim7", "mM7", "+7", "+maj7"})


def _seventh_mark(q: str, notation: SM.Notation) -> str:
    """The suffix of a seventh chord without its 7: '' (V7, ii7), 'maj', 'ø', '°', '+' ..."""
    M = notation.maj7[:-1]  # "maj" | "Δ" | "M"
    sym = notation.symbols
    return {"7": "", "m7": "", "maj7": M, "m7b5": "ø" if sym else "m7b5",
            "dim7": "°" if sym else "dim", "mM7": f"({M})", "+7": "+", "+maj7": f"+{M}"}[q]


def roman(root: int, quality: str, bass: int, key: int, level: str = "full",
          slash: bool | None = None, upper_structure: bool = True, seventh_in_upper: bool = True,
          notation: SM.Notation = SM.DEFAULT_NOTATION, next_root: int | None = None) -> str:
    """Roman numeral with figured-bass inversions in a local key (0..23)."""
    dc, ctx, droot, fbass = _functional(root, quality, bass, key, level, slash, upper_structure,
                                        seventh_in_upper, next_root)
    if ctx is None:
        return "N"
    ref = ctx.tonic
    deg = degree_str(droot, ref)
    acc, num = deg[:-1], int(deg[-1])
    numeral = ROMAN[num - 1]
    q = dc.quality
    if q in T.LOWER_CASE:
        numeral = numeral.lower()
    head = acc + numeral
    if fbass is None:
        return head + _roman_suffix(q, notation)
    tn = T.tone(q, (fbass.pc - droot.pc) % 12) if dc.kind == "inversion" else None
    if tn is not None:
        role = tn.degree.lstrip("b#")
        if q in _TRIAD_MARK and role in ("3", "5"):
            return head + _figure_triad(q, role, notation)
        if q in _SEVENTHS and role in ("3", "5", "7"):
            fig = {"3": "6/5", "5": "4/3", "7": "4/2"}[role]
            return head + _seventh_mark(q, notation) + fig
    return head + _roman_suffix(q, notation) + "/" + degree_str(fbass, ref) + HAT


def _figure_triad(q: str, role: str, notation: SM.Notation) -> str:
    mark = {"maj": "", "min": "", "dim": "°" if notation.symbols else "dim", "aug": "+"}[q]
    return mark + ("6" if role == "3" else "6/4")


def degrees(root: int, quality: str, bass: int, key: int, **kw) -> dict:
    """The analysis-JSON ``degree`` object: root/bass semitones above the local tonic plus the
    Nashville and Roman strings (``kw`` as for ``nashville``)."""
    if quality == T.N or root < 0:
        return {"root": None, "bass": None, "nashville": "N", "roman": "N"}
    tonic = S.key_tonic_pc(key)
    return {"root": (root - tonic) % 12, "bass": (bass - tonic) % 12,
            "nashville": nashville(root, quality, bass, key, **kw),
            "roman": roman(root, quality, bass, key, **kw)}
