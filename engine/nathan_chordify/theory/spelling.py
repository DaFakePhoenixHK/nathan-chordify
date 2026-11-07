"""Key-aware chord spelling (DESIGN §5.12 step 7 v1.1, review A18).

Rules, applied in the local key at the segment start (``key`` is 0..23 as in
``KeyTrack.per_beat``: 0–11 major tonic pc, 12–23 minor tonic pc + 12), after transposing the
chord and the key by ``transpose`` semitones:

1. **Key names**: fewest accidentals; ties go to sharps (F# over Gb, D#m over Ebm). The
   ``pref`` toggle ("sharp" / "flat") picks the other enharmonic key when it has ≤ 7
   accidentals (C# vs Db, Gb vs F#, A#m vs Bbm ...). Tonic names B#, E#, Cb and Fb are avoided
   unless ``theoretical`` is set, so "flat" in B major still gives B.
2. **Diatonic roots** use the scale letters (natural minor for minor keys).
3. **Non-diatonic roots**, functional spelling:
   - diminished family (dim, dim7, m7b5) = leading-tone chords → sharp side (#1°7, #4ø7, #5°7);
     with context, a dim chord that resolves *down* a semitone uses the flat side;
   - with context (``next_root``), a dominant-family (or major) chord that resolves down a 5th is
     spelled a 5th above its target's letter (secondary dominants), and a dominant chord that
     resolves down a semitone one letter above its target (tritone substitution);
   - major keys: ♭II, ♭III, ♭VI, ♭VII (and anything else on those degrees) → flat side
     (borrowed chords, Neapolitan, tritone substitutes); a dominant chord on #4 → sharp side;
   - minor keys: the raised 3rd, 6th and 7th (Picardy, melodic and harmonic minor) → sharp
     side; ♭II → flat side; a dominant chord on #4 → sharp side;
   - no rule → fewest accidentals; ties → the preferred side (``pref``; "auto" = the key's side,
     sharps for C major / A minor).
   (iv in a major key has a diatonic root, so it needs no root rule.)
4. **Chord tones and the bass** are spelled by interval from the root (E/G#, not E/Ab). Non-chord
   basses use the generic degrees +2 = 2nd, +5 = 4th, +10 = minor 7th.
5. **Display fixes**: a double sharp/flat in a root, bass or tone name is replaced by its
   enharmonic (D#7/F𝄪 → D#7/G). Roots and basses avoid B#, E#, Cb and Fb unless ``theoretical``
   (#1°7 in B shows as C°7). Bass and tones are spelled from the *displayed* root, so the name
   stays internally consistent (C°7, not C°7 with B#-based tones).

Nashville / Roman numbers (numbers.py) use the *functional* spelling (before the display fixes),
which is why #1°7 in B is numbered "#1°7" although the letter label reads "C°7".
"""
from __future__ import annotations

from dataclasses import dataclass

from . import templates as T

LETTERS = "CDEFGAB"
NATURAL_PC = (0, 2, 4, 5, 7, 9, 11)
MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)
ODD_NAMES = frozenset({"B#", "E#", "Cb", "Fb"})
PREFS = ("auto", "sharp", "flat")

_ACC_IN = {"#": 1, "♯": 1, "x": 2, "𝄪": 2, "b": -1, "♭": -1, "𝄫": -2}


@dataclass(frozen=True, order=True)
class Note:
    """A spelled note name: letter index (0 = C .. 6 = B) and accidental (−2..+2)."""
    letter: int
    acc: int

    @property
    def pc(self) -> int:
        return (NATURAL_PC[self.letter] + self.acc) % 12

    def __str__(self) -> str:
        return LETTERS[self.letter] + ("#" * self.acc if self.acc > 0 else "b" * -self.acc)

    def up(self, degree: str) -> Note:
        """The note a Harte degree above this one ("3" of B is D#, "b7" of F# is E)."""
        return note_at(self.letter + T.degree_step(degree), self.pc + T.degree_semitones(degree))


def note_at(letter: int, pc: int) -> Note:
    """The note with this letter that sounds pitch class ``pc`` (accidental may be large)."""
    letter %= 7
    acc = ((pc - NATURAL_PC[letter] + 6) % 12) - 6
    return Note(letter, acc)


def parse_note(name: str) -> Note:
    """Parse "C", "f#", "Bb", "C##", "Cx", "Ebb", "C♯", "B♭" into a Note."""
    s = name.strip()
    if not s or s[0].upper() not in LETTERS:
        raise ValueError(f"bad note name {name!r}")
    letter = LETTERS.index(s[0].upper())
    acc = 0
    for ch in s[1:]:
        if ch not in _ACC_IN:
            raise ValueError(f"bad note name {name!r}")
        acc += _ACC_IN[ch]
    if abs(acc) > 2:
        raise ValueError(f"too many accidentals in {name!r}")
    return Note(letter, acc)


def pc_of(name: str) -> int:
    return parse_note(name).pc


def simple(n: Note) -> Note:
    """Replace a double sharp / double flat by its enharmonic (F## → G, Bbb → A, E## → F#)."""
    if abs(n.acc) < 2:
        return n
    step = 1 if n.acc > 0 else -1
    for k in (1, 2):
        cand = note_at(n.letter + step * k, n.pc)
        if abs(cand.acc) <= 1:
            return cand
    return n  # unreachable for |acc| == 2


def display(n: Note, theoretical: bool = False) -> Note:
    """Display form of a root or bass: no double accidentals; no B#/E#/Cb/Fb unless theoretical."""
    n = simple(n)
    if not theoretical and str(n) in ODD_NAMES:
        n = note_at(n.letter + (1 if n.acc > 0 else -1), n.pc)
    return n


# ---------------------------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------------------------

def key_tonic_pc(key: int) -> int:
    return key % 12


def key_is_minor(key: int) -> bool:
    if not 0 <= key <= 23:
        raise ValueError(f"key must be 0..23, got {key}")
    return key >= 12


def scale_notes(tonic: Note, minor: bool) -> tuple[Note, ...]:
    steps = MINOR_STEPS if minor else MAJOR_STEPS
    return tuple(note_at(tonic.letter + i, tonic.pc + s) for i, s in enumerate(steps))


def key_signature(tonic: Note, minor: bool) -> int:
    """Sharps (> 0) or flats (< 0) in the key signature (theoretical keys exceed ±7)."""
    major_tonic = note_at(tonic.letter + 2, tonic.pc + 3) if minor else tonic
    return sum(n.acc for n in scale_notes(major_tonic, False))


def tonic_note(pc: int, minor: bool, pref: str = "auto", theoretical: bool = False) -> Note:
    """Spell a key's tonic: fewest accidentals, ties to sharps; ``pref`` picks a valid enharmonic."""
    if pref not in PREFS:
        raise ValueError(f"pref must be one of {PREFS}")
    cands = [note_at(L, pc) for L in range(7)]
    cands = [n for n in cands if abs(n.acc) <= 1]
    valid = [n for n in cands if abs(key_signature(n, minor)) <= 7
             and (theoretical or str(n) not in ODD_NAMES)] or cands

    def best(pool: list[Note]) -> Note:
        return min(pool, key=lambda n: (abs(key_signature(n, minor)), key_signature(n, minor) < 0))

    if pref == "sharp":
        pool = [n for n in valid if key_signature(n, minor) >= 0]
        if pool:
            return best(pool)
    elif pref == "flat":
        pool = [n for n in valid if key_signature(n, minor) <= 0]
        if pool:
            return best(pool)
    return best(valid)


@dataclass(frozen=True)
class KeyContext:
    tonic: Note
    minor: bool
    scale: tuple[Note, ...]
    signature: int
    prefer_sharps: bool       # tie-break side for "no rule" roots

    def note_for(self, pc: int) -> Note | None:
        pc %= 12
        for n in self.scale:
            if n.pc == pc:
                return n
        return None

    @property
    def name(self) -> str:
        return str(self.tonic) + ("m" if self.minor else "")


def key_context(key: int, transpose: int = 0, pref: str = "auto", theoretical: bool = False) -> KeyContext:
    minor = key_is_minor(key)
    pc = (key_tonic_pc(key) + transpose) % 12
    tonic = tonic_note(pc, minor, pref, theoretical)
    sig = key_signature(tonic, minor)
    prefer = {"sharp": True, "flat": False}.get(pref, sig >= 0)
    return KeyContext(tonic, minor, scale_notes(tonic, minor), sig, prefer)


def key_name(key: int, transpose: int = 0, pref: str = "auto", theoretical: bool = False) -> str:
    """"B", "Db", "G#m" ... for a key index 0..23."""
    return key_context(key, transpose, pref, theoretical).name


# ---------------------------------------------------------------------------------------------
# Chord roots
# ---------------------------------------------------------------------------------------------

def root_note(pc: int, quality: str, ctx: KeyContext, next_pc: int | None = None) -> Note:
    """Functional spelling of a chord root (before the display fixes), per the module rules."""
    pc %= 12
    n = ctx.note_for(pc)
    if n is not None:
        return n
    below = ctx.note_for(pc - 1)
    above = ctx.note_for(pc + 1)
    assert below is not None and above is not None  # both scales have only whole/half steps
    sharp = note_at(below.letter, pc)
    flat = note_at(above.letter, pc)
    q = quality

    if next_pc is not None:
        nxt = next_pc % 12
        if q in T.DIM_FAMILY:
            if nxt == (pc + 1) % 12:
                return sharp
            if nxt == (pc - 1) % 12:
                return flat
        if (q in T.DOMINANT or q == "maj") and nxt == (pc + 5) % 12:
            target = root_note(nxt, "maj", ctx)
            return note_at(target.letter + 4, pc)          # a 5th above the target's letter
        if q in T.DOMINANT and nxt == (pc - 1) % 12:
            target = root_note(nxt, "maj", ctx)
            return note_at(target.letter + 1, pc)          # tritone sub: one letter above

    if q in T.DIM_FAMILY:
        return sharp
    rel = (pc - ctx.tonic.pc) % 12
    if ctx.minor:
        if rel in (4, 9, 11):
            return sharp
        if rel == 1:
            return flat
    else:
        if rel in (1, 3, 8, 10):
            return flat
    if rel == 6 and q in T.DOMINANT:
        return sharp
    if abs(sharp.acc) != abs(flat.acc):
        return sharp if abs(sharp.acc) < abs(flat.acc) else flat
    return sharp if ctx.prefer_sharps else flat


def is_diatonic_root(pc: int, ctx: KeyContext) -> bool:
    return ctx.note_for(pc) is not None


# ---------------------------------------------------------------------------------------------
# Whole chords
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Spelled:
    root: str                 # display root ("C" for B#°7 in B)
    bass: str                 # display bass (== root in root position)
    quality: str
    tones: tuple[str, ...]    # display names of the template tones, by interval (for diagrams)
    key: str                  # display key name ("B", "Db", "G#m")
    root_note: Note | None = None
    bass_note: Note | None = None
    func_root: Note | None = None   # functional spelling (B#), used by numbers.py
    func_bass: Note | None = None

    @property
    def slash(self) -> bool:
        return self.bass != self.root


def tone_names(root: Note, quality: str) -> tuple[str, ...]:
    """Template tones spelled by interval from ``root``, double accidentals simplified."""
    return tuple(str(simple(root.up(t.degree))) for t in T.TEMPLATE[quality])


def name_pc(root: Note, quality: str, pc: int) -> str:
    """Name any pitch class relative to a spelled chord root (chord tones by their degree)."""
    deg = T.bass_degree(quality, (pc - root.pc) % 12)
    return str(simple(root.up(deg)))


def spell(root: int, quality: str, bass: int, key: int, transpose: int = 0, pref: str = "auto",
          theoretical: bool = False, next_root: int | None = None) -> Spelled:
    """Spell a (root, quality, bass) chord in a local key, after transposing by ``transpose``.

    ``next_root`` (untransposed pc of the following chord, optional) enables the
    secondary-dominant, tritone-substitution and descending-diminished context rules.
    """
    ctx = key_context(key, transpose, pref, theoretical)
    if quality == T.N or root < 0:
        return Spelled("N", "N", T.N, (), ctx.name)
    t = transpose
    r = (root + t) % 12
    b = (bass + t) % 12
    nr = None if next_root is None or next_root < 0 else (next_root + t) % 12
    func_root = root_note(r, quality, ctx, nr)
    disp_root = display(func_root, theoretical)
    deg = T.bass_degree(quality, (b - r) % 12)
    func_bass = func_root.up(deg)
    disp_bass = display(disp_root.up(deg), theoretical)
    return Spelled(str(disp_root), str(disp_bass), quality, tone_names(disp_root, quality), ctx.name,
                   disp_root, disp_bass, func_root, func_bass)


def spell_candidate(c, key: int, transpose: int = 0, pref: str = "auto", theoretical: bool = False,
                    next_root: int | None = None) -> Spelled:
    """``spell`` for a ``types.Candidate`` (or any object with root/quality/bass)."""
    return spell(c.root, c.quality, c.bass, key, transpose, pref, theoretical, next_root)
