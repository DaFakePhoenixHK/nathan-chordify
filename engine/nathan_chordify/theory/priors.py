"""Key and idiom priors (DESIGN §5.10 table v1.1, review A10) and idiom trigrams (§6.4).

``prior(cp, c, key)`` scores a transition cp → c in nats. Positive rules **stack** (every
matching rule adds) and the sum is capped at ``cap`` (``DecoderParams.prior_cap`` = +3.0). The
two penalties apply only when no positive rule matched (they add up: −2.5 when both hold).

``trigram(l, p, r, key)`` returns +1.5 when (l, p, r) matches an idiom of §6.4, plus +1.5 when
bass(l) → bass(p) → bass(r) is monotone with steps of 1–2 semitones.

Degrees are semitones above the tonic of the local key. **Minor keys are mapped through their
relative major** for every degree-based rule (as §6.4 prescribes for trigrams), so in A minor
F → G → Am is the 4 → 5 → 6m walk-up. "Diatonic" for a minor key is the natural minor plus its
raised leading tone (so V and V7 are diatonic and i → V is not penalised).

Families (by the §5.11 core quality, which strips 9/11/13 extensions):
- IV-family / ♭VII-family / ♭VI: core ∈ {maj, maj7, 7, 6} on degree 5 / 10 / 8;
- iv-family: core ∈ {min, m6, m7} on degree 5;
- ♭VII as a target: core ∈ {maj, 7}; I as a target: core ∈ {maj, maj7, 6} on degree 0;
- dominant family: 7, 9, 13, 7b9, 7#9, 7b13, +7, and a major triad acting as V
  (i.e. any maj triad whose root moves up a 4th).
The augmented rule is literal to the table: I → I+ / I+7 / I+maj7, and I+ family → IV-family.
The passing-diminished trigram also accepts a major chord (or a bass) a whole step above x,
because x → #x°7 → (x+2) works the same over a major target (IV → #iv°7 → V).
Chords may be ``types.Candidate`` objects or (root, quality, bass) triples.
"""
from __future__ import annotations

from . import templates as T

MAJOR = frozenset((0, 2, 4, 5, 7, 9, 11))

_MAJORISH = frozenset({"maj", "maj7", "7", "6"})
_IV_CORE = _MAJORISH
_iv_CORE = frozenset({"min", "m6", "m7"})
_I_CORE = frozenset({"maj", "maj7", "6"})
_bVII_TARGET = frozenset({"maj", "7"})
_SUS_TARGET = frozenset({"maj", "7", "9", "13"})
_M7S = frozenset({"m7", "m9", "m11"})
_LINE_MINOR = ("min", "mM7", "m7", "m6")
_LINE_MAJOR = ("maj", "aug", "6", "7")
IDIOM_BONUS = 1.5
BASS_LINE_BONUS = 1.5


def _rqb(c) -> tuple[int, str, int]:
    if isinstance(c, tuple):
        return c
    return (c.root, c.quality, c.bass)


def ref_tonic(key: int) -> int:
    """Major-key reference tonic: the tonic, or the relative major's tonic for a minor key."""
    return key % 12 if key < 12 else (key % 12 + 3) % 12


def diatonic_set(key: int) -> frozenset[int]:
    if key < 12:
        return frozenset((key + i) % 12 for i in MAJOR)
    t = key % 12
    return frozenset((t + i) % 12 for i in (0, 2, 3, 5, 7, 8, 10, 11))


def is_diatonic(c, key: int) -> bool:
    r, q, b = _rqb(c)
    if q == T.N:
        return False
    pcs = {(r + i) % 12 for i in T.tones(q)} | {b % 12}
    return pcs <= diatonic_set(key)


def _deg(pc: int, key: int) -> int:
    return (pc - ref_tonic(key)) % 12


def _core(q: str) -> str:
    return T.CORE.get(q, q)


def _signed(a: int, b: int) -> int:
    """Shortest signed distance from pc a to pc b (−5..6)."""
    d = (b - a) % 12
    return d - 12 if d > 6 else d


def _is(c, key: int, deg: int, cores: frozenset[str]) -> bool:
    r, q, _ = c
    return q != T.N and _deg(r, key) == deg and _core(q) in cores


def _iv5_type(c) -> bool:
    r, q, b = c
    return q in ("9sus4", "13sus4") or (q in ("maj", "maj7") and (b - r) % 12 == 2)


def prior(cp, c, key: int, cap: float = 3.0) -> float:
    """§5.10 transition prior cp → c in the local key (0..23), stacked and capped (nats)."""
    a, b = _rqb(cp), _rqb(c)
    if a[1] == T.N or b[1] == T.N or a == b:
        return 0.0
    r1, q1, b1 = a
    r2, q2, b2 = b
    d1 = _deg(r1, key)
    bonus = 0.0
    if is_diatonic(a, key) and is_diatonic(b, key):
        bonus += 1.0
    if 1 <= abs(_signed(b1, b2)) <= 2 and (b1 != r1 or b2 != r2):
        bonus += 1.5                                       # stepwise bass with a slash chord
    if (q1 in T.DOMINANT or q1 == "maj") and r2 == (r1 + 5) % 12:
        bonus += 1.5                                       # (secondary) dominant → target
    if q1 in T.DOMINANT and r1 == (r2 + 1) % 12:
        bonus += 1.5                                       # tritone substitution
    if q1 in T.SUS4_FAMILY and r1 == r2 and q2 in _SUS_TARGET:
        bonus += 2.0                                       # sus resolution
    if _iv5_type(a) and r2 == (b1 + 5) % 12:
        bonus += 1.5                                       # IV/5 → I
    if q1 in T.DIM_FAMILY and r2 == (r1 + 1) % 12:
        bonus += 1.5                                       # leading-tone / passing diminished
    if ((q1 == "maj" and d1 == 0 and q2 in T.AUG_FAMILY and r2 == r1)
            or (q1 in T.AUG_FAMILY and d1 == 0 and _is(b, key, 5, _IV_CORE))):
        bonus += 1.0                                       # I → I+(7) → IV
    if (_is(a, key, 5, _IV_CORE) or _is(a, key, 10, _IV_CORE)) and _is(b, key, 5, _iv_CORE):
        bonus += 1.0                                       # borrowed iv (incl. ♭VII7 → iv)
    if _is(b, key, 10, _bVII_TARGET) and (_is(a, key, 5, _IV_CORE) or _is(a, key, 8, _MAJORISH)):
        bonus += 0.8                                       # IV / ♭VI → ♭VII
    if _is(b, key, 0, _I_CORE) and _is(a, key, 10, _IV_CORE):
        bonus += 0.8                                       # ♭VII → I
    if q1 in _M7S and q2 in _M7S and r2 == (r1 - 1) % 12:
        bonus += 1.0                                       # chromatic descending m7
    if r1 == r2 and b1 == b2 and _line_step(q1, q2):
        bonus += 1.0                                       # line cliché
    if bonus > 0:
        return min(bonus, cap)
    pen = 0.0
    if not is_diatonic(b, key):
        pen -= 1.0
    if q1 == q2 and abs(_signed(r1, r2)) == 1:
        pen -= 1.5
    return pen


def _line_step(q1: str, q2: str) -> bool:
    for chain in (_LINE_MINOR, _LINE_MAJOR):
        if q1 in chain and q2 in chain and chain.index(q2) == chain.index(q1) + 1:
            return True
    return False


# ---------------------------------------------------------------------------------------------
# §6.4 trigrams
# ---------------------------------------------------------------------------------------------

_MAJ = frozenset({"maj", "maj7", "6"})          # "1", "4" ... as major chords (by core)
_DOM_OR_MAJ = frozenset({"maj", "7"})           # "b7(7)", "5", "3(7)"
_MIN = frozenset({"min", "m7", "m6"})           # "6m", "2m", "4m(6/7)" (by core)


def _d(c, key):  # degree of root
    return _deg(c[0], key)


def _bd(c, key):  # degree of bass
    return _deg(c[2], key)


def _maj(c, deg, key, bass_deg=None, cores=_MAJ):
    return _is(c, key, deg, cores) and (bass_deg is None or _bd(c, key) == bass_deg)


def _min(c, deg, key, bass_deg=None):
    return _is(c, key, deg, _MIN) and (bass_deg is None or _bd(c, key) == bass_deg)


def _idiom(l, p, r, key) -> str | None:
    """Name of the first §6.4 idiom that (l, p, r) matches, or None."""
    lq, pq, rq = l[1], p[1], r[1]
    # augmented passing: 1 → 1+(7) → 4(maj7)
    if (_maj(l, 0, key) and _d(p, key) == 0 and pq in ("aug", "+7")
            and _is(r, key, 5, frozenset({"maj", "maj7"}))):
        return "augmented passing"
    # borrowed: 4maj7 → b7(7) → 4m(6/7)
    if (_is(l, key, 5, frozenset({"maj", "maj7"})) and _is(p, key, 10, _DOM_OR_MAJ)
            and _min(r, 5, key)):
        return "borrowed"
    # borrowed ♭VI–♭VII: b6 → b7 → 1
    if _maj(l, 8, key, cores=_MAJORISH) and _is(p, key, 10, _MAJORISH) and _maj(r, 0, key):
        return "bVI-bVII-I"
    # descending Canon: 1 → 5/7 → 6m; 6m → 6m/5 → 4; 4 → 1/3 → 2m
    if _maj(l, 0, key) and _maj(p, 7, key, 11, _DOM_OR_MAJ | _MAJ) and _min(r, 9, key):
        return "descending canon"
    if _min(l, 9, key) and _min(p, 9, key, 7) and _maj(r, 5, key):
        return "descending canon"
    if _maj(l, 5, key) and _maj(p, 0, key, 4) and _min(r, 2, key):
        return "descending canon"
    # descending chromatic bass under the tonic: 1 → 1/7 → 1/b7 → 4/6
    tonic_any = frozenset({"maj", "maj7", "7", "6"})
    if (_maj(l, 0, key, 0, tonic_any) and _maj(p, 0, key, 11, tonic_any)
            and _maj(r, 0, key, 10, tonic_any)):
        return "chromatic bass"
    if (_maj(l, 0, key, 11, tonic_any) and _maj(p, 0, key, 10, tonic_any)
            and _maj(r, 5, key, 9)):
        return "chromatic bass"
    # walk-up: 2m → 1/3 → 4; 4 → 5 → 6m; 5/7 → 1
    if _min(l, 2, key) and _maj(p, 0, key, 4) and _maj(r, 5, key):
        return "walk-up"
    if _maj(l, 5, key) and _maj(p, 7, key, None, _DOM_OR_MAJ | _MAJ) and _min(r, 9, key):
        return "walk-up"
    if _maj(p, 7, key, 11, _DOM_OR_MAJ | _MAJ) and _maj(r, 0, key):
        return "walk-up"
    # rising minor 7ths: 2m7 → 3m7 → 4; 6m7 → 7m7 → x
    if _min(l, 2, key) and _min(p, 4, key) and _maj(r, 5, key):
        return "rising m7"
    if _min(l, 9, key) and _min(p, 11, key) and rq != T.N:
        return "rising m7"
    # chromatic descending minor 7ths (direction continues): 3m7 → b3m7 → 2m7 and any x
    if (lq in _M7S and pq in _M7S and rq in _M7S and p[0] == (l[0] - 1) % 12
            and r[0] == (p[0] - 1) % 12):
        return "chromatic m7"
    # passing diminished: x → #x°7 → (x+2)m (also a major target, or the target's bass)
    if (pq in ("dim", "dim7") and p[0] == (l[0] + 1) % 12
            and ((r[0] == (p[0] + 1) % 12 and _core(rq) in (_MIN | _MAJ | {"7"}))
                 or r[2] == (p[0] + 1) % 12)):
        return "passing diminished"
    # secondary dominant: 1 → 3(7)/#5 → 6m (and other bass-step forms): V/x of a diatonic
    # minor chord, i.e. a dominant whose 3rd is chromatic
    if ((pq in T.DOMINANT or pq == "maj") and r[0] == (p[0] + 5) % 12
            and _core(rq) in _MIN and _d(r, key) in (2, 4, 9)):
        return "secondary dominant"
    # tritone substitution: 2m7 → b2(7) → 1
    if _min(l, 2, key) and _d(p, key) == 1 and pq in T.DOMINANT and _maj(r, 0, key):
        return "tritone substitution"
    # line cliché (any root): min → mM7 → m7 → m6; maj → aug → 6 → 7
    if l[0] == p[0] == r[0] and l[2] == p[2] == r[2] and _line_step(lq, pq) and _line_step(pq, rq):
        return "line cliche"
    # sus resolution: Xsus4 → X
    if pq in T.SUS4_FAMILY and r[0] == p[0] and rq in _SUS_TARGET:
        return "sus resolution"
    # four over five: 4/5 → 1 (any IV/5-type chord resolving to the chord a 4th above its bass)
    if _iv5_type(p) and r[0] == (p[2] + 5) % 12 and rq != T.N:
        return "four over five"
    # pedal neighbour: 1 → 4/1 → 1
    if (l[0] == r[0] and l[2] == r[2] == p[2] and p[0] == (l[0] + 5) % 12
            and _core(lq) in _MAJ and _core(pq) in _MAJ):
        return "pedal neighbour"
    # royal road: 4maj7 → 5 (or 5/4) → 3m7 → 6m
    if (_is(l, key, 5, frozenset({"maj", "maj7"})) and _is(p, key, 7, _DOM_OR_MAJ)
            and _bd(p, key) in (7, 5) and _min(r, 4, key)):
        return "royal road"
    if (_is(l, key, 7, _DOM_OR_MAJ) and _bd(l, key) in (7, 5) and _min(p, 4, key)
            and _min(r, 9, key)):
        return "royal road"
    return None


def bass_monotone(l, p, r) -> bool:
    """bass(l) → bass(p) → bass(r) moves in one direction by 1–2 semitones per step."""
    s1, s2 = _signed(l[2], p[2]), _signed(p[2], r[2])
    return 1 <= abs(s1) <= 2 and 1 <= abs(s2) <= 2 and (s1 > 0) == (s2 > 0)


def idiom(l, p, r, key: int) -> str | None:
    a, b, c = _rqb(l), _rqb(p), _rqb(r)
    if T.N in (a[1], b[1], c[1]):
        return None
    return _idiom(a, b, c, key)


def trigram(l, p, r, key: int) -> float:
    """§6.4 salience bonus for passing chord p between l and r (nats, 0 / 1.5 / 3.0)."""
    a, b, c = _rqb(l), _rqb(p), _rqb(r)
    if T.N in (a[1], b[1], c[1]):
        return 0.0
    out = IDIOM_BONUS if _idiom(a, b, c, key) else 0.0
    if bass_monotone(a, b, c):
        out += BASS_LINE_BONUS
    return out
