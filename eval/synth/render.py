"""Synthetic songs with exact chord labels: the D2 / mini-D2 renderer (DESIGN §13.3).

A small NumPy "band" renders a chord progression to 44.1 kHz stereo audio:

- piano: additive tones with inharmonic, decaying partials; voice-led right-hand voicings that are
  re-struck on every chord change (optional rhythmic comping) plus a left-hand bass note.
  ``piano="salamander"`` (CLI ``--piano salamander``) plays the same notes on the sampled Salamander
  Grand Piano instead (``train/piano.py``; needs ``models/piano/salamander``);
- bass: a harmonic-rich pluck playing each chord's own bass note (slash basses included);
- pad (optional): slow strings that SUSTAIN the previous main chord across passing chords, which
  is what masks short passing chords in real worship mixes;
- noise drums (kick / snare / hat), a vocal-like melody (vibrato, formants) and a synthetic
  reverb (exponential noise tail, RT60 0.5–3 s).

Each item is written as ``<id>.wav``, ``<id>.lab`` (start end Harte), ``<id>.beats`` (time and
beat-in-bar, 1 = downbeat) and ``<id>.json`` (parameters + segments with ``passing`` flags and
idiom ``tags``). Labels are exact: every chord boundary is a (swung) grid time; note onsets are
humanised by a few ms only.

Progression DSL (bar text, DESIGN §0.1), self-contained (it does not use theory/chart.py)::

    | B . . B+7 | Emaj7 . A7 Em6 |

- one bar between ``|`` marks, one token per beat; ``.`` continues the previous chord;
- a bar with fewer tokens than beats and no dots or brackets is split evenly (``| B E |`` = 2 + 2);
  with dots or brackets, the last chord runs to the end of the bar;
- ``(A7 Em6)`` splits one beat evenly (``.`` allowed inside); ``N`` = no chord; ``%`` repeats the
  previous bar; a trailing ``?`` (unsure) is ignored; ``[Section]`` labels and text outside the
  outermost ``|`` marks are ignored;
- extensions for synthetic charts: a leading ``*`` marks a passing chord and ``@tag`` adds an idiom
  tag (``*A7@borrowed_iv``).

A list of ``(beats, chord[, mark])`` tuples also works (chord = symbol or Harte label; mark =
``"*"``, ``"*tag"`` or ``"tag"``).

CLI::

    python -m eval.synth.render --set mini-d2 --out eval/data/d2/
    python -m eval.synth.render --chart "| C . . *C+7 | Fmaj7 . . . |" --key D --bpm 80 --out eval/data/tmp
    python -m eval.synth.render --list
    python -m eval.synth.render --set mini-d2 --piano salamander --out eval/data/d2_salamander/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter

SR = 44100
MAX_SECONDS = 30.0

# ------------------------------------------------------------------------------------ pitch names

_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLATS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
KEYS = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_FLAT_KEYS = {1, 3, 5, 8, 10}  # Db Eb F Ab Bb are spelled with flats


def note_pc(name: str) -> int:
    """Pitch class of a note name ('F#', 'Bb', 'E')."""
    name = name.strip().replace("♯", "#").replace("♭", "b")
    if not name or name[0] not in _LETTERS or any(ch not in "#b" for ch in name[1:]):
        raise ValueError(f"bad note name: {name!r}")
    return (_LETTERS[name[0]] + name.count("#") - name[1:].count("b")) % 12


def pc_name(pc: int, flats: bool = False) -> str:
    return (FLATS if flats else SHARPS)[pc % 12]


def key_uses_flats(key: str) -> bool:
    return note_pc(key) in _FLAT_KEYS


# --------------------------------------------------------------------------- chord symbols → Harte

# Typed suffix → Harte quality. Conventions follow the engine (theory/harte.py):
# "11" = 9sus4 (F#11 = E/F#), "2" = sus2, "+7" = aug(b7), "add9" = maj(9).
_SUFFIX = {
    "": "maj", "maj": "maj", "M": "maj", "m": "min", "min": "min", "-": "min",
    "dim": "dim", "o": "dim", "aug": "aug", "+": "aug",
    "sus2": "sus2", "2": "sus2", "sus4": "sus4", "sus": "sus4", "5": "5",
    "6": "maj6", "m6": "min6", "7": "7", "maj7": "maj7", "M7": "maj7", "m7": "min7", "-7": "min7",
    "m7b5": "hdim7", "m7-5": "hdim7", "ø": "hdim7", "ø7": "hdim7", "dim7": "dim7", "o7": "dim7",
    "mM7": "minmaj7", "m(maj7)": "minmaj7", "mmaj7": "minmaj7",
    "+maj7": "aug(7)", "maj7#5": "aug(7)",
    "+7": "aug(b7)", "7#5": "aug(b7)", "7+5": "aug(b7)", "aug7": "aug(b7)", "7+": "aug(b7)",
    "7sus4": "sus4(b7)", "7sus": "sus4(b7)",
    "add9": "maj(9)", "add2": "maj(9)", "(add9)": "maj(9)", "madd9": "min(9)", "m(add9)": "min(9)",
    "6/9": "maj6(9)", "69": "maj6(9)",
    "9": "9", "maj9": "maj9", "M9": "maj9", "m9": "min9", "m11": "min11",
    "11": "sus4(b7,9)", "9sus4": "sus4(b7,9)", "9sus": "sus4(b7,9)", "13sus4": "sus4(b7,9,13)",
    "13": "13", "7b9": "7(b9)", "7#9": "7(#9)", "7b13": "7(b13)", "maj7#11": "maj7(#11)",
}
_SYMBOL_RE = re.compile(r"^([A-G][#b]*)(.*?)(?:/([A-G][#b]*))?$")
_BASS_DEGREE = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")
# Upper extensions that mir_eval's quality bitmaps leave out (they only hold the 7th chord).
_EXTENSIONS = {"9": (2,), "maj9": (2,), "min9": (2,), "11": (2, 5), "min11": (2, 5),
               "13": (2, 9), "maj13": (2, 9), "min13": (2, 9)}


def symbol_to_harte(symbol: str) -> str:
    """Typed chord symbol ('G#m7/F#', 'B+7', 'F#11', 'Em/A') → Harte label ('G#:min7/b7', ...).

    Harte labels ('C:maj7/3') and 'N' pass through after validation."""
    s = symbol.strip().rstrip("?").replace("♯", "#").replace("♭", "b")
    if s.upper() in ("N", "NC", "N.C."):
        return "N"
    if ":" in s:
        mir_eval.chord.encode(s)
        return s
    s = s.replace("Δ7", "maj7").replace("Δ", "maj7").replace("°7", "dim7").replace("°", "dim")
    m = _SYMBOL_RE.match(s)
    if not m or m.group(2) not in _SUFFIX:
        raise ValueError(f"unknown chord symbol: {symbol!r}")
    root, quality = m.group(1), _SUFFIX[m.group(2)]
    note_pc(root)
    label = f"{root}:{quality}"
    if m.group(3):
        iv = (note_pc(m.group(3)) - note_pc(root)) % 12
        if iv:
            label += "/" + ("#5" if iv == 8 and quality.startswith("aug") else _BASS_DEGREE[iv])
    mir_eval.chord.encode(label)
    return label


@lru_cache(maxsize=None)
def chord_tones(label: str) -> tuple[int, tuple[int, ...], int]:
    """(root pc, sorted chord-tone pcs incl. 9/11/13 but not an out-of-chord bass, bass pc).

    'N' → (-1, (), -1)."""
    if label in ("N", "X"):
        return -1, (), -1
    root, quality, degrees, bass = mir_eval.chord.split(label)
    r = mir_eval.chord.pitch_class_to_semitone(root)
    ivs = {i for i, on in enumerate(mir_eval.chord.QUALITIES[quality]) if on} | {0}
    ivs |= set(_EXTENSIONS.get(quality, ()))
    for d in degrees:
        semis = mir_eval.chord.scale_degree_to_semitone(d.lstrip("*")) % 12
        (ivs.discard if d.startswith("*") else ivs.add)(semis)
    b = (r + mir_eval.chord.scale_degree_to_semitone(bass)) % 12
    return r, tuple(sorted((r + i) % 12 for i in ivs)), b


def transpose_label(label: str, semitones: int, flats: bool = False) -> str:
    """Transpose a Harte label (the bass is relative to the root, so only the root changes)."""
    if label in ("N", "X") or semitones % 12 == 0:
        return label
    m = re.match(r"^([A-G][#b]*)(.*)$", label)
    return pc_name(note_pc(m.group(1)) + semitones, flats) + m.group(2)


def transpose_text(text: str, semitones: int, flats: bool = False) -> str:
    """Transpose every note name in a chart or symbol ('| C . *G/B |' → '| D . *A/C# |')."""
    if semitones % 12 == 0:
        return text
    return re.sub(r"(?<![A-Za-z#])([A-G][#b]*)",
                  lambda m: pc_name(note_pc(m.group(1)) + semitones, flats), text)


# ------------------------------------------------------------------------------- progression DSL

@dataclass
class Chord:
    """One chord of a progression: its length in beats and its exact Harte label."""
    beats: float
    label: str
    symbol: str = ""
    passing: bool = False
    tags: tuple[str, ...] = ()


def _parse_token(tok: str, default_tag: str | None) -> Chord:
    passing = tok.startswith("*")
    sym, *tags = tok.lstrip("*").split("@")
    tags = tuple(t for t in tags if t)
    if passing and not tags and default_tag:
        tags = (default_tag,)
    sym = sym.rstrip("?")
    return Chord(0.0, symbol_to_harte(sym), sym, passing, tags)


def _bar_tokens(bar: str) -> list[str]:
    """Split a bar into tokens; '(A7 Em6)' is one bracketed beat, 'Cm(maj7)' stays one chord."""
    toks, i, n = [], 0, len(bar)
    while i < n:
        if bar[i].isspace():
            i += 1
            continue
        j, depth = i, 0
        while j < n:
            ch = bar[j]
            if ch.isspace() and depth == 0:
                break
            depth += (ch == "(") - (ch == ")")
            j += 1
            if bar[i] == "(" and depth == 0:  # a bracketed beat ends at its ")"
                break
        if depth != 0:
            raise ValueError(f"unbalanced brackets in bar {bar!r}")
        toks.append(bar[i:j])
        i = j
    return toks


def _bar_units(bar: str, beats_per_bar: int) -> list[tuple[float, str]]:
    """(beats, token) pairs of one bar; token '.' continues the previous chord."""
    tokens = _bar_tokens(bar)
    if not tokens:
        raise ValueError("empty bar")
    timed = "." in tokens or any(t.startswith("(") for t in tokens)
    if len(tokens) > beats_per_bar and timed:
        raise ValueError(f"bar {bar!r} has more than {beats_per_bar} beats")
    unit = 1.0 if timed else beats_per_bar / len(tokens)
    out: list[tuple[float, str]] = []
    for tok in tokens:
        if tok.startswith("("):
            inner = tok[1:-1].split()
            if not inner:
                raise ValueError(f"empty brackets in bar {bar!r}")
            out += [(unit / len(inner), t) for t in inner]
        else:
            out.append((unit, tok))
    if timed and len(tokens) < beats_per_bar:  # the last chord runs to the bar end
        out.append((float(beats_per_bar - len(tokens)), "."))
    return out


def parse_chart(text: str, beats_per_bar: int = 4, default_tag: str | None = None) -> list[Chord]:
    """Parse bar text (see the module docstring) into chords with durations in beats."""
    chords: list[Chord] = []
    prev: list[tuple[float, str]] | None = None
    for line in text.splitlines():
        line = re.sub(r"\[[^\]]*\]", " ", line)
        if "|" not in line:
            continue
        line = line[line.index("|") + 1: line.rindex("|")]
        for bar in line.split("|"):
            bar = bar.strip()
            if not bar:
                continue
            if bar == "%":
                if prev is None:
                    raise ValueError("'%' with no previous bar")
                units = prev
            else:
                units = _bar_units(bar, beats_per_bar)
            prev = units
            for beats, tok in units:
                if tok == ".":
                    if not chords:
                        raise ValueError("'.' before the first chord")
                    chords[-1].beats += beats
                else:
                    chord = _parse_token(tok, default_tag)
                    chord.beats = beats
                    chords.append(chord)
    if not chords:
        raise ValueError("no bars found (bars are written between '|' marks)")
    return chords


def parse_progression(prog, beats_per_bar: int = 4, default_tag: str | None = None) -> list[Chord]:
    """Bar text, a list of Chord, or a list of (beats, chord[, mark]) tuples → list of Chord."""
    if isinstance(prog, str):
        return parse_chart(prog, beats_per_bar, default_tag)
    out = []
    for item in prog:
        if isinstance(item, Chord):
            out.append(replace(item))
            continue
        beats, chord, *rest = item
        mark = str(rest[0]) if rest else ""
        passing, tag = mark.startswith("*"), mark.lstrip("*")
        tags = (tag,) if tag else ((default_tag,) if passing and default_tag else ())
        out.append(Chord(float(beats), symbol_to_harte(chord), str(chord), passing, tags))
    return out


def transpose(chords: list[Chord], semitones: int, flats: bool = False) -> list[Chord]:
    return [replace(c, label=transpose_label(c.label, semitones, flats),
                    symbol=transpose_text(c.symbol, semitones, flats)) for c in chords]


# --------------------------------------------------------------------------------- idiom library

@dataclass(frozen=True)
class Idiom:
    """A DESIGN §6.4 idiom as a 4-bar loop; passing chords are marked ``*`` (½ and 1 beat)."""
    name: str
    pattern: str
    chart: str
    key: str = "C"

    def chords(self, key: str | None = None) -> list[Chord]:
        chords = parse_chart(self.chart, 4, default_tag=self.name)
        key = key or self.key
        return transpose(chords, (note_pc(key) - note_pc(self.key)) % 12, key_uses_flats(key))

    def chart_in(self, key: str) -> str:
        return transpose_text(self.chart, (note_pc(key) - note_pc(self.key)) % 12, key_uses_flats(key))


IDIOMS: tuple[Idiom, ...] = (
    Idiom("augmented_passing", "1 → 1+(7) → 4maj7",
          "| C . . *C+7 | Fmaj7 . . . | C . . (C *C+) | Fmaj7 . G . |"),
    Idiom("borrowed_iv", "4maj7 → b7(7) → 4m6 → 1 (incl. ♭VII7 → iv6)",
          "| Fmaj7 . *Bb7 *Fm6 | C . . . | Fmaj7 . . (*Bb7 *Fm6) | C . G . |"),
    Idiom("descending_canon", "1 → 5/7 → 6m → 6m/5 → 4 → 1/3 → 2m",
          "| C . . *G/B | Am . . (Am *Am/G) | F . . *C/E | Dm . G . |"),
    Idiom("walk_up", "1 → 2m → 1/3 → 4; 4 → 5 → 6m; 5/7 → 1",
          "| C . *Dm *C/E | F . . *G | Am . . (Am *G/B) | C . G . |"),
    Idiom("rising_m7", "2m7 → 3m7 → 4; 6m7 → 7m7 → 1",
          "| Dm7 . . *Em7 | Fmaj7 . . . | Am7 . . (Am7 *Bm7) | C . G . |"),
    Idiom("passing_diminished", "1 → #1°7 → 2m; 2m → #2°7 → 1/3; 4 → #4°7 → 1/5",
          "| C . . *C#dim7 | Dm7 . . (Dm7 *D#dim7) | C/E . F *F#dim7 | C/G . G . |"),
    Idiom("secondary_dominant", "1 → 3(7)/#5 → 6m",
          "| C . . *E7/G# | Am . . . | F . . (F *E/G#) | Am . G . |"),
    Idiom("sus_resolution", "5sus4 → 5; 1sus4 → 1",
          "| C . . . | Fmaj7 . *Gsus4 G | C . (*Csus4 C) . | F . (*Gsus4 G) . |"),
    Idiom("four_over_five", "4/5 → 1",
          "| C . . . | Am7 . . *F/G | C . . . | Dm7 . . (Dm7 *F/G) |"),
    Idiom("pedal_neighbour", "1 → 4/1 → 1 (and 5 → 1/5 → 5)",
          "| C . . *F/C | C . . . | G . . (G *C/G) | G . . . |"),
    Idiom("royal_road", "4maj7 → 5/4 (or 5) → 3m7 → 6m",
          "| Fmaj7 . . *G/F | Em7 . Am . | Fmaj7 . . (Fmaj7 *G7) | Em7 . Am . |"),
    # 回家 pre-chorus (RESEARCH §5.3, B major ~65 BPM): I I+7 | IVmaj7 ♭VII7 iv6 | I I/5 I7 |
    # IVmaj7 ♭VII7 iv7 | ii7 iii7 | ii7/3 | V11 V. B+7 is ~0.5 s (½ beat), A7/Em6 ~1 beat each.
    Idiom("huijia_prechorus", "I I+7 | IVmaj7 bVII7 iv6 | I I/5 I7 | IVmaj7 bVII7 iv7 | ii7 iii7 | ii7/3 | V11 V",
          "| B . . (B *B+7@augmented_passing) | Emaj7 . *A7@borrowed_iv *Em6@borrowed_iv "
          "| B . *B/F# *B7 | Emaj7 . *A7@borrowed_iv *Em7@borrowed_iv "
          "| C#m7 . D#m7@rising_m7 . | C#m7/E . . . | F#11 . F# . |", key="B"),
)
IDIOMS_BY_NAME = {i.name: i for i in IDIOMS}


# ------------------------------------------------------------------------------------ parameters

_METERS = {"4/4": (4, 2), "3/4": (3, 2), "6/8": (2, 3)}  # beats per bar, subdivisions per beat
PIANOS = ("additive", "salamander")


@dataclass
class RenderParams:
    """Render settings. Levels are dB relative to the piano; None switches a part off.

    In 6/8 a beat is a dotted quarter (two beats per bar, three eighths per beat)."""
    bpm: float = 90.0
    meter: str = "4/4"
    swing: float = 0.5           # 0.5 = straight; up to 0.75 delays the off-beat eighth (simple meters)
    repeats: int = 1
    comping: str = "block"       # block | quarters | eighths | arp (passing chords are always block)
    piano_lh: bool = True
    bass_pattern: str = "hold"   # hold | eighths
    bass_db: float | None = -2.0
    pad_db: float | None = -9.0  # D2 sweeps -18 … 0 dB
    drums_db: float | None = -9.0
    vocal_db: float | None = -5.0
    rt60: float = 1.2            # reverb RT60 in s (0 = dry)
    reverb_wet: float = 0.3
    humanize_ms: float = 4.0
    lead_in: float = 0.5
    tail: float = 1.0
    seed: int = 0
    piano: str = "additive"      # additive | salamander (sampled piano, train/piano.py)

    def check(self) -> None:
        if self.meter not in _METERS:
            raise ValueError(f"meter must be one of {sorted(_METERS)}")
        if not 30 <= self.bpm <= 300:
            raise ValueError("bpm must be within 30–300")
        if not 0.5 <= self.swing <= 0.75:
            raise ValueError("swing must be within 0.5–0.75")
        if self.comping not in ("block", "quarters", "eighths", "arp"):
            raise ValueError(f"unknown comping {self.comping!r}")
        if self.bass_pattern not in ("hold", "eighths"):
            raise ValueError(f"unknown bass pattern {self.bass_pattern!r}")
        if self.piano not in PIANOS:
            raise ValueError(f"piano must be one of {PIANOS}")


def stable_seed(*parts) -> int:
    """A seed that is the same in every Python run (``hash()`` is salted)."""
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) & 0x7FFFFFFF


# ------------------------------------------------------------------------------------- timeline

@dataclass
class _Seg:
    b0: float
    b1: float
    label: str
    symbol: str
    passing: bool
    tags: tuple[str, ...]
    t0: float = 0.0
    t1: float = 0.0


@dataclass
class Note:
    part: str    # piano_rh | piano_lh | bass | pad | vocal
    midi: int
    t0: float
    t1: float    # note-off; the release follows
    vel: float = 0.7


def _beat_time(b, p: RenderParams) -> np.ndarray:
    """Time (s) of beat position(s) b counted from the first chord, swing applied."""
    b = np.asarray(b, dtype=float)
    n = np.floor(b + 1e-9)
    f = np.clip(b - n, 0.0, None)
    if p.swing != 0.5 and _METERS[p.meter][1] == 2:
        s = p.swing
        f = np.where(f < 0.5, f * 2 * s, s + (f - 0.5) * 2 * (1 - s))
    return p.lead_in + (n + f) * 60.0 / p.bpm


def _timeline(chords: list[Chord], p: RenderParams) -> tuple[list[_Seg], float]:
    segs, b = [], 0.0
    for _ in range(p.repeats):
        for c in chords:
            segs.append(_Seg(b, b + c.beats, c.label, c.symbol, c.passing, c.tags))
            b += c.beats
    times = _beat_time([s.b0 for s in segs] + [b], p)
    for i, s in enumerate(segs):
        s.t0, s.t1 = float(times[i]), float(times[i + 1])
    return segs, b


def _label_segments(segs: list[_Seg], p: RenderParams, end_t: float, duration: float) -> list[dict]:
    """Merged .lab segments (identical neighbours joined), N in the lead-in and the tail."""
    out: list[dict] = []
    if p.lead_in > 0:
        out.append(dict(t0=0.0, t1=p.lead_in, label="N", symbol="N", beats=0.0, passing=False, tags=[]))
    for s in segs:
        last = out[-1] if out else None
        if last and last["label"] == s.label:
            last["t1"] = s.t1
            last["beats"] += s.b1 - s.b0
            last["passing"] = last["passing"] and s.passing
            last["tags"] = sorted(set(last["tags"]) | set(s.tags))
        else:
            out.append(dict(t0=s.t0, t1=s.t1, label=s.label, symbol=s.symbol, beats=s.b1 - s.b0,
                            passing=s.passing, tags=sorted(s.tags)))
    out.append(dict(t0=end_t, t1=duration, label="N", symbol="N", beats=0.0, passing=False, tags=[]))
    for d in out:
        d["t0"], d["t1"], d["beats"] = round(d["t0"], 6), round(d["t1"], 6), round(d["beats"], 6)
    return out


# ------------------------------------------------------------------------------- note planning

_DROP_ORDER = (7, 0, 2, 9, 5)  # intervals dropped first when a chord has more than 4 tones


def _upper_structure(label: str, max_notes: int = 4) -> list[int]:
    root, tones, _ = chord_tones(label)
    ivs = sorted({(t - root) % 12 for t in tones})
    for d in _DROP_ORDER:
        if len(ivs) <= max_notes:
            break
        if d in ivs:
            ivs.remove(d)
    return [(root + i) % 12 for i in ivs]


def _voicing(pcs: list[int], prev: list[int] | None, lo: int, hi: int) -> list[int]:
    """Close-position voicing of `pcs` within [lo, hi] that moves least from `prev`."""
    pcs = sorted(set(pcs))
    center = (lo + hi) / 2
    best = None
    for rot in range(len(pcs)):
        order = pcs[rot:] + pcs[:rot]
        base = lo + (order[0] - lo) % 12
        notes = [base]
        for pc in order[1:]:
            notes.append(notes[-1] + ((pc - notes[-1]) % 12 or 12))
        if notes[-1] > hi:
            continue
        if prev:
            cost = sum(min(abs(m - q) for q in prev) for m in notes) + sum(min(abs(m - q) for m in notes) for q in prev)
        else:
            cost = 0.0
        cost += 0.15 * abs(float(np.mean(notes)) - center)
        if best is None or (cost, notes) < best:
            best = (cost, notes)
    return best[1]


def _nearest(pc: int, prev: int, lo: int, hi: int) -> int:
    return min((m for m in range(lo, hi + 1) if m % 12 == pc), key=lambda m: (abs(m - prev), m))


def _jitter(rng, p: RenderParams) -> float:
    s = p.humanize_ms / 1000.0
    return float(np.clip(rng.normal(0.0, s), -2.5 * s, 2.5 * s)) if s > 0 else 0.0


def _plan_piano(segs: list[_Seg], p: RenderParams, rng) -> list[Note]:
    sub = 1.0 / _METERS[p.meter][1]
    notes: list[Note] = []
    prev_v, prev_lh = None, 48
    for s in segs:
        if s.label == "N":
            continue
        voicing = _voicing(_upper_structure(s.label), prev_v, 55, 79)
        prev_v = voicing
        style = "block" if s.passing else p.comping
        accent = (0.72 + rng.uniform(-0.05, 0.05)) * (0.8 if s.passing else 1.0)  # passing chords are soft
        if style == "arp":
            for j, m in enumerate(voicing):
                b = s.b0 + j * sub
                if b < s.b1 - 1e-9:
                    notes.append(Note("piano_rh", m, float(_beat_time(b, p)) + _jitter(rng, p), s.t1,
                                      accent if j == 0 else 0.55))
        else:
            step = {"block": None, "quarters": 1.0, "eighths": sub}[style]
            strikes = [s.b0]
            if step:
                k = np.floor(s.b0 / step + 1e-9) + 1
                while k * step < s.b1 - 1e-9:
                    strikes.append(k * step)
                    k += 1
            times = list(_beat_time(strikes, p)) + [s.t1]
            for i, b in enumerate(strikes):
                t0 = float(times[i]) + _jitter(rng, p)
                vel = accent if i == 0 else 0.5 + rng.uniform(-0.04, 0.04)
                notes += [Note("piano_rh", m, t0, float(times[i + 1]), vel) for m in voicing]
        if p.piano_lh:
            prev_lh = _nearest(chord_tones(s.label)[2], prev_lh, 40, 55)
            notes.append(Note("piano_lh", prev_lh, s.t0 + _jitter(rng, p), s.t1, accent * 0.85))
    return notes


def _plan_bass(segs: list[_Seg], p: RenderParams, rng) -> list[Note]:
    sub = 1.0 / _METERS[p.meter][1]
    notes, prev = [], 38
    for s in segs:
        if s.label == "N":
            continue
        prev = _nearest(chord_tones(s.label)[2], prev, 31, 50)
        if p.bass_pattern == "eighths" and not s.passing:
            b = s.b0
            while b < s.b1 - 1e-9:
                b1 = min(b + sub, s.b1)
                t0, t1 = _beat_time([b, b1], p)
                notes.append(Note("bass", prev, float(t0) + _jitter(rng, p), float(t1) - 0.03,
                                  0.85 if b == s.b0 else 0.65))
                b = b1
        else:
            notes.append(Note("bass", prev, s.t0 + _jitter(rng, p), s.t1 - 0.02, 0.85))
    return notes


def _plan_pad(segs: list[_Seg], end_t: float) -> list[Note]:
    """Pad chords follow the main (non-passing) chords only and are held until the next main
    chord starts, so a passing chord sounds over the previous chord's pad."""
    main = [s for s in segs if not s.passing and s.label != "N"]
    spans: list[list] = []
    for s in main:
        if spans and spans[-1][0] == s.label:
            continue
        spans.append([s.label, s.t0])
    notes, prev = [], None
    for i, (label, t0) in enumerate(spans):
        t1 = spans[i + 1][1] if i + 1 < len(spans) else end_t
        voicing = _voicing(_upper_structure(label), prev, 52, 76)
        prev = voicing
        notes += [Note("pad", m, t0, t1, 0.6) for m in voicing]
    return notes


def _plan_vocal(segs: list[_Seg], total_beats: float, p: RenderParams, rng) -> list[Note]:
    """A simple melody of chord tones; a note held over several chords uses their common tones."""
    chords = [s for s in segs if s.label != "N"]

    def tones(s):
        _, t, b = chord_tones(s.label)
        return set(t) | {b}

    lengths = (1.0, 1.0, 2.0) if p.bpm < 90 else (1.0, 2.0, 2.0)
    notes, b, prev = [], 0.0, 67
    while b < total_beats - 1e-9:
        length = min(float(rng.choice(lengths)), total_beats - b)
        if rng.random() < 0.15:  # breath
            b += length
            continue
        over = [s for s in chords if s.b0 < b + length - 1e-9 and s.b1 > b + 1e-9]
        if not over:
            b += length
            continue
        common = set.intersection(*(tones(s) for s in over))
        if not common:
            common, length = tones(over[0]), over[0].b1 - b
        target = prev + int(rng.integers(-3, 4))
        midi = min((m for m in range(60, 77) if m % 12 in common), key=lambda m: (abs(m - target), m))
        t0, t1 = _beat_time([b, b + length], p)
        notes.append(Note("vocal", midi, float(t0), float(t1) - 0.04, 0.8))
        prev, b = midi, b + length
    return notes


def _plan_drums(total_beats: float, p: RenderParams, rng) -> list[tuple[str, float, float]]:
    bpb, sub = _METERS[p.meter]
    hits = []
    for k in range(int(np.ceil(total_beats - 1e-9))):
        pos = k % bpb
        if pos == 0 or (bpb == 4 and pos == 2):
            hits.append(("kick", k, 1.0))
        if (bpb == 4 and pos in (1, 3)) or (bpb == 3 and pos == 2) or (bpb == 2 and pos == 1):
            hits.append(("snare", k, 0.9))
        if bpb == 4 and pos == 2 and rng.random() < 0.3:
            hits.append(("kick", k + 0.5, 0.7))
        hits += [("hat", k + j / sub, 0.6 if j == 0 else 0.4) for j in range(sub)]
    return [(kind, float(_beat_time(b, p)) + _jitter(rng, p), vel)
            for kind, b, vel in hits if b < total_beats - 1e-9]


# ------------------------------------------------------------------------------------ synthesis

_VEL_LEVELS = (0.4, 0.65, 0.9)
_TABLE_N = 4096


def _midi_hz(m) -> np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


@lru_cache(maxsize=None)
def _piano_template(midi: int, level: int) -> np.ndarray:
    """One piano note: inharmonic partials (f_k = k·f0·sqrt(1 + B·k²)), two-stage decay that is
    faster for higher partials, a hammer-position comb, two detuned strings and a thump."""
    vel = _VEL_LEVELS[level]
    f0 = float(_midi_hz(midi))
    inharm = 1.2e-4 * 2 ** ((midi - 60) / 24)
    tau = float(np.clip(6.0 * 2 ** (-(midi - 40) / 20), 0.5, 8.0))
    n = int(min(5.0, 2.5 * tau + 0.5) * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(1000 * midi + level)
    out = np.zeros(n)
    tilt = 1.9 - 0.9 * vel  # louder = brighter
    for k in range(1, 25):
        fk = k * f0 * np.sqrt(1 + inharm * k * k)
        if fk > 9000:
            break
        amp = k ** -tilt * (abs(np.sin(np.pi * k / 7.3)) + 0.05)
        ts = tau / (1 + 0.25 * (k - 1))
        m = min(n, int(ts * 7 * SR) + 1)
        tt = t[:m]
        env = 0.6 * np.exp(-tt / (ts / 8)) + 0.4 * np.exp(-tt / ts)
        ph = rng.uniform(0, 2 * np.pi, 2)
        if k <= 3:
            d = fk * 3e-4
            wave = 0.5 * (np.sin(2 * np.pi * (fk - d) * tt + ph[0]) + np.sin(2 * np.pi * (fk + d) * tt + ph[1]))
        else:
            wave = np.sin(2 * np.pi * fk * tt + ph[0])
        out[:m] += amp * env * wave
    m = int(0.02 * SR)
    thump = lfilter([0.15], [1, -0.85], rng.standard_normal(m)) * np.exp(-np.arange(m) / (0.004 * SR))
    out[:m] += 0.3 * vel * thump * np.max(np.abs(out[:m]))
    out *= np.minimum(1.0, t / 0.002)
    fade = int(0.3 * SR)
    out[-fade:] *= np.linspace(1.0, 0.0, fade)
    out /= np.max(np.abs(out)) + 1e-12
    return (vel * out).astype(np.float32)


@lru_cache(maxsize=None)
def _wavetable(kind: str) -> np.ndarray:
    x = np.arange(_TABLE_N) / _TABLE_N
    if kind == "bass_bright":
        amps = {k: 1 / k / (1 + (k / 10) ** 2) for k in range(1, 41)}
    elif kind == "bass_dark":
        amps = {k: 1 / k ** 2 for k in range(1, 9)}
    else:  # pad: a soft saw
        amps = {k: 1 / k / (1 + (k / 6) ** 2) for k in range(1, 31)}
    w = sum(a * np.sin(2 * np.pi * k * x) for k, a in amps.items())
    w /= np.max(np.abs(w))
    return np.append(w, w[0])


def _osc(kind: str, freq: float, n: int, phase: float = 0.0) -> np.ndarray:
    idx = ((phase + freq * np.arange(n) / SR) % 1.0) * _TABLE_N
    i = idx.astype(np.int64)
    frac = idx - i
    tab = _wavetable(kind)
    return tab[i] * (1 - frac) + tab[i + 1] * frac


def _bass_note(midi: int, dur: float) -> np.ndarray:
    f0 = float(_midi_hz(midi))
    n = int((dur + 0.1) * SR)
    t = np.arange(n) / SR
    w = 0.25 + 0.75 * np.exp(-t / 0.12)  # the pluck gets darker
    x = w * _osc("bass_bright", f0, n) + (1 - w) * _osc("bass_dark", f0, n)
    env = np.minimum(1.0, t / 0.004) * (0.55 + 0.45 * np.exp(-t / 0.2)) * np.exp(-t / 3.0)
    off = min(n, int(dur * SR))
    env[off:] *= np.exp(-np.arange(n - off) / (0.02 * SR))
    return x * env


def _pad_note(midi: int, dur: float, release: float) -> tuple[np.ndarray, np.ndarray]:
    f0 = float(_midi_hz(midi))
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    env = 1 - np.exp(-t / 0.15)  # slow attack (~0.45 s)
    off = min(n, int(dur * SR))
    env[off:] *= np.exp(-np.arange(n - off) / (release / 4 * SR))
    a = _osc("pad", f0 * 2 ** (-6 / 1200), n, phase=(midi * 0.137) % 1)
    b = _osc("pad", f0 * 2 ** (6 / 1200), n, phase=(midi * 0.291) % 1)
    return env * (0.8 * a + 0.2 * b), env * (0.2 * a + 0.8 * b)


_VOWELS = ((730, 1090, 2440), (570, 840, 2410), (530, 1840, 2480), (300, 870, 2240), (270, 2290, 3010))


def _smooth(x: np.ndarray, seconds: float) -> np.ndarray:
    a = 1 - np.exp(-1 / (seconds * SR))
    return lfilter([a], [1, a - 1], x - x[0]) + x[0]


def _vocal(notes: list[Note], n: int, rng) -> np.ndarray:
    """Sung-ish line: glided pitch with delayed vibrato, 12 harmonics shaped by 3 formants."""
    out = np.zeros(n)
    if not notes:
        return out
    pitch = np.full(n, float(notes[0].midi))
    amp, gate = np.zeros(n), np.zeros(n)
    formants = np.tile(np.array(_VOWELS[0], dtype=float)[:, None], (1, n))
    for nt in notes:
        i0, i1 = int(nt.t0 * SR), min(n, int(nt.t1 * SR))
        if i1 <= i0:
            continue
        pitch[i0:] = nt.midi
        formants[:, i0:] = np.array(_VOWELS[int(rng.integers(len(_VOWELS)))], dtype=float)[:, None]
        k = np.arange(i1 - i0) / SR
        amp[i0:i1] = np.minimum(1.0, k / 0.05) * np.minimum(1.0, (k[-1] - k) / 0.08 + 1e-3)
        gate[i0:i1] = np.clip((k - 0.15) / 0.25, 0.0, 1.0)
    pitch = _smooth(pitch, 0.03)
    formants = np.vstack([_smooth(f, 0.05) for f in formants])
    t = np.arange(n) / SR
    rate = 5.4 + 0.3 * rng.random()
    f0 = _midi_hz(pitch + 0.3 * gate * np.sin(2 * np.pi * rate * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    active = np.flatnonzero(amp > 0)
    lo, hi = active[0], active[-1] + 1
    for k in range(1, 13):
        fk = k * f0[lo:hi]
        gain = 0.1 + sum(g / (1 + ((fk - formants[j, lo:hi]) / bw) ** 2)
                         for j, (g, bw) in enumerate(((1.0, 90.0), (0.6, 110.0), (0.3, 170.0))))
        out[lo:hi] += k ** -0.7 * gain * np.sin(k * phase[lo:hi])
    return out * amp


def _drum_kit(rng) -> dict[str, np.ndarray]:
    t = np.arange(int(0.4 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(48 + 85 * np.exp(-t / 0.03)) / SR) * np.exp(-t / 0.16)
    kick[: int(0.003 * SR)] += 0.3 * rng.standard_normal(int(0.003 * SR))
    ts = t[: int(0.25 * SR)]
    b, a = butter(2, [1200 / (SR / 2), 7000 / (SR / 2)], btype="band")
    snare = lfilter(b, a, rng.standard_normal(len(ts))) * np.exp(-ts / 0.09)
    snare = snare / np.max(np.abs(snare)) + 0.5 * np.sin(2 * np.pi * 190 * ts) * np.exp(-ts / 0.05)
    th = t[: int(0.1 * SR)]
    b, a = butter(2, 7000 / (SR / 2), btype="high")
    hat = lfilter(b, a, rng.standard_normal(len(th))) * np.exp(-th / 0.025)
    return {k: v / np.max(np.abs(v)) for k, v in (("kick", kick), ("snare", snare), ("hat", hat))}


def _reverb_ir(rt60: float, rng) -> np.ndarray:
    """Stereo IR (2, n): pre-delay, a few early reflections, an exponential noise tail that
    reaches -60 dB at RT60 and darkens over time. Unit energy per channel."""
    n = int(min(4.0, 1.1 * rt60) * SR)
    t = np.arange(n) / SR
    env = 10 ** (-3 * t / rt60)
    darken = np.exp(-t / (0.25 * rt60))
    pre = int(0.015 * SR)
    irs = []
    for _ in range(2):
        bright = rng.standard_normal(n)
        dark = lfilter([0.3], [1, -0.7], rng.standard_normal(n)) * 2.0
        ir = np.zeros(n + pre)
        ir[pre:] = env * (darken * bright + (1 - darken) * dark)
        for d, g in zip(rng.uniform(0.005, 0.04, 6), rng.uniform(0.3, 0.7, 6)):
            ir[int(d * SR)] += g * rng.choice((-1, 1)) * 4
        irs.append(ir / np.sqrt(np.sum(ir ** 2)))
    return np.vstack(irs)


def _salamander_piano(notes: list[Note], n: int, seed: int) -> np.ndarray:
    """The piano notes on the sampled Salamander Grand Piano (stereo samples, so no panning)."""
    from train.piano import get_piano
    pn = get_piano()
    played = [(nt.midi, int(np.clip(round(nt.vel * 127), 1, 127)), max(0.0, nt.t0), max(0.02, nt.t1 - nt.t0))
              for nt in notes]
    y = pn.render(played, duration=n / SR, seed=seed, pedal_noise=False)
    if pn.sr != SR:
        raise ValueError("the sampler must run at 44.1 kHz")
    return y[:n].T.astype(np.float64)


def _pan(pos: float) -> tuple[float, float]:
    theta = (pos + 1) * np.pi / 4
    return float(np.cos(theta)), float(np.sin(theta))


def _add(bus: np.ndarray, x: np.ndarray, t0: float, pan: float, gain: float = 1.0) -> None:
    i0 = int(round(t0 * SR))
    if i0 < 0:
        x, i0 = x[-i0:], 0
    i1 = min(bus.shape[1], i0 + len(x))
    if i1 <= i0:
        return
    gl, gr = _pan(pan)
    bus[0, i0:i1] += gain * gl * x[: i1 - i0]
    bus[1, i0:i1] += gain * gr * x[: i1 - i0]


# ---------------------------------------------------------------------------------- render + I/O

@dataclass
class Rendered:
    audio: np.ndarray       # (n, 2) float32 at SR
    segments: list[dict]    # .lab rows + symbol, beats, passing, tags
    beats: np.ndarray       # beat times (s)
    beat_pos: np.ndarray    # beat in bar, 1 = downbeat
    notes: list[Note]
    params: RenderParams
    end_time: float = 0.0   # end of the last chord

    @property
    def duration(self) -> float:
        return len(self.audio) / SR


def render(progression, params: RenderParams | None = None) -> Rendered:
    """Render a progression (bar text, list of Chord or (beats, chord[, mark]) tuples)."""
    p = params or RenderParams()
    p.check()
    bpb, _ = _METERS[p.meter]
    chords = parse_progression(progression, bpb)
    segs, total = _timeline(chords, p)
    end_t = float(_beat_time(total, p))
    duration = end_t + p.tail
    n = int(round(duration * SR))
    rng = lambda k: np.random.default_rng([p.seed, k])  # noqa: E731 - one stream per part

    notes = _plan_piano(segs, p, rng(1)) + _plan_bass(segs, p, rng(2))
    if p.pad_db is not None:
        notes += _plan_pad(segs, end_t)
    if p.vocal_db is not None:
        notes += _plan_vocal(segs, total, p, rng(3))
    hits = _plan_drums(total, p, rng(4)) if p.drums_db is not None else []

    buses = {k: np.zeros((2, n)) for k in ("piano", "bass", "pad", "drums", "vocal")}
    if p.piano == "salamander":
        buses["piano"] = _salamander_piano([x for x in notes if x.part.startswith("piano")], n, p.seed)
    for nt in notes:
        dur = max(0.02, nt.t1 - nt.t0)
        if nt.part.startswith("piano") and p.piano == "salamander":
            continue
        if nt.part.startswith("piano"):
            level = int(np.argmin([abs(nt.vel - v) for v in _VEL_LEVELS]))
            tmpl = _piano_template(nt.midi, level)
            m = min(len(tmpl), int((dur + 0.35) * SR))
            x = tmpl[:m].astype(np.float64)
            k = int(dur * SR)
            if k < m:
                x[k:] *= np.exp(-np.arange(m - k) / (0.07 * SR))  # damper
            pan = -0.25 if nt.part == "piano_lh" else 0.15 + 0.01 * (nt.midi - 67)
            _add(buses["piano"], x, nt.t0, pan, nt.vel / _VEL_LEVELS[level])
        elif nt.part == "bass":
            _add(buses["bass"], _bass_note(nt.midi, dur), nt.t0, 0.0, nt.vel)
        elif nt.part == "pad":
            release = 1.2 if nt.t1 < end_t - 1e-6 else 0.25
            left, right = _pad_note(nt.midi, dur, release)
            _add(buses["pad"], left, nt.t0, -1.0, nt.vel * 0.707)
            _add(buses["pad"], right, nt.t0, 1.0, nt.vel * 0.707)
    if p.vocal_db is not None:
        _add(buses["vocal"], _vocal([x for x in notes if x.part == "vocal"], n, rng(5)), 0.0, 0.0)
    if hits:
        kit = _drum_kit(rng(6))
        pans = {"kick": 0.0, "snare": -0.1, "hat": 0.3}
        for kind, t0, vel in hits:
            _add(buses["drums"], kit[kind], t0, pans[kind], vel)

    levels = {"piano": 0.0, "bass": p.bass_db, "pad": p.pad_db, "drums": p.drums_db, "vocal": p.vocal_db}
    sends = {"piano": 1.0, "bass": 0.1, "pad": 1.0, "drums": 0.35, "vocal": 1.0}
    i0, i1 = int(p.lead_in * SR), int(end_t * SR)
    mix, send = np.zeros((2, n)), np.zeros(n)
    for name, bus in buses.items():
        rms = float(np.sqrt(np.mean(bus[:, i0:i1] ** 2))) if i1 > i0 else 0.0
        if levels[name] is None or rms == 0.0:
            continue
        bus *= 0.1 / rms * 10 ** (levels[name] / 20)  # every part at -20 dB RMS + its level
        mix += bus
        send += sends[name] * bus.mean(axis=0)
    if p.rt60 > 0 and p.reverb_wet > 0:
        ir = _reverb_ir(p.rt60, rng(7))
        for ch in range(2):
            mix[ch] += p.reverb_wet * fftconvolve(send, ir[ch])[:n]
    fade = int(0.1 * SR)
    mix[:, -fade:] *= np.linspace(1.0, 0.0, fade)
    mix *= 0.89 / (np.max(np.abs(mix)) + 1e-12)  # peak at -1 dBFS

    n_beats = int(np.ceil(total - 1e-9))
    beats = np.round(_beat_time(np.arange(n_beats), p), 6)
    return Rendered(audio=mix.T.astype(np.float32), segments=_label_segments(segs, p, end_t, n / SR),
                    beats=beats, beat_pos=np.arange(n_beats) % bpb + 1, notes=notes, params=p,
                    end_time=round(end_t, 6))


def write_item(r: Rendered, out_dir, item_id: str, meta: dict | None = None, fmt: str = "wav") -> Path:
    """Write <out_dir>/<item_id>/<item_id>.{wav,lab,beats,json}; returns the item folder."""
    folder = Path(out_dir) / item_id
    folder.mkdir(parents=True, exist_ok=True)
    audio_name = f"{item_id}.{fmt}"
    sf.write(folder / audio_name, r.audio, SR, subtype="PCM_16", format=fmt.upper())
    (folder / f"{item_id}.lab").write_text(
        "".join(f"{s['t0']:.6f}\t{s['t1']:.6f}\t{s['label']}\n" for s in r.segments), encoding="utf-8")
    (folder / f"{item_id}.beats").write_text(
        "".join(f"{t:.6f}\t{int(k)}\n" for t, k in zip(r.beats, r.beat_pos)), encoding="utf-8")
    info = {"id": item_id, **(meta or {}), "params": asdict(r.params), "sr": SR,
            "duration": round(r.duration, 6), "end_time": r.end_time, "audio": audio_name,
            "lab": f"{item_id}.lab", "beats": f"{item_id}.beats", "segments": r.segments}
    (folder / f"{item_id}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    return folder


# ------------------------------------------------------------------------------------- mini-D2

MINI_D2_TEMPOS = (65, 110)
PAD_SWEEP = (-18.0, -14.0, -10.0, -6.0, -3.0, 0.0)
RT60_SWEEP = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
MAX_REPEATS = 2


@dataclass
class ItemSpec:
    item_id: str
    idiom: Idiom
    key: str
    params: RenderParams
    meta: dict = field(default_factory=dict)


def mini_d2_keys(index: int) -> tuple[str, str, str]:
    """Three keys a major third apart; over the 12 idioms every key is used three times
    (the 回家 item, index 11, gets G, B and Eb: B is its original key)."""
    return tuple(KEYS[(5 * index + d) % 12] for d in (0, 4, 8))


def item_id(idiom: str, key: str, bpm: float) -> str:
    return f"{idiom}_{key.replace('#', 's')}_{int(round(bpm)):03d}"


def mini_d2_items() -> list[ItemSpec]:
    """12 idioms × 3 keys × 2 tempos with deterministic seeds; pad level and RT60 are swept
    (each idiom gets all six pad levels and all six RT60 values)."""
    items = []
    for i, idiom in enumerate(IDIOMS):
        beats = sum(c.beats for c in idiom.chords())
        for ki, key in enumerate(mini_d2_keys(i)):
            for ti, bpm in enumerate(MINI_D2_TEMPOS):
                j = 2 * ki + ti
                seed = stable_seed(idiom.name, key, bpm)
                u = np.random.default_rng(seed).random(6)
                fast = bpm >= 90
                styles = ("block", "quarters", "eighths") if fast else ("block", "eighths", "arp")
                lead_in, tail = 0.5, 1.0
                phrase_s = beats * 60.0 / bpm
                repeats = int(max(1, min(MAX_REPEATS, (MAX_SECONDS - lead_in - tail) // phrase_s)))
                params = RenderParams(
                    bpm=float(bpm), repeats=repeats, comping=styles[int(u[0] * 3)],
                    bass_pattern="eighths" if fast and u[1] < 0.5 else "hold",
                    pad_db=PAD_SWEEP[(j + i) % 6], rt60=RT60_SWEEP[(j + 2 * i) % 6],
                    drums_db=-9.0 if (fast or u[2] < 0.5) else None,
                    vocal_db=-5.0 if u[3] < 0.75 else None,
                    swing=0.62 if fast and u[4] < 0.25 else 0.5,
                    lead_in=lead_in, tail=tail, seed=seed)
                meta = {"set": "mini-d2", "idiom": idiom.name, "pattern": idiom.pattern, "key": key,
                        "bpm": float(bpm), "chart": idiom.chart_in(key)}
                items.append(ItemSpec(item_id(idiom.name, key, bpm), idiom, key, params, meta))
    return items


def render_item(spec: ItemSpec) -> Rendered:
    return render(spec.idiom.chords(spec.key), spec.params)


def render_set(out_dir, only: list[str] | None = None, fmt: str = "wav", verbose: bool = True,
               piano: str = "additive") -> list[dict]:
    """Render mini-D2 into out_dir (one folder per item) and write out_dir/index.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [s for s in mini_d2_items() if not only or s.idiom.name in only]
    if piano != "additive":
        specs = [replace(s, params=replace(s.params, piano=piano)) for s in specs]
    index = []
    for k, spec in enumerate(specs, 1):
        t = time.perf_counter()
        r = render_item(spec)
        write_item(r, out_dir, spec.item_id, spec.meta, fmt)
        passing = [s["beats"] for s in r.segments if s["passing"]]
        index.append({"id": spec.item_id, **{k2: spec.meta[k2] for k2 in ("idiom", "key", "bpm")},
                      "duration": round(r.duration, 3), "repeats": spec.params.repeats,
                      "pad_db": spec.params.pad_db, "rt60": spec.params.rt60,
                      "n_passing": len(passing), "passing_beats": sorted(set(passing))})
        if verbose:
            print(f"[{k:2d}/{len(specs)}] {spec.item_id:34s} {r.duration:5.1f} s  "
                  f"{len(passing)} passing  ({time.perf_counter() - t:.1f} s)", flush=True)
    (out_dir / "index.json").write_text(json.dumps({"set": "mini-d2", "items": index}, indent=1), encoding="utf-8")
    return index


# ----------------------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.synth.render", description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", choices=["mini-d2"], help="render a whole set")
    ap.add_argument("--only", help="comma-separated idiom names (with --set)")
    ap.add_argument("--chart", help="render one bar-text chart instead of a set")
    ap.add_argument("--chart-key", default="C", help="key the --chart is written in")
    ap.add_argument("--key", help="render the --chart in this key (default: --chart-key)")
    ap.add_argument("--name", default="custom", help="item id for --chart")
    ap.add_argument("--bpm", type=float, default=90.0)
    ap.add_argument("--meter", default="4/4", choices=sorted(_METERS))
    ap.add_argument("--swing", type=float, default=0.5)
    ap.add_argument("--pad-db", type=float, default=-9.0)
    ap.add_argument("--rt60", type=float, default=1.2)
    ap.add_argument("--comping", default="block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", default="wav", choices=["wav", "flac"])
    ap.add_argument("--piano", default="additive", choices=list(PIANOS),
                    help="piano sound: additive synth (default) or the sampled Salamander Grand Piano")
    ap.add_argument("--out", default="eval/data/d2")
    ap.add_argument("--list", action="store_true", help="list the idiom library")
    a = ap.parse_args(argv)

    if a.list:
        for idiom in IDIOMS:
            beats = sorted({c.beats for c in idiom.chords() if c.passing})
            print(f"{idiom.name:20s} passing {beats} beats  {idiom.pattern}\n    {idiom.chart}")
        return 0
    if a.set:
        only = [s.strip() for s in a.only.split(",")] if a.only else None
        unknown = set(only or ()) - set(IDIOMS_BY_NAME)
        if unknown:
            ap.error(f"unknown idioms: {sorted(unknown)}")
        index = render_set(a.out, only, a.format, piano=a.piano)
        size = sum(f.stat().st_size for f in Path(a.out).rglob("*") if f.is_file())
        print(f"{len(index)} items, {size / 1e6:.1f} MB in {a.out}")
        return 0
    if a.chart:
        key = a.key or a.chart_key
        chords = transpose(parse_chart(a.chart, _METERS[a.meter][0], default_tag="passing"),
                           (note_pc(key) - note_pc(a.chart_key)) % 12, key_uses_flats(key))
        p = RenderParams(bpm=a.bpm, meter=a.meter, swing=a.swing, pad_db=a.pad_db, rt60=a.rt60,
                         comping=a.comping, seed=a.seed, piano=a.piano)
        r = render(chords, p)
        folder = write_item(r, a.out, a.name, {"key": key, "bpm": a.bpm, "chart": a.chart}, a.format)
        print(f"wrote {folder} ({r.duration:.1f} s)")
        return 0
    ap.error("give --set, --chart or --list")
    return 2


if __name__ == "__main__":
    sys.exit(main())
