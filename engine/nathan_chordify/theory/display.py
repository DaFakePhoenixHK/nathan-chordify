"""display.json builder (DESIGN §5.12 step 12): every transposition × accidental preference, server-side.

The client never spells chords: it picks variant "<t>:<pref>" and the voicings for "<t>".
"""
from __future__ import annotations

from .. import schema
from . import harte as H
from . import simplify as SM
from . import spelling as S
from . import voicing as V

PREFS = S.PREFS                      # ("auto", "sharp", "flat")


def key_index(tonic: str, mode: str) -> int:
    """KeyTrack convention: 0–11 major tonic pc, 12–23 minor."""
    return S.parse_note(tonic).pc + (12 if mode == "minor" else 0)


def _key_at(analysis: schema.Analysis, t: float) -> int:
    segs = analysis.key.segments
    for seg in segs:
        if seg.start <= t < seg.end:
            return key_index(seg.tonic, seg.mode)
    last = segs[-1] if segs else None
    g = analysis.key.global_
    return key_index(last.tonic, last.mode) if last else key_index(g.tonic, g.mode)


def chord_triples(analysis: schema.Analysis) -> list[tuple[int, str, int]]:
    return [H.parse_harte(c.harte, strict=False) for c in analysis.chords]


def build(analysis: schema.Analysis, upper_structure: bool = True, rootless: bool = False) -> schema.Display:
    chords = analysis.chords
    triples = chord_triples(analysis)
    keys = [_key_at(analysis, c.start) for c in chords]
    # next chord's root (time order) drives secondary-dominant / passing-°7 spelling (§5.12 step 7)
    order = sorted(range(len(chords)), key=lambda i: (chords[i].start, chords[i].layer != "main"))
    next_root: dict[int, int | None] = {}
    for a, b in zip(order, order[1:] + [None]):
        next_root[a] = triples[b][0] if b is not None and triples[b][0] >= 0 else None
    seg_keys = [key_index(s.tonic, s.mode) for s in analysis.key.segments] or \
               [key_index(analysis.key.global_.tonic, analysis.key.global_.mode)]
    bass_keys = [_key_at(analysis, b.start) for b in analysis.bass_line]

    variants: dict[str, schema.DisplayVariant] = {}
    for t in range(12):
        for pref in PREFS:
            labels: dict[str, list[str]] = {}
            for i, c in enumerate(chords):
                r, q, b = triples[i]
                labels[c.id] = [SM.label(r, q, b, keys[i], transpose=t, pref=pref, level=lv,
                                         upper_structure=upper_structure, next_root=next_root[i])
                                for lv in SM.LEVELS]
            bass_names = [S.spell(bn.midi % 12, "maj", bn.midi % 12, k, t, pref).root
                          for bn, k in zip(analysis.bass_line, bass_keys)]
            variants[f"{t}:{pref}"] = schema.DisplayVariant(
                key_names=[S.key_name(k, t, pref) for k in seg_keys], chords=labels, bass_line=bass_names)

    base = V.voice_sequence(triples, upper_structure=upper_structure, rootless=rootless)
    voicings: dict[str, dict[str, schema.Voicing]] = {}
    for t in range(12):
        moved = V.transpose_voicings(base, t)
        voicings[str(t)] = {c.id: schema.Voicing(lh=list(v.lh), rh=list(v.rh)) for c, v in zip(chords, moved)}
    return schema.Display(variants=variants, voicings=voicings)
