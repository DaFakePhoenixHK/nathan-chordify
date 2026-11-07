// Chord label selection. THE CLIENT NEVER SPELLS CHORDS (§5.12 step 12): letter labels come from
// display.variants["<t>:<acc>"].chords[id] = [simple, standard, full]; Nashville / Roman strings come from
// chord.degree (they don't change with transposition). Only presentation-level string edits happen here:
// dropping a "/bass" suffix (Slash toggle, Simple level) and keeping the degree's root numeral at Simple.
import type { Analysis, ChordOut, Display, DisplayVariant } from '../api/types';
import type { Acc, Level, Notation } from '../app/store';
import type { Seg } from './model';

export const LEVEL_INDEX: Record<Level, number> = { simple: 0, standard: 1, full: 2 };

const mod12 = (n: number) => ((n % 12) + 12) % 12;

/** Label transposition: the song's transpose t plus the play-in offset p. */
export function shownShift(transpose: number, playIn: number): number {
  return mod12(transpose + playIn);
}

export function variantOf(display: Display | null, shift: number, acc: Acc): DisplayVariant | null {
  if (!display) return null;
  return display.variants[`${mod12(shift)}:${acc}`] ?? display.variants[`${mod12(shift)}:auto`] ?? null;
}

const LETTER_BASS = /\/[A-G](?:#|b|♯|♭|𝄪|𝄫)*$/;
const NASH_BASS = /\/[#b♯♭]?[1-7]$/;
const ROMAN_NONCHORD_BASS = /\/[#b♯♭]?[1-7]̂$/;

export function hasLetterBass(label: string): boolean {
  return LETTER_BASS.test(label);
}

export function stripLetterBass(label: string): string {
  return label.replace(LETTER_BASS, '');
}

export function isNoChord(c: ChordOut): boolean {
  return c.harte === 'N' || c.quality === 'N' || c.root == null;
}

export interface LabelCtx {
  variant: DisplayVariant | null;
  level: Level;
  slash: boolean;
  notation: Notation;
}

/** Letter label at the level, from display.json (falls back to analysis labels for t = 0). */
export function letterLabel(c: ChordOut, ctx: LabelCtx): string {
  const arr = ctx.variant?.chords[c.id];
  return arr?.[LEVEL_INDEX[ctx.level]] ?? c.label[ctx.level];
}

export function chordLabel(c: ChordOut, ctx: LabelCtx): string {
  if (isNoChord(c)) return 'N.C.';
  const letters = letterLabel(c, ctx);
  if (ctx.notation === 'letters') return ctx.slash ? letters : stripLetterBass(letters);

  // Degrees: one string per chord (full detail). The letter label at this level says whether a bass shows.
  const deg = ctx.notation === 'nashville' ? c.degree.nashville : c.degree.roman;
  const showBass = ctx.slash && hasLetterBass(letters);
  if (ctx.level === 'simple') {
    // Root numeral + the simple label's quality mark (m, °) — no theory, just the server's pieces.
    if (ctx.notation === 'nashville') {
      const root = /^[#b♯♭]?[1-7]/.exec(deg)?.[0];
      const suffix = /^[A-G](?:#|b|♯|♭)*(.*)$/.exec(stripLetterBass(letters))?.[1] ?? '';
      return root ? root + suffix : deg;
    }
    const root = /^[#b♯♭]?(?:VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)/.exec(deg)?.[0];
    const dim = /°|dim/.test(letters) ? '°' : '';
    return root ? root + dim : deg;
  }
  if (showBass) return deg;
  return ctx.notation === 'nashville' ? deg.replace(NASH_BASS, '') : deg.replace(ROMAN_NONCHORD_BASS, '');
}

/** Label of a timeline segment (handles Everything-mode bass passing tones drawn as slash chords). */
export function segLabel(seg: Seg, ctx: LabelCtx): string {
  const base = chordLabel(seg.chord, ctx);
  if (seg.bassTone < 0 || ctx.notation !== 'letters' || ctx.level === 'simple' || isNoChord(seg.chord)) return base;
  const bassName = ctx.variant?.bass_line[seg.bassTone];
  return bassName ? `${stripLetterBass(base)}/${bassName}` : base;
}

/** Labels for every seg plus whether each one should be drawn (it differs from the previous one). */
export function segLabels(segs: Seg[], ctx: LabelCtx): { text: string[]; show: boolean[] } {
  const text = segs.map((s) => segLabel(s, ctx));
  const show = text.map((t, i) => i === 0 || t !== text[i - 1]);
  return { text, show };
}

/** Display label for an alternative/equivalent Harte string: reuse a chord in this song with that Harte. */
export function harteLabel(a: Analysis, harte: string, ctx: LabelCtx): { text: string; matched: boolean } {
  if (harte === 'N') return { text: 'N.C.', matched: true };
  const c = a.chords.find((x) => x.harte === harte);
  if (c) return { text: chordLabel(c, { ...ctx, level: ctx.notation === 'letters' ? 'full' : ctx.level, slash: true }), matched: true };
  return { text: harte, matched: false };
}

/** Key name for a key segment at a label shift. */
export function keyName(display: Display | null, a: Analysis, seg: number, shift: number, acc: Acc): string {
  const v = variantOf(display, shift, acc);
  const name = v?.key_names[seg];
  if (name) return name;
  const s = a.key.segments[seg];
  return s ? `${s.tonic}${s.mode === 'minor' ? 'm' : ''}` : a.key.global.tonic;
}

export function signed(n: number): string {
  return n > 0 ? `+${n}` : n < 0 ? `−${Math.abs(n)}` : '0';
}
