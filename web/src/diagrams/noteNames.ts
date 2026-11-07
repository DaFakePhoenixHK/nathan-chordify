// Note-name helpers for diagram labels only (DESIGN §12: note names on diagram keys are client-side;
// chord spelling is server-only). A pressed key whose pitch class matches the chord label's root or bass
// letter reuses the server's spelling; other keys follow the key's sharp/flat preference.
import type { Acc } from '../app/store';

const SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
const LETTER_PC: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const FLAT_KEYS = new Set(['F', 'Dm', 'Gm', 'Cm', 'Fm']);

export function pcOfName(name: string): number | null {
  const m = /^([A-G])((?:#|b|♯|♭)*)/.exec(name);
  if (!m) return null;
  let pc = LETTER_PC[m[1]];
  for (const ch of m[2]) pc += ch === '#' || ch === '♯' ? 1 : -1;
  return ((pc % 12) + 12) % 12;
}

export function prefersFlats(acc: Acc, keyName: string | undefined): boolean {
  if (acc === 'flat') return true;
  if (acc === 'sharp') return false;
  if (!keyName) return false;
  return /^[A-G]b/.test(keyName) || FLAT_KEYS.has(keyName);
}

/** MIDI -> display name for the notes of one voicing. */
export function namesFor(notes: number[], letterLabel: string, flats: boolean): Map<number, string> {
  const spelled = new Map<number, string>();
  const root = /^([A-G](?:#|b)?)/.exec(letterLabel)?.[1];
  const bass = /\/([A-G](?:#|b)?)$/.exec(letterLabel)?.[1];
  for (const n of [root, bass]) {
    if (!n) continue;
    const pc = pcOfName(n);
    if (pc != null) spelled.set(pc, n);
  }
  const table = flats ? FLAT : SHARP;
  const out = new Map<number, string>();
  for (const m of notes) {
    const pc = ((m % 12) + 12) % 12;
    out.set(m, spelled.get(pc) ?? table[pc]);
  }
  return out;
}
