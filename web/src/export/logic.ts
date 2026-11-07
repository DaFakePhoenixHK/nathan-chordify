// Export logic (2026-11-04): the sheet-export URL, which sheet's lyrics to use, and the ChordPro line
// parser for the print view. No chord theory here: every label comes from the server.
import type { SheetsState } from '../api/types';
import type { Acc, Level } from '../app/store';

export type SheetFormat = 'txt' | 'chordpro' | 'json' | 'pdf';

export interface SheetExportParams {
  format: SheetFormat;
  /** Label transposition (song transpose + play-in), semitones. */
  transpose: number;
  level: Level;
  slash: boolean;
  /** Put the chords over the words when the song has lyrics. */
  lyrics: boolean;
  acc: Acc;
  /** The sheet whose words to use (the applied / selected one). */
  sheet?: string | null;
}

const enc = encodeURIComponent;

/** GET /api/tracks/{key}/export/sheet?… */
export function exportSheetUrl(key: string, p: SheetExportParams): string {
  const q = new URLSearchParams();
  q.set('format', p.format);
  q.set('transpose', String(Math.round(p.transpose)));
  q.set('level', p.level);
  q.set('slash', p.slash ? '1' : '0');
  q.set('lyrics', p.lyrics ? '1' : '0');
  q.set('acc', p.acc);
  if (p.sheet) q.set('sheet', p.sheet);
  return `/api/tracks/${enc(key)}/export/sheet?${q.toString()}`;
}

/** The sheet whose words go under our chords: the applied one, else the one shown on the grid, else the
 *  best result that has words. Null when no sheet has words. */
export function pickLyricsSheet(st: SheetsState | null, sheetSel: string | null): string | null {
  if (!st) return null;
  const has = (id: string | null | undefined) => !!id && st.results.some((r) => r.id === id && r.lyrics_text);
  if (has(st.applied?.id)) return st.applied!.id;
  if (has(sheetSel)) return sheetSel;
  return st.results.find((r) => r.lyrics_text)?.id ?? null;
}

export function hasLyrics(st: SheetsState | null): boolean {
  return !!st && st.results.some((r) => !!r.lyrics_text);
}

/** A chord sheet's file name: "<title> (<key>) chords.<ext>" without the characters Windows refuses. */
export function sheetFileName(title: string, key: string, ext: string): string {
  const bad = new Set('<>:"/\\|?*');
  const stem =
    Array.from(title)
      .map((ch) => (bad.has(ch) || ch.charCodeAt(0) < 32 ? '_' : ch))
      .join('')
      .trim()
      .replace(/[ .]+$/, '')
      .slice(0, 100) || 'song';
  return `${stem} (${key}) chords.${ext}`;
}

// ---------------------------------------------------------------- ChordPro lines → tokens for the print view

export type ProLine = { kind: 'blank' } | { kind: 'label'; text: string } | { kind: 'text'; tokens: { chord: string | null; text: string }[] };

const LABEL_WORDS = /^(intro|verse|chorus|bridge|pre[- ]?chorus|tag|outro|ending|interlude|coda|refrain|instrumental|solo|vamp|turnaround|part|section|主歌|副歌|導歌|前奏|間奏|尾奏|過門|橋段|結尾)/i;
const CHORD_LIKE = /^[A-G](?:#|b|♯|♭)?[^\s]*$/;

/** A whole-line "[…]" is a section label unless it reads like a chord ("[Verse 1]" vs "[C]"). */
export function isLabelLine(line: string): string | null {
  const m = /^\s*\[([^[\]]+)\]\s*$/.exec(line);
  if (!m) return null;
  const inner = m[1];
  if (LABEL_WORDS.test(inner.trim())) return inner.trim();
  return CHORD_LIKE.test(inner.trim()) ? null : inner.trim();
}

/** "[C]Twinkle [G]twinkle" → tokens [{C,"Twinkle "},{G,"twinkle"}]; text before the first chord has chord null. */
export function parseProLine(line: string): ProLine {
  if (!line.trim()) return { kind: 'blank' };
  const label = isLabelLine(line);
  if (label !== null) return { kind: 'label', text: label };
  if (/^\s*\{[^}]*\}\s*$/.test(line)) return { kind: 'label', text: line.trim().replace(/^\{(?:comment|c):?\s*/i, '').replace(/\}$/, '') };
  const tokens: { chord: string | null; text: string }[] = [];
  const re = /\[([^[\]]+)\]/g;
  let pos = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(line))) {
    const pre = line.slice(pos, m.index);
    if (tokens.length) tokens[tokens.length - 1].text += pre;
    else if (pre) tokens.push({ chord: null, text: pre });
    tokens.push({ chord: m[1], text: '' });
    pos = m.index + m[0].length;
  }
  const tail = line.slice(pos);
  if (tokens.length) tokens[tokens.length - 1].text += tail;
  else tokens.push({ chord: null, text: tail });
  return { kind: 'text', tokens };
}

export function parseChordPro(text: string): ProLine[] {
  return text.replace(/\r\n?/g, '\n').split('\n').map(parseProLine);
}

/** Save text as a file from the browser (the demo has no server to download from). */
export function downloadText(name: string, text: string, type = 'text/plain;charset=utf-8'): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}
