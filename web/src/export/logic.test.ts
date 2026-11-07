import { describe, expect, it } from 'vitest';
import type { SheetResult, SheetsState } from '../api/types';
import { exportSheetUrl, hasLyrics, isLabelLine, parseChordPro, parseProLine, pickLyricsSheet, sheetFileName } from './logic';

const result = (over: Partial<SheetResult>): SheetResult => ({
  id: 'r',
  source: 'cantonhymn',
  url: null,
  title: '回家',
  artist: null,
  year: null,
  key_written: 'B',
  key_recording: 'B',
  transpose: 0,
  meter: '4/4',
  n_chords: 4,
  match: { title: 1, align: 0.9, score: 0.9 },
  verdict: 'good',
  chart_text: '| B . . . |',
  aligned: [],
  warnings: [],
  ...over,
});

const state = (results: SheetResult[], applied: string | null = null): SheetsState => ({
  status: 'done',
  error: null,
  query: { title: '回家', artist: '', terms: [] },
  results,
  applied: applied ? { id: applied, group: 'g', mode: 'refine', ts: 't' } : null,
  ts: null,
});

describe('the export URL', () => {
  it('carries the view settings and encodes the key and the sheet id', () => {
    const url = exportSheetUrl('yt-Y7boUFbM76k', { format: 'txt', transpose: -1, level: 'simple', slash: false, lyrics: true, acc: 'flat', sheet: 'ch:1' });
    expect(url.startsWith('/api/tracks/yt-Y7boUFbM76k/export/sheet?')).toBe(true);
    const q = new URLSearchParams(url.split('?')[1]);
    expect(Object.fromEntries(q)).toEqual({ format: 'txt', transpose: '-1', level: 'simple', slash: '0', lyrics: '1', acc: 'flat', sheet: 'ch:1' });
    const u2 = exportSheetUrl('a b', { format: 'chordpro', transpose: 14.4, level: 'full', slash: true, lyrics: false, acc: 'auto' });
    expect(u2).toContain('/api/tracks/a%20b/export/sheet?');
    expect(u2).toContain('transpose=14');
    expect(u2).toContain('slash=1&lyrics=0');
    expect(u2).not.toContain('sheet=');
  });

  it('names the file like the server does and strips the characters Windows refuses', () => {
    expect(sheetFileName('回家', 'B', 'txt')).toBe('回家 (B) chords.txt');
    expect(sheetFileName('詩歌-回家 [歌詞] Kelly:DD', 'Bb', 'cho')).toBe('詩歌-回家 [歌詞] Kelly_DD (Bb) chords.cho');
    expect(sheetFileName('   ', 'C', 'txt')).toBe('song (C) chords.txt');
  });
});

describe('which sheet gives the words', () => {
  const noWords = result({ id: 'a' });
  const words = result({ id: 'b', lyrics_text: '[B]啦\n' });
  const words2 = result({ id: 'c', source: 'simplyworship', lyrics_text: '[E]la\n' });

  it('prefers the applied sheet, then the one shown on the grid, then the first with words', () => {
    expect(pickLyricsSheet(null, null)).toBeNull();
    expect(pickLyricsSheet(state([noWords]), null)).toBeNull();
    expect(pickLyricsSheet(state([noWords, words, words2]), null)).toBe('b');
    expect(pickLyricsSheet(state([noWords, words, words2]), 'c')).toBe('c');
    expect(pickLyricsSheet(state([noWords, words, words2], 'c'), 'b')).toBe('c');
    // an applied sheet without words does not win
    expect(pickLyricsSheet(state([noWords, words, words2], 'a'), null)).toBe('b');
    expect(hasLyrics(state([noWords]))).toBe(false);
    expect(hasLyrics(state([noWords, words]))).toBe(true);
  });
});

describe('ChordPro lines for the print view', () => {
  it('tells section labels from chords', () => {
    expect(isLabelLine('[Verse 1]')).toBe('Verse 1');
    expect(isLabelLine('[副歌]')).toBe('副歌');
    expect(isLabelLine('[C]')).toBeNull();
    expect(isLabelLine('[F#m7/A]')).toBeNull();
    expect(isLabelLine('[C]word')).toBeNull();
    expect(isLabelLine('[Pre-Chorus 2]')).toBe('Pre-Chorus 2');
  });

  it('splits a line into chord + text tokens', () => {
    expect(parseProLine('')).toEqual({ kind: 'blank' });
    expect(parseProLine('[Chorus]')).toEqual({ kind: 'label', text: 'Chorus' });
    expect(parseProLine('{comment: Bridge}')).toEqual({ kind: 'label', text: 'Bridge' });
    expect(parseProLine('[C]Twinkle [G]twinkle')).toEqual({ kind: 'text', tokens: [{ chord: 'C', text: 'Twinkle ' }, { chord: 'G', text: 'twinkle' }] });
    expect(parseProLine('oh [Am]little [F]star')).toEqual({
      kind: 'text',
      tokens: [{ chord: null, text: 'oh ' }, { chord: 'Am', text: 'little ' }, { chord: 'F', text: 'star' }],
    });
    expect(parseProLine('just words')).toEqual({ kind: 'text', tokens: [{ chord: null, text: 'just words' }] });
    expect(parseProLine('[C] [G]')).toEqual({ kind: 'text', tokens: [{ chord: 'C', text: ' ' }, { chord: 'G', text: '' }] });
  });

  it('parses a whole text with Windows line ends', () => {
    const lines = parseChordPro('[Verse 1]\r\n[C]月亮[G]代表\r\n\r\n[Am]我的心');
    expect(lines.map((l) => l.kind)).toEqual(['label', 'text', 'blank', 'text']);
  });
});
