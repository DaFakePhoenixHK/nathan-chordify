import { describe, expect, it } from 'vitest';
import analysisJson from '../../../shared/fixtures/huijia.analysis.json';
import type { Analysis, SheetAligned, SheetsState, SheetResult } from '../api/types';
import { buildSquares, buildTimeline, segAt } from '../grid/model';
import { chordLabel } from '../grid/labels';
import {
  bannerOffer,
  countDiffRuns,
  dismissBanner,
  flagsForSquares,
  isBannerDismissed,
  keyShift,
  looksLikeUrl,
  normLabel,
  pollSheets,
  sheetLabelAt,
  sheetMarks,
  sourceKey,
  verdictClass,
  verdictKey,
  type KV,
} from './logic';

const a = analysisJson as unknown as Analysis;

const result = (over: Partial<SheetResult> = {}): SheetResult => ({
  id: 'r1',
  source: 'cantonhymn',
  url: null,
  title: '回家',
  artist: null,
  year: null,
  key_written: 'B',
  key_recording: 'B',
  transpose: 0,
  meter: '4/4',
  n_chords: 10,
  match: { title: 1, align: 0.9, score: 0.9 },
  verdict: 'good',
  chart_text: '| B . . . |',
  aligned: [],
  warnings: [],
  ...over,
});

const state = (over: Partial<SheetsState> = {}): SheetsState => ({
  status: 'done',
  error: null,
  query: { title: '回家', artist: '', terms: ['回家'] },
  results: [result()],
  applied: null,
  ts: '2026-11-03T00:00:00Z',
  ...over,
});

describe('verdicts and card texts', () => {
  it('maps a verdict to its plain-words string and colour class', () => {
    expect(verdictKey('good')).toBe('verdict_good');
    expect(verdictKey('maybe')).toBe('verdict_maybe');
    expect(verdictKey('different')).toBe('verdict_different');
    expect(verdictClass('maybe')).toBe('maybe');
  });

  it('names known sources and leaves unknown sites to the caller', () => {
    expect(sourceKey('cantonhymn')).toBe('source_cantonhymn');
    expect(sourceKey('pasted')).toBe('source_pasted');
    expect(sourceKey('hkchords')).toBeNull();
  });

  it('describes a shifted sheet only when it was shifted and its key is known', () => {
    expect(keyShift(result({ transpose: -1, key_written: 'C' }))).toEqual({ from: 'C', to: 'B' });
    expect(keyShift(result({ transpose: 0, key_written: 'B' }))).toBeNull();
    expect(keyShift(result({ transpose: 2, key_written: null }))).toBeNull();
  });

  it('tells a pasted link from pasted chart text', () => {
    expect(looksLikeUrl(' https://cantonhymn.net/song/abc ')).toBe(true);
    expect(looksLikeUrl('| B . . . |\nhttps://x.y')).toBe(false);
    expect(looksLikeUrl('B E F#')).toBe(false);
  });
});

/** Our chord at time t in the small examples below. */
const ours = (t: number) => (t < 2 ? 'B' : t < 4 ? 'E7' : 'N.C.');

const fakeKv = (): KV => {
  const m = new Map<string, string>();
  return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => void m.set(k, v) };
};

describe('"Sheet says" marks (which beat squares to highlight)', () => {
  const aligned: SheetAligned[] = [
    { t0: 0, t1: 2, harte: 'B:maj', labels: ['B', 'B', 'B'] },
    { t0: 2, t1: 4, harte: 'E:maj7/7', labels: ['E', 'Emaj7/D#', 'Emaj7/D#'] },
  ];
  const squares = [
    { t0: 0, t1: 1, beat: 1 },
    { t0: 1, t1: 2, beat: 2 },
    { t0: 2, t1: 3, beat: 3 },
    { t0: 3, t1: 4, beat: 4 },
    { t0: 4, t1: 5, beat: 1 },
  ];
  it('finds the sheet label under a time, stripping the bass when slash chords are off', () => {
    expect(sheetLabelAt(aligned, 0.5, 2, true)).toBe('B');
    expect(sheetLabelAt(aligned, 2.5, 2, true)).toBe('Emaj7/D#');
    expect(sheetLabelAt(aligned, 2.5, 2, false)).toBe('Emaj7');
    expect(sheetLabelAt(aligned, 2.5, 0, true)).toBe('E');
    expect(sheetLabelAt(aligned, 4.5, 2, true)).toBeNull();
    expect(sheetLabelAt([], 1, 2, true)).toBeNull();
  });

  it('highlights squares whose sheet label differs from ours at the shown level', () => {
    const full = sheetMarks(aligned, squares, ours, 2, true, true);
    expect(full.map((m) => m.label)).toEqual(['B', 'B', 'Emaj7/D#', 'Emaj7/D#', null]);
    expect(full.map((m) => m.differs)).toEqual([false, false, true, true, false]);
    expect(full.map((m) => m.show)).toEqual([true, false, true, false, false]);
    expect(countDiffRuns(full)).toBe(1);
    // Simple level: "E" vs our "E7" still differs as text (we never re-spell; the server's simple label decides)
    const simple = sheetMarks(aligned, squares, ours, 0, true, true);
    expect(simple.map((m) => m.differs)).toEqual([false, false, true, true, false]);
    // A bar start always draws the label again
    const marks = sheetMarks(aligned, [{ t0: 0, t1: 1, beat: 1 }, { t0: 1, t1: 2, beat: 1 }], ours, 2, true, true);
    expect(marks.map((m) => m.show)).toEqual([true, true]);
  });

  it('never compares when the user has transposed (labels are for the original key)', () => {
    const m = sheetMarks(aligned, squares, ours, 2, true, false);
    expect(m.some((x) => x.differs)).toBe(false);
    expect(m[2].label).toBe('Emaj7/D#');
  });

  it('treats every "no chord" spelling as the same thing', () => {
    expect(normLabel('N')).toBe('N.C.');
    expect(normLabel('N.C.')).toBe('N.C.');
    expect(normLabel(' B/D# ')).toBe('B/D#');
    const m = sheetMarks([{ t0: 0, t1: 1, harte: 'N', labels: ['N', 'N', 'N'] }], [{ t0: 0, t1: 1 }], () => 'N.C.', 2, true, true);
    expect(m[0].differs).toBe(false);
  });

  it('on the 回家 fixture, three changed chords give three disagreeing stretches at Full and one at Simple', () => {
    const changes: Record<string, string[]> = {
      c0002: ['D#m', 'D#m', 'D#m'], // D#m7 → D#m (quality only)
      c0009: ['E', 'Emaj7', 'Emaj7'], // Emaj7/D# → Emaj7 (bass only)
      c0012: ['C#m', 'C#m7', 'C#m7'], // E/F# → C#m7 (root)
    };
    const main = a.chords.filter((c) => c.layer === 'main' && c.harte !== 'N').toSorted((p, q) => p.start - q.start);
    const al: SheetAligned[] = main.map((c) => ({
      t0: c.start,
      t1: c.end,
      harte: c.harte,
      labels: changes[c.id] ?? [c.label.simple, c.label.standard, c.label.full],
    }));
    const sqs = buildSquares(a);
    const segs = buildTimeline(a, 0);
    const oursAt = (level: 'simple' | 'standard' | 'full', slash: boolean) => (t: number) => {
      const k = segAt(segs, t);
      return k < 0 ? null : chordLabel(segs[k].chord, { variant: null, level, slash, notation: 'letters' });
    };
    expect(countDiffRuns(sheetMarks(al, sqs, oursAt('full', true), 2, true, true))).toBe(3);
    // slash chords off: "Emaj7/D#" becomes "Emaj7" on both sides → 2
    expect(countDiffRuns(sheetMarks(al, sqs, oursAt('full', false), 2, false, true))).toBe(2);
    expect(countDiffRuns(sheetMarks(al, sqs, oursAt('simple', true), 0, true, true))).toBe(1);
    // the sheet has nothing over the silent intro
    expect(sheetMarks(al, sqs, oursAt('full', true), 2, true, true)[0].label).toBeNull();
  });

  it('marks the squares a flagged spot overlaps', () => {
    const flags = [{ t0: 1.5, t1: 2.5, detected: 'E7', sheet: 'E' }];
    const f = flagsForSquares(flags, squares);
    expect(f.map((x) => (x ? 1 : 0))).toEqual([0, 1, 1, 0, 0]);
  });
});

describe('polling while the server searches', () => {
  it('keeps asking every interval until the status leaves "searching"', async () => {
    const answers: SheetsState[] = [state({ status: 'searching', results: [] }), state({ status: 'searching', results: [] }), state()];
    const slept: number[] = [];
    const st = await pollSheets(() => Promise.resolve(answers.shift()!), { intervalMs: 1500, sleep: async (ms) => void slept.push(ms) });
    expect(st?.status).toBe('done');
    expect(slept).toEqual([1500, 1500]);
  });

  it('stops with null when cancelled and gives up after maxMs', async () => {
    let calls = 0;
    const get = () => {
      calls++;
      return Promise.resolve(state({ status: 'searching' }));
    };
    expect(await pollSheets(get, { cancelled: () => true })).toBeNull();
    expect(calls).toBe(0);
    const st = await pollSheets(get, { intervalMs: 1000, maxMs: 2000, sleep: async () => undefined });
    expect(st?.status).toBe('searching');
    expect(calls).toBe(3);
  });
});

describe('banner', () => {
  it('offers the best result only when it looks good, the search is done and nothing is applied', () => {
    expect(bannerOffer(state(), false)?.id).toBe('r1');
    expect(bannerOffer(state(), true)).toBeNull();
    expect(bannerOffer(state({ status: 'searching' }), false)).toBeNull();
    expect(bannerOffer(state({ results: [result({ verdict: 'maybe' })] }), false)).toBeNull();
    expect(bannerOffer(state({ applied: { id: 'r1', group: 'g', mode: 'refine', ts: 'x' } }), false)).toBeNull();
    expect(bannerOffer(state({ results: [] }), false)).toBeNull();
    expect(bannerOffer(null, false)).toBeNull();
  });

  it('remembers "Not now" per song', () => {
    const kv = fakeKv();
    expect(isBannerDismissed(kv, 'yt-a')).toBe(false);
    dismissBanner(kv, 'yt-a');
    expect(isBannerDismissed(kv, 'yt-a')).toBe(true);
    expect(isBannerDismissed(kv, 'yt-b')).toBe(false);
    dismissBanner(kv, 'yt-a'); // twice is fine
    expect(JSON.parse(kv.getItem('nc.sheets.notnow')!)).toEqual(['yt-a']);
    kv.setItem('nc.sheets.notnow', 'garbage');
    expect(isBannerDismissed(kv, 'yt-a')).toBe(false);
  });
});
