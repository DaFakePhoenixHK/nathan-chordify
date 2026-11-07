import { describe, expect, it } from 'vitest';
import analysisJson from '../../../shared/fixtures/huijia.analysis.json';
import displayJson from '../../../shared/fixtures/huijia.display.json';
import type { Analysis, Display } from '../api/types';
import { buildBassTimeline, buildGrid, buildTimeline, passingVisible, segAt } from './model';
import { chordLabel, stripLetterBass, variantOf } from './labels';

const a = analysisJson as unknown as Analysis;
const d = displayJson as unknown as Display;
const byId = (id: string) => a.chords.find((c) => c.id === id)!;

describe('visible timeline (§5.11)', () => {
  it('shows only the main layer at Detail 0', () => {
    const segs = buildTimeline(a, 0);
    expect(segs.length).toBe(a.chords.filter((c) => c.layer === 'main').length);
    expect(segs.every((s) => !s.passing)).toBe(true);
  });

  it('is sorted, gap-free and non-overlapping at every stop', () => {
    for (const detail of [0, 0.5, 0.8, 1]) {
      const segs = buildTimeline(a, detail);
      for (let i = 1; i < segs.length; i++) {
        expect(segs[i].start).toBeGreaterThanOrEqual(segs[i - 1].start);
        expect(Math.abs(segs[i].start - segs[i - 1].end)).toBeLessThan(1e-3);
      }
    }
  });

  it('applies "s > 0 and s >= min_detail"', () => {
    const p = { ...byId('c0007'), min_detail: 0.8 };
    expect(passingVisible(p, 0)).toBe(false);
    expect(passingVisible(p, 0.79)).toBe(false);
    expect(passingVisible(p, 0.8)).toBe(true);
    const all = buildTimeline(a, 1).filter((s) => s.passing).length;
    expect(all).toBe(a.chords.filter((c) => c.layer === 'passing').length);
    expect(buildTimeline(a, 0.8).filter((s) => s.passing).length).toBeLessThan(all);
  });

  it('finds the sounding segment by time', () => {
    const segs = buildTimeline(a, 1);
    const i = segAt(segs, 16.4);
    expect(segs[i].chord.id).toBe('c0007');
  });
});

describe('grid model', () => {
  const segs = buildTimeline(a, 1);
  const g = buildGrid(a, segs, buildBassTimeline(a.bass_line ?? [], 1));

  it('has one square per metered beat and free-time strips at both ends', () => {
    expect(g.squares.length).toBe(a.beats.filter((b) => b.bar >= 0).length);
    expect(g.lead?.t1).toBeCloseTo(a.bars[0].start, 3);
    expect(g.tail?.t0).toBeCloseTo(a.bars[a.bars.length - 1].end, 3);
  });

  it('splits squares on the 1/4-beat cell grid, parts covering each square exactly', () => {
    for (const s of g.squares) {
      const total = s.parts.reduce((x, p) => x + (p.f1 - p.f0), 0);
      expect(total).toBeCloseTo(1, 6);
      for (const p of s.parts) expect((p.f0 * 4) % 1).toBeCloseTo(0, 6);
    }
  });

  it('draws the half-beat passing F#/E in the second half of beat 15 (B ┊ F#/E)', () => {
    const sq = g.squares[14];
    expect(sq.parts.length).toBe(2);
    const p = sq.parts[1];
    expect(p.f0).toBeCloseTo(0.5, 6);
    expect(p.isStart).toBe(true);
    expect(segs[p.seg].chord.id).toBe('c0007');
    expect(segs[p.seg].passing).toBe(true);
  });

  it('keeps the flat part list in time order (binary search for the highlight)', () => {
    for (let i = 1; i < g.flat.length; i++) expect(g.flat[i].t).toBeGreaterThanOrEqual(g.flat[i - 1].t - 1e-9);
  });
});

describe('labels come from display.json (the client never spells)', () => {
  const ctx = { variant: variantOf(d, 1, 'auto'), level: 'standard' as const, slash: true, notation: 'letters' as const };

  it('uses the transposed variant', () => {
    expect(chordLabel(byId('c0005'), ctx)).toBe('Fmaj7/E');
    expect(chordLabel(byId('c0005'), { ...ctx, slash: false })).toBe('Fmaj7');
  });

  it('uses degree strings for 級數 / Roman, unchanged by transposition', () => {
    expect(chordLabel(byId('c0005'), { ...ctx, notation: 'nashville' })).toBe('4maj7/3');
    expect(chordLabel(byId('c0005'), { ...ctx, notation: 'nashville', level: 'simple' })).toBe('4');
    expect(chordLabel(byId('c0001'), { ...ctx, notation: 'nashville', level: 'simple' })).toBe('6m');
  });

  it('never strips a 6/9 quality as if it were a bass', () => {
    expect(stripLetterBass('C6/9')).toBe('C6/9');
    expect(stripLetterBass('F#/A#')).toBe('F#');
    expect(stripLetterBass('Bb/D')).toBe('Bb');
  });
});
