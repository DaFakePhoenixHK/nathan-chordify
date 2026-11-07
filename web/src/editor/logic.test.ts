import { describe, expect, it } from 'vitest';
import analysisJson from '../../../shared/fixtures/huijia.analysis.json';
import type { Analysis, EditOp } from '../api/types';
import { buildTimeline, gtVisible } from '../grid/model';
import { medianIbi, nearestBeat, nudgeNearest, tapBpm, tapInto } from '../gt/tap';
import { composeChord, historyRows, redoCandidate, undoCandidate } from './logic';

const a = analysisJson as unknown as Analysis;

const op = (seq: number, device: string, kind: EditOp['kind'] = 'set_chord', active = true, target = {}): EditOp => ({
  id: `o${seq}`,
  seq,
  device,
  kind,
  target,
  value: {},
  active,
});

describe('undo / redo (revert ops, §5.13)', () => {
  it('undoes this device’s latest live op first, then anyone’s', () => {
    const h = [op(1, 'phone-1'), op(2, 'desktop-a'), op(3, 'phone-1'), op(4, 'desktop-a', 'revert', true, { op_id: 'o2' })];
    h[1].active = false; // o2 was reverted by o4
    expect(undoCandidate(h, 'desktop-a')?.id).toBe('o3'); // desktop's own o2 is already undone → falls back to the latest live op
    expect(undoCandidate(h, 'phone-1')?.id).toBe('o3');
    expect(undoCandidate([op(1, 'x', 'set_chord', false)], 'x')).toBeNull();
    expect(undoCandidate([op(1, 'a'), op(2, 'b')], 'a')?.id).toBe('o1');
  });

  it('redoes the newest undo that is still in effect', () => {
    const h = [op(1, 'a'), op(2, 'a', 'revert'), op(3, 'a', 'revert', false)];
    expect(redoCandidate(h, ['o2', 'o3'])).toBe('o2');
    expect(redoCandidate(h, [])).toBeNull();
    expect(historyRows(h).map((o) => o.seq)).toEqual([3, 2, 1]);
  });

  it('composes picker chords as typed text', () => {
    expect(composeChord('F#', '', 'A#')).toBe('F#/A#');
    expect(composeChord('C#', 'm7', '')).toBe('C#m7');
    expect(composeChord('E', '', 'E')).toBe('E');
  });
});

describe('ground-truth view (§8.4, review E3)', () => {
  it('shows the main layer plus only the passing chords Nathan made or accepted', () => {
    const p = a.chords.find((c) => c.layer === 'passing')!;
    expect(gtVisible(p)).toBe(false);
    expect(gtVisible({ ...p, edited: true })).toBe(true);
    expect(gtVisible({ ...p, edited: true, suggestion: true })).toBe(false);
    const segs = buildTimeline(a, 1, true);
    expect(segs.every((s) => !s.passing)).toBe(true);
    const edited = { ...a, chords: a.chords.map((c) => (c.id === p.id ? { ...c, edited: true } : c)) };
    expect(buildTimeline(edited, 0, true).filter((s) => s.passing).map((s) => s.chord.id)).toEqual([p.id]);
  });
});

const grid = (n: number, t0 = 1, ibi = 0.9) => Array.from({ length: n }, (_, i) => Math.round((t0 + i * ibi) * 1e4) / 1e4);

describe('tapped beats (§8.4)', () => {

  it('adds taps and estimates the tempo', () => {
    let b: number[] = [];
    let prev: number | null = null;
    for (const t of grid(8)) {
      b = tapInto(b, t, prev);
      prev = t;
    }
    expect(b).toEqual(grid(8));
    expect(medianIbi(b)).toBeCloseTo(0.9);
    expect(tapBpm(b)).toBeCloseTo(66.67, 1);
  });

  it('re-tapping a stretch replaces it as you go', () => {
    const saved = grid(12);
    let b = saved;
    let prev: number | null = null;
    for (const t of [3.73, 4.63, 5.53]) {
      // shifted by +0.03 s
      b = tapInto(b, t, prev);
      prev = t;
    }
    expect(b.length).toBe(12);
    expect(b.slice(3, 6)).toEqual([3.73, 4.63, 5.53]);
    expect(b.slice(0, 3)).toEqual(saved.slice(0, 3));
    expect(b.slice(6)).toEqual(saved.slice(6));
  });

  it('nudges the nearest beat', () => {
    const b = nudgeNearest(grid(4), 2.0, 0.02);
    expect(b[1]).toBeCloseTo(1.92);
    expect(nearestBeat(b, 2.6)).toBe(2);
    expect(nearestBeat([], 1)).toBe(-1);
  });
});
