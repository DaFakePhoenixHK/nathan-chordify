// Pure layout model for the chord grid. No chord theory here: labels come from display.json (labels.ts).
import type { Analysis, BassNoteOut, ChordOut, KeyChangeOut } from '../api/types';

const EPS = 1e-4;

// ---------------------------------------------------------------- visible timeline (§5.11)

/** One stretch of the visible chord sequence (main layer with the visible passing chords cut in). */
export interface Seg {
  chord: ChordOut;
  /** Index in analysis.chords. */
  ci: number;
  start: number;
  end: number;
  /** Passing-layer chord (drawn smaller, accent colour). */
  passing: boolean;
  /** A main chord resuming after a passing chord cut into it. */
  resumed: boolean;
  /** Index into analysis.bass_line for a bass passing tone drawn as a slash chord (Everything), else -1. */
  bassTone: number;
}

/** A passing chord is shown when s > 0 and s >= min_detail (§5.11). */
export function passingVisible(c: ChordOut, detail: number): boolean {
  return c.layer === 'passing' && detail > 0 && detail >= (c.min_detail ?? 0) - 1e-9;
}

/** Bass passing tones show in the bass lane from Detailed (s >= 0.8), as slash chords in the grid at Everything (s = 1). */
export const BASS_TONE_LANE_MIN = 0.8;
export const BASS_TONE_GRID_MIN = 1;

function carve<T extends { start: number; end: number }>(list: T[], ins: T, resume: (s: T) => T): T[] {
  const out: T[] = [];
  for (const s of list) {
    if (s.end <= ins.start + EPS || s.start >= ins.end - EPS) {
      out.push(s);
      continue;
    }
    if (s.start < ins.start - EPS) out.push({ ...s, end: ins.start });
    if (s.end > ins.end + EPS) out.push(resume({ ...s, start: ins.end }));
  }
  let k = out.findIndex((s) => s.start >= ins.start - EPS);
  if (k < 0) k = out.length;
  out.splice(k, 0, ins);
  return out;
}

/**
 * Ground-truth view (§8.4, review E3): the main layer plus the passing chords Nathan made or accepted.
 * Our own passing guesses and pending suggestions stay hidden so they can't anchor the annotation.
 */
export function gtVisible(c: ChordOut): boolean {
  return c.layer === 'passing' && !!c.edited && !c.suggestion;
}

export function buildTimeline(a: Analysis, detail: number, gt = false): Seg[] {
  const idx = a.chords.map((c, i) => ({ c, i }));
  let segs: Seg[] = idx
    .filter((x) => x.c.layer === 'main')
    .toSorted((p, q) => p.c.start - q.c.start)
    .map((x) => ({ chord: x.c, ci: x.i, start: x.c.start, end: x.c.end, passing: false, resumed: false, bassTone: -1 }));
  const resume = (s: Seg): Seg => ({ ...s, resumed: true });
  const show = gt ? gtVisible : (c: ChordOut) => passingVisible(c, detail);
  for (const x of idx.filter((y) => show(y.c)).toSorted((p, q) => p.c.start - q.c.start)) {
    segs = carve(segs, { chord: x.c, ci: x.i, start: x.c.start, end: x.c.end, passing: true, resumed: false, bassTone: -1 }, resume);
  }
  if (!gt && detail >= BASS_TONE_GRID_MIN) {
    (a.bass_line ?? []).forEach((b, bi) => {
      if (b.role !== 'passing_tone') return;
      const host = segs.find((s) => s.start <= b.start + EPS && s.end > b.start + EPS);
      if (!host) return;
      segs = carve(segs, { ...host, start: b.start, end: b.end, passing: true, resumed: false, bassTone: bi }, resume);
    });
  }
  return segs;
}

/** Signature of what's visible at this detail value, so the timeline is only rebuilt when it changes. */
function visibilitySig(a: Analysis, detail: number, gt: boolean): string {
  if (gt) return 'G';
  let sig = detail >= BASS_TONE_GRID_MIN ? 'T' : 'F';
  for (const c of a.chords) if (c.layer === 'passing') sig += passingVisible(c, detail) ? '1' : '0';
  return sig;
}

let tlCache: { a: Analysis; sig: string; segs: Seg[] } | null = null;
/** Memoized: the same array instance as long as the visible set is unchanged (stable seg indices). */
export function getTimeline(a: Analysis, detail: number, gt = false): Seg[] {
  const sig = visibilitySig(a, detail, gt);
  if (tlCache && tlCache.a === a && tlCache.sig === sig) return tlCache.segs;
  const segs = buildTimeline(a, detail, gt);
  tlCache = { a, sig, segs };
  return segs;
}

// ---------------------------------------------------------------- bass lane

export interface BassSeg {
  bi: number;
  start: number;
  end: number;
  /** Bass of a passing chord. */
  passing: boolean;
  /** role = passing_tone (greyed). */
  tone: boolean;
}

export function buildBassTimeline(bass: BassNoteOut[], detail: number): BassSeg[] {
  const idx = bass.map((b, bi) => ({ b, bi }));
  let segs: BassSeg[] = idx
    .filter((x) => x.b.role === 'chord_bass' && x.b.min_detail == null)
    .toSorted((p, q) => p.b.start - q.b.start)
    .map((x) => ({ bi: x.bi, start: x.b.start, end: x.b.end, passing: false, tone: false }));
  const extra = idx
    .filter((x) =>
      x.b.role === 'passing_tone'
        ? detail >= BASS_TONE_LANE_MIN
        : x.b.min_detail != null && detail > 0 && detail >= x.b.min_detail - 1e-9,
    )
    .toSorted((p, q) => p.b.start - q.b.start);
  for (const x of extra) {
    segs = carve(segs, { bi: x.bi, start: x.b.start, end: x.b.end, passing: x.b.role !== 'passing_tone', tone: x.b.role === 'passing_tone' }, (s) => s);
  }
  return segs;
}

// ---------------------------------------------------------------- squares and positions

export interface Part {
  /** Seg index (chord lane) or BassSeg index (bass lane); -1 = nothing sounding. */
  seg: number;
  f0: number;
  f1: number;
  /** The seg starts in this square (draw its label); otherwise it continues from before. */
  isStart: boolean;
}

export interface Square {
  k: number;
  bar: number;
  /** 1-based beat in the bar. */
  beat: number;
  t0: number;
  t1: number;
  /** Columns this square spans (1, or a whole free-time bar). */
  span: number;
  free: boolean;
  cpb: number;
  parts: Part[];
  bass: Part[];
}

export interface BarView {
  i: number;
  /** Indices into GridModel.squares. */
  squares: number[];
  beats: number;
  keyChange: KeyChangeOut | null;
  /** Key segment index for the key-change badge (display key_names index). */
  keySeg: number;
  section: string | null;
}

export interface Strip {
  kind: 'lead' | 'tail';
  t0: number;
  t1: number;
  parts: Part[];
  bass: Part[];
}

export interface FlatPart {
  t: number;
  seg: number;
  /** Square index, or -1 (lead strip) / -2 (tail strip). */
  sq: number;
}

export interface GridModel {
  squares: Square[];
  bars: BarView[];
  lead: Strip | null;
  tail: Strip | null;
  beatsPerBar: number;
  /** Every chord-lane part in time order; DOM elements carry data-pi = index here. */
  flat: FlatPart[];
}

function cellsPerBeat(a: Analysis, bar: number): number {
  let cpb = 4;
  for (const s of a.tempo.subdivision ?? []) if (s.from_bar <= bar) cpb = s.cells_per_beat;
  return cpb;
}

export function keySegmentIndexAt(a: Analysis, t: number): number {
  const segs = a.key.segments;
  for (let i = 0; i < segs.length; i++) if (t >= segs[i].start - 0.05 && t < segs[i].end - 0.05) return i;
  return Math.max(0, segs.length - 1);
}

/** Index of the key segment that matches the global key (display key_names index). */
export function globalKeySegment(a: Analysis): number {
  const segs = a.key.segments;
  const g = a.key.global;
  const i = segs.findIndex((s) => s.tonic === g.tonic && s.mode === g.mode);
  if (i >= 0) return i;
  let best = 0;
  segs.forEach((s, j) => {
    if (s.end - s.start > segs[best].end - segs[best].start) best = j;
  });
  return best;
}

export function buildSquares(a: Analysis): Square[] {
  const squares: Square[] = [];
  const beats = a.beats.filter((b) => b.bar >= 0).toSorted((p, q) => p.t - q.t);
  const barsByIdx = new Map(a.bars.map((b) => [b.i, b]));
  const freeDone = new Set<number>();
  for (let j = 0; j < beats.length; j++) {
    const b = beats[j];
    const bar = barsByIdx.get(b.bar);
    if (!bar) continue;
    if (bar.free) {
      if (freeDone.has(bar.i)) continue;
      freeDone.add(bar.i);
      squares.push({ k: squares.length, bar: bar.i, beat: 1, t0: bar.start, t1: bar.end, span: Math.max(1, bar.beats), free: true, cpb: 0, parts: [], bass: [] });
      continue;
    }
    const nb = beats[j + 1];
    const t1 = nb && (nb.bar === b.bar || nb.bar === b.bar + 1) ? nb.t : bar.end;
    if (t1 <= b.t) continue;
    squares.push({ k: squares.length, bar: bar.i, beat: b.beat, t0: b.t, t1, span: 1, free: false, cpb: cellsPerBeat(a, bar.i), parts: [], bass: [] });
  }
  return squares;
}

/** Quantized position in "square units": k + fraction (fraction snapped to the cell grid). */
export function makePosOf(squares: Square[]): (t: number) => number {
  const n = squares.length;
  return (t: number) => {
    if (n === 0 || t < squares[0].t0 - EPS) return -Infinity;
    let lo = 0;
    let hi = n - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (squares[mid].t0 <= t + EPS) lo = mid;
      else hi = mid - 1;
    }
    const s = squares[lo];
    if (t >= s.t1 - EPS) return lo === n - 1 ? (t > s.t1 + EPS ? Infinity : n) : lo + 1;
    let f = (t - s.t0) / (s.t1 - s.t0);
    if (!s.free) f = Math.round(f * s.cpb) / s.cpb;
    return lo + Math.min(Math.max(f, 0), 1);
  };
}

function fillGaps(parts: Part[]): Part[] {
  const out: Part[] = [];
  let at = 0;
  for (const p of parts.toSorted((x, y) => x.f0 - y.f0)) {
    if (p.f0 > at + 1e-6) out.push({ seg: -1, f0: at, f1: p.f0, isStart: true });
    out.push(p);
    at = Math.max(at, p.f1);
  }
  if (at < 1 - 1e-6) out.push({ seg: -1, f0: at, f1: 1, isStart: true });
  return out;
}

function distribute(
  items: { start: number; end: number }[],
  squares: Square[],
  posOf: (t: number) => number,
  pick: (s: Square) => Part[],
): void {
  const n = squares.length;
  items.forEach((it, si) => {
    const qs = posOf(it.start);
    const qe = posOf(it.end);
    if (!(qe > qs)) return;
    const k0 = Math.max(0, Math.floor(qs + 1e-9));
    const k1 = Math.min(n - 1, Math.ceil(qe - 1e-9) - 1);
    for (let k = k0; k <= k1; k++) {
      const f0 = Math.max(qs - k, 0);
      const f1 = Math.min(qe - k, 1);
      if (f1 - f0 > 1e-6) pick(squares[k]).push({ seg: si, f0, f1, isStart: qs >= k - 1e-9 });
    }
  });
}

function stripParts(items: { start: number; end: number }[], t0: number, t1: number): Part[] {
  const d = t1 - t0;
  const parts: Part[] = [];
  items.forEach((it, si) => {
    if (it.end <= t0 + EPS || it.start >= t1 - EPS) return;
    parts.push({
      seg: si,
      f0: Math.max(0, (it.start - t0) / d),
      f1: Math.min(1, (it.end - t0) / d),
      isStart: it.start >= t0 - 0.02,
    });
  });
  return fillGaps(parts);
}

export function buildGrid(a: Analysis, segs: Seg[], bassSegs: BassSeg[]): GridModel {
  const squares = buildSquares(a);
  const posOf = makePosOf(squares);
  distribute(segs, squares, posOf, (s) => s.parts);
  distribute(bassSegs, squares, posOf, (s) => s.bass);
  for (const s of squares) {
    s.parts = fillGaps(s.parts);
    s.bass = fillGaps(s.bass);
  }

  const changes = a.key.changes ?? [];
  const bars: BarView[] = [];
  const byBar = new Map<number, number[]>();
  for (const s of squares) {
    const l = byBar.get(s.bar);
    if (l) l.push(s.k);
    else byBar.set(s.bar, [s.k]);
  }
  let prevSection: string | null = null;
  for (const bar of a.bars) {
    const sq = byBar.get(bar.i);
    if (!sq) continue;
    const kc = changes.find((c) => c.bar === bar.i) ?? null;
    const section = bar.section ?? null;
    bars.push({
      i: bar.i,
      squares: sq,
      beats: bar.free ? Math.max(1, bar.beats) : sq.length,
      keyChange: kc,
      keySeg: kc ? keySegmentIndexAt(a, kc.t + 0.06) : -1,
      section: section && section !== prevSection ? section : null,
    });
    prevSection = section;
  }

  const duration = a.source.duration;
  const first = squares[0];
  const last = squares[squares.length - 1];
  const lead: Strip | null =
    first && first.t0 > 0.25
      ? { kind: 'lead', t0: 0, t1: first.t0, parts: stripParts(segs, 0, first.t0), bass: stripParts(bassSegs, 0, first.t0) }
      : null;
  const tail: Strip | null =
    last && duration - last.t1 > 0.25
      ? { kind: 'tail', t0: last.t1, t1: duration, parts: stripParts(segs, last.t1, duration), bass: stripParts(bassSegs, last.t1, duration) }
      : null;

  const flat: FlatPart[] = [];
  const pushStrip = (st: Strip | null, sq: number) => {
    if (!st) return;
    for (const p of st.parts) flat.push({ t: st.t0 + p.f0 * (st.t1 - st.t0), seg: p.seg, sq });
  };
  pushStrip(lead, -1);
  for (const s of squares) for (const p of s.parts) flat.push({ t: s.t0 + p.f0 * (s.t1 - s.t0), seg: p.seg, sq: s.k });
  pushStrip(tail, -2);

  const beatsPerBar = Math.max(1, ...a.meter.map((m) => m.num), ...bars.map((b) => b.beats));
  return { squares, bars, lead, tail, beatsPerBar, flat };
}

let gridCache: { a: Analysis; segs: Seg[]; bass: BassSeg[]; g: GridModel } | null = null;
export function getGrid(a: Analysis, segs: Seg[], bass: BassSeg[]): GridModel {
  if (gridCache && gridCache.a === a && gridCache.segs === segs && gridCache.bass === bass) return gridCache.g;
  const g = buildGrid(a, segs, bass);
  gridCache = { a, segs, bass, g };
  return g;
}

let bassCache: { a: Analysis; sig: string; segs: BassSeg[] } | null = null;
export function getBassTimeline(a: Analysis, detail: number): BassSeg[] {
  const bl = a.bass_line ?? [];
  let sig = detail >= BASS_TONE_LANE_MIN ? 'T' : 'F';
  for (const b of bl) if (b.min_detail != null && b.role !== 'passing_tone') sig += detail > 0 && detail >= b.min_detail - 1e-9 ? '1' : '0';
  if (bassCache && bassCache.a === a && bassCache.sig === sig) return bassCache.segs;
  const segs = buildBassTimeline(bl, detail);
  bassCache = { a, sig, segs };
  return segs;
}

/** Group bars into rows of `barsPerRow`. */
export function rowsOf(g: GridModel, barsPerRow: number): BarView[][] {
  const rows: BarView[][] = [];
  for (let i = 0; i < g.bars.length; i += barsPerRow) rows.push(g.bars.slice(i, i + barsPerRow));
  return rows;
}

/** Index of the last element of a sorted array that is <= t (or -1). */
export function lastAtOrBefore(starts: ArrayLike<number>, t: number): number {
  let lo = 0;
  let hi = starts.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (starts[mid] <= t) {
      ans = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return ans;
}

/** Seg index sounding at time t, or -1. */
export function segAt(segs: Seg[], t: number): number {
  let lo = 0;
  let hi = segs.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (segs[mid].start <= t + EPS) {
      ans = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return ans >= 0 && t < segs[ans].end + EPS ? ans : -1;
}
