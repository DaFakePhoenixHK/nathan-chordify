// Song-level actions shared by the transport, the grid menu and the keyboard shortcuts.
import { engine } from '../player/engine';
import { revealCurrent } from '../player/sync';
import { getBassTimeline, getGrid, getTimeline, type GridModel } from '../grid/model';
import { setLoop, useApp } from './store';

export function gridModel(): GridModel | null {
  const { analysis, detail, gtMode } = useApp.getState();
  if (!analysis) return null;
  return getGrid(analysis, getTimeline(analysis, detail, gtMode), getBassTimeline(analysis, detail));
}
const model = gridModel;

/** The beat square `dir` squares away from the one containing t (editor Tab / Shift+Tab). */
export function neighborSquare(t: number, dir: number): { t0: number; t1: number } | null {
  const g = model();
  if (!g || g.squares.length === 0) return null;
  const sq = g.squares;
  let k = sq.findIndex((s) => s.t0 <= t + 1e-3 && t < s.t1 - 1e-3);
  if (k < 0) k = dir > 0 ? sq.findIndex((s) => s.t0 > t) - 1 : sq.findIndex((s) => s.t0 > t);
  const j = Math.min(sq.length - 1, Math.max(0, k + dir));
  return { t0: sq[j].t0, t1: sq[j].t1 };
}

/** The bar containing t (or the last one before it). */
export function barAt(t: number): { i: number; start: number; end: number } | null {
  const a = useApp.getState().analysis;
  if (!a || a.bars.length === 0) return null;
  let best = a.bars[0];
  for (const b of a.bars) if (b.start <= t + 0.01) best = b;
  return best;
}

/** The beat square [t0, t1) containing t, or null outside the metered bars. */
export function squareAt(t: number): { t0: number; t1: number } | null {
  const g = model();
  if (!g) return null;
  const sq = g.squares;
  let lo = 0;
  let hi = sq.length - 1;
  let k = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (sq[mid].t0 <= t + 1e-3) {
      k = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  if (k < 0 || t >= sq[k].t1) return null;
  return { t0: sq[k].t0, t1: sq[k].t1 };
}

/** `[` / A button: loop start at the start of the current beat. */
export function markLoopA(): void {
  const t = engine.time;
  const s = squareAt(t);
  const a = s ? s.t0 : t;
  const { b } = useApp.getState().loop;
  setLoop(a, b != null && b > a ? b : null);
}

/** `]` / B button: loop end at the end of the current beat. */
export function markLoopB(): void {
  const t = engine.time;
  const s = squareAt(t);
  const b = s ? s.t1 : t;
  const { a } = useApp.getState().loop;
  if (a == null) setLoop(Math.max(0, b - 4), b);
  else setLoop(a, b);
}

export function clearLoop(): void {
  setLoop(null, null);
}

export function loopBar(barIdx: number): void {
  const a = useApp.getState().analysis;
  const bar = a?.bars.find((b) => b.i === barIdx);
  if (!bar) return;
  setLoop(bar.start, bar.end);
  engine.seek(bar.start);
}

/** ← / → : previous / next bar. */
export function stepBar(dir: -1 | 1): void {
  const a = useApp.getState().analysis;
  if (!a || a.bars.length === 0) return;
  const t = engine.time;
  const bars = a.bars;
  let idx = -1;
  for (let i = 0; i < bars.length; i++) if (bars[i].start <= t + 0.01) idx = i;
  let target: number;
  if (dir > 0) {
    target = idx + 1 < bars.length ? bars[idx + 1].start : t;
  } else if (idx < 0) {
    target = 0;
  } else {
    // within the first 0.35 s of a bar, ← goes to the previous bar; otherwise to this bar's start
    target = t - bars[idx].start > 0.35 || idx === 0 ? bars[idx].start : bars[idx - 1].start;
  }
  engine.seek(target);
  revealCurrent();
}

export function seekTo(t: number): void {
  engine.seek(t);
}
