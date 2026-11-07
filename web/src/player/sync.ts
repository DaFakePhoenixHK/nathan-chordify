// The highlight loop (§8.3): requestAnimationFrame reads the audio clock, binary-searches the grid parts and
// toggles CSS classes directly. React only hears about it when the current chord changes (nowSeg).
import { useApp } from '../app/store';
import { lastAtOrBefore, segAt, type Seg } from '../grid/model';

export interface Clock {
  readonly time: number;
  readonly paused: boolean;
  seek(t: number): void;
}

export interface GridTargets {
  starts: Float64Array;
  parts: HTMLElement[];
  /** Square (or strip) element of each part. */
  boxes: HTMLElement[];
  /** Row index of each part. */
  rows: Int32Array;
  rowEls: HTMLElement[];
  scroller: HTMLElement | null;
}

export interface ClockEls {
  time?: HTMLElement | null;
  seek?: HTMLInputElement | null;
}

let clock: Clock | null = null;
let grid: GridTargets | null = null;
let segs: Seg[] = [];
let clockEls: ClockEls = {};
let cur = -1;
let curBox: HTMLElement | null = null;
let curRow = -1;
let lastTimeText = '';
let rafId = 0;
let running = false;
let pending = 0;
let userScrollAt = 0;

/** Dev-only counters used to check the M3 gate (60 fps, no per-frame renders). */
export const perf = {
  frames: 0,
  maxGapMs: 0,
  lastFrame: 0,
  renders: {} as Record<string, number>,
  /** Dev inspection: the clock and the part start times the highlight uses. */
  clock: null as Clock | null,
  starts: null as Float64Array | null,
};
if (import.meta.env.DEV) (window as unknown as { ncPerf: typeof perf }).ncPerf = perf;

export function registerClock(c: Clock): void {
  clock = c;
  if (import.meta.env.DEV) perf.clock = c;
}

export function setTimeline(s: Seg[]): void {
  segs = s;
  requestTick();
}

export function setClockEls(els: ClockEls): void {
  clockEls = { ...clockEls, ...els };
  lastTimeText = '';
  requestTick();
}

export function setGridTargets(t: GridTargets | null): void {
  if (grid) {
    grid.parts[cur]?.classList.remove('is-now');
    curBox?.classList.remove('is-now');
  }
  grid = t;
  if (import.meta.env.DEV) perf.starts = t?.starts ?? null;
  cur = -1;
  curBox = null;
  curRow = -1;
  requestTick();
}

/** The user scrolled the grid by hand: pause auto-follow for a few seconds. */
export function noteUserScroll(): void {
  userScrollAt = performance.now();
}

export function formatTime(t: number, tenths = false): string {
  if (!Number.isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return tenths ? `${m}:${s.toFixed(1).padStart(4, '0')}` : `${m}:${Math.floor(s).toString().padStart(2, '0')}`;
}

function autoscroll(rowEl: HTMLElement, sc: HTMLElement): void {
  if (performance.now() - userScrollAt < 4000) return;
  if (sc.offsetParent === null || sc.clientHeight === 0) return; // hidden (phone Diagrams tab)
  const top = rowEl.offsetTop;
  const rel = top - sc.scrollTop;
  const h = sc.clientHeight;
  // Keep the current row in the upper third of the view.
  if (rel < 0 || rel > h / 3) {
    sc.scrollTo({ top: Math.max(0, top - h * 0.12), behavior: 'smooth' });
  }
}

function tick(): void {
  if (!clock) return;
  let t = clock.time;
  const st = useApp.getState();

  // A/B loop (seek-based, ±30 ms; gapless loops arrive with signalsmith-stretch in M7).
  const { a, b } = st.loop;
  if (!clock.paused && a != null && b != null && b > a + 0.05 && t >= b) {
    clock.seek(a);
    t = a;
  }

  // Current chord (for NOW/NEXT/THEN): only a store write when it changes.
  const ns = segAt(segs, t);
  if (ns !== st.nowSeg) useApp.setState({ nowSeg: ns });

  if (grid) {
    const i = lastAtOrBefore(grid.starts, t);
    if (i !== cur) {
      grid.parts[cur]?.classList.remove('is-now');
      const box = i >= 0 ? grid.boxes[i] : null;
      if (box !== curBox) {
        curBox?.classList.remove('is-now');
        box?.classList.add('is-now');
        curBox = box;
      }
      if (i >= 0) grid.parts[i].classList.add('is-now');
      cur = i;
      const row = i >= 0 ? grid.rows[i] : -1;
      if (row !== curRow) {
        curRow = row;
        const rowEl = row >= 0 ? grid.rowEls[row] : null;
        if (rowEl && grid.scroller && !clock.paused) autoscroll(rowEl, grid.scroller);
      }
    }
  }

  const tt = formatTime(t, true);
  if (tt !== lastTimeText && clockEls.time) {
    clockEls.time.textContent = tt;
    lastTimeText = tt;
  }
  const sk = clockEls.seek;
  if (sk && sk.dataset.dragging !== '1') sk.value = String(t);
}

function frame(now: number): void {
  if (!running) return;
  if (perf.lastFrame) perf.maxGapMs = Math.max(perf.maxGapMs, now - perf.lastFrame);
  perf.lastFrame = now;
  perf.frames++;
  tick();
  rafId = requestAnimationFrame(frame);
}

export function startLoop(): void {
  if (running) return;
  running = true;
  perf.lastFrame = 0;
  rafId = requestAnimationFrame(frame);
}

export function stopLoop(): void {
  running = false;
  cancelAnimationFrame(rafId);
}

/** One tick on the next frame (paused seeks, layout changes). */
export function requestTick(): void {
  if (running || pending) return;
  pending = requestAnimationFrame(() => {
    pending = 0;
    tick();
  });
}

/** Scroll the current row into view now (e.g. after a seek while paused). */
export function revealCurrent(): void {
  if (!grid || cur < 0 || !grid.scroller) return;
  const rowEl = grid.rowEls[grid.rows[cur]];
  if (rowEl) {
    userScrollAt = 0;
    autoscroll(rowEl, grid.scroller);
  }
}

if (import.meta.env.DEV) (perf as typeof perf & { tick: () => void }).tick = tick;
