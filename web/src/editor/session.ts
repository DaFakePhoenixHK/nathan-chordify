// Editor session actions (§8.4): edit / ground-truth modes, selection, typed entry. Shared by the
// grid, the chord editor, the toolbars and the keyboard shortcuts.
import { api } from '../api/client';
import type { ChordOut } from '../api/types';
import { barAt, gridModel, neighborSquare, squareAt } from '../app/actions';
import { setLoop, setRate, showToast, useApp, type Sel } from '../app/store';
import { tNow } from '../i18n';
import { engine } from '../player/engine';
import { getTimeline, segAt } from '../grid/model';
import { shownShift } from '../grid/labels';
import { playChord } from '../gt/synth';
import { pollEdits, refreshHistory } from './ops';

function songKey(): string | null {
  const r = useApp.getState().route;
  return r.page === 'song' ? r.key : null;
}

let pollTimer = 0;
function startPolling(on: boolean): void {
  window.clearInterval(pollTimer);
  pollTimer = on ? window.setInterval(() => void pollEdits(), 10_000) : 0;
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && useApp.getState().editMode) void pollEdits();
});

export function setEditMode(on: boolean): void {
  const key = songKey();
  if (!key) return;
  if (on && api.mock) showToast(tNow('editNeedsServer'), 'error');
  const st = useApp.getState();
  useApp.setState({
    editMode: on,
    gtMode: on ? st.gtMode : false,
    sel: on ? st.sel : null,
    tapMode: on ? st.tapMode : false,
    historyOpen: on ? st.historyOpen : false,
    menu: null,
  });
  if (!on) stopBarLoop();
  startPolling(on);
  if (on) {
    void refreshHistory(key);
    void pollEdits();
  }
}

export function toggleEditMode(): void {
  setEditMode(!useApp.getState().editMode);
}

export async function loadGt(): Promise<void> {
  const key = songKey();
  if (!key || api.mock) return;
  try {
    const gt = await api.gtState(key);
    if (songKey() === key) useApp.setState({ gt });
  } catch (e) {
    showToast(e instanceof Error ? e.message : String(e), 'error');
  }
}

export function setGtMode(on: boolean): void {
  if (on && !useApp.getState().editMode) setEditMode(true);
  useApp.setState({ gtMode: on, tapMode: false });
  if (on) void loadGt();
  else stopBarLoop();
}

// ---------------------------------------------------------------- selection

export function selectChord(c: ChordOut): void {
  useApp.setState({ sel: { kind: 'chord', id: c.id, t0: c.start, t1: c.end, layer: c.layer }, entry: '' });
}

export function selectRange(t0: number, t1: number): void {
  if (t1 <= t0) return;
  useApp.setState({ sel: { kind: 'range', t0, t1 }, entry: '' });
}

export function clearSel(): void {
  useApp.setState({ sel: null, entry: '' });
}

/** Tab / Shift+Tab: the next / previous beat square becomes the selection. */
export function moveSel(dir: 1 | -1): void {
  const { sel } = useApp.getState();
  const sq = sel ? neighborSquare(sel.t0 + 1e-3, dir) : neighborSquare(engine.time, 0);
  if (sq) {
    useApp.setState({ sel: { kind: 'range', t0: sq.t0, t1: sq.t1 }, entry: '' });
    revealSel(sq.t0);
  }
}

/** The beat square of the selection start (for "just this beat"). */
export function beatOf(sel: Sel): { t0: number; t1: number } | null {
  return squareAt(Math.max(sel.t0, Math.min(engine.time, sel.t1 - 1e-3)) + 1e-4) ?? squareAt(sel.t0 + 1e-4);
}

/** Start typed entry with a first character (keyboard-first, §8.4). */
export function startEntry(text: string): void {
  useApp.setState((s) => ({ entry: text, entryFocus: s.entryFocus + 1 }));
}

function revealSel(t: number): void {
  const k = gridModel()?.squares.findIndex((s) => Math.abs(s.t0 - t) < 1e-3) ?? -1;
  if (k >= 0) document.querySelector<HTMLElement>(`[data-sq="${k}"]`)?.scrollIntoView({ block: 'nearest' });
}

// ---------------------------------------------------------------- ground-truth tools

let barLoopPrevRate: number | null = null;

export function barLoopOn(): boolean {
  return barLoopPrevRate != null;
}

export function stopBarLoop(): void {
  if (barLoopPrevRate == null) return;
  setRate(barLoopPrevRate);
  barLoopPrevRate = null;
  setLoop(null, null);
}

/** L: loop the bar under the playhead at 70% (§8.4); again to stop. */
export function toggleBarLoop(): void {
  if (barLoopPrevRate != null) {
    stopBarLoop();
    useApp.setState({}); // re-render the toolbar
    return;
  }
  const t = useApp.getState().sel?.t0 ?? engine.time;
  const bar = barAt(t + 1e-3);
  if (!bar) return;
  barLoopPrevRate = useApp.getState().rate;
  setRate(0.7);
  setLoop(bar.start, bar.end);
  engine.seek(bar.start);
  void engine.play();
}

/** P: play the chord at the selection (or the playhead) on the minimal synth, with the server's voicing. */
export function playChordAt(): void {
  const { analysis, display, detail, gtMode, sel, transpose, playIn } = useApp.getState();
  if (!analysis) return;
  const shift = shownShift(transpose, playIn);
  let id: string | null = sel?.kind === 'chord' ? sel.id : null;
  if (!id) {
    const segs = getTimeline(analysis, detail, gtMode);
    const k = segAt(segs, (sel?.t0 ?? engine.time) + 1e-3);
    id = k >= 0 ? segs[k].chord.id : null;
  }
  if (!id) return;
  const v = display?.voicings[String(shift)]?.[id] ?? (shift === 0 ? analysis.chords.find((c) => c.id === id)?.voicing : undefined);
  const notes = [...(v?.lh ?? []), ...(v?.rh ?? [])];
  if (notes.length === 0) return;
  engine.pause();
  playChord(notes);
}
