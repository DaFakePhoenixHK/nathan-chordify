import { useEffect } from 'react';
import { engine } from '../player/engine';
import { markLoopA, markLoopB, stepBar } from './actions';
import { cycleDetail, nudgeRate, nudgeTranspose, useApp } from './store';
import { chordById, deleteChange, redo, undo } from '../editor/ops';
import { clearSel, moveSel, playChordAt, startEntry, toggleBarLoop, toggleEditMode } from '../editor/session';
import { nudgeTap, stopTapping, tapNow } from '../gt/GtBar';

function isTyping(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  if (el.isContentEditable) return true;
  if (el instanceof HTMLTextAreaElement || el instanceof HTMLSelectElement) return true;
  if (el instanceof HTMLInputElement) return !['range', 'checkbox', 'radio', 'button'].includes(el.type);
  return false;
}

/** Edit mode (§8.4, keyboard-first): Ctrl+Z / Ctrl+Shift+Z, Tab, letters start typed entry, Del, L, P;
 *  tap mode owns Space and ←/→. Returns true when the key was handled. */
function editKeys(e: KeyboardEvent): boolean {
  const st = useApp.getState();
  if (st.tapMode) {
    if (e.code === 'Space' || e.key === ' ') {
      tapNow();
      (document.activeElement as HTMLElement | null)?.blur?.();
      return true;
    }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      nudgeTap(e.key === 'ArrowLeft' ? -0.02 : 0.02);
      return true;
    }
  }
  if (!st.editMode) return false;
  if (e.key === 'Tab') {
    moveSel(e.shiftKey ? -1 : 1);
    return true;
  }
  if (st.sel) {
    if (/^[a-gnA-GN]$/.test(e.key)) {
      startEntry(e.key.toUpperCase());
      return true;
    }
    if (e.key === 'Enter') {
      startEntry('');
      return true;
    }
    if ((e.key === 'Delete' || e.key === 'Backspace') && st.sel.kind === 'chord') {
      const c = chordById(st.analysis, st.sel.id);
      if (c) void deleteChange(c);
      return true;
    }
  }
  if (st.gtMode && e.code === 'KeyL') {
    toggleBarLoop();
    return true;
  }
  if (st.gtMode && e.code === 'KeyP') {
    playChordAt();
    return true;
  }
  return false;
}

/** Keyboard shortcuts of §8.1 and §8.4 (song page only). */
export function useShortcuts(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const st = useApp.getState();
      const typing = isTyping(e.target);
      if ((e.ctrlKey || e.metaKey) && !e.altKey && st.editMode && !typing) {
        const k = e.key.toLowerCase();
        if (k === 'z' || k === 'y') {
          e.preventDefault();
          void (k === 'y' || e.shiftKey ? redo() : undo());
          return;
        }
      }
      if (e.ctrlKey || e.metaKey || e.altKey || typing) return;
      if (e.key === 'Escape') {
        if (st.menu || st.sheetOpen || st.chartOpen || st.sheetsOpen) useApp.setState({ menu: null, sheetOpen: false, chartOpen: false, sheetsOpen: false });
        else if (st.tapMode) void stopTapping();
        else if (st.sel) clearSel();
        return;
      }
      if (editKeys(e)) {
        e.preventDefault();
        return;
      }
      const plus = e.code === 'Equal' || e.code === 'NumpadAdd' || e.key === '+';
      const minus = e.code === 'Minus' || e.code === 'NumpadSubtract' || e.key === '-' || e.key === '_';
      let handled = true;
      if (e.code === 'Space' || e.key === ' ') {
        engine.toggle();
        (document.activeElement as HTMLElement | null)?.blur?.();
      } else if (e.key === 'ArrowLeft') stepBar(-1);
      else if (e.key === 'ArrowRight') stepBar(1);
      else if (e.key === '[') markLoopA();
      else if (e.key === ']') markLoopB();
      else if (plus) {
        // Shift + "+" = tempo; "+" (or "=" key) = transpose
        if (e.shiftKey && e.code !== 'NumpadAdd') nudgeRate(0.05);
        else nudgeTranspose(1);
      } else if (minus) {
        if (e.shiftKey) nudgeRate(-0.05);
        else nudgeTranspose(-1);
      } else if (e.code === 'KeyD') cycleDetail();
      else if (e.code === 'KeyE') toggleEditMode();
      else handled = false;
      if (handled) e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}
