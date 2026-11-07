// Pure editor logic (no DOM, no store): unit-tested in logic.test.ts.
import type { EditOp } from '../api/types';

/** The op Ctrl+Z reverts: this device's latest op still in effect, else anyone's. */
export function undoCandidate(history: EditOp[], dev: string): EditOp | null {
  const live = history.filter((o) => o.active !== false && o.kind !== 'revert');
  for (let i = live.length - 1; i >= 0; i--) if (live[i].device === dev) return live[i];
  return live.length ? live[live.length - 1] : null;
}

/** The undo (a revert op id) Ctrl+Shift+Z reverts, from this session's undo stack (newest last). */
export function redoCandidate(history: EditOp[], undone: string[]): string | null {
  for (let i = undone.length - 1; i >= 0; i--) {
    const op = history.find((o) => o.id === undone[i]);
    if (op && op.active !== false) return op.id;
  }
  return null;
}

/** Ops the history list shows, newest first. */
export function historyRows(history: EditOp[]): EditOp[] {
  return history.toReversed();
}

/** Compose a typed chord from the picker: root + quality suffix + optional "/bass". */
export function composeChord(root: string, suffix: string, bass: string): string {
  return root + suffix + (bass && bass !== root ? `/${bass}` : '');
}
