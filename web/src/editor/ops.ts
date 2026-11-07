// Edit ops (§5.13, §8.4): client-made ids, one queue per tab, idempotent retries, undo = revert op.
// No chord theory here: chords go to the server as Harte (from POST /api/theory/parse) or as typed text.
import { api, ApiError } from '../api/client';
import type { Analysis, ChordOut, EditKind, EditOp, EditsPosted } from '../api/types';
import { showToast, useApp, type Sel } from '../app/store';
import { PHONE_QUERY } from '../app/hooks';
import { tNow } from '../i18n';
import { redoCandidate, undoCandidate } from './logic';

// ---------------------------------------------------------------- ids

/** uuid4 from getRandomValues (crypto.randomUUID needs a secure context; the phone may use plain HTTP). */
export function newId(): string {
  const b = new Uint8Array(16);
  crypto.getRandomValues(b);
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

let device: string | null = null;
/** "desktop-ab12" / "phone-ab12", remembered per browser. */
export function deviceId(): string {
  if (device) return device;
  const kind = window.matchMedia(PHONE_QUERY).matches ? 'phone' : 'desktop';
  try {
    device = localStorage.getItem('nc.device');
    if (!device) {
      device = `${kind}-${newId().slice(0, 4)}`;
      localStorage.setItem('nc.device', device);
    }
  } catch {
    device = `${kind}-${newId().slice(0, 4)}`;
  }
  return device;
}

export function makeOp(kind: EditKind, target: Record<string, unknown> = {}, value: Record<string, unknown> = {}): EditOp {
  return { id: newId(), kind, target, value, device: deviceId() };
}

// ---------------------------------------------------------------- song helpers

export function chordById(a: Analysis | null, id: string): ChordOut | null {
  return a?.chords.find((c) => c.id === id) ?? null;
}

/** The local key at time t as "B:major" (for typed entry and labels). */
export function keyAt(a: Analysis, t: number): string {
  const segs = a.key.segments;
  const s = segs.find((x) => x.start <= t + 1e-6 && t < x.end) ?? segs[segs.length - 1];
  return s ? `${s.tonic}:${s.mode}` : `${a.key.global.tonic}:${a.key.global.mode}`;
}

export function chordTarget(c: ChordOut): Record<string, unknown> {
  return { chord_id: c.id, t0: c.start, t1: c.end, layer: c.layer };
}

export function selTarget(sel: Sel): Record<string, unknown> {
  return sel.kind === 'chord'
    ? { chord_id: sel.id, t0: sel.t0, t1: sel.t1, layer: sel.layer }
    : { t0: round4(sel.t0), t1: round4(sel.t1), mode: 'range', layer: 'main' };
}

const round4 = (x: number) => Math.round(x * 1e4) / 1e4;

// ---------------------------------------------------------------- undo / redo

export { historyRows, redoCandidate, undoCandidate } from './logic';

// ---------------------------------------------------------------- queue

let chain: Promise<unknown> = Promise.resolve();
const undoStack: string[] = [];
const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

function songKey(): string | null {
  const r = useApp.getState().route;
  return r.page === 'song' ? r.key : null;
}

/** Put a server answer on screen (unless it is older than what's there). */
export function applyResult(key: string, r: { analysis: Analysis; display: unknown; rev?: number }): void {
  const st = useApp.getState();
  if (songKey() !== key) return;
  const rev = r.rev ?? r.analysis.edits?.rev ?? 0;
  if (rev < st.rev && st.analysis?.source.key === key) return;
  useApp.setState({ analysis: r.analysis, display: r.display as never, rev });
}

export async function refreshHistory(key: string): Promise<void> {
  try {
    const log = await api.getEdits(key);
    if (songKey() === key) useApp.setState({ history: log.ops });
  } catch {
    /* the next edit or poll retries */
  }
}

async function postWithRetry(key: string, ops: EditOp[]): Promise<EditsPosted> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await api.postEdits(key, ops, deviceId());
    } catch (e) {
      // 4xx: the op is wrong, retrying won't help. 0 (offline) / 5xx: retry with the same ids.
      const retry = e instanceof ApiError && (e.status === 0 || e.status >= 500) && e.status !== 501;
      if (!retry || attempt >= 4) throw e;
      if (attempt === 1) showToast(tNow('editRetrying'));
      await delay(400 * 2 ** attempt);
    }
  }
}

/** Send ops (in order, one request at a time). Resolves with the server's answer, or null on failure. */
export function sendOps(ops: EditOp[], opts: { quiet?: boolean } = {}): Promise<EditsPosted | null> {
  const key = songKey();
  if (!key || ops.length === 0) return Promise.resolve(null);
  useApp.setState((s) => ({ pendingOps: s.pendingOps + ops.length }));
  const run = chain.then(async () => {
    try {
      const r = await postWithRetry(key, ops);
      applyResult(key, r);
      for (const n of r.notes ?? []) if (!opts.quiet) showToast(tNow('editNote', { note: n.note }));
      void refreshHistory(key);
      return r;
    } catch (e) {
      showToast(`${tNow('editFailed')}: ${e instanceof Error ? e.message : String(e)}`, 'error');
      return null;
    } finally {
      useApp.setState((s) => ({ pendingOps: Math.max(0, s.pendingOps - ops.length) }));
    }
  });
  chain = run;
  return run;
}

/** Another device may have edited: reload when the server's rev moved on. */
export async function pollEdits(): Promise<void> {
  const key = songKey();
  const st = useApp.getState();
  if (!key || st.pendingOps > 0 || api.mock) return;
  try {
    const log = await api.getEdits(key, st.rev);
    if (log.rev === st.rev || songKey() !== key) return;
    const a = await api.getAnalysis(key);
    const d = await api.getDisplay(key, a);
    applyResult(key, { analysis: a, display: d, rev: a.edits?.rev ?? log.rev });
    await refreshHistory(key);
  } catch {
    /* offline: try again later */
  }
}

// ---------------------------------------------------------------- actions

function edit(ops: EditOp[]): Promise<EditsPosted | null> {
  undoStack.length = 0; // a new edit ends the redo chain
  return sendOps(ops);
}

/** Set the selection to a chord: `harte` (concert pitch, from the parser / alternatives) or typed `text` at `shift`. */
export function setChord(sel: Sel, chord: { harte?: string; text?: string; shift?: number }): Promise<EditsPosted | null> {
  const value: Record<string, unknown> = chord.harte ? { harte: chord.harte } : { text: chord.text, shift: chord.shift ?? 0 };
  if (chord.text && chord.harte) value.text = chord.text;
  return edit([makeOp('set_chord', selTarget(sel), value)]);
}

export function insertRest(sel: Sel): Promise<EditsPosted | null> {
  return edit([makeOp('insert_rest', selTarget(sel))]);
}

export function deleteChange(c: ChordOut): Promise<EditsPosted | null> {
  return edit([makeOp('delete_change', chordTarget(c))]);
}

function suggestionOp(kind: 'accept_suggestion' | 'reject_suggestion', c: ChordOut): EditOp {
  return makeOp(kind, { chord_id: c.id, t0: c.start, t1: c.end }, { harte: c.harte, min_detail: c.min_detail ?? 0.05 });
}

export function acceptSuggestions(cs: ChordOut[]): Promise<EditsPosted | null> {
  return edit(cs.map((c) => suggestionOp('accept_suggestion', c)));
}

export function rejectSuggestions(cs: ChordOut[]): Promise<EditsPosted | null> {
  return edit(cs.map((c) => suggestionOp('reject_suggestion', c)));
}

export function halfDoubleTempo(factor: 0.5 | 2): Promise<EditsPosted | null> {
  return edit([makeOp('half_double_tempo', {}, { factor })]);
}

export function setSection(t0: number, label: string | null): Promise<EditsPosted | null> {
  return edit([makeOp('set_section', { t0 }, { label })]);
}

export function setKey(key: string, t0: number | null): Promise<EditsPosted | null> {
  return edit([makeOp('set_key', t0 == null ? {} : { t0 }, { key })]);
}

/** Toggle one op from the history list (revert it, or revert its revert). */
export function toggleOp(op: EditOp): Promise<EditsPosted | null> {
  return edit([makeOp('revert', { op_id: op.id })]);
}

export async function undo(): Promise<void> {
  const { history } = useApp.getState();
  const target = undoCandidate(history, deviceId());
  if (!target) {
    showToast(tNow('nothingToUndo'));
    return;
  }
  const op = makeOp('revert', { op_id: target.id });
  const r = await sendOps([op]);
  if (r) {
    undoStack.push(op.id);
    showToast(tNow('undone', { what: describeOp(target) }));
  }
}

export async function redo(): Promise<void> {
  const { history } = useApp.getState();
  const id = redoCandidate(history, undoStack);
  if (!id) {
    showToast(tNow('nothingToRedo'));
    return;
  }
  const r = await sendOps([makeOp('revert', { op_id: id })]);
  if (r) undoStack.splice(undoStack.lastIndexOf(id), 1);
}

/** A short text for the history list and toasts (op values are concert-pitch Harte: shown as typed when known). */
export function describeOp(op: EditOp): string {
  const v = op.value as Record<string, unknown>;
  const t = op.target as Record<string, unknown>;
  const at = typeof t.t0 === 'number' ? ` @${fmt(t.t0)}` : '';
  switch (op.kind) {
    case 'set_chord':
      return `${tNow('op_set_chord')} ${String(v.text ?? v.harte ?? '')}${at}`;
    case 'insert_rest':
      return `${tNow('op_insert_rest')}${at}`;
    case 'delete_change':
      return `${tNow('op_delete_change')}${at}`;
    case 'accept_suggestion':
      return `${tNow('op_accept')} ${String(v.harte ?? '')}${at}`;
    case 'reject_suggestion':
      return `${tNow('op_reject')} ${String(v.harte ?? '')}${at}`;
    case 'half_double_tempo':
      return v.factor === 2 ? tNow('doubleTempo') : tNow('halfTempo');
    case 'set_section':
      return `${tNow('op_section')} ${String(v.label ?? '—')}${at}`;
    case 'set_key':
      return `${tNow('op_key')} ${String(v.key ?? '')}${at}`;
    case 'revert':
      return t.group ? tNow('op_revert_group') : tNow('op_revert');
    default:
      return op.kind;
  }
}

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = t - 60 * m;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}
