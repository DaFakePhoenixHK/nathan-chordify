// Pure helpers for the "find the sheet online" feature (no DOM, no store): unit-tested in logic.test.ts.
// The web never spells chords: the sheet's labels come spelled from the server, ours from display.json.
import type { SheetAligned, SheetFlag, SheetResult, SheetsState, SheetVerdict } from '../api/types';
import type { StringKey } from '../i18n/strings';
import { stripLetterBass } from '../grid/labels';

const EPS = 1e-3;

// ---------------------------------------------------------------- verdicts, sources

export function verdictKey(v: SheetVerdict): StringKey {
  return `verdict_${v}`;
}

/** CSS class / colour family of a verdict: good = green, maybe = amber, different = grey. */
export function verdictClass(v: SheetVerdict): 'good' | 'maybe' | 'different' {
  return v;
}

const SOURCES = new Set(['cantonhymn', 'simplyworship', 'worshiptogether', 'ultimateguitar', 'pasted']);

/** String key of a source's display name, or null for a site we don't know (show the raw name). */
export function sourceKey(src: string): StringKey | null {
  return SOURCES.has(src) ? (`source_${src}` as StringKey) : null;
}

/** "Written in C → shifted to B" data, or null when the sheet wasn't shifted (or its key is unknown). */
export function keyShift(r: Pick<SheetResult, 'transpose' | 'key_written' | 'key_recording'>): { from: string; to: string } | null {
  if (r.transpose === 0 || !r.key_written) return null;
  return { from: r.key_written, to: r.key_recording };
}

export function termsText(st: SheetsState | null): string {
  return st?.query.terms.join(' ') ?? '';
}

// ---------------------------------------------------------------- banner

/** The result the banner offers: the best one, when the search is done, it looks good and nothing is applied yet. */
export function bannerOffer(st: SheetsState | null, dismissed: boolean): SheetResult | null {
  if (!st || dismissed || st.status !== 'done' || st.applied) return null;
  const best = st.results[0];
  return best && best.verdict === 'good' ? best : null;
}

/** The subset of Storage we use (so tests can pass a Map-backed fake). */
export interface KV {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export const DISMISS_KEY = 'nc.sheets.notnow';
const DISMISS_MAX = 200;

function readDismissed(kv: KV): string[] {
  try {
    const v = JSON.parse(kv.getItem(DISMISS_KEY) ?? '[]') as unknown;
    return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
  } catch {
    return [];
  }
}

export function isBannerDismissed(kv: KV, songKey: string): boolean {
  return readDismissed(kv).includes(songKey);
}

/** "Not now" is remembered per song (the newest 200 songs). */
export function dismissBanner(kv: KV, songKey: string): void {
  const list = readDismissed(kv).filter((k) => k !== songKey);
  list.push(songKey);
  try {
    kv.setItem(DISMISS_KEY, JSON.stringify(list.slice(-DISMISS_MAX)));
  } catch {
    /* private mode: the dismissal lasts for this session only */
  }
}

// ---------------------------------------------------------------- "Sheet says" marks on the beat grid

export function normLabel(s: string): string {
  const x = s.trim();
  return x === 'N' || x === 'NC' || x === 'N.C.' || x === '' ? 'N.C.' : x;
}

/** The sheet's label sounding at t: labels[level], without the "/bass" when slash chords are off. */
export function sheetLabelAt(aligned: SheetAligned[], t: number, level: number, slash: boolean): string | null {
  let lo = 0;
  let hi = aligned.length - 1;
  let k = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (aligned[mid].t0 <= t + EPS) {
      k = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  if (k < 0 || t >= aligned[k].t1 - EPS) return null;
  const l = aligned[k].labels[Math.min(level, aligned[k].labels.length - 1)] ?? null;
  return l == null ? null : slash ? l : stripLetterBass(l);
}

export interface SheetMark {
  /** The sheet's chord over this beat square, or null where the sheet has nothing. */
  label: string | null;
  /** Differs from our chord at the grid's label level (only when `compare` is on). */
  differs: boolean;
  /** Draw the label (a new chord, or the first beat of a bar); otherwise a dot. */
  show: boolean;
}

export interface SquareLike {
  t0: number;
  t1: number;
  /** 1-based beat in the bar (1 = bar start). */
  beat?: number;
}

/**
 * One mark per beat square. `ours(t)` returns our label at the same level/slash setting (null = nothing sounding).
 * `compare` is false when the user has transposed: the sheet's labels are spelled for the original key, so a
 * comparison would only show the transposition.
 */
export function sheetMarks(
  aligned: SheetAligned[],
  squares: SquareLike[],
  ours: (t: number) => string | null,
  level: number,
  slash: boolean,
  compare: boolean,
): SheetMark[] {
  const out: SheetMark[] = [];
  let prev: string | null = null;
  for (let i = 0; i < squares.length; i++) {
    const s = squares[i];
    const mid = (s.t0 + s.t1) / 2;
    const label = sheetLabelAt(aligned, mid, level, slash);
    let differs = false;
    if (compare && label != null) {
      const mine = ours(mid);
      differs = mine != null && normLabel(mine) !== normLabel(label);
    }
    const show = label != null && (i === 0 || label !== prev || s.beat === 1);
    out.push({ label, differs, show });
    prev = label;
  }
  return out;
}

/** Number of separate disagreeing stretches (adjacent differing squares with the same sheet label count once). */
export function countDiffRuns(marks: SheetMark[]): number {
  let n = 0;
  for (let i = 0; i < marks.length; i++) {
    const m = marks[i];
    if (!m.differs) continue;
    const p = marks[i - 1];
    if (!p || !p.differs || p.label !== m.label) n++;
  }
  return n;
}

/** The flagged spot (kept as detected after an apply) overlapping each square, or null. */
export function flagsForSquares(flags: SheetFlag[], squares: SquareLike[]): (SheetFlag | null)[] {
  return squares.map((s) => flags.find((f) => f.t0 < s.t1 - EPS && f.t1 > s.t0 + EPS) ?? null);
}

// ---------------------------------------------------------------- polling

export interface PollOpts {
  intervalMs?: number;
  /** Give up (returning the last state) after this long. */
  maxMs?: number;
  sleep?: (ms: number) => Promise<void>;
  /** Stop early (returns null): the song was closed, or a newer search took over. */
  cancelled?: () => boolean;
}

const realSleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** GET the sheet state every 1.5 s while it says "searching"; resolves with the first non-searching state. */
export async function pollSheets(get: () => Promise<SheetsState>, opts: PollOpts = {}): Promise<SheetsState | null> {
  const interval = opts.intervalMs ?? 1500;
  const maxMs = opts.maxMs ?? 120_000;
  const sleep = opts.sleep ?? realSleep;
  const cancelled = opts.cancelled ?? (() => false);
  let waited = 0;
  for (;;) {
    if (cancelled()) return null;
    const st = await get();
    if (st.status !== 'searching' || waited >= maxMs) return st;
    await sleep(interval);
    waited += interval;
  }
}

/** Is the pasted text a single link rather than a chart? */
export function looksLikeUrl(text: string): boolean {
  const t = text.trim();
  return /^https?:\/\/\S+$/i.test(t);
}
