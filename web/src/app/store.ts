import { create } from 'zustand';
import { get as idbGet, set as idbSet } from 'idb-keyval';
import type { Analysis, Display, EditOp, GtState, Layer, ServerSettings, SheetApplyMode, SheetFlag, SheetSource, SheetsState } from '../api/types';

export type Level = 'simple' | 'standard' | 'full';
export type Notation = 'letters' | 'nashville' | 'roman';
export type Acc = 'auto' | 'sharp' | 'flat';
export type Lang = 'en' | 'zh-TW';
export type Theme = 'auto' | 'light' | 'dark';

export type Route = { page: 'library' } | { page: 'connect' } | { page: 'phone' } | { page: 'song'; key: string };

/** Detail stops (§6.2): Simple 0 / Standard 0.5 / Detailed 0.8 / Everything 1.0. */
export const DETAIL_STOPS = [0, 0.5, 0.8, 1] as const;
export const PRESETS = [
  { id: 'simple', level: 'simple', detail: 0 },
  { id: 'standard', level: 'standard', detail: 0.5 },
  { id: 'detailed', level: 'full', detail: 0.8 },
  { id: 'everything', level: 'full', detail: 1 },
] as const;
export type PresetId = (typeof PRESETS)[number]['id'];

export const RATE_MIN = 0.25;
export const RATE_MAX = 2;
export const TRANSPOSE_MIN = -12;
export const TRANSPOSE_MAX = 12;

export interface Prefs {
  detail: number;
  level: Level;
  slash: boolean;
  notation: Notation;
  acc: Acc;
  noteNames: boolean;
  lang: Lang;
  theme: Theme;
  volume: number;
  /** Volume of the piano-like chord voice ("Chords only" / "Piano + chords" sounds). */
  chordsVolume: number;
}

/** "Sound" selector (2026-11-04): what plays. */
export type MixId = 'original' | 'novocals' | 'harmony' | 'chords' | 'piano_chords';

/**
 * "Audio" choice (2026-11-04, user's request): does the recording follow Transpose?
 * `transposed` = the server moves the recording's pitch to the transposed key (same speed);
 * `original` = the recording plays in its own key and only the chord names move.
 */
export type AudioKey = 'original' | 'transposed';

export interface MixUi {
  mix: MixId;
  /** A practice mix is being made on the server (first use takes a few seconds). */
  mixBusy: boolean;
  /** false once the server said the song has no separated stems (options greyed with a hint). */
  mixAvail: boolean | null;
  /** The player URL without a ?mix= / &transpose= part (the recording as it is). */
  audioBase: string | null;
  /** Whether the recording follows Transpose (remembered per song). */
  audioKey: AudioKey;
  /** The shifted audio is being made on the server (the first time takes several seconds). */
  shiftBusy: boolean;
  /** "Export" menu + the print view. */
  exportOpen: boolean;
  printOpen: boolean;
  /** The print view calls window.print() once its content has loaded. */
  printAuto: boolean;
}

/** Per-song settings, remembered in IndexedDB (idb-keyval) per track key. */
export interface SongPrefs {
  transpose: number;
  /** Play-in offset p in semitones (-6..+5): labels shown at transposition t + p; keyboard hint −p. */
  playIn: number;
  rate: number;
}

export interface JobState {
  jobId: string | null;
  key: string | null;
  label: string;
  stage: string;
  pct: number;
  msg: string;
  etaS: number | null;
  status: 'uploading' | 'running' | 'done' | 'error';
  /** Stage ids in the order they were first seen. */
  seen: string[];
  error: { code: string; msg: string; hint?: string } | null;
  connectionLost: boolean;
}

export interface Loop {
  a: number | null;
  b: number | null;
}

export interface MenuState {
  seg: number;
  x: number;
  y: number;
}

export interface Toast {
  id: number;
  msg: string;
  kind: 'info' | 'error';
}

/** Editor selection (§8.4): one chord (edited in place) or a time range on the beat grid. */
export type Sel =
  | { kind: 'chord'; id: string; t0: number; t1: number; layer: Layer }
  | { kind: 'range'; t0: number; t1: number };

export interface EditState {
  /** Edit mode (E key / ✎ button). */
  editMode: boolean;
  /** Ground-truth mode (§8.4): implies edit mode; the grid shows the main layer + Nathan's chords only. */
  gtMode: boolean;
  sel: Sel | null;
  /** Text typed into the chord entry (shared by the keyboard shortcuts and the picker). */
  entry: string;
  /** Bumped to focus the chord entry field. */
  entryFocus: number;
  /** The op log (GET /edits), oldest first. */
  history: EditOp[];
  /** Server rev of the analysis on screen. */
  rev: number;
  /** Ops sent but not yet confirmed by the server. */
  pendingOps: number;
  historyOpen: boolean;
  gt: GtState | null;
  tapMode: boolean;
  /** Beats being tapped (saved GT beats + this session's taps), or null outside tap mode. */
  tapDraft: number[] | null;
  tapDirty: boolean;
  chartOpen: boolean;
}

/** "Find the sheet online" (2026-11-03): per-song state of the sheet lookup. */
export interface SheetsUi {
  /** The sheet panel (drawer below the toolbar; a bottom sheet on phones). */
  sheetsOpen: boolean;
  sheets: SheetsState | null;
  /** Result whose chords show as the "Sheet says" row under the grid (before applying). */
  sheetSel: string | null;
  /** Spots kept as detected after an apply: marked in the grid until Undo. */
  sheetFlags: SheetFlag[];
  sheetSummary: { n_changed: number; n_kept: number; source: SheetSource; mode: SheetApplyMode } | null;
  /** "Not now" on the banner for this song (also remembered in localStorage). */
  sheetBannerHidden: boolean;
}

export interface AppState extends Prefs, SongPrefs, EditState, SheetsUi, MixUi {
  /** Server-side settings (GET /api/settings), loaded on demand. */
  serverSettings: ServerSettings | null;
  route: Route;
  analysis: Analysis | null;
  display: Display | null;
  loading: boolean;
  loadError: string | null;
  audioSrc: string | null;
  playing: boolean;
  duration: number;
  loop: Loop;
  /** Index of the current segment in the visible timeline. Written by the sync loop only when it changes. */
  nowSeg: number;
  job: JobState | null;
  menu: MenuState | null;
  sheetOpen: boolean;
  mobileTab: 'grid' | 'diagrams';
  toast: Toast | null;
  /** Bumped to force the open song to reload (e.g. after a re-analysis of the same key). */
  songRev: number;
}

const PREFS_KEY = 'nc.prefs.v1';

export const DEFAULT_PREFS: Prefs = {
  detail: 0.8, // Detailed preset: the default for Nathan (§6.2)
  level: 'full',
  slash: true,
  notation: 'letters',
  acc: 'auto',
  noteNames: true,
  lang: 'en',
  theme: 'auto',
  volume: 1,
  chordsVolume: 0.7,
};

export const DEFAULT_SONG: SongPrefs = { transpose: 0, playIn: 0, rate: 1 };

export const DEFAULT_EDIT: EditState = {
  editMode: false,
  gtMode: false,
  sel: null,
  entry: '',
  entryFocus: 0,
  history: [],
  rev: 0,
  pendingOps: 0,
  historyOpen: false,
  gt: null,
  tapMode: false,
  tapDraft: null,
  tapDirty: false,
  chartOpen: false,
};

export const DEFAULT_SHEETS: SheetsUi = {
  sheetsOpen: false,
  sheets: null,
  sheetSel: null,
  sheetFlags: [],
  sheetSummary: null,
  sheetBannerHidden: false,
};

export const DEFAULT_MIX: MixUi = {
  mix: 'original',
  mixBusy: false,
  mixAvail: null,
  audioBase: null,
  audioKey: 'transposed',
  shiftBusy: false,
  exportOpen: false,
  printOpen: false,
  printAuto: false,
};

function loadPrefs(): Prefs {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (!raw) return DEFAULT_PREFS;
    return { ...DEFAULT_PREFS, ...(JSON.parse(raw) as Partial<Prefs>) };
  } catch {
    return DEFAULT_PREFS;
  }
}

function parseRoute(): Route {
  const h = location.hash.replace(/^#\/?/, '');
  const m = /^song\/(.+)$/.exec(h);
  if (m) return { page: 'song', key: decodeURIComponent(m[1]) };
  if (h === 'connect' || h === 'phone') return { page: h };
  return { page: 'library' };
}

export const useApp = create<AppState>()(() => ({
  ...loadPrefs(),
  ...DEFAULT_SONG,
  route: parseRoute(),
  analysis: null,
  display: null,
  loading: false,
  loadError: null,
  audioSrc: null,
  playing: false,
  duration: 0,
  loop: { a: null, b: null },
  nowSeg: -1,
  job: null,
  menu: null,
  sheetOpen: false,
  mobileTab: 'grid',
  toast: null,
  songRev: 0,
  ...DEFAULT_EDIT,
  ...DEFAULT_SHEETS,
  ...DEFAULT_MIX,
  serverSettings: null,
}));

// Persist prefs whenever one of them changes.
const PREF_KEYS = Object.keys(DEFAULT_PREFS) as (keyof Prefs)[];
useApp.subscribe((s, prev) => {
  if (!PREF_KEYS.some((k) => s[k] !== prev[k])) return;
  const out = {} as Record<string, unknown>;
  for (const k of PREF_KEYS) out[k] = s[k];
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(out));
  } catch {
    /* private mode: prefs just don't persist */
  }
});

// Persist per-song prefs.
useApp.subscribe((s, prev) => {
  if (s.route.page !== 'song') return;
  if (s.transpose === prev.transpose && s.playIn === prev.playIn && s.rate === prev.rate) return;
  const v: SongPrefs = { transpose: s.transpose, playIn: s.playIn, rate: s.rate };
  idbSet(`song:${s.route.key}`, v).catch(() => undefined);
});

export async function loadSongPrefs(key: string): Promise<SongPrefs> {
  try {
    const v = (await idbGet(`song:${key}`)) as Partial<SongPrefs> | undefined;
    return { ...DEFAULT_SONG, ...v };
  } catch {
    return DEFAULT_SONG;
  }
}

window.addEventListener('hashchange', () => {
  useApp.setState({ route: parseRoute(), menu: null, sheetOpen: false, ...DEFAULT_EDIT, ...DEFAULT_SHEETS, ...DEFAULT_MIX });
});

export function navigate(route: Route): void {
  const hash = route.page === 'song' ? `#/song/${encodeURIComponent(route.key)}` : route.page === 'library' ? '#/' : `#/${route.page}`;
  if (location.hash !== hash) location.hash = hash;
  else useApp.setState({ route });
}

// ---------------------------------------------------------------- small actions

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function setTranspose(t: number): void {
  useApp.setState({ transpose: clamp(Math.round(t), TRANSPOSE_MIN, TRANSPOSE_MAX) });
}

export function nudgeTranspose(d: number): void {
  setTranspose(useApp.getState().transpose + d);
}

export function setRate(r: number): void {
  // 1% steps (§7.2)
  const v = clamp(Math.round(r * 100) / 100, RATE_MIN, RATE_MAX);
  useApp.setState({ rate: v });
}

export function nudgeRate(d: number): void {
  setRate(useApp.getState().rate + d);
}

export function setDetail(d: number): void {
  useApp.setState({ detail: clamp(d, 0, 1) });
}

export function cycleDetail(): void {
  const d = useApp.getState().detail;
  const i = DETAIL_STOPS.findIndex((s) => s > d + 1e-6);
  setDetail(i < 0 ? DETAIL_STOPS[0] : DETAIL_STOPS[i]);
}

export function applyPreset(id: PresetId): void {
  const p = PRESETS.find((x) => x.id === id);
  if (p) useApp.setState({ level: p.level, detail: p.detail });
}

/** The preset matching the current level + detail, if any. */
export function currentPreset(level: Level, detail: number): PresetId | null {
  const p = PRESETS.find((x) => x.level === level && Math.abs(x.detail - detail) < 1e-6);
  return p ? p.id : null;
}

let toastSeq = 0;
export function showToast(msg: string, kind: Toast['kind'] = 'info'): void {
  const id = ++toastSeq;
  useApp.setState({ toast: { id, msg, kind } });
  window.setTimeout(() => {
    if (useApp.getState().toast?.id === id) useApp.setState({ toast: null });
  }, 4500);
}

export function setLoop(a: number | null, b: number | null): void {
  if (a != null && b != null && b < a) [a, b] = [b, a];
  useApp.setState({ loop: { a, b } });
}
