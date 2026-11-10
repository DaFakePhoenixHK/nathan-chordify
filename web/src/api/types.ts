// Hand-generated from shared/analysis.schema.json and shared/display.schema.json (schema v1.0).
// Keep in sync with engine/nathan_chordify/schema.py. Field docs: DESIGN.md §5.13.

// ---------------------------------------------------------------- analysis.json

export interface Alternative {
  harte: string;
  conf: number;
}

export interface AudioInfo {
  play: string;
  stems?: string[];
}

export interface BarOut {
  i: number;
  start: number;
  end: number;
  beats: number;
  section?: string | null;
  free?: boolean;
}

export type BassRole = 'chord_bass' | 'passing_tone';

export interface BassNoteOut {
  start: number;
  end: number;
  note: string;
  midi: number;
  conf: number;
  role: BassRole;
  min_detail?: number | null;
}

export interface BeatOut {
  i: number;
  t: number;
  /** -1 for beats outside the metered bars (free time). */
  bar: number;
  /** 1-based beat within the bar; 0 outside bars. */
  beat: number;
  conf: number;
}

export type Layer = 'main' | 'passing';
export type ChordRole = 'chord' | 'bass_event';

export interface Labels {
  simple: string;
  standard: string;
  full: string;
}

export interface Degree {
  root: number | null;
  bass: number | null;
  nashville: string;
  roman: string;
}

export interface Voicing {
  lh?: number[];
  rh?: number[];
}

export interface ChordOut {
  id: string;
  start: number;
  end: number;
  /** Fractional index into beats[]; null inside free time. */
  start_beat: number | null;
  end_beat: number | null;
  /** -1 inside trailing free time. */
  bar: number;
  /** 1-based, fractional (2.5 = halfway through beat 2). */
  beat_in_bar: number | null;
  dur_beats: number | null;
  harte: string;
  root: string | null;
  quality: string;
  bass: string | null;
  label: Labels;
  degree: Degree;
  layer: Layer;
  role?: ChordRole;
  /** null for main chords; a passing chord is shown when s > 0 and s >= min_detail (§5.11). */
  min_detail?: number | null;
  conf: number;
  salience?: number | null;
  equivalents?: string[];
  alternatives?: Alternative[];
  voicing?: Voicing;
  sources?: Record<string, number>;
  edited?: boolean;
  suggestion?: boolean;
}

export type KeyChangeKind = 'step_up' | 'modulation' | 'relative' | 'parallel';
export type Mode = 'major' | 'minor';

export interface KeyChangeOut {
  t: number;
  bar: number;
  from: string;
  to: string;
  kind: KeyChangeKind;
}

export interface KeyGlobal {
  tonic: string;
  mode: Mode;
  conf: number;
}

export interface KeySegment {
  start: number;
  end: number;
  tonic: string;
  mode: Mode;
}

export interface KeyInfo {
  global: KeyGlobal;
  segments: KeySegment[];
  changes?: KeyChangeOut[];
}

export interface MeterOut {
  from_bar: number;
  num: number;
  den: number;
}

export interface SectionOut {
  label: string;
  group?: string | null;
  start: number;
  end: number;
  source: 'user' | 'auto';
}

export type SourceType = 'youtube' | 'url' | 'upload';

export interface SourceInfo {
  key: string;
  type: SourceType;
  url?: string | null;
  title: string;
  artist?: string | null;
  duration: number;
}

export interface Stats {
  n_main: number;
  n_passing: number;
  runtime_s?: Record<string, number>;
}

export interface SubdivisionOut {
  from_bar: number;
  cells_per_beat: 2 | 3 | 4 | 6;
}

export interface TimeRegion {
  start: number;
  end: number;
}

export interface TempoInfo {
  bpm: number;
  subdivision: SubdivisionOut[];
  free_time?: TimeRegion[];
}

export interface EditsInfo {
  rev?: number;
  n_ops?: number;
}

export interface GtInfo {
  status?: 'none' | 'draft' | 'reviewed';
  tapped_beats?: boolean;
  chart?: string | null;
}

export interface Analysis {
  schema_version?: string;
  engine_version: string;
  params_hash: string;
  calibration?: string | null;
  mode: 'full' | 'fast';
  created_at: string;
  source: SourceInfo;
  audio: AudioInfo;
  display: string;
  key: KeyInfo;
  tempo: TempoInfo;
  meter: MeterOut[];
  beats: BeatOut[];
  bars: BarOut[];
  sections?: SectionOut[];
  chords: ChordOut[];
  bass_line?: BassNoteOut[];
  lyrics?: string | null;
  edits?: EditsInfo;
  gt?: GtInfo;
  stats: Stats;
}

// ---------------------------------------------------------------- display.json

/** [simple, standard, full] */
export type LevelLabels = string[];

export interface DisplayVariant {
  /** One per key segment (analysis.key.segments order). */
  key_names: string[];
  /** chord id -> [simple, standard, full] */
  chords: Record<string, LevelLabels>;
  /** Parallel to analysis.bass_line. */
  bass_line: string[];
}

export interface Display {
  /** "<t>:<auto|sharp|flat>", t = 0..11 */
  variants: Record<string, DisplayVariant>;
  /** "<t>" -> chord id -> voicing */
  voicings: Record<string, Record<string, Voicing>>;
}

// ---------------------------------------------------------------- server API (§11)

export interface TrackSummary {
  key: string;
  title: string;
  artist?: string | null;
  duration?: number | null;
  /** Display name of the global key, e.g. "B" or "G#m", when the server provides it. */
  key_name?: string | null;
  bpm?: number | null;
  created_at?: string | null;
  source_type?: SourceType | null;
  /** Kept on this phone or tablet. */
  saved?: boolean;
}

export interface JobCreated {
  job_id: string;
  key: string;
}

export interface UploadCreated {
  upload_id: string;
  key: string;
  /** Optional: some servers queue the job directly on upload. */
  job_id?: string;
}

export interface ProgressEvent {
  stage: string;
  pct: number;
  msg?: string;
  eta_s?: number | null;
}

export interface DoneEvent {
  analysis_url?: string;
  display_url?: string;
}

export interface ErrorEvent {
  code: string;
  msg: string;
  hint?: string;
}

// ---------------------------------------------------------------- M4: edits, typed entry, ground truth (§5.13, §8.4, §11)

export type EditKind =
  | 'set_chord'
  | 'delete_change'
  | 'insert_rest'
  | 'accept_suggestion'
  | 'reject_suggestion'
  | 'half_double_tempo'
  | 'set_section'
  | 'set_key'
  | 'revert';

/** One line of cache/<key>/edits.jsonl. `id` is made by the client (idempotency key); the server assigns `seq`/`ts`. */
export interface EditOp {
  id: string;
  kind: EditKind;
  target: Record<string, unknown>;
  value: Record<string, unknown>;
  device?: string;
  seq?: number;
  ts?: string;
  /** GET /edits only: false once the op has been reverted. */
  active?: boolean;
}

export interface EditsLog {
  rev: number;
  n_ops: number;
  ops: EditOp[];
}

export interface EditsPosted {
  rev: number;
  n_ops: number;
  appended: { id: string; seq: number }[];
  duplicates: { id: string; seq: number }[];
  notes: { op: string; note: string }[];
  analysis: Analysis;
  display: Display;
}

/** POST /api/theory/parse for one typed chord. */
export interface ParsedChord {
  ok: boolean;
  input?: string;
  error?: string;
  hint?: string;
  harte?: string;
  root?: string | null;
  quality?: string;
  bass?: string | null;
  label?: Labels;
  /** [simple, standard, full] at the display shift. */
  shown?: string[];
  degree?: Degree;
  voicing?: Voicing;
}

export interface PickerVocab {
  key: string;
  shown_key: string;
  roots: { pc: number; name: string }[];
  qualities: { q: string; suffix: string; label: string }[];
}

export interface GtUnsure {
  t0: number;
  t1: number;
  harte: string;
  flag: string;
}

export interface GtChartInfo {
  text: string;
  import_id: string;
  group: string;
  source: 'tapped' | 'detected' | 'lab';
  start: number | null;
  start_beat: number | null;
  n_chords: number;
  unsure: GtUnsure[];
  warnings: string[];
  ts: string;
  format: 'chart' | 'lab';
  /** The beats were re-tapped after this chart was aligned. */
  stale: boolean;
}

export interface GtState {
  status: 'none' | 'draft' | 'reviewed';
  reviewed: boolean;
  reviewed_ts: string | null;
  beats: number[];
  beats_ts: string | null;
  chart: GtChartInfo | null;
  exported: Record<string, string> | null;
  exported_ts: string | null;
  draft_chart: boolean;
}

export interface GtImported {
  gt: GtState;
  analysis: Analysis;
  display: Display;
  rev: number;
  n_chords: number;
  unsure: GtUnsure[];
  warnings: string[];
  appended: number;
}

export interface GtExported {
  files: Record<string, string>;
  songs_yaml: string;
  reviewed: boolean;
  warnings: string[];
  n_chords: number;
  gt: GtState;
}

// ---------------------------------------------------------------- sheets: "find the chord sheet online and refine" (2026-11-03)

export type SheetSource = 'cantonhymn' | 'simplyworship' | 'worshiptogether' | 'ultimateguitar' | 'pasted';
export type SheetVerdict = 'good' | 'maybe' | 'different';
export type SheetApplyMode = 'refine' | 'sheet';

/** One chord of the sheet laid on the recording's timeline; `labels` are spelled by the server for transpose 0. */
export interface SheetAligned {
  t0: number;
  t1: number;
  harte: string;
  /** [simple, standard, full] */
  labels: LevelLabels;
}

export interface SheetMatch {
  title: number;
  align: number;
  score: number;
}

export interface SheetResult {
  id: string;
  source: SheetSource;
  url: string | null;
  title: string;
  artist: string | null;
  year: number | null;
  /** Key the sheet is written in (null when the site doesn't say). */
  key_written: string | null;
  key_recording: string;
  /** Semitones the sheet was shifted to match the recording. */
  transpose: number;
  meter: string | null;
  n_chords: number;
  match: SheetMatch;
  verdict: SheetVerdict;
  /** The sheet page links the same YouTube video. */
  video_match?: boolean;
  /** Other sites with the identical chart. */
  also?: string[];
  /** Bar-text chart: chords + section labels only. */
  chart_text: string;
  aligned: SheetAligned[];
  warnings: string[];
  /** Chords over the words, ChordPro-style ("[C]word"), when the site had them (2026-11-04). */
  lyrics_text?: string | null;
}

export interface SheetApplied {
  id: string;
  /** Edit group of the apply (one undoable step). */
  group: string;
  mode: SheetApplyMode;
  ts: string;
}

export interface SheetsState {
  status: 'idle' | 'searching' | 'done' | 'error';
  error: { code: string; msg: string } | null;
  query: { title: string; artist: string; terms: string[] };
  /** Best first. */
  results: SheetResult[];
  applied: SheetApplied | null;
  ts: string | null;
}

/** A spot where the recording and the sheet disagree and the detected chord was kept. */
export interface SheetFlag {
  t0: number;
  t1: number;
  detected: string;
  sheet: string;
}

export interface SheetApplySummary {
  n_changed: number;
  n_kept: number;
  flagged: SheetFlag[];
}

export interface SheetApplyResult {
  analysis: Analysis;
  display: Display;
  rev: number;
  applied: SheetApplied;
  summary: SheetApplySummary;
}

export interface SheetUndoResult {
  analysis: Analysis;
  display: Display;
  rev: number;
  applied: null;
}

/** Server-side settings (GET/PATCH /api/settings). */
export interface ServerSettings {
  sheets_auto_lookup: boolean;
}

// ---------------------------------------------------------------- practice mixes + the printable sheet (2026-11-04)

/** GET /api/tracks/{key}/mixes */
export type MixStatus = 'ready' | 'makeable' | 'no_stems' | 'missing';

/** GET /api/tracks/{key}/export/sheet?format=json */
export interface SheetExport {
  title: string;
  artist: string | null;
  /** Key name after the transposition. */
  key: string;
  key_original: string;
  transpose: number;
  bpm: number;
  meter: string;
  level: string;
  slash: boolean;
  /** The bar chart: "[Verse]" lines and "| B . . . | E . . . |" lines. */
  chart_text: string;
  key_changes: string[];
  intro: string[];
  /** "[C]word" text with OUR chords placed over the sheet's words (or the sheet's own chords, see lyrics_ours). */
  lyrics_chordpro: string | null;
  /** The same in two lines (chords above the words), for plain text. */
  lyrics_text: string | null;
  lyrics_ours: boolean;
  lyrics_source: string | null;
  lyrics_sheet_id: string | null;
  notes: string[];
  made: string;
  /** The whole .txt file. */
  text: string;
  /** The whole .cho file, or null without lyrics. */
  chordpro: string | null;
}
