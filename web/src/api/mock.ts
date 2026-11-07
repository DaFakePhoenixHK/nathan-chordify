// `?mock=1`: the UI runs without the server. Data come from shared/fixtures (served by the dev-only
// middleware in vite.config.ts under /__mock/); the audio is the cached test song when present, or a file
// the user picks. Jobs are simulated.
import { ApiError, type Api, type JobHandlers } from './client';
import type {
  Analysis,
  ChordOut,
  Display,
  ServerSettings,
  SheetAligned,
  SheetApplyMode,
  SheetExport,
  SheetFlag,
  SheetResult,
  SheetsState,
  TrackSummary,
} from './types';
import type { SheetExportParams } from '../export/logic';

export const MOCK_KEY = 'yt-Y7boUFbM76k';
const RENAME_KEY = 'nc.mock.renames';

let fixtures: Promise<[Analysis, Display]> | null = null;
let pickedAudio: { url: string; name: string } | null = null;
let jobSeq = 0;

function loadFixtures(): Promise<[Analysis, Display]> {
  fixtures ??= Promise.all([
    fetch('/__mock/analysis.json').then((r) => {
      if (!r.ok) throw new Error('Mock fixtures need the Vite dev server (npm run dev).');
      return r.json() as Promise<Analysis>;
    }),
    fetch('/__mock/display.json').then((r) => {
      if (!r.ok) throw new Error('Mock fixtures need the Vite dev server (npm run dev).');
      return r.json() as Promise<Display>;
    }),
  ]).catch((e: unknown) => {
    fixtures = null;
    throw e;
  });
  return fixtures;
}

function renames(): Record<string, { title?: string; artist?: string | null }> {
  try {
    return JSON.parse(localStorage.getItem(RENAME_KEY) ?? '{}') as Record<string, { title?: string; artist?: string | null }>;
  } catch {
    return {};
  }
}

/** Use a local file as the mock song's audio. */
export function setMockAudioFile(file: File): string {
  if (pickedAudio) URL.revokeObjectURL(pickedAudio.url);
  pickedAudio = { url: URL.createObjectURL(file), name: file.name };
  return pickedAudio.url;
}

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- sheets: fake search results for 回家

const SETTINGS_KEY = 'nc.mock.settings';
let sheetsState: SheetsState | null = null;
let sheetsTimer = 0;
let mockRev = 0;
let pasteSeq = 0;

const nowIso = () => new Date().toISOString();

function mainChords(a: Analysis): ChordOut[] {
  return a.chords.filter((c) => c.layer === 'main' && c.harte !== 'N').toSorted((p, q) => p.start - q.start);
}

/** The fixture's own chords laid on its timeline (spelled like the grid: display variant "0:auto"), with a few
 *  of them changed so the grid marks show. */
function alignedFrom(a: Analysis, d: Display, tweak: (c: ChordOut, i: number) => { harte: string; labels: string[] } | null): SheetAligned[] {
  const spelled = d.variants['0:auto']?.chords ?? {};
  return mainChords(a).map((c, i) => {
    const tw = tweak(c, i);
    return { t0: c.start, t1: c.end, harte: tw?.harte ?? c.harte, labels: tw?.labels ?? spelled[c.id] ?? [c.label.simple, c.label.standard, c.label.full] };
  });
}

const MOCK_SECTIONS: Record<number, string> = { 0: 'Intro', 4: 'Verse 1', 20: 'Chorus', 28: 'Verse 2', 44: 'Chorus', 56: 'Tag' };

/** Bar text (| chords per beat |) from an aligned sheet, four bars a line, with section labels. */
function chartTextFrom(a: Analysis, aligned: SheetAligned[]): string {
  const lines: string[] = [];
  const byBar = new Map<number, number[]>();
  for (const b of a.beats) if (b.bar >= 0) byBar.set(b.bar, [...(byBar.get(b.bar) ?? []), b.t]);
  const labelAt = (t: number) => aligned.find((s) => s.t0 <= t + 1e-3 && t < s.t1)?.labels[2] ?? null;
  let row: string[] = [];
  let prev: string | null = null;
  const flush = () => {
    if (row.length) lines.push(`| ${row.join(' | ')} |`);
    row = [];
  };
  for (const bar of a.bars) {
    const beats = byBar.get(bar.i);
    if (!beats?.length) continue;
    if (MOCK_SECTIONS[bar.i]) {
      flush();
      lines.push(`[${MOCK_SECTIONS[bar.i]}]`);
    }
    row.push(
      beats
        .map((t) => {
          const l = labelAt(t + 0.05);
          if (!l) return 'N';
          const out = l === prev ? '.' : l;
          prev = l;
          return out;
        })
        .join(' '),
    );
    if (row.length === 4) flush();
  }
  flush();
  return lines.join('\n');
}

const SIMPLE = (c: ChordOut, d: Display) => {
  const l = d.variants['0:auto']?.chords[c.id]?.[0] ?? c.label.simple;
  return { harte: c.harte, labels: [l, l, l] };
};

function mockResults(a: Analysis, d: Display): SheetResult[] {
  // 1) CantonHymn, a good match: 3 disagreements with our chords (2 quality-only, 1 root).
  const changes: Record<string, { harte: string; labels: string[] }> = {
    c0002: { harte: 'D#:min', labels: ['D#m', 'D#m', 'D#m'] },
    c0009: { harte: 'E:maj7', labels: ['E', 'Emaj7', 'Emaj7'] },
    c0012: { harte: 'C#:min7', labels: ['C#m', 'C#m7', 'C#m7'] },
  };
  const good = alignedFrom(a, d, (c) => changes[c.id] ?? null);
  // 2) Simply Worship, another version: written in C, plainer chords every 4th change.
  const maybe = alignedFrom(a, d, (c, i) => (i % 4 === 1 ? SIMPLE(c, d) : null));
  // 3) A different song with the same title.
  const cycle = [
    { harte: 'B:maj', labels: ['B', 'B', 'B'] },
    { harte: 'F#:maj', labels: ['F#', 'F#', 'F#'] },
    { harte: 'G#:min', labels: ['G#m', 'G#m', 'G#m'] },
    { harte: 'E:maj', labels: ['E', 'E', 'E'] },
  ];
  const different = alignedFrom(a, d, (_c, i) => cycle[(i >> 1) % 4]).filter((_s, i) => i % 3 !== 2);
  return [
    {
      id: 'ch-1',
      source: 'cantonhymn',
      url: 'https://cantonhymn.net/song/回家-eternity-girls',
      title: '回家',
      artist: 'Eternity Girls 永恆音樂事工',
      year: 2007,
      key_written: 'B',
      key_recording: 'B',
      transpose: 0,
      meter: '4/4',
      n_chords: good.length,
      match: { title: 1, align: 0.91, score: 0.93 },
      verdict: 'good',
      video_match: true,
      chart_text: chartTextFrom(a, good),
      aligned: good,
      warnings: [],
      lyrics_text: mockLyrics(good),
    },
    {
      id: 'sw-1',
      source: 'simplyworship',
      url: 'https://simplyworship.org/?s=回家',
      title: '回家 (Going Home)',
      artist: 'Eternity Girls',
      year: 2009,
      key_written: 'C',
      key_recording: 'B',
      transpose: -1,
      meter: '4/4',
      n_chords: maybe.length,
      match: { title: 0.9, align: 0.62, score: 0.66 },
      verdict: 'maybe',
      also: ['hkchords'],
      chart_text: chartTextFrom(a, maybe),
      aligned: maybe,
      warnings: ['The chorus has two extra bars in this version.'],
    },
    {
      id: 'ch-2',
      source: 'cantonhymn',
      url: 'https://cantonhymn.net/song/回家-stream-of-praise',
      title: '回家',
      artist: '讚美之泉 Stream of Praise',
      year: 2003,
      key_written: 'G',
      key_recording: 'B',
      transpose: 4,
      meter: '4/4',
      n_chords: 28,
      match: { title: 1, align: 0.18, score: 0.24 },
      verdict: 'different',
      chart_text: chartTextFrom(a, different),
      aligned: different,
      warnings: ['Only 18% of the bars line up with the recording.'],
    },
  ];
}

/** Demo words: "啦" syllables under the sheet's chords, two chords a line, a label every eight. */
function mockLyrics(aligned: SheetAligned[]): string {
  const out: string[] = [];
  aligned.forEach((s, i) => {
    if (i % 16 === 0) out.push('', `[${i < 32 ? `Verse ${i / 16 + 1}` : 'Chorus'}]`);
    const tok = `[${s.labels[2]}]啦啦啦`;
    if (i % 2 === 0) out.push(tok);
    else out[out.length - 1] += ` ${tok}`;
  });
  return out.join('\n').trim() + '\n';
}

/** The demo's chord sheet: the bar chart from the fixture labels at the display shift (the labels are the
 *  server's; the browser only picks them), plus the sheet's own words when a sheet has them. */
function mockExport(a: Analysis, d: Display, p: SheetExportParams, st: SheetsState | null): SheetExport {
  const shift = ((p.transpose % 12) + 12) % 12;
  const v = d.variants[`${shift}:${p.acc}`] ?? d.variants[`${shift}:auto`];
  const li = { simple: 0, standard: 1, full: 2 }[p.level];
  const strip = (l: string) => (p.slash ? l : l.replace(/\/[A-G](?:#|b|♯|♭)*$/, ''));
  const label = (c: ChordOut) => (c.harte === 'N' ? 'N' : strip(v?.chords[c.id]?.[li] ?? c.label[p.level]));
  const mains = a.chords.filter((c) => c.layer === 'main').toSorted((x, y) => x.start - y.start);
  const byBar = new Map<number, number[]>();
  for (const b of a.beats) if (b.bar >= 0) byBar.set(b.bar, [...(byBar.get(b.bar) ?? []), b.t]);
  const lines: string[] = [];
  let row: string[] = [];
  let last: string | null = null;
  const flush = () => {
    if (row.length) lines.push(`| ${row.join(' | ')} |`);
    row = [];
  };
  for (const bar of a.bars) {
    if (MOCK_SECTIONS[bar.i]) {
      flush();
      lines.push(`[${MOCK_SECTIONS[bar.i]}]`);
    }
    const toks = (byBar.get(bar.i) ?? [bar.start]).map((t) => {
      const c = mains.find((x) => x.start - 1e-3 <= t + 0.02 && t + 0.02 < x.end - 1e-3);
      const l = c ? label(c) : null;
      if (!l || l === last) return '.';
      last = l;
      return l;
    });
    row.push(toks.join(' '));
    if (row.length === 4) flush();
  }
  flush();
  const gi = Math.max(0, a.key.segments.findIndex((s) => s.tonic === a.key.global.tonic && s.mode === a.key.global.mode));
  const key = v?.key_names[gi] ?? a.key.global.tonic;
  const key0 = d.variants['0:auto']?.key_names[gi] ?? a.key.global.tonic;
  const sheet = st?.results.find((r) => r.id === (p.sheet ?? '') && r.lyrics_text) ?? st?.results.find((r) => r.lyrics_text) ?? null;
  const chart_text = lines.join('\n') + '\n';
  const header = [
    `# ${a.source.title}${a.source.artist ? ` — ${a.source.artist}` : ''}`,
    `# Key: ${key}`,
    `# Tempo: ${Math.round(a.tempo.bpm)} BPM · Meter: 4/4 · Chords: ${p.level}`,
    '# Demo: the real server places our chords over the words.',
  ];
  const lyrics = p.lyrics && sheet?.lyrics_text ? sheet.lyrics_text : null;
  const notes = lyrics ? ["The chords over the words are the sheet's own: the demo has no server to line them up."] : [];
  return {
    title: a.source.title,
    artist: a.source.artist ?? null,
    key,
    key_original: key0,
    transpose: p.transpose,
    bpm: Math.round(a.tempo.bpm),
    meter: '4/4',
    level: p.level,
    slash: p.slash,
    chart_text,
    key_changes: [],
    intro: [],
    lyrics_chordpro: lyrics,
    lyrics_text: lyrics,
    lyrics_ours: false,
    lyrics_source: sheet?.source ?? null,
    lyrics_sheet_id: sheet?.id ?? null,
    notes,
    made: new Date().toISOString().slice(0, 10),
    text: `${header.join('\n')}\n${chart_text}${lyrics ? `\n# The sheet's own chords over the words\n${lyrics}` : ''}`,
    chordpro: lyrics ? `{title: ${a.source.title}}\n{key: ${key}}\n\n${lyrics}\n{start_of_tab}\n${chart_text}{end_of_tab}\n` : null,
  };
}

function sheetsIdle(a: Analysis): SheetsState {
  return {
    status: 'idle',
    error: null,
    query: { title: a.source.title.replace(/\s*\[.*?\]\s*/g, '').replace(/^詩歌-/, ''), artist: a.source.artist ?? '', terms: ['回家', 'Eternity Girls'] },
    results: [],
    applied: null,
    ts: null,
  };
}

async function sheetsNow(): Promise<SheetsState> {
  const [a] = await loadFixtures();
  sheetsState ??= sheetsIdle(a);
  return sheetsState;
}

/** The first two spots where the sheet's full label differs from ours (for the apply summary). */
function flaggedFor(a: Analysis, d: Display, r: SheetResult): SheetFlag[] {
  const out: SheetFlag[] = [];
  const spelled = d.variants['0:auto']?.chords ?? {};
  for (const s of r.aligned) {
    const c = a.chords.find((x) => x.layer === 'main' && Math.abs(x.start - s.t0) < 1e-3);
    const mine = c ? (spelled[c.id]?.[2] ?? c.label.full) : null;
    if (c && mine && mine !== s.labels[2]) out.push({ t0: s.t0, t1: s.t1, detected: mine, sheet: s.labels[2] });
    if (out.length === 2) break;
  }
  return out;
}

function readSettings(): ServerSettings {
  try {
    return { sheets_auto_lookup: true, ...(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}') as Partial<ServerSettings>) };
  } catch {
    return { sheets_auto_lookup: true };
  }
}

/** Editing and ground truth need the real server (they write the song's edit log and eval/gt). */
const needsServer = (): Promise<never> =>
  Promise.reject(new ApiError(501, 'mock', 'Editing needs the Nathan Chordify server (demo mode is read-only).'));

export const mockApi: Api = {
  mock: true,
  async listTracks(): Promise<TrackSummary[]> {
    const [a, d] = await loadFixtures();
    const gi = Math.max(
      0,
      a.key.segments.findIndex((s) => s.tonic === a.key.global.tonic && s.mode === a.key.global.mode),
    );
    const r = renames()[MOCK_KEY] ?? {};
    return [
      {
        key: MOCK_KEY,
        title: r.title ?? a.source.title,
        artist: r.artist !== undefined ? r.artist : (a.source.artist ?? null),
        duration: a.source.duration,
        key_name: d.variants['0:auto']?.key_names[gi] ?? a.key.global.tonic,
        bpm: a.tempo.bpm,
        created_at: a.created_at,
        source_type: a.source.type,
      },
    ];
  },
  async getAnalysis() {
    const [a] = await loadFixtures();
    const r = renames()[MOCK_KEY] ?? {};
    return {
      ...a,
      source: {
        ...a.source,
        title: r.title ?? a.source.title,
        artist: r.artist !== undefined ? r.artist : a.source.artist,
      },
    };
  },
  async getDisplay() {
    const [, d] = await loadFixtures();
    return d;
  },
  async audioUrl() {
    if (pickedAudio) return pickedAudio.url;
    try {
      const r = await fetch('/__mock/audio', { method: 'HEAD' });
      if (r.ok) return '/__mock/audio';
    } catch {
      /* no dev server */
    }
    return null;
  },
  async createJob() {
    jobSeq += 1;
    return { job_id: `mock-${jobSeq}`, key: MOCK_KEY };
  },
  async uploadAndQueue(file, onProgress) {
    for (let p = 0; p <= 100; p += 25) {
      onProgress(p);
      await delay(120);
    }
    setMockAudioFile(file);
    jobSeq += 1;
    return { job_id: `mock-${jobSeq}`, key: MOCK_KEY };
  },
  watchJob(_jobId: string, h: JobHandlers) {
    const stages: [string, number][] = [
      ['ingest', 8],
      ['separation', 30],
      ['rhythm', 6],
      ['evidence', 14],
      ['decode', 8],
      ['output', 4],
    ];
    const total = stages.reduce((s, x) => s + x[1], 0);
    let cancelled = false;
    void (async () => {
      h.onConnection(true);
      let elapsed = 0;
      for (const [stage, w] of stages) {
        for (let i = 0; i <= 4; i++) {
          if (cancelled) return;
          h.onProgress({
            stage,
            pct: i * 25,
            msg: `${stage} ${i * 25}%`,
            eta_s: Math.round((total - elapsed - (w * i) / 4) * 0.1),
          });
          await delay(110);
        }
        elapsed += w;
      }
      if (!cancelled) h.onDone({ analysis_url: `/api/tracks/${MOCK_KEY}` });
    })();
    return () => {
      cancelled = true;
    };
  },
  async redecode() {
    await delay(200);
    return 'unsupported';
  },
  getEdits: async () => ({ rev: 0, n_ops: 0, ops: [] }),
  postEdits: needsServer,
  parse: needsServer,
  vocab: needsServer,
  gtState: needsServer,
  gtPutBeats: needsServer,
  gtClearBeats: needsServer,
  gtImport: needsServer,
  gtReview: needsServer,
  gtExport: needsServer,
  gtDraftChart: needsServer,
  async sheets() {
    await delay(80);
    return sheetsNow();
  },
  async sheetsSearch(_key, req) {
    const [a, d] = await loadFixtures();
    const cur = await sheetsNow();
    window.clearTimeout(sheetsTimer);
    const terms = req.query?.trim() ? req.query.trim().split(/\s+/) : cur.query.terms;
    sheetsState = { ...cur, status: 'searching', error: null, query: { ...cur.query, terms }, results: [] };
    sheetsTimer = window.setTimeout(() => {
      sheetsState = { ...(sheetsState ?? cur), status: 'done', results: mockResults(a, d), ts: nowIso() };
    }, 1500);
    return sheetsState;
  },
  async sheetsAdd(_key, req) {
    const [a, d] = await loadFixtures();
    const cur = await sheetsNow();
    await delay(300);
    const text = req.text?.trim() ?? '';
    if (!text && !req.url?.trim()) throw new ApiError(400, 'bad_chart', 'Paste a chart or a link.');
    pasteSeq += 1;
    const aligned = alignedFrom(a, d, (c, i) => (i % 6 === 3 ? SIMPLE(c, d) : null));
    const res: SheetResult = {
      id: `pasted-${pasteSeq}`,
      source: 'pasted',
      url: req.url?.trim() || null,
      title: cur.query.title,
      artist: null,
      year: null,
      key_written: null,
      key_recording: 'B',
      transpose: 0,
      meter: null,
      n_chords: text ? (text.match(/\b[A-G](?:#|b)?[A-Za-z0-9()+/#b-]*/g) ?? []).length : aligned.length,
      match: { title: 1, align: 0.7, score: 0.7 },
      verdict: 'maybe',
      chart_text: text || chartTextFrom(a, aligned),
      aligned,
      warnings: req.url ? ['Read from the link you pasted.'] : [],
    };
    sheetsState = {
      ...cur,
      status: cur.status === 'searching' ? 'searching' : 'done',
      results: [...cur.results, res].toSorted((p, q) => q.match.score - p.match.score),
      ts: nowIso(),
    };
    return sheetsState;
  },
  async sheetsApply(_key, id, mode: SheetApplyMode) {
    const [a, d] = await loadFixtures();
    const cur = await sheetsNow();
    const r = cur.results.find((x) => x.id === id);
    if (!r) throw new ApiError(404, 'no_sheet', 'That sheet is not in the list any more.');
    await delay(400);
    mockRev += 1;
    const applied = { id, group: `sheet-${id}-${mockRev}`, mode, ts: nowIso() };
    sheetsState = { ...cur, applied };
    const flagged = mode === 'refine' ? flaggedFor(a, d, r) : [];
    return {
      analysis: { ...a, edits: { rev: mockRev, n_ops: 1 } },
      display: d,
      rev: mockRev,
      applied,
      summary: { n_changed: mode === 'refine' ? 12 : 15, n_kept: flagged.length, flagged },
    };
  },
  async sheetsUndo() {
    const [a, d] = await loadFixtures();
    const cur = await sheetsNow();
    await delay(250);
    mockRev += 1;
    sheetsState = { ...cur, applied: null };
    return { analysis: { ...a, edits: { rev: mockRev, n_ops: 2 } }, display: d, rev: mockRev, applied: null };
  },
  async getSettings() {
    return readSettings();
  },
  async patchSettings(patch) {
    const s = { ...readSettings(), ...patch };
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    } catch {
      /* ignore */
    }
    return s;
  },
  async exportSheet(_key, p) {
    const [a, d] = await loadFixtures();
    await delay(150);
    return mockExport(a, d, p, sheetsState);
  },
  exportSheetUrl() {
    return null;
  },
  async renameTrack(key, patch) {
    const all = renames();
    all[key] = { ...all[key], ...patch };
    try {
      localStorage.setItem(RENAME_KEY, JSON.stringify(all));
    } catch {
      /* ignore */
    }
  },
};
