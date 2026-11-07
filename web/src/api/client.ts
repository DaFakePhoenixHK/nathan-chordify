import type {
  Analysis,
  Display,
  DoneEvent,
  EditOp,
  EditsLog,
  EditsPosted,
  ErrorEvent,
  GtExported,
  GtImported,
  GtState,
  JobCreated,
  ParsedChord,
  PickerVocab,
  ProgressEvent,
  ServerSettings,
  SheetApplyMode,
  SheetApplyResult,
  SheetExport,
  SheetsState,
  SheetUndoResult,
  TrackSummary,
  UploadCreated,
} from './types';
import { exportSheetUrl, type SheetExportParams } from '../export/logic';
import { watchJobEvents } from './sse';
import { mockApi } from './mock';

export class ApiError extends Error {
  status: number;
  code: string;
  hint?: string;
  constructor(status: number, code: string, msg: string, hint?: string) {
    super(msg);
    this.status = status;
    this.code = code;
    this.hint = hint;
  }
}

export interface JobHandlers {
  onProgress(e: ProgressEvent): void;
  onDone(e: DoneEvent): void;
  onError(e: ErrorEvent): void;
  /** true = connected, false = connection lost (EventSource retries by itself). */
  onConnection(ok: boolean): void;
}

/** A re-scan returns the song with the new chords as pending suggestions (accept / reject = edit ops), or 'unsupported'. */
export type RedecodeResult = { analysis: Analysis; display: Display; n_suggestions?: number } | 'unsupported';

export interface ParseRequest {
  text?: string;
  texts?: string[];
  hartes?: string[];
  /** Local key, e.g. "B:major". */
  key: string;
  /** Display transposition: typed text is read at this shift; hartes are labelled at it. */
  shift: number;
  pref: string;
}

export interface ParseResponse extends ParsedChord {
  results?: ParsedChord[];
  labels?: Record<string, string[]>;
}

export interface ChartImport {
  text: string;
  format?: 'chart' | 'lab';
  start?: number | null;
  beats?: 'auto' | 'tapped' | 'detected';
  import_id: string;
  device: string;
  replace?: boolean;
}

/** M4 editor + ground-truth routes (§11). */
export interface EditApi {
  getEdits(key: string, since?: number): Promise<EditsLog>;
  postEdits(key: string, ops: EditOp[], device: string): Promise<EditsPosted>;
  parse(req: ParseRequest): Promise<ParseResponse>;
  vocab(key: string, shift: number, pref: string): Promise<PickerVocab>;
  gtState(key: string): Promise<GtState>;
  gtPutBeats(key: string, beats: number[], range?: [number, number]): Promise<GtState>;
  gtClearBeats(key: string): Promise<GtState>;
  gtImport(key: string, req: ChartImport): Promise<GtImported>;
  gtReview(key: string, reviewed: boolean): Promise<{ gt: GtState; files?: Record<string, string>; warnings?: string[] }>;
  gtExport(key: string): Promise<GtExported>;
  gtDraftChart(key: string): Promise<{ text: string; kind: string; file: string }>;
}

/** Chord sheets found online (CantonHymn, Simply Worship, WorshipTogether, Ultimate Guitar) or pasted. */
export interface SheetsApi {
  sheets(key: string): Promise<SheetsState>;
  /** Starts a search; the answer is "searching" — poll `sheets()` every 1.5 s until "done" / "error". */
  sheetsSearch(key: string, req: { query?: string; force?: boolean }): Promise<SheetsState>;
  sheetsAdd(key: string, req: { text?: string; url?: string }): Promise<SheetsState>;
  sheetsApply(key: string, id: string, mode: SheetApplyMode): Promise<SheetApplyResult>;
  sheetsUndo(key: string): Promise<SheetUndoResult>;
  getSettings(): Promise<ServerSettings>;
  patchSettings(patch: Partial<ServerSettings>): Promise<ServerSettings>;
}

/** The printable chord sheet (2026-11-04): GET /api/tracks/{key}/export/sheet. */
export interface ExportApi {
  /** The sheet as parts (format=json) for the print view and the clipboard. */
  exportSheet(key: string, p: SheetExportParams): Promise<SheetExport>;
  /** A download URL for format=txt|chordpro, or null when the files must be made in the browser (demo). */
  exportSheetUrl(key: string, p: SheetExportParams): string | null;
}

export interface Api extends EditApi, SheetsApi, ExportApi {
  readonly mock: boolean;
  listTracks(): Promise<TrackSummary[]>;
  getAnalysis(key: string): Promise<Analysis>;
  getDisplay(key: string, analysis: Analysis): Promise<Display>;
  /** URL for the <audio> element, or null when there is no audio (mock mode before a file is picked). */
  audioUrl(key: string, analysis: Analysis): Promise<string | null>;
  createJob(source: string): Promise<JobCreated>;
  /** Uploads a file; returns the job to watch (queues one when the upload route doesn't). */
  uploadAndQueue(file: File, onProgress: (pct: number) => void): Promise<JobCreated>;
  watchJob(jobId: string, h: JobHandlers): () => void;
  redecode(key: string, region: [number, number]): Promise<RedecodeResult>;
  renameTrack(key: string, patch: { title?: string; artist?: string | null }): Promise<void>;
}

// ---------------------------------------------------------------- real server (§11)

async function parseError(res: Response): Promise<ApiError> {
  let code = `http_${res.status}`;
  let msg = res.statusText || `HTTP ${res.status}`;
  let hint: string | undefined;
  try {
    const j = (await res.json()) as Record<string, unknown>;
    const d = (typeof j.detail === 'object' && j.detail ? j.detail : j) as Record<string, unknown>;
    if (typeof d.code === 'string') code = d.code;
    if (typeof d.msg === 'string') msg = d.msg;
    else if (typeof d.message === 'string') msg = d.message;
    else if (typeof j.detail === 'string') msg = j.detail;
    if (typeof d.hint === 'string') hint = d.hint;
  } catch {
    /* not JSON */
  }
  return new ApiError(res.status, code, msg, hint);
}

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, { ...init, headers: { Accept: 'application/json', ...init?.headers } });
  } catch (e) {
    throw new ApiError(0, 'offline', e instanceof Error ? e.message : String(e));
  }
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

function postJson<T>(url: string, body: unknown, method = 'POST'): Promise<T> {
  return getJson<T>(url, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}

const enc = encodeURIComponent;

/** Accepts `[...]`, `{tracks: [...]}` or `{items: [...]}` and tolerant field names. */
const str = (v: unknown) => (typeof v === 'string' ? v : null);
const num = (v: unknown) => (typeof v === 'number' ? v : null);

export function normalizeTracks(raw: unknown): TrackSummary[] {
  const r = raw as Record<string, unknown>;
  const list = (Array.isArray(raw) ? raw : (r?.tracks ?? r?.items ?? [])) as Record<string, unknown>[];
  return list
    .map((it) => {
      const key = str(it.key) ?? str(it.id) ?? '';
      return {
        key,
        title: str(it.title) ?? key,
        artist: str(it.artist),
        duration: num(it.duration),
        key_name: str(it.key_name) ?? str(it.global_key) ?? str(it.song_key),
        bpm: num(it.bpm) ?? num(it.tempo),
        created_at: str(it.created_at) ?? str(it.analysed_at) ?? str(it.updated_at),
        source_type: (str(it.source_type) ?? str(it.type)) as TrackSummary['source_type'],
      };
    })
    .filter((t) => t.key);
}

function uploadXhr(file: File, onProgress: (pct: number) => void): Promise<UploadCreated> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/uploads');
    xhr.responseType = 'json';
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) onProgress((100 * e.loaded) / e.total);
    });
    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as UploadCreated);
      else {
        const j = (xhr.response ?? {}) as Record<string, unknown>;
        const d = (typeof j.detail === 'object' && j.detail ? j.detail : j) as Record<string, unknown>;
        reject(
          new ApiError(
            xhr.status,
            typeof d.code === 'string' ? d.code : `http_${xhr.status}`,
            typeof d.msg === 'string' ? d.msg : typeof j.detail === 'string' ? j.detail : `HTTP ${xhr.status}`,
            typeof d.hint === 'string' ? d.hint : undefined,
          ),
        );
      }
    });
    xhr.addEventListener('error', () => reject(new ApiError(0, 'offline', 'Network error during upload')));
    const fd = new FormData();
    fd.append('file', file, file.name);
    xhr.send(fd);
  });
}

const serverApi: Api = {
  mock: false,
  async listTracks() {
    return normalizeTracks(await getJson<unknown>('/api/tracks'));
  },
  getAnalysis(key) {
    return getJson<Analysis>(`/api/tracks/${enc(key)}`);
  },
  getDisplay(key, analysis) {
    return getJson<Display>(analysis.display || `/api/tracks/${enc(key)}/display`);
  },
  async audioUrl(key, analysis) {
    return analysis.audio?.play || `/api/tracks/${enc(key)}/audio`;
  },
  createJob(source) {
    return postJson<JobCreated>('/api/jobs', { source, mode: 'full' });
  },
  async uploadAndQueue(file, onProgress) {
    const up = await uploadXhr(file, onProgress);
    if (up.job_id) return { job_id: up.job_id, key: up.key };
    return postJson<JobCreated>('/api/jobs', { upload_id: up.upload_id, mode: 'full' });
  },
  watchJob(jobId, h) {
    return watchJobEvents(`/api/jobs/${enc(jobId)}/events`, h);
  },
  async redecode(key, region) {
    let res: Response;
    try {
      res = await fetch(`/api/tracks/${enc(key)}/redecode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // a region makes the server re-decode it with the more sensitive settings (DESIGN §5.11)
        body: JSON.stringify({ region }),
      });
    } catch (e) {
      throw new ApiError(0, 'offline', e instanceof Error ? e.message : String(e));
    }
    if (res.status === 501 || res.status === 404 || res.status === 405) return 'unsupported';
    if (!res.ok) throw await parseError(res);
    return (await res.json()) as { analysis: Analysis; display: Display };
  },
  async renameTrack(key, patch) {
    await postJson<unknown>(`/api/tracks/${enc(key)}`, patch, 'PATCH');
  },
  getEdits(key, since = 0) {
    return getJson<EditsLog>(`/api/tracks/${enc(key)}/edits?since=${since}`);
  },
  postEdits(key, ops, device) {
    return postJson<EditsPosted>(`/api/tracks/${enc(key)}/edits`, { ops, device });
  },
  parse(req) {
    return postJson<ParseResponse>('/api/theory/parse', req);
  },
  vocab(key, shift, pref) {
    return getJson<PickerVocab>(`/api/theory/vocab?key=${enc(key)}&shift=${shift}&pref=${enc(pref)}`);
  },
  gtState(key) {
    return getJson<GtState>(`/api/tracks/${enc(key)}/gt`);
  },
  gtPutBeats(key, beats, range) {
    return postJson<GtState>(
      `/api/tracks/${enc(key)}/gt/beats`,
      range ? { beats, t0: range[0], t1: range[1] } : { beats },
      'PUT',
    );
  },
  gtClearBeats(key) {
    return getJson<GtState>(`/api/tracks/${enc(key)}/gt/beats`, { method: 'DELETE' });
  },
  gtImport(key, req) {
    return postJson<GtImported>(`/api/tracks/${enc(key)}/gt/chart`, req);
  },
  gtReview(key, reviewed) {
    return postJson<{ gt: GtState; files?: Record<string, string>; warnings?: string[] }>(`/api/tracks/${enc(key)}/gt/review`, {
      reviewed,
    });
  },
  gtExport(key) {
    return postJson<GtExported>(`/api/tracks/${enc(key)}/gt/export`, {});
  },
  gtDraftChart(key) {
    return getJson<{ text: string; kind: string; file: string }>(`/api/tracks/${enc(key)}/gt/draft-chart`);
  },
  sheets(key) {
    return getJson<SheetsState>(`/api/tracks/${enc(key)}/sheets`);
  },
  sheetsSearch(key, req) {
    return postJson<SheetsState>(`/api/tracks/${enc(key)}/sheets/search`, req);
  },
  sheetsAdd(key, req) {
    return postJson<SheetsState>(`/api/tracks/${enc(key)}/sheets/add`, req);
  },
  sheetsApply(key, id, mode) {
    return postJson<SheetApplyResult>(`/api/tracks/${enc(key)}/sheets/${enc(id)}/apply`, { mode });
  },
  sheetsUndo(key) {
    return getJson<SheetUndoResult>(`/api/tracks/${enc(key)}/sheets/applied`, { method: 'DELETE' });
  },
  getSettings() {
    return getJson<ServerSettings>('/api/settings');
  },
  patchSettings(patch) {
    return postJson<ServerSettings>('/api/settings', patch, 'PATCH');
  },
  exportSheet(key, p) {
    return getJson<SheetExport>(exportSheetUrl(key, { ...p, format: 'json' }));
  },
  exportSheetUrl(key, p) {
    return exportSheetUrl(key, p);
  },
};

export const IS_MOCK = (() => {
  const v = new URLSearchParams(location.search).get('mock');
  return v != null && v !== '0' && v !== 'false';
})();

export const api: Api = IS_MOCK ? mockApi : serverApi;
