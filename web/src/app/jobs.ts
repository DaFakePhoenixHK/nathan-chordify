// Analysis jobs: POST /api/jobs (or upload first), then follow SSE progress (§11).
import { api, ApiError } from '../api/client';
import type { JobCreated } from '../api/types';
import { tNow } from '../i18n';
import { navigate, showToast, useApp, type JobState } from './store';

let stopWatch: (() => void) | null = null;

function patchJob(p: Partial<JobState>): void {
  const j = useApp.getState().job;
  if (j) useApp.setState({ job: { ...j, ...p } });
}

function newJob(label: string, status: JobState['status']): void {
  stopWatch?.();
  stopWatch = null;
  useApp.setState({
    job: {
      jobId: null,
      key: null,
      label,
      stage: status === 'uploading' ? 'upload' : 'queued',
      pct: 0,
      msg: '',
      etaS: null,
      status,
      error: null,
      seen: [],
      connectionLost: false,
    },
  });
}

function fail(e: unknown): void {
  const err =
    e instanceof ApiError
      ? {
          code: e.code,
          // 0 = network error; 502–504 = the dev proxy can't reach 127.0.0.1:8765
          msg: e.status === 0 || (e.status >= 502 && e.status <= 504) ? tNow('libraryOffline') : e.message,
          hint: e.hint,
        }
      : { code: 'unknown', msg: e instanceof Error ? e.message : String(e) };
  patchJob({ status: 'error', error: err });
}

function watch(j: JobCreated): void {
  patchJob({ jobId: j.job_id, key: j.key, status: 'running', stage: 'queued', pct: 0 });
  stopWatch = api.watchJob(j.job_id, {
    onProgress(e) {
      const cur = useApp.getState().job;
      const seen = cur && !cur.seen.includes(e.stage) ? [...cur.seen, e.stage] : (cur?.seen ?? []);
      patchJob({ stage: e.stage, pct: e.pct, msg: e.msg ?? '', etaS: e.eta_s ?? null, seen, connectionLost: false });
    },
    onDone() {
      patchJob({ status: 'done', pct: 100, connectionLost: false });
      const st = useApp.getState();
      if (st.route.page === 'song' && st.route.key === j.key) useApp.setState({ songRev: st.songRev + 1 });
      else navigate({ page: 'song', key: j.key });
      window.setTimeout(() => {
        if (useApp.getState().job?.jobId === j.job_id && useApp.getState().job?.status === 'done') useApp.setState({ job: null });
      }, 2500);
    },
    onError(e) {
      patchJob({ status: 'error', error: { code: e.code, msg: e.msg, hint: e.hint } });
    },
    onConnection(ok) {
      patchJob({ connectionLost: !ok });
    },
  });
}

export function looksLikeUrl(s: string): boolean {
  return /^(https?:\/\/|www\.|youtu\.be\/|(m\.|music\.)?youtube\.com\/)/i.test(s.trim());
}

export async function analyzeInput(text: string): Promise<void> {
  const s = text.trim();
  if (!s) return;
  if (!looksLikeUrl(s)) {
    showToast(tNow('searchComingSoon'));
    return;
  }
  const source = /^https?:\/\//i.test(s) ? s : `https://${s}`;
  newJob(source, 'running');
  try {
    watch(await api.createJob(source));
  } catch (e) {
    fail(e);
  }
}

export async function analyzeFile(file: File): Promise<void> {
  newJob(file.name, 'uploading');
  try {
    const j = await api.uploadAndQueue(file, (pct) => patchJob({ pct }));
    watch(j);
  } catch (e) {
    fail(e);
  }
}

export function dismissJob(): void {
  stopWatch?.();
  stopWatch = null;
  useApp.setState({ job: null });
}
