import type { JobHandlers } from './client';
import type { DoneEvent, ErrorEvent, ProgressEvent } from './types';

function parse<T>(data: unknown): T | null {
  if (typeof data !== 'string' || !data) return null;
  try {
    return JSON.parse(data) as T;
  } catch {
    return null;
  }
}

/**
 * Job progress over SSE (§11): named events `progress`, `done`, `error`.
 * Unnamed messages carrying `{type|event: ...}` are accepted too.
 * Returns a function that closes the stream.
 */
export function watchJobEvents(url: string, h: JobHandlers): () => void {
  const es = new EventSource(url);
  let finished = false;
  const finish = () => {
    finished = true;
    es.close();
  };

  const onProgress = (d: ProgressEvent | null) => {
    if (d) h.onProgress(d);
  };
  const onDone = (d: DoneEvent | null) => {
    finish();
    h.onDone(d ?? {});
  };
  const onError = (d: ErrorEvent | null) => {
    finish();
    h.onError(d ?? { code: 'unknown', msg: 'Analysis failed' });
  };

  es.addEventListener('open', () => h.onConnection(true));
  es.addEventListener('progress', (e) => onProgress(parse<ProgressEvent>((e as MessageEvent).data)));
  es.addEventListener('done', (e) => onDone(parse<DoneEvent>((e as MessageEvent).data)));
  es.addEventListener('error', (e) => {
    if (finished) return;
    const data = e instanceof MessageEvent ? parse<ErrorEvent>(e.data) : null;
    if (data) onError(data);
    else h.onConnection(false); // transport error: EventSource reconnects on its own
  });
  es.addEventListener('message', (e) => {
    const d = parse<Record<string, unknown>>(e.data);
    if (!d) return;
    const type = (d.type ?? d.event) as string | undefined;
    const payload = (d.data ?? d) as Record<string, unknown>;
    if (type === 'progress') onProgress(payload as unknown as ProgressEvent);
    else if (type === 'done') onDone(payload as DoneEvent);
    else if (type === 'error') onError(payload as unknown as ErrorEvent);
  });
  return finish;
}
