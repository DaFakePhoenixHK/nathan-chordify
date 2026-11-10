import type { JobHandlers } from './client';

export interface JobSnapshot {
  state: string;
  stage: string;
  pct: number;
  msg?: string;
  eta_s?: number | null;
  error?: { code: string; msg: string; hint?: string } | null;
}

/**
 * Job progress by asking every moment, for connections that can't hold a live stream open (public
 * tunnels buffer it, and a phone suspends the page). A lost connection is reported and retried: the
 * job keeps running on the computer. Returns a function that stops asking.
 */
export function pollJob(load: () => Promise<JobSnapshot>, h: JobHandlers, everyMs = 1500): () => void {
  let stopped = false;
  let timer = 0;
  const tick = async () => {
    if (stopped) return;
    try {
      const j = await load();
      if (stopped) return;
      h.onConnection(true);
      if (j.state === 'done') {
        stopped = true;
        h.onDone({});
        return;
      }
      if (j.state === 'error' || j.state === 'cancelled') {
        stopped = true;
        h.onError(j.error ?? { code: j.state, msg: j.msg || 'Analysis stopped' });
        return;
      }
      h.onProgress({ stage: j.stage, pct: j.pct, msg: j.msg, eta_s: j.eta_s });
    } catch {
      if (stopped) return;
      h.onConnection(false);
    }
    timer = window.setTimeout(() => void tick(), everyMs);
  };
  void tick();
  return () => {
    stopped = true;
    window.clearTimeout(timer);
  };
}
