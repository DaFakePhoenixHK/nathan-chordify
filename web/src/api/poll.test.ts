import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { JobHandlers } from './client';
import { pollJob, type JobSnapshot } from './poll';

function handlers() {
  return { onProgress: vi.fn(), onDone: vi.fn(), onError: vi.fn(), onConnection: vi.fn() } satisfies JobHandlers;
}

const running: JobSnapshot = { state: 'running', stage: 'chords', pct: 40, msg: 'x', eta_s: 9 };

describe('pollJob', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal('window', globalThis);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('reports progress, then the end, and stops asking', async () => {
    const load = vi.fn<() => Promise<JobSnapshot>>().mockResolvedValueOnce(running).mockResolvedValueOnce({ ...running, state: 'done' });
    const h = handlers();
    pollJob(load, h, 100);
    await vi.advanceTimersByTimeAsync(0);
    expect(h.onProgress).toHaveBeenCalledWith({ stage: 'chords', pct: 40, msg: 'x', eta_s: 9 });
    await vi.advanceTimersByTimeAsync(100);
    expect(h.onDone).toHaveBeenCalledOnce();
    await vi.advanceTimersByTimeAsync(500);
    expect(load).toHaveBeenCalledTimes(2);
  });

  it('passes on the error the computer gave', async () => {
    const h = handlers();
    pollJob(async () => ({ ...running, state: 'error', error: { code: 'bad', msg: 'no' } }), h, 100);
    await vi.advanceTimersByTimeAsync(0);
    expect(h.onError).toHaveBeenCalledWith({ code: 'bad', msg: 'no' });
  });

  it('keeps trying through a lost connection', async () => {
    const load = vi.fn<() => Promise<JobSnapshot>>().mockRejectedValueOnce(new Error('net')).mockResolvedValueOnce(running);
    const h = handlers();
    const stop = pollJob(load, h, 100);
    await vi.advanceTimersByTimeAsync(0);
    expect(h.onConnection).toHaveBeenLastCalledWith(false);
    await vi.advanceTimersByTimeAsync(100);
    expect(h.onConnection).toHaveBeenLastCalledWith(true);
    expect(h.onProgress).toHaveBeenCalledOnce();
    stop();
  });

  it('stops when asked to', async () => {
    const load = vi.fn(async () => running);
    const stop = pollJob(load, handlers(), 100);
    await vi.advanceTimersByTimeAsync(0);
    stop();
    await vi.advanceTimersByTimeAsync(500);
    expect(load).toHaveBeenCalledOnce();
  });
});
