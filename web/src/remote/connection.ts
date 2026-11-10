// How this app reaches the computer that does the chord analysis. On the computer itself nothing
// is needed. A phone holds a "link": the computer's address and its secret phone code.
import { useSyncExternalStore } from 'react';

export interface Link {
  /** The computer's HTTPS address; '' when the computer served this very page. */
  server: string;
  token: string;
}

export type LaptopState = 'unknown' | 'online' | 'offline';

/** Built for a web host (GitHub Pages): no server stands behind this page, so it must be paired. */
export const HOSTED = import.meta.env.VITE_HOSTED === '1';

const KEY = 'nc.link.v1';
const PAIR_MARK = '#/pair?';

const SERVER_OK = /^https:\/\/[^/?#\s]+$/i;
const SERVER_DEV = /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i;

/** Reads a pairing link (or just its "#/pair?…" part). Null when it isn't one. */
export function parsePair(text: string): Link | null {
  const s = text.trim();
  const at = s.indexOf(PAIR_MARK);
  if (at < 0) return null;
  const q = new URLSearchParams(s.slice(at + PAIR_MARK.length));
  const token = q.get('t') ?? '';
  if (token.length < 20 || !/^[\w-]+$/.test(token)) return null;
  const server = (q.get('s') ?? '').replace(/\/+$/, '');
  if (server && !SERVER_OK.test(server) && !SERVER_DEV.test(server)) return null;
  return { server, token };
}

function readStored(): Link | null {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? 'null') as Link | null;
    return v && typeof v.token === 'string' && typeof v.server === 'string' ? v : null;
  } catch {
    return null;
  }
}

let link: Link | null = readStored();
let laptop: LaptopState = 'unknown';
let version = 0;
const listeners = new Set<() => void>();

function changed(): void {
  version++;
  listeners.forEach((fn) => fn());
}

export function getLink(): Link | null {
  return link;
}

export function setLink(next: Link | null): void {
  link = next;
  laptop = 'unknown';
  try {
    if (next) localStorage.setItem(KEY, JSON.stringify(next));
    else localStorage.removeItem(KEY);
  } catch {
    /* private browsing: the link lasts until the page closes */
  }
  changed();
}

/** A hosted copy of the app that doesn't know any computer yet. */
export function needsPairing(): boolean {
  return HOSTED && !link;
}

export function reportLaptop(state: LaptopState): void {
  if (state === laptop) return;
  laptop = state;
  changed();
}

export function laptopState(): LaptopState {
  return laptop;
}

/** Re-renders when the link or the computer's reachability changes. */
export function useConnection(): { link: Link | null; laptop: LaptopState } {
  useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => version,
  );
  return { link, laptop };
}

export function apiUrl(path: string): string {
  if (!link?.server || /^https?:/i.test(path)) return path;
  return `${link.server}${path}`;
}

export function authHeaders(): Record<string, string> {
  return link ? { Authorization: `Bearer ${link.token}` } : {};
}

/** A URL for something that can't send headers (<audio>, a plain download). */
export function mediaUrl(path: string): string {
  if (!link) return path;
  const url = apiUrl(path);
  return `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(link.token)}`;
}

/** Statuses that mean "the computer can't be reached or doesn't know this phone". */
export function isDownStatus(status: number): boolean {
  return status === 0 || status === 401 || status === 502 || status === 503 || status === 504 || (status >= 520 && status <= 530);
}

export type LinkCheck = 'ok' | 'bad_code' | 'unreachable';

/** Tries a link before keeping it. */
export async function checkLink(l: Link): Promise<LinkCheck> {
  const ctl = new AbortController();
  const timer = window.setTimeout(() => ctl.abort(), 12000);
  try {
    const res = await fetch(`${l.server}/api/tracks`, {
      headers: { Accept: 'application/json', Authorization: `Bearer ${l.token}` },
      signal: ctl.signal,
    });
    if (res.ok) return 'ok';
    return res.status === 401 ? 'bad_code' : 'unreachable';
  } catch {
    return 'unreachable';
  } finally {
    window.clearTimeout(timer);
  }
}

// A pairing link opened directly (the phone's camera, or a pasted link): keep it, then forget the
// address bar copy so the code doesn't stay in the browser history.
if (typeof location !== 'undefined' && location.hash.startsWith(PAIR_MARK)) {
  const fromUrl = parsePair(location.hash);
  if (fromUrl) setLink(fromUrl);
  history.replaceState(null, '', `${location.pathname}${location.search}#/`);
}
