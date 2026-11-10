// Songs kept on this phone or tablet, so they open with no computer and no internet (at church).
// Plain storage only: this file knows nothing about the server.
import { createStore, del, get, keys, set } from 'idb-keyval';
import type { Analysis, Display, TrackSummary } from '../api/types';
import type { ServerMix } from '../player/mix';

export interface SavedInfo extends TrackSummary {
  saved_at: string;
  bytes: number;
  /** Which recordings are stored, as "mix:semitones". */
  audio: string[];
}

export interface SavedAudio {
  mix: ServerMix;
  shift: number;
  blob: Blob;
}

const store = createStore('nc-saved', 'songs');
const infoKey = (k: string) => `i:${k}`;
const analysisKey = (k: string) => `a:${k}`;
const displayKey = (k: string) => `d:${k}`;
const audioKey = (k: string, mix: string, shift: number) => `b:${k}:${mix}:${shift}`;

const fromDevice = new WeakSet<object>();
const blobUrls = new Map<string, string>();

/** True for a song that was opened from this device's storage rather than from the computer. */
export function isFromDevice(a: Analysis): boolean {
  return fromDevice.has(a);
}

export async function listSaved(): Promise<SavedInfo[]> {
  const all = (await keys(store)).filter((k) => String(k).startsWith('i:'));
  const infos = await Promise.all(all.map((k) => get<SavedInfo>(k, store)));
  const found = infos.filter((v): v is SavedInfo => !!v);
  found.sort((a, b) => a.title.localeCompare(b.title));
  return found;
}

export async function isSaved(key: string): Promise<boolean> {
  return (await get(infoKey(key), store)) !== undefined;
}

export async function savedAnalysis(key: string): Promise<Analysis | undefined> {
  const a = await get<Analysis>(analysisKey(key), store);
  if (a) fromDevice.add(a);
  return a;
}

export function savedDisplay(key: string): Promise<Display | undefined> {
  return get<Display>(displayKey(key), store);
}

/** A playable address for a stored recording, or null when that version wasn't saved. */
export async function savedAudioUrl(key: string, mix: ServerMix, shift: number): Promise<string | null> {
  const id = audioKey(key, mix, shift);
  const known = blobUrls.get(id);
  if (known) return known;
  const kept = await get<{ type: string; data: ArrayBuffer }>(id, store);
  if (!kept) return null;
  const url = URL.createObjectURL(new Blob([kept.data], { type: kept.type }));
  blobUrls.set(id, url);
  return url;
}

export async function removeSong(key: string): Promise<void> {
  const info = await get<SavedInfo>(infoKey(key), store);
  for (const a of info?.audio ?? []) {
    const [mix, shift] = a.split(':');
    const id = audioKey(key, mix, Number(shift));
    const url = blobUrls.get(id);
    if (url) URL.revokeObjectURL(url);
    blobUrls.delete(id);
    await del(id, store);
  }
  await del(analysisKey(key), store);
  await del(displayKey(key), store);
  await del(infoKey(key), store);
}

export async function putSong(key: string, analysis: Analysis, display: Display, audio: SavedAudio[], summary: TrackSummary): Promise<void> {
  // the recordings first: a song only counts as saved once everything is in
  // as plain bytes: Safari refuses to store Blobs in private browsing
  for (const a of audio) await set(audioKey(key, a.mix, a.shift), { type: a.blob.type || 'audio/mp4', data: await a.blob.arrayBuffer() }, store);
  await set(analysisKey(key), analysis, store);
  await set(displayKey(key), display, store);
  const info: SavedInfo = {
    ...summary,
    saved_at: new Date().toISOString(),
    bytes: audio.reduce((n, a) => n + a.blob.size, 0),
    audio: audio.map((a) => `${a.mix}:${a.shift}`),
  };
  await set(infoKey(key), info, store);
  // iPhones may clear website data when space runs low, unless the site asks to keep it
  void navigator.storage?.persist?.().catch(() => undefined);
}
