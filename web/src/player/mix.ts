// "Sound" selector logic (2026-11-04, user's request: "play without the singer"). Pure functions, tested.
//
//   original      the recording                                   /audio
//   novocals      recording without the singer                    /audio?mix=novocals   (bass + drums + other)
//   harmony       band only: no singer, no drums                  /audio?mix=harmony    (bass + other)
//   chords        no recording: a piano-like voice plays the main chords in time (the <audio> keeps running
//                 muted, so the clock, loop, tempo and seek all stay the same)
//   piano_chords  recording without the singer + the piano chords
import type { Analysis, Display, Voicing } from '../api/types';
import type { AudioKey, MixId } from '../app/store';
import { getTimeline } from '../grid/model';
import { isNoChord } from '../grid/labels';

export type ServerMix = 'original' | 'novocals' | 'harmony';

export interface MixDef {
  id: MixId;
  /** Which player file to load. */
  audio: ServerMix;
  /** The recording is muted (the chord voice carries the sound). */
  muted: boolean;
  /** The chord voice plays. */
  synth: boolean;
  /** Needs the separated stems on the server. */
  needsStems: boolean;
}

export const MIXES: MixDef[] = [
  { id: 'original', audio: 'original', muted: false, synth: false, needsStems: false },
  { id: 'novocals', audio: 'novocals', muted: false, synth: false, needsStems: true },
  { id: 'harmony', audio: 'harmony', muted: false, synth: false, needsStems: true },
  { id: 'chords', audio: 'original', muted: true, synth: true, needsStems: false },
  { id: 'piano_chords', audio: 'novocals', muted: false, synth: true, needsStems: true },
];

export const MIX_IDS = MIXES.map((m) => m.id);

export function mixDef(id: MixId): MixDef {
  return MIXES.find((m) => m.id === id) ?? MIXES[0];
}

export function isMixId(v: unknown): v is MixId {
  return typeof v === 'string' && (MIX_IDS as string[]).includes(v);
}

/**
 * The player URL for a server mix moved by `transpose` semitones: `?mix=` and `&transpose=` are added (or
 * replaced, or removed when back to the plain recording) on the song's audio URL. The shift is folded to
 * -6..5 first (the server moves the pitch at most half an octave: +7 is the same key as -5).
 */
export function mixAudioUrl(base: string, mix: ServerMix, transpose = 0): string {
  if (base.startsWith('blob:') || base.startsWith('data:')) return base; // a local file: no server mixes
  const [path, query = ''] = base.split('?', 2);
  const params = new URLSearchParams(query);
  if (mix === 'original') params.delete('mix');
  else params.set('mix', mix);
  const n = foldShift(transpose);
  if (n === 0) params.delete('transpose');
  else params.set('transpose', String(n));
  const q = params.toString();
  return q ? `${path}?${q}` : path;
}

/** The same key with the smallest move: -6..5 semitones (+7 → -5, -8 → +4, +12 → 0). */
export function foldShift(n: number): number {
  const m = Math.round(n);
  return ((((m + 6) % 12) + 12) % 12) - 6;
}

/** How many semitones the RECORDING moves: Transpose when the Audio choice is "transposed", else 0. */
export function audioShift(audioKey: AudioKey, transpose: number): number {
  return audioKey === 'transposed' ? foldShift(transpose) : 0;
}

/**
 * How many semitones the chord voice moves. "Chords only" (the recording is muted) always follows
 * Transpose; "Piano + chords" must agree with the recording it plays over: Transpose when the audio
 * follows the transpose, else 0 (the recording's own key).
 */
export function synthShift(def: MixDef, audioKey: AudioKey, transpose: number): number {
  if (!def.synth) return 0;
  if (def.muted) return transpose;
  return audioKey === 'transposed' ? transpose : 0;
}

export function isAudioKey(v: unknown): v is AudioKey {
  return v === 'original' || v === 'transposed';
}

/** The analysis says whether the separated stems exist (audio.stems lists them; fast mode has none). */
export function stemsListed(a: Analysis | null): boolean {
  const st = a?.audio?.stems ?? [];
  return st.includes('vocals') || (st.includes('bass') && st.includes('other'));
}

// ---------------------------------------------------------------- remembered per song (localStorage)

export interface KV {
  getItem(k: string): string | null;
  setItem(k: string, v: string): void;
}

export const MIX_KEY = 'nc.mix.v1';
const MIX_MAX = 300;

function readAll(kv: KV): Record<string, MixId> {
  try {
    const v = JSON.parse(kv.getItem(MIX_KEY) ?? '{}') as unknown;
    if (!v || typeof v !== 'object') return {};
    const out: Record<string, MixId> = {};
    for (const [k, m] of Object.entries(v as Record<string, unknown>)) if (isMixId(m)) out[k] = m;
    return out;
  } catch {
    return {};
  }
}

export function recallMix(kv: KV, songKey: string): MixId {
  return readAll(kv)[songKey] ?? 'original';
}

/** Remember the chosen sound for this song (the newest 300 songs). "original" clears the entry. */
export function rememberMix(kv: KV, songKey: string, mix: MixId): void {
  const all = readAll(kv);
  delete all[songKey];
  if (mix !== 'original') all[songKey] = mix;
  const keys = Object.keys(all).slice(-MIX_MAX);
  const out: Record<string, MixId> = {};
  for (const k of keys) out[k] = all[k];
  try {
    kv.setItem(MIX_KEY, JSON.stringify(out));
  } catch {
    /* private mode: the choice lasts for this session only */
  }
}

// The Audio choice (original / transposed recording), remembered per song the same way. The default is
// "transposed" (a beginner who picks an easy key expects the recording to follow), so only "original" is
// stored.
export const AUDIO_KEY_KEY = 'nc.audiokey.v1';

function readAudioKeys(kv: KV): Record<string, AudioKey> {
  try {
    const v = JSON.parse(kv.getItem(AUDIO_KEY_KEY) ?? '{}') as unknown;
    if (!v || typeof v !== 'object') return {};
    const out: Record<string, AudioKey> = {};
    for (const [k, m] of Object.entries(v as Record<string, unknown>)) if (isAudioKey(m)) out[k] = m;
    return out;
  } catch {
    return {};
  }
}

export function recallAudioKey(kv: KV, songKey: string): AudioKey {
  return readAudioKeys(kv)[songKey] ?? 'transposed';
}

export function rememberAudioKey(kv: KV, songKey: string, v: AudioKey): void {
  const all = readAudioKeys(kv);
  delete all[songKey];
  if (v !== 'transposed') all[songKey] = v;
  const keys = Object.keys(all).slice(-MIX_MAX);
  const out: Record<string, AudioKey> = {};
  for (const k of keys) out[k] = all[k];
  try {
    kv.setItem(AUDIO_KEY_KEY, JSON.stringify(out));
  } catch {
    /* private mode: the choice lasts for this session only */
  }
}

// ---------------------------------------------------------------- the chord voice's timeline

export interface Strike {
  /** Song time (s). */
  t: number;
  /** MIDI notes (left hand + right hand from the server's voicing). */
  notes: number[];
  /** How long it rings (s of song time, before the tempo). */
  dur: number;
  /** 1 on the chord change, softer on the later downbeats. */
  vel: number;
}

const mod12 = (n: number) => ((n % 12) + 12) % 12;
const MAX_RING = 3.5;

/**
 * The strikes of the chord voice: the MAIN layer of the song (detail 0: no passing chords), each chord
 * struck when it starts and again on every later downbeat while it lasts. `shift` = semitones the voicing
 * is moved (Transpose for "Chords only"; 0 for "Piano + chords", which must match the recording).
 */
export function buildStrikes(a: Analysis | null, display: Display | null, shift: number): Strike[] {
  if (!a) return [];
  const segs = getTimeline(a, 0);
  const voicings = display?.voicings[String(mod12(shift))];
  const downbeats = a.bars.filter((b) => !b.free).map((b) => b.start);
  const out: Strike[] = [];
  for (const s of segs) {
    if (isNoChord(s.chord)) continue;
    const v: Voicing | undefined = voicings?.[s.chord.id] ?? (mod12(shift) === 0 ? s.chord.voicing : undefined);
    const notes = [...(v?.lh ?? []), ...(v?.rh ?? [])];
    if (notes.length === 0) continue;
    const times = [s.start];
    for (const d of downbeats) if (d > s.start + 0.15 && d < s.end - 0.15) times.push(d);
    times.forEach((t, i) => {
      const next = times[i + 1] ?? s.end;
      out.push({ t, notes, dur: Math.min(MAX_RING, Math.max(0.2, next - t)), vel: i === 0 ? 1 : 0.6 });
    });
  }
  return out.toSorted((x, y) => x.t - y.t);
}

/** Index of the first strike at or after t (binary search). */
export function firstAtOrAfter(list: { t: number }[], t: number): number {
  let lo = 0;
  let hi = list.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid].t < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
