import { describe, expect, it } from 'vitest';
import analysisJson from '../../../shared/fixtures/huijia.analysis.json';
import displayJson from '../../../shared/fixtures/huijia.display.json';
import type { Analysis, Display } from '../api/types';
import {
  audioShift,
  buildStrikes,
  firstAtOrAfter,
  foldShift,
  isMixId,
  MIXES,
  mixAudioUrl,
  mixDef,
  recallAudioKey,
  recallMix,
  rememberAudioKey,
  rememberMix,
  stemsListed,
  synthShift,
  type KV,
} from './mix';

const a = analysisJson as unknown as Analysis;
const d = displayJson as unknown as Display;

function kv(): KV & { store: Record<string, string> } {
  const store: Record<string, string> = {};
  return {
    store,
    getItem: (k) => store[k] ?? null,
    setItem: (k, v) => {
      store[k] = v;
    },
  };
}

describe('the Sound selector', () => {
  it('builds the player URL for each server mix (and leaves local files alone)', () => {
    expect(mixAudioUrl('/api/tracks/yt-X/audio', 'novocals')).toBe('/api/tracks/yt-X/audio?mix=novocals');
    expect(mixAudioUrl('/api/tracks/yt-X/audio', 'harmony')).toBe('/api/tracks/yt-X/audio?mix=harmony');
    expect(mixAudioUrl('/api/tracks/yt-X/audio?mix=harmony', 'original')).toBe('/api/tracks/yt-X/audio');
    expect(mixAudioUrl('/api/tracks/yt-X/audio?mix=harmony&x=1', 'novocals')).toBe('/api/tracks/yt-X/audio?mix=novocals&x=1');
    expect(mixAudioUrl('blob:http://localhost/abc', 'novocals')).toBe('blob:http://localhost/abc');
  });

  it('adds the transpose to the player URL (folded to the smallest move) and drops it at 0', () => {
    expect(mixAudioUrl('/api/tracks/yt-X/audio', 'original', -3)).toBe('/api/tracks/yt-X/audio?transpose=-3');
    expect(mixAudioUrl('/api/tracks/yt-X/audio', 'novocals', 2)).toBe('/api/tracks/yt-X/audio?mix=novocals&transpose=2');
    expect(mixAudioUrl('/api/tracks/yt-X/audio', 'harmony', 7)).toBe('/api/tracks/yt-X/audio?mix=harmony&transpose=-5');
    expect(mixAudioUrl('/api/tracks/yt-X/audio?mix=novocals&transpose=2', 'original', 0)).toBe('/api/tracks/yt-X/audio');
    expect(mixAudioUrl('/api/tracks/yt-X/audio?mix=novocals&transpose=2', 'novocals', -1)).toBe('/api/tracks/yt-X/audio?mix=novocals&transpose=-1');
    expect(mixAudioUrl('/api/tracks/yt-X/audio', 'original', 12)).toBe('/api/tracks/yt-X/audio');
    expect(mixAudioUrl('/api/tracks/yt-X/audio', 'original')).toBe('/api/tracks/yt-X/audio');
    expect(mixAudioUrl('blob:http://localhost/abc', 'original', -3)).toBe('blob:http://localhost/abc');
  });

  it('folds a transposition to -6..5: the same key with the smallest move', () => {
    expect([0, 1, 5, 6, 7, 11, 12, -1, -6, -7, -12].map(foldShift)).toEqual([0, 1, 5, -6, -5, -1, 0, -1, -6, 5, 0]);
    expect(audioShift('transposed', 7)).toBe(-5);
    expect(audioShift('transposed', -3)).toBe(-3);
    expect(audioShift('original', 7)).toBe(0);
  });

  it('moves the chord voice with the recording it plays over', () => {
    expect(synthShift(mixDef('chords'), 'original', 3)).toBe(3); // the recording is muted: practice key
    expect(synthShift(mixDef('chords'), 'transposed', 3)).toBe(3);
    expect(synthShift(mixDef('piano_chords'), 'original', 3)).toBe(0); // the recording plays in its own key
    expect(synthShift(mixDef('piano_chords'), 'transposed', 3)).toBe(3); // both moved: they agree
    expect(synthShift(mixDef('novocals'), 'transposed', 3)).toBe(0); // no chord voice
  });

  it('remembers the Audio choice per song and defaults to "transposed"', () => {
    const s = kv();
    expect(recallAudioKey(s, 'yt-A')).toBe('transposed');
    rememberAudioKey(s, 'yt-A', 'original');
    expect(recallAudioKey(s, 'yt-A')).toBe('original');
    expect(JSON.parse(s.store['nc.audiokey.v1'])).toEqual({ 'yt-A': 'original' });
    rememberAudioKey(s, 'yt-A', 'transposed');
    expect(recallAudioKey(s, 'yt-A')).toBe('transposed');
    expect(JSON.parse(s.store['nc.audiokey.v1'])).toEqual({});
    s.store['nc.audiokey.v1'] = '{"yt-B": "weird", "yt-C": "original"}';
    expect(recallAudioKey(s, 'yt-B')).toBe('transposed');
    expect(recallAudioKey(s, 'yt-C')).toBe('original');
    s.store['nc.audiokey.v1'] = 'nope';
    expect(recallAudioKey(s, 'yt-C')).toBe('transposed');
  });

  it('knows which sounds need the separated stems and which play the chord voice', () => {
    expect(MIXES.map((m) => m.id)).toEqual(['original', 'novocals', 'harmony', 'chords', 'piano_chords']);
    expect(mixDef('chords')).toMatchObject({ audio: 'original', muted: true, synth: true, needsStems: false });
    expect(mixDef('piano_chords')).toMatchObject({ audio: 'novocals', muted: false, synth: true, needsStems: true });
    expect(mixDef('harmony').needsStems && mixDef('novocals').needsStems).toBe(true);
    expect(mixDef('original')).toMatchObject({ muted: false, synth: false, needsStems: false });
    expect(isMixId('harmony')).toBe(true);
    expect(isMixId('karaoke')).toBe(false);
  });

  it('reads the stems list of the analysis', () => {
    expect(stemsListed(a)).toBe(true);
    expect(stemsListed({ ...a, audio: { play: '/x', stems: [] } })).toBe(false);
    expect(stemsListed({ ...a, audio: { play: '/x' } })).toBe(false);
    expect(stemsListed(null)).toBe(false);
  });

  it('remembers the choice per song in localStorage and forgets "original"', () => {
    const s = kv();
    expect(recallMix(s, 'yt-A')).toBe('original');
    rememberMix(s, 'yt-A', 'harmony');
    rememberMix(s, 'yt-B', 'chords');
    expect(recallMix(s, 'yt-A')).toBe('harmony');
    expect(recallMix(s, 'yt-B')).toBe('chords');
    rememberMix(s, 'yt-A', 'original');
    expect(recallMix(s, 'yt-A')).toBe('original');
    expect(JSON.parse(s.store['nc.mix.v1'])).toEqual({ 'yt-B': 'chords' });
    s.store['nc.mix.v1'] = '{"yt-C": "karaoke", "yt-D": "novocals"}';
    expect(recallMix(s, 'yt-C')).toBe('original');
    expect(recallMix(s, 'yt-D')).toBe('novocals');
    s.store['nc.mix.v1'] = 'not json';
    expect(recallMix(s, 'yt-D')).toBe('original');
  });
});

describe('the chord voice timeline', () => {
  it('strikes every main chord (no passing chords, no N.C.) with the server voicing, sorted in time', () => {
    const strikes = buildStrikes(a, d, 0);
    const mains = a.chords.filter((c) => c.layer === 'main' && c.harte !== 'N');
    expect(strikes.length).toBeGreaterThanOrEqual(mains.length);
    for (let i = 1; i < strikes.length; i++) expect(strikes[i].t).toBeGreaterThanOrEqual(strikes[i - 1].t);
    const first = mains[0];
    const s0 = strikes.find((s) => Math.abs(s.t - first.start) < 1e-6)!;
    const v = d.voicings['0'][first.id];
    expect(s0.notes).toEqual([...(v.lh ?? []), ...(v.rh ?? [])]);
    expect(s0.vel).toBe(1);
    expect(s0.dur).toBeGreaterThan(0.2);
    // nothing plays where there is no chord
    const nc = a.chords.filter((c) => c.layer === 'main' && c.harte === 'N');
    for (const c of nc) expect(strikes.some((s) => Math.abs(s.t - c.start) < 1e-6 && s.notes.length)).toBe(false);
    // a long chord rings again on the downbeats inside it (softer)
    const again = strikes.filter((s) => s.vel < 1);
    expect(again.length).toBeGreaterThan(0);
    for (const s of again) expect(a.bars.some((b) => Math.abs(b.start - s.t) < 1e-6)).toBe(true);
  });

  it('moves the voicing with the transposition', () => {
    const s0 = buildStrikes(a, d, 0);
    const s2 = buildStrikes(a, d, 2);
    expect(s2.length).toBe(s0.length);
    const n0 = s0.find((s) => s.notes.length)!.notes;
    const n2 = s2.find((s) => s.notes.length)!.notes;
    expect(n2.length).toBe(n0.length);
    // display.voicings["2"] is the server's transposed voicing: every note moves by 2 (or an octave of it)
    for (let i = 0; i < n0.length; i++) expect((n2[i] - n0[i] - 2) % 12).toBe(0);
    expect(buildStrikes(null, d, 0)).toEqual([]);
  });

  it('finds the first strike at or after a time', () => {
    const list = [{ t: 1 }, { t: 2 }, { t: 2 }, { t: 5 }];
    expect(firstAtOrAfter(list, 0)).toBe(0);
    expect(firstAtOrAfter(list, 2)).toBe(1);
    expect(firstAtOrAfter(list, 2.5)).toBe(3);
    expect(firstAtOrAfter(list, 9)).toBe(4);
  });
});
