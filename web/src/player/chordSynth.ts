// The chord voice (2026-11-04): a soft piano-like Web Audio sound that plays the main chords in time with
// the player ("Chords only" and "Piano + chords" sounds). Strikes are scheduled a little ahead of the
// <audio> clock, song time → audio-context time divided by the playback rate, so tempo, loops and seeks
// all keep working. Patterns from Chordify_clone/web/src/audio (scheduler + synth), trimmed.
import { useApp } from '../app/store';
import { engine } from './engine';
import { buildStrikes, firstAtOrAfter, mixDef, synthShift, type Strike } from './mix';

let ctx: AudioContext | null = null;
let bus: GainNode | null = null;
let vol = 0.7;

interface Voice {
  gain: GainNode;
  oscs: OscillatorNode[];
  end: number;
}
const voices: Voice[] = [];

/** Dev-only counters (the scheduler really plays). */
export const synthStats = { strikes: 0 };
if (import.meta.env.DEV) (window as unknown as { ncSynth: typeof synthStats }).ncSynth = synthStats;

/** Slider 0..1 → gain (softer than linear). */
const curve = (v: number) => Math.max(0, Math.min(1, v)) ** 1.6;

function makeCtx(): AudioContext | null {
  if (ctx) return ctx;
  const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AC) return null;
  ctx = new AC();
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -14;
  comp.ratio.value = 4;
  comp.connect(ctx.destination);
  bus = ctx.createGain();
  bus.gain.value = curve(vol);
  bus.connect(comp);
  return ctx;
}

/** Create / resume the context (call from a user gesture: the Sound menu, the play button). */
export function unlockSynth(): void {
  const ac = makeCtx();
  if (ac && ac.state === 'suspended') void ac.resume();
}

export function setChordsVolume(v: number): void {
  vol = v;
  if (ctx && bus) bus.gain.setTargetAtTime(curve(v), ctx.currentTime, 0.03);
}

const freq = (midi: number) => 440 * 2 ** ((midi - 69) / 12);

/** Partials of the piano-ish voice: [frequency ratio, level, wave, detune cents]. */
const PARTIALS: [number, number, OscillatorType, number][] = [
  [1, 0.7, 'triangle', -3],
  [1, 0.45, 'sine', 4],
  [2, 0.16, 'sine', 0],
  [3, 0.05, 'sine', 2],
];

/** Strike MIDI notes at audio time `when` (slightly rolled, bass first) and let them ring `dur` seconds. */
export function strike(notes: number[], when: number, dur: number, velocity = 1): void {
  const ac = makeCtx();
  if (!ac || !bus || notes.length === 0) return;
  synthStats.strikes++;
  const t0base = Math.max(when, ac.currentTime + 0.005);
  const sorted = notes.toSorted((a, b) => a - b);
  const level = (0.28 * velocity) / Math.sqrt(sorted.length);
  const ring = Math.max(0.25, dur);
  sorted.forEach((m, i) => {
    const t0 = t0base + i * 0.012;
    const g = ac.createGain();
    const lp = ac.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(Math.min(9000, freq(m) * 6), t0);
    lp.frequency.exponentialRampToValueAtTime(Math.min(4000, freq(m) * 2.5), t0 + 0.6);
    lp.Q.value = 0.3;
    const peak = level * (m < 52 ? 1.15 : 1);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(peak, t0 + 0.006);
    g.gain.exponentialRampToValueAtTime(peak * 0.45, t0 + 0.18);
    g.gain.exponentialRampToValueAtTime(peak * 0.16, t0 + Math.min(ring, 1.6));
    g.gain.setTargetAtTime(0.0001, t0 + ring, 0.12);
    lp.connect(g).connect(bus!);
    const oscs: OscillatorNode[] = [];
    for (const [ratio, lv, type, cents] of PARTIALS) {
      const o = ac.createOscillator();
      o.type = type;
      o.frequency.value = freq(m) * ratio;
      o.detune.value = cents;
      const pg = ac.createGain();
      pg.gain.value = lv;
      o.connect(pg).connect(lp);
      o.start(t0);
      o.stop(t0 + ring + 0.8);
      oscs.push(o);
    }
    voices.push({ gain: g, oscs, end: t0 + ring + 0.8 });
  });
  for (let i = voices.length - 1; i >= 0; i--) if (voices[i].end < ac.currentTime) voices.splice(i, 1);
}

/** Fade out every chord still ringing (pause / seek / sound change). */
export function silence(): void {
  if (!ctx) return;
  const now = ctx.currentTime;
  for (const v of voices) {
    try {
      v.gain.gain.cancelScheduledValues(now);
      v.gain.gain.setTargetAtTime(0.0001, now, 0.03);
      for (const o of v.oscs) o.stop(now + 0.25);
    } catch {
      /* already stopped */
    }
  }
  voices.length = 0;
}

/** Play one chord now (a tap on a diagram or the chord entry). */
export function previewChord(notes: number[]): void {
  unlockSynth();
  if (!ctx) return;
  strike(notes, ctx.currentTime + 0.02, 1.4, 1);
}

// ---------------------------------------------------------------- the scheduler

const LOOKAHEAD = 0.15; // s of song time per tick (× rate)
const TICK_MS = 25;

let strikes: Strike[] = [];
let timer = 0;
let until: number | null = null;
let lastT = 0;
let wired = false;

function synthOn(): boolean {
  return mixDef(useApp.getState().mix).synth;
}

function reset(): void {
  silence();
  until = null;
}

function rebuild(): void {
  const s = useApp.getState();
  const def = mixDef(s.mix);
  // "Chords only" follows Transpose (practice in the easy key); "Piano + chords" must match the recording,
  // which follows Transpose only when the Audio choice is "transposed".
  strikes = def.synth ? buildStrikes(s.analysis, s.display, synthShift(def, s.audioKey, s.transpose)) : [];
  reset();
}

function tick(): void {
  if (engine.paused || !synthOn() || !ctx) return;
  const st = useApp.getState();
  const t = engine.time;
  const r = Math.max(0.25, st.rate);
  if (until != null && (t < lastT - 0.05 || t > until + 0.5)) reset(); // a jump the events didn't see
  lastT = t;
  const from = until ?? t - 0.01;
  const to = t + LOOKAHEAD * r;
  if (to <= from) return;
  const now = ctx.currentTime;
  const loopEnd = st.loop.a != null && st.loop.b != null ? st.loop.b : Infinity;
  for (let i = firstAtOrAfter(strikes, from); i < strikes.length && strikes[i].t < to; i++) {
    const e = strikes[i];
    if (e.t >= loopEnd - 0.01) break;
    strike(e.notes, now + Math.max(0, e.t - t) / r, Math.min(e.dur, loopEnd - e.t) / r, e.vel);
  }
  until = to;
}

function start(): void {
  if (timer) return;
  until = null;
  lastT = engine.time;
  if (synthOn()) unlockSynth();
  timer = window.setInterval(tick, TICK_MS);
  tick();
}

function stop(): void {
  window.clearInterval(timer);
  timer = 0;
  reset();
}

/** Wire the chord voice to the player and the store (once; called from the Sound selector / song page). */
export function initChordSynth(): void {
  if (wired) return;
  wired = true;
  const el = engine.el;
  el.addEventListener('play', start);
  el.addEventListener('playing', () => {
    until = null;
  });
  el.addEventListener('pause', stop);
  el.addEventListener('seeking', reset);
  el.addEventListener('ratechange', reset);
  setChordsVolume(useApp.getState().chordsVolume);
  rebuild();
  useApp.subscribe((s, p) => {
    if (s.analysis !== p.analysis || s.display !== p.display || s.transpose !== p.transpose || s.mix !== p.mix || s.audioKey !== p.audioKey) {
      rebuild();
      if (s.mix !== p.mix && mixDef(s.mix).synth && !engine.paused) {
        unlockSynth();
        if (!timer) start();
      }
    }
    if (s.chordsVolume !== p.chordsVolume) setChordsVolume(s.chordsVolume);
    if (s.loop !== p.loop) reset();
  });
}
