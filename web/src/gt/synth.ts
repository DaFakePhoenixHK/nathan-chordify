// A minimal Web Audio chord voice for Ground-truth mode (§8.4 "Synth A/B"): plays the server's voicing
// (LH bass + RH notes) so a chord can be checked by ear against the recording. The v1.1 ChordSynth
// (electric-piano voice, scheduled with playback) replaces it in M7.

type Ctx = AudioContext;
let ctx: Ctx | null = null;

function audio(): Ctx | null {
  if (ctx) return ctx;
  const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AC) return null;
  ctx = new AC();
  return ctx;
}

const freq = (midi: number) => 440 * 2 ** ((midi - 69) / 12);

/** Play MIDI notes as a soft, slightly rolled chord (must be called from a user gesture on iOS). */
export function playChord(notes: number[], dur = 1.6): void {
  const ac = audio();
  if (!ac || notes.length === 0) return;
  if (ac.state === 'suspended') void ac.resume();
  const now = ac.currentTime + 0.02;
  const master = ac.createGain();
  master.gain.value = 0.22 / Math.sqrt(notes.length);
  master.connect(ac.destination);
  notes
    .toSorted((a, b) => a - b)
    .forEach((m, i) => {
      const t0 = now + i * 0.018;
      for (const [mult, level, type] of [
        [1, 1, 'triangle'],
        [2, 0.18, 'sine'],
      ] as const) {
        const o = ac.createOscillator();
        o.type = type;
        o.frequency.value = freq(m) * mult;
        const g = ac.createGain();
        g.gain.setValueAtTime(0.0001, t0);
        g.gain.exponentialRampToValueAtTime(level, t0 + 0.012);
        g.gain.exponentialRampToValueAtTime(level * 0.35, t0 + 0.45);
        g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
        o.connect(g).connect(master);
        o.start(t0);
        o.stop(t0 + dur + 0.05);
      }
    });
  window.setTimeout(() => master.disconnect(), (dur + 0.3) * 1000);
}
