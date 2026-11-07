// Tapped ground-truth beats (§8.4, review E1): pure helpers, unit-tested.

/** Median inter-beat interval of a sorted list (null with fewer than 2 beats). */
export function medianIbi(beats: number[]): number | null {
  if (beats.length < 2) return null;
  const d = beats.slice(1).map((b, i) => b - beats[i]).toSorted((a, b) => a - b);
  return d[Math.floor(d.length / 2)];
}

/** BPM of the last few taps (song time, so it doesn't depend on the playback speed). */
export function tapBpm(beats: number[], last = 8): number | null {
  const ibi = medianIbi(beats.slice(-last));
  return ibi && ibi > 0 ? 60 / ibi : null;
}

/**
 * Add a tap at song time t to a sorted beat list, replacing as you go: the beats between the previous
 * tap of this run and t are dropped (re-tapping a stretch overwrites it), and so is any beat within
 * 40% of a beat of t. `prev` is null for the first tap of a run (after a seek or a pause).
 */
export function tapInto(beats: number[], t: number, prev: number | null): number[] {
  const ibi = medianIbi(beats) ?? 0.9;
  const near = Math.max(0.08, 0.4 * ibi);
  const out = beats.filter((b) => {
    if (Math.abs(b - t) < near) return false;
    if (prev != null && b > prev + 1e-6 && b < t) return false;
    return true;
  });
  out.push(Math.round(t * 1e4) / 1e4);
  return out.toSorted((a, b) => a - b);
}

/** Move the beat nearest t by dt seconds (←/→ nudge), keeping the list sorted. */
export function nudgeNearest(beats: number[], t: number, dt: number): number[] {
  if (beats.length === 0) return beats;
  let k = 0;
  for (let i = 1; i < beats.length; i++) if (Math.abs(beats[i] - t) < Math.abs(beats[k] - t)) k = i;
  const out = beats.slice();
  out[k] = Math.round((out[k] + dt) * 1e4) / 1e4;
  return out.toSorted((a, b) => a - b);
}

/** Index of the beat nearest t (or -1). */
export function nearestBeat(beats: number[], t: number): number {
  let k = -1;
  for (let i = 0; i < beats.length; i++) if (k < 0 || Math.abs(beats[i] - t) < Math.abs(beats[k] - t)) k = i;
  return k;
}
