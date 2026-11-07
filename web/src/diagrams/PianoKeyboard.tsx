import { memo } from 'react';

const BLACK = new Set([1, 3, 6, 8, 10]);
export const isBlack = (m: number) => BLACK.has(((m % 12) + 12) % 12);

const W = 14; // white key width (SVG units)
const H = 64; // white key height
const BW = 9; // black key width
const BH = 40; // black key height

export interface KeyboardProps {
  lo: number;
  hi: number;
  lh: number[];
  rh: number[];
  /** MIDI -> note name, shown on pressed keys when given. */
  names?: Map<number, string> | null;
  label?: string;
}

/** Song-wide keyboard range for a set of voicings: at least 2 octaves, white-key aligned. */
export function keyboardRange(voicings: { lh?: number[]; rh?: number[] }[]): [number, number] {
  let mn = Infinity;
  let mx = -Infinity;
  for (const v of voicings) {
    for (const n of [...(v.lh ?? []), ...(v.rh ?? [])]) {
      if (n < mn) mn = n;
      if (n > mx) mx = n;
    }
  }
  if (!Number.isFinite(mn)) return [48, 76];
  let lo = mn - 2;
  let hi = mx + 2;
  if (hi - lo < 24) {
    const pad = 24 - (hi - lo);
    lo -= Math.floor(pad / 2);
    hi += Math.ceil(pad / 2);
  }
  while (isBlack(lo)) lo--;
  while (isBlack(hi)) hi++;
  return [lo, hi];
}

function KeyboardImpl({ lo, hi, lh, rh, names, label }: KeyboardProps) {
  const whites: number[] = [];
  for (let m = lo; m <= hi; m++) if (!isBlack(m)) whites.push(m);
  const whiteX = new Map<number, number>();
  whites.forEach((m, i) => whiteX.set(m, i * W));
  const lhSet = new Set(lh);
  const rhSet = new Set(rh);
  const width = whites.length * W;
  const fillOf = (m: number, black: boolean) =>
    lhSet.has(m) ? 'var(--lh)' : rhSet.has(m) ? 'var(--rh)' : black ? 'var(--key-black)' : 'var(--key-white)';

  const blacks: { m: number; x: number }[] = [];
  for (let m = lo; m <= hi; m++) {
    if (!isBlack(m)) continue;
    const left = whiteX.get(m - 1);
    if (left == null) continue;
    blacks.push({ m, x: left + W - BW / 2 });
  }

  return (
    <svg className="kbd" viewBox={`0 0 ${width} ${H + 1}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={label}>
      {whites.map((m) => {
        const x = whiteX.get(m)!;
        const on = lhSet.has(m) || rhSet.has(m);
        return (
          <g key={m}>
            <rect x={x + 0.5} y={0.5} width={W - 1} height={H} rx={2} fill={fillOf(m, false)} stroke="var(--key-edge)" strokeWidth={1} />
            {m === 60 && !on && <circle cx={x + W / 2} cy={H - 6} r={1.6} fill="var(--key-edge)" />}
            {on && names && (
              <text x={x + W / 2} y={H - 5} textAnchor="middle" className={`kbd-name ${lhSet.has(m) ? 'lh' : 'rh'}`}>
                {names.get(m)}
              </text>
            )}
          </g>
        );
      })}
      {blacks.map(({ m, x }) => {
        const on = lhSet.has(m) || rhSet.has(m);
        return (
          <g key={m}>
            <rect x={x} y={0.5} width={BW} height={BH} rx={1.5} fill={fillOf(m, true)} stroke="var(--key-edge)" strokeWidth={0.8} />
            {on && names && (
              <text x={x + BW / 2} y={BH - 4} textAnchor="middle" className={`kbd-name ${lhSet.has(m) ? 'lh' : 'rh'}`}>
                {names.get(m)}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

export const PianoKeyboard = memo(KeyboardImpl);
