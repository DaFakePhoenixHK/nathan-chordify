import { useMemo, useState } from 'react';
import { useApp } from '../app/store';
import { useLabelCtx, useRenderCount } from '../app/hooks';
import { useT } from '../i18n';
import type { Voicing } from '../api/types';
import { getTimeline, keySegmentIndexAt, type Seg } from '../grid/model';
import { chordLabel, isNoChord, segLabels } from '../grid/labels';
import { keyboardRange, PianoKeyboard } from './PianoKeyboard';
import { namesFor, prefersFlats } from './noteNames';

type Mode = 'side' | 'mini' | 'full';
const NO_SEGS: Seg[] = [];

/** NOW / NEXT / THEN chords with piano diagrams (LH bass in its own colour, RH from the server's voicing). */
export function PianoPanel({ mode }: { mode: Mode }) {
  useRenderCount(`PianoPanel:${mode}`);
  const t = useT();
  const analysis = useApp((s) => s.analysis);
  const display = useApp((s) => s.display);
  const detail = useApp((s) => s.detail);
  const gtMode = useApp((s) => s.gtMode);
  const nowSeg = useApp((s) => s.nowSeg);
  const noteNames = useApp((s) => s.noteNames);
  const acc = useApp((s) => s.acc);
  const ctx = useLabelCtx();
  const [open, setOpen] = useState(true);

  const segs = analysis ? getTimeline(analysis, detail, gtMode) : NO_SEGS;
  const labels = useMemo(() => segLabels(segs, ctx), [segs, ctx]);
  const voicings = display?.voicings[String(ctx.shift)];
  const range = useMemo(() => keyboardRange(voicings ? Object.values(voicings) : []), [voicings]);

  if (!analysis) return null;

  // NOW, then the next two chords whose label differs.
  const picks: number[] = [];
  let i = nowSeg >= 0 ? nowSeg : 0;
  while (i > 0 && !labels.show[i]) i--; // a continuation of the same label: name the chord that started it
  if (segs[i]) picks.push(i);
  for (let j = i + 1; j < segs.length && picks.length < 3; j++) if (labels.show[j]) picks.push(j);

  const voicingOf = (s: Seg): Voicing => voicings?.[s.chord.id] ?? (ctx.shift === 0 ? (s.chord.voicing ?? {}) : {});

  const card = (si: number, role: 'now' | 'next' | 'then') => {
    const s = segs[si];
    const v = voicingOf(s);
    const lh = v.lh ?? [];
    const rh = v.rh ?? [];
    const nc = isNoChord(s.chord);
    const letters = chordLabel(s.chord, { ...ctx, notation: 'letters', slash: true });
    const keyName = ctx.variant?.key_names[keySegmentIndexAt(analysis, s.start)];
    const names = noteNames && !nc ? namesFor([...lh, ...rh], letters, prefersFlats(acc, keyName)) : null;
    const title = labels.text[si];
    return (
      <div key={role} className={`pcard ${role}${s.passing ? ' passing' : ''}`}>
        <div className="pcard-head">
          <span className="pcard-role">{t(`role_${role}`)}</span>
          <span className="pcard-label">{nc ? t('noChord') : title}</span>
          {ctx.notation !== 'letters' && !nc && <span className="pcard-sub">{letters}</span>}
        </div>
        {(mode !== 'mini' || open) && (
          <PianoKeyboard lo={range[0]} hi={range[1]} lh={lh} rh={rh} names={names} label={title} />
        )}
      </div>
    );
  };

  const roles: ('now' | 'next' | 'then')[] = ['now', 'next', 'then'];
  const shown = mode === 'mini' ? picks.slice(0, 2) : picks;

  return (
    <section className={`piano-panel ${mode}`} aria-label={t('piano')}>
      {mode !== 'mini' && (
        <div className="pp-head">
          <span className="pp-title">{t('piano')}</span>
          <span className="legend">
            <i className="sw lh" /> {t('lh')} <i className="sw rh" /> {t('rh')}
          </span>
          <label className="toggle small">
            <input
              type="checkbox"
              checked={noteNames}
              onChange={(e) => useApp.setState({ noteNames: e.target.checked })}
            />
            <span>{t('noteNames')}</span>
          </label>
        </div>
      )}
      <div className="pcards">{shown.map((si, k) => card(si, roles[k]))}</div>
      {mode === 'mini' && (
        <button className="pp-collapse" onClick={() => setOpen(!open)} aria-expanded={open}>
          {open ? '▾' : '▴'}
        </button>
      )}
    </section>
  );
}
