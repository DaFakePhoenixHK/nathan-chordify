// The "Sound" selector in the transport (2026-11-04): a button with a small menu of what plays, plus the
// chord-voice volume. Icon-only on phones. Options that need the separated stems are greyed with a hint
// when the song has none.
import { useEffect, useRef, useState, type MouseEvent as RMouseEvent } from 'react';
import { useApp, type MixId } from '../app/store';
import { api } from '../api/client';
import { useT, type StringKey } from '../i18n';
import { MIXES, mixDef, stemsListed } from './mix';
import { setMix } from './mixActions';

const noFocus = (e: RMouseEvent) => e.preventDefault();

function SoundIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <path d="M4 9v6h4l5 4V5L8 9H4z" fill="currentColor" />
      <path d="M16 8.5a5 5 0 0 1 0 7" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" />
      <path d="M18.5 5.5a9 9 0 0 1 0 13" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" />
    </svg>
  );
}

export function MixSelect({ phone }: { phone: boolean }) {
  const t = useT();
  const mix = useApp((s) => s.mix);
  const busy = useApp((s) => s.mixBusy);
  const avail = useApp((s) => s.mixAvail);
  const analysis = useApp((s) => s.analysis);
  const audioBase = useApp((s) => s.audioBase);
  const chordsVolume = useApp((s) => s.chordsVolume);
  const audioKey = useApp((s) => s.audioKey);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const localFile = api.mock && !!audioBase && audioBase.startsWith('blob:');
  const stemsOk = stemsListed(analysis) && avail !== false && !localFile;
  const def = mixDef(mix);
  const label = t(`mix_${mix}` as StringKey);
  const stemsHint = localFile ? t('mixMockFile') : t('mixNoStems');

  const pick = (id: MixId) => {
    setOpen(false);
    void setMix(id);
  };

  return (
    <div className="mix-wrap" ref={ref}>
      <button
        className={`mix-btn${mix !== 'original' ? ' on' : ''}${busy ? ' busy' : ''}`}
        onMouseDown={noFocus}
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`${t('sound')}: ${label}`}
        title={`${t('sound')}: ${label} — ${t('soundTitle')}`}
        disabled={!audioBase}
      >
        <SoundIcon />
        {!phone && <span className="mix-label">{label}</span>}
        {busy && <span className="sheets-spin" aria-hidden="true" />}
      </button>
      {open && (
        <div className="mix-pop" role="menu" aria-label={t('sound')}>
          <div className="mix-head">{t('sound')}</div>
          {MIXES.map((m) => {
            const off = m.needsStems && !stemsOk;
            return (
              <button
                key={m.id}
                role="menuitemradio"
                aria-checked={mix === m.id}
                className={`mix-opt${mix === m.id ? ' on' : ''}`}
                disabled={off}
                title={off ? stemsHint : t(`mix_${m.id}_help` as StringKey)}
                onMouseDown={noFocus}
                onClick={() => pick(m.id)}
              >
                <span className="mix-check" aria-hidden="true">
                  {mix === m.id ? '●' : '○'}
                </span>
                <span className="mix-opt-text">
                  <span className="mix-opt-name">{t(`mix_${m.id}` as StringKey)}</span>
                  <span className="mix-opt-help">{off ? stemsHint : t(`mix_${m.id}_help` as StringKey)}</span>
                </span>
              </button>
            );
          })}
          <label className="mix-vol" title={t('chordsVolume')}>
            <span>{t('chordsVolume')}</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={chordsVolume}
              aria-label={t('chordsVolume')}
              onChange={(e) => useApp.setState({ chordsVolume: Number(e.target.value) })}
            />
          </label>
          <div className="mix-note">{t(audioKey === 'transposed' ? 'mixKeyNoteFollow' : 'mixKeyNote')}</div>
        </div>
      )}
      {!phone && def.synth && !open && (
        <input
          className="mix-vol-inline"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={chordsVolume}
          aria-label={t('chordsVolume')}
          title={t('chordsVolume')}
          onChange={(e) => useApp.setState({ chordsVolume: Number(e.target.value) })}
        />
      )}
    </div>
  );
}
