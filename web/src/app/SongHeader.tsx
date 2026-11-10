import { useApp } from './store';
import { PHONE_QUERY, useMedia } from './hooks';
import { useT } from '../i18n';
import { globalKeySegment } from '../grid/model';
import { keyName, signed } from '../grid/labels';
import { NotationSwitch } from '../controls/Controls';
import { ExportButton } from '../export/ExportMenu';
import { SaveButton } from '../offline/SaveButton';
import { HOSTED, useConnection } from '../remote/connection';

/** Title, key badge (transpose / play-in hint), key changes, BPM, meter, notation (§8.1, §7.2). */
export function SongHeader() {
  const t = useT();
  const analysis = useApp((s) => s.analysis);
  const display = useApp((s) => s.display);
  const transpose = useApp((s) => s.transpose);
  const playIn = useApp((s) => s.playIn);
  const acc = useApp((s) => s.acc);
  const rate = useApp((s) => s.rate);
  const mobileTab = useApp((s) => s.mobileTab);
  const phone = useMedia(PHONE_QUERY);
  const { link } = useConnection();
  const canSave = HOSTED || !!link;
  if (!analysis) return null;

  const gi = globalKeySegment(analysis);
  const orig = keyName(display, analysis, gi, 0, acc);
  const sounding = keyName(display, analysis, gi, transpose, acc);
  const shown = keyName(display, analysis, gi, transpose + playIn, acc);
  const bpm = analysis.tempo.bpm;
  const meter = analysis.meter[0];
  const changes = analysis.key.changes ?? [];
  const kbd = t('kbdHint', { n: signed(-playIn) });

  const badge = (
    <span className="key-badge" title={playIn !== 0 ? kbd : undefined}>
      <span className="kb-label">{t('keyLabel')}</span> <b>{orig}</b>
      {transpose !== 0 && (
        <>
          {' '}
          → <b>{sounding}</b> <small>({signed(transpose)})</small>
        </>
      )}
      {playIn !== 0 && (
        <>
          {' '}
          ▸ {t('playIn')} <b>{shown}</b> <small className="kbd-hint">({t('kbdShort', { n: signed(-playIn) })})</small>
        </>
      )}
    </span>
  );

  if (phone) {
    return (
      <div className="song-header phone">
        <div className="sh-row">
          <a className="icon-btn back" href="#/" aria-label={t('back')}>
            ←
          </a>
          <span className="sh-title" title={analysis.source.title}>
            {analysis.source.title}
          </span>
          {badge}
          <span className="sh-bpm">♩{Math.round(bpm * rate)}</span>
          {canSave && <SaveButton phone />}
          <ExportButton phone />
          <button className="icon-btn" onClick={() => useApp.setState({ sheetOpen: true })} aria-label={t('settings')}>
            ⋯
          </button>
        </div>
        {playIn !== 0 && <div className="sh-hint">{kbd}</div>}
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={mobileTab === 'grid'} className={mobileTab === 'grid' ? 'on' : ''} onClick={() => useApp.setState({ mobileTab: 'grid' })}>
            {t('tabGrid')}
          </button>
          <button role="tab" aria-selected={mobileTab === 'diagrams'} className={mobileTab === 'diagrams' ? 'on' : ''} onClick={() => useApp.setState({ mobileTab: 'diagrams' })}>
            {t('tabDiagrams')}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="song-header">
      <div className="sh-main">
        <h1 className="sh-title" title={analysis.source.title}>
          {analysis.source.title}
          {analysis.source.artist && <span className="sh-artist"> · {analysis.source.artist}</span>}
        </h1>
        {badge}
        {changes.length > 0 && (
          <span className="kc-list">
            {changes.map((c) => {
              const seg = analysis.key.segments.findIndex((s) => Math.abs(s.start - c.t) < 0.06);
              const name = seg >= 0 ? keyName(display, analysis, seg, transpose + playIn, acc) : c.to;
              return (
                <span key={c.bar} className="kc-chip" title={t('keyChangeAt', { bar: c.bar + 1, name })}>
                  {t('bar')} {c.bar + 1} → {name}
                </span>
              );
            })}
          </span>
        )}
        <span className="sh-bpm" title={t('bpm')}>
          ♩ {Math.round(bpm)}
          {rate !== 1 && <small> → {Math.round(bpm * rate)}</small>}
        </span>
        {meter && (
          <span className="sh-meter">
            {meter.num}/{meter.den}
          </span>
        )}
        <span className="sh-spacer" />
        {canSave && <SaveButton phone={false} />}
        <ExportButton phone={false} />
        <NotationSwitch compact />
      </div>
      {playIn !== 0 && <div className="sh-hint">{kbd}</div>}
    </div>
  );
}
