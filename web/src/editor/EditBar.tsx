import { useState } from 'react';
import { useApp } from '../app/store';
import { PHONE_QUERY, useMedia } from '../app/hooks';
import { barAt } from '../app/actions';
import { useT } from '../i18n';
import { engine } from '../player/engine';
import { GtBar } from '../gt/GtBar';
import {
  acceptSuggestions,
  describeOp,
  halfDoubleTempo,
  historyRows,
  redo,
  rejectSuggestions,
  setKey,
  setSection,
  toggleOp,
  undo,
} from './ops';
import { setEditMode, setGtMode } from './session';

const noFocus = (e: { preventDefault(): void }) => e.preventDefault();
const TONICS = ['C', 'C#', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];

function History() {
  const t = useT();
  const history = useApp((s) => s.history);
  const rows = historyRows(history);
  return (
    <div className="history" role="list" aria-label={t('history')}>
      {rows.length === 0 && <div className="muted">{t('historyEmpty')}</div>}
      {rows.map((op) => {
        const off = op.active === false;
        const isRevert = op.kind === 'revert';
        return (
          <div key={op.id} role="listitem" className={`hist-row${off ? ' off' : ''}${isRevert ? ' rv' : ''}`}>
            <span className="hist-seq">#{op.seq}</span>
            <span className="hist-what">{describeOp(op)}</span>
            <span className="hist-dev">{op.device}</span>
            {!isRevert && (
              <button className="pill small" onMouseDown={noFocus} onClick={() => void toggleOp(op)}>
                {off ? t('historyRedo') : t('undo')}
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Section label and key from the current bar on (set_section / set_key ops). */
function SectionKeyForm() {
  const t = useT();
  const analysis = useApp((s) => s.analysis);
  const sel = useApp((s) => s.sel);
  const [label, setLabel] = useState('');
  const [tonic, setTonic] = useState('');
  const [minor, setMinor] = useState(false);
  const [fromHere, setFromHere] = useState(false);
  if (!analysis) return null;
  const bar = barAt((sel?.t0 ?? engine.time) + 1e-3);
  return (
    <div className="edit-forms">
      <form
        className="ef-group"
        onSubmit={(e) => {
          e.preventDefault();
          if (bar) void setSection(bar.start, label.trim() || null);
          setLabel('');
        }}
      >
        <span className="ce-title">{bar ? t('sectionAt', { bar: bar.i + 1 }) : t('sectionLabel')}</span>
        <input className="ef-input" value={label} placeholder={t('sectionPlaceholder')} onChange={(e) => setLabel(e.target.value)} maxLength={40} />
        <button type="submit" disabled={!bar}>
          {label.trim() ? t('setLabel') : t('remove')}
        </button>
      </form>
      <form
        className="ef-group"
        onSubmit={(e) => {
          e.preventDefault();
          if (!tonic) return;
          void setKey(`${tonic}:${minor ? 'minor' : 'major'}`, fromHere && bar ? bar.start : null);
        }}
      >
        <span className="ce-title">{t('keyLabelShort')}</span>
        <select className="select" value={tonic} onChange={(e) => setTonic(e.target.value)} aria-label={t('keyLabelShort')}>
          <option value="">—</option>
          {TONICS.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
        <select className="select" value={minor ? 'm' : ''} onChange={(e) => setMinor(e.target.value === 'm')} aria-label="mode">
          <option value="">major</option>
          <option value="m">minor</option>
        </select>
        <select className="select" value={fromHere ? '1' : ''} onChange={(e) => setFromHere(e.target.value === '1')}>
          <option value="">{t('keyWhole')}</option>
          <option value="1">{t('keyFromBar', { bar: (bar?.i ?? 0) + 1 })}</option>
        </select>
        <button type="submit" disabled={!tonic}>
          {t('setLabel')}
        </button>
      </form>
    </div>
  );
}

/** Edit-mode toolbar (§8.4): undo / redo, history, half / double tempo, suggestions, sections, key, GT mode. */
export function EditBar() {
  const t = useT();
  const editMode = useApp((s) => s.editMode);
  const gtMode = useApp((s) => s.gtMode);
  const analysis = useApp((s) => s.analysis);
  const historyOpen = useApp((s) => s.historyOpen);
  const history = useApp((s) => s.history);
  const pending = useApp((s) => s.pendingOps);
  const tapMode = useApp((s) => s.tapMode);
  const phone = useMedia(PHONE_QUERY);
  const [more, setMore] = useState(false);
  if (!editMode || !analysis) return null;
  if (phone && tapMode) {
    // phone: while tapping, the TAP pad gets the room
    return (
      <div className="edit-bar gt">
        <GtBar />
      </div>
    );
  }
  const sugg = analysis.chords.filter((c) => c.suggestion);
  const nOps = history.filter((o) => o.kind !== 'revert').length;
  return (
    <div className={`edit-bar${gtMode ? ' gt' : ''}`}>
      <div className="eb-row">
        <span className="eb-tag">✎ {t('editing')}</span>
        <button onMouseDown={noFocus} onClick={() => void undo()} title={`${t('undo')} (Ctrl+Z)`}>
          ↶ {phone ? '' : t('undo')}
        </button>
        <button onMouseDown={noFocus} onClick={() => void redo()} title={`${t('redo')} (Ctrl+Shift+Z)`}>
          ↷ {phone ? '' : t('redo')}
        </button>
        <button
          className={historyOpen ? 'on' : ''}
          onMouseDown={noFocus}
          onClick={() => useApp.setState({ historyOpen: !historyOpen })}
          aria-expanded={historyOpen}
        >
          {t('history')} ({nOps})
        </button>
        {(!phone || more) && (
          <>
            <button onMouseDown={noFocus} onClick={() => void halfDoubleTempo(0.5)} title={t('halfTempoTitle')}>
              {t('halfTempo')}
            </button>
            <button onMouseDown={noFocus} onClick={() => void halfDoubleTempo(2)} title={t('doubleTempoTitle')}>
              {t('doubleTempo')}
            </button>
          </>
        )}
        <button className={more ? 'on' : ''} onMouseDown={noFocus} onClick={() => setMore(!more)} aria-expanded={more}>
          {phone ? '⋯' : `${t('sectionLabel')} / ${t('keyLabelShort')} ▾`}
        </button>
        {sugg.length > 0 && (
          <span className="eb-sugg">
            <span className="chip suggest-chip">{t('suggestionsN', { n: sugg.length })}</span>
            <button onMouseDown={noFocus} onClick={() => void acceptSuggestions(sugg)}>
              ✓ {t('acceptAll')}
            </button>
            <button onMouseDown={noFocus} onClick={() => void rejectSuggestions(sugg)}>
              ✗ {t('rejectAll')}
            </button>
          </span>
        )}
        <label className="toggle eb-gt" title={t('gtTitle')}>
          <input type="checkbox" checked={gtMode} onChange={(e) => setGtMode(e.target.checked)} />
          <span>{t('gtMode')}</span>
        </label>
        {pending > 0 && <span className="ce-saving">{t('saving')}</span>}
        <span className="sh-spacer" />
        <button className="primary" onMouseDown={noFocus} onClick={() => setEditMode(false)} title="E">
          {t('editDone')}
        </button>
      </div>
      {!phone && <div className="eb-hint">{t('editHint')}</div>}
      {phone && !gtMode && <div className="eb-hint">{t('editHintPhone')}</div>}
      {more && <SectionKeyForm />}
      {historyOpen && <History />}
      {gtMode && <GtBar />}
    </div>
  );
}
