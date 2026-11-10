import { useEffect, useRef, type MouseEvent as RMouseEvent } from 'react';
import { useApp } from '../app/store';
import { PHONE_QUERY, useMedia, useRenderCount } from '../app/hooks';
import { clearLoop, markLoopA, markLoopB, stepBar } from '../app/actions';
import { toggleEditMode } from '../editor/session';
import { SheetButton } from '../sheets/SheetPanel';
import { useT } from '../i18n';
import { engine } from './engine';
import { IS_IOS } from '../app/platform';
import { unlockSynth } from './chordSynth';
import { mixDef } from './mix';
import { MixSelect } from './MixSelect';
import { formatTime, setClockEls } from './sync';

/** Buttons shouldn't keep focus (Space must stay play/pause, not "click the last button"). */
const noFocus = (e: RMouseEvent) => e.preventDefault();

export function Transport() {
  useRenderCount('Transport');
  const t = useT();
  const playing = useApp((s) => s.playing);
  const duration = useApp((s) => s.duration);
  const loop = useApp((s) => s.loop);
  const audioSrc = useApp((s) => s.audioSrc);
  const volume = useApp((s) => s.volume);
  const editMode = useApp((s) => s.editMode);
  const phone = useMedia(PHONE_QUERY);
  const timeRef = useRef<HTMLSpanElement>(null);
  const seekRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setClockEls({ time: timeRef.current, seek: seekRef.current });
    return () => setClockEls({ time: null, seek: null });
  }, []);

  const loopOn = loop.a != null && loop.b != null;
  const pct = (v: number) => (duration > 0 ? `${(100 * v) / duration}%` : '0%');
  const disabled = !audioSrc;

  return (
    <div className="transport">
      <div className="tp-buttons">
        {!phone && (
          <button className="icon-btn" onMouseDown={noFocus} onClick={() => stepBar(-1)} aria-label={t('prevBar')} title={`${t('prevBar')} (←)`} disabled={disabled}>
            ⏮
          </button>
        )}
        <button
          className="play-btn"
          onMouseDown={noFocus}
          onClick={() => {
            if (mixDef(useApp.getState().mix).synth) unlockSynth(); // iPhones only allow sound to start from a tap
            engine.toggle();
          }}
          aria-label={playing ? t('pause') : t('play')}
          title={`${playing ? t('pause') : t('play')} (Space)`}
          disabled={disabled}
        >
          {playing ? (
            <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
              <rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" />
              <rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
              <path d="M8 5.5v13l11-6.5z" fill="currentColor" />
            </svg>
          )}
        </button>
        {!phone && (
          <button className="icon-btn" onMouseDown={noFocus} onClick={() => stepBar(1)} aria-label={t('nextBar')} title={`${t('nextBar')} (→)`} disabled={disabled}>
            ⏭
          </button>
        )}
      </div>
      <span className="tp-time">
        <span ref={timeRef}>0:00.0</span>
        {!phone && <span className="tp-dur"> / {formatTime(duration)}</span>}
      </span>
      <div className="tp-seek">
        {loop.a != null && (
          <div
            className={`tp-loop${loopOn ? '' : ' only-a'}`}
            style={{ left: pct(loop.a), width: loopOn ? pct((loop.b ?? loop.a) - loop.a) : '2px' }}
          />
        )}
        <input
          ref={seekRef}
          type="range"
          min={0}
          max={duration || 1}
          step={0.01}
          defaultValue={0}
          aria-label={t('seek')}
          disabled={disabled}
          onPointerDown={(e) => {
            e.currentTarget.dataset.dragging = '1';
          }}
          onPointerUp={(e) => {
            e.currentTarget.dataset.dragging = '0';
          }}
          onPointerCancel={(e) => {
            e.currentTarget.dataset.dragging = '0';
          }}
          onInput={(e) => engine.seek(Number(e.currentTarget.value))}
          onKeyDown={(e) => {
            // the global shortcuts own the arrow keys and Space
            if (e.key === ' ' || e.key.startsWith('Arrow')) e.preventDefault();
          }}
        />
      </div>
      <div className="tp-loopbtns" role="group" aria-label={t('loop')}>
        <button
          className={`loop-btn${loop.a != null ? ' on' : ''}`}
          onMouseDown={noFocus}
          onClick={markLoopA}
          title={`${t('loopSetA')} ([)`}
          aria-label={t('loopSetA')}
          disabled={disabled}
        >
          {phone ? 'A' : `⟳ ${t('loopA')}`}
        </button>
        <button
          className={`loop-btn${loop.b != null ? ' on' : ''}`}
          onMouseDown={noFocus}
          onClick={markLoopB}
          title={`${t('loopSetB')} (])`}
          aria-label={t('loopSetB')}
          disabled={disabled}
        >
          {t('loopB')}
        </button>
        {loop.a != null && (
          <button className="loop-btn clear" onMouseDown={noFocus} onClick={clearLoop} title={t('loopClear')} aria-label={t('loopClear')}>
            ×
          </button>
        )}
        {!phone && loopOn && (
          <span className="tp-loopinfo">
            {formatTime(loop.a!, true)}–{formatTime(loop.b!, true)}
          </span>
        )}
      </div>
      <button
        className={`edit-btn${editMode ? ' on' : ''}`}
        onMouseDown={noFocus}
        onClick={toggleEditMode}
        aria-pressed={editMode}
        title={t('editTitle')}
        aria-label={t('editTitle')}
      >
        ✎{phone ? '' : ` ${t('edit')}`}
      </button>
      <SheetButton phone={phone} />
      <MixSelect phone={phone} />
      {!phone && !IS_IOS && (
        <label className="tp-vol" title={t('volume')}>
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M4 9v6h4l5 4V5L8 9H4z" fill="currentColor" /><path d="M16 8.5a5 5 0 0 1 0 7" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" /></svg>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={volume}
            aria-label={t('volume')}
            onChange={(e) => useApp.setState({ volume: Number(e.target.value) })}
          />
        </label>
      )}
    </div>
  );
}
