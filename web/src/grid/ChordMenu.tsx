import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { api, ApiError } from '../api/client';
import { PHONE_QUERY, useLabelCtx, useMedia } from '../app/hooks';
import { loopBar, seekTo } from '../app/actions';
import { showToast, useApp } from '../app/store';
import { useT } from '../i18n';
import { engine } from '../player/engine';
import { formatTime } from '../player/sync';
import { getTimeline } from './model';
import { chordLabel, harteLabel, isNoChord, segLabel } from './labels';
import { acceptSuggestions, applyResult, rejectSuggestions } from '../editor/ops';
import { selectChord, setEditMode } from '../editor/session';

const close = () => useApp.setState({ menu: null });

/** Right-click / long-press menu: alternatives, equivalents, loop, re-scan (§8.1). */
export function ChordMenu() {
  const t = useT();
  const menu = useApp((s) => s.menu);
  const analysis = useApp((s) => s.analysis);
  const detail = useApp((s) => s.detail);
  const gtMode = useApp((s) => s.gtMode);
  const ctx = useLabelCtx();
  const phone = useMedia(PHONE_QUERY);
  const box = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!menu) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    const onDown = (e: PointerEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) close();
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('pointerdown', onDown, true);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('pointerdown', onDown, true);
    };
  }, [menu]);

  // Keep the popover on screen.
  useLayoutEffect(() => {
    if (!menu || phone || !box.current) {
      setPos(null);
      return;
    }
    const r = box.current.getBoundingClientRect();
    const left = Math.max(8, Math.min(menu.x, window.innerWidth - r.width - 8));
    const top = menu.y + r.height + 8 > window.innerHeight ? Math.max(8, menu.y - r.height) : menu.y;
    setPos({ left, top });
  }, [menu, phone]);

  if (!menu || !analysis) return null;
  const segs = getTimeline(analysis, detail, gtMode);
  const seg = segs[menu.seg];
  if (!seg) return null;
  const c = seg.chord;
  const label = segLabel(seg, ctx);
  const nash = isNoChord(c) || ctx.notation === 'nashville' ? '' : c.degree.nashville;
  const letters = ctx.notation !== 'letters' ? chordLabel(c, { ...ctx, notation: 'letters' }) : '';
  const alts = (c.alternatives ?? []).map((a) => ({ ...harteLabel(analysis, a.harte, ctx), conf: a.conf }));
  const eqs = (c.equivalents ?? []).map((h) => harteLabel(analysis, h, ctx));
  const anyUnmatched = alts.some((a) => !a.matched) || eqs.some((e) => !e.matched);
  const bar = analysis.bars.find((b) => b.i === c.bar);

  const rescan = async () => {
    const { loop } = useApp.getState();
    const region: [number, number] =
      loop.a != null && loop.b != null && loop.a <= seg.start + 1e-3 && loop.b >= seg.end - 1e-3
        ? [loop.a, loop.b]
        : bar
          ? [bar.start, bar.end]
          : [seg.start, seg.end];
    setBusy(true);
    try {
      const r = await api.redecode(analysis.source.key, region);
      if (r === 'unsupported') showToast(t('rescanComingSoon'));
      else {
        // new chords come back as pending suggestions (dashed): accept / reject them here or in the edit bar
        applyResult(analysis.source.key, r);
        showToast(r.n_suggestions != null ? `${t('rescanDone')} ${t('suggestionsN', { n: r.n_suggestions })}` : t('rescanDone'));
      }
      close();
    } catch (e) {
      showToast(`${t('rescanFailed')}: ${e instanceof ApiError ? e.message : String(e)}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  const style = phone ? undefined : { left: pos?.left ?? menu.x, top: pos?.top ?? menu.y, visibility: pos ? undefined : ('hidden' as const) };

  return (
    <>
      {phone && <div className="sheet-backdrop" onClick={close} />}
      <div ref={box} className={phone ? 'chord-menu sheet' : 'chord-menu popover'} style={style} role="dialog" aria-label={label}>
        <div className="cm-head">
          <span className={`cm-label${seg.passing ? ' passing' : ''}`}>{label}</span>
          {letters && <span className="cm-sub">{letters}</span>}
          {nash && <span className="cm-sub">{nash}</span>}
          <button className="icon-btn cm-close" onClick={close} aria-label={t('close')}>
            ×
          </button>
        </div>
        <div className="cm-meta">
          {formatTime(seg.start, true)} · {seg.passing ? t('layerPassing') : t('layerMain')}
          {c.role === 'bass_event' ? ` · ${t('bassEvent')}` : ''} · {t('confidence')} {Math.round(c.conf * 100)}%
        </div>
        {alts.length > 0 && (
          <div className="cm-section">
            <div className="cm-title">{t('alternatives')}</div>
            <ul className="cm-list">
              {alts.map((a, i) => (
                <li key={i}>
                  <span className={a.matched ? 'cm-alt' : 'cm-alt harte'}>{a.text}</span>
                  <span className="cm-conf">
                    <span className="cm-bar" style={{ width: `${Math.round(a.conf * 100)}%` }} />
                    {Math.round(a.conf * 100)}%
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {eqs.length > 0 && (
          <div className="cm-section">
            <div className="cm-title">{t('equivalents')}</div>
            <div className="cm-eqs">
              {eqs.map((e, i) => (
                <span key={i} className={e.matched ? 'chip' : 'chip harte'}>
                  {e.text}
                </span>
              ))}
            </div>
          </div>
        )}
        {anyUnmatched && <div className="cm-note">{t('originalKeyNote')}</div>}
        <div className="cm-actions">
          <button
            onClick={() => {
              seekTo(seg.start);
              void engine.play();
              close();
            }}
          >
            ▶ {t('playFromHere')}
          </button>
          {bar && (
            <button
              onClick={() => {
                loopBar(bar.i);
                close();
              }}
            >
              ⟳ {t('loopThisBar')}
            </button>
          )}
          <button onClick={() => void rescan()} disabled={busy}>
            ⟲ {t('rescan')}
          </button>
          {c.suggestion && (
            <>
              <button
                className="primary"
                onClick={() => {
                  void acceptSuggestions([c]);
                  close();
                }}
              >
                ✓ {t('accept')} ({t('suggestion')})
              </button>
              <button
                onClick={() => {
                  void rejectSuggestions([c]);
                  close();
                }}
              >
                ✗ {t('reject')}
              </button>
            </>
          )}
          <button
            onClick={() => {
              if (!useApp.getState().editMode) setEditMode(true);
              selectChord(c);
              close();
            }}
          >
            ✎ {t('editChord')}
          </button>
        </div>
      </div>
    </>
  );
}
