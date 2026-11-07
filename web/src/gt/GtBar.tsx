import { useEffect, useRef, useState } from 'react';
import { api, ApiError } from '../api/client';
import type { ChordOut } from '../api/types';
import { setLoop, setRate, showToast, useApp } from '../app/store';
import { barAt } from '../app/actions';
import { PHONE_QUERY, useMedia } from '../app/hooks';
import { tNow, useT } from '../i18n';
import { engine } from '../player/engine';
import { formatTime } from '../player/sync';
import { applyResult, deviceId, newId, refreshHistory } from '../editor/ops';
import { barLoopOn, playChordAt, selectChord, toggleBarLoop } from '../editor/session';
import { nudgeNearest, tapBpm, tapInto } from './tap';

const noFocus = (e: { preventDefault(): void }) => e.preventDefault();

function songKey(): string | null {
  const r = useApp.getState().route;
  return r.page === 'song' ? r.key : null;
}

// ---------------------------------------------------------------- tap beats (§8.4: at 70%, Space on every beat)

let lastTap: { t: number; wall: number } | null = null;
let tapPrevRate: number | null = null;

export function startTapping(): void {
  const st = useApp.getState();
  tapPrevRate = st.rate;
  setRate(0.7);
  lastTap = null;
  useApp.setState({ tapMode: true, tapDraft: [...(st.gt?.beats ?? [])], tapDirty: false });
}

/** Space / TAP: the first press starts playback, then every press is a beat at the song position. */
export function tapNow(): void {
  const st = useApp.getState();
  if (!st.tapMode || !st.tapDraft) return;
  if (engine.paused) {
    void engine.play();
    lastTap = null;
    return;
  }
  const t = engine.time;
  const now = performance.now();
  const prev = lastTap && now - lastTap.wall < 3000 && t > lastTap.t ? lastTap.t : null;
  lastTap = { t, wall: now };
  useApp.setState({ tapDraft: tapInto(st.tapDraft, t, prev), tapDirty: true });
  const pad = document.querySelector('.tap-pad');
  pad?.classList.remove('flash');
  void (pad as HTMLElement | null)?.offsetWidth;
  pad?.classList.add('flash');
}

/** ←/→ in tap mode: move the beat nearest the playhead by 20 ms. */
export function nudgeTap(dt: number): void {
  const st = useApp.getState();
  if (!st.tapDraft?.length) return;
  useApp.setState({ tapDraft: nudgeNearest(st.tapDraft, engine.time, dt), tapDirty: true });
}

export async function saveTaps(): Promise<boolean> {
  const key = songKey();
  const st = useApp.getState();
  if (!key || !st.tapDraft) return false;
  try {
    const gt = st.tapDraft.length >= 2 ? await api.gtPutBeats(key, st.tapDraft) : await api.gtClearBeats(key);
    useApp.setState({ gt, tapDirty: false, tapDraft: [...gt.beats] });
    showToast(tNow('beatsSaved', { n: gt.beats.length }));
    return true;
  } catch (e) {
    showToast(e instanceof Error ? e.message : String(e), 'error');
    return false;
  }
}

export async function stopTapping(): Promise<void> {
  const st = useApp.getState();
  if (st.tapDirty && !(await saveTaps())) return;
  engine.pause();
  if (tapPrevRate != null) setRate(tapPrevRate);
  tapPrevRate = null;
  useApp.setState({ tapMode: false, tapDraft: null, tapDirty: false });
}

function TapPanel() {
  const t = useT();
  const draft = useApp((s) => s.tapDraft) ?? [];
  const dirty = useApp((s) => s.tapDirty);
  const bpm = tapBpm(draft);
  return (
    <div className="tap-panel">
      <button className="tap-pad" onMouseDown={noFocus} onPointerDown={(e) => { e.preventDefault(); tapNow(); }}>
        {t('tap')}
      </button>
      <div className="tap-info">
        <b>{t('tapsN', { n: draft.length })}</b>
        {bpm && <span> · ♩ {bpm.toFixed(1)}</span>}
        <div className="muted small">{t('tapHelp')}</div>
      </div>
      <div className="tap-actions">
        <button className="primary" onMouseDown={noFocus} onClick={() => void saveTaps()} disabled={!dirty}>
          {t('saveBeats')}
        </button>
        <button onMouseDown={noFocus} onClick={() => useApp.setState({ tapDraft: [], tapDirty: true })}>
          {t('clearBeats')}
        </button>
        <button onMouseDown={noFocus} onClick={() => void stopTapping()}>
          {t('editDone')}
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- chart import

const close = () => useApp.setState({ chartOpen: false });

function ChartDialog() {
  const t = useT();
  const gt = useApp((s) => s.gt);
  const phone = useMedia(PHONE_QUERY);
  const [text, setText] = useState(gt?.chart?.text ?? '');
  const [start, setStart] = useState<string>(gt?.chart?.start != null ? gt.chart.start.toFixed(2) : '');
  const [beats, setBeats] = useState<'auto' | 'tapped' | 'detected'>('auto');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const area = useRef<HTMLTextAreaElement>(null);
  useEffect(() => area.current?.focus(), []);

  const loadDraft = async () => {
    const key = songKey();
    if (!key) return;
    try {
      const d = await api.gtDraftChart(key);
      setText(d.text);
      setMsg(d.file);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    }
  };

  const submit = async () => {
    const key = songKey();
    if (!key || !text.trim()) return;
    setBusy(true);
    setMsg(null);
    try {
      const s = start.trim() === '' ? null : Number(start.replace(',', '.'));
      const r = await api.gtImport(key, {
        text,
        start: s != null && Number.isFinite(s) ? s : null,
        beats,
        import_id: newId().slice(0, 13),
        device: deviceId(),
        replace: true,
      });
      applyResult(key, { analysis: r.analysis, display: r.display, rev: r.rev });
      useApp.setState({ gt: r.gt, chartOpen: false });
      void refreshHistory(key);
      showToast(t('imported', { n: r.n_chords, u: r.unsure.length }) + (r.warnings.length ? ` · ${r.warnings.join(' · ')}` : ''));
    } catch (e) {
      setMsg(e instanceof ApiError ? `${e.message}${e.hint ? ` — ${e.hint}` : ''}` : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="sheet-backdrop" onClick={close} />
      <div className={`chart-dialog${phone ? ' sheet' : ''}`} role="dialog" aria-label={t('chartTitle')} onPointerDown={(e) => e.stopPropagation()}>
        <div className="sheet-head">
          <b>{t('chartTitle')}</b>
          <button className="icon-btn" onClick={close} aria-label={t('close')}>
            ×
          </button>
        </div>
        <p className="muted small">{t('chartHelp')}</p>
        <textarea
          ref={area}
          className="chart-text"
          value={text}
          spellCheck={false}
          placeholder={'[Pre-chorus]\n| B . . (. B+7) | Emaj7 . A7 Em6 |'}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.stopPropagation()}
        />
        <div className="chart-opts">
          <label className="ef-group">
            <span className="ce-title">{t('chartStart')}</span>
            <input className="ef-input num" inputMode="decimal" value={start} placeholder="auto" onChange={(e) => setStart(e.target.value)} onKeyDown={(e) => e.stopPropagation()} />
            <span className="muted">s</span>
            <button type="button" onClick={() => setStart(engine.time.toFixed(2))}>
              {t('usePlayhead')} ({formatTime(engine.time, true)})
            </button>
          </label>
          <label className="ef-group">
            <span className="ce-title">{t('beatsSource')}</span>
            <select className="select" value={beats} onChange={(e) => setBeats(e.target.value as typeof beats)}>
              {(['auto', 'tapped', 'detected'] as const).map((b) => (
                <option key={b} value={b}>
                  {t(`beats_${b}`)}
                </option>
              ))}
            </select>
          </label>
        </div>
        {msg && <div className="chart-msg">{msg}</div>}
        <div className="chart-actions">
          {gt?.draft_chart && (
            <button type="button" onClick={() => void loadDraft()}>
              {t('loadDraft')}
            </button>
          )}
          <span className="sh-spacer" />
          <button type="button" onClick={close}>
            {t('cancel')}
          </button>
          <button type="button" className="primary" disabled={busy || !text.trim()} onClick={() => void submit()}>
            {t('importChart')}
          </button>
        </div>
      </div>
    </>
  );
}

// ---------------------------------------------------------------- the GT toolbar

function unsureChords(chords: ChordOut[]): ChordOut[] {
  return chords.filter((c) => c.edited && c.conf < 0.5).toSorted((a, b) => a.start - b.start);
}

function goTo(c: ChordOut): void {
  selectChord(c);
  const bar = barAt(c.start + 1e-3);
  if (bar) setLoop(bar.start, bar.end);
  engine.seek(bar ? bar.start : c.start);
  document.querySelector('.esel')?.scrollIntoView({ block: 'nearest' });
}

/** Ground-truth mode (§8.4, §13.1): tap beats, paste chart, bar loop at 70%, chord preview, unsure spots, review, save. */
export function GtBar() {
  const t = useT();
  const gt = useApp((s) => s.gt);
  const analysis = useApp((s) => s.analysis);
  const tapMode = useApp((s) => s.tapMode);
  const chartOpen = useApp((s) => s.chartOpen);
  const sel = useApp((s) => s.sel);
  useApp((s) => s.loop); // re-render when the bar loop changes
  const [busy, setBusy] = useState(false);
  const phone = useMedia(PHONE_QUERY);
  if (!analysis) return null;
  if (phone && tapMode) {
    return (
      <div className="gt-bar tapping">
        <TapPanel />
      </div>
    );
  }
  const key = analysis.source.key;
  const unsure = unsureChords(analysis.chords);
  const at = sel ? unsure.findIndex((c) => sel.t0 >= c.start - 1e-3 && sel.t0 < c.end) : -1;
  const status = gt?.status ?? analysis.gt?.status ?? 'none';

  const review = async (on: boolean) => {
    setBusy(true);
    try {
      const r = await api.gtReview(key, on);
      useApp.setState({ gt: r.gt });
      if (r.files) showToast(t('gtSaved', { files: Object.values(r.files).join(', ') }));
    } catch (e) {
      showToast(e instanceof Error ? e.message : String(e), 'error');
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    setBusy(true);
    try {
      const r = await api.gtExport(key);
      useApp.setState({ gt: r.gt });
      showToast([t('gtSaved', { files: Object.values(r.files).join(', ') }), ...r.warnings].join(' · '));
    } catch (e) {
      showToast(e instanceof Error ? e.message : String(e), 'error');
    } finally {
      setBusy(false);
    }
  };

  const step = (dir: 1 | -1) => {
    if (unsure.length === 0) return;
    const i = at < 0 ? (dir > 0 ? unsure.findIndex((c) => c.start > engine.time) : -1) : at + dir;
    const j = ((i < 0 ? (dir > 0 ? 0 : unsure.length - 1) : i) + unsure.length) % unsure.length;
    goTo(unsure[j]);
  };

  return (
    <div className="gt-bar">
      <div className="eb-row">
        <span className={`chip gt-status ${status}`} title={t('gtTitle')}>
          {t('gtMode')}: {t(`gt_${status}`)}
        </span>
        <button className={tapMode ? 'on' : ''} onMouseDown={noFocus} onClick={() => (tapMode ? void stopTapping() : startTapping())}>
          {t('tapBeats')} ({gt?.beats.length ?? 0})
        </button>
        <button onMouseDown={noFocus} onClick={() => useApp.setState({ chartOpen: true })}>
          {t('pasteChart')}
        </button>
        <button className={barLoopOn() ? 'on' : ''} onMouseDown={noFocus} onClick={toggleBarLoop} title={t('barLoopTitle')}>
          ⟳ {t('barLoop')}
        </button>
        <button onMouseDown={noFocus} onClick={playChordAt} title={t('chordPreview')}>
          ♪ {t('chordPreview')}
        </button>
        {unsure.length > 0 && (
          <span className="gt-unsure">
            <button className="icon-btn" onMouseDown={noFocus} onClick={() => step(-1)} aria-label="previous unsure">
              ◀
            </button>
            <span>
              {t('unsure')} {at >= 0 ? at + 1 : '–'}/{unsure.length}
            </span>
            <button className="icon-btn" onMouseDown={noFocus} onClick={() => step(1)} aria-label="next unsure">
              ▶
            </button>
          </span>
        )}
        {gt?.chart?.stale && (
          <span className="gt-stale">
            {t('staleChart')}{' '}
            <button onMouseDown={noFocus} onClick={() => useApp.setState({ chartOpen: true })}>
              {t('realign')}
            </button>
          </span>
        )}
        <span className="sh-spacer" />
        <label className="toggle">
          <input type="checkbox" checked={!!gt?.reviewed} disabled={busy} onChange={(e) => void review(e.target.checked)} />
          <span>{t('reviewed')}</span>
        </label>
        <button className="primary" disabled={busy} onMouseDown={noFocus} onClick={() => void save()}>
          {t('saveGt')}
        </button>
      </div>
      {tapMode && <TapPanel />}
      {chartOpen && <ChartDialog />}
    </div>
  );
}

