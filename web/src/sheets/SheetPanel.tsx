// "Find the sheet online and refine the chords" (2026-11-03): the button, the banner, the panel and its actions.
// Sheets are a reference, not the truth: "Use it" = refine (the recording decides); "as written" is secondary.
import { useEffect, useMemo, useState, type MouseEvent as RMouseEvent } from 'react';
import { api, ApiError } from '../api/client';
import type { Analysis, ServerSettings, SheetApplyMode, SheetResult, SheetsState } from '../api/types';
import { showToast, useApp } from '../app/store';
import { PHONE_QUERY, useLabelCtx, useMedia } from '../app/hooks';
import { tNow, useT, type StringKey } from '../i18n';
import { engine } from '../player/engine';
import { formatTime } from '../player/sync';
import { applyResult, refreshHistory } from '../editor/ops';
import { buildSquares, buildTimeline, segAt } from '../grid/model';
import { chordLabel, LEVEL_INDEX } from '../grid/labels';
import {
  bannerOffer,
  countDiffRuns,
  dismissBanner,
  isBannerDismissed,
  keyShift,
  looksLikeUrl,
  pollSheets,
  sheetMarks,
  sourceKey,
  termsText,
  verdictKey,
} from './logic';

const noFocus = (e: RMouseEvent) => e.preventDefault();

function songKey(): string | null {
  const r = useApp.getState().route;
  return r.page === 'song' ? r.key : null;
}

// ---------------------------------------------------------------- the flow (search, poll, apply, undo)

let pollToken = 0;

const EMPTY: SheetsState = { status: 'idle', error: null, query: { title: '', artist: '', terms: [] }, results: [], applied: null, ts: null };

function setSheets(key: string, st: SheetsState): void {
  if (songKey() !== key) return;
  const cur = useApp.getState();
  // When a good sheet turns up and nothing is applied yet, show its chords on the grid (the banner offers it).
  const best = st.status === 'done' && !st.applied ? st.results[0] : undefined;
  const autoSel = cur.sheetSel == null && !cur.sheetBannerHidden && best?.verdict === 'good' ? best.id : cur.sheetSel;
  const keepSel = autoSel && st.results.some((r) => r.id === autoSel) ? autoSel : null;
  useApp.setState({ sheets: st, sheetSel: st.applied ? null : keepSel });
}

/** Plain words for the API's error codes. */
export function sheetErrorText(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.code === 'sheets_unreachable' || e.status === 502 || e.status === 0) return tNow('sheetsErrorUnreachable');
    if (e.code === 'not_analysed' || e.status === 409) return tNow('sheetsErrorNotAnalysed');
    if (e.code === 'bad_chart' || e.status === 400) return tNow('sheetsErrorBadChart');
    if (e.code === 'no_sheet') return tNow('sheetsNone');
    return e.hint ? `${e.message} — ${e.hint}` : e.message;
  }
  return e instanceof Error ? e.message : String(e);
}

function failed(key: string, e: unknown): void {
  const code = e instanceof ApiError ? e.code : 'error';
  const cur = useApp.getState().sheets ?? EMPTY;
  setSheets(key, { ...cur, status: 'error', error: { code, msg: sheetErrorText(e) } });
}

async function followSearch(key: string): Promise<void> {
  const token = ++pollToken;
  const st = await pollSheets(() => api.sheets(key), {
    intervalMs: 1500,
    cancelled: () => token !== pollToken || songKey() !== key,
  }).catch((e: unknown) => {
    failed(key, e);
    return null;
  });
  if (st) setSheets(key, st);
}

export async function ensureSettings(): Promise<ServerSettings | null> {
  const cur = useApp.getState().serverSettings;
  if (cur) return cur;
  try {
    const s = await api.getSettings();
    useApp.setState({ serverSettings: s });
    return s;
  } catch {
    return null; // an older server: behave as "on"
  }
}

export async function setAutoLookup(on: boolean): Promise<void> {
  try {
    const s = await api.patchSettings({ sheets_auto_lookup: on });
    useApp.setState({ serverSettings: s });
  } catch (e) {
    showToast(sheetErrorText(e), 'error');
  }
}

/** Opening a song: load the sheet state, follow a running search, start one when the setting is on. */
export async function loadSheets(key: string): Promise<void> {
  useApp.setState({ sheetBannerHidden: isBannerDismissed(localStorage, key) });
  let st: SheetsState;
  try {
    st = await api.sheets(key);
  } catch (e) {
    // 404 / 405 / 501: a server without the feature; 409: not analysed yet. Nothing to show, no toast.
    if (e instanceof ApiError && [404, 405, 409, 501].includes(e.status)) return;
    failed(key, e);
    return;
  }
  setSheets(key, st);
  if (st.status === 'searching') return followSearch(key);
  if (st.status === 'idle') {
    const s = await ensureSettings();
    if ((s?.sheets_auto_lookup ?? true) && songKey() === key) await startSearch(key);
  }
}

export async function startSearch(key: string, query?: string): Promise<void> {
  pollToken++;
  try {
    const st = await api.sheetsSearch(key, query?.trim() ? { query: query.trim(), force: true } : {});
    useApp.setState({ sheetSel: null });
    setSheets(key, st);
    if (st.status === 'searching') await followSearch(key);
  } catch (e) {
    failed(key, e);
  }
}

/** Paste box: a link or chart text. Resolves with an error text to show, or null when added. */
export async function addSheet(key: string, text: string): Promise<string | null> {
  try {
    const st = await api.sheetsAdd(key, looksLikeUrl(text) ? { url: text.trim() } : { text });
    setSheets(key, st);
    const added = st.results.find((r) => r.source === 'pasted');
    if (added) useApp.setState({ sheetSel: added.id });
    return null;
  } catch (e) {
    return sheetErrorText(e);
  }
}

export async function applySheet(id: string, mode: SheetApplyMode): Promise<void> {
  const key = songKey();
  const r = useApp.getState().sheets?.results.find((x) => x.id === id);
  if (!key || !r) return;
  try {
    const res = await api.sheetsApply(key, id, mode);
    applyResult(key, res);
    useApp.setState((s) => ({
      sheets: s.sheets ? { ...s.sheets, applied: res.applied } : s.sheets,
      sheetSel: null,
      sheetFlags: res.summary.flagged,
      sheetSummary: { n_changed: res.summary.n_changed, n_kept: res.summary.n_kept, source: r.source, mode },
      sheetBannerHidden: true,
    }));
    void refreshHistory(key);
    const src = sourceKey(r.source);
    showToast(`${tNow(`applied_${mode}`, { source: src ? tNow(src) : r.source })} · ${tNow('appliedChanged', { n: res.summary.n_changed })}`);
  } catch (e) {
    showToast(sheetErrorText(e), 'error');
  }
}

export async function undoSheet(): Promise<void> {
  const key = songKey();
  if (!key) return;
  try {
    const res = await api.sheetsUndo(key);
    applyResult(key, res);
    useApp.setState((s) => ({ sheets: s.sheets ? { ...s.sheets, applied: null } : s.sheets, sheetFlags: [], sheetSummary: null }));
    void refreshHistory(key);
    showToast(tNow('appliedUndone'));
  } catch (e) {
    showToast(sheetErrorText(e), 'error');
  }
}

/** After an edit-toolbar Undo the server may have dropped the applied sheet: re-read and clear our marks. */
async function recheckApplied(key: string): Promise<void> {
  try {
    const st = await api.sheets(key);
    if (songKey() !== key || st.applied) return;
    useApp.setState({ sheets: st, sheetFlags: [], sheetSummary: null });
  } catch {
    /* the next change retries */
  }
}

export function dismissSheetBanner(): void {
  const key = songKey();
  if (key) dismissBanner(localStorage, key);
  useApp.setState({ sheetBannerHidden: true, sheetSel: null });
}

export function toggleSheetsPanel(): void {
  useApp.setState((s) => ({ sheetsOpen: !s.sheetsOpen }));
}

const closePanel = () => useApp.setState({ sheetsOpen: false });

// ---------------------------------------------------------------- the "Find sheet" button (next to Edit)

function SheetIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
      <path d="M6 3h9l4 4v14H6z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      <path d="M15 3v4h4M9 11h6M9 15h6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

export function SheetButton({ phone }: { phone: boolean }) {
  const t = useT();
  const open = useApp((s) => s.sheetsOpen);
  const n = useApp((s) => s.sheets?.results.length ?? 0);
  const searching = useApp((s) => s.sheets?.status === 'searching');
  const applied = useApp((s) => !!s.sheets?.applied);
  const found = n === 1 ? t('sheetsFound', { n }) : t('sheetsFoundMany', { n });
  return (
    <button
      className={`sheets-btn${open ? ' on' : ''}`}
      onMouseDown={noFocus}
      onClick={toggleSheetsPanel}
      aria-pressed={open}
      aria-label={t('findSheet')}
      title={n > 0 ? `${t('findSheetTitle')} · ${found}` : t('findSheetTitle')}
    >
      <SheetIcon />
      {!phone && <span>{t('findSheet')}</span>}
      {searching && <span className="sheets-spin" aria-hidden="true" />}
      {!searching && n > 0 && (
        <span className={`sheets-badge${applied ? ' applied' : ''}`} aria-label={found}>
          {applied ? '✓' : n}
        </span>
      )}
    </button>
  );
}

// ---------------------------------------------------------------- the banner above the grid

export function SheetBanner() {
  const t = useT();
  const sheets = useApp((s) => s.sheets);
  const hidden = useApp((s) => s.sheetBannerHidden);
  const open = useApp((s) => s.sheetsOpen);
  const r = bannerOffer(sheets, hidden);
  if (!r || open) return null;
  const src = sourceKey(r.source);
  return (
    <div className="sheet-banner" role="status">
      <span className="sb-text">{t('sheetBanner', { source: src ? t(src) : r.source })}</span>
      <span className="sb-actions">
        <button className="primary" onMouseDown={noFocus} onClick={() => void applySheet(r.id, 'refine')} title={t('useItHelp')}>
          {t('useIt')}
        </button>
        <button onMouseDown={noFocus} onClick={() => useApp.setState({ sheetsOpen: true, sheetSel: r.id })}>
          {t('showChart')}
        </button>
        <button onMouseDown={noFocus} onClick={dismissSheetBanner}>
          {t('notNow')}
        </button>
      </span>
    </div>
  );
}

// ---------------------------------------------------------------- the settings toggle

export function SheetsAutoToggle() {
  const t = useT();
  const s = useApp((x) => x.serverSettings);
  useEffect(() => {
    void ensureSettings();
  }, []);
  return (
    <label className="toggle" title={t('sheetsAutoLookupHelp')}>
      <input type="checkbox" checked={s?.sheets_auto_lookup ?? true} disabled={!s} onChange={(e) => void setAutoLookup(e.target.checked)} />
      <span>{t('sheetsAutoLookup')}</span>
    </label>
  );
}

// ---------------------------------------------------------------- result cards

/** How many stretches of this sheet disagree with our chords at the grid's label level (null when transposed). */
function useDiffCount(r: SheetResult, analysis: Analysis): number | null {
  const ctx = useLabelCtx();
  return useMemo(() => {
    if (ctx.shift !== 0) return null;
    const squares = buildSquares(analysis);
    const segs = buildTimeline(analysis, 0);
    const ours = (t: number) => {
      const k = segAt(segs, t);
      return k < 0 ? null : chordLabel(segs[k].chord, { ...ctx, notation: 'letters' });
    };
    return countDiffRuns(sheetMarks(r.aligned, squares, ours, LEVEL_INDEX[ctx.level], ctx.slash, true));
  }, [r, analysis, ctx]);
}

function siteName(s: string, t: (k: StringKey) => string): string {
  const k = sourceKey(s);
  return k ? t(k) : s;
}

function SheetCard({ r, analysis, selected, isApplied, anyApplied }: { r: SheetResult; analysis: Analysis; selected: boolean; isApplied: boolean; anyApplied: boolean }) {
  const t = useT();
  const [chartOpen, setChartOpen] = useState(false);
  const [sure, setSure] = useState(false);
  const [busy, setBusy] = useState(false);
  const diffs = useDiffCount(r, analysis);
  const shift = keyShift(r);
  const different = r.verdict === 'different';
  const apply = async (mode: SheetApplyMode) => {
    setBusy(true);
    try {
      await applySheet(r.id, mode);
    } finally {
      setBusy(false);
    }
  };
  const toggleSel = () => useApp.setState({ sheetSel: selected ? null : r.id });
  const srcKey = sourceKey(r.source);
  return (
    <div className={`sp-card ${r.verdict}${selected ? ' sel' : ''}${isApplied ? ' applied' : ''}`}>
      <div className="sp-head">
        <span className={`src-badge ${r.source}`}>{srcKey ? t(srcKey) : r.source}</span>
        <b className="sp-title">{r.title}</b>
        {r.artist && <span className="sp-artist">{r.artist}</span>}
        {r.year != null && <span className="muted">{r.year}</span>}
        {r.video_match && <span className="sp-tag video">▶ {t('sameVideo')}</span>}
        {isApplied && <span className="sp-tag applied">✓ {t('inUse')}</span>}
      </div>
      <div className="sp-meta muted">
        {shift ? (
          <span>{t('writtenIn', { from: shift.from, to: shift.to })}</span>
        ) : r.key_written ? (
          <span>
            {t('keyLabel')} {r.key_written}
          </span>
        ) : null}
        {r.meter && <span>{r.meter}</span>}
        <span>{t('chordsN', { n: r.n_chords })}</span>
        {r.also && r.also.length > 0 && <span>{t('alsoOn', { sites: r.also.map((s) => siteName(s, t)).join(', ') })}</span>}
      </div>
      <div className={`sp-verdict ${r.verdict}`}>
        <span className="sp-dot" aria-hidden="true" />
        <span>{t(verdictKey(r.verdict))}</span>
        {diffs != null && <span className="muted"> · {diffs > 0 ? t('sheetDisagreeN', { n: diffs }) : t('sheetAgree')}</span>}
      </div>
      {r.warnings.map((w) => (
        <div key={w} className="sp-warn">
          ⚠ {w}
        </div>
      ))}
      <div className="sp-actions">
        <button className="primary" disabled={busy || anyApplied} onMouseDown={noFocus} onClick={() => void apply('refine')} title={t('useItHelp')}>
          {t('useIt')}
        </button>
        <button
          className="sp-aswritten"
          disabled={busy || anyApplied || (different && !sure)}
          onMouseDown={noFocus}
          onClick={() => void apply('sheet')}
          title={different && !sure ? t('useAsWrittenNote') : undefined}
        >
          {t('useAsWritten')}
        </button>
        {!anyApplied && (
          <button className={selected ? 'on' : ''} onMouseDown={noFocus} onClick={toggleSel} aria-pressed={selected}>
            {selected ? t('previewRowOff') : t('previewRow')}
          </button>
        )}
        <button onMouseDown={noFocus} onClick={() => setChartOpen(!chartOpen)} aria-expanded={chartOpen}>
          {chartOpen ? t('hideChart') : t('showChart')}
        </button>
        {r.url && (
          <a className="sp-link" href={r.url} target="_blank" rel="noopener noreferrer">
            {t('openSource')} ↗
          </a>
        )}
      </div>
      <div className="sp-help muted">{t('useItHelp')}</div>
      {different && !anyApplied && (
        <label className="toggle sp-sure">
          <input type="checkbox" checked={sure} onChange={(e) => setSure(e.target.checked)} />
          <span>
            {t('useAsWrittenNote')} <b>{t('useAnyway')}</b>
          </span>
        </label>
      )}
      {chartOpen && <pre className="sp-chart">{r.chart_text}</pre>}
    </div>
  );
}

// ---------------------------------------------------------------- applied note + flagged spots

function seekFlag(t0: number): void {
  engine.seek(Math.max(0, t0 - 0.05));
}

function AppliedNote() {
  const t = useT();
  const summary = useApp((s) => s.sheetSummary);
  const flags = useApp((s) => s.sheetFlags);
  const [busy, setBusy] = useState(false);
  if (!summary) return null;
  const src = sourceKey(summary.source);
  const undo = async () => {
    setBusy(true);
    try {
      await undoSheet();
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="sp-applied">
      <div className="sp-applied-row">
        <span>
          ✓ {t(`applied_${summary.mode}`, { source: src ? t(src) : summary.source })} · {t('appliedChanged', { n: summary.n_changed })} ·{' '}
          {t('appliedKept', { n: summary.n_kept })}
        </span>
        <button disabled={busy} onMouseDown={noFocus} onClick={() => void undo()}>
          ↶ {t('appliedUndo')}
        </button>
      </div>
      {flags.length > 0 && (
        <div className="sp-flags">
          <span className="muted">{t('flaggedTitle')}</span>
          {flags.map((f) => (
            <button key={`${f.t0}`} className="sp-flag" onMouseDown={noFocus} onClick={() => seekFlag(f.t0)}>
              <span className="sp-flag-t">{formatTime(f.t0, true)}</span> {t('flaggedSays', { detected: f.detected, sheet: f.sheet })}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- paste box

function PasteBox({ songKey: key }: { songKey: string }) {
  const t = useT();
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const submit = async () => {
    if (!text.trim()) return;
    setBusy(true);
    setMsg(null);
    const err = await addSheet(key, text);
    setBusy(false);
    if (err) setMsg(err);
    else setText('');
  };
  return (
    <div className="sp-paste">
      <label className="sp-paste-label" htmlFor="sp-paste-text">
        {t('pasteSheet')}
      </label>
      <div className="sp-paste-row">
        <textarea
          id="sp-paste-text"
          className="sp-paste-text"
          rows={text.includes('\n') ? 5 : 1}
          value={text}
          spellCheck={false}
          placeholder={'https://…  |  | B . . . | E . F# . |'}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            e.stopPropagation();
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) void submit();
          }}
        />
        <button className="primary" disabled={busy || !text.trim()} onMouseDown={noFocus} onClick={() => void submit()}>
          {t('addSheet')}
        </button>
      </div>
      <div className="muted small">{t('pasteSheetHelp')}</div>
      {msg && <div className="chart-msg">{msg}</div>}
    </div>
  );
}

// ---------------------------------------------------------------- the panel

function StatusLine({ st }: { st: SheetsState | null }) {
  const t = useT();
  if (!st || st.status === 'idle') return <div className="sp-status muted">{t('sheetsIdle')}</div>;
  if (st.status === 'searching')
    return (
      <div className="sp-status searching" role="status" aria-live="polite">
        <span className="sheets-spin" aria-hidden="true" /> {t('sheetsSearching')}
      </div>
    );
  if (st.status === 'error') return <div className="sp-status error">{st.error?.msg ?? t('errorGeneric')}</div>;
  if (st.results.length === 0) return <div className="sp-status">{t('sheetsNone')}</div>;
  const when = st.ts ? new Date(st.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : null;
  return when ? <div className="sp-status muted small">{t('sheetsLastSearch', { when })}</div> : null;
}

/** The sheet panel: a drawer under the toolbar (desktop) or a bottom sheet (phone). */
export function SheetPanel() {
  const t = useT();
  const open = useApp((s) => s.sheetsOpen);
  const sheets = useApp((s) => s.sheets);
  const sel = useApp((s) => s.sheetSel);
  const analysis = useApp((s) => s.analysis);
  const phone = useMedia(PHONE_QUERY);
  const [query, setQuery] = useState<string | null>(null);
  const key = analysis?.source.key ?? null;
  const applied = sheets?.applied ?? null;

  // The apply is one edit group: an Undo from the edit toolbar removes it on the server → re-read the state.
  useEffect(() => {
    if (!key) return;
    return useApp.subscribe((s, prev) => {
      if (s.rev !== prev.rev && s.sheets?.applied) void recheckApplied(key);
    });
  }, [key]);

  if (!open || !analysis || !key) return null;
  const terms = query ?? termsText(sheets);
  const searching = sheets?.status === 'searching';

  const body = (
    <>
      <div className="sheet-head">
        <b>{t('sheetsTitle')}</b>
        <button className="icon-btn" onClick={closePanel} aria-label={t('close')}>
          ×
        </button>
      </div>
      <p className="muted small sp-intro">{t('sheetsIntro')}</p>
      <form
        className="sp-query"
        onSubmit={(e) => {
          e.preventDefault();
          setQuery(null);
          void startSearch(key, terms);
        }}
      >
        <input
          className="ef-input sp-terms"
          value={terms}
          aria-label={t('sheetsQuery')}
          placeholder={t('sheetsQuery')}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.stopPropagation()}
        />
        <button type="submit" disabled={searching || !terms.trim()}>
          {t('searchAgain')}
        </button>
      </form>
      <StatusLine st={sheets} />
      {applied && <AppliedNote />}
      {sheets && sheets.results.length > 0 && (
        <div className="sp-cards">
          {sheets.results.map((r) => (
            <SheetCard key={r.id} r={r} analysis={analysis} selected={sel === r.id && !applied} isApplied={applied?.id === r.id} anyApplied={!!applied} />
          ))}
        </div>
      )}
      <PasteBox songKey={key} />
    </>
  );

  if (phone) {
    return (
      <>
        <div className="sheet-backdrop" onClick={closePanel} />
        <div className="sheet sheet-panel modal" role="dialog" aria-label={t('sheetsTitle')}>
          {body}
        </div>
      </>
    );
  }
  return (
    <div className="sheet-panel drawer" role="region" aria-label={t('sheetsTitle')}>
      {body}
    </div>
  );
}
