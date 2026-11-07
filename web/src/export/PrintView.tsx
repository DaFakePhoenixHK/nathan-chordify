// The print view (2026-11-04): the chord sheet the server built (export/sheet.py) laid out for paper: the
// header, the bar chart, and the chords over the words. Follows the current Transpose / Level / Slash.
// "Save as PDF" is the browser's print dialog (no PDF library in the venv).
import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { api } from '../api/client';
import type { SheetExport } from '../api/types';
import { useApp } from '../app/store';
import { useT, type StringKey } from '../i18n';
import { sourceKey } from '../sheets/logic';
import { currentExportParams } from './ExportMenu';
import { parseChordPro } from './logic';

const close = () => useApp.setState({ printOpen: false, printAuto: false });

function Lyrics({ text }: { text: string }) {
  const lines = useMemo(() => parseChordPro(text), [text]);
  return (
    <div className="pv-lyrics">
      {lines.map((ln, i) => {
        if (ln.kind === 'blank') return <div key={i} className="pv-blank" />;
        if (ln.kind === 'label')
          return (
            <h3 key={i} className="pv-section">
              {ln.text}
            </h3>
          );
        return (
          <div key={i} className="pv-line">
            {ln.tokens.map((tk, j) => (
              <span key={j} className="pv-tok">
                <span className="pv-chord">{tk.chord ?? ' '}</span>
                <span className="pv-word">{tk.text || ' '}</span>
              </span>
            ))}
          </div>
        );
      })}
    </div>
  );
}

export function PrintView() {
  const t = useT();
  const open = useApp((s) => s.printOpen);
  const auto = useApp((s) => s.printAuto);
  const route = useApp((s) => s.route);
  const transpose = useApp((s) => s.transpose);
  const playIn = useApp((s) => s.playIn);
  const level = useApp((s) => s.level);
  const slash = useApp((s) => s.slash);
  const acc = useApp((s) => s.acc);
  const sheets = useApp((s) => s.sheets);
  const sheetSel = useApp((s) => s.sheetSel);
  const rev = useApp((s) => s.rev);
  const [data, setData] = useState<SheetExport | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Every setting that changes the sheet is a dependency; the params come from the store at that moment.
  useEffect(() => {
    if (!open || route.page !== 'song') return;
    const key = route.key;
    const p = currentExportParams('json');
    if (!p) return;
    let cancelled = false;
    api
      .exportSheet(key, p)
      .then((ex) => {
        if (cancelled) return;
        setData(ex);
        setError(null);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
    // oxlint-disable-next-line react/exhaustive-effect-dependencies
  }, [open, route, transpose, playIn, level, slash, acc, sheets, sheetSel, rev]);

  useEffect(() => {
    if (!open) return;
    document.body.classList.add('printing');
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.classList.remove('printing');
    };
  }, [open]);

  useEffect(() => {
    if (!open || !auto || !data) return;
    const id = window.setTimeout(() => {
      useApp.setState({ printAuto: false });
      window.print();
    }, 350);
    return () => window.clearTimeout(id);
  }, [open, auto, data]);

  if (!open) return null;
  const src = data?.lyrics_source ? sourceKey(data.lyrics_source) : null;
  const srcName = data?.lyrics_source ? (src ? t(src) : data.lyrics_source) : '';
  return createPortal(
    <div className="print-root" role="dialog" aria-label={t('printTitle')}>
      <div className="print-tools no-print">
        <b>{t('printTitle')}</b>
        <span className="pv-hint">{t('savePdfHint')}</span>
        <span className="sh-spacer" />
        <button className="pill primary" onClick={() => window.print()} disabled={!data}>
          {t('print')}
        </button>
        <button className="pill" onClick={close}>
          {t('close')}
        </button>
      </div>
      {error && <div className="print-page pv-error">{`${t('exportFailed')}: ${error}`}</div>}
      {!data && !error && <div className="print-page muted">{t('printLoading')}</div>}
      {data && (
        <article className="print-page">
          <header className="pv-head">
            <h1>{data.title}</h1>
            {data.artist && <div className="pv-artist">{data.artist}</div>}
            <div className="pv-meta">
              <span>
                {t('keyLabel')}: <b>{data.key}</b>
                {data.transpose % 12 !== 0 && <small> ({t('originalKey', { key: data.key_original })})</small>}
              </span>
              <span>
                ♩ <b>{data.bpm}</b> {t('bpm')}
              </span>
              <span>{data.meter}</span>
              <span>{t(`level_${data.level}` as StringKey)}</span>
              {!data.slash && <span>{t('noSlashShort')}</span>}
            </div>
            {data.key_changes.map((kc) => (
              <div key={kc} className="pv-note">
                {t('printKeyChange', { where: kc })}
              </div>
            ))}
            {data.intro.length > 0 && <div className="pv-note">{t('printIntro', { chords: data.intro.join(' ') })}</div>}
          </header>
          <section>
            <h2>{t('printChartTitle')}</h2>
            <div className="pv-legend">{t('printLegend')}</div>
            <pre className="pv-chart">{data.chart_text}</pre>
          </section>
          {data.lyrics_chordpro && (
            <section className="pv-lyrics-sec">
              <h2>{t('printLyricsTitle')}</h2>
              <div className="pv-legend">{data.lyrics_ours ? t('printLyricsOurs', { source: srcName }) : t('printLyricsSheet', { source: srcName })}</div>
              <Lyrics text={data.lyrics_chordpro} />
            </section>
          )}
          <footer className="pv-foot">{t('printMadeBy', { date: data.made })}</footer>
        </article>
      )}
    </div>,
    document.body,
  );
}
