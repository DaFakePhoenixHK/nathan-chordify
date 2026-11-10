// "Export" (2026-11-04): Print / Save as PDF (the print view), Download text (bar chart), Download ChordPro
// (chords over the words; off when the song has no words yet), Copy to clipboard. The sheet follows the
// current Transpose / Play-in, Level, Slash toggle and accidentals; the server spells every chord.
import { useEffect, useRef, useState, type MouseEvent as RMouseEvent } from 'react';
import { api } from '../api/client';
import { showToast, useApp } from '../app/store';
import { shownShift } from '../grid/labels';
import { useT } from '../i18n';
import { copyLater, IS_NATIVE, shareOrDownload } from '../app/platform';
import { getLink } from '../remote/connection';
import { downloadText, exportSheetUrl, hasLyrics, pickLyricsSheet, sheetFileName, type SheetExportParams, type SheetFormat } from './logic';

const noFocus = (e: RMouseEvent) => e.preventDefault();
const close = () => useApp.setState({ exportOpen: false });

/** The export parameters of the current view. */
export function currentExportParams(format: SheetFormat): SheetExportParams | null {
  const s = useApp.getState();
  if (s.route.page !== 'song' || !s.analysis) return null;
  return {
    format,
    transpose: shownShift(s.transpose, s.playIn),
    level: s.level,
    slash: s.slash,
    lyrics: true,
    acc: s.acc,
    sheet: pickLyricsSheet(s.sheets, s.sheetSel),
  };
}

function songKey(): string | null {
  const r = useApp.getState().route;
  return r.page === 'song' ? r.key : null;
}

async function download(format: 'txt' | 'chordpro'): Promise<void> {
  const key = songKey();
  const p = currentExportParams(format);
  if (!key || !p) return;
  const url = getLink() ? null : api.exportSheetUrl(key, p); // a phone can't save a link: it gets the file itself
  if (url) {
    const a = document.createElement('a');
    a.href = url;
    a.download = '';
    document.body.appendChild(a);
    a.click();
    a.remove();
    return;
  }
  // the demo: build the file here
  const ex = await api.exportSheet(key, { ...p, format: 'json' });
  const text = format === 'chordpro' ? ex.chordpro : ex.text;
  if (!text) throw new Error('no_lyrics');
  await shareOrDownload(sheetFileName(ex.title, ex.key, format === 'chordpro' ? 'cho' : 'txt'), text, downloadText);
}

async function copy(): Promise<void> {
  const key = songKey();
  const p = currentExportParams('json');
  if (!key || !p) return;
  await copyLater(api.exportSheet(key, p).then((ex) => ex.text));
}

export function ExportButton({ phone }: { phone: boolean }) {
  const t = useT();
  const open = useApp((s) => s.exportOpen);
  const sheets = useApp((s) => s.sheets);
  const analysis = useApp((s) => s.analysis);
  const ref = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const lyrics = hasLyrics(sheets);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) useApp.setState({ exportOpen: false });
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') useApp.setState({ exportOpen: false });
    };
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  if (!analysis) return null;
  const run = (what: string, fn: () => Promise<void>) => {
    setBusy(what);
    fn()
      .then(() => {
        if (what === 'copy') showToast(t('copied'));
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        showToast(`${what === 'copy' ? t('copyFailed') : t('exportFailed')}: ${msg === 'no_lyrics' ? t('noLyricsForExport') : msg}`, 'error');
      })
      .finally(() => setBusy(null));
    close();
  };
  const openPrint = (auto: boolean, pdf: boolean) => {
    close();
    useApp.setState({ printOpen: true, printAuto: auto });
    if (pdf) showToast(t('savePdfHint'));
  };

  return (
    <div className="export-wrap" ref={ref}>
      <button
        className={`export-btn${open ? ' on' : ''}`}
        onMouseDown={noFocus}
        onClick={() => useApp.setState({ exportOpen: !open })}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t('export')}
        title={t('exportTitle')}
      >
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path d="M12 3v11m0 0l-4-4m4 4l4-4" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M5 15v4h14v-4" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" />
        </svg>
        {!phone && <span>{t('export')}</span>}
        {busy && <span className="sheets-spin" aria-hidden="true" />}
      </button>
      {open && (
        <div className="export-menu" role="menu" aria-label={t('export')}>
          <button role="menuitem" onMouseDown={noFocus} onClick={() => openPrint(true, false)}>
            {IS_NATIVE ? t('printTitle') : `🖨 ${t('print')}`}
          </button>
          {!IS_NATIVE && (
            <button role="menuitem" onMouseDown={noFocus} onClick={() => openPrint(true, true)} title={t('savePdfHint')}>
              📄 {t('savePdf')}
            </button>
          )}
          <button role="menuitem" onMouseDown={noFocus} onClick={() => run('txt', () => download('txt'))}>
            ⬇ {t('downloadText')}
          </button>
          <button
            role="menuitem"
            onMouseDown={noFocus}
            disabled={!lyrics}
            title={lyrics ? undefined : t('noLyricsForExport')}
            onClick={() => run('chordpro', () => download('chordpro'))}
          >
            ⬇ {t('downloadChordpro')}
          </button>
          {!lyrics && <div className="export-note">{t('noLyricsForExport')}</div>}
          <button role="menuitem" onMouseDown={noFocus} onClick={() => run('copy', copy)}>
            ⎘ {t('copyText')}
          </button>
        </div>
      )}
    </div>
  );
}

export { exportSheetUrl };
