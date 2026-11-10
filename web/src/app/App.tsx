import { useEffect, useState } from 'react';
import { useApp } from './store';
import { PHONE_QUERY, useMedia, useThemeAttr } from './hooks';
import { analyzeFile } from './jobs';
import { JobPanel, TopBar } from './TopBar';
import { SongPage } from './SongPage';
import { LibraryPage } from '../library/LibraryPage';
import { ConnectPage } from '../remote/ConnectPage';
import { PhonePage } from '../remote/PhonePage';
import { useT } from '../i18n';

function Toast() {
  const toast = useApp((s) => s.toast);
  if (!toast) return null;
  return (
    <div className={`toast ${toast.kind}`} role="status" onClick={() => useApp.setState({ toast: null })}>
      {toast.msg}
    </div>
  );
}

/** Drop an audio file anywhere to analyse it. */
function DropOverlay() {
  const t = useT();
  const [over, setOver] = useState(false);
  useEffect(() => {
    let depth = 0;
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes('Files');
    const enter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth++;
      setOver(true);
    };
    const leave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setOver(false);
    };
    const overFn = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    };
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setOver(false);
      const f = e.dataTransfer?.files?.[0];
      if (f) void analyzeFile(f);
    };
    window.addEventListener('dragenter', enter);
    window.addEventListener('dragleave', leave);
    window.addEventListener('dragover', overFn);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragenter', enter);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('dragover', overFn);
      window.removeEventListener('drop', drop);
    };
  }, []);
  if (!over) return null;
  return (
    <div className="drop-overlay">
      <div>{t('dropHere')}</div>
    </div>
  );
}

export function App() {
  useThemeAttr();
  const route = useApp((s) => s.route);
  const phone = useMedia(PHONE_QUERY);
  const onSong = route.page === 'song';
  return (
    <div className={`app${phone ? ' phone' : ''}`}>
      {!(phone && onSong) && <TopBar />}
      <JobPanel />
      <main className="page">
        {route.page === 'song' && <SongPage key={route.key} songKey={route.key} />}
        {route.page === 'library' && <LibraryPage />}
        {route.page === 'connect' && <ConnectPage />}
        {route.page === 'phone' && <PhonePage />}
      </main>
      <Toast />
      <DropOverlay />
    </div>
  );
}
