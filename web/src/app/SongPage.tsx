import { useEffect, useRef } from 'react';
import { api } from '../api/client';
import { setMockAudioFile } from '../api/mock';
import { loadSongPrefs, useApp } from './store';
import { PHONE_QUERY, WIDE_QUERY, useMedia, useRenderCount } from './hooks';
import { useShortcuts } from './shortcuts';
import { SongHeader } from './SongHeader';
import { useT } from '../i18n';
import { engine } from '../player/engine';
import { setTimeline } from '../player/sync';
import { Transport } from '../player/Transport';
import { ChordGrid } from '../grid/ChordGrid';
import { ChordMenu } from '../grid/ChordMenu';
import { getTimeline } from '../grid/model';
import { PianoPanel } from '../diagrams/PianoPanel';
import { ControlsCompact, ControlsDock, SettingsSheet } from '../controls/Controls';
import { EditBar } from '../editor/EditBar';
import { ChordEditor } from '../editor/ChordEditor';
import { loadSheets, SheetBanner, SheetPanel } from '../sheets/SheetPanel';
import { applyRememberedMix } from '../player/mixActions';
import { PrintView } from '../export/PrintView';

function MockBar() {
  const t = useT();
  const audioSrc = useApp((s) => s.audioSrc);
  const input = useRef<HTMLInputElement>(null);
  return (
    <div className={`mock-bar${audioSrc ? '' : ' warn'}`}>
      <span>{t('mockBanner')}</span>
      {!audioSrc && <span>{t('mockNoAudio')}</span>}
      <button className="pill" onClick={() => input.current?.click()}>
        {t('mockPickAudio')}
      </button>
      <input
        ref={input}
        type="file"
        accept="audio/*,video/*,.m4a,.mp3,.wav,.flac,.ogg,.webm"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) engine.load(setMockAudioFile(f));
          e.target.value = '';
        }}
      />
    </div>
  );
}

export function SongPage({ songKey }: { songKey: string }) {
  useRenderCount('SongPage');
  const t = useT();
  const analysis = useApp((s) => s.analysis);
  const loading = useApp((s) => s.loading);
  const loadError = useApp((s) => s.loadError);
  const songRev = useApp((s) => s.songRev);
  const detail = useApp((s) => s.detail);
  const mobileTab = useApp((s) => s.mobileTab);
  const gtMode = useApp((s) => s.gtMode);
  const phone = useMedia(PHONE_QUERY);
  const wide = useMedia(WIDE_QUERY);
  useShortcuts();

  // songRev is an intentional trigger: bumping it reloads the same song (re-scan, finished re-analysis).
  useEffect(() => {
    let cancelled = false;
    const keep = useApp.getState().analysis?.source.key === songKey; // re-load of the same song: keep position
    const pos = keep ? engine.time : 0;
    useApp.setState({ loading: true, loadError: null, menu: null, ...(keep ? {} : { analysis: null, display: null, nowSeg: -1, loop: { a: null, b: null } }) });
    if (!keep) engine.load(null);
    (async () => {
      const prefs = await loadSongPrefs(songKey);
      const a = await api.getAnalysis(songKey);
      const d = await api.getDisplay(songKey, a);
      const src = await api.audioUrl(songKey, a);
      if (cancelled) return;
      useApp.setState({ analysis: a, display: d, loading: false, rev: a.edits?.rev ?? 0, audioBase: src, ...(keep ? {} : prefs) });
      // a re-load of the same song keeps its current player file (the Sound + Audio choice already applied)
      if (!keep || !useApp.getState().audioSrc) engine.load(src);
      if (keep && pos > 0) engine.seek(pos);
      if (!keep && src) applyRememberedMix(songKey); // the Sound chosen for this song last time
      void loadSheets(songKey); // online chord sheets: state, running search, or auto-lookup
    })().catch((e: unknown) => {
      if (!cancelled) useApp.setState({ loading: false, loadError: e instanceof Error ? e.message : String(e) });
    });
    return () => {
      cancelled = true;
    };
    // oxlint-disable-next-line react/exhaustive-effect-dependencies
  }, [songKey, songRev]);

  useEffect(() => () => engine.pause(), []);

  useEffect(() => {
    if (analysis) setTimeline(getTimeline(analysis, detail, gtMode));
    else setTimeline([]);
  }, [analysis, detail, gtMode]);

  useEffect(() => {
    if (analysis) document.title = `${analysis.source.title} · Nathan Chordify`;
    return () => {
      document.title = 'Nathan Chordify';
    };
  }, [analysis]);

  if (loadError && !analysis) {
    return (
      <div className="center-msg">
        <p>{t('loadFailed')}</p>
        <p className="muted">{loadError}</p>
        <button className="pill" onClick={() => useApp.setState((s) => ({ songRev: s.songRev + 1 }))}>
          {t('retry')}
        </button>
      </div>
    );
  }
  if (!analysis) return <div className="center-msg muted">{loading ? t('loadingSong') : ''}</div>;

  const diagramsTab = phone && mobileTab === 'diagrams';
  return (
    <div className={`song${phone ? ' is-phone' : ''}`}>
      <SongHeader />
      {api.mock && <MockBar />}
      <EditBar />
      <SheetPanel />
      <div className="song-main">
        <div className={`grid-col${diagramsTab ? ' hidden' : ''}`}>
          <SheetBanner />
          <ChordGrid />
        </div>
        {diagramsTab && <PianoPanel mode="full" />}
        {wide && <PianoPanel mode="side" />}
      </div>
      {!wide && !diagramsTab && <PianoPanel mode="mini" />}
      <ChordEditor />
      <div className="dock">
        <Transport />
        {phone ? <ControlsCompact /> : <ControlsDock />}
      </div>
      <ChordMenu />
      <SettingsSheet />
      <PrintView />
    </div>
  );
}
