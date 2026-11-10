import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError } from '../api/client';
import type { TrackSummary } from '../api/types';
import { navigate, useApp } from '../app/store';
import { analyzeFile } from '../app/jobs';
import { formatTime } from '../player/sync';
import { needsPairing, useConnection } from '../remote/connection';
import { useT } from '../i18n';

const open = (key: string) => navigate({ page: 'song', key });

function RenameRow({ track, onDone }: { track: TrackSummary; onDone: (saved: boolean) => void }) {
  const t = useT();
  const [title, setTitle] = useState(track.title);
  const [artist, setArtist] = useState(track.artist ?? '');
  const [err, setErr] = useState<string | null>(null);
  return (
    <form
      className="rename"
      onSubmit={(e) => {
        e.preventDefault();
        api
          .renameTrack(track.key, { title: title.trim() || track.title, artist: artist.trim() || null })
          .then(() => onDone(true))
          .catch((x: unknown) => setErr(x instanceof Error ? x.message : String(x)));
      }}
    >
      <input value={title} onChange={(e) => setTitle(e.target.value)} aria-label={t('titleField')} placeholder={t('titleField')} autoFocus />
      <input value={artist} onChange={(e) => setArtist(e.target.value)} aria-label={t('artistField')} placeholder={t('artistField')} />
      <button type="submit" className="primary">
        {t('save')}
      </button>
      <button type="button" onClick={() => onDone(false)}>
        {t('cancel')}
      </button>
      {err && <span className="error-text">{err}</span>}
    </form>
  );
}

/** WP12 (M3): the analysed songs, open, rename, upload. */
export function LibraryPage() {
  const t = useT();
  const [tracks, setTracks] = useState<TrackSummary[] | null>(null);
  const [error, setError] = useState<{ offline: boolean; msg: string } | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const jobDone = useApp((s) => s.job?.status === 'done');
  const { link, laptop } = useConnection();
  const notPaired = needsPairing();
  const away = !!link && laptop === 'offline';
  const file = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    api
      .listTracks()
      .then((list) => {
        setTracks(list);
        setError(null);
      })
      .catch((e: unknown) => {
        setTracks([]);
        setError({
          offline: e instanceof ApiError && (e.status === 0 || e.status >= 500),
          msg: e instanceof Error ? e.message : String(e),
        });
      });
  }, []);

  useEffect(load, [load, link]);
  useEffect(() => {
    if (jobDone) load();
  }, [jobDone, load]);

  return (
    <div className="library">
      <div className="lib-head">
        <h1>{t('libraryTitle')}</h1>
        <button className="pill" onClick={() => file.current?.click()}>
          ⤒ {t('uploadFile')}
        </button>
        <input
          ref={file}
          type="file"
          hidden
          accept="audio/*,video/*,.m4a,.mp3,.wav,.flac,.ogg,.webm,.mp4,.mov"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void analyzeFile(f);
            e.target.value = '';
          }}
        />
      </div>
      {api.mock && <div className="mock-bar">{t('mockBanner')}</div>}
      {(notPaired || away) && (
        <div className="lib-connect">
          <p>{notPaired ? t('libraryNotPaired') : t('libraryAway')}</p>
          <a className="pill primary" href="#/connect">
            {t('connectTitle')}
          </a>
        </div>
      )}
      {tracks === null && <p className="muted">{t('libraryLoading')}</p>}
      {error && (
        <div className="lib-error">
          <p>{error.offline ? t('libraryOffline') : `${t('errorGeneric')}: ${error.msg}`}</p>
          <button className="pill" onClick={load}>
            {t('retry')}
          </button>
        </div>
      )}
      {tracks && tracks.length === 0 && !error && <p className="muted">{notPaired || away ? t('libraryNoneSaved') : t('libraryEmpty')}</p>}
      {tracks && tracks.length > 0 && (
        <ul className="track-list">
          {tracks.map((tr) => (
            <li key={tr.key} className="track">
              {renaming === tr.key ? (
                <RenameRow
                  track={tr}
                  onDone={(saved) => {
                    setRenaming(null);
                    if (saved) load();
                  }}
                />
              ) : (
                <>
                  <button className="track-main" onClick={() => open(tr.key)}>
                    <span className="track-title">{tr.title}</span>
                    <span className="track-meta">
                      {tr.saved && <span className="saved-chip">✓ {t('savedChip')}</span>}
                      {tr.artist && <span>{tr.artist}</span>}
                      {tr.key_name && (
                        <span className="key-chip">
                          {t('keyLabel')} {tr.key_name}
                        </span>
                      )}
                      {tr.bpm != null && <span>♩ {Math.round(tr.bpm)}</span>}
                      {tr.duration != null && <span>{formatTime(tr.duration)}</span>}
                      {tr.created_at && (
                        <span className="muted">
                          {t('analysedOn')} {new Date(tr.created_at).toLocaleDateString()}
                        </span>
                      )}
                    </span>
                  </button>
                  <div className="track-actions">
                    <button onClick={() => setRenaming(tr.key)}>{t('rename')}</button>
                    <button className="primary" onClick={() => open(tr.key)}>
                      {t('open')}
                    </button>
                  </div>
                </>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
