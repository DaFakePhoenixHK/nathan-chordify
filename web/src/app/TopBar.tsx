import { useRef, useState } from 'react';
import { useApp, type Theme } from './store';
import { analyzeFile, analyzeInput } from './jobs';
import { stageName, useT } from '../i18n';

function Logo() {
  return (
    <svg viewBox="0 0 64 64" width="26" height="26" aria-hidden="true">
      <rect width="64" height="64" rx="14" fill="var(--logo-bg)" />
      <g fill="#f5f5f2">
        <rect x="10" y="14" width="8" height="36" rx="2" />
        <rect x="20" y="14" width="8" height="36" rx="2" />
        <rect x="30" y="14" width="8" height="36" rx="2" />
        <rect x="40" y="14" width="8" height="36" rx="2" />
      </g>
      <rect x="16" y="14" width="5" height="22" rx="1.5" fill="#e8a33a" />
      <rect x="36" y="14" width="5" height="22" rx="1.5" fill="#e8a33a" />
      <rect x="26" y="14" width="5" height="22" rx="1.5" fill="#3b82c4" />
    </svg>
  );
}

const THEME_NEXT: Record<Theme, Theme> = { auto: 'light', light: 'dark', dark: 'auto' };
const THEME_ICON: Record<Theme, string> = { auto: '◐', light: '○', dark: '●' };

export function TopBar() {
  const t = useT();
  const [text, setText] = useState('');
  const file = useRef<HTMLInputElement>(null);
  const lang = useApp((s) => s.lang);
  const theme = useApp((s) => s.theme);
  const busy = useApp((s) => s.job?.status === 'running' || s.job?.status === 'uploading');

  return (
    <header className="topbar">
      <a className="brand" href="#/" title={t('library')}>
        <Logo />
        <span className="brand-name">{t('appName')}</span>
      </a>
      <form
        className="analyze-form"
        onSubmit={(e) => {
          e.preventDefault();
          void analyzeInput(text).then(() => {
            if (useApp.getState().job) setText('');
          });
        }}
      >
        <input
          className="analyze-input"
          type="text"
          inputMode="url"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={t('inputPlaceholder')}
          aria-label={t('inputPlaceholder')}
          enterKeyHint="go"
        />
        <button type="button" className="icon-btn upload" title={t('uploadFile')} aria-label={t('uploadFile')} onClick={() => file.current?.click()}>
          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
            <path d="M12 16V4m0 0l-5 5m5-5l5 5M5 20h14" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
        <button type="submit" className="primary" disabled={busy || !text.trim()}>
          {t('analyze')}
        </button>
        <input
          ref={file}
          type="file"
          accept="audio/*,video/*,.m4a,.mp3,.wav,.flac,.ogg,.webm,.mp4,.mov"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void analyzeFile(f);
            e.target.value = '';
          }}
        />
      </form>
      <nav className="topnav">
        <a href="#/" className="navlink">
          {t('library')}
        </a>
        <button className="icon-btn" onClick={() => useApp.setState({ lang: lang === 'en' ? 'zh-TW' : 'en' })} title={t('language')}>
          {lang === 'en' ? '中文' : 'EN'}
        </button>
        <button
          className="icon-btn"
          onClick={() => useApp.setState({ theme: THEME_NEXT[theme] })}
          title={`${t('theme')}: ${t(`theme_${theme}`)}`}
          aria-label={`${t('theme')}: ${t(`theme_${theme}`)}`}
        >
          {THEME_ICON[theme]}
        </button>
      </nav>
    </header>
  );
}

export function JobPanel() {
  const t = useT();
  const job = useApp((s) => s.job);
  const lang = useApp((s) => s.lang);
  if (!job) return null;
  const title =
    job.status === 'uploading'
      ? t('jobUploading')
      : job.status === 'done'
        ? t('jobDone')
        : job.status === 'error'
          ? t('jobFailed')
          : t('jobTitle');
  const stages = job.seen.length ? job.seen : job.status === 'running' ? ['queued'] : [];
  return (
    <div className={`job-panel ${job.status}`} role="status" aria-live="polite">
      <div className="job-head">
        <b>{title}</b>
        <span className="job-label" title={job.label}>
          {job.label}
        </span>
        {(job.status === 'error' || job.status === 'done') && (
          <button className="icon-btn" onClick={() => useApp.setState({ job: null })} aria-label={t('jobDismiss')}>
            ×
          </button>
        )}
      </div>
      {job.status !== 'error' && (
        <>
          <ol className="stages">
            {stages.map((s) => {
              const cur = s === job.stage && job.status !== 'done';
              return (
                <li key={s} className={cur ? 'cur' : 'done'}>
                  <span className="mark">{cur ? '●' : '✓'}</span> {stageName(lang, s)}
                  {cur && <span className="pct"> {Math.round(job.pct)}%</span>}
                </li>
              );
            })}
          </ol>
          <div className="progress">
            <div style={{ width: `${Math.min(100, Math.max(0, job.status === 'done' ? 100 : job.pct))}%` }} />
          </div>
          <div className="job-msg">
            {job.msg}
            {job.etaS != null && job.etaS > 0 && <span className="muted"> · {t('jobEta', { s: Math.round(job.etaS) })}</span>}
          </div>
        </>
      )}
      {job.connectionLost && job.status === 'running' && <div className="job-warn">{t('jobConnectionLost')}</div>}
      {job.error && (
        <div className="job-error">
          <b>{job.error.msg}</b>
          {job.error.hint && <p>{job.error.hint}</p>}
          <code>{job.error.code}</code>
        </div>
      )}
    </div>
  );
}
