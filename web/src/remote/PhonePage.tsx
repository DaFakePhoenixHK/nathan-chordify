import { useCallback, useEffect, useState } from 'react';
import { useT } from '../i18n';

interface PhoneStatus {
  token: string;
  public_url: string | null;
  app_url: string;
  app_ready: boolean;
  pair_link: string | null;
  link: { state: 'off' | 'starting' | 'on' | 'error'; url: string | null; error: string | null };
}

async function qrDataUrl(text: string): Promise<string> {
  const { default: qrcode } = await import('qrcode-generator');
  const q = qrcode(0, 'M');
  q.addData(text);
  q.make();
  return q.createDataURL(8, 16);
}

const post = (body?: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
});

/** On the computer: switch the phone link on, and show the code a phone scans to connect. */
export function PhonePage() {
  const t = useT();
  const [st, setSt] = useState<PhoneStatus | null>(null);
  const [qr, setQr] = useState<{ link: string; img: string } | null>(null);
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = useCallback((url = '/api/phone/status', init?: RequestInit) => {
    return fetch(url, init)
      .then((r) => (r.ok ? (r.json() as Promise<PhoneStatus>) : Promise.reject(new Error(String(r.status)))))
      .then(setSt)
      .catch(() => setFailed(true));
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // while the link starts (a few seconds), ask again
  const starting = st?.link.state === 'starting' || (st?.link.state === 'error' && !!st.link.error);
  useEffect(() => {
    if (!starting) return;
    const id = window.setInterval(() => void load(), 1500);
    return () => window.clearInterval(id);
  }, [starting, load]);

  useEffect(() => {
    const link = st?.pair_link;
    if (link) void qrDataUrl(link).then((img) => setQr({ link, img }));
  }, [st?.pair_link]);

  if (failed) return <div className="center-msg">{t('phoneOnlyHere')}</div>;
  if (!st) return <div className="center-msg muted">{t('libraryLoading')}</div>;

  const toggle = (on: boolean) => {
    setBusy(true);
    void load('/api/phone/link', post({ on })).finally(() => setBusy(false));
  };
  const state = st.link.state;
  const img = qr && qr.link === st.pair_link ? qr.img : null;
  return (
    <div className="connect">
      <h1>{t('phoneTitle')}</h1>
      <p>{t('phoneLinkHelp')}</p>
      {state === 'off' && (
        <button className="pill primary connect-scan" disabled={busy} onClick={() => toggle(true)}>
          {t('phoneLinkStart')}
        </button>
      )}
      {state === 'starting' && <p className="muted">{t('phoneLinkStarting')}</p>}
      {state === 'error' && <p className="error-text">{st.link.error}</p>}
      {st.pair_link && (
        <>
          <p>{t('phoneScanHelp')}</p>
          {img && <img className="phone-qr" src={img} alt={t('phoneQrAlt')} />}
          {!st.app_ready && <p className="connect-install">{t('phoneAppNotPublished')}</p>}
          <p className="muted">{t('phoneSecret')}</p>
          <button
            className="pill"
            onClick={() => {
              if (window.confirm(t('phoneNewCodeAsk'))) void load('/api/phone/reset', post());
            }}
          >
            {t('phoneNewCode')}
          </button>
        </>
      )}
      {state !== 'off' && (
        <button className="pill" disabled={busy} onClick={() => toggle(false)}>
          {t('phoneLinkStop')}
        </button>
      )}
    </div>
  );
}
