import { useEffect, useRef, useState } from 'react';
import { useT, type StringKey } from '../i18n';
import { IS_IOS, IS_STANDALONE } from '../app/platform';
import { navigate } from '../app/store';
import { checkLink, parsePair, setLink, useConnection, type Link } from './connection';

/** Camera → QR text. Returns a function that turns the camera off. */
async function startScanner(video: HTMLVideoElement, onText: (s: string) => void): Promise<() => void> {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false });
  let stopped = false;
  let timer = 0;
  const stop = () => {
    stopped = true;
    window.clearTimeout(timer);
    stream.getTracks().forEach((tr) => tr.stop());
    video.srcObject = null;
  };
  try {
    video.srcObject = stream;
    video.setAttribute('playsinline', '');
    await video.play();
    const { default: jsQR } = await import('jsqr');
    const canvas = document.createElement('canvas');
    const g = canvas.getContext('2d', { willReadFrequently: true });
    const look = () => {
      if (stopped || !g) return;
      if (video.videoWidth) {
        const scale = Math.min(1, 720 / Math.max(video.videoWidth, video.videoHeight));
        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        g.drawImage(video, 0, 0, canvas.width, canvas.height);
        const code = jsQR(g.getImageData(0, 0, canvas.width, canvas.height).data, canvas.width, canvas.height);
        if (code?.data) {
          onText(code.data);
          return;
        }
      }
      timer = window.setTimeout(look, 180);
    };
    look();
  } catch (e) {
    stop();
    throw e;
  }
  return stop;
}

/** On the phone or tablet: join this app to the computer that does the analysis. */
export function ConnectPage() {
  const t = useT();
  const { link, laptop } = useConnection();
  const [scanning, setScanning] = useState(false);
  const [text, setText] = useState('');
  const [problem, setProblem] = useState<StringKey | null>(null);
  const [busy, setBusy] = useState(false);
  const video = useRef<HTMLVideoElement>(null);

  const tryLink = async (l: Link) => {
    setBusy(true);
    setProblem(null);
    const r = await checkLink(l);
    setBusy(false);
    if (r === 'ok') {
      setLink(l);
      navigate({ page: 'library' });
    } else {
      setProblem(r === 'bad_code' ? 'connectBadCode' : 'connectUnreachable');
    }
  };

  const acceptText = (s: string) => {
    const l = parsePair(s);
    if (l) void tryLink(l);
    else setProblem('connectNotALink');
  };

  useEffect(() => {
    if (!scanning || !video.current) return;
    let stop: (() => void) | null = null;
    let live = true;
    startScanner(video.current, (s) => {
      setScanning(false);
      acceptText(s);
    })
      .then((fn) => {
        if (live) stop = fn;
        else fn();
      })
      .catch(() => {
        setScanning(false);
        setProblem('connectNoCamera');
      });
    return () => {
      live = false;
      stop?.();
    };
    // oxlint-disable-next-line react/exhaustive-effect-dependencies
  }, [scanning]);

  return (
    <div className="connect">
      <h1>{t('connectTitle')}</h1>
      {link && <p className={`connect-state ${laptop}`}>{laptop === 'offline' ? t('connectStateOff') : t('connectStateOn')}</p>}
      <ol className="connect-steps">
        <li>{t('connectStep1')}</li>
        <li>{t('connectStep2')}</li>
      </ol>
      {scanning ? (
        <div className="connect-camera">
          <video ref={video} muted playsInline />
          <button className="pill" onClick={() => setScanning(false)}>
            {t('cancel')}
          </button>
        </div>
      ) : (
        <button className="pill primary connect-scan" disabled={busy} onClick={() => setScanning(true)}>
          {busy ? t('connectChecking') : t('connectScan')}
        </button>
      )}
      <form
        className="connect-paste"
        onSubmit={(e) => {
          e.preventDefault();
          acceptText(text);
        }}
      >
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder={t('connectPaste')} aria-label={t('connectPaste')} autoCapitalize="off" autoCorrect="off" spellCheck={false} />
        <button type="submit" className="pill" disabled={busy || !text.trim()}>
          {t('connectGo')}
        </button>
      </form>
      {problem && <p className="error-text">{t(problem)}</p>}
      {IS_IOS && !IS_STANDALONE && (
        <div className="connect-install">
          <b>{t('installTitle')}</b>
          <p>{t('installSteps')}</p>
        </div>
      )}
      <p className="muted">{t('connectSavedNote')}</p>
      {link && (
        <button className="pill" onClick={() => setLink(null)}>
          {t('connectForget')}
        </button>
      )}
    </div>
  );
}
