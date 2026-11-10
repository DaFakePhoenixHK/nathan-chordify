import { useEffect, useState } from 'react';
import { ApiError } from '../api/client';
import type { Analysis, TrackSummary } from '../api/types';
import { showToast, useApp } from '../app/store';
import { useT, tNow } from '../i18n';
import { audioShift, mixAudioUrl, mixDef, stemsListed, type ServerMix } from '../player/mix';
import { isFromDevice, isSaved, putSong, removeSong, type SavedAudio } from './saved';

function summaryOf(a: Analysis, key: string): TrackSummary {
  return { key, title: a.source.title, artist: a.source.artist ?? null, duration: useApp.getState().duration || null, key_name: null, bpm: a.tempo.bpm, created_at: a.created_at };
}

async function fetchRecording(base: string, mix: ServerMix, shift: number): Promise<Blob | null> {
  const res = await fetch(mixAudioUrl(base, mix, shift));
  if (res.status === 409) return null; // this song has no separated voice
  if (!res.ok) throw new ApiError(res.status, `http_${res.status}`, res.statusText);
  const data = await res.blob();
  return data.type ? data : data.slice(0, data.size, 'audio/mp4');
}

/** The recording, the version without the singer, and whatever is being listened to right now. */
async function saveThisSong(key: string, onStep: (done: number, total: number) => void): Promise<void> {
  const st = useApp.getState();
  const { analysis, display, audioBase } = st;
  if (!analysis || !display || !audioBase) return;
  const wanted: { mix: ServerMix; shift: number }[] = [{ mix: 'original', shift: 0 }];
  if (stemsListed(analysis)) wanted.push({ mix: 'novocals', shift: 0 });
  const nowMix = mixDef(st.mix).audio;
  const nowShift = audioShift(st.audioKey, st.transpose);
  if (!wanted.some((w) => w.mix === nowMix && w.shift === nowShift)) wanted.push({ mix: nowMix, shift: nowShift });
  const audio: SavedAudio[] = [];
  for (const [i, w] of wanted.entries()) {
    onStep(i, wanted.length);
    const blob = await fetchRecording(audioBase, w.mix, w.shift);
    if (blob) audio.push({ ...w, blob });
  }
  await putSong(key, analysis, display, audio, summaryOf(analysis, key));
}

/** Keep this song on the phone or tablet (works with no computer and no internet). */
export function SaveButton({ phone }: { phone: boolean }) {
  const t = useT();
  const analysis = useApp((s) => s.analysis);
  const route = useApp((s) => s.route);
  const key = route.page === 'song' ? route.key : null;
  const [saved, setSaved] = useState(false);
  const [step, setStep] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    if (key) void isSaved(key).then((v) => live && setSaved(v));
    return () => {
      live = false;
    };
    // oxlint-disable-next-line react/exhaustive-effect-dependencies -- re-check when the song reloads
  }, [key, analysis]);

  if (!key || !analysis) return null;

  const onClick = async () => {
    if (step) return;
    try {
      if (saved) {
        if (!window.confirm(tNow('saveRemoveAsk'))) return;
        await removeSong(key);
        setSaved(false);
        showToast(tNow('saveRemoved'));
        return;
      }
      if (isFromDevice(analysis)) return;
      setStep('…');
      await saveThisSong(key, (done, total) => setStep(`${done + 1}/${total}`));
      setSaved(true);
      showToast(tNow('saveDone'));
    } catch (e) {
      const full = e instanceof DOMException && e.name === 'QuotaExceededError';
      showToast(full ? tNow('saveFull') : `${tNow('saveFailed')}: ${e instanceof Error ? e.message : tNow('errorGeneric')}`, 'error');
    } finally {
      setStep(null);
    }
  };

  const label = step ? t('saveBusy', { step }) : saved ? t('saveOn') : t('saveOff');
  return (
    <button className={`${phone ? 'icon-btn' : 'pill'} save-btn${saved ? ' on' : ''}`} onClick={() => void onClick()} title={label} aria-label={label} aria-pressed={saved}>
      {step && phone ? (
        step
      ) : (
        <>
          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
            <rect x="6.5" y="2.5" width="11" height="19" rx="2.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
            {saved ? (
              <path d="M9.5 12.2l2 2 3.2-4" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" />
            ) : (
              <path d="M12 7.5v7m0 0l-2.6-2.6M12 14.5l2.6-2.6" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" />
            )}
          </svg>
          {!phone && <span> {label}</span>}
        </>
      )}
    </button>
  );
}
