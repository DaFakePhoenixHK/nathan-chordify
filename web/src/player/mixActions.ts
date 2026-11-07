// Switching the Sound (2026-11-04) and the Audio choice (original / transposed recording, 2026-11-04):
// checks the server file exists (HEAD makes it on first use: a mix in a few seconds, a shifted copy in
// ~15 s for a 5-minute song), swaps the player file while keeping the position and the play state, mutes
// the recording for "Chords only", and remembers both choices per song in localStorage.
import { ApiError } from '../api/client';
import { showToast, useApp, type AudioKey, type MixId } from '../app/store';
import { tNow } from '../i18n';
import { engine } from './engine';
import { initChordSynth, unlockSynth } from './chordSynth';
import { audioShift, mixAudioUrl, mixDef, recallAudioKey, recallMix, rememberAudioKey, rememberMix, stemsListed } from './mix';

function songKey(): string | null {
  const r = useApp.getState().route;
  return r.page === 'song' ? r.key : null;
}

async function headOk(url: string): Promise<void> {
  let res: Response;
  try {
    res = await fetch(url, { method: 'HEAD', headers: { Accept: 'audio/mp4' } });
  } catch (e) {
    throw new ApiError(0, 'offline', e instanceof Error ? e.message : String(e));
  }
  if (res.ok) return;
  let code = `http_${res.status}`;
  let msg = res.statusText || `HTTP ${res.status}`;
  let hint: string | undefined;
  try {
    const r2 = await fetch(url, { method: 'GET', headers: { Accept: 'application/json' } });
    const j = (await r2.json()) as Record<string, unknown>;
    if (typeof j.code === 'string') code = j.code;
    if (typeof j.msg === 'string') msg = j.msg;
    if (typeof j.hint === 'string') hint = j.hint;
  } catch {
    /* not JSON */
  }
  throw new ApiError(res.status, code, msg, hint);
}

let seq = 0;

/**
 * Load the player file for a sound + the current Audio choice and Transpose. Keeps the play position
 * and the play / pause state; the old file keeps playing while the server makes a new one.
 */
async function applySource(id: MixId, remember: boolean): Promise<void> {
  const st = useApp.getState();
  const key = songKey();
  if (!key || !st.analysis || !st.audioBase) return;
  const def = mixDef(id);
  const my = ++seq;
  initChordSynth();
  if (def.synth) unlockSynth();
  const base = st.audioBase;
  if (def.needsStems) {
    if (base.startsWith('blob:')) {
      showToast(tNow('mixMockFile'), 'error');
      return;
    }
    if (!stemsListed(st.analysis) || st.mixAvail === false) {
      useApp.setState({ mixAvail: false });
      showToast(tNow('mixNoStems'), 'error');
      return;
    }
  }
  const shift = audioShift(st.audioKey, st.transpose);
  const url = mixAudioUrl(base, def.audio, shift);
  try {
    if ((def.needsStems || shift !== 0) && url !== st.audioSrc) {
      useApp.setState({ mixBusy: def.needsStems, shiftBusy: shift !== 0 });
      await headOk(url);
      if (my !== seq) return; // a newer choice took over
    }
    await engine.swapSource(url);
    if (my !== seq) return;
    engine.setMuted(def.muted);
    useApp.setState({ mix: id, mixBusy: false, shiftBusy: false, ...(def.needsStems ? { mixAvail: true } : {}) });
    if (remember) rememberMix(localStorage, key, id);
  } catch (e) {
    if (my !== seq) return;
    useApp.setState({ mixBusy: false, shiftBusy: false });
    if (e instanceof ApiError && e.code === 'no_stems') {
      useApp.setState({ mixAvail: false });
      showToast(tNow('mixNoStems'), 'error');
    } else if (e instanceof ApiError && (e.code === 'shift_failed' || e.code === 'bad_transpose')) {
      // the recording stays in its own key for this song; the chord names still move
      useApp.setState({ audioKey: 'original' });
      showToast(`${tNow('audioKeyFailed')}: ${e.message}`, 'error');
    } else {
      const msg = e instanceof Error ? e.message : String(e);
      showToast(`${tNow('mixFailed')}: ${msg}`, 'error');
    }
  }
}

/** Choose what plays. Keeps the play position and the play / pause state; the choice is remembered per song. */
export async function setMix(id: MixId, remember = true): Promise<void> {
  await applySource(id, remember);
}

/** Choose whether the recording follows Transpose. Remembered per song. */
export function setAudioKey(v: AudioKey, remember = true): void {
  const key = songKey();
  if (useApp.getState().audioKey !== v) useApp.setState({ audioKey: v });
  if (remember && key) rememberAudioKey(localStorage, key, v);
  scheduleRefresh(0);
}

// Transpose clicks come in bursts (+ + +): wait a moment, then load the file for the final value.
let refreshTimer = 0;
const REFRESH_DELAY_MS = 350;

function scheduleRefresh(delay = REFRESH_DELAY_MS): void {
  window.clearTimeout(refreshTimer);
  refreshTimer = window.setTimeout(() => {
    refreshTimer = 0;
    const st = useApp.getState();
    if (!st.analysis || !st.audioBase) return;
    const want = mixAudioUrl(st.audioBase, mixDef(st.mix).audio, audioShift(st.audioKey, st.transpose));
    if (want !== st.audioSrc) void applySource(st.mix, false);
    else useApp.setState({ shiftBusy: false });
  }, delay);
}

let wired = false;

/** Follow Transpose / the Audio choice while a song is open (once). */
function initMixActions(): void {
  if (wired) return;
  wired = true;
  useApp.subscribe((s, p) => {
    if (s.route.page !== 'song' || !s.analysis || !s.audioBase) return;
    if (s.analysis !== p.analysis) return; // a song (re)load: applyRememberedMix takes care of it
    if (s.transpose !== p.transpose || s.audioKey !== p.audioKey) {
      if (audioShift(s.audioKey, s.transpose) !== audioShift(p.audioKey, p.transpose)) {
        useApp.setState({ shiftBusy: audioShift(s.audioKey, s.transpose) !== 0 });
        scheduleRefresh();
      }
    }
  });
}

/** On opening a song: start from the sound and the Audio choice remembered for it. */
export function applyRememberedMix(key: string): void {
  initChordSynth();
  initMixActions();
  const audioKey = recallAudioKey(localStorage, key);
  const id = recallMix(localStorage, key);
  useApp.setState({ audioKey });
  if (id === 'original' && audioShift(audioKey, useApp.getState().transpose) === 0) {
    engine.setMuted(false);
    useApp.setState({ mix: 'original' });
    return;
  }
  void applySource(id, false);
}
