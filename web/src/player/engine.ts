// v1 (M3) playback engine (§8.3): a plain <audio> element. Tempo = playbackRate with preservesPitch;
// loops are seek-based (handled by the sync loop). signalsmith-stretch replaces this in M7.
import { useApp } from '../app/store';
import { registerClock, requestTick, startLoop, stopLoop } from './sync';
import { initAudioSession } from '../app/platform';

type PitchAudio = HTMLAudioElement & { preservesPitch?: boolean; webkitPreservesPitch?: boolean; mozPreservesPitch?: boolean };

function setMediaAction(a: MediaSessionAction, fn: MediaSessionActionHandler): void {
  try {
    navigator.mediaSession.setActionHandler(a, fn);
  } catch {
    /* not offered on this version */
  }
}

class AudioEngine {
  readonly el: PitchAudio;
  private wakeLock: { release(): Promise<void> } | null = null;
  private holdAwake = false;
  /** iPhones only load a file's details once it plays, and ignore a seek made before that. */
  private pendingSeek: number | null = null;

  constructor() {
    initAudioSession();
    const el = document.createElement('audio') as PitchAudio;
    el.preload = 'auto';
    el.setAttribute('playsinline', '');
    this.el = el;
    this.applyPitch();
    el.addEventListener('play', () => {
      useApp.setState({ playing: true });
      startLoop();
      void this.lockScreen(true);
    });
    el.addEventListener('pause', () => {
      useApp.setState({ playing: false });
      stopLoop();
      requestTick();
      void this.lockScreen(this.holdAwake);
    });
    el.addEventListener('ended', () => {
      const { loop } = useApp.getState();
      if (loop.a != null && loop.b != null) {
        this.seek(loop.a);
        void this.play();
      }
    });
    const dur = () => useApp.setState({ duration: Number.isFinite(el.duration) ? el.duration : 0 });
    el.addEventListener('durationchange', dur);
    el.addEventListener('loadedmetadata', () => {
      dur();
      this.applyRate(useApp.getState().rate);
      if (this.pendingSeek != null) {
        el.currentTime = this.pendingSeek;
        this.pendingSeek = null;
      }
      requestTick();
    });
    el.addEventListener('seeked', () => requestTick());
    el.addEventListener('timeupdate', () => {
      if (el.paused) requestTick();
    });
    useApp.subscribe((s, prev) => {
      if (s.rate !== prev.rate) this.applyRate(s.rate);
      if (s.volume !== prev.volume) el.volume = Math.min(1, Math.max(0, s.volume));
    });
    el.volume = useApp.getState().volume;
    // the system drops the screen lock whenever the app is hidden: take it again on return
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState !== 'visible') return;
      this.wakeLock = null;
      if (this.holdAwake || !el.paused) void this.lockScreen(true);
    });
    this.wireLockScreenControls();
  }

  /** Play / pause and seeking from the iPhone lock screen and Control Centre. */
  private wireLockScreenControls(): void {
    if (!('mediaSession' in navigator)) return;
    setMediaAction('play', () => void this.play());
    setMediaAction('pause', () => this.pause());
    setMediaAction('seekto', (d) => {
      if (d.seekTime != null) this.seek(d.seekTime);
    });
  }

  /** What the lock screen shows. */
  setNowPlaying(title: string | null, artist?: string | null): void {
    if ('mediaSession' in navigator) {
      navigator.mediaSession.metadata = title ? new MediaMetadata({ title, artist: artist ?? '' }) : null;
    }
  }

  /** Keep the screen on while a song is open: playing along live, nothing is "playing". */
  setHoldAwake(on: boolean): void {
    this.holdAwake = on;
    void this.lockScreen(on || !this.el.paused);
  }

  private applyPitch(): void {
    this.el.preservesPitch = true;
    this.el.webkitPreservesPitch = true;
    this.el.mozPreservesPitch = true;
  }

  private applyRate(r: number): void {
    this.applyPitch();
    this.el.defaultPlaybackRate = r;
    this.el.playbackRate = r;
  }

  private async lockScreen(on: boolean): Promise<void> {
    // Wake Lock keeps a phone screen on during playback (§8.2). Best effort.
    try {
      if (on && !this.wakeLock && 'wakeLock' in navigator) {
        const wl = (navigator as unknown as { wakeLock: { request(t: 'screen'): Promise<{ release(): Promise<void> }> } }).wakeLock;
        this.wakeLock = await wl.request('screen');
      } else if (!on && this.wakeLock) {
        await this.wakeLock.release();
        this.wakeLock = null;
      }
    } catch {
      this.wakeLock = null;
    }
  }

  load(src: string | null): void {
    this.pause();
    this.pendingSeek = null;
    if (!src) {
      this.el.removeAttribute('src');
      this.el.load();
      useApp.setState({ duration: 0, audioSrc: null });
      return;
    }
    if (this.el.getAttribute('src') !== src) {
      this.el.src = src;
      this.el.load();
    }
    this.applyRate(useApp.getState().rate);
    useApp.setState({ audioSrc: src });
    requestTick();
  }

  /**
   * Change the file but keep the position, the tempo and the play / pause state (the Sound selector:
   * original ↔ no singer ↔ band only). Resolves once the new file's metadata is in (or after 8 s).
   */
  async swapSource(src: string): Promise<void> {
    const el = this.el;
    if (el.getAttribute('src') === src) return;
    const t = el.currentTime || 0;
    const wasPlaying = !el.paused;
    el.pause();
    el.src = src;
    el.load();
    await new Promise<void>((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        el.removeEventListener('loadedmetadata', finish);
        el.removeEventListener('error', finish);
        resolve();
      };
      el.addEventListener('loadedmetadata', finish);
      el.addEventListener('error', finish);
      // a paused iPhone may not load the new file at all until play, so don't hold the screen up
      window.setTimeout(finish, wasPlaying ? 8000 : 1500);
    });
    this.applyRate(useApp.getState().rate);
    this.seek(t);
    useApp.setState({ audioSrc: src });
    requestTick();
    if (wasPlaying) await this.play();
  }

  /** "Chords only": the recording keeps running (it is the clock) but is not heard. */
  setMuted(on: boolean): void {
    this.el.muted = on;
  }

  get time(): number {
    return this.pendingSeek ?? (this.el.currentTime || 0);
  }

  get paused(): boolean {
    return this.el.paused;
  }

  async play(): Promise<void> {
    if (!this.el.getAttribute('src')) return;
    try {
      await this.el.play();
    } catch {
      /* autoplay policy / no source: the UI stays paused */
    }
  }

  pause(): void {
    if (!this.el.paused) this.el.pause();
  }

  toggle(): void {
    if (this.el.paused) void this.play();
    else this.pause();
  }

  seek(t: number): void {
    const d = Number.isFinite(this.el.duration) ? this.el.duration : Infinity;
    const to = Math.min(Math.max(0, t), Math.max(0, d - 0.05));
    if (this.el.readyState < 1) this.pendingSeek = to;
    else this.el.currentTime = to;
    requestTick();
  }
}

export const engine = new AudioEngine();
registerClock(engine);
