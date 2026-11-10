// What kind of device this is, and the small things Apple devices need.

const nav = (typeof navigator === 'undefined' ? {} : navigator) as Navigator & { standalone?: boolean; audioSession?: { type: string } };

/** iPhone or iPad. iPadOS reports itself as a Mac, so touch points tell them apart. */
export const IS_IOS = /iP(hone|ad|od)/.test(nav.userAgent ?? '') || (nav.platform === 'MacIntel' && (nav.maxTouchPoints ?? 0) > 1);

/** Running inside the installed iPhone / iPad app. */
export const IS_NATIVE = typeof location !== 'undefined' && location.protocol === 'capacitor:';

/** Opened from the home-screen icon rather than a browser tab. */
export const IS_STANDALONE =
  IS_NATIVE || nav.standalone === true || (typeof matchMedia !== 'undefined' && matchMedia('(display-mode: standalone)').matches);

export const IS_TOUCH = typeof matchMedia !== 'undefined' && matchMedia('(pointer: coarse)').matches;

/** Without this the ring/silent switch mutes the chord voice (Web Audio). */
export function initAudioSession(): void {
  try {
    if (nav.audioSession) nav.audioSession.type = 'playback';
  } catch {
    /* older iOS */
  }
}

/** Hand a text file to the iOS share sheet (Files, Notes, AirDrop); falls back to a download. */
export async function shareOrDownload(name: string, text: string, download: (name: string, text: string) => void): Promise<void> {
  if (IS_NATIVE) {
    const { Share } = await import('@capacitor/share');
    await Share.share({ title: name, text, dialogTitle: name }).catch(() => undefined); // closing the sheet is not an error
    return;
  }
  const file = new File([text], name, { type: 'text/plain' });
  if (nav.canShare?.({ files: [file] })) {
    try {
      await nav.share({ files: [file] });
      return;
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return; // the user closed the sheet
    }
  }
  download(name, text);
}

/**
 * Copy text that is still being fetched. Safari only allows copying during the tap itself, so the
 * clipboard is opened right away and promised the text.
 */
export async function copyLater(text: Promise<string>): Promise<void> {
  if (IS_IOS && typeof ClipboardItem !== 'undefined' && navigator.clipboard?.write) {
    const blob = text.then((s) => new Blob([s], { type: 'text/plain' }));
    await navigator.clipboard.write([new ClipboardItem({ 'text/plain': blob })]);
    return;
  }
  await navigator.clipboard.writeText(await text);
}
