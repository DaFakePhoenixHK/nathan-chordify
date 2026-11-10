import { useEffect, useMemo, useSyncExternalStore } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useApp } from './store';
import { shownShift, variantOf, type LabelCtx } from '../grid/labels';
import { perf } from '../player/sync';

/** A phone held either way up: narrow, or too short for the tablet layout. */
export const PHONE_QUERY = '(max-width: 639px), (max-height: 500px)';
export const SHORT_QUERY = '(max-height: 500px)';
export const WIDE_QUERY = '(min-width: 1024px)';

export function useMedia(query: string): boolean {
  return useSyncExternalStore(
    (cb) => {
      const m = window.matchMedia(query);
      m.addEventListener('change', cb);
      return () => m.removeEventListener('change', cb);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
}

/** Everything a chord label depends on. */
export function useLabelCtx(): LabelCtx & { shift: number } {
  const { display, transpose, playIn, acc, level, slash, notation } = useApp(
    useShallow((s) => ({
      display: s.display,
      transpose: s.transpose,
      playIn: s.playIn,
      acc: s.acc,
      level: s.level,
      slash: s.slash,
      notation: s.notation,
    })),
  );
  const shift = shownShift(transpose, playIn);
  return useMemo(
    () => ({ variant: variantOf(display, shift, acc), level, slash, notation, shift }),
    [display, shift, acc, level, slash, notation],
  );
}

/** Dev-only render counter (checks "no per-frame React re-renders" in the M3 gate). */
export function useRenderCount(name: string): void {
  if (import.meta.env.DEV) perf.renders[name] = (perf.renders[name] ?? 0) + 1;
}

/** Apply the theme to <html data-theme>. */
export function useThemeAttr(): void {
  const theme = useApp((s) => s.theme);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'auto') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', theme);
  }, [theme]);
  const lang = useApp((s) => s.lang);
  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);
}
