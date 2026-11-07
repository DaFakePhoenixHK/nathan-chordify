import { useCallback } from 'react';
import { useApp, type Lang } from '../app/store';
import { en, zhTW, type StringKey } from './strings';

export type { StringKey };

const TABLES: Record<Lang, Record<StringKey, string>> = { en, 'zh-TW': zhTW };

export function translate(lang: Lang, key: StringKey, vars?: Record<string, string | number>): string {
  let s = TABLES[lang][key] ?? en[key] ?? key;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.replace(`{${k}}`, String(v));
  return s;
}

/** Translation function bound to the current UI language. */
export function useT(): (key: StringKey, vars?: Record<string, string | number>) => string {
  const lang = useApp((s) => s.lang);
  return useCallback((key, vars) => translate(lang, key, vars), [lang]);
}

/** Non-React access (event handlers outside components). */
export function tNow(key: StringKey, vars?: Record<string, string | number>): string {
  return translate(useApp.getState().lang, key, vars);
}

/** Translated stage name, falling back to the raw stage id. */
export function stageName(lang: Lang, stage: string): string {
  const k = `stage_${stage}` as StringKey;
  return k in en ? translate(lang, k) : stage;
}
