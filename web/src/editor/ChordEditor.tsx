import { useEffect, useMemo, useRef, useState, type KeyboardEvent as RKeyboardEvent } from 'react';
import { api } from '../api/client';
import type { ChordOut, ParsedChord, PickerVocab } from '../api/types';
import { showToast, useApp } from '../app/store';
import { PHONE_QUERY, useLabelCtx, useMedia } from '../app/hooks';
import { useT } from '../i18n';
import { chordLabel, isNoChord } from '../grid/labels';
import { getTimeline, segAt } from '../grid/model';
import { formatTime } from '../player/sync';
import { playChord } from '../gt/synth';
import { acceptSuggestions, chordById, deleteChange, insertRest, keyAt, rejectSuggestions, setChord } from './ops';
import { beatOf, clearSel, moveSel, selectRange } from './session';
import { composeChord } from './logic';

const vocabCache = new Map<string, Promise<PickerVocab>>();
function vocabFor(key: string, shift: number, pref: string): Promise<PickerVocab> {
  const k = `${key}|${shift}|${pref}`;
  let p = vocabCache.get(k);
  if (!p) {
    p = api.vocab(key, shift, pref);
    p.catch(() => vocabCache.delete(k));
    vocabCache.set(k, p);
  }
  return p;
}

/** Where a time sits in the bars: "Bar 12 · beat 3". */
function barBeat(beats: { t: number; bar: number; beat: number }[], t: number): { bar: number; beat: number } | null {
  let b = null;
  for (const x of beats) if (x.t <= t + 1e-3 && x.bar >= 0) b = x;
  return b ? { bar: b.bar + 1, beat: b.beat } : null;
}

const noFocus = (e: { preventDefault(): void }) => e.preventDefault();

interface Pick {
  root: string | null;
  suffix: string;
  bass: string;
}
const NO_PICK: Pick = { root: null, suffix: '', bass: '' };

/** The chord picker of edit mode (§8.4): typed entry (validated by the server), top alternatives,
 *  equivalents, root / quality / bass pickers, delete change, rest, accept / reject, play. */
export function ChordEditor() {
  const t = useT();
  const sel = useApp((s) => s.sel);
  const editMode = useApp((s) => s.editMode);
  const analysis = useApp((s) => s.analysis);
  const display = useApp((s) => s.display);
  const entry = useApp((s) => s.entry);
  const entryFocus = useApp((s) => s.entryFocus);
  const detail = useApp((s) => s.detail);
  const gtMode = useApp((s) => s.gtMode);
  const acc = useApp((s) => s.acc);
  const pending = useApp((s) => s.pendingOps);
  const ctx = useLabelCtx();
  const phone = useMedia(PHONE_QUERY);
  const input = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<ParsedChord | null>(null);
  const [altLabels, setAltLabels] = useState<Record<string, string[]>>({});
  const [vocab, setVocab] = useState<PickerVocab | null>(null);
  // picker state belongs to one selection: a new selection starts with fresh pickers
  const [pickState, setPickState] = useState<Pick & { sel: string }>({ ...NO_PICK, sel: '' });

  const chord: ChordOut | null = sel?.kind === 'chord' ? chordById(analysis, sel.id) : null;
  const t0 = chord?.start ?? sel?.t0 ?? 0;
  const t1 = chord?.end ?? sel?.t1 ?? 0;
  const key = analysis ? keyAt(analysis, t0) : 'C:major';
  const shift = ctx.shift;
  const segs = analysis ? getTimeline(analysis, detail, gtMode) : [];
  const here: ChordOut | null = chord ?? (segs[segAt(segs, t0 + 1e-3)]?.chord ?? null);
  const hartes = useMemo(() => {
    if (!here) return [];
    const alts = (here.alternatives ?? []).slice(0, 3).map((a) => a.harte);
    return [...new Set([...alts, ...(here.equivalents ?? [])])];
  }, [here]);

  // focus the entry when a shortcut starts typing
  useEffect(() => {
    if (entryFocus > 0) input.current?.focus();
  }, [entryFocus]);

  const selKey = sel ? `${sel.kind}:${sel.kind === 'chord' ? sel.id : ''}:${sel.t0}:${sel.t1}` : '';
  const pick: Pick = pickState.sel === selKey ? pickState : NO_PICK;
  const setPick = (p: Pick) => setPickState({ ...p, sel: selKey });

  // typed entry → server parse (debounced): harte, labels at the shift, or an error with a hint
  // (a preview counts only while preview.input === entry, so a stale one is never used)
  useEffect(() => {
    if (!entry.trim() || !editMode) return;
    let live = true;
    const id = window.setTimeout(() => {
      api
        .parse({ text: entry, key, shift, pref: acc })
        .then((r) => live && setPreview(r))
        .catch(() => live && setPreview(null));
    }, 150);
    return () => {
      live = false;
      window.clearTimeout(id);
    };
  }, [entry, key, shift, acc, editMode]);

  // labels of the alternatives / equivalents at this transposition
  useEffect(() => {
    if (hartes.length === 0 || !editMode) return;
    let live = true;
    api
      .parse({ hartes, key, shift, pref: acc })
      .then((r) => live && setAltLabels(r.labels ?? {}))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [hartes, key, shift, acc, editMode]);

  useEffect(() => {
    if (!editMode || !sel) return;
    let live = true;
    vocabFor(key, shift, acc)
      .then((v) => live && setVocab(v))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [key, shift, acc, editMode, sel]);

  if (!editMode || !sel || !analysis) return null;

  const LEVEL = { simple: 0, standard: 1, full: 2 }[ctx.level];
  const lettersCtx = { ...ctx, notation: 'letters' as const, level: 'full' as const, slash: true };
  const hereLabel = here ? (isNoChord(here) ? 'N.C.' : chordLabel(here, lettersCtx)) : '—';
  const okPreview = preview && preview.ok && preview.input === entry ? preview : null;
  const badPreview = preview && !preview.ok && preview.input === entry ? preview : null;
  const shownPreview = okPreview?.shown?.[Math.max(LEVEL, 1)] ?? null;
  const bb = barBeat(analysis.beats, t0);
  const sq = beatOf(sel);
  const half = sq ? (sq.t0 + sq.t1) / 2 : 0;
  const isOrig = (h: string) => here?.alternatives?.[0]?.harte === h && here.edited;

  const compose = (p: Pick) => {
    setPick(p);
    if (!p.root) return;
    useApp.setState({ entry: composeChord(p.root, p.suffix, p.bass) });
  };

  const commit = (advance: boolean) => {
    const text = entry.trim();
    if (!text) {
      if (advance) moveSel(1);
      return;
    }
    if (badPreview) {
      showToast(`${badPreview.error ?? ''} ${badPreview.hint ?? ''}`.trim(), 'error');
      return;
    }
    void setChord(sel, okPreview?.harte ? { harte: okPreview.harte, text } : { text, shift });
    useApp.setState({ entry: '' });
    setPick(NO_PICK);
    if (advance) moveSel(1);
  };

  const onKey = (e: RKeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      commit(false);
    } else if (e.key === 'Tab') {
      e.preventDefault();
      if (entry.trim()) commit(true);
      else moveSel(e.shiftKey ? -1 : 1);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      if (entry) useApp.setState({ entry: '' });
      else {
        clearSel();
        input.current?.blur();
      }
    }
    e.stopPropagation();
  };

  const play = () => {
    const v = okPreview?.voicing ?? (here ? (display?.voicings[String(shift)]?.[here.id] ?? (shift === 0 ? here.voicing : undefined)) : undefined);
    const notes = [...(v?.lh ?? []), ...(v?.rh ?? [])];
    if (notes.length) playChord(notes);
  };

  const pickHarte = (h: string) => void setChord(sel, { harte: h });

  const meta = chord
    ? `${formatTime(t0, true)}–${formatTime(t1, true)} · ${chord.layer === 'passing' ? t('layerPassing') : t('layerMain')}${
        chord.suggestion ? ` · ${t('suggestion')}` : ''
      }${chord.edited ? ` · ${t('edited')}` : ` · ${t('confidence')} ${Math.round(chord.conf * 100)}%`}`
    : `${bb ? t('selBar', { bar: bb.bar, beat: bb.beat }) : ''} · ${t('selRange', { from: formatTime(t0, true), to: formatTime(t1, true) })}`;

  const body = (
    <section className={`chord-editor${phone ? ' sheet' : ''}`} aria-label={t('editChord')} onPointerDown={(e) => e.stopPropagation()}>
      <div className="ce-head">
        <span className={`ce-cur${chord?.layer === 'passing' ? ' passing' : ''}${chord?.suggestion ? ' suggest' : ''}`}>{hereLabel}</span>
        <span className="ce-arrow">→</span>
        <span className={`ce-new${badPreview ? ' bad' : ''}`}>{shownPreview ?? (badPreview ? '?' : '…')}</span>
        <span className="ce-meta">{meta}</span>
        {pending > 0 && <span className="ce-saving">{t('saving')}</span>}
        <button className="icon-btn ce-close" onMouseDown={noFocus} onClick={clearSel} aria-label={t('close')}>
          ×
        </button>
      </div>
      <div className="ce-entry">
        <input
          ref={input}
          className={`ce-input${badPreview ? ' bad' : okPreview ? ' ok' : ''}`}
          value={entry}
          placeholder={t('typeChord')}
          autoComplete="off"
          autoCapitalize="off"
          spellCheck={false}
          enterKeyHint="done"
          aria-label={t('typeChord')}
          aria-invalid={!!badPreview}
          onChange={(e) => useApp.setState({ entry: e.target.value })}
          onKeyDown={onKey}
        />
        <button className="primary" onMouseDown={noFocus} onClick={() => commit(false)} disabled={!entry.trim() || !!badPreview}>
          {t('setChord')} ⏎
        </button>
        <button onMouseDown={noFocus} onClick={play} title={t('playChord')} aria-label={t('playChord')}>
          ▶
        </button>
        {badPreview && (
          <span className="ce-error" role="alert">
            {badPreview.error} <span className="muted">{badPreview.hint}</span>
          </span>
        )}
      </div>
      {hartes.length > 0 && (
        <div className="ce-row ce-alts">
          <span className="ce-title">{t('topAlternatives')}</span>
          {(here?.alternatives ?? []).slice(0, 3).map((a) => (
            <button key={a.harte} className="chip ce-alt" onMouseDown={noFocus} onClick={() => pickHarte(a.harte)} title={a.harte}>
              <b>{altLabels[a.harte]?.[Math.max(LEVEL, 1)] ?? a.harte}</b>
              <small>{isOrig(a.harte) ? t('was', { label: '' }).trim() : `${Math.round(a.conf * 100)}%`}</small>
            </button>
          ))}
          {(here?.equivalents ?? []).map((h) => (
            <button key={h} className="chip ce-eq" onMouseDown={noFocus} onClick={() => pickHarte(h)} title={`${t('equivalents')}: ${h}`}>
              {altLabels[h]?.[2] ?? h}
            </button>
          ))}
        </div>
      )}
      {vocab && (
        <div className="ce-row ce-pick">
          <span className="ce-title">{t('root')}</span>
          <div className="ce-roots" role="group" aria-label={t('root')}>
            {vocab.roots.map((r) => (
              <button
                key={r.pc}
                className={pick.root === r.name ? 'on' : ''}
                onMouseDown={noFocus}
                onClick={() => compose({ ...pick, root: r.name })}
              >
                {r.name}
              </button>
            ))}
          </div>
          <label className="ce-sel">
            <span className="ce-title">{t('quality')}</span>
            <select className="select" value={pick.suffix} onChange={(e) => compose({ ...pick, suffix: e.target.value })}>
              {vocab.qualities.map((q) => (
                <option key={q.q} value={q.suffix}>
                  {q.label}
                </option>
              ))}
            </select>
          </label>
          <label className="ce-sel">
            <span className="ce-title">{t('bassPick')}</span>
            <select className="select" value={pick.bass} onChange={(e) => compose({ ...pick, bass: e.target.value })}>
              <option value="">{t('noBass')}</option>
              {vocab.roots.map((r) => (
                <option key={r.pc} value={r.name}>
                  /{r.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}
      <div className="ce-row ce-actions">
        {sq && (
          <div className="segmented" role="group" aria-label={t('thisBeat')}>
            <button onMouseDown={noFocus} onClick={() => selectRange(sq.t0, sq.t1)} className={sel.kind === 'range' && sel.t0 === sq.t0 && sel.t1 === sq.t1 ? 'on' : ''}>
              {t('wholeBeat')}
            </button>
            <button onMouseDown={noFocus} onClick={() => selectRange(sq.t0, half)} className={sel.kind === 'range' && sel.t0 === sq.t0 && sel.t1 === half ? 'on' : ''}>
              {t('firstHalf')}
            </button>
            <button onMouseDown={noFocus} onClick={() => selectRange(half, sq.t1)} className={sel.kind === 'range' && sel.t0 === half && sel.t1 === sq.t1 ? 'on' : ''}>
              {t('secondHalf')}
            </button>
          </div>
        )}
        {chord && (
          <button onMouseDown={noFocus} onClick={() => void deleteChange(chord)} title="Del">
            {t('deleteChange')}
          </button>
        )}
        <button onMouseDown={noFocus} onClick={() => void insertRest(sel)}>
          {t('rest')}
        </button>
        {chord?.suggestion && (
          <>
            <button className="primary" onMouseDown={noFocus} onClick={() => void acceptSuggestions([chord])}>
              ✓ {t('accept')}
            </button>
            <button onMouseDown={noFocus} onClick={() => void rejectSuggestions([chord])}>
              ✗ {t('reject')}
            </button>
          </>
        )}
      </div>
    </section>
  );
  if (!phone) return body;
  return (
    <>
      <div className="sheet-backdrop light" onClick={clearSel} />
      {body}
    </>
  );
}
