import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  type CSSProperties,
  type MouseEvent as RMouseEvent,
  type PointerEvent as RPointerEvent,
  type ReactNode,
} from 'react';
import { useApp, setLoop, type Loop } from '../app/store';
import { PHONE_QUERY, useLabelCtx, useMedia, useRenderCount } from '../app/hooks';
import { useT } from '../i18n';
import { engine } from '../player/engine';
import { noteUserScroll, setGridTargets } from '../player/sync';
import { buildTimeline, getBassTimeline, getGrid, getTimeline, rowsOf, segAt, type GridModel, type Part, type Strip } from './model';
import { chordLabel, isNoChord, LEVEL_INDEX, segLabels } from './labels';
import { selectChord, selectRange, startEntry } from '../editor/session';
import { flagsForSquares, sheetMarks } from '../sheets/logic';

/** Break long labels at the slash so "Emaj7/D#" can wrap inside a narrow square. */
function withWbr(text: string): ReactNode {
  const i = text.lastIndexOf('/');
  if (i <= 0 || (text.slice(0, i).endsWith('6') && text.slice(i + 1).startsWith('9'))) return text; // keep "6/9" intact
  return (
    <>
      {text.slice(0, i)}
      <wbr />
      {text.slice(i)}
    </>
  );
}

/** Edit mode: a double click on a chord focuses the chord entry. */
function onDoubleClick(e: RMouseEvent): void {
  if (!useApp.getState().editMode || !(e.target as HTMLElement).closest('[data-pi]')) return;
  e.preventDefault();
  startEntry('');
}

function openMenu(partEl: HTMLElement, x: number, y: number): void {
  const seg = Number(partEl.dataset.seg ?? -1);
  if (seg >= 0) useApp.setState({ menu: { seg, x, y } });
}

function applyLoopClasses(sqEls: HTMLElement[], g: GridModel, loop: Loop): void {
  const { a, b } = loop;
  for (let k = 0; k < sqEls.length; k++) {
    const el = sqEls[k];
    const s = g.squares[k];
    if (!el || !s) continue;
    const inLoop = a != null && b != null && s.t0 < b - 1e-3 && s.t1 > a + 1e-3;
    el.classList.toggle('in-loop', inLoop);
    el.classList.toggle('loop-a', a != null && b == null && s.t0 <= a + 1e-3 && s.t1 > a + 1e-3);
  }
}

export function ChordGrid() {
  useRenderCount('ChordGrid');
  const t = useT();
  const analysis = useApp((s) => s.analysis);
  const detail = useApp((s) => s.detail);
  const editMode = useApp((s) => s.editMode);
  const gtMode = useApp((s) => s.gtMode);
  const sel = useApp((s) => (s.editMode ? s.sel : null));
  const ctx = useLabelCtx();
  const phone = useMedia(PHONE_QUERY);
  const scroller = useRef<HTMLDivElement>(null);
  const gridEl = useRef<HTMLDivElement>(null);
  const sqEls = useRef<HTMLElement[]>([]);

  if (!analysis) throw new Error('ChordGrid needs an analysis');
  const segs = getTimeline(analysis, detail, gtMode);
  const bassSegs = getBassTimeline(analysis, detail);
  const g = getGrid(analysis, segs, bassSegs);
  const labels = useMemo(() => segLabels(segs, ctx), [segs, ctx]);
  const bassText = useMemo(() => {
    const names = bassSegs.map((b) => ctx.variant?.bass_line[b.bi] ?? analysis.bass_line?.[b.bi]?.note ?? '');
    return { names, show: names.map((n, i) => i === 0 || n !== names[i - 1]) };
  }, [bassSegs, ctx.variant, analysis]);

  // "Sheet says" row (sheets/): the selected online sheet's chords under the grid, before it is applied.
  const sheetSel = useApp((s) => s.sheetSel);
  const sheetRes = useApp((s) => (s.sheetSel && !s.sheets?.applied ? (s.sheets?.results.find((r) => r.id === s.sheetSel) ?? null) : null));
  const sheetFlags = useApp((s) => s.sheetFlags);
  const sheetMainSegs = useMemo(() => (sheetRes ? buildTimeline(analysis, 0) : null), [analysis, sheetRes]);
  const marks = useMemo(() => {
    if (!sheetRes || !sheetMainSegs) return null;
    const ours = (tm: number) => {
      const k = segAt(sheetMainSegs, tm);
      return k < 0 ? null : chordLabel(sheetMainSegs[k].chord, { ...ctx, notation: 'letters' });
    };
    return sheetMarks(sheetRes.aligned, g.squares, ours, LEVEL_INDEX[ctx.level], ctx.slash, ctx.shift === 0);
  }, [sheetRes, sheetMainSegs, g, ctx]);
  const flagBySq = useMemo(() => (sheetFlags.length ? flagsForSquares(sheetFlags, g.squares) : null), [sheetFlags, g]);
  void sheetSel;

  const barsPerRow = Math.max(1, (phone ? 2 : 4) / (g.beatsPerBar >= 6 ? 2 : 1));
  const rows = useMemo(() => rowsOf(g, barsPerRow), [g, barsPerRow]);
  const cols = barsPerRow * g.beatsPerBar;
  const colStyle: CSSProperties = { gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` };

  const offsets = useMemo(() => {
    const sq = new Int32Array(g.squares.length);
    let o = g.lead?.parts.length ?? 0;
    for (const s of g.squares) {
      sq[s.k] = o;
      o += s.parts.length;
    }
    return { sq, tail: o };
  }, [g]);

  // After every commit: hand the DOM targets to the rAF sync loop and re-apply imperative classes.
  useLayoutEffect(() => {
    const root = gridEl.current;
    if (!root) return;
    const n = g.flat.length;
    const parts: HTMLElement[] = Array.from({ length: n });
    const boxes: HTMLElement[] = Array.from({ length: n });
    const rowsIdx = new Int32Array(n);
    root.querySelectorAll<HTMLElement>('[data-pi]').forEach((el) => {
      const i = Number(el.dataset.pi);
      parts[i] = el;
      boxes[i] = el.closest<HTMLElement>('[data-box]') ?? el;
      rowsIdx[i] = Number(el.closest<HTMLElement>('[data-row]')?.dataset.row ?? -1);
    });
    const rowEls: HTMLElement[] = [];
    root.querySelectorAll<HTMLElement>('[data-row]').forEach((el) => {
      rowEls[Number(el.dataset.row)] = el;
    });
    const sq: HTMLElement[] = [];
    root.querySelectorAll<HTMLElement>('[data-sq]').forEach((el) => {
      sq[Number(el.dataset.sq)] = el;
    });
    sqEls.current = sq;
    setGridTargets({
      starts: Float64Array.from(g.flat, (p) => p.t),
      parts,
      boxes,
      rows: rowsIdx,
      rowEls,
      scroller: scroller.current,
    });
    applyLoopClasses(sq, g, useApp.getState().loop);
  });

  useEffect(() => () => setGridTargets(null), []);

  // Loop shading without re-rendering the grid.
  useEffect(
    () =>
      useApp.subscribe((s, prev) => {
        if (s.loop !== prev.loop) applyLoopClasses(sqEls.current, g, s.loop);
      }),
    [g],
  );

  // ---------------------------------------------------------------- pointer interactions
  const drag = useRef<{ pi: number; sq: number; cur: number; moved: boolean; touch: boolean; x: number; y: number } | null>(null);
  const longPress = useRef<number>(0);

  const paintSel = (a: number, b: number) => {
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    sqEls.current.forEach((el, k) => el?.classList.toggle('sel', k >= lo && k <= hi));
  };
  const clearSel = () => sqEls.current.forEach((el) => el?.classList.remove('sel'));
  const cancelLongPress = () => {
    if (longPress.current) window.clearTimeout(longPress.current);
    longPress.current = 0;
  };

  const onPointerDown = (e: RPointerEvent) => {
    const partEl = (e.target as HTMLElement).closest<HTMLElement>('[data-pi]');
    if (!partEl || (e.pointerType === 'mouse' && e.button !== 0)) return;
    const sqEl = partEl.closest<HTMLElement>('[data-sq]');
    const sq = sqEl ? Number(sqEl.dataset.sq) : -1;
    const touch = e.pointerType !== 'mouse';
    drag.current = { pi: Number(partEl.dataset.pi), sq, cur: sq, moved: false, touch, x: e.clientX, y: e.clientY };
    // Mouse drags keep receiving events (and the release) even outside the grid.
    if (!touch) {
      try {
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      } catch {
        /* pointer already gone */
      }
    }
    if (touch) {
      cancelLongPress();
      const { clientX: x, clientY: y } = e;
      longPress.current = window.setTimeout(() => {
        longPress.current = 0;
        drag.current = null; // the release must not seek
        openMenu(partEl, x, y);
      }, 550);
    }
  };

  const onPointerMove = (e: RPointerEvent) => {
    const d = drag.current;
    if (!d) return;
    if (d.touch) {
      if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 10) {
        cancelLongPress();
        drag.current = null;
      }
      return;
    }
    if (d.sq >= 0) trackSquare(d, e.clientX, e.clientY);
  };

  /** Mouse drag: follow the square under the pointer (the drag lives from pointerdown to pointerup). */
  const trackSquare = (d: NonNullable<typeof drag.current>, x: number, y: number) => {
    const el = document.elementFromPoint(x, y)?.closest<HTMLElement>('[data-sq]');
    if (!el) return;
    const k = Number(el.dataset.sq);
    if (k !== d.cur) {
      d.cur = k;
      d.moved = k !== d.sq;
      if (d.moved) paintSel(d.sq, k);
      else clearSel();
    }
  };

  const onPointerUp = (e: RPointerEvent) => {
    cancelLongPress();
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    if (!d.touch && d.sq >= 0) trackSquare(d, e.clientX, e.clientY);
    if (d.moved) {
      const lo = g.squares[Math.min(d.sq, d.cur)];
      const hi = g.squares[Math.max(d.sq, d.cur)];
      clearSel();
      if (lo && hi) {
        if (useApp.getState().editMode) selectRange(lo.t0, hi.t1); // edit mode: drag selects beats
        else {
          setLoop(lo.t0, hi.t1);
          engine.seek(lo.t0);
        }
      }
      return;
    }
    const p = g.flat[d.pi];
    if (p && useApp.getState().editMode) {
      // edit mode: a click selects the chord (no seek, so a playing bar loop keeps going)
      if (p.seg >= 0) selectChord(segs[p.seg].chord);
      else if (p.sq >= 0) selectRange(g.squares[p.sq].t0, g.squares[p.sq].t1);
      return;
    }
    if (p) engine.seek(p.t);
  };

  const onPointerCancel = () => {
    cancelLongPress();
    drag.current = null;
    clearSel();
  };

  const onContextMenu = (e: RMouseEvent) => {
    const partEl = (e.target as HTMLElement).closest<HTMLElement>('[data-pi]');
    if (!partEl) return;
    e.preventDefault();
    cancelLongPress();
    drag.current = null;
    openMenu(partEl, e.clientX, e.clientY);
  };

  // ---------------------------------------------------------------- rendering helpers
  const renderPart = (p: Part, pi: number, rowStart: boolean) => {
    const grow = p.f1 - p.f0;
    if (p.seg < 0) return <div key={pi} className="part empty" data-pi={pi} data-seg={-1} style={{ flexGrow: grow }} />;
    const seg = segs[p.seg];
    const c = seg.chord;
    const text = labels.text[p.seg];
    const draw = p.isStart && labels.show[p.seg];
    const cont = !p.isStart && rowStart;
    const cls = ['part'];
    if (seg.passing) cls.push('passing');
    if (draw) cls.push('start');
    if (c.conf < 0.5) cls.push('lowconf');
    if (isNoChord(c)) cls.push('nc');
    if (grow < 0.99) cls.push('split');
    if (grow <= 0.5) cls.push('narrow');
    const longest = Math.max(...text.split('/').map((s) => s.length)); // a slash chord wraps at the slash
    if (longest >= 5) cls.push(longest >= 6 ? 'len6' : 'len5');
    if (c.suggestion) cls.push('suggest');
    if (c.edited) cls.push('edited');
    if (gtMode && c.edited && c.conf < 0.5) cls.push('unsure'); // an unsure spot of the imported chart
    if (sel?.kind === 'chord' && sel.id === c.id) cls.push('esel');
    const style = { flexGrow: grow, '--conf': c.conf.toFixed(2) } as CSSProperties;
    const title = `${text} · ${Math.round(c.conf * 100)}%${seg.passing ? ` · ${t('passing')}` : ''}${c.conf < 0.5 ? ` · ${t('lowConfidence')}` : ''}${
      c.suggestion ? ` · ${t('suggestion')}` : ''
    }${c.edited ? ` · ${t('edited')}` : ''}`;
    return (
      <div key={pi} className={cls.join(' ')} data-pi={pi} data-seg={p.seg} style={style} title={title}>
        {draw ? (
          <span className="lbl">{withWbr(text)}</span>
        ) : cont ? (
          <span className="lbl cont">{withWbr(text)}</span>
        ) : (
          <span className="dot">·</span>
        )}
      </div>
    );
  };

  const renderBass = (p: Part, key: number) => {
    const grow = p.f1 - p.f0;
    if (p.seg < 0) return <div key={key} className="bpart empty" style={{ flexGrow: grow }} />;
    const b = bassSegs[p.seg];
    const cls = ['bpart'];
    if (b.tone) cls.push('tone');
    else if (b.passing) cls.push('passing');
    const draw = p.isStart && bassText.show[p.seg];
    return (
      <div key={key} className={cls.join(' ')} style={{ flexGrow: grow }}>
        {draw ? bassText.names[p.seg] : <span className="dot">·</span>}
      </div>
    );
  };

  const renderStrip = (st: Strip, row: number, base: number) => (
    <div className="row strip-row" data-row={row} key={`strip-${st.kind}`}>
      <div className="strip" data-box="" title={t('freeTime')}>
        <span className="strip-tag">{t('freeTime')}</span>
        <div className="parts">{st.parts.map((p, j) => renderPart(p, base + j, j === 0))}</div>
      </div>
      <div className="strip-bass">{st.bass.map((p, j) => renderBass(p, j))}</div>
    </div>
  );

  let rowIdx = 0;
  const leadRow = g.lead ? renderStrip(g.lead, rowIdx++, 0) : null;
  const barRows = rows.map((bars) => {
    const r = rowIdx++;
    let first = true;
    return (
      <div className="row" data-row={r} key={`r${bars[0].i}`}>
        <div className="cells" style={colStyle}>
          {bars.map((bar) =>
            bar.squares.map((k, j) => {
              const s = g.squares[k];
              const rowStart = first;
              first = false;
              const kcName = bar.keyChange ? (ctx.variant?.key_names[bar.keySeg] ?? bar.keyChange.to) : null;
              const inSel = sel?.kind === 'range' && s.t0 < sel.t1 - 1e-3 && s.t1 > sel.t0 + 1e-3;
              const band = inSel
                ? { left: Math.max(0, (sel.t0 - s.t0) / (s.t1 - s.t0)), right: Math.min(1, (sel.t1 - s.t0) / (s.t1 - s.t0)) }
                : null;
              const flag = flagBySq?.[k] ?? null;
              const diff = marks?.[k]?.differs ?? false;
              return (
                <div
                  key={k}
                  className={`sq${j === 0 ? ' bar-start' : ''}${s.free ? ' free' : ''}${inSel ? ' esel-sq' : ''}${diff ? ' sheet-diff' : ''}${flag ? ' sheet-flag' : ''}`}
                  data-box=""
                  data-sq={k}
                  style={s.span > 1 ? { gridColumn: `span ${s.span}` } : undefined}
                  title={flag ? t('flaggedSays', { detected: flag.detected, sheet: flag.sheet }) : undefined}
                >
                  {band && (
                    <span className="esel-band" style={{ left: `${band.left * 100}%`, width: `${(band.right - band.left) * 100}%` }} />
                  )}
                  {j === 0 && <span className="barno">{bar.i + 1}</span>}
                  {j === 0 && kcName && (
                    <span className="kc" title={t('keyChangeAt', { bar: bar.i + 1, name: kcName })}>
                      → {kcName}
                    </span>
                  )}
                  {j === 0 && bar.section && <span className="section">{bar.section}</span>}
                  <div className="parts">{s.parts.map((p, pj) => renderPart(p, offsets.sq[k] + pj, rowStart && pj === 0))}</div>
                </div>
              );
            }),
          )}
        </div>
        <div className="brow" style={colStyle}>
          {bars.map((bar) =>
            bar.squares.map((k, j) => {
              const s = g.squares[k];
              return (
                <div
                  key={k}
                  className={`bsq${j === 0 ? ' bar-start' : ''}`}
                  style={s.span > 1 ? { gridColumn: `span ${s.span}` } : undefined}
                >
                  {s.bass.map((p, pj) => renderBass(p, pj))}
                </div>
              );
            }),
          )}
        </div>
        {marks && (
          <div className={`srow${ctx.shift !== 0 ? ' greyed' : ''}`} style={colStyle} aria-label={t('sheetSays')}>
            {bars.map((bar, bi) =>
              bar.squares.map((k, j) => {
                const s = g.squares[k];
                const m = marks[k];
                return (
                  <div
                    key={k}
                    className={`ssq${j === 0 ? ' bar-start' : ''}${m.differs ? ' diff' : ''}`}
                    style={s.span > 1 ? { gridColumn: `span ${s.span}` } : undefined}
                    title={m.label ? `${t('sheetSays')}: ${m.label}` : undefined}
                  >
                    {bi === 0 && j === 0 && (
                      <span className="srow-tag" onClick={() => useApp.setState({ sheetsOpen: true })}>
                        {t('sheetSays')}
                        {ctx.shift !== 0 && <small> · {t('sheetSaysOriginalKey')}</small>}
                      </span>
                    )}
                    {m.label == null ? '' : m.show ? m.label : <span className="dot">·</span>}
                  </div>
                );
              }),
            )}
          </div>
        )}
      </div>
    );
  });
  const tailRow = g.tail ? renderStrip(g.tail, rowIdx++, offsets.tail) : null;

  return (
    <div className="grid-scroll" ref={scroller} onWheel={noteUserScroll} onTouchMove={noteUserScroll}>
      <div
        className={`grid${editMode ? ' editing' : ''}${gtMode ? ' gt' : ''}`}
        ref={gridEl}
        onDoubleClick={onDoubleClick}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerCancel}
        onContextMenu={onContextMenu}
      >
        {leadRow}
        {barRows}
        {tailRow}
      </div>
    </div>
  );
}
