import { useState, type ReactNode } from 'react';
import {
  applyPreset,
  currentPreset,
  DETAIL_STOPS,
  nudgeRate,
  nudgeTranspose,
  PRESETS,
  RATE_MAX,
  RATE_MIN,
  setDetail,
  setRate,
  TRANSPOSE_MAX,
  TRANSPOSE_MIN,
  useApp,
  type Acc,
  type AudioKey,
  type Lang,
  type Level,
  type Notation,
  type Theme,
} from '../app/store';
import { api } from '../api/client';
import { useT, type StringKey } from '../i18n';
import { globalKeySegment } from '../grid/model';
import { keyName, signed } from '../grid/labels';
import { SheetsAutoToggle } from '../sheets/SheetPanel';
import { setAudioKey } from '../player/mixActions';

// ---------------------------------------------------------------- small building blocks

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { v: T; label: ReactNode; title?: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.v}
          role="radio"
          aria-checked={value === o.v}
          className={value === o.v ? 'on' : ''}
          title={o.title}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => onChange(o.v)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Group({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return (
    <div className={`ctl-group${className ? ` ${className}` : ''}`}>
      <span className="ctl-label">{label}</span>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------- transpose / play in

export function useKeyNames() {
  const analysis = useApp((s) => s.analysis);
  const display = useApp((s) => s.display);
  const acc = useApp((s) => s.acc);
  const gi = analysis ? globalKeySegment(analysis) : 0;
  return (shift: number) => (analysis ? keyName(display, analysis, gi, shift, acc) : '');
}

export function TransposeControl() {
  const t = useT();
  const transpose = useApp((s) => s.transpose);
  const names = useKeyNames();
  return (
    <Group label={t('transpose')}>
      <div className="stepper">
        <button onMouseDown={(e) => e.preventDefault()} onClick={() => nudgeTranspose(-1)} disabled={transpose <= TRANSPOSE_MIN} aria-label={t('transposeDown')} title={`${t('transposeDown')} (−)`}>
          −
        </button>
        <span className="stepper-val" aria-live="polite">
          <b>{names(transpose)}</b>
          {transpose !== 0 && <small> {signed(transpose)}</small>}
        </span>
        <button onMouseDown={(e) => e.preventDefault()} onClick={() => nudgeTranspose(1)} disabled={transpose >= TRANSPOSE_MAX} aria-label={t('transposeUp')} title={`${t('transposeUp')} (+)`}>
          +
        </button>
      </div>
    </Group>
  );
}

function AudioKeyIcon() {
  // a note with an up and a down arrow: "the recording moves with the key"
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <circle cx="8" cy="17" r="3" fill="currentColor" />
      <path d="M11 17V5l5 2" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M19 7l-3 4h6l-3-4zm0 11l-3-4h6l-3 4z" fill="currentColor" />
    </svg>
  );
}

/**
 * "Audio: Original / Transposed" (2026-11-04): does the recording follow Transpose? Always shown; only
 * meaningful when Transpose is not 0 (a grey hint says so). `compact` = the icon-only toggle for phones.
 */
export function AudioKeyControl({ compact }: { compact?: boolean }) {
  const t = useT();
  const transpose = useApp((s) => s.transpose);
  const audioKey = useApp((s) => s.audioKey);
  const busy = useApp((s) => s.shiftBusy);
  const audioBase = useApp((s) => s.audioBase);
  const localFile = api.mock && !!audioBase && audioBase.startsWith('blob:');
  const idle = transpose === 0;
  const help = t(`audioKey_${audioKey}_help` as StringKey);
  const hint = busy ? t('audioKeyPreparing') : localFile ? t('mixMockFile') : idle ? t('audioKeyNoTranspose') : '';
  if (compact) {
    const on = audioKey === 'transposed';
    return (
      <button
        className={`pill audiokey-btn${on ? ' on' : ''}${idle ? ' idle' : ''}`}
        onClick={() => setAudioKey(on ? 'original' : 'transposed')}
        aria-pressed={on}
        aria-label={`${t('audioKey')}: ${t(`audioKey_${audioKey}` as StringKey)}`}
        title={`${t('audioKey')}: ${t(`audioKey_${audioKey}` as StringKey)} — ${hint || help}`}
        disabled={localFile}
      >
        <AudioKeyIcon />
        {busy && <span className="sheets-spin" aria-hidden="true" />}
      </button>
    );
  }
  return (
    <Group label={t('audioKey')} className={`audiokey${idle ? ' idle' : ''}`}>
      <span title={t('audioKeyTitle')}>
        <Segmented<AudioKey>
          label={t('audioKey')}
          value={audioKey}
          onChange={(v) => setAudioKey(v)}
          options={(['original', 'transposed'] as const).map((v) => ({ v, label: t(`audioKey_${v}`), title: t(`audioKey_${v}_help`) }))}
        />
      </span>
      {hint && (
        <span className={`audiokey-hint${busy ? ' busy' : ''}`} aria-live="polite">
          {busy && <span className="sheets-spin" aria-hidden="true" />}
          {hint}
        </span>
      )}
    </Group>
  );
}

export function PlayInControl() {
  const t = useT();
  const transpose = useApp((s) => s.transpose);
  const playIn = useApp((s) => s.playIn);
  const names = useKeyNames();
  const opts: number[] = [];
  for (let p = -6; p <= 5; p++) opts.push(p);
  return (
    <Group label={t('playIn')}>
      <select
        className="select"
        value={playIn}
        onChange={(e) => useApp.setState({ playIn: Number(e.target.value) })}
        aria-label={t('playIn')}
      >
        {opts.map((p) => (
          <option key={p} value={p}>
            {p === 0 ? `${t('sameKey')} (${names(transpose)})` : `${names(transpose + p)} · ${t('kbdShort', { n: signed(-p) })}`}
          </option>
        ))}
      </select>
    </Group>
  );
}

// ---------------------------------------------------------------- tempo

function parseTempo(text: string, bpm: number): number | null {
  const m = /^\s*(\d+(?:[.,]\d+)?)\s*(%|bpm)?\s*$/i.exec(text);
  if (!m) return null;
  const v = Number(m[1].replace(',', '.'));
  if (!Number.isFinite(v) || v <= 0) return null;
  return m[2]?.toLowerCase() === 'bpm' ? v / bpm : v / 100;
}

function TempoField({ value, onCommit, label, suffix, width }: { value: string; onCommit: (s: string) => void; label: string; suffix: string; width: number }) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <label className="num-field">
      <input
        inputMode="decimal"
        aria-label={label}
        style={{ width: `calc(${width}ch + 16px)` }}
        value={draft ?? value}
        onFocus={(e) => e.currentTarget.select()}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          if (draft != null && draft !== value) onCommit(draft);
          setDraft(null);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur();
          if (e.key === 'Escape') {
            setDraft(null);
            e.currentTarget.blur();
          }
        }}
      />
      <span>{suffix}</span>
    </label>
  );
}

export function TempoControl() {
  const t = useT();
  const rate = useApp((s) => s.rate);
  const bpm = useApp((s) => s.analysis?.tempo.bpm ?? 0);
  const pct = Math.round(rate * 100);
  const commit = (unit: '%' | 'bpm') => (s: string) => {
    const hasUnit = /%|bpm/i.test(s);
    const r = parseTempo(hasUnit ? s : `${s}${unit}`, bpm || 60);
    if (r != null) setRate(r);
  };
  return (
    <Group label={t('tempo')}>
      <div className="stepper tempo">
        <button onMouseDown={(e) => e.preventDefault()} onClick={() => nudgeRate(-0.05)} disabled={rate <= RATE_MIN} title={`${t('tempoDown')} (Shift −)`} aria-label={t('tempoDown')}>
          −5%
        </button>
        <span className="tempo-fields" title={t('tempoInputHint')}>
          <TempoField value={String(pct)} onCommit={commit('%')} label={`${t('tempo')} %`} suffix="%" width={4} />
          {bpm > 0 && <TempoField value={String(Math.round(bpm * rate))} onCommit={commit('bpm')} label={t('bpm')} suffix={t('bpm')} width={4} />}
        </span>
        <button onMouseDown={(e) => e.preventDefault()} onClick={() => nudgeRate(0.05)} disabled={rate >= RATE_MAX} title={`${t('tempoUp')} (Shift +)`} aria-label={t('tempoUp')}>
          +5%
        </button>
      </div>
    </Group>
  );
}

// ---------------------------------------------------------------- detail / level / slash / notation

export function DetailControl() {
  const t = useT();
  const detail = useApp((s) => s.detail);
  const level = useApp((s) => s.level);
  const preset = currentPreset(level, detail);
  return (
    <Group label={t('detail')} className="detail">
      <div className="detail-body">
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={detail}
          list="detail-stops"
          aria-label={t('detail')}
          aria-valuetext={detail.toFixed(2)}
          onChange={(e) => setDetail(Number(e.target.value))}
          onKeyDown={(e) => {
            if (e.key === ' ') e.preventDefault();
          }}
        />
        <datalist id="detail-stops">
          {DETAIL_STOPS.map((s) => (
            <option key={s} value={s} />
          ))}
        </datalist>
        <div className="presets">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              className={`chip${preset === p.id ? ' on' : Math.abs(p.detail - detail) < 1e-6 ? ' half' : ''}`}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => applyPreset(p.id)}
              title={`${t(`preset_${p.id}` as StringKey)}: ${t('level')} ${t(`level_${p.level}` as StringKey)}, ${t('detail')} ${p.detail}`}
            >
              {t(`preset_${p.id}` as StringKey)}
            </button>
          ))}
        </div>
      </div>
    </Group>
  );
}

export function LevelControl() {
  const t = useT();
  const level = useApp((s) => s.level);
  return (
    <Group label={t('level')}>
      <Segmented<Level>
        label={t('level')}
        value={level}
        onChange={(v) => useApp.setState({ level: v })}
        options={(['simple', 'standard', 'full'] as const).map((v) => ({ v, label: t(`level_${v}`) }))}
      />
    </Group>
  );
}

export function SlashToggle() {
  const t = useT();
  const slash = useApp((s) => s.slash);
  return (
    <label className="toggle">
      <input type="checkbox" checked={slash} onChange={(e) => useApp.setState({ slash: e.target.checked })} />
      <span>{t('slash')}</span>
    </label>
  );
}

export function NotationSwitch({ compact }: { compact?: boolean }) {
  const t = useT();
  const notation = useApp((s) => s.notation);
  return (
    <Segmented<Notation>
      label={t('notation')}
      value={notation}
      onChange={(v) => useApp.setState({ notation: v })}
      options={[
        { v: 'letters', label: compact ? 'A B C' : t('notation_letters'), title: t('notation_letters') },
        { v: 'nashville', label: compact ? '1 2 3' : t('notation_nashville'), title: t('notation_nashville') },
        { v: 'roman', label: compact ? 'I ii' : t('notation_roman'), title: t('notation_roman') },
      ]}
    />
  );
}

export function AccControl() {
  const t = useT();
  const acc = useApp((s) => s.acc);
  return (
    <Group label={t('accidentals')}>
      <Segmented<Acc>
        label={t('accidentals')}
        value={acc}
        onChange={(v) => useApp.setState({ acc: v })}
        options={(['auto', 'sharp', 'flat'] as const).map((v) => ({ v, label: t(`acc_${v}`) }))}
      />
    </Group>
  );
}

export function LangControl() {
  const t = useT();
  const lang = useApp((s) => s.lang);
  return (
    <Group label={t('language')}>
      <Segmented<Lang>
        label={t('language')}
        value={lang}
        onChange={(v) => useApp.setState({ lang: v })}
        options={[
          { v: 'en', label: 'EN' },
          { v: 'zh-TW', label: '中文' },
        ]}
      />
    </Group>
  );
}

export function ThemeControl() {
  const t = useT();
  const theme = useApp((s) => s.theme);
  return (
    <Group label={t('theme')}>
      <Segmented<Theme>
        label={t('theme')}
        value={theme}
        onChange={(v) => useApp.setState({ theme: v })}
        options={(['auto', 'light', 'dark'] as const).map((v) => ({ v, label: t(`theme_${v}`) }))}
      />
    </Group>
  );
}

/** Desktop dock: every M3 control in two rows. */
export function ControlsDock() {
  const t = useT();
  return (
    <div className="controls">
      <div className="ctl-row">
        <TransposeControl />
        <AudioKeyControl />
        <PlayInControl />
        <TempoControl />
      </div>
      <div className="ctl-row">
        <DetailControl />
        <LevelControl />
        <SlashToggle />
        <AccControl />
        <SheetsAutoToggle />
      </div>
      <div className="shortcuts-hint">{t('shortcuts')}</div>
    </div>
  );
}

/** Phone: compact row above the bottom sheet (§8.2). */
export function ControlsCompact() {
  const t = useT();
  const transpose = useApp((s) => s.transpose);
  const rate = useApp((s) => s.rate);
  const names = useKeyNames();
  return (
    <div className="controls-compact">
      <div className="stepper">
        <button onClick={() => nudgeTranspose(-1)} aria-label={t('transposeDown')}>
          −
        </button>
        <span className="stepper-val">
          <b>{names(transpose)}</b>
          {transpose !== 0 && <small> {signed(transpose)}</small>}
        </span>
        <button onClick={() => nudgeTranspose(1)} aria-label={t('transposeUp')}>
          +
        </button>
      </div>
      <AudioKeyControl compact />
      <button className="pill rate-pill" onClick={() => useApp.setState({ sheetOpen: true })} aria-label={t('tempo')}>
        {Math.round(rate * 100)}%
      </button>
      <button className="pill settings-pill" onClick={() => useApp.setState({ sheetOpen: true })} aria-label={t('settings')} title={t('settings')}>
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path d="M4 7h10M18 7h2M4 17h2M10 17h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          <circle cx="16" cy="7" r="2.2" fill="none" stroke="currentColor" strokeWidth="2" />
          <circle cx="8" cy="17" r="2.2" fill="none" stroke="currentColor" strokeWidth="2" />
        </svg>
      </button>
    </div>
  );
}

const closeSheet = () => useApp.setState({ sheetOpen: false });

/** Phone bottom sheet with every setting. */
export function SettingsSheet() {
  const t = useT();
  const open = useApp((s) => s.sheetOpen);
  const noteNames = useApp((s) => s.noteNames);
  if (!open) return null;
  return (
    <>
      <div className="sheet-backdrop" onClick={closeSheet} />
      <div className="sheet settings-sheet" role="dialog" aria-label={t('settings')}>
        <div className="sheet-head">
          <b>{t('settings')}</b>
          <button className="icon-btn" onClick={closeSheet} aria-label={t('close')}>
            ×
          </button>
        </div>
        <TransposeControl />
        <AudioKeyControl />
        <PlayInControl />
        <TempoControl />
        <DetailControl />
        <LevelControl />
        <SlashToggle />
        <Group label={t('notation')}>
          <NotationSwitch />
        </Group>
        <AccControl />
        <label className="toggle">
          <input
            type="checkbox"
            checked={noteNames}
            onChange={(e) => useApp.setState({ noteNames: e.target.checked })}
          />
          <span>{t('noteNames')}</span>
        </label>
        <SheetsAutoToggle />
        <LangControl />
        <ThemeControl />
      </div>
    </>
  );
}
