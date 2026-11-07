"""WP8 orchestration (DESIGN §5, steps 1–11): source → analysis.json + display.json.

M1 scope: one chord model (lv-chordia on the mix and the harmonic stem), the bass tracker, the
calibrated HSMM (main + detail layers) and post-processing. M5 adds consonance-ACE (same two inputs;
fusion picks up ``ace@…`` evidence automatically; decode states saved before M5 list only lv, so
``load_state`` also loads any cached ``evidence/ace@*.npz``) and the piano transcription below.
Sections and the repetition prior are still to come.

Piano transcription (M5, §5.6/§5.7): ``analyze`` transcribes stems/other.flac once
(``notes.piano_amt``, cached as cache/<key>/notes.npz) when ``DecoderParams.use_sym`` or
``amt_bass`` is on. At decode time (``_Run.build_all``, so ``redecode`` can switch both) the notes
become the symbolic chroma (``notes.symbolic.cell_chroma`` → x for the ``sym`` source, dch as the
phantom rule's onsets) and the left-hand bass fallback (``notes.bass.with_left_hand``). The decode
state keeps the stem-only bass track; the notes are re-read from notes.npz.

Re-decoding (M2): ``analyze`` saves everything the decoder needs — the final beat grid (after the
octave check), the bass track, the first key estimate and the HCDF peaks — as
``cache/<key>/decode_state.v1.npz`` (no pickles; the chord-model evidence stays in
``evidence/*.npz``). ``redecode(key, params, region)`` re-runs steps 3–11 (fusion, pruning, HSMM,
layers, post-processing, key refinement, spelling, voicing, JSON) from that state in a few
seconds, without audio, stems or GPU. With ``region=(t0, t1)`` it also re-scans that region
"more sensitively" (§5.11) and returns the new chords there as suggestions.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import ENGINE_VERSION, config, ingest, rhythm, schema, separation
from .decode import evidence as EV
from .decode import hsmm, layers, post
from .decode import params as DP
from .notes import bass as bass_mod
from .notes import piano_amt as AMT
from .notes import symbolic as SYMB
from .theory import display as DISP
from .theory import harte as H
from .theory import key as KEY
from .theory import numbers as NUM
from .theory import simplify as SM
from .theory import spelling as S
from .theory import voicing as VOI
from .theory.vocab import candidates
from .types import (BassTrack, ChordEvidence, DecoderParams, KeyChange, KeyTrack, NoteEvents, Progress,
                    RhythmGrid, Segment)

KEY_REDECODE_BEATS = 8           # §5 step 10
STATE_NAME = "decode_state.v1.npz"
STATE_VERSION = 1


def _noop(stage: str, pct: float, msg: str) -> None:
    pass


def params_from(overrides: dict | DecoderParams | None) -> DecoderParams:
    if isinstance(overrides, DecoderParams):
        return overrides
    p = DecoderParams()
    return replace(p, **overrides) if overrides else p


def params_hash(p: DecoderParams, mode: str) -> str:
    blob = json.dumps({"engine": ENGINE_VERSION, "mode": mode, "p": asdict(p)}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------------------------
# Decoding loop (§5 steps 3–10)
# ---------------------------------------------------------------------------------------------

class _Run:
    """Holds the per-song state so the octave and key loops can re-run parts of it."""

    def __init__(self, grid, evid, bass, key, vocab, p, hcdf, notes: NoteEvents | None = None):
        self.grid, self.evid, self.bass, self.key = grid, evid, bass, key   # bass: the stem's track
        self.vocab, self.p, self.hcdf = vocab, p, hcdf
        self.notes = notes                  # piano AMT notes (§5.7), or None
        self.ev_all = None
        self.sym_x = self.onsets = None
        self.bass_eff = bass

    def build_all(self):
        """Full-vocabulary evidence on the current grid (+ the M5 symbolic chroma and left hand)."""
        C_all = np.arange(len(self.vocab.cands))
        self.sym_x = self.onsets = None
        if self.notes is not None and self.p.use_sym:
            self.sym_x, self.onsets = SYMB.cell_chroma(self.notes, self.grid)
        use_lh = self.notes is not None and self.p.amt_bass and self.bass is not None
        self.bass_eff = bass_mod.with_left_hand(self.bass, self.notes) if use_lh else self.bass
        self.ev_all = EV.build(self.grid, self.evid, self.sym_x, self.bass_eff, self.vocab, C_all, self.p)

    def sub(self, C):
        """Evidence over a candidate subset, sliced from ``ev_all`` (same numbers as a rebuild:
        the calibrated log-softmax runs over the FULL vocabulary, §5.9 step 3)."""
        return EV.subset(self.ev_all, C)

    def first_main(self):
        C0 = EV.prune(self.ev_all, self.vocab, None, self.p)
        return hsmm.decode(self.sub(C0), self.grid, self.key, self.vocab, self.p, "main", hcdf=self.hcdf)

    def layers(self, main0):
        C = EV.prune(self.ev_all, self.vocab, main0, self.p)
        ev = self.sub(C)
        detail = hsmm.decode(ev, self.grid, self.key, self.vocab, self.p, "detail", hcdf=self.hcdf)
        main = hsmm.decode(ev, self.grid, self.key, self.vocab, self.p, "main", hcdf=self.hcdf)
        segs = layers.build(detail, main, ev, self.grid, self.key, self.vocab, self.p, hcdf=self.hcdf)
        segs = post.finalize(segs, self.bass_eff, self.key, self.grid, self.vocab, ev, self.p, onsets=self.onsets)
        return segs, ev


def _key_changed_beats(a: KeyTrack, b: KeyTrack) -> int:
    n = min(len(a.per_beat), len(b.per_beat))
    return int(np.sum(np.asarray(a.per_beat[:n]) != np.asarray(b.per_beat[:n])))


def _finish(run: _Run, main0):
    """§5 steps 6–10 after the first main decode: layers, post, key refinement (+ one re-run)."""
    segs, ev = run.layers(main0)
    key = run.key
    key2 = KEY.refine(key, segs, run.grid, None)                      # §5 step 10
    if _key_changed_beats(key, key2) >= KEY_REDECODE_BEATS:
        key = run.key = key2
        segs, ev = run.layers(main0)
    else:
        key = run.key = key2
    return segs, ev, key


# ---------------------------------------------------------------------------------------------
# Decode state (cache/<key>/decode_state.v1.npz)
# ---------------------------------------------------------------------------------------------

@dataclass
class DecodeState:
    """Everything steps 3–11 need, without audio or GPU."""
    dir: Path
    key: str
    meta: dict                     # asset info for the JSON, "stems", "mode", "runtime_s", "evidence"
    grid: RhythmGrid
    bass: BassTrack
    key0: KeyTrack                 # KEY.estimate on the final grid (before refinement)
    hcdf: np.ndarray
    evid: list[ChordEvidence] = field(default_factory=list)
    notes: NoteEvents | None = None     # piano AMT notes, re-read from notes.npz (not stored in the state)


_GRID_ARRAYS = ("beats", "beat_conf", "bar_of_beat", "beat_in_bar", "subdiv", "cell_t", "cell_beat", "cell_pos")


def save_state(st: DecodeState) -> Path:
    arrays: dict[str, np.ndarray] = {f"grid__{k}": np.asarray(getattr(st.grid, k)) for k in _GRID_ARRAYS}
    arrays["bass__frame_t"] = np.asarray(st.bass.frame_t, dtype=np.float32)
    arrays["bass__salience"] = np.asarray(st.bass.salience, dtype=np.float32)
    arrays["bass__energy_db"] = np.asarray(st.bass.energy_db, dtype=np.float32)
    arrays["key__per_beat"] = np.asarray(st.key0.per_beat, dtype=np.int64)
    arrays["hcdf"] = np.asarray(st.hcdf if st.hcdf is not None else np.zeros(0), dtype=float)
    g = st.grid
    meta = {
        "version": STATE_VERSION, "engine": ENGINE_VERSION, "cache_key": st.key, "meta": st.meta,
        "grid": {"bars": [list(map(int, b)) for b in g.bars], "meter": [list(map(int, m)) for m in g.meter],
                 "bpm": float(g.bpm), "ibi_ref": float(g.ibi_ref), "free": [list(map(float, f)) for f in g.free],
                 "edited": bool(g.edited)},
        "bass_notes": [[float(a), float(b), int(m), float(c), str(s)] for a, b, m, c, s in st.bass.notes],
        "key": {"global_tonic": int(st.key0.global_tonic), "global_mode": st.key0.global_mode,
                "global_conf": float(st.key0.global_conf),
                "changes": [{"t": float(c.t), "bar": int(c.bar), "from_key": int(c.from_key),
                             "to_key": int(c.to_key), "kind": c.kind} for c in st.key0.changes]},
    }
    arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
    path = Path(st.dir) / STATE_NAME
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)
    return path


def state_dir(target: str | Path) -> Path:
    p = Path(target)
    if p.is_dir():
        return p
    if p.name == STATE_NAME:
        return p.parent
    return config.CACHE_DIR / str(target)


def load_state(target: str | Path) -> DecodeState:
    """The decode state of a cache key (or a cache folder); evidence is re-read from evidence/."""
    from .models.base import load_evidence
    d = state_dir(target)
    path = d / STATE_NAME
    if not path.exists():
        raise FileNotFoundError(f"no decode state in {d}: run analyze first")
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arr = {k: z[k] for k in z.files if k != "meta"}
    gm = meta["grid"]
    grid = RhythmGrid(beats=arr["grid__beats"], beat_conf=arr["grid__beat_conf"],
                      bar_of_beat=arr["grid__bar_of_beat"], beat_in_bar=arr["grid__beat_in_bar"],
                      bars=[tuple(b) for b in gm["bars"]], meter=[tuple(m) for m in gm["meter"]],
                      bpm=gm["bpm"], ibi_ref=gm["ibi_ref"], subdiv=arr["grid__subdiv"],
                      free=[tuple(f) for f in gm["free"]], cell_t=arr["grid__cell_t"],
                      cell_beat=arr["grid__cell_beat"], cell_pos=arr["grid__cell_pos"], edited=gm["edited"])
    bass = BassTrack(frame_t=arr["bass__frame_t"].astype(float), salience=arr["bass__salience"].astype(float),
                     energy_db=arr["bass__energy_db"].astype(float),
                     notes=[(a, b, int(m), c, s) for a, b, m, c, s in meta["bass_notes"]])
    km = meta["key"]
    key0 = KeyTrack(km["global_tonic"], km["global_mode"], km["global_conf"], arr["key__per_beat"],
                    [KeyChange(**c) for c in km["changes"]])
    m = meta["meta"]
    files = list(m.get("evidence", []))
    files += sorted(f.name for f in (d / "evidence").glob(f"ace@*.v{_ace_version()}.npz") if f.name not in files)
    evid = [load_evidence(d / "evidence" / f) for f in files]          # states saved before M5 list only lv
    return DecodeState(dir=d, key=meta["cache_key"], meta=m, grid=grid, bass=bass, key0=key0, hcdf=arr["hcdf"],
                       evid=evid, notes=AMT.load_cached(d))


def _ace_version() -> int:
    from .models.ace import Ace
    return int(Ace.adapter_version)


def run_from_state(st: DecodeState, p: DecoderParams) -> _Run:
    run = _Run(st.grid, st.evid, st.bass, st.key0, candidates(), p, st.hcdf if len(st.hcdf) else None,
               notes=st.notes)
    run.build_all()
    return run


def decode_state(st: DecodeState, p: DecoderParams):
    """Steps 3–10 from a saved state: (run, segments, evidence, refined key)."""
    run = run_from_state(st, p)
    main0 = run.first_main()
    segs, ev, key = _finish(run, main0)
    return run, segs, ev, key


# ---------------------------------------------------------------------------------------------
# JSON assembly (§5.12 steps 7–12, §5.13)
# ---------------------------------------------------------------------------------------------

def _key_parts(k: int) -> tuple[str, str]:
    name = S.key_name(k)
    minor = S.key_is_minor(k)
    return (name[:-1] if minor and name.endswith("m") else name), ("minor" if minor else "major")


def _beat_of_time(grid: RhythmGrid, t: float) -> int:
    b = int(np.searchsorted(grid.beats, t + 1e-6) - 1)
    return min(max(b, 0), len(grid.beats) - 1)


def _num(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


def _key_info(key: KeyTrack, grid: RhythmGrid, duration: float) -> schema.KeyInfo:
    tonic, mode = _key_parts(key.global_tonic + (12 if key.global_mode == "minor" else 0))
    segs: list[schema.KeySegment] = []
    pb = np.asarray(key.per_beat)
    start_b = 0
    for b in range(1, len(pb) + 1):
        if b == len(pb) or pb[b] != pb[start_b]:
            t0 = 0.0 if start_b == 0 else float(grid.beats[start_b])
            t1 = duration if b == len(pb) else float(grid.beats[b])
            st, sm = _key_parts(int(pb[start_b]))
            segs.append(schema.KeySegment(start=round(t0, 3), end=round(t1, 3), tonic=st, mode=sm))
            start_b = b
    if not segs:
        segs = [schema.KeySegment(start=0.0, end=duration, tonic=tonic, mode=mode)]
    changes = []
    for ch in key.changes:
        ft, fm = _key_parts(ch.from_key)
        tt, tm = _key_parts(ch.to_key)
        changes.append(schema.KeyChangeOut(t=round(ch.t, 3), bar=ch.bar, from_=f"{ft}:{fm}", to=f"{tt}:{tm}",
                                           kind=ch.kind))
    return schema.KeyInfo(global_=schema.KeyGlobal(tonic=tonic, mode=mode, conf=round(float(key.global_conf), 3)),
                          segments=segs, changes=changes)


def _rhythm_info(grid: RhythmGrid, duration: float):
    beats = [schema.BeatOut(i=i, t=round(float(t), 4), bar=int(grid.bar_of_beat[i]),
                            beat=int(grid.beat_in_bar[i]), conf=round(float(grid.beat_conf[i]), 3))
             for i, t in enumerate(grid.beats)]
    bars = []
    for i, (b0, n) in enumerate(grid.bars):
        start = float(grid.beats[b0])
        end = float(grid.beats[b0 + n]) if b0 + n < len(grid.beats) else min(duration, start + n * grid.ibi_ref)
        bars.append(schema.BarOut(i=i, start=round(start, 4), end=round(end, 4), beats=int(n)))
    subdiv, last = [], None
    for i, (b0, _) in enumerate(grid.bars):
        v = int(np.asarray(grid.subdiv)[b0])
        if v != last:
            subdiv.append(schema.SubdivisionOut(from_bar=i, cells_per_beat=v))
            last = v
    if not subdiv:
        subdiv = [schema.SubdivisionOut(from_bar=0, cells_per_beat=int(np.asarray(grid.subdiv)[0]) if len(grid.subdiv) else 4)]
    tempo = schema.TempoInfo(bpm=round(float(grid.bpm), 2), subdivision=subdiv,
                             free_time=[schema.TimeRegion(start=round(a, 3), end=round(b, 3)) for a, b in grid.free])
    meter = [schema.MeterOut(from_bar=int(a), num=int(n), den=int(d)) for a, n, d in grid.meter] or \
            [schema.MeterOut(from_bar=0, num=4, den=4)]
    return beats, bars, tempo, meter


def _chords(segs: list[Segment], run: _Run, ev, key: KeyTrack, grid: RhythmGrid):
    vocab = run.vocab
    segs = sorted(segs, key=lambda s: (s.t0, s.layer != "main"))
    triples = [(vocab.cands[s.cand].root, vocab.cands[s.cand].quality, vocab.cands[s.cand].bass) for s in segs]
    keys = [int(key.per_beat[_beat_of_time(grid, s.t0)]) if len(key.per_beat) else 0 for s in segs]
    nxt = [next((triples[j][0] for j in range(i + 1, len(segs)) if triples[j][0] >= 0), None)
           for i in range(len(segs))]
    voicings = VOI.voice_sequence(triples)
    col = {int(c): j for j, c in enumerate(ev.C)}
    src_names = [n for n in EV.sources(ev)]
    out, bass_line = [], []
    for i, (s, (r, q, b), k) in enumerate(zip(segs, triples, keys)):
        hk = dict(key=k, next_root=nxt[i])
        harte = H.to_harte_in_key(r, q, b, k, next_root=nxt[i])
        siu = post.seventh_in_upper(s, ev, vocab)
        if q == "N" or r < 0:
            root_name = bass_name = None
        else:
            sp = S.spell(r, q, b, k, next_root=nxt[i])
            root_name, bass_name = sp.root, sp.bass
        lab = schema.Labels(**SM.labels(r, q, b, k, next_root=nxt[i], seventh_in_upper=siu))
        deg = schema.Degree(**NUM.degrees(r, q, b, k))
        sb, eb = _num(grid.cell_beat[s.c0]), _num(grid.cell_beat[s.c1])
        bi = _beat_of_time(grid, s.t0)
        bar = int(grid.bar_of_beat[bi])
        bib = None if sb is None or grid.beat_in_bar[bi] == 0 else round(int(grid.beat_in_bar[bi]) + (sb - math.floor(sb)), 4)
        members = [m for m in vocab.classes[s.cls] if m != s.cand] if s.cls >= 0 else []
        equiv = [H.to_harte_in_key(vocab.cands[m].root, vocab.cands[m].quality, vocab.cands[m].bass, **hk)
                 for m in members]
        alts = [schema.Alternative(harte=H.to_harte_in_key(vocab.cands[a].root, vocab.cands[a].quality,
                                                            vocab.cands[a].bass, **hk), conf=round(float(c), 3))
                for a, c in s.alternatives]
        sources = {}
        j = col.get(int(s.cand))
        if j is not None and s.c1 > s.c0:
            for n in src_names:
                part = ev.parts.get(n)
                if part is not None and part.ndim == 2 and part.shape[1] == len(ev.C):
                    w = EV.source_activity(ev, n, s.c0, s.c1)
                    if w > 0:
                        sources[n] = round(float(np.exp(part[s.c0:s.c1, j]).mean()), 3)
        v = voicings[i]
        cid = f"c{i:04d}"
        out.append(schema.ChordOut(
            id=cid, start=round(s.t0, 4), end=round(s.t1, 4), start_beat=sb, end_beat=eb, bar=bar,
            beat_in_bar=bib, dur_beats=None if sb is None or eb is None else round(eb - sb, 4),
            harte=harte, root=root_name, quality=q, bass=bass_name, label=lab, degree=deg, layer=s.layer,
            role=s.role, min_detail=None if s.layer == "main" else (None if s.min_detail is None else round(s.min_detail, 3)),
            conf=round(float(s.conf), 3), salience=None if s.salience is None else round(float(s.salience), 3),
            equivalents=equiv, alternatives=alts, voicing=schema.Voicing(lh=list(v.lh), rh=list(v.rh)),
            sources=sources, edited=s.edited, suggestion=s.suggestion))
        if r >= 0 and v.lh:
            passing_tone = s.role == "bass_event" and post.is_bass_passing_tone(s, grid, run.p)
            bass_line.append(schema.BassNoteOut(
                start=round(s.t0, 4), end=round(s.t1, 4), note=bass_name or "", midi=int(v.lh[0]),
                conf=round(float(s.conf), 3), role="passing_tone" if passing_tone else "chord_bass",
                min_detail=out[-1].min_detail if s.layer == "passing" else None))
    return out, bass_line


def _assemble(st: DecodeState, p: DecoderParams, mode: str, run: _Run, segs: list[Segment], ev, key: KeyTrack,
              timing: dict[str, float]) -> tuple[schema.Analysis, schema.Display]:
    m = st.meta
    grid = run.grid
    duration = float(m.get("duration_s") or float(grid.cell_t[-1]))
    beats, bars, tempo, meter = _rhythm_info(grid, duration)
    chords, bass_line = _chords(segs, run, ev, key, grid)
    analysis = schema.Analysis(
        engine_version=ENGINE_VERSION, params_hash=params_hash(p, mode), mode=mode,
        calibration=DP.CALIBRATION_ID if p.temp else None,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source=schema.SourceInfo(key=st.key, type=m.get("source_type", "upload"), url=m.get("source_url"),
                                 title=m.get("title") or st.key, artist=m.get("artist"),
                                 duration=round(duration, 3)),
        audio=schema.AudioInfo(play=f"/api/tracks/{st.key}/audio", stems=list(m.get("stems", []))),
        display=f"/api/tracks/{st.key}/display",
        key=_key_info(key, grid, duration), tempo=tempo, meter=meter, beats=beats, bars=bars, sections=[],
        chords=chords, bass_line=bass_line,
        stats=schema.Stats(n_main=sum(c.layer == "main" for c in chords),
                           n_passing=sum(c.layer == "passing" for c in chords), runtime_s=dict(timing)))
    return analysis, DISP.build(analysis)


def _write(d: Path, analysis: schema.Analysis, display: schema.Display) -> None:
    (d / f"analysis.v{ENGINE_VERSION}-{analysis.params_hash}.json").write_text(
        analysis.model_dump_json(by_alias=True, indent=1), "utf-8")
    (d / "analysis.json").write_text(analysis.model_dump_json(by_alias=True, indent=1), "utf-8")
    (d / "display.json").write_text(display.model_dump_json(by_alias=True), "utf-8")


# ---------------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------------

def analyze(source: str, mode: str = "full", params: dict | None = None, progress: Progress | None = None,
            device: str = "cuda", write: bool = True) -> schema.Analysis:
    """Full analysis of one song (§5). Writes analysis.json / display.json into cache/<key>/,
    and the decode state used by ``redecode``."""
    progress = progress or _noop
    p = params_from(params)
    timing: dict[str, float] = {}

    def lap(name, t0):
        timing[name] = round(time.perf_counter() - t0, 2)

    t = time.perf_counter()
    asset = ingest.fetch(source, progress)
    lap("ingest", t)
    t = time.perf_counter()
    stems = separation.separate(asset, mode, progress)
    lap("separation", t)
    t = time.perf_counter()
    grid = rhythm.track(asset, stems, None, progress)
    hcdf = rhythm.hcdf_peaks(stems)
    lap("rhythm", t)

    t = time.perf_counter()
    from .models.ace import Ace
    from .models.base import evidence_path
    from .models.lv_chordia import LvChordia
    inputs = [asset.wav] + ([stems.harmonic] if Path(stems.harmonic) != Path(asset.wav) else [])
    evid: list[ChordEvidence] = []
    ev_files: list[str] = []
    for i, (model, label) in enumerate(((LvChordia(), "lv-chordia"), (Ace(), "consonance-ACE"))):   # §5.5 (ACE: M5 gate passed)
        progress("evidence", 0.4 * i, f"Reading chords ({label})")
        evid += [model.run_cached(w, device, asset.dir) for w in inputs]
        ev_files += [evidence_path(asset.dir, model.evidence_name(w), model.adapter_version).name for w in inputs]
    lap("evidence", t)

    notes = None                                                      # §5.7 piano transcription (M5)
    if (p.use_sym or p.amt_bass) and stems.model != "none" and separation.available(Path(stems.other)):
        t = time.perf_counter()
        progress("evidence", 0.8, "Transcribing the piano")
        notes = AMT.transcribe(stems.other, device, cache_dir=asset.dir)
        lap("notes", t)

    t = time.perf_counter()
    progress("decode", 0.0, "Tracking the bass")
    bass = bass_mod.track(stems, grid, None)
    key = KEY.estimate(stems, grid, bass)
    vocab = candidates()
    run = _Run(grid, evid, bass, key, vocab, p, hcdf, notes=notes)
    run.build_all()
    progress("decode", 0.3, "Decoding chords")
    main0 = run.first_main()
    new_grid = rhythm.check_octave(grid, main0)                       # §5 step 5 (once)
    if new_grid is not None:
        grid = run.grid = new_grid
        bass = run.bass = bass_mod.track(stems, grid, None)
        key = run.key = KEY.estimate(stems, grid, bass)
        run.build_all()
        main0 = run.first_main()
    key0 = run.key
    segs, ev, key = _finish(run, main0)
    lap("decode", t)

    available = [n for n in ("vocals", "drums", "bass", "other") if separation.available(Path(getattr(stems, n)))]
    st = DecodeState(dir=asset.dir, key=asset.key, grid=grid, bass=bass, key0=key0,
                     hcdf=np.asarray(hcdf if hcdf is not None else np.zeros(0), dtype=float), evid=evid,
                     notes=notes,
                     meta={"source_type": asset.source_type, "source_url": asset.source_url, "title": asset.title,
                           "artist": asset.artist, "duration_s": float(asset.duration_s), "stems": available,
                           "mode": mode, "evidence": ev_files})
    t = time.perf_counter()
    progress("output", 0.0, "Spelling and voicing")
    analysis, display = _assemble(st, p, mode, run, segs, ev, key, timing)
    lap("output", t)
    analysis.stats.runtime_s.update(timing)
    if write:
        save_state(st)
        _write(asset.dir, analysis, display)
    progress("done", 1.0, f"{len(analysis.chords)} chords")
    return analysis


def redecode(target: str | Path, params: dict | DecoderParams | None = None,
             region: tuple[float, float] | None = None, *, write: bool = False,
             progress: Progress | None = None) -> schema.Analysis:
    """Re-run steps 3–11 from the cached decode state (no audio, no GPU; seconds).

    ``params``: DecoderParams overrides. ``region=(t0, t1)`` (seconds): additionally re-scan that
    region ± 2 bars "more sensitively" (§5.11: λ_scale × 0.6, w_prior × 0.5, PT bonus 1.5) with the
    main chords just outside it as fixed boundaries; chords the re-scan finds that the normal
    decode does not show at Everything come back as ``suggestion: true`` passing chords
    (min_detail 0.05, so they are visible from Standard). ``write=True`` replaces analysis.json /
    display.json (the server writes only after the user accepts suggestions)."""
    progress = progress or _noop
    t = time.perf_counter()
    st = load_state(target)
    p = params_from(params)
    progress("decode", 0.0, "Re-decoding from cached evidence")
    run, segs, ev, key = decode_state(st, p)
    if region is not None:
        segs = segs + rescan(run, segs, ev, float(region[0]), float(region[1]))
    timing = {"redecode": round(time.perf_counter() - t, 2)}
    analysis, display = _assemble(st, p, st.meta.get("mode", "full"), run, segs, ev, key, timing)
    if write:
        _write(st.dir, analysis, display)
    progress("done", 1.0, f"{len(analysis.chords)} chords")
    return analysis


def rescan(run: _Run, segs: list[Segment], ev, t0: float, t1: float) -> list[Segment]:
    """§5.11 "Re-scan this section (more sensitive)": new chords in [t0, t1] ± 2 bars as suggestions."""
    grid, vocab = run.grid, run.vocab
    pr = DP.fast_rescan(run.p)
    r0, r1 = hsmm.rescan_region(grid, t0, t1)
    mains = sorted((s for s in segs if s.layer == "main"), key=lambda s: s.c0)

    def main_at(g: int) -> int | None:
        for s in mains:
            if s.c0 <= g < s.c1:
                return int(s.cand)
        return None

    G = len(grid.cell_t) - 1
    left = main_at(r0 - 1) if r0 > 0 else None
    right = main_at(r1) if r1 < G else None
    res = hsmm.decode(ev, grid, run.key, vocab, pr, "detail", region=(r0, r1), boundary=(left, right),
                      hcdf=run.hcdf)
    shown = post.pieces(segs)
    out: list[Segment] = []
    for R in res.segments:
        if R.c1 <= r0 or R.c0 >= r1 or vocab.cands[R.cand].quality == "N":
            continue
        cover = sum(min(c1, R.c1) - max(c0, R.c0) for c0, c1, s in shown
                    if s.cls == R.cls and c1 > R.c0 and c0 < R.c1)
        if 2 * cover >= R.c1 - R.c0:
            continue                                      # the normal decode already shows it
        conf = float(res.post_cls[R.c0:R.c1, R.cls].mean())
        out.append(Segment(c0=R.c0, c1=R.c1, t0=R.t0, t1=R.t1, cand=R.cand, cls=R.cls, layer="passing",
                           conf=conf, salience=None, min_detail=0.05, suggestion=True,
                           alternatives=layers.class_alternatives(res, ev.C, vocab, R.c0, R.c1, R.cls)))
    return out


def load_cached(key: str) -> schema.Analysis | None:
    p = config.CACHE_DIR / key / "analysis.json"
    return schema.Analysis.model_validate_json(p.read_text("utf-8")) if p.exists() else None
