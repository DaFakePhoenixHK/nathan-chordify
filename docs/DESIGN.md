# Nathan_chordify: implementation design (v1.1, 2026-10-13)

> v1.1 folds in the two reviews in docs/reviews/. Section 17 lists every change.

Nathan_chordify is a self-hosted alternative to Chordify for worship and gospel piano. You paste a YouTube link or upload audio, and it detects chords, including the soft and short passing chords that Chordify drops, then shows them synced to playback.

The analysis runs on Nathan's laptop. The app is available as:
- a local web UI;
- a Windows desktop window;
- an installable phone web app (PWA) that connects to the laptop.

All source facts and URLs are in [RESEARCH.md](RESEARCH.md); the two adversarial reviews are in [reviews/](reviews/). This document records **decisions**. Where a fact is unverified, it says so (§16), and milestone M0's smoke test and spikes settle it before anything else is built.

---

## 0. Decisions at a glance

| Area | Decision |
|---|---|
| Engine | Python 3.12 package `nathan_chordify` in `engine/`. One venv at `Nathan_chordify\.venv`. |
| Build order | Milestones **M0–M8** (§14): environment and spikes → engine v0 with one model → evaluation and calibration → browser MVP → editor and ground-truth mode → accuracy round 2 → desktop and phone → v1.1 parity → v2. |
| GPU stack | `torch==2.14.1+cu130` and `torchaudio==2.11.0+cu130`; torch installed with `--no-cache-dir`. If the M0 smoke test fails, fall back to `torch==2.11.0+cu130`. No torchcodec. All decoding goes through the ffmpeg CLI to WAV, then soundfile. |
| Download | `yt-dlp[default]>=2026.8.19` with `deno==2.9.7` as the JS runtime and Node 24 as the fallback runtime. App-local FFmpeg 9.0.2 (Gyan full_build-shared). Always `noplaylist=True`. Maximum song length 30 min (configurable). |
| Separation | demucs 4.1.0 `htdemucs_ft` (vocals / drums / bass / other), with the mix tensor kept on the CPU. "Harmonic" = bass + other. Stems are stored as FLAC. |
| Beats / meter | Beat This! 1.1.0 (`Audio2Beats`, checkpoint `final0`, no DBN; madmom not installed). Tempo-octave rule plus an octave check after the first decode, a per-section triplet test (2, 3, 4 or 6 cells per beat), a free-time grid for rubato passages, and a user half/double toggle. |
| Key | Profile key on the harmonic stem → 24-state local-key HMM with a log-softmax emission → chord-based refinement on 16-beat windows plus a truck-driver (step-up) test. Key changes are marked in the JSON and the UI. |
| Chord evidence | **lv-chordia** is the only model in engine v0 (M1): root+triad, bass and 7/9/11/13 heads, run on the mix and the harmonic stem in 120 s chunks, the two runs merged into **one** source. **consonance-ACE** and **ChordMini BTC-CL** join in M5, each only if it passes the component gate (§13.6). |
| Bass | Our own bass tracker on the bass stem: 12 bins/octave CQT with `filter_scale=0.5`, onset-segmented notes, overlap-weighted per-cell bass probabilities. It is the only bass source besides the piano left hand; the models' bass heads count half, inside their own source. |
| Notes | ByteDance `piano-transcription-inference` 0.0.6 on the `other` stem (M5). Upper notes become saturating symbolic chroma; notes below MIDI 55 feed the bass term. |
| Fusion / decoding | Each source becomes a calibrated log-softmax over the candidates (fitted temperature, clipped at −6); sources are combined by a weighted **mean**, and evidence is measured per beat (κ_b ≈ 10). A **semi-Markov (HSMM)** decode on the sub-beat grid with metric change costs, key and idiom priors, duration priors and (M5) a repetition prior. Confidence, salience and pruning work on **equivalence classes** of chords. Two layers (main + passing). |
| Output | JSON schema v1 (§5.13), plus `display.json` with labels and voicings precomputed for every transposition. Exports: `.lab` and JSON in v1; MIDI (DAW and notation versions), MusicXML, ChordPro, iReal Pro and print/PDF in v1.1. Edits flow into every export. |
| Server | FastAPI 0.143.0 + uvicorn 0.54.0 on `127.0.0.1:8765`. One long-lived, pre-warmed GPU worker process. SSE for progress. SQLite library. Append-only edit operations. |
| Frontend | Vite 8.3.4 + React 19.3.0 + TypeScript 7.0.2, linted with oxlint. v1 plays through `<audio>` with `preservesPitch`; v1.1 adds **signalsmith-stretch 1.3.2** (real pitch shift, gapless loops). **No chord theory in TypeScript**: the server sends every label. Piano SVG diagrams only. |
| Desktop | pywebview 6.2.1 + pythonnet 3.2.1 (WebView2) window around the local server, started from `Nathan_chordify.bat` or a desktop shortcut. stdout and stderr go to a log file under `pythonw`. |
| Phone | PWA served over HTTPS by **Tailscale serve**, so no firewall rule is needed. Surfshark stays on (standing rule); M6 tests Tailscale alongside it (§10). The phone needs the laptop on, except for songs saved offline (v1.1). The mkcert LAN fallback is deferred. |
| Storage | 37 GB free on C:. The song cache is capped at 10 GB; stems and evidence are kept 30 days; stems are FLAC. |
| Chord labels | Three levels (Simple / Standard / Full) plus an independent **Detail** slider for passing chords, and toggles for slash chords and upper-structure display. Letters, Nashville (級數) or Roman numerals. |

### 0.1 Questions for Nathan (none of them blocks M0–M3)

1. **Test song: RESOLVED.** Nathan confirmed the Cantonese **回家 by Eternity Girls** (B major, about 65 BPM), YouTube `Y7boUFbM76k`.
2. **Your chords for 回家.** They override every published chart (§13.1). Please send them as bar text, with section labels on their own lines:
   ```
   [Verse 1]
   | B . . B+7 | Emaj7 . A7 Em6 |
   ```
   - One bar between `|` marks, one token per beat; `.` means "the same chord continues".
   - A bar with fewer tokens than beats and no dots is split evenly: `| B E |` = 2 beats each.
   - Brackets split one beat: `(A7 Em6)`. `N` = no chord, `%` = repeat the previous bar, a trailing `?` = unsure (`Em/G?`).
3. **Surfshark VPN is on.** Surfshark and Tailscale are both WireGuard VPNs and commonly conflict, and YouTube bot checks hit VPN exit IPs more often. **Resolved 2026-10-13: Surfshark is never paused** (Nathan's rule). YouTube downloads use the nightly yt-dlp, back off on rate limits, or Nathan supplies audio files; the phone path is tested with Surfshark on in M6. Surfshark Antivirus may also quarantine `deno.exe`, FFmpeg DLLs or model files; the smoke test reports missing files (§12).
4. **Disk space.** C: has 37 GB free. The install takes about 7–8 GB (est.), and the smoke test requires at least 15 GB free afterwards. The song cache is capped at 10 GB (FLAC stems, 30-day retention). Is that OK, or should the cache live on another drive?
5. **Tailscale account.** The recommended phone setup needs a free Tailscale account, which you create and sign in to yourself.
6. **Personal use only?** Some optional weights are non-commercial or have an unknown licence (§15). v1 uses only MIT, BSD or CC-BY parts, except where §15 says otherwise.
7. **UI language.** Default zh-TW with an English toggle?
8. **Ground truth.** Can you tap the beats and check the low-confidence (L) spots in 回家, annotate 2 more songs by the end of M4, and reach 8 songs by the end of M5 (§13.3)? This is the single biggest factor in accuracy.
9. **Chordify baseline.** Can you copy Chordify's chord sequence for 回家 and 3 of the other songs into the same bar-text format? It is the only way to show that we beat Chordify (§13.4, B4).

---

## 1. Goals, non-goals and honest accuracy expectations

**Goals**
1. Detect worship and gospel harmony at piano level: 7ths, sus, add9, slash chords and inversions, passing diminished and augmented chords, borrowed chords and walk-ups. Show them at **sub-beat** resolution, which Chordify's one-chord-per-beat model cannot.
2. Match every Chordify Free, Premium and Premium Plus feature (§7) for one user on one laptop and a phone. v1 covers the core, v1.1 reaches Premium parity and v2 adds the Toolkit; §7 says which phase delivers each feature.
3. Make every result **correctable and re-analysable**: alternatives, region re-scan and an editor (v1), and an editable beat grid and meter (v1.1).

**Non-goals**
- Guitar, ukulele or mandolin diagrams.
- A public song catalog, or multi-user hosting.
- Selling the app (licences, §15).

**What accuracy we can expect, honestly**
- State of the art on standard datasets:
  - about 82–84% maj/min WCSR (lv-chordia 82.62, ChordFormer 84.09);
  - about 70–72% sevenths;
  - about 63–67% tetrads;
  - inversion-aware scores lower still (consonance-ACE MajMin-inv 75.6).
- Expert annotators agree only **73%** on maj/min labels and **54%** on detailed labels (CASD).
- **100% is not achievable**, and no tool gets there.

Our targets (§13) are therefore relative:
- match or beat the best single model on standard metrics;
- recover **≥60% of short passing chords at ≥70% precision**;
- beat Chordify's own output (baseline B4) on the same songs.

The UI is built on the assumption that some labels will be wrong: confidence shading, top-3 alternatives, one-click fixes, and edits that feed back into tuning.

---

## 2. Architecture

```
                       ┌──────────────────────── Laptop (Windows 11, RTX 4070 8 GB) ────────────────────────┐
 YouTube / SoundCloud ─┤                                                                                     │
 URL or uploaded file  │  FastAPI server process (uvicorn, 127.0.0.1:8765)                                   │
                       │   ├─ REST /api/*  ├─ SSE /api/jobs/{id}/events  ├─ WS /api/live (v2)                │
                       │   ├─ static web app (web/dist)                                                      │
                       │   ├─ SQLite data/library.sqlite  (tracks, jobs, setlists, edit-op index)            │
                       │   └─ job queue (FIFO) ──mp.Queue──►  GPU worker (spawn, 1 slot, pre-warmed)         │
                       │                                      │ ingest: yt-dlp+deno → ffmpeg → audio.wav     │
                       │                                      │ separation: htdemucs_ft → FLAC stems         │
                       │                                      │ rhythm: Beat This! → beats/downbeats/grid    │
                       │                                      │ evidence: lv-chordia (+ ACE, BTC-CL in M5)   │
                       │                                      │ bass tracker (+ piano AMT → chroma, M5)      │
                       │                                      │ key → calibrated fusion → HSMM main/detail   │
                       │                                      │ layers → spelling/degrees/voicings           │
                       │                                      └─► cache/<key>/ analysis+display ; SSE        │
                       │                                                                                     │
                       │  Desktop launcher (pythonw -m nathan_chordify.desktop, stdout → logs\)              │
                       │   └─ hosts the uvicorn thread + pywebview window (Edge WebView2)                    │
                       │  tailscaled: `tailscale serve --bg 8765` → https://<pc>.<tailnet>.ts.net            │
                       └──────────────────────────────────────────────┬──────────────────────────────────────┘
                                                                      │ HTTPS (Tailscale, WireGuard)
                                                               Phone PWA (Android Chrome / iOS Safari)
                                                               same React app, offline cache (v1.1)
```

**Processes**

| # | Process | What it does |
|---|---|---|
| 1 | Desktop launcher | Hosts the server in a thread, plus the pywebview window. `Nathan_chordify_server.bat` runs the server headless for phone-only use. |
| 2 | GPU worker | Started with `multiprocessing` `spawn`; its target is `nathan_chordify.server.worker:main`, and every launcher keeps its start-up code under `if __name__ == "__main__":`. It lives as long as the server and is **pre-warmed** at start-up (imports torch, loads lv-chordia and Beat This!; a cold start costs 15–30 s). It runs one job at a time and calls `torch.cuda.empty_cache()` between stages. If it crashes (e.g. CUDA OOM), the server restarts it and retries the job once in a lower-memory configuration (§5.14). |
| 3 | Live worker (v2) | Microphone chord detection. Uses CUDA when no analysis job is running, otherwise the CPU. |

**Data flow:** the browser talks only to the server. The server never runs models in-process, so it stays responsive while a 2-minute analysis runs.

---

## 3. Repository layout and ownership

The WP codes say which work package owns each file (§14). **A file has exactly one owner.** The shared contract files belong to WP0 and change only through a contract-change request to the lead.

```
Nathan_chordify\
├─ .venv\                                (local, not committed)
├─ Nathan_chordify.bat                   WP13  desktop launcher
├─ Nathan_chordify_server.bat            WP13  headless server (phone only)
├─ README.md                             WP0
├─ docs\  RESEARCH.md  DESIGN.md  reviews\                                      lead
├─ engine\
│  ├─ pyproject.toml  requirements.txt  constraints.txt  requirements.lock     WP0
│  ├─ nathan_chordify\
│  │  ├─ __init__.py          ENGINE_VERSION                                    WP0
│  │  ├─ config.py            paths; env set before torch/HF import (§9, §12)   WP0
│  │  ├─ logsetup.py          stdout/stderr → logs\ under pythonw (§9)          WP0
│  │  ├─ types.py             internal dataclasses (§4)                         WP0
│  │  ├─ schema.py            pydantic output models (§5.13)                    WP0
│  │  ├─ progress.py          Progress callback protocol                        WP0
│  │  ├─ ingest\              resolve.py youtube.py decode.py cache.py search.py WP1
│  │  ├─ separation\          demucs_sep.py mixes.py                            WP2
│  │  ├─ rhythm\              beats.py meter.py grid.py hcdf.py                 WP3
│  │  ├─ models\  base.py (WP0)  lv_chordia.py (WP4a)  ace.py (WP4b)  btc_cl.py (WP4c)
│  │  ├─ notes\               bass.py piano_amt.py symbolic.py                  WP5
│  │  ├─ theory\              vocab.py templates.py harte.py spelling.py key.py
│  │  │                       numbers.py simplify.py voicing.py priors.py
│  │  │                       chart.py display.py                               WP6
│  │  ├─ decode\              params.py calibrate.py evidence.py hsmm.py
│  │  │                       layers.py post.py sections.py                     WP7
│  │  ├─ pipeline.py  cli.py                                                    WP8
│  │  ├─ export\              lab.py midi.py musicxml.py chordpro.py ireal.py text.py  WP10
│  │  ├─ server\              app.py worker.py jobs.py db.py storage.py edits.py gt.py
│  │  │                       routes_tracks.py routes_jobs.py routes_setlists.py
│  │  │                       routes_export.py routes_lyrics.py routes_search.py
│  │  │                       routes_system.py lan.py                           WP9
│  │  ├─ server\live.py                                                         WP15
│  │  └─ desktop.py                                                             WP13
│  └─ tests\  test_<package>.py (each WP owns its own test files)
├─ shared\
│  ├─ analysis.schema.json  display.schema.json   (generated from schema.py)    WP0
│  ├─ fixtures\huijia_mock.analysis.json  huijia_mock.display.json              WP0
│  └─ test_vectors\spelling.json numbers.json simplify.json voicing.json
│                  classes.json  (Python tests only)                            WP6
├─ third_party\  consonance-ACE\  ChordMini\  VERSIONS.txt (pinned commits)      WP4b/WP4c
├─ models\  hf\  torch\  piano_transcription\  MANIFEST.json (sha256)           WP0 script
├─ tools\ffmpeg\bin\ffmpeg.exe ffprobe.exe                                      WP0 script
├─ cache\<key>\…   data\library.sqlite   logs\                                  (runtime)
├─ web\
│  ├─ package.json  package-lock.json  vite.config.ts  tsconfig.json
│  │  index.html  .oxlintrc.json                                                WP0
│  ├─ pwa.config.ts  public\manifest icons                                     WP13
│  └─ src\
│     ├─ shims.d.ts  (declare module 'signalsmith-stretch')                     WP0
│     ├─ main.tsx  app\ (shell, routes, layout, store)                          WP11
│     ├─ api\ (types.ts, client.ts, sse.ts)                                     WP0 (types) / WP11 (client)
│     ├─ player\  grid\  diagrams\  controls\  i18n\                            WP11
│     ├─ library\ editor\ gt\ setlists\ lyrics\ exports\ settings\              WP12
│     └─ toolkit\  (tuner, metronome, trainer, live)                            WP15
├─ eval\  __init__.py  gt\  synth\  chord_metrics.py  calibration.py  checklist.py
│         run_eval.py  tune.py  songs.yaml  reports\                            WP14
└─ scripts\ install.ps1 fetch_models.py smoke_test.py make_shortcut.ps1
            tailscale_setup.md mkcert_setup.ps1                               WP0 / WP13
```

- WP0 creates a stub `index.ts` in every `web/src/<feature>` folder, exporting a placeholder component under an agreed name. WP11's router can then import WP12 and WP15 pages before they exist.
- **There is no client-side chord theory.** Labels, degrees and voicings for every transposition come from the server (`display.json`, §5.12 step 12). The shared test vectors are run by Python tests only.
- The evaluation package is `eval/` and runs as a module from the repo root: `python -m eval.run_eval`. Its metrics file is `chord_metrics.py`: consonance-ACE imports a top-level package named `metrics` through a `sys.path` hack, so no module of ours may use that name.

---

## 4. Core internal types and interfaces (`engine/nathan_chordify/types.py`, owned by WP0)

```python
Progress = Callable[[str, float, str], None]     # (stage, pct 0..1, message)

@dataclass(frozen=True)
class AudioAsset:
    key: str                  # "yt-Y7boUFbM76k" | "sha256-<hex64>" | "url-<extractor>-<id>"
    dir: Path                 # cache/<key>/
    wav: Path                 # cache/<key>/audio.wav  (44.1 kHz, stereo, PCM_16; evictable, re-decoded from source.*)
    play: Path                # cache/<key>/play.m4a   (AAC 192k, faststart) for the web player
    duration_s: float
    title: str
    artist: str | None
    source_type: Literal["youtube", "url", "upload"]
    source_url: str | None

@dataclass(frozen=True)
class Stems:
    mix: Path; vocals: Path; drums: Path; bass: Path; other: Path     # stems/*.flac (16-bit FLAC)
    harmonic: Path            # bass + other (no vocals, no drums)
    model: str                # "htdemucs_ft" | "none" (debug fast mode: harmonic = mix, bass = low-passed mix)

@dataclass
class RhythmGrid:
    beats: np.ndarray         # [B] seconds
    beat_conf: np.ndarray     # [B]
    bar_of_beat: np.ndarray   # [B] int (−1 for pickup beats and free-time beats)
    beat_in_bar: np.ndarray   # [B] int, 1-based (0 in free time)
    bars: list[tuple[int, int]]          # (first_beat_idx, beats_in_bar); mixed meters allowed
    meter: list[tuple[int, int, int]]    # (from_bar, numerator, denominator)
    bpm: float
    ibi_ref: float            # median inter-beat interval (s); scales evidence in free-time cells
    subdiv: np.ndarray        # [B] cells per beat ∈ {2, 3, 4, 6}, constant within a section (§5.3)
    free: list[tuple[float, float]]      # free-time regions (s), drawn unmetered
    cell_t: np.ndarray        # [G+1] cell boundaries in seconds
    cell_beat: np.ndarray     # [G+1] boundaries in beat units (beat index + fraction; NaN inside free time)
    cell_pos: np.ndarray      # [G] int8: 0 downbeat, 1 mid-bar beat, 2 beat, 3 half/third, 4 quarter/sixth, 5 free time
    edited: bool = False      # True when user grid edits were applied

@dataclass
class RhythmOverrides:        # built from the edit ops (§5.13); rhythm.track re-applies them
    tempo_factor: float = 1.0                                       # 0.5 | 1 | 2
    downbeat_shift: int = 0
    moved_beats: dict[int, float] = field(default_factory=dict)     # beat idx → new time (s)
    tapped_beats: list[float] | None = None                         # replaces the tracker (tap tempo, GT beats)
    meter: list[tuple[int, int, int]] = field(default_factory=list) # (from_bar, num, den)
    subdiv: list[tuple[int, int]] = field(default_factory=list)     # (from_bar, cells per beat)
    free: list[tuple[float, float]] = field(default_factory=list)   # forced free-time regions

@dataclass
class ChordEvidence:          # one model on one input, at the model's native hop
    name: str                 # "lv@mix", "lv@harmonic", "ace@mix", "ace@harmonic", "btc@harmonic"
    frame_t: np.ndarray       # [T] frame centre times (s), already corrected by offset_s
    offset_s: float           # measured adapter latency (§5.5)
    root: np.ndarray | None   # [T,13] probs (12 pcs + N)
    bass: np.ndarray | None   # [T,13]
    notes: np.ndarray | None  # [T,12] independent probabilities
    heads: dict[str, np.ndarray]           # model-specific heads, e.g. lv: triad/seventh/ninth/...
    head_labels: dict[str, list[str]]
    label_probs: np.ndarray | None         # [T,V] for label classifiers (BTC-CL)
    labels: list[str] | None               # Harte labels of label_probs columns

@dataclass
class NoteEvents:
    onset: np.ndarray; offset: np.ndarray; pitch: np.ndarray; velocity: np.ndarray  # [N]
    offset_pedal: np.ndarray  # [N] offset extended to the sustain-pedal release
    source: str               # "piano_amt@other"

@dataclass
class BassTrack:
    frame_t: np.ndarray       # [T] (11.6 ms hop)
    salience: np.ndarray      # [T,12] pitch-class salience, rows sum to 1
    energy_db: np.ndarray     # [T]
    notes: list[tuple[float, float, int, float, str]]  # (t0, t1, midi, conf, "bass_stem" | "amt_lh"), latency-compensated

@dataclass
class KeyChange:
    t: float; bar: int
    from_key: int; to_key: int            # 0..23 as in KeyTrack.per_beat
    kind: Literal["step_up", "modulation", "relative", "parallel"]

@dataclass
class KeyTrack:
    global_tonic: int; global_mode: Literal["major", "minor"]; global_conf: float
    per_beat: np.ndarray      # [B] int 0..23 (0–11 major tonic pc, 12–23 minor)
    changes: list[KeyChange]

@dataclass(frozen=True)
class Candidate:
    root: int; quality: str; bass: int    # pcs 0..11, quality from §5.8; N is (-1, "N", -1)
    pcs: frozenset[int]                   # template tones (required + optional) ∪ {bass}
    cls: int                              # equivalence-class id (§5.8)

@dataclass
class Vocab:
    cands: list[Candidate]
    tau: np.ndarray           # [V,12] template probabilities (§5.9, sym)
    classes: list[list[int]]  # class id → member candidate indices

@dataclass
class SymParams:              # §5.7
    tau: float = 0.4; beta: float = 0.5; alpha: float = 0.7; theta: float = 0.25
    split_midi: int = 55; min_velocity: int = 20; min_dur_s: float = 0.06

@dataclass
class DecoderParams:          # decode/params.py; every constant of §5.9–5.12 lives here
    kappa_b: float = 10.0                                   # evidence per beat; FIXED while the rest is tuned
    temp: dict[str, float] = field(default_factory=dict)    # T_s per source, fitted by NLL (decode/calibrate.py)
    clip: float = -6.0
    w_src: dict[str, float] = field(default_factory=lambda: {"lv": 1.0, "ace": 1.0, "btc": 0.5, "sym": 1.0, "bas": 2.0})
    w_short: float = 1.0
    lam: dict[int, float] = field(default_factory=lambda: {0: 2.0, 1: 2.5, 2: 3.0, 3: 4.5, 4: 6.0, 5: 4.0})
    lam_scale: float = 1.0; lam_main_mult: float = 1.5
    w_prior: float = 1.0; prior_cap: float = 3.0
    w_dur: float = 1.0; dur_median_beats: float = 2.0; dur_sigma: float = 0.9
    pt_bonus: float = 1.0; rep_bonus: float = 1.0
    slider: tuple[float, float, float] = (-2.0, -0.5, 1.5)  # (a, b, c) of the min_detail mapping (§5.11), refitted in M2
    demote: float = 0.3
    prune_top: int = 10; prune_cap: int = 250
    bass_event_max_s: float = 0.3

@dataclass
class Evidence:
    C: np.ndarray             # [K] candidate indices (the pruned set, §5.8)
    E: np.ndarray             # [G,K] nats
    E_short: np.ndarray       # [G,K] the same with the fast sources up-weighted (w_short); used by segments ≤ ½ beat
    parts: dict[str, np.ndarray]   # calibrated ℓ_s [G,K] per source (debugging, and "sources" in the JSON)

@dataclass
class Segment:
    c0: int; c1: int          # cells [c0, c1)
    t0: float; t1: float
    cand: int                 # display candidate (index into Vocab.cands)
    cls: int
    layer: Literal["main", "passing"]
    role: Literal["chord", "bass_event"] = "chord"
    conf: float = 0.0         # class-level mean posterior
    salience: float | None = None
    min_detail: float | None = None
    alternatives: list[tuple[int, float]] = field(default_factory=list)   # (cand, class conf), other classes
    suggestion: bool = False; edited: bool = False

@dataclass
class DecodeResult:
    segments: list[Segment]
    post: np.ndarray          # [G,K] cell marginals
    post_cls: np.ndarray      # [G,n_cls] class marginals (sum over members)
    pb: np.ndarray            # [G] boundary probability
    score: float              # Viterbi path score

@dataclass
class EditOp:                 # one line of cache/<key>/edits.jsonl (append-only, §5.13)
    id: str                   # uuid4 made by the client; idempotency key
    seq: int                  # assigned by the server on append
    ts: str                   # ISO 8601
    device: str               # "desktop" | "phone-<n>"
    kind: Literal["set_chord", "delete_change", "insert_rest", "accept_suggestion", "reject_suggestion",
                  "set_meter", "set_subdiv", "set_free_time", "shift_downbeat", "move_beat",
                  "half_double_tempo", "tap_beats", "set_section", "set_key", "revert"]
    target: dict              # {"chord_id": …} | {"t0": …, "t1": …} | {"bar": …} | {"op_id": …}
    value: dict               # e.g. {"harte": "B:aug(b7)"}
```

The starting value of `slider` is a placeholder; M2 fits it (§5.11).

**Module entry points.** Each WP implements exactly these signatures:

```python
ingest.resolve(source: str) -> tuple[str, dict]                         # cache key + metadata, no download
ingest.fetch(source: str, progress: Progress) -> AudioAsset
ingest.search(query: str, n: int = 8) -> list[dict]                     # ytsearch (v1.1)
separation.separate(asset: AudioAsset, mode: str, progress) -> Stems
separation.practice_mix(asset, stems, gains_db: dict[str, float]) -> Path   # cached m4a (v1.1)
rhythm.track(asset, stems, overrides: RhythmOverrides | None, progress) -> RhythmGrid
rhythm.check_octave(grid: RhythmGrid, main: DecodeResult) -> RhythmGrid | None   # §5.3; None = unchanged
models.base.ChordModel.run(wav: Path, device: str) -> ChordEvidence     # lv_chordia/ace/btc_cl subclasses
notes.bass.track(stems: Stems, grid: RhythmGrid, amt: NoteEvents | None) -> BassTrack
notes.bass.cell_probs(bass: BassTrack, grid: RhythmGrid) -> np.ndarray          # P_bass [G,12] (§5.6); re-run on grid changes
notes.piano_amt.transcribe(wav: Path, device: str) -> NoteEvents
notes.symbolic.cell_chroma(notes: NoteEvents, grid: RhythmGrid, p: SymParams) -> tuple[np.ndarray, np.ndarray]  # x, dch [G,12]
theory.key.estimate(stems, grid, bass: BassTrack) -> KeyTrack
theory.key.refine(key: KeyTrack, segments: list[Segment], grid, sections: list[dict] | None) -> KeyTrack
theory.vocab.candidates() -> Vocab                                       # with equivalence classes
theory.priors.prior(cp: Candidate, c: Candidate, key: int) -> float     # §5.10 table, stacked and capped
theory.priors.trigram(l: Candidate, p: Candidate, r: Candidate, key: int) -> float   # §6.4
theory.chart.parse(text: str) -> Chart                                   # §0.1 bar-text format
theory.chart.align(chart: Chart, beats: np.ndarray) -> list[Segment]
theory.chart.format(segments: list[Segment], grid: RhythmGrid) -> str
theory.display.build(analysis: schema.Analysis) -> schema.Display       # 12 transpositions × accidental preference
decode.calibrate.fit(evidence_sets, gt) -> dict[str, float]             # temperatures T_s
decode.evidence.build(grid, evid: list[ChordEvidence], sym: np.ndarray | None, bass: BassTrack,
                      vocab: Vocab, C: np.ndarray, p: DecoderParams) -> Evidence
decode.evidence.prune(ev_all: Evidence, vocab: Vocab, main: DecodeResult | None, p) -> np.ndarray   # §5.8
decode.hsmm.decode(ev: Evidence, grid, key, vocab, p, level: Literal["detail", "main"],
                   region: tuple[int, int] | None = None,                 # cell range (re-scan); None = whole song
                   boundary: tuple[int | None, int | None] = (None, None),# candidates fixed just outside the region
                   rep: np.ndarray | None = None) -> DecodeResult         # consensus class per cell (M5)
decode.layers.build(detail, main, ev, grid, key, vocab, p) -> list[Segment]
decode.post.finalize(segments, bass, key, grid, vocab, ev, p) -> list[Segment]
decode.sections.find(grid, stems, post_cls) -> tuple[list[dict], np.ndarray]   # auto sections + consensus class per cell
pipeline.analyze(source: str, mode: Literal["full", "fast"], params: dict, progress) -> schema.Analysis
pipeline.redecode(key: str, params: dict, region: tuple[float, float] | None) -> schema.Analysis   # cached evidence only
server.edits.apply(analysis: schema.Analysis, ops: list[EditOp]) -> tuple[schema.Analysis, schema.Display]
```

---

## 5. Analysis pipeline

**Mode.** The app always runs **Full**, which runs every step. `nchord analyze --mode fast` is a CLI/debug option only: it skips separation and piano transcription (the harmonic stem is the mix, the bass stem is the mix low-passed at 300 Hz). The two-phase "fast preview, then full" flow is **deferred**. If it ever returns, Full must reuse Fast's beats and evidence; otherwise one GPU slot simply pays for both runs.

**Length.** The default maximum is 30 min (Settings can raise it). lv-chordia and consonance-ACE run in overlapping chunks (§5.5), so VRAM does not grow with song length.

**Orchestration** (WP8 `pipeline.py`; it owns every loop between WP3, WP6 and WP7):

```
 1  ingest.fetch → separation.separate → rhythm.track → model evidence → notes.bass.track
    (M5: piano_amt.transcribe → symbolic.cell_chroma)
 2  key = theory.key.estimate(...)
 3  C0  = evidence.prune(ev_all, vocab, main=None)        # class-level top-10 per cell (§5.8)
    ev0 = evidence.build(..., C0)
 4  main0 = hsmm.decode(ev0, level="main")
 5  if rhythm.check_octave(grid, main0) returns a new grid: re-pool the cached evidence onto it, redo 3–4 (once)
 6  C  = evidence.prune(ev_all, vocab, main=main0)        # adds idiom-completion bridges, cap 250
    ev = evidence.build(..., C)
 7  M5: detail0 = hsmm.decode(ev, "detail"); sections, rep = decode.sections.find(grid, stems, detail0.post_cls)
    (§5.15; before M5 rep = None and sections = [])
 8  detail = hsmm.decode(ev, "detail", rep=rep);  main = hsmm.decode(ev, "main", rep=rep)
 9  segments = layers.build(detail, main, ...) → post.finalize(...)
10  key' = theory.key.refine(key, segments, grid, sections)
    if key' changes the local key on ≥ 8 beats: redo 8–9 once with key'
11  spelling, numbers, voicings → analysis.json;  theory.display.build → display.json
```

### 5.1 Ingest (WP1)

1. **Classify the source.**
   - A YouTube URL in any form (watch, youtu.be, shorts, music.youtube, with or without `&list=`/`start_radio`/`pp=`) → `extract_info(url, download=False)` with `noplaylist=True` gives the canonical 11-character ID. The key is `yt-<id>`.
   - Any other URL yt-dlp supports (SoundCloud, Bilibili, …) → `url-<extractor>-<id>`. **Deezer is unsupported** (DRM).
   - An upload or local file → its SHA-256 is computed while streaming → `sha256-<hex>`.
2. **Use the cache.** If `cache/<key>/audio.wav` exists, skip the download. If only `source.*` is left (evicted, §5.14), decode again.
3. **Download with yt-dlp**, with these options:
   ```python
   {"format": "bestaudio/best", "noplaylist": True, "outtmpl": str(dir / "source.%(ext)s"),
    "ffmpeg_location": str(FFMPEG_BIN),
    "js_runtimes": {"deno": {"path": deno.find_deno_bin()}, "node": {}},   # Node is a fallback only when listed
    "retries": 3, "socket_timeout": 20, "max_filesize": 1_000_000_000,
    "match_filter": yt_dlp.utils.match_filter_func(f"duration <= {MAX_LEN_S}"),   # 1800 by default
    "progress_hooks": [hook], "quiet": True}
   ```
   - Save `meta.json` with title, uploader, duration, thumbnail URL and upload date.
   - No yt-dlp post-processors are used, so every ffmpeg call is ours and runs with `creationflags=subprocess.CREATE_NO_WINDOW`.
4. **When the download fails**, show these options to the user, in this order. **Never suggest pausing or bypassing the VPN** (standing rule from Nathan, 2026-10-13; see CLAUDE.md). VPN exit IPs trigger more bot checks, so the ladder has to work with the VPN on:
   1. "Update downloader": the server stops the GPU worker, runs `pip install -U --pre "yt-dlp[default]"` (nightly; on 2026-10-13 the stable build got HTTP 403 and the nightly worked) in a separate process, restarts the worker and retries (§9).
   2. On HTTP 429 / "confirm you're not a bot": **back off**. Show "YouTube is rate-limiting this connection; try again in about 30–60 min", and refuse automatic retries for 30 min, because every retry extends the block.
   3. The `bgutil-ytdlp-pot-provider==2.0.2` plugin, installed by the same button flow after the user confirms.
   4. A setting to use browser cookies (`cookiesfrombrowser=("firefox",)`), opt-in only.
   5. Always available: "Upload the audio file instead".
5. **Decode once** (both with `CREATE_NO_WINDOW`).
   - `ffmpeg -nostdin -y -i source.* -vn -ac 2 -ar 44100 -c:a pcm_s16le audio.wav`
   - `ffmpeg -nostdin -y -i audio.wav -c:a aac -b:a 192k -movflags +faststart play.m4a`
6. **Limits:** 30 min and 1 GB by default, both configurable (Chordify allows 20 min and 32 MB).

### 5.2 Separation (WP2)

- **Model:** demucs 4.1.0 `htdemucs_ft`, called in-process (no CLI):
  ```python
  sep = demucs.api.Separator(model="htdemucs_ft", device="cuda", shifts=1, overlap=0.25, segment=7.8)
  _, stems = sep.separate_tensor(mix_cpu_float32, 44100)   # the mix tensor stays on the CPU
  ```
  - `apply_model` allocates whole-track buffers on the mix tensor's device, so the mix must be a **CPU** float32 tensor while the model runs on CUDA. `segment` must be ≤ 7.8 s.
  - Output: `stems/{vocals,drums,bass,other}.flac` and `stems/harmonic.flac` = bass + other (44.1 kHz, 16-bit FLAC via soundfile). No stem WAVs are written.
- **If memory runs out:** use `segment=6`, then fall back to CPU (§5.14).
- **Practice mixes** (v1.1, for the player's stem mixer) are rendered on demand with ffmpeg `amix`.
  - Gains are rounded to 3 dB steps, and each mix is cached as `stems/mix_<hash>.m4a`.
  - Presets: *No vocals*, *Piano focus* (other +3, bass 0, drums −12, vocals −inf), *Bass focus*, *Original*.
- **BS-RoFormer-SW** (6 stems, with a real piano stem) is a **phase-2, evaluation-gated** option (WP16, M8). Reasons: unknown licence, an extra 700 MB, and audio-separator pins `rotary-embedding-torch<0.7`.

### 5.3 Beats, downbeats, meter and grid (WP3)

```python
from beat_this.inference import Audio2Beats
beats, downbeats = Audio2Beats(checkpoint_path="final0", device="cuda", dbn=False)(mono_mix_44k, 44100)
ibi  = median(diff(beats));  bpm = 60 / ibi
bpb_seq = beats-per-bar between consecutive downbeats;  bpb = mode(bpb_seq)

# tempo-octave rule (回家 is ~65 BPM; ChordU reported 130.2 = eighth notes)
if bpb == 8 and bpm >= 110:               drop every other beat (keep downbeat-aligned) → bpb = 4, bpm /= 2
elif bpb in (6, 12) and bpm >= 110:       compound meter 6/8 or 12/8: beat = dotted quarter = 3 tracked beats
elif bpm < 45:                            insert midpoints → bpm *= 2
bars keep their own length (mixed meter allowed); beats before the first downbeat → pickup bar (−1)

# free time (rubato intros, ballad tags), tested on sliding 8-beat windows
free = mean(beat_conf) < 0.5 or std(ibi) / mean(ibi) > 0.15
   → 0.2 s cells, each boundary snapped to the strongest onset within ±60 ms; cell_pos = 5; flat change cost;
     drawn unmetered (no bar lines); evidence scaled by ibi_ref (§5.9)

# triplet test, per section (8-bar blocks until sections exist), on the harmonic stem's onset strength
r3 = mean onset strength at 1/3 and 2/3 of each beat  ÷  mean at 1/4, 1/2 and 3/4
ternary = compound meter or r3 > 1.2
subdiv  = (6 if ibi >= 0.9 s else 3) if ternary else (4 if ibi >= 0.75 s else 2)
          # 回家: ibi ≈ 0.917 s, binary → 4 cells per beat, cell ≈ 0.23 s
cell boundaries = each beat interval split evenly into subdiv
cell_pos: 0 downbeat | 1 mid-bar beat (beat bpb/2+1 when bpb is even) | 2 other beat
          | 3 half (or 1/3, 2/3) | 4 quarter (or sixth) | 5 free time
HCDF: harmonic change detection on 12-bin chroma of the harmonic stem (hop 1024 @ 22.05 kHz),
      peaks above mean + 1.5·std → used only as a +0.5 nat boundary bonus (§5.10)

# octave check after the first main decode (rhythm.check_octave, any meter; pipeline step 5)
if ≥ 80% of main-layer changes fall on even-indexed tracked beats (0, 2, 4, … from each downbeat)
   and the halved tempo stays ≥ 45 BPM: halve the tempo (keep those beats), rebuild the grid, decode again
```

**User overrides**, stored as edit ops:
- half/double tempo (v1, M4);
- move a beat, tap tempo, shift the downbeat by ±n beats, set the meter or the subdivision from bar *k* onward, mark a region as free time (v1.1, M7).

`rhythm.track(..., overrides)` re-applies them. Only decoding is redone: evidence is cached on the 11.6 ms timeline (§5.9) and re-pooled onto the new grid.

### 5.4 Key (WP6)

1. **Profiles.** Compute beat-synchronous `chroma_cqt` of the harmonic stem (hop 4096 @ 22.05 kHz). Score the 24 keys with Krumhansl-Kessler and Temperley profiles (mean of the two correlations), plus 0.3 × the correlation of the bass-salience histogram with the tonic-weighted profile.
2. **Local key** (revised 2026-10-13 after 回家 drifted to G#m/E/C#m; see `theory/key.py`). An HMM over beats with 24 states.
   - Emission: **log-softmax over the 24 keys of cos/T**, T = 0.02, where cos is the cosine similarity of an 8-beat chroma window to each key profile, **divided by 8** because every beat sits in 8 overlapping windows (effective T = 0.16).
   - Moves to a key one accidental away (vi, IV, V, ii, iii) cost **12 extra nats**: chroma cannot tell them apart once a worship song borrows ♭VII/iv or tonicises vi, so those are left to the chord-based pass. Step-ups (+1/+2) and distant modulations keep the weights below.
   - Self-transition 0.995 per beat.
   - Moves to the relative, parallel, dominant or subdominant key, and **same-mode moves up 1 or 2 semitones**, are twice as likely as other moves.
   - Key segments shorter than 16 beats are merged.
3. **Chord-based refinement** after the first decode (pipeline step 10), as a **chord-likelihood HMM** (revised 2026-10-13): each decoded chord scores log P(chord | key) from a degree × triad-family table (I/IV/V/vi dominate; secondary dominants and borrowed chords are rare; a minor key needs its major V to beat its relative major); the path starts and ends in the global key; every change costs 16 nats; the global key gets +0.4 nats per beat where the chroma pass agrees. A chroma-only change is dropped unless the chords support it. Chord-based keys are more reliable for worship songs that sit on IV and V.
4. **Truck-driver test.** At each section boundary in the last two thirds of the song (auto sections from M5, 8-bar blocks before that), transpose the following section's chord classes down 1 and 2 semitones and compare them beat by beat with each earlier section. If ≥ 70% of the beats match, mark a key change of +1 or +2 at that boundary (`kind = "step_up"`).
5. If the local key changes on ≥ 8 beats, WP8 re-decodes once (the priors depend on it), then spelling runs (§5.12).

- **Spelling of key names:** fewest accidentals per key segment, ties to sharps (F# over Gb), with a user toggle.
- **Key changes** are written to `key.changes` (§5.13). The UI shows a badge at that bar, and Nashville numbering restarts there (§5.12 step 8).

### 5.5 Chord-model evidence (WP4a/b/c)

| Adapter | Milestone | Input(s) | What it exposes (before any CRF or argmax) | Native hop | Chunking |
|---|---|---|---|---|---|
| `lv_chordia.py` | M1 | mix, harmonic (resampled to 22.05 kHz) | Softmax of each head, averaged over the 5 ensemble models: root+triad, bass(13), seventh, ninth, eleventh, thirteenth. The package's CRF is **not** used. | 23 ms | 120 s windows with 10 s overlap; 5 s cropped at each interior edge |
| `ace.py` | M5, gated | mix, harmonic | root(13) softmax, bass(13) softmax, notes(12) sigmoid, from `conformer_decomposed_smooth.ckpt` | 23 ms (CQT hop 512 @ 22.05 kHz) | 20 s windows with 5 s overlap (its own inference uses 20 s chunks without overlap); 2.5 s cropped at each interior edge |
| `btc_cl.py` | M5, last, gated | harmonic | 170-class softmax from `btc_model_best.pth` | read from ChordMini's config by WP4c (unverified) | as the original code |

**Rules every adapter follows**
- Load weights with `torch.load(..., weights_only=True)`; every third-party checkpoint has been verified to load that way. The BTC checkpoint was saved on `mps`, so the adapter calls `torch.load(path, map_location=device, weights_only=True)` itself.
- Keep weights on the GPU between jobs.
- Write intermediate output to `cache/<key>/evidence/<name>.v<adapter_version>.npz`.
- **Measure the time offset.** Each adapter's latency is measured on D2 (exact labels) by cross-correlating its label-change curve with the GT boundaries, stored as `offset_s` and subtracted from `frame_t`. Acceptance: residual |offset| < 20 ms.

**lv-chordia specifics (WP4a)**
- It silently runs with **random weights** if its `.sdict` files are missing, so the adapter asserts that `<sys.prefix>/share/lv-chordia/cache_data/*.sdict` exists before the first run.
- It takes file paths: each 120 s window is written to `cache/<key>/tmp/lv_<i>.wav` and passed as an **absolute** path.
- It uses the GPU whenever one is visible (`use_gpu = cuda.device_count() > 0`). Its CPU fallback therefore runs in a one-off subprocess with `CUDA_VISIBLE_DEVICES=""` set before torch is imported.
- **M0 GO/NO-GO spike:** read the six per-head softmax outputs without forking, and record the class lists in `head_labels`. NO-GO → vendor the model file into `third_party/` with the MIT notice; if that also fails, M1 uses consonance-ACE as its single model.

**consonance-ACE specifics (WP4b)**
- **Never `pip install` the ACE repo**: it pins beat-this 0.1, lightning 2.5.1 and torch 2.6.0. Its runtime dependencies are pinned in `requirements.txt` instead (§12).
- The adapter puts `third_party/consonance-ACE` on `sys.path` and imports `ACE.mir_evaluation` **before** `ACE.preprocess.audio_processor`, which does a bare `from transforms import CQTransform`.

### 5.6 Bass tracking (WP5; M1)

```python
x  = bass stem (mono, 22.05 kHz)                     # debug fast mode: mix low-passed at 300 Hz
C  = |CQT(x)|, fmin = E1 (41.2 Hz), 12 bins/octave, filter_scale = 0.5 (Q ≈ 8.4, ≈ 0.2 s filters at 41 Hz),
     6 octaves, hop 256 (11.6 ms)
S[f0] = Σ_{h=1..4} 0.8^(h-1) · C[f0 + round(12·log2(h))]    for f0 ∈ [E1, G3]      # harmonic summation
sal[t, pc] = softmax_pc( max_octave S[t, pc] / 0.5 )                                # pitch-class salience
onsets = onset_detect(x, backtrack=True), shifted by the filter latency measured on D2
notes  = segments between onsets; pc = argmax of mean sal over the segment after a 30 ms attack skip;
         conf = mean max-salience; segments shorter than 60 ms are merged into the previous note
left hand (M5): where the rolling 1 s RMS of the bass stem < −45 dBFS, use piano-AMT notes below MIDI 55 instead
P_bass(g, pc) = normalize( Σ_n ov(n, g) · conf_n · 1[pc_n = pc]  +  0.05 · (1 − Σ_n ov(n, g)) / 12 )
                # ov(n, g) = fraction of cell g covered by note n
```

- **Fallback:** if the CQT version fails the acceptance test below, WP5 switches the note pitch to `librosa.pyin` (fmin 40 Hz, frame 2048 @ 22.05 kHz, hop 256) with the same onset segmentation.
- **P_bass comes only from this tracker** (and the AMT left hand). The models' bass heads stay inside their own sources at half weight (§5.9), so bass evidence is not counted twice.
- Octave errors don't matter, because only the pitch class is used.
- **Acceptance (WP5):** synthetic bass lines with 16th notes at 65 BPM and 8th notes at 130 BPM, over E1–G3: ≥90% pitch-class accuracy per note and onset error ≤ 40 ms.

### 5.7 Note transcription and symbolic chroma (WP5; M5)

```python
audio = soundfile.read(stems/other.flac) → mono → soxr to 16 kHz, float32      # not the package's load_audio
amt   = PianoTranscription(device="cuda",
          checkpoint_path="models/piano_transcription/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth")
out   = amt.transcribe(audio, None)          # the second argument (MIDI output path) is required; None = no file
keep notes with velocity ≥ 20, duration ≥ 60 ms, 33 ≤ pitch ≤ 96; offset_pedal = offset extended to the pedal release
pitch < 55 → bass term (left hand, §5.6);  pitch ≥ 55 → symbolic chroma

# per cell g = [a, b), Δg = b − a, v_n = velocity / 127, off'_n = offset_pedal
ch[g, pc]  = Σ_{n: pc(n)=pc} v_n · (1/Δg) ∫_{max(a,on_n)}^{min(b,off'_n)} exp(−(t − on_n)/τ) dt
           + β · v_n · 1[on_n ∈ [a − 0.05, b)]                            τ = 0.4 s, β = 0.5
x[g, pc]   = 1 − exp(−ch[g, pc] / θ)                                       θ = 0.25: saturating presence, no max-normalisation
dch[g, pc] = max(0, ch[g, pc] − α · ch[g−1, pc])                          α = 0.7: onset feature only (phantom rule §5.12, HCDF)
```

- The package's own `load_audio` is not used: it passes a positional resample argument that librosa 0.11 rejects (TypeError).
- The checkpoint is prefetched by `scripts/fetch_models.py`; the package's `wget` auto-download does not exist on Windows.
- In debug fast mode, and before M5, there are no notes: the symbolic term is dropped (w_sym = 0).

### 5.8 Chord candidates and equivalence classes (WP6 `vocab.py`)

**Qualities.** 32 internal qualities plus N; templates are given as semitones from the root.

```
maj 0 4 7 | min 0 3 7 | dim 0 3 6 | aug 0 4 8 | sus2 0 2 7 | sus4 0 5 7 | 5 0 7 | 6 0 4 7 9 | m6 0 3 7 9
7 0 4 7 10 | maj7 0 4 7 11 | m7 0 3 7 10 | m7b5 0 3 6 10 | dim7 0 3 6 9 | mM7 0 3 7 11 | +maj7 0 4 8 11
7sus4 0 5 7 10 | +7 0 4 8 10 | add9 0 2 4 7 | madd9 0 2 3 7 | 6/9 0 2 4 7 9 | 9 0 2 4 7 10
maj9 0 2 4 7 11 | m9 0 2 3 7 10 | m11 0 2 3 5 7 10 (2 optional) | 9sus4 0 2 5 7 10 | 13sus4 0 2 5 7 9 10 (2 optional)
13 0 4 7 9 10 (2 optional) | 7b9 0 1 4 7 10 | 7#9 0 3 4 7 10 | 7b13 0 4 7 8 10 | maj7#11 0 4 6 7 11
```

- **Template roles.** The perfect 5th is *optional* except in dim, aug, m7b5, dim7, +7, +maj7 and 5. Tones marked "optional" above are optional. All other listed tones are *required*. Any tone not listed is *absent*.
- **Allowed bass notes:**
  - any chord tone except the 9th, 11th or 13th;
  - for maj, min, sus2, sus4, add9, **maj7, m7 and 6**, also the non-chord intervals **+2, +5, +10**. These cover IV/5 (F/G = maj/+2), Em/A (min/+5), F#/B (maj/+5), G/F (maj/+10), C#m7/F# (m7/+5), Emaj7/F# (maj7/+2) and B6/C# (6/+2).
- **Equivalence classes.** Candidates with the same bass pc and the same pc-set (template tones ∪ bass) form a class. Two classes are merged when they share the bass and one's set equals the other's minus a single optional perfect 5th. Examples:
  - the four spellings of a dim7 over the same bass, and the aug rotations;
  - E6 = C#m7/E; Csus2 = Gsus4/C;
  - F#9sus4 = C#m7/F#, merged with E/F#; Emaj7/F# = F#13sus4;
  - B+7 = B7b13 without its 5th.

  **Confidence, salience and pruning are computed per class**, so equivalent spellings no longer split the posterior (a passing °7 used to land at conf ≤ 0.25 and be dropped). The display label inside a class is chosen afterwards (§5.12 step 6). WP6 ships `classes.json` test vectors.
- **Size.** About 1,900 (root, quality, bass) candidates plus N (est.).
- **Pruning** (`decode.evidence.prune`): two passes and a cap.
  1. **C0** = ∪ over cells of all members of the **top-10 classes** (ranked by their best member's E) in that cell and its ±2 neighbours.
  2. **Idiom completion**, after the first main decode (pipeline step 6). For every adjacent pair of main chords A → B, add the bridge candidates:
     - the °7 a semitone below root(B) (#x°7 → x);
     - V7/B (the dominant 7th on root(B) + 7);
     - A and B over each pitch class on the stepwise bass path between bass(A) and bass(B) (bass-step slash chords);
     - Bsus4, and Aaug / A+7.
  3. **Cap: 250 candidates.** Over the cap, whole classes with the lowest best E are dropped; bridge candidates are never dropped.

  Without completion, the top-10 per cell fills with variants of the sounding chord, and once-per-song passing chords (B+7, Em6) never enter C.
- **Passing-type set PT** (these get the short-duration bonus): dim, dim7, m7b5, aug, +7, +maj7, sus2, sus4, 7sus4, 9sus4, 13sus4, and **any chord whose bass ≠ root**.

### 5.9 Fusion: calibrated cell evidence (WP7 `evidence.py`, `calibrate.py`)

```python
# 1) One source per model. A model's head probabilities are averaged over its two inputs FIRST:
#    p = 0.6·p@harmonic + 0.4·p@mix   (the two runs of one network are strongly correlated)
lv(t,c)  = log p_lv.root_triad[t, (r, triad(q))] + 0.5 · log p_lv.bass[t, b]
         + Σ_{h ∈ seventh,ninth,eleventh,thirteenth} max(log 1e-4, log p_lv.h[t, ext_h(q)])   # ext_h(q) = "none" if q lacks it
ace(t,c) = log p_ace.root[t, r] + 0.5 · log p_ace.bass[t, b]
         + 0.5 · Σ_pc max(log 1e-4, τ_c(pc)·log n_t(pc) + (1 − τ_c(pc))·log(1 − n_t(pc)))   # τ_c: 0.85 req / 0.5 opt / 0.10 absent
btc(t,c) = log Σ_{l ∈ M(r,q)} p_btc[t, l]             # M maps to the nearest BTC-170 labels; bass ignored
sym(g,c) = Σ_pc [ x_g(pc)·log τ'_c(pc) + (1 − x_g(pc))·log(1 − τ'_c(pc)) ]
           # τ' = τ, except the bass pc gets 0.5 unless it is also an upper tone (RH need not play the bass)
bas(g,b) = log( 0.9 · P_bass(g, b) + 0.1/12 )          # depends only on the bass pc
N (no chord): lv/ace N classes; sym uses τ = 0.1 for all pcs; bas uses 1/12

# 2) Time. Interpolate each model's frame log-scores (offset-corrected, §5.5) onto the 11.6 ms timeline,
#    then integrate over each cell (overlap-weighted mean of the 11.6 ms frames)      → raw_s(g, c)

# 3) Calibrate each source s: fitted temperature T_s, log-softmax over C, clip
ℓ_s(g,c) = max(−6, log_softmax_{c ∈ C}( raw_s(g,c) / T_s ))       # bas: the softmax runs over the 12 bass pcs

# 4) Weighted MEAN over the active sources, measured per beat
B(g,c)  = Σ_s w_s · ℓ_s(g,c) / Σ_{s active} w_s
          w: lv 1.0, ace 1.0, btc 0.5, sym 1.0, bas = w_bas 2.0 (tuned in [1, 3])
E(g,c)  = κ_b · (Δg / ibi_g) · B(g,c)       κ_b = 10 per beat; ibi_g = the local beat length (ibi_ref in free time)
E_short = the same with w_bas and w_sym multiplied by (1 + w_short), w_short = 1.0   # for segments ≤ ½ beat (§5.10)
```

Why it is built this way:
- **One source per model.** Counting the mix and harmonic runs of the same network twice would double its weight.
- **Calibrated log-softmax.** Each source becomes a calibrated distribution over C; the clip at −6 stops any single source from vetoing a candidate. T_s is fitted by NLL on D2 and the reviewed D1 songs (`decode/calibrate.py`) before any other tuning.
- **Weighted mean.** The scale stays the same however many sources are active (M1 has only lv and bas; M5 adds ace, sym and btc if they pass the gate).
- **Per beat.** Evidence measured in beats makes the change costs tempo-independent.
- **w_short.** The networks smooth over time internally even without their CRF, so in very short segments the fast sources (our bass tracker and the symbolic chroma) get more weight.

**Calibration targets (WP7 acceptance, gate of M2)**
- **Break-even table** (`eval/calibration.py`). For each pattern (A–P–B, A–P–A), each P length (¼, ½, 1 beat) and each metric position, it lists the per-beat bracket advantage δ* (nats of B(g, P) − B(g, A) per beat) at which P is decoded. Targets: **δ* ≈ 0.7 nats for a half-beat A–P–B** and **δ* ≈ 1.3 nats for a half-beat A–P–A**.
  - Worked check: a half-beat P at δ = 0.7 earns κ_b · ½ · 0.7 = 3.5 nats. That must just pay for one extra change (λ at P's position × λ_scale), the duration prior's hidden cost (§5.10) and the priors, minus the PT bonus.
  - **κ_b stays fixed.** λ_scale, the priors and the PT bonus are tuned relative to it to hit the targets.
- **Reliability.** A reliability diagram of the class posteriors (detail lattice) against GT on D2 + reviewed D1, with **ECE ≤ 0.1** (10 bins). The Detail slider (§5.11) ships only after this gate passes.

All weights and constants live in `decode/params.py` (`DecoderParams`, §4). They are tuned in WP14 (§13.5).

### 5.10 Decoding: semi-Markov on the sub-beat grid (WP7 `hsmm.py`)

```python
# Inputs: Evidence (E, E_short over C), grid, KeyTrack, DecoderParams P, optional rep (consensus class per cell, M5)
Ecum, Ecum_s = prefix sums of E and E_short over cells
seg(i, j, c) = (Ecum_s if dur_beats(i, j) <= 0.5 else Ecum)[j, c] − (same)[i, c]      # cells i..j−1

λ = λ_scale · {0: 2.0, 1: 2.5, 2: 3.0, 3: 4.5, 4: 6.0, 5: 4.0}   # change cost (nats) by cell_pos of boundary cell i
                                                              # 5 = free time: flat, no metric position
def T(cp, c, i):                                      # transition cp → c at boundary cell i
    if cp == c: return −0.3                            # re-entry of the same candidate, merged later
    k = local_key(i)
    return (−λ[cell_pos[i]] + P.w_prior · prior(cp, c, k)          # prior: table below, stacked, capped at +3.0
            + (0.6 if 1 <= bass_dist(cp, c) <= 2 else 0)            # stepwise bass
            + (0.5 if hcdf_peak_within(cell_t[i], 0.06) else 0))

def D(n, c, i):                                       # duration prior for a segment of n cells starting at cell i
    d = dur_beats(i, i + n)
    v = P.w_dur · lognormal_logpdf(d, median=2.0, sigma=0.9)        # w_dur = 1.0
    if c in PT and d <= 1.0: v += P.pt_bonus                         # +1.0
    if rep is not None and cls(c) == majority(rep[i:i+n]): v += P.rep_bonus   # repetition prior, +1.0 (M5)
    return v

Lmax = 4 bars of cells (64 at 4/4 with subdiv 4; 16 s in free time); minimum segment = 1 cell
# Viterbi (segmental), over region (r0, r1) — the whole song unless re-scanning:
Δ[i, c] = max_{cp ∈ C} δ[i, cp] + T(cp, c, i)         # computed once per boundary i: O(G·|C|²)
δ[j, c] = max_{1 ≤ n ≤ min(Lmax, j − r0)} ( (L0[c] if j−n == r0 else Δ[j−n, c]) + seg(j−n, j, c) + D(n, c, j−n) )
# boundary = (left, right): L0[c] = T(left, c, r0) (0 when left is None); the final step adds T(c, right, r1)
# backtrack → detail path (segments with label, cells)
# Forward–backward with logsumexp, same recursions, temperature 1.0 →
#   post[g, c] = P(cell g has label c),  post_cls[g, k] = Σ_{c ∈ k} post[g, c],  pb[g] = P(boundary at g)
# Complexity for 回家 (G ≈ 1,080 cells, |C| ≤ 250): G·|C|² ≈ 68 M + G·Lmax·|C| ≈ 17 M → about 1 s in NumPy
```

**The duration prior is also a change cost.** Splitting one segment into two adds a D term, which costs about 1–2.5 nats at w_dur = 1 (for example, about 1.1 nats when a 4-beat chord is split into 3½ + ½, before the PT bonus). The break-even table (§5.9) includes it.

**The prior** (`theory/priors.py`) works on degrees in the local key and is additive, in nats. **Positive rules stack** (every matching rule adds), **capped at +3.0** per transition. The two penalties apply only when no positive rule matched.

| Rule (cp → c) | Bonus |
|---|---|
| Both chords diatonic to the local key | +1.0 |
| Bass moves by 1–2 semitones and at least one of the chords is a slash chord (Canon, walk-ups, a descending chromatic bass under the tonic) | +1.5 |
| cp is dominant-family (7, 9, 13, 7b9, 7#9, 7b13, +7, or major acting as V) and root(c) = root(cp) + 5 (secondary dominant → target) | +1.5 |
| Tritone substitution: cp is dominant-family and root(cp) = root(c) + 1 (♭II7 → I, and ♭II7 of any target) | +1.5 |
| cp ∈ {sus4, 7sus4, 9sus4, 13sus4} with the same root, and c ∈ {maj, 7, 9, 13} | +2.0 |
| cp is IV/5-type (9sus4, 13sus4, or maj/maj7 with bass +2) and root(c) = bass(cp) + 5 | +1.5 |
| cp ∈ {dim, dim7, m7b5} and root(c) = root(cp) + 1 (leading-tone or passing diminished) | +1.5 |
| cp = I (maj), c ∈ {I+, I+7, I+maj7} on the same root, or cp ∈ {I+, I+7, I+maj7} and c = IV-family | +1.0 |
| Borrowed iv: cp ∈ IV-family ∪ ♭VII-family (maj, maj7, 7 or 9 on degree 4 or ♭7) and c ∈ iv-family (min, m6, m7 on degree 4). Covers 回家's A7 → Em6 (♭VII7 → iv). | +1.0 |
| c = ♭VII (maj, 7 or 9) and cp = IV-family or ♭VI | +0.8 |
| c = I and cp = ♭VII-family (completes ♭VI → ♭VII → I) | +0.8 |
| Chromatic descending minor 7ths: cp, c ∈ {m7, m9, m11} and root(c) = root(cp) − 1 (the trigram adds +1.5 when the direction continues, §6.4) | +1.0 |
| Line cliché: same root and bass, cp → c one step along min → mM7 → m7 → m6, or maj → aug → 6 → 7 | +1.0 |
| c is not diatonic and no positive rule matched | −1.0 |
| Same quality, root moves by a semitone, and no positive rule matched | −1.5 |

**The main layer** is decoded with the same code on the **beat grid**:
- cells are beats, and E is summed over each beat's sub-cells (E_short is not used: main segments are ≥ 1 beat);
- in free time, groups of 3 cells (0.6 s) act as beats;
- change costs are λ_main = 1.5 × λ_scale × {downbeat 2.0, mid-bar 2.5, beat 3.0, free time 4.0};
- the minimum segment is 1 beat.

### 5.11 Layers, salience and the Detail control (WP7 `layers.py`)

```python
detail = hsmm.decode(ev, grid, key, vocab, P, level="detail", rep=rep)
main   = hsmm.decode(ev, grid, key, vocab, P, level="main",   rep=rep)

# same_chord(R, M): same root, same bass and the same core quality. The core strips only the 9/11/13
#   extensions and alterations: add9→maj, madd9→min, 6/9→6, 9/13/7b9/7#9/7b13→7, maj9/maj7#11→maj7, m9/m11→m7.
#   sus, 6, 7, aug and dim are NOT extensions: B → B7 and Bsus4 → B are real changes.
for each run R in detail.segments:
    M = the main segment with the largest overlap with R
    if same_chord(R, M):
        mark R "main"; if R covers ≥ 50% of M and adds an extension, use R's richer label for M there
    elif R.dur_beats <= 2.0:
        R is a passing candidate
    else:  # long disagreement between the layers: both compared on the DETAIL lattice
        if mean_{g∈R} post_cls_detail[g, cls(R)] > mean_{g∈R} post_cls_detail[g, cls(M)] + 0.10:
            promote R into main (split M)
        else: keep M; add R's class to M.alternatives

for each passing candidate P with neighbours L, Rn (in detail):
    merged = best of (merge P into L, merge P into Rn)          # rescoring only touches 3 segments
    salience(P) = score(detail) − score(merged) + priors.trigram(L, P, Rn, key)     # §6.4
    conf(P)     = mean_{g ∈ P} post_cls[g, cls(P)]               # class level (§5.8)
    P.min_detail = clamp( σ(a·logit(conf(P)) + b·salience(P) + c), 0.05, 1.00 )      # a, b < 0
# post.finalize may later add the A–P–A demotion (§5.12 step 3); nothing is deleted
```

**The slider mapping.** (a, b, c) are fitted in M2 on the passing candidates of D2 and the reviewed D1 songs, each labelled hit or miss by the short-segment matcher (§13.2). The fit makes the passing chords shown at Standard (s = 0.5) and Detailed (s = 0.8) reach a precision of about **0.85** and **0.70**. Everything (s = 1.0) shows every passing candidate, so its precision (target about **0.50**) is set by λ_scale and the PT bonus, not by (a, b, c). The per-stop precision targets are also constraints in the tuning objective (§13.5).

**What the user controls**
- **Detail slider** s ∈ [0, 1], with stops Simple 0, Standard 0.5, Detailed 0.8 and Everything 1.0.
- A passing chord is shown when **s > 0 and s ≥ min_detail**. At s = 0 only the main layer is shown.
- Nothing the decoder found is deleted, so Everything shows every passing candidate, including demoted neighbour chords.
- This is filtered **client-side**, so moving the slider is instant.
- **"Re-scan this section (more sensitive)"** on a selected region calls `POST /api/tracks/{key}/redecode` with `region=[t0, t1]`. It re-decodes that region ±2 bars with λ_scale × 0.6, w_prior × 0.5 and the PT bonus at +1.5, with `boundary` fixed to the main chords just outside the region. New chords come back as dashed *suggestions* that the user accepts or rejects. It uses cached evidence only, so it takes under 2 s.
- **Bass passing tones** (§5.12 step 1) appear in the bass lane from Detailed (s ≥ 0.8), and are drawn in the grid as slash chords at Everything (s = 1.0). §6.2 says the same.

### 5.12 Post-processing: bass events, phantoms, spelling, numbers, voicings (WP7 `post.py`, WP6)

1. **Bass events.** A segment P whose non-bass pcs ⊆ a neighbour N's pcs, with only the bass different, is a **bass event**. It keeps N's label with P's bass as a slash (C → C/B) and is **never relabelled** (no "C°" over a held B).
   - If it is shorter than **min(½ beat, 0.3 s)**, it is a **bass passing tone**: merged into N and recorded in `bass_line` with `role="passing_tone"`. It shows in the bass lane from Detailed, and as the slash chord N/bass in the grid at Everything. (At 120 BPM half a beat is only 0.25 s, so fast walk-ups are no longer merged away.)
   - Otherwise it stays as a slash chord, and §5.11 decides its layer.
2. **Transition phantoms.** At a boundary, the old chord's reverb mixes with the new attack (B → Emaj7 can read as Emaj7/B or Bsus4). P between A and B is merged into the neighbour that shares the most pitch classes with it when **all** of these hold:
   - P ≤ ½ beat;
   - pcs(P) ⊆ pcs(A) ∪ pcs(B);
   - bass(P) ∈ {bass(A), bass(B)};
   - there is no new onset inside P with a pc outside pcs(A) ∪ pcs(B): piano-AMT onsets (dch > 0) from M5, a chroma-flux peak on the harmonic stem before that.

   Checked against 回家: B+7, A7, E/D#, B/D# and F#/A# all survive this rule.
3. **A–P–A neighbours.** P (≤ 1 beat) between two segments with the same label A is **never deleted**. It keeps its min_detail when one of these holds; otherwise it is **demoted**: min_detail += 0.3, capped at 1.0, so Everything still shows it.
   - bass(P) = bass(A) (pedal neighbour, 1 – 4/1 – 1), or bass(P) within ±1–2 semitones of bass(A);
   - P is IV of A or V of A (including V/7 and vii°/A);
   - P ∈ {A sus4, A sus2, A add9, A 7} **and** the added tone is supported by the harmonic or symbolic evidence (symbolic presence ≥ 0.5, or the lv@harmonic head for that tone ≥ 0.5), so a vocal melody note cannot create it.
4. **Merge** adjacent segments that have the same label.
5. **7th only in the bass.** When the bass is the chord's 7th and the upper evidence for that pc is < 0.5, display the triad over the bass (C/B, Am/G#), with the 7th chord as an alternative.
6. **Display label inside the class.** Pick the class member with the best sum of the key/idiom prior given the neighbours (`theory.priors`) and the model root head's mass on its root. For example, a passing °7 is named from the root that resolves up a semitone into the next chord.
7. **Spelling.** Use the local key at the segment start.
   - Diatonic roots use the scale letters.
   - Secondary dominants (dominant-family resolving down a 5th) and leading-tone diminished chords use the sharp side.
   - Borrowed ♭III, ♭VI, ♭VII and iv use the flat side.
   - Chord tones and the bass are spelled by interval from the root (E/G#, not E/A♭).
   - **Roots and basses avoid B#, E#, Cb and Fb by default**: #1°7 in B is shown as C°7. The "theoretical spelling" toggle (§6.3) restores them.
   - A double sharp or double flat is always shown as its enharmonic equivalent, in chord names and in diagram note names (D#7/F𝄪 is shown as D#7/G).
   - Key names: fewest accidentals; ties go to sharps (F# over Gb). There is a user toggle.
8. **Numbers.**
   - Nashville / 級數: `"1"`, `"5/7"`, `"6m"`, `"4maj7"`, `"b7(7)"`, `"3(7)/7"`, `"#1°7"`.
   - Roman: case follows quality, inversions use figured bass (6, 6/4, 6/5, 4/3, 4/2), and a non-chord bass uses `/5̂`.
   - Numbering **restarts at every key change** (`key.changes`), relative to the new key.
9. **Upper-structure display** (piano default ON). 9sus4 is shown as the upper triad over the bass (F#9sus4 → **E/F#**; G9sus4 → F/G), and 13sus4 as the upper maj7 (F#13sus4 → **Emaj7/F#**). The alternatives `F#11`, `F#9sus4` and `F#13sus4` are offered in the chord menu.
10. **Voicings** (`theory/voicing.py`; one implementation, run on the server):
    - LH: the bass note in MIDI 36–52, nearest to the previous LH note.
    - RH: 3–4 notes in MIDI 55–79, always including the 3rd (or sus tone), the 7th and the 9th when present. The 5th is dropped first.
    - **Inversions** (the bass is a chord tone): RH keeps the root and drops the bass pc, doubling the root when fewer than 3 notes remain (B/D#: LH D#, RH B–F#–B, not D#–F#, which sounds like D#m).
    - **Non-chord bass** (Em/A, F#/B, E/F#) and upper-structure slash chords: RH plays the full upper chord.
    - Rootless voicings are an option in Settings, off by default.
    - The RH inversion minimises cost = total semitone movement from the previous RH voicing, **−2** when the top note moves by 2 semitones or less, and +3 per minor 2nd below MIDI 60.
11. **Alternatives.** The top 3 *other classes* by posterior mass over the segment's cells, excluding classes equal to it at the current level, each with its display label (step 6). `equivalents` lists the other members of the chord's own class.
12. **Display variants.** `theory.display.build` writes `display.json`. For each transposition t ∈ 0..11 semitones and each accidental preference (auto, sharp, flat) it holds the key names, the labels at the three levels and the bass-line note names. Voicings are stored once per t: transposed, then the whole song shifted by −12, 0 or +12 to minimise notes outside the ranges. Degrees don't change with transposition and stay in `analysis.json`. Size: about 100–200 KB per song (est.). `server/edits.py` rebuilds it after every edit. **The client never spells chords.**

### 5.13 Output JSON (schema v1; `schema.py` → `shared/analysis.schema.json`, `shared/display.schema.json`)

```jsonc
{
  "schema_version": "1.0",
  "engine_version": "0.1.0",
  "params_hash": "a1b2c3d4", "calibration": "cal-0001", "mode": "full",
  "created_at": "2026-10-13T12:00:00Z",
  "source": { "key": "yt-Y7boUFbM76k", "type": "youtube", "url": "https://www.youtube.com/watch?v=Y7boUFbM76k",
              "title": "詩歌-回家 [歌詞]", "artist": null, "duration": 247.0 },
  "audio": { "play": "/api/tracks/yt-Y7boUFbM76k/audio", "stems": ["vocals","drums","bass","other"] },
  "display": "/api/tracks/yt-Y7boUFbM76k/display",
  "key":   { "global": { "tonic": "B", "mode": "major", "conf": 0.87 },
             "segments": [ { "start": 0.0, "end": 247.0, "tonic": "B", "mode": "major" } ],
             "changes": [] },     // e.g. { "t": 201.3, "bar": 55, "from": "B:major", "to": "C#:major", "kind": "step_up" }
  "tempo": { "bpm": 65.4,
             "subdivision": [ { "from_bar": 0, "cells_per_beat": 4 } ],
             "free_time": [] },   // e.g. [ { "start": 0.0, "end": 2.91 } ] for a rubato intro
  "meter": [ { "from_bar": 0, "num": 4, "den": 4 } ],
  "beats": [ { "i": 0, "t": 2.91, "bar": 0, "beat": 1, "conf": 0.93 } ],
  "bars":  [ { "i": 0, "start": 2.91, "end": 6.58, "beats": 4, "section": "Intro", "free": false } ],
  "sections": [ { "label": "Intro", "group": "A", "start": 2.9, "end": 32.4, "source": "user|auto" } ],
  "chords": [
    { "id": "c0042", "start": 61.38, "end": 61.84,
      "start_beat": 64.5, "end_beat": 65.0, "bar": 16, "beat_in_bar": 2.5, "dur_beats": 0.5,
      "harte": "B:aug(b7)", "root": "B", "quality": "+7", "bass": "B",
      "label": { "simple": "B", "standard": "B+", "full": "B+7" },
      "degree": { "root": 0, "bass": 0, "nashville": "1+(7)", "roman": "I+7" },
      "layer": "passing", "role": "chord", "min_detail": 0.35, "conf": 0.69, "salience": 2.8,
      "equivalents": [ "B:7(b13)" ],
      "alternatives": [ { "harte": "B:7", "conf": 0.18 }, { "harte": "B:maj", "conf": 0.07 } ],
      "voicing": { "lh": [47], "rh": [63, 67, 69] },
      "sources": { "lv": 0.61, "ace": 0.55, "btc": 0.40, "sym": 0.72, "bass": 0.88 },
      "edited": false, "suggestion": false }
  ],
  "bass_line": [ { "start": 61.38, "end": 61.84, "note": "B", "midi": 47, "conf": 0.9,
                   "role": "chord_bass", "min_detail": null } ],
                   // a bass passing tone has "role": "passing_tone" and a min_detail
  "lyrics": null,
  "edits": { "rev": 0, "n_ops": 0 },
  "gt": { "status": "none", "tapped_beats": false, "chart": null },   // status: "none" | "draft" | "reviewed"
  "stats": { "n_main": 112, "n_passing": 37, "runtime_s": { "separation": 48.1, "evidence": 21.0, "decode": 0.9 } }
}
```

```jsonc
// display.json — GET /api/tracks/{key}/display (theory/display.py)
{ "variants": {
    "0:auto": { "key_names": ["B"],                          // one per key segment
                "chords": { "c0042": ["B", "B+", "B+7"] },     // simple, standard, full
                "bass_line": ["B"] },
    "1:auto": { "key_names": ["C"], "chords": { "c0042": ["C", "C+", "C+7"] }, "bass_line": ["C"] }
    // … 36 variants: t = 0..11 × auto | sharp | flat
  },
  "voicings": { "0": { "c0042": { "lh": [47], "rh": [63, 67, 69] } },
                "1": { "c0042": { "lh": [48], "rh": [64, 68, 70] } } } }   // … 12 transpositions
```

**Field rules**
- `layer` is `"main"` or `"passing"`; `role` is `"chord"` or `"bass_event"`.
- `min_detail` is null for main chords.
- `harte` is the canonical label, used for evaluation and exports.
- `conf` is class-level (§5.8). `equivalents` are the other members of the chord's class; `alternatives` are other classes.
- `label.*` and `degree.*` are for transpose 0 and the default accidental preference. Every other transposition comes from `display.json`. The client never re-spells.
- Times are seconds in the original audio. Tempo changes are a playback concern only.
- `tempo.free_time` regions are drawn unmetered; their bars have `"free": true`.
- `key.changes` marks key changes; Nashville and Roman numbers restart there.
- `sections[].group` links repeats of the same material (A, B, C …); `source` is `"auto"` (§5.15) or `"user"`.
- **Edits** are `cache/<key>/edits.jsonl`, an append-only list of `EditOp` lines (§4).
  - The client POSTs new ops with client-made ids. The phone's offline queue replays them on reconnect, and duplicates are ignored by id.
  - The server assigns `seq` and applies every non-reverted op in `seq` order; on the same target, the later op wins. It then re-spells and re-voices (`server/edits.py` → `theory.display.build`) and returns the analysis, the display file and the new `rev`.
  - Undo is a `revert` op. Nothing is ever rewritten, so there is no last-writer-wins on the whole list.
  - `GET /api/tracks/{key}` returns the analysis with the edits applied, plus `original` on request.
- **Every export uses the edited analysis.**
- `gt` is the ground-truth status of this song (§13.1).
- `.lab` export: `start end harte` per line, main + passing at the chosen detail level.

### 5.14 Runtime, VRAM and cache

**Full mode on the RTX 4070 laptop** (estimates; M0 and M1 measure them):

| Step | Time |
|---|---|
| Download + decode | 5–15 s |
| Worker cold start (first job after start-up only; the worker is pre-warmed) | 15–30 s |
| htdemucs_ft | 40–60 s |
| Beat This! | 2–5 s |
| lv-chordia × 2 inputs | 15–30 s |
| ACE × 2 (M5) | 10–15 s |
| BTC-CL (M5) | under 5 s |
| Piano transcription (M5) | 10–20 s |
| Bass + key + fusion + decode (including the octave and key loops) | under 10 s |

**Target:** ≤3 min for a 4-minute song.

**VRAM**
- Resident weights are small next to the activations. The known checkpoints total under 0.5 GB (Beat This! 81 MB, ACE 57.6 MB, BTC-CL 35.9 MB, piano transcription 172 MB), plus htdemucs_ft and lv-chordia; M0 measures the total.
- A peak below 5 GB of the 8 GB is realistic for songs up to about 8 min, as long as the demucs mix tensor stays on the CPU and lv-chordia runs in chunks. With chunking, longer songs (up to the 30 min limit) don't raise the peak; the M0 spike measures a 10-minute file.
- After a CUDA OOM:
  1. `empty_cache()`, then retry with demucs `segment=6`;
  2. unload the chord models during separation;
  3. use the CPU for that stage (lv-chordia in a one-off subprocess with `CUDA_VISIBLE_DEVICES=""`, §5.5).

**Cache layout**
- `cache/<key>/`: `meta.json source.* audio.wav play.m4a stems/*.flac stems/mix_*.m4a`.
- `evidence/*.npz`: model outputs, versioned per adapter.
- `notes.npz`, `rhythm.json`.
- `analysis.v<ENGINE>-<params_hash>.json`, `display.json`, `edits.jsonl`, `gt/`, `lyrics.lrc`.

A new `ENGINE_VERSION` re-runs only the stages whose adapter versions changed.

**Storage** (`server/storage.py`, WP9; C: has only 37 GB free)
- A song with WAV stems takes about 250 MB; FLAC stems cut that (M1 measures by how much).
- **Retention:** stems and evidence of songs not opened for **30 days** are deleted (Settings → Storage).
- **Size cap: 10 GB** for `cache/`. When it is exceeded, the least recently opened songs lose their stems, then their evidence, then `audio.wav` (re-decoded from `source.*` on demand). `analysis*.json`, `display.json`, `edits.jsonl`, `play.m4a` and `source.*` are always kept.
- The UI warns when C: drops below 15 GB free, and the server refuses new jobs below 5 GB.

### 5.15 Repeats and automatic sections (WP7 `sections.py`; engine in M5, UI in M7)

Worship choruses repeat 3–6 times. Decoding each repeat on its own wastes that redundancy, and Chordify-style section labels need the same analysis.

1. **Self-similarity.** Per beat, concatenate the harmonic-stem chroma (12) and the class posteriors of a first detail decode reduced to root + maj/min (25). Build the cosine self-similarity matrix and its time-lag form. Repeats are diagonal stripes at bar-multiple lags, ≥ 4 bars long, with mean similarity ≥ 0.7.
2. **Groups.** Bars linked by stripes are clustered into groups (A, B, C …), and each run of bars in one group becomes a section with `group` set.
3. **Repetition prior.** For each aligned beat position of a group, the consensus class is the class that ≥ 2/3 of the repeats agree on; otherwise there is none. `rep[g]` carries it per cell, and the HSMM adds **+1 nat per segment** whose class matches it (`D`, §5.10). It stays soft on purpose: 回家's third chorus differs from the others, and the evidence must still win there.
4. **Labels**, from position and vocal-stem energy:
   - before the first vocal entry → Intro; after the last → Outro (Tag when it repeats the end of a chorus);
   - groups without vocals between sung sections → Interlude;
   - the sung group with the most repeats (ties: higher vocal and mix energy) → Chorus;
   - a group that always directly precedes the Chorus → Pre-chorus;
   - a sung group that appears once, late in the song → Bridge; other sung groups → Verse.
5. `set_section` edit ops override the auto labels. The truck-driver test (§5.4) uses these sections.

---

## 6. Chord vocabulary and simplify levels (WP6 `simplify.py`)

### 6.1 Levels

| Internal quality | Simple | Standard | Full |
|---|---|---|---|
| maj, min | maj, min | same | same |
| 5 (no 3rd) | maj | 5 | 5 |
| dim | dim (°) | dim | dim |
| aug | maj | aug (+) | aug |
| sus2, sus4 | maj | sus2, sus4 | same |
| 6, m6 | maj, min | 6, m6 | same |
| 7, maj7, m7 | maj, maj, min | same | same |
| m7b5 (ø), dim7 | dim | m7b5, dim7 | same |
| mM7 | min | m(maj7) | same |
| 7sus4 | maj | 7sus4 | same |
| +7, +maj7 | maj | aug | +7, +maj7 |
| add9, madd9 | maj, min | add9, m(add9) | same |
| 6/9 | maj | 6 | 6/9 |
| 9, maj9, m9 | maj, maj, min | 7, maj7, m7 | 9, maj9, m9 |
| m11 | min | m7 | m11 |
| maj7#11 | maj | maj7 | maj7#11 |
| 9sus4 | maj on the chord root (F/G → G) | upper-structure slash (F/G) | F/G (alternatives G9sus4, G11) |
| 13sus4 | maj on the chord root (Emaj7/F# → F#) | upper-structure slash (Emaj7/F#) | Emaj7/F# (alternative F#13sus4) |
| 13, 7b9, 7#9, 7b13 | maj | 7 | 13, 7b9, 7#9, 7b13 |

**Bass per level**
- Simple: never shows a bass.
- Standard: shows a slash for inversions (bass is a chord tone) and for upper-structure chords.
- Full: shows every slash, including non-chord bass (Em/A, F#/B).
- When the 7th sounds only in the bass, Standard and Full show the triad over the bass (C/B, §5.12 step 5).
- The **Slash chords** toggle overrides this per view.

### 6.2 Presets (the chip next to the Detail slider)

| Preset | Level | Detail | Notes |
|---|---|---|---|
| Simple | Simple | 0 | Comparable with Chordify's output (baseline B5, §13.4) |
| Standard | Standard | 0.5 | Default for first-time viewing |
| Detailed | Full | 0.8 | **Default for Nathan.** The bass lane shows bass passing tones. |
| Everything | Full | 1.0 | Also draws bass passing tones as slash chords in the grid, and shows demoted neighbour chords |

### 6.3 Notation toggles

- Letters (B, F#/A#), Nashville / 級數 (1, 5/7, 6m), or Roman numerals (I, V6, vi).
- Sharps/flats preference.
- `maj7` vs `Δ7` vs `M7`.
- `°` / `ø` symbols or `dim` / `m7b5`.
- Theoretical spelling (allow B#, E#, Cb and Fb roots): off by default.
- Rootless RH voicings: off by default.

### 6.4 Idiom trigrams for salience (`theory.priors.trigram`, WP6)

Written as degrees in the local major key. Minor keys are mapped through their relative major. `trigram(L, P, R, key)` returns +1.5 when (L, P, R) matches an idiom below, plus +1.5 when bass(L) → bass(P) → bass(R) is monotone with steps of at most 2 semitones.

| Idiom | Pattern |
|---|---|
| Augmented passing | 1 → 1+(7) → 4(maj7) |
| Borrowed | 4maj7 → b7(7) → 4m(6/7) (回家: Emaj7 → A7 → Em6) |
| Borrowed ♭VI–♭VII | b6 → b7 → 1 |
| Descending Canon | 1 → 5/7 → 6m; 6m → 6m/5 → 4; 4 → 1/3 → 2m |
| Descending chromatic bass under the tonic | 1 → 1/7 → 1/b7 → 4/6 (scored on each consecutive triple) |
| Walk-up | 2m → 1/3 → 4; 4 → 5 → 6m; 5/7 → 1 |
| Rising minor 7ths | 2m7 → 3m7 → 4; 6m7 → 7m7 → x |
| Chromatic descending minor 7ths | 3m7 → b3m7 → 2m7 (matches only when the direction continues) |
| Passing diminished | x → #x°7 → (x+2)m (all degrees) |
| Secondary dominant | 1 → 3(7)/#5 → 6m (and other bass-step forms) |
| Tritone substitution | 2m7 → b2(7) → 1 |
| Line cliché | 1m → 1m(maj7) → 1m7 → 1m6; 1 → 1+ → 1(6) → 1(7) (scored on each consecutive triple) |
| Sus resolution | Xsus4 → X |
| Four over five | 4/5 → 1 |
| Pedal neighbour | 1 → 4/1 → 1 |
| Royal road | 4maj7 → 5 (or 5/4) → 3m7 → 6m |

These are starting values. WP14 reweights them from Nathan's annotated songs.

---

## 7. Feature map: every Chordify feature, plus what goes beyond it

**Phases.** **v1** = milestones M3–M4, the MVP Nathan uses every day. **v1.1** = M7, Chordify Premium parity. **v2** = M8, the Toolkit and the gated models. The engine milestones (M1, M2, M5) and the platform milestone (M6) are in §14.

**Said plainly:** v1 does **not** reach full Premium parity. MIDI, PDF/print, setlists, offline mode, lyrics, audio pitch shift, gapless loops, the metronome and stem mixes arrive in v1.1, and the Toolkit in v2.

**The phone app needs the laptop** switched on, awake and reachable (over Tailscale). The only exception is songs saved offline (v1.1), which play without it.

### 7.1 Chordify Free and Basic

| Chordify feature | Our implementation | Ships / WP |
|---|---|---|
| Chords for any song (36M catalog) | Any YouTube link, yt-dlp-supported URL or uploaded file is analysed on demand and cached in the local library. **Deezer is unsupported** (DRM). A search box that runs `ytsearch8:<query>` (yt-dlp, `extract_flat`) when the input isn't a URL. | Analysis v1 (M3); search v1.1 (M7) / WP1, WP9, WP12 |
| Animated chord diagrams (guitar/piano/ukulele) | **Piano only.** v1: NOW / NEXT diagrams with voice-led voicings, the LH bass in its own colour, and RH chord tones. v1.1: the full animated strip and the Summary view. | v1 (M3); v1.1 (M7) / WP11 |
| YouTube links | Yes, plus SoundCloud and Bilibili via yt-dlp. Mix and playlist links are reduced to the single video. | v1 (M1 CLI, M3 UI) / WP1 |
| Library sync between desktop and mobile | One server library (SQLite). Desktop and phone see the same songs, edits and (v1.1) setlists. | v1 (M3; phone from M6) / WP9, WP12 |
| Simplify Chords | Simple / Standard / Full levels plus the Detail slider (§6). | v1 (M3) / WP6, WP11 |
| Multi-string tuner (free) | Chromatic tuner from the mic (polyphonic guitar mode isn't needed for piano). | v2 (M8) / WP15 |
| Daily limit / Basic "unlimited" | No limits. | v1 |

### 7.2 Chordify Premium

| Chordify feature | Our implementation | Ships / WP |
|---|---|---|
| Transpose (±1 semitone per click, key shown) | −12…+12 semitones. Key badge "Key B → C (+1)". Every label and diagram is re-spelled from the server's precomputed `display.json`. v1: labels and diagrams only; the audio keeps its pitch. v1.1: **real audio pitch shift** (signalsmith-stretch), on by default with an "audio follows transpose" toggle. | Labels v1 (M3); audio v1.1 (M7) / WP6, WP11 |
| Key changes | A key badge per key segment, shown at the bar where the key changes. Nashville and Roman numbering restart there. Transpose shifts every segment by the same amount (B → C and C# → D at +1), and Play-in-key keeps the same offset across the change. The modulation detector improves in M5. | v1 (M3); detector M5 / WP6, WP11 |
| Capo | Piano equivalent: **Play in key** (keyboard transpose). Chords are shown in an easier key, e.g. C, with the hint "set your keyboard's Transpose to −1". The sounding audio stays the same. | v1 (M3) / WP11 |
| Tempo 50–200% in 5% steps | **25–200% in 1% steps**: ±5% buttons and a typed % or BPM field, pitch preserved. v1 uses `<audio>` `playbackRate` with `preservesPitch`; v1.1 moves it onto signalsmith-stretch. | v1 (M3) / WP11 |
| Loop: drag over beat squares | v1: drag over cells and **A/B buttons** (seek-based loop). v1.1: **gapless** loops, count-in (0/1/2 bars), a loop counter and a **speed trainer** (+5% per pass up to a target). | v1 (M3); v1.1 (M7) / WP11 |
| Song and chord volume (+ metronome) | v1: song volume. v1.1: a mixer with the song **or a stem practice mix** (No vocals / Piano focus / Bass focus / custom), chord-synth volume, and a metronome that follows the detected beats. | v1 (M3); v1.1 (M7) / WP11, WP2 |
| MIDI download (DAW tempo-mapped + fixed tempo, bassline track) | Both versions. Tracks: Chords (the actual piano voicings), Bass, Markers (sections). Transpose, level, detail **and edits** are applied. The tempo is in the file name (`回家 - B - 65bpm - daw.mid`). | v1.1 (M7) / WP10 |
| PDF / print (chord sheet + diagrams) | Print view (A4 `@page`): grid sheet, Chords & Lyrics sheet, **Nashville chart**, piano-diagram page. Follows transpose, play-in key, level, notation **and edits**. Print-to-PDF works from WebView2, Android and iOS. | v1.1 (M7) / WP12 |
| Other exports (competitors offer MusicXML, iReal Pro, ChordPro) | MusicXML, ChordPro, **iReal Pro** (import link; sub-beat chords rounded to the beat), text, `.lab` and JSON, all with edits applied. | `.lab`/JSON v1 (M1); the rest v1.1 (M7) / WP10 |
| Setlists (unlimited, offline, rename) | Setlists where **each entry stores its own key, preset, tempo and loop**. A whole-setlist PDF. Available offline. | v1.1 (M7) / WP12 |
| Offline mode (apps only, no original audio) | PWA offline **with audio**: "Save offline" stores the analysis, `display.json` and audio in Cache Storage. Without the laptop, the phone can play only these songs. | v1.1 (M7) / WP13 |
| Upload own files (≤20 min, ≤32 MB) | Any format ffmpeg can decode (mp3, m4a, wav, flac, ogg, mp4, mov), up to 30 min (configurable) and 1 GB. Drag-and-drop or the phone's file picker. | v1 (M1 CLI, M3 UI) / WP1, WP12 |
| Edit chords (web only) | Editor on desktop **and** phone. v1: chord picker with **top-3 alternatives** and equivalent spellings, typed entry (`F#/A#`, `C#m7`, `N`, parsed by the server), root/quality/bass pickers, delete change, rest, accept/reject suggestions, undo, half/double tempo. Edits are append-only operations shared by desktop and phone. v1.1: **any meter** (3/4, 4/4, 6/8, 12/8, mixed), shift bar lines, **beat-grid editing** (move beat, tap tempo, subdivision, free time), preview sound in the pickers, and history with a diff against the original. | v1 (M4); v1.1 (M7) / WP12, WP9 |
| Edits flow into MIDI and PDF | Every export and print view uses the edited analysis. | v1.1 (M7) / WP10, WP12 |
| Views: Animated diagrams, Summary, Grid, Chords & Lyrics | v1: Grid (with a **bass lane**) and NOW / NEXT diagrams. v1.1: Summary, Chords & Lyrics, and a **Nashville chart** view. | v1 (M3); v1.1 (M7) / WP11, WP12 |
| Chords & Lyrics details (text size, karaoke, pinnable diagrams) | Chords anchored to the lyric character sounding at the chord's start (per character for Chinese), adjustable text size, karaoke highlighting of the current line, and pinnable diagrams. | v1.1 (M7) / WP12 |
| Rename songs | Title and artist editing in the library. | v1 (M3) / WP12 |
| Edits used to train the algorithm | **Ground-truth mode** (§8.4): reviewed songs export a `.lab` and tapped beats into `eval/gt/`, which tune the decoder (§13). | v1 (M4) / WP12, WP9, WP14 |
| Re-analysis (users ask; Chordify can't) | Re-analyse at any time, re-decode with new parameters, and re-scan a region. | v1 (M3) / WP9 |
| Premium demo channels | Not applicable. | — |

### 7.3 Chordify Premium Plus (Toolkit)

| Chordify feature | Our implementation | Ships / WP |
|---|---|---|
| Live chord detection (mic) | Mic → AudioWorklet (16 kHz mono) → WebSocket `/api/live` → live worker. lv-chordia runs on a rolling 4 s window every 0.5 s and the current chord + bass come back. Target latency ≤1 s (est.). | v2 (M8) / WP15 |
| Chromatic tuner (adjustable A4, cents/Hz) | In the browser (MPM pitch detection), A4 415–466 Hz, cents and Hz readout. | v2 (M8) / WP15 |
| Metronome with drumbeat (4/4, 3/4, 1/1) | Standalone metronome in any meter (incl. 6/8, 12/8) with synthesised kick/snare/hat patterns (ballad, pop 4/4, 6/8) and a visual flash. The in-song metronome is in v1.1. | v2 (M8) / WP15 |
| Chord trainer | Uses **Web MIDI** from Nathan's digital piano (Chrome/Edge desktop and Android; iOS Safari has no Web MIDI, so mic detection there). Shows a chord, checks the pitch classes and optional bass, and scores speed and accuracy. Also drills the song's own chords. | v2 (M8) / WP15 |
| Generate Lyrics | v1.1: paste or import LRC, **tap-to-sync** (tap at each line start while playing), optional LRCLIB lookup through the server's proxy. v2: ASR on the vocal stem (model chosen and verified then; Cantonese support is a known weak spot). | v1.1 (M7) / WP12, WP9; v2 (M8) / WP16 |
| Learn Songs (Beta) | **Practice mode** per section: loop ladder (70 → 85 → 100%), and **step mode**, where playback waits at each chord change until it is played on the MIDI keyboard. | v2 (M8) / WP15 |
| Search for songs using chords you know | Library filter "songs whose chords ⊆ {selected chords}" at the chosen level (`GET /api/tracks?chords_subset=`). | v1.1 (M7) / WP9, WP12 |

### 7.4 Beyond Chordify

1. **Passing-chord layer.** Sub-beat chords, class-level confidence, the Detail slider, and a "Re-scan section" button (§5.11). v1.
2. **Slash chords and inversions**, with a **bass lane** that shows walking-bass passing tones. v1.
3. **32-quality vocabulary** (add9, 9sus4 and 13sus4 as upper-structure slash chords, +7, m7b5, dim7, 6/9, m11, maj7#11, 7b13, …). v1.
4. **Nashville / 級數 and Roman numerals**, plus local-key display with renumbering at key changes (catches a last chorus raised a step). v1.
5. **Piano voicing diagrams** with LH/RH split and voice leading, plus a "show note names / degrees" toggle. v1.
6. **Top-3 alternatives, equivalent spellings** and confidence shading. Low-confidence cells are dashed. v1.
7. **Ground-truth mode** with chart import and tapped beats, so Nathan's corrections become test data. v1.
8. **Repeat-aware decoding** and automatic section labels (§5.15). Engine M5, UI v1.1.
9. **Stem practice mixes**, and real pitch shift on transpose. v1.1.
10. **Editable beat grid**, any meter, half/double tempo and free-time passages. Half/double and free-time display v1; grid editing v1.1.
11. **Exports:** MusicXML, ChordPro, iReal Pro, `.lab`, JSON and a Nashville PDF. `.lab`/JSON v1, the rest v1.1.
12. **zh-TW / English UI**, with Traditional Chinese print support. v1.
13. **Play-along scoring** with a MIDI keyboard. v2.

---

## 8. UI

### 8.1 Desktop layout (≥1024 px)

The mock shows v1.1. v1 (M3) has no Count-in, Mix or "♪audio" transpose checkbox, and its Export menu holds `.lab` and JSON only.

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ ☰  Nathan Chordify   [ 搜尋或貼上 YouTube 連結 / drop audio file …………… ]  [分析 Analyze]  ⚙ │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ 回家 · Eternity Girls     Key B ▸ play in C (kbd −1)   ♩ 65   4/4   [A B C | 1 2 3 | I ii]  │
│ Sections:  Intro  Verse  Pre  Chorus  Interlude  Pre  Chorus  Interlude  Chorus  Tag       │
├──────────────────────────────────────────────────────────────┬────────────────────────────┤
│ CHORD GRID (4 bars per row, one square per beat)             │ PIANO                      │
│ ║ B    │  ·  │ B+7┊·│  ·  ║Emaj7│  ·  │ A7  │ Em6 ║ …         │  NOW  Emaj7               │
│ ║ B    │  ·  │B/F#│ B7  ║Emaj7│  ·  │ A7  │ Em7 ║ …         │  ▁▁█▁▁█▁█▁▁ (LH E ● RH)    │
│   (passing chords in amber, smaller; split squares for       │  NEXT A7    THEN Em6        │
│    sub-beat chords; dashed = low confidence; ▪ = now)        │  [Animated | Summary]       │
│ bass lane: B · · · A#│ E · A · E …  (walking tones in grey)  │                             │
├──────────────────────────────────────────────────────────────┴────────────────────────────┤
│ ⏮ ▶ ⏭  1:01.4 / 4:07  ━━━━━━━━●━━━━━━━━━━━━━━   Loop ⟳ [A][B]  Count-in 1 bar   ✎ Edit    │
│ Transpose [−] B [+] ♪audio ☑   Play in [C ▾]   Tempo [−] 100% [+] (65 BPM)   Mix ▾        │
│ Detail ○───●──────○  Simple · Standard · Detailed · Everything   Level [Full ▾]  Slash ☑  │
│ Export ▾ (MIDI · MusicXML · ChordPro · iReal · PDF/Print · .lab)  Re-scan section ⟲       │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

**Interactions**
- **Clicking**
  - Click a cell to seek. Drag across cells to set a loop.
  - Right-click (long-press on a phone) a chord to open the chord menu: alternatives, equivalent spellings, edit, play chord (v1.1), show voicing, "Re-scan around here".
- **Keyboard shortcuts**
  - Space: play/pause. ←/→: previous/next bar.
  - `[` and `]`: loop A/B.
  - `+`/`−`: transpose. Shift `+`/`−`: tempo.
  - `E`: edit mode. `D`: cycle the Detail stops.
- **Grid layout**
  - 4/4 uses 4 bars per row. 6/8 and 12/8 use 2 bars per row, with dotted-quarter beat squares split in three.
  - Free-time passages are drawn as an unmetered timeline strip with no bar lines.
  - A key change shows a badge at its bar ("→ C#"); Nashville numbers restart there.
  - The current row stays in the upper third of the view.

**Library page** (WP12)
- v1: recent songs, upload, rename, and a job panel showing live progress (stage names and %) from SSE.
- v1.1: search, filters (key, BPM, analysed date, "fits my chords") and setlists.

### 8.2 Phone layout (portrait, 360–430 px, with a 16 px gutter)

```
┌──────────────────────────────┐
│ ←  回家          Key B ♩65  ⋯ │
│ [Grid] [Diagrams] [Lyrics]    │
│ ║ B   · │B+7┊· ·║Emaj7 · A7 Em6║   2 bars per row; passing chords split squares
│ ║ B   · │B/F# B7║Emaj7 · A7 Em7║
│ …                             │
│ ┌ NOW Emaj7 ─────┐┌ NEXT A7 ┐ │   collapsible mini piano diagrams
│ └────────────────┘└─────────┘ │
│ ▶  1:01 ━━━━●━━━━━━━━  ⟳ A B   │   sticky transport
│ [−] T [+]   100%   Detail ▾   │   bottom sheet: level, slash, notation, mix, export
└──────────────────────────────┘
```

- Landscape uses 4 bars per row with the diagrams on the right.
- **Wake Lock** keeps the screen on during playback.
- Every tap target is at least 44 px.
- On iOS, playback starts from a tap ("Tap to start"), see §8.3.
- The library and setlist pages are lists, with a "Save offline" toggle on each song (v1.1).

### 8.3 Playback engine (WP11 `player/`)

**v1 (M3): `<audio>` engine**
- `<audio src="play.m4a">` with `playbackRate` 0.25–2.0 and `preservesPitch = true`.
- A/B and drag loops: the rAF loop seeks back to A when `currentTime` passes B (not gapless; about ±30 ms).
- **Highlighting:** a `requestAnimationFrame` loop reads `audio.currentTime`, binary-searches the cells and toggles CSS classes. **No React re-render per frame.**
- Transpose changes labels and diagrams only (from `display.json`).

**v1.1 (M7): signalsmith-stretch engine**
- **Signal chain:** fetch `play.m4a` (or a practice mix) → `decodeAudioData` → **signalsmith-stretch** AudioWorklet node → song gain. Alongside it run a ChordSynth gain (an additive electric-piano voice playing the server's voicings, scheduled 100 ms ahead) and a Metronome gain (clicks on the detected beats, accent on downbeats). All feed the destination.
- **Controls:**
  - Tempo → `schedule({rate})`.
  - Transpose with audio → `schedule({semitones})`.
  - Loop → `schedule({loopStart, loopEnd})` (gapless).
  - Count-in → metronome clicks, then start.
- **Clock:** the song position comes from the stretch node's input time. WP11 checks this API against the 1.3.2 README; the fallback is integrating `AudioContext.currentTime` × rate.
- The package ships no TypeScript types; `web/src/shims.d.ts` declares `declare module 'signalsmith-stretch'`.
- **The `<audio>` engine stays as the fallback**: no AudioWorklet or WASM, songs over 12 min on a phone, and iOS Safari until AudioWorklet + WASM is verified on a real device.

**iOS audio**
- Set `navigator.audioSession.type = "playback"` where it exists, so the ring/silent switch doesn't mute Web Audio.
- Create or resume the `AudioContext` only from a user tap (a "Tap to start" overlay).

### 8.4 Editor and Ground-truth mode (WP12, M4)

**Editor** (desktop and phone)
- The chord menu sets a chord: alternatives, equivalent spellings, root/quality/bass pickers, and typed entry, which `POST /api/theory/parse` turns into Harte and labels (or an error).
- Delete a change, insert a rest, accept or reject a suggestion, undo (a `revert` op), half/double tempo.
- **Keyboard-first:** type a chord name and press Enter to set the selected beat; Tab moves to the next beat; Ctrl+Z undoes.
- Every action is one `EditOp` appended through `POST /api/tracks/{key}/edits` (§5.13).

**Ground-truth mode** (per song; the route to D1, §13.3)
- **Tap beats:** play at 70% and tap Space on every beat; nudge a beat with ←/→. Saved as the song's GT beats, independent of our beat tracker.
- **Import chart:** paste bar text (§0.1 format) or a `.lab`. It is aligned to the tapped beats (`theory.chart.align`) and loaded as a draft edit layer, with H/M/L flags per bar (L bars highlighted).
- New songs start from **our main layer only** (no passing chords), so our passing-chord guesses don't anchor the annotation.
- **Bar loop at 70%** on one key. **Synth A/B:** plays the current GT chord on a minimal Web Audio voice, alternating with the audio (the v1.1 ChordSynth replaces it later).
- **Save as ground truth** writes `eval/gt/<key>.lab`, `<key>.beats` and `<key>.chart.txt`. Ticking "Reviewed" sets `reviewed: true` in `eval/songs.yaml`.

---

## 9. Windows desktop app (WP13; M6)

- **`nathan_chordify/desktop.py`**
  1. **First lines, before any other import:** `if sys.stdout is None: sys.stdout = sys.stderr = open(LOG, "a", encoding="utf-8", buffering=1)` (`logsetup.py`, LOG = `logs\desktop.log`). Under `pythonw.exe`, `sys.stdout` is None and uvicorn's log formatter calls `isatty()`, which crashes with `ValueError: Unable to configure formatter 'default'` (verified). With the redirect, `/health` returns 200 under pythonw (verified). The worker entry does the same with `logs\worker.log`.
  2. `config.py` sets `HF_HOME`, `TORCH_HOME`, `HF_HUB_DISABLE_PROGRESS_BARS=1`, `HF_HUB_DISABLE_SYMLINKS_WARNING=1` and `MPLBACKEND=Agg` before torch or huggingface_hub is imported. All weights are prefetched (§12), so nothing downloads at run time.
  3. All start-up code sits under `if __name__ == "__main__":`, because `spawn` re-imports the main module. The worker target is `nathan_chordify.server.worker:main`.
  4. If `GET http://127.0.0.1:8765/health` already answers, reuse that server.
  5. Otherwise start `uvicorn.Server(Config(app, host="127.0.0.1", port=8765))` in a daemon thread. The FastAPI lifespan spawns the GPU worker and **pre-warms** it in the background (torch import, lv-chordia and Beat This! loaded; 15–30 s).
  6. Poll `/health` for up to 30 s. The window opens before the pre-warm finishes; jobs wait for it.
  7. `webview.create_window("Nathan Chordify", "http://127.0.0.1:8765", width=1400, height=900, min_size=(900, 600))`, then `webview.start(private_mode=False, storage_path=%LOCALAPPDATA%\Nathan_chordify\webview)`, so IndexedDB and settings persist.
  8. When the window closes, stop the worker and the server.
- **`Nathan_chordify.bat`:** `start "" "%~dp0.venv\Scripts\pythonw.exe" -m nathan_chordify.desktop`.
- **`Nathan_chordify_server.bat`:** headless server only, for phone use while the window is closed.
- **`scripts/make_shortcut.ps1`** creates a Desktop `.lnk` with WScript.Shell, pointing at `pythonw.exe -m nathan_chordify.desktop`, with the icon from `web/public/icons/icon.ico`.
- **Child processes:** every ffmpeg and pip subprocess runs with `creationflags=subprocess.CREATE_NO_WINDOW`, so no console window flashes under pythonw.
- **"Update downloader"** first stops the GPU worker (a loaded `.pyd` can't be replaced on Windows), then runs `python -m pip install -U "yt-dlp[default]"` in a separate process, then restarts the worker.
- **While a job runs,** the worker calls `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` so the laptop doesn't sleep mid-analysis, and releases it afterwards.
- **Microphone in WebView2** (v2 tuner and live detection). Verified in pywebview's WebView2: `isSecureContext`, `getUserMedia`, AudioWorklet, WASM, service worker and IndexedDB all work. pywebview has no `PermissionRequested` handler, so WebView2 shows its default permission prompt. Optional: auto-allow the microphone for `127.0.0.1` by subscribing to `CoreWebView2.PermissionRequested` on the UI thread through `window.native.Invoke(...)`. If the prompt misbehaves, a menu item "Open in Edge" opens `http://127.0.0.1:8765` in the browser.
- **Not used:** PyInstaller, Electron, Tauri (see RESEARCH §4.9).

---

## 10. Phone PWA, HTTPS and firewall (WP13; M6, offline in M7)

**PWA**
- `vite-plugin-pwa` (`pwa.config.ts`), with a manifest (name, zh-TW short name, icons 192/512/maskable, `display: standalone`) and Workbox:
  - precache the app shell;
  - `NetworkFirst` for `/api/tracks*` JSON;
  - `CacheFirst` with the range-requests plugin for audio and `display.json`, **only** for songs marked "Save offline" (v1.1);
  - IndexedDB (idb-keyval) queues edit ops made offline and replays them on reconnect (ids make the replay idempotent).
- The phone doesn't analyse anything; it submits jobs and watches SSE progress. **It needs the laptop on, awake and reachable**, except for songs saved offline.
- **Content-Security-Policy** (set by the server): `default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; media-src 'self' blob: data:; img-src 'self' blob: data:; connect-src 'self'`. It must allow `blob:`, `data:` and `'wasm-unsafe-eval'` for signalsmith-stretch's WASM and decoded audio.
- **iOS audio:** `navigator.audioSession.type = "playback"` and a tap to start (§8.3).

**Before Tailscale: Surfshark.** Surfshark's WireGuard VPN is up on the laptop, and two WireGuard VPNs at once commonly conflict. **Standing rule: Surfshark is never paused or disconnected, and we never suggest it** (Nathan travels and uses several devices). M6 must therefore find a phone path that works with Surfshark on: first test Tailscale alongside Surfshark as-is; if they conflict, present the options to Nathan (e.g. a Surfshark split-tunnel exclusion for Tailscale only, if he wants that) and let him decide. On the phone, if Surfshark also runs there, Tailscale and Surfshark cannot both be active; this is a real limitation to show Nathan before M6, not something to work around silently.

**Primary: Tailscale serve.** No firewall rule needed. Nathan does these steps himself; the agents do not create accounts or change settings.
1. Install Tailscale for Windows and sign in.
2. In the admin console, enable **MagicDNS** and **HTTPS Certificates**. Rename the PC to something neutral (e.g. `music-pc`), because names appear in Certificate Transparency logs.
3. Run `tailscale serve --bg 8765`. This persists, and gives `https://music-pc.<tailnet>.ts.net`. SSE passes through (verified in the review).
4. Install Tailscale on the phone with the same account. Open the URL, then use "Add to Home Screen" (iOS Safari) or "Install app" (Android Chrome).
5. uvicorn stays bound to `127.0.0.1`, so nothing is exposed on the LAN. Settings → Phone shows the URL and a QR code.

**Fallback: mkcert on the LAN — deferred.** It is built only if Tailscale can't be used (decided in M6). It also needs the Wi-Fi profile switched from Public to Private and Surfshark's "Invisible on LAN" turned off. The plan (`scripts/mkcert_setup.ps1`, run by Nathan):
1. Install mkcert v1.4.4. Run `mkcert -install`, then `mkcert -cert-file certs\lan.pem -key-file certs\lan-key.pem <LAN-IP> <pcname>.local localhost`.
2. Start the server with `--lan`. A second `uvicorn.Server` binds `0.0.0.0:8443` with `ssl_certfile`/`ssl_keyfile`.
3. **Pairing:** LAN mode requires a 6-digit code shown on the desktop. The phone enters it once and gets an HttpOnly cookie. Requests without the cookie get 401.
4. Copy **only `rootCA.pem`** to the phone, never the key.
   - Android: Settings → Security → Encryption & credentials → Install a certificate → CA certificate.
   - iOS: install the profile, then General → About → Certificate Trust Settings → enable full trust.
5. Reserve the PC's IP in the router's DHCP settings.
6. **Firewall** (admin PowerShell, mkcert path only):
   ```powershell
   Get-NetConnectionProfile          # Wi-Fi must be Private; if not: Set-NetConnectionProfile -InterfaceAlias "Wi-Fi" -NetworkCategory Private
   New-NetFirewallRule -DisplayName "Nathan_chordify HTTPS 8443" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8443 -Profile Private
   ```

**Rejected:** self-signed certificates (no PWA install) and Cloudflare quick tunnels (no SSE, the hostname changes, and the app would be exposed publicly).

---

## 11. Server API (WP9; FastAPI, JSON, single origin, no CORS)

| Method & path | Purpose | Ships |
|---|---|---|
| `GET /health` | Liveness, plus worker state (`warming`, `ready`, `busy`) | v1 |
| `GET /api/system/info` | Versions, GPU, ffmpeg/deno/yt-dlp versions, free disk, phone URL | v1 |
| `POST /api/system/update-downloader` | Stops the worker, runs `pip install -U "yt-dlp[default]"` in a separate process, restarts the worker; after confirmation in the UI | v1 |
| `GET /api/system/storage` / `POST /api/system/storage/cleanup` | Cache size, retention (30 days) and cap (10 GB); run the clean-up now | v1 |
| `POST /api/uploads` (multipart, streamed) | Returns `{upload_id, key}` (sha256) | v1 |
| `POST /api/jobs` `{source, params}` | Returns `{job_id, key}`; the job is queued (always Full; fast mode is CLI-only) | v1 |
| `GET /api/jobs/{id}/events` (SSE) | Events: `progress {stage, pct, msg, eta_s}`, `done {analysis_url, display_url}`, `error {code, msg, hint}`. Download hints follow the §5.1 ladder and never suggest pausing the VPN. | v1 |
| `DELETE /api/jobs/{id}` | Cancel | v1 |
| `GET /api/tracks?q=&chords_subset=&level=` | Library list. `chords_subset=B,E,F#` returns songs whose chords ⊆ that set at `level` | v1; filter v1.1 |
| `GET /api/tracks/{key}` | Analysis with edits applied (`?original=1` returns the raw analysis) | v1 |
| `GET /api/tracks/{key}/display` | `display.json`: labels and voicings for 12 transpositions × 3 accidental preferences | v1 |
| `PATCH /api/tracks/{key}` | Rename (title/artist), tags | v1 |
| `GET /api/tracks/{key}/edits` | The op log and the current `rev` | v1 |
| `POST /api/tracks/{key}/edits` `{ops: [...]}` | Appends ops (idempotent by op id); returns the merged analysis, the display file and the new `rev` | v1 |
| `POST /api/theory/parse` `{text, key}` | Typed chord entry → Harte and labels, or an error with a hint | v1 |
| `POST /api/tracks/{key}/redecode` `{params, region?}` | Fast re-decode from cached evidence (§5.11) | v1 |
| `POST /api/tracks/{key}/reanalyze` `{}` | Full re-run from cached audio | v1 |
| `GET /api/tracks/{key}/audio?mix=original\|novocals\|piano\|bass\|custom&g=…` | m4a with HTTP Range support (mixes v1.1) | v1 |
| `GET /api/tracks/{key}/export/{fmt}?transpose=&play_in=&level=&detail=&notation=` | fmt ∈ lab, json (v1); midi-daw, midi-fixed, musicxml, chordpro, ireal, text (v1.1). Always uses the edited analysis. | v1 / v1.1 |
| `POST /api/tracks/{key}/gt/import` `{format: "chart"\|"lab", text}` | Parses and aligns a chart or `.lab` to the tapped beats (or the detected grid until beats are tapped) and loads it as the draft GT edit layer | v1 (M4) |
| `PUT /api/tracks/{key}/gt/beats` `{beats: [...]}` | Stores the tapped GT beats | v1 (M4) |
| `GET /api/tracks/{key}/gt/export?fmt=lab\|beats\|chart&reviewed=` | Writes `eval/gt/<key>.*` (and `songs.yaml` when reviewed) and returns the file | v1 (M4) |
| `GET /api/search?q=` | `ytsearch8` results: id, title, channel, duration, thumbnail (`ingest.search`) | v1.1 |
| `GET/PUT /api/tracks/{key}/lyrics` | LRC text with timings (paste, import, tap-sync) | v1.1 |
| `GET /api/lyrics/lrclib?title=&artist=&duration=` | Server-side proxy to LRCLIB, only when the user presses "Find lyrics" | v1.1 |
| `DELETE /api/tracks/{key}` | Remove the song and its cache, after confirmation in the UI | v1 |
| `GET/POST/PUT/DELETE /api/setlists[/{id}]` | Setlists and their entries (key, preset, tempo, loop) | v1.1 |
| `WS /api/live` | Mic PCM frames in, `{label, bass, conf}` out | v2 |

Route owners: `routes_tracks.py` (incl. the chords-subset query), `edits.py`, `gt.py`, `routes_search.py`, `routes_lyrics.py` (incl. the LRCLIB proxy), `routes_export.py`, `routes_jobs.py`, `routes_setlists.py`, `routes_system.py` and `storage.py`, all WP9.

---

## 12. Install steps and pinned versions (Windows 11, Python 3.12.4, RTX 4070 8 GB, driver 591.86 / CUDA ≤13.1)

Nathan runs these steps (they download about 4–5 GB, and take about 7–8 GB on C: once installed, est.) after approving them. **Nothing has been installed yet.** git 2.50.0 and git-lfs 3.6.1 are already present; ffmpeg, yt-dlp, deno and Tailscale are not.

**Optional cleanup:** delete the leftover `~ip*` folders that pip's warning points to in the global site-packages.

```powershell
$ProgressPreference = 'SilentlyContinue'      # the progress bar makes Invoke-WebRequest very slow
cd C:\Users\User\Desktop\Nathan_chordify
$env:HF_HOME = "$PWD\models\hf"; $env:TORCH_HOME = "$PWD\models\torch"   # before anything imports torch or huggingface_hub
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip setuptools wheel

# 1) GPU stack (cu130 runs natively on driver 591.86 / CUDA 13.1); no pip cache copy of the 2 GB wheel
.\.venv\Scripts\python -m pip install --no-cache-dir torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python -c "import torch,torchaudio;print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0),torchaudio.__version__);print(torchaudio.transforms.MelSpectrogram().cuda()(torch.randn(1,22050,device='cuda')).shape)"
#    expect: 2.14.1+cu130 13.0 True NVIDIA GeForce RTX 4070 Laptop GPU 2.11.0+cu130, then a mel shape
#    if this fails: reinstall with torch==2.11.0 and set torch==2.11.0+cu130 in engine\constraints.txt

# 2) Engine dependencies, with torch locked so that nothing can swap in a CPU build
.\.venv\Scripts\python -m pip install -r engine\requirements.txt -c engine\constraints.txt
.\.venv\Scripts\python -m pip install -e engine -c engine\constraints.txt

# 3) Model repos (pinned: record the commit hashes). Never pip-install these repos.
git clone https://github.com/andreamust/consonance-ACE third_party\consonance-ACE
git clone https://github.com/ptnghia-j/ChordMini third_party\ChordMini
git -C third_party\consonance-ACE rev-parse HEAD   # → third_party\VERSIONS.txt
git -C third_party\ChordMini rev-parse HEAD        # → third_party\VERSIONS.txt

# 4) FFmpeg 9.0.2 (Gyan full_build-shared), app-local, no admin needed; safe to re-run
if (-not (Test-Path .\tools\ffmpeg\bin\ffmpeg.exe)) {
  $z = "$env:TEMP\ffmpeg-9.0.2.zip"
  Invoke-WebRequest https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-full_build-shared.zip -OutFile $z
  Expand-Archive $z -DestinationPath .\tools -Force
  if (-not (Test-Path .\tools\ffmpeg)) { Rename-Item .\tools\ffmpeg-9.0.2-full_build-shared ffmpeg }
}

# 5) Weights: htdemucs_ft (HF adefossez/HTDemucs-ft), Beat This! final0 (81 MB) and the piano-transcription
#    checkpoint (Zenodo 4034264, 172 MB) into .\models; checks that lv-chordia's bundled .sdict files exist
.\.venv\Scripts\python scripts\fetch_models.py      # records the sha256 of every file in models\MANIFEST.json

# 6) Freeze and smoke-test
.\.venv\Scripts\python -m pip freeze > engine\requirements.lock
.\.venv\Scripts\python scripts\smoke_test.py

# 7) Frontend, then the desktop shortcut
cd web; npm ci; npm run build; cd ..
powershell -ExecutionPolicy Bypass -File scripts\make_shortcut.ps1
```

**`engine/constraints.txt`**
```
torch==2.14.1+cu130
torchaudio==2.11.0+cu130
numpy==2.5.4
```

**`engine/requirements.txt`** (every pin verified on PyPI with a cp312-win_amd64, abi3 or pure wheel)
```
numpy==2.5.4
scipy==1.18.1
numba==0.68.0
llvmlite==0.50.0
soundfile==0.14.0
soxr==1.1.0
audioread==3.1.0
librosa==0.11.0            # 1.0.0 removed APIs that the model code uses; fall back to 1.0.0 + patches if 0.11 fails the smoke test
mir_eval==0.8.2
pretty_midi==0.2.11.post0
mido==1.3.3
music21==10.5.0
matplotlib==3.11.2         # verified pin; runs headless with MPLBACKEND=Agg
einops==0.8.2
huggingface-hub==2.2.0     # htdemucs_ft weights come from the HF hub
safetensors==0.8.0
demucs==4.1.0
rotary-embedding-torch==0.9.1   # Beat This! dependency
beat-this==1.1.0
lv-chordia==1.1.0
piano-transcription-inference==0.0.6
torchlibrosa==0.1.0
lightning==2.6.6           # consonance-ACE runtime set (instead of installing the repo's own pins)
torchmetrics==1.9.0
gin-config==0.5.0
jams==0.3.5
pumpp==0.6.0
harte-library==0.4.5
fastapi==0.143.0
uvicorn[standard]==0.54.0
python-multipart==0.0.32
pydantic==2.14.1
pywebview==6.2.1
pythonnet==3.2.1
deno==2.9.7
yt-dlp[default]>=2026.8.19 # deliberately unpinned upward; updated from the app
```

**Not installed in v1:**
- madmom
- basic-pitch
- chord-extractor
- vamp
- audio-separator / onnxruntime-gpu (phase 2, WP16)
- torchcodec
- tensorflow

**Only if YouTube starts failing:** `bgutil-ytdlp-pot-provider==2.0.2`.

**`web/package.json`** (exact pins)
```json
{ "dependencies": { "react": "19.3.0", "react-dom": "19.3.0", "tonal": "6.4.3", "signalsmith-stretch": "1.3.2",
                    "zustand": "5.0.15", "idb-keyval": "6.3.0", "workbox-window": "7.4.1" },
  "devDependencies": { "vite": "8.3.4", "@vitejs/plugin-react": "6.1.2", "typescript": "7.0.2", "vite-plugin-pwa": "2.0.0" } }
```
- **tonal 6.4.3.** The 6.5.0 publish is broken: it lacks `dist/index.js` (TS2307 in tsc, and Node's import fails). With 6.4.3, `npm install`, tsc 7, `vite build` and the PWA's generateSW all pass (verified). tonal is used only for note-name helpers (diagram note labels, tuner, trainer); chord spelling is server-only.
- WP0 adds **`@types/react`**, **`@types/react-dom`** and **`oxlint`** with `npm i -D -E`; their exact versions are resolved then and locked in `package-lock.json`.
- **Linting:** TypeScript 7 has no classic compiler API, so typescript-eslint can't be used. Type-check with `tsc --noEmit` / `tsc -b` (verified) and lint with oxlint.
- **Scaffold:** `create-vite` has no 8.3.4 release; WP0 uses `npm create vite@9.2.1` or writes the scaffold by hand.
- **`src/shims.d.ts`:** `declare module 'signalsmith-stretch';` (the package ships no types).

**`scripts/smoke_test.py` must pass before any other WP merges.** It checks, in this order:
1. `torchaudio.transforms.MelSpectrogram().cuda()(torch.randn(1, 22050, device="cuda"))` runs (torch 2.14.1 with torchaudio 2.11 is documented as ABI-compatible but was never executed here). Fix hint: the torch 2.11.0+cu130 fallback.
2. CUDA is available and the device name is right.
3. At least **15 GB free** on the install drive.
4. `ffmpeg -version` and `ffprobe` work from `tools\ffmpeg\bin`, and `deno --version` works.
5. Every pinned package imports (with `MPLBACKEND=Agg`).
6. librosa 0.11 works with scipy 1.18 and numpy 2.5 (`librosa.cqt` with 12 bins/octave and `filter_scale=0.5` on 5 s of noise).
7. Beat This! `Audio2Beats(checkpoint_path="final0", device="cuda", dbn=False)` runs on a 20 s synthetic 65 BPM click track and finds 65 ± 2 BPM.
8. lv-chordia's `.sdict` files exist; it runs on a synthetic C–G–Am–F WAV (absolute path) and returns all six heads.
9. demucs runs on 10 s of audio with the model on CUDA and the mix tensor on the CPU.
10. The piano-transcription checkpoint loads, and `transcribe(audio, None)` runs on 5 s of audio.
11. The ACE and BTC-CL checkpoints load with `weights_only=True` (BTC with `map_location`); file sizes 57.6 MB and 35.9 MB (ordinary git blobs, not LFS); ACE imports in the required order (§5.5).
12. `yt-dlp --simulate --no-playlist <test URL>` resolves formats. Fix hint: update yt-dlp to the nightly (`--pre`); on HTTP 429 wait 30–60 min (never suggest pausing the VPN).
13. Under `pythonw.exe`, the server starts with stdout redirected and `/health` returns 200.
14. Every file in `models\MANIFEST.json` and `tools\ffmpeg\bin` is still present and matches its sha256. Fix hint: check the Surfshark Antivirus quarantine.

Each check prints PASS/FAIL with a fix hint.

---

## 13. Evaluation plan (WP14)

### 13.1 Ground truth for the test song (回家, confirmed; §0.1)

1. **Chart.** `eval/gt/yt-Y7boUFbM76k.chart.txt`, in the intake format of §0.1. It starts as the RESEARCH §5.3 consensus chart with each bar marked H, M or L. **Nathan's chords replace it when he sends them**: they override every published chart.
2. **Beats.** Nathan taps `eval/gt/yt-Y7boUFbM76k.beats` in Ground-truth mode (§8.4). Our beat tracker is never used for GT, so the GT is not aligned to our own output.
3. **Align.** `theory.chart.align` maps the bars onto the tapped beats, giving a draft `.lab` (beat resolution, sub-beat where the chart uses brackets).
4. **Review.** Nathan checks it by ear and at the piano in Ground-truth mode, with the (L) bars flagged. He saves `eval/gt/yt-Y7boUFbM76k.lab` and sets `reviewed: true` in `eval/songs.yaml`.
5. **Until it is reviewed,** reports label this song "draft GT" and show only the checklist and coarse metrics (root and majmin at bar level).
6. **Passing-chord checklist** `eval/gt/yt-Y7boUFbM76k.checklist.json`. Each **occurrence** is one item: `{"t0", "t1", "label", "tol", "accept", "conf"}`.
   - A point time t means the window [t − 0.3, t + 0.3] s; a range is used as given.
   - A hit needs a chord **shown at Detailed (0.8)** that overlaps the window and matches root + quality family + bass. "Also accepted" labels score 1; labels marked (0.5) score 0.5.
   - The times are in this video. They are re-timed against the tapped beats once those exist.
   - The score is the mean over the 24 scored occurrences.
7. **False-positive gate:** at Detailed, at most **3 passing chords per minute** that match no GT segment, counted inside H-confidence bars.

| # | Expected | Occurrences (s, this video) | n | Also accepted |
|---|---|---|---|---|
| 1 | B+7 | 61.4, 135.5 | 2 | B+ (full); B7 (0.5) |
| 2 | A7 → Em6/Em7/Em | 63.7–65.5, 71.0–72.4, 130.9–131.8 | 3 | A, A9; Em |
| 3 | Em/A | 37.7–39.4 | 1 | A9sus4, A7sus4 (bass A plus E-G-B on top) |
| 4 | C#m7 → D#m7 | 47.1, 72.8, 139.7 | 3 | D#m |
| 5 | D#7/A# → G#m (F#/A# at 195.9) | 103.8, 170.0, 210.6, 195.9 | 4 | D#/A#; D#7 (0.5) |
| 6 | F#/A#, F#7/A# | 96.9, 98.2 | 2 | either one |
| 7 | G#m7 → A#m7 | 163.6, 189.4 | 2 | A#m |
| 8 | E, B/D#, C#m7, B, E, B/D#, C#m7, G#m, F#, F#/C# | 109.8–124.0 | 1 | scored as the fraction of the 10 with the right bass |
| 9 | B/D#; B/F# | 84.4; 218.5 | 2 | — |
| 10 | E/F# (F#11) | 55.0, 79.3, 236.2 | 3 | F#9sus4; F#7sus4 or F#sus4 (0.5) |
| 11 | A9 | 226.4–229.7 | 1 | A7, A, Aadd9 |
| 12 | D#7sus4 → D#7/G | 4.7, 12.1 | — | **Not scored until Nathan confirms** (L) |
| 13 | Em/G | 52.6 | — | **Not scored until Nathan confirms** (L) |

### 13.2 Metrics (`eval/chord_metrics.py`, `eval/calibration.py`)

- **Standard:** `mir_eval==0.8.2` `chord.evaluate`, reporting root, majmin, **majmin_inv**, mirex, thirds(_inv), triads(_inv), sevenths, **sevenths_inv**, tetrads, **tetrads_inv**, seg, overseg and underseg.
- **Custom:**
  - **Out-of-chord-bass accuracy**: F/G, Em/A and similar fall outside *_inv, so the bass pitch class is checked separately.
  - **Short-segment P/R/F1**: reference segments ≤1 beat, and separately ≤0.6 s. A hit needs ≥50% overlap, the onset within ±max(70 ms, ¼ beat), and a label match under root, majmin_inv and sevenths_inv. The same matcher labels passing candidates hit/miss for the slider fit (§5.11).
  - **Passing-chord precision per Detail stop** (targets 0.85 / 0.70 / 0.50 at Standard / Detailed / Everything).
  - **Accuracy by duration bin**: (0, ½], (½, 1], (1, 2], (2, 4], >4 beats.
  - **Boundary F**: 100 ms and ¼-beat windows.
  - **Flicker rate**: unmatched estimated segments under ½ beat per minute. Target <2.
  - **Recall per idiom**: passing °7, augmented, borrowed (incl. ♭VII7 → iv), descending slash, walk-up, secondary dominant, tritone substitution, line cliché, sus→resolution, 4/5.
  - **Bass pitch-class accuracy per segment.**
  - **Checklist score** and the **false-positive gate** (§13.1).
- **Calibration** (`calibration.py`): reliability diagram and ECE of the class posteriors (10 bins), the break-even table (§5.9) and the measured adapter offsets.
- **Every metric is reported at each Detail stop** (0, 0.5, 0.8, 1.0) to give a precision/recall curve for passing chords.
- **On draft GT**, only the checklist and the coarse metrics are reported.

### 13.3 Data

| Set | Contents |
|---|---|
| D1 (dev) | Nathan's worship songs (Cantonese and Mandarin), annotated in Ground-truth mode with tapped beats, sub-beat chords and slash chords. **Staged:** 3 songs (回家 + 2) by the end of M4, **8 by the end of M5**. Annotation starts from our main layer only. **Coverage:** ≥2 songs at 100–130 BPM, ≥1 in 6/8 or 12/8, ≥1 with a modulation, ≥1 with a free-time intro, ≥1 Mandarin song, ≥1 solo-piano arrangement. 回家's shortest changes are half-beat at 65 BPM, so it never exercises 16th-note cells; the faster songs and D2 do. |
| D2 (synthetic) | `eval/synth/render.py`: a **sampled piano** (Salamander Grand Piano, CC-BY; downloaded once after Nathan approves), bass, a pad that sustains the previous chord, noise drums, and a convolution reverb from a synthetic IR. It renders the §6.4 idioms and the 回家 pre-chorus in 12 keys at 60–140 BPM, sweeping pad level (−18…0 dB) and reverb (RT60 0.5–3 s). The labels are exact. **mini-D2** (M2): 12 idioms × 3 keys × 2 tempos. D2 is used for **calibration** (T_s, ECE, break-even, adapter offsets) and **unit tests**; in the tuning objective it never weighs more than D1. |
| D3 (optional regression) | Public annotations (Isophonics, Billboard, RWC) only for audio Nathan already owns. No audio is downloaded for this. |

### 13.4 Baselines and ablations

**Baselines**
- B1: lv-chordia alone, through its own CRF (`submission` dictionary, d = 30).
- B2: consonance-ACE alone (`--chord-min-duration 0.5`), from M5.
- B3: BTC-CL alone, from M5.
- **B4: Chordify's output, mandatory** for 回家 and at least 3 D1 songs. Nathan copies Chordify's sequence from its page in the bar-text format (§0.1). "Beats Chordify" is claimed only against B4.
- **B5: our main layer at Simple** (Detail 0): the like-for-like comparison with Chordify's one-chord-per-beat output.

**Ablations** (each switched off once):
- separation;
- sub-beat grid (beat grid only);
- bass term;
- symbolic chroma;
- priors (`w_prior = 0`);
- calibration (raw weighted sums instead of §5.9);
- equivalence classes;
- idiom completion;
- repetition prior;
- `w_short = 0`;
- HSMM vs a frame HMM with a uniform penalty;
- two layers vs detail only;
- each model removed in turn.

### 13.5 Tuning

- **Fixed, never tuned:** κ_b = 10 per beat. The temperatures T_s are fitted by NLL (`decode/calibrate.py`) before any tuning, and refitted whenever a source changes.
- `eval/tune.py` searches the other `DecoderParams`: λ_scale, w_prior, w_sym, w_bas ∈ [1, 3], w_dur, PT bonus, w_short, rep_bonus, the source weights and the slider mapping (a, b, c). 200 Sobol samples are followed by coordinate descent.
- Validation is leave-one-song-out over D1, with D2 included at no more than D1's weight.
- Objective per set: **J = 0.4·sevenths_inv + 0.4·shortseg_F1(detail = 0.8) + 0.2·bass_acc − 0.02·max(0, flicker − 2) − Σ_{s ∈ {0.5, 0.8, 1.0}} max(0, prec*(s) − 0.05 − prec(s))**, with prec* = 0.85 / 0.70 / 0.50.
- Combined: **J_total = 0.6·J(D1) + 0.4·J(D2)**. In M2, before D1 exists, tuning uses mini-D2 plus the 回家 checklist.
- **Never tune on the test song alone.** With one song, overfitting is guaranteed.

### 13.6 Accuracy gates per milestone (targets, not promises)

| Gate | Criteria |
|---|---|
| M1 | End-to-end on 回家 in ≤3 min with lv-chordia only. The JSON and `display.json` validate. B1 is measured. |
| M2 | ECE ≤ 0.1. Break-even δ* within ±0.2 nats of the targets (0.7 half-beat A–P–B, 1.3 half-beat A–P–A). On mini-D2 and the 回家 checklist: **beats B1 on short-segment recall with flicker ≤ 2/min**. Checklist ≥ 0.55. |
| M5 (tuned, LOSO on 8 D1 songs) | sevenths_inv ≥ best baseline + 3 pts. **Short-segment F1 ≥ 0.6 at precision ≥ 0.7** (Detailed). Passing precision per stop within 0.05 of 0.85 / 0.70 / 0.50. Flicker < 2/min. Bass accuracy ≥ 85%. Checklist ≥ 0.75 (stretch goal 0.9) and the false-positive gate met. On 回家 and ≥3 songs: B5 ≥ B4 on majmin, and ours at Detailed > B4 on short-segment recall. |
| M5 component gate | ACE, piano transcription + symbolic chroma, the repetition prior, the modulation detector and BTC-CL are each kept only if LOSO short-segment F1 rises by **≥0.02** with **no** majmin_inv loss. |
| Phase-2 gate (WP16, M8) | A new component (BACHI, BS-RoFormer-SW, S-KEY) is kept only if it raises short-segment F1 by ≥0.03 without lowering majmin_inv by more than 0.5 pt. |

### 13.7 The iteration loop

1. Every change runs `python -m eval.run_eval --set D1,D2` from the repo root.
2. The report has per-song tables, a diff against the last run, confusion matrices, idiom recall, calibration plots, and a **"listen list"**: the 20 worst segments as deep links (`/#/track/<key>?t=<s>&loop=<a>,<b>`) so they can be checked by ear at once.
3. Corrections Nathan makes in everyday use can be promoted into D1 once he marks them reviewed in Ground-truth mode.

---

## 14. Milestones and work packages

### 14.1 Milestones (build order)

| M | Contents | Main WPs | Exit gate |
|---|---|---|---|
| **M0** Environment and spikes | venv, torch (`--no-cache-dir`), FFmpeg, yt-dlp + deno, prefetched weights, `smoke_test.py` passing (§12). WP0 contracts (§4, the §5.13 schemas, mock fixtures, web scaffold). **Spikes:** (a) **download 回家** (`Y7boUFbM76k`): GO/NO-GO; on NO-GO try the nightly yt-dlp, then a 30-min back-off, then the PO-token plugin, and otherwise Nathan supplies the audio file (never pause Surfshark). (b) **lv-chordia raw per-head probabilities**: GO/NO-GO (§5.5). (c) htdemucs_ft + Beat This! on 回家: bars ≈ 3.67 s, VRAM and timing logged, plus one 10-minute file. Ask Nathan for his chords (§0.1). | WP0 (+ WP1–WP3, WP4a for the spikes) | The smoke test passes on Nathan's laptop; both GO/NO-GO decisions are recorded. |
| **M1** Engine v0 + CLI | **One model.** ingest → htdemucs_ft → Beat This! (octave rule and check, triplet test, free time) → lv-chordia (mix and harmonic merged, chunked) → bass tracker (§5.6) → calibrated evidence (§5.9) → class pruning + idiom completion → HSMM main + detail → layers (demotion, phantom rule, bass-event rule) → key, spelling, Nashville, voicings → `analysis.json`, `display.json`, `.lab`. `nchord analyze <URL> --out a.json --lab a.lab --chart` (`--chart` prints the bar-text chart). | WP1–WP3, WP4a, WP5 (bass), WP6, WP7, WP8, WP10 (`.lab`) | §13.6 M1 gate. |
| **M2** Evaluation harness + calibration | `chord_metrics` (mir_eval, short-segment F1, flicker, per-stop precision), checklist + false-positive gate, baselines B1 and B5 (B4 as soon as Nathan sends Chordify's chords), mini-D2, temperature fit, adapter offsets, ECE and the break-even table. Tune λ_scale, w_bas, the PT bonus and the slider mapping. | WP14, WP7 | §13.6 M2 gate. |
| **M3** Browser MVP | FastAPI + one pre-warmed GPU worker + SSE + audio route + upload + library list + storage limits. React sub-beat grid with bass lane, synced to `<audio>`. Tempo 25–200% with `preservesPitch`, A/B and drag loop, transpose and play-in from server-precomputed labels, key-change badges, Detail / Level / Slash / Nashville, NOW/NEXT piano diagram, alternatives, re-scan, zh-TW/en strings. | WP9, WP11, WP12 (library list), WP6 (`display.py`) | Plays 回家 in sync (±30 ms visual), 60 fps on the laptop with no per-frame React renders; phone layout at 375 px with no horizontal scroll (in a desktop browser until M6). |
| **M4** Minimal editor + GT mode | Set chord, delete, accept/reject, undo, half/double tempo; append-only edit ops (`server/edits.py`); chart import and alignment (`theory/chart.py`, `server/gt.py`); tapped GT beats; `.lab`/beats/chart export; the GT-mode tools of §8.4. **Nathan annotates 回家 + 2 songs** (D1 = 3). | WP12, WP9, WP6 | Edits from desktop and phone, interleaved, merge without loss; 3 reviewed D1 songs. |
| **M5** Accuracy round 2 | In this order: consonance-ACE; piano transcription → symbolic chroma and left-hand bass; repetition prior + auto sections (§5.15); modulation detector (§5.4 steps 3–4); BTC-CL last. **Each is kept only if LOSO short-segment F1 rises by ≥0.02 with no majmin_inv loss.** D1 grows to 8 songs. | WP4b, WP5, WP7, WP6, WP4c, WP14 | §13.6 M5 gate. |
| **M6** Desktop + phone | pywebview window, `.bat` files and desktop shortcut (§9); PWA installed on the phone over Tailscale serve, with the Surfshark set-up (§10). | WP13 | The desktop window opens in ≤10 s under pythonw; the phone installs the PWA and runs a job over Tailscale. |
| **M7** v1.1 parity | signalsmith-stretch engine (real pitch shift, gapless loops, count-in, speed trainer), ChordSynth, metronome, stem practice mixes; exports MIDI (DAW + fixed), MusicXML, ChordPro, iReal Pro, text and print/PDF, all with edits; setlists; offline PWA; lyrics (paste, LRC, tap-sync, LRCLIB proxy, Chords & Lyrics view); beat-grid editing UI; YouTube and library search, the chords-subset filter; auto-section UI; Summary and Nashville chart views. | WP1, WP2, WP9, WP10, WP11, WP12, WP13 | Every v1.1 row of §7 works on desktop and phone; a song saved offline plays in airplane mode; MIDI and MusicXML open in MuseScore. |
| **M8** v2 | Toolkit (tuner, standalone metronome, chord trainer, live detection, practice and step modes) and the phase-2 models (BACHI, BS-RoFormer-SW, S-KEY, ASR lyrics), each behind a flag. | WP15, WP16 | A live chord within ≤1 s on a piano recording; the trainer works with a USB MIDI keyboard; the phase-2 gate (§13.6). |

**Order and critical path**

```
M0 ─► M1 ─┬─► M2 ──────────────────────────────────────┐
          └─► M3 ─► M4 ─► Nathan annotates D1 (3 → 8) ─┴─► M5 ─► M7 ─► M8
                └─► M6 (any time after M3)
```

- **The critical path runs through Nathan's annotation.** M5's LOSO tuning needs D1; D1 needs M4's Ground-truth mode; GT mode needs the GT routes (§11) and `theory/chart.py`.
- M2 runs alongside M3. The frontend develops against `shared/fixtures/huijia_mock.*.json` until M1 produces real output.
- **Deferred** (not in any milestone until a need appears): the two-phase fast preview, the mkcert/LAN fallback.

### 14.2 Work packages (each owns disjoint files per §3)

| WP | Delivers (interface) | Milestone(s) | Depends on | Acceptance |
|---|---|---|---|---|
| **WP0 Foundation and contracts** (lead) | Folder skeleton. `pyproject`, requirements, constraints. `config.py` (paths; sets `HF_HOME`/`TORCH_HOME`/`HF_HUB_*`/`MPLBACKEND` before torch is imported; ffmpeg on PATH). `logsetup.py`. `types.py` with every type of §4, `schema.py` → analysis and display schemas. `progress.py`. `models/base.py`. Hand-made mock fixtures (pre-chorus with passing chords). `web` skeleton (by hand or `create-vite@9.2.1`) + `api/types.ts` + `shims.d.ts` + oxlint + feature stubs. `scripts/install.ps1`, `fetch_models.py`, `smoke_test.py`. | M0 | — | `smoke_test.py` passes on Nathan's laptop. The mocks validate in pytest, and in TS via a type-checked import. |
| WP1 Ingest | `ingest.resolve/fetch` (§5.1). Cache keys and `meta.json`. Error ladder with user hints (§5.1; never "pause the VPN"). `ingest.search` (v1.1). | M0 spike, M1; search M7 | WP0 | Tests: URL normalisation (watch, youtu.be, shorts, `&list=RD…`, `&pp=`) → `yt-Y7boUFbM76k`. An upload's sha256 key. An ffmpeg decode to 44.1 kHz PCM16 with `CREATE_NO_WINDOW`. |
| WP2 Separation | `separate()` with the CPU mix tensor, FLAC stems, OOM fallback (§5.2). `practice_mix()` (v1.1). | M0 spike, M1; mixes M7 | WP0 | Stems written as FLAC. harmonic = bass + other within −60 dB of the residual. A 4-minute song in ≤90 s on the GPU. |
| WP3 Rhythm | `rhythm.track()` with the octave rule, pickups, mixed meter, triplet test, subdiv ∈ {2, 3, 4, 6}, free-time grid, `cell_pos`, HCDF, overrides; `check_octave()` (§5.3). | M0 spike, M1 | WP0 | Synthetic click tracks (65/130 BPM, 3/4, 6/8, swung 8ths, a rubato intro) give the right bpm, bpb, subdiv and free-time regions. On 回家, bars come out ≈3.67 s. |
| WP4a lv-chordia adapter | `LvChordia.run()` → heads, with chunking, the `.sdict` assert and the measured offset (§5.5). | M0 spike, M1 | WP0 | Head shapes and labels recorded. Synthetic C–G–Am–F: the frame argmax is right on ≥90% of frames. A 10-minute file runs in chunks with flat VRAM. Offset residual < 20 ms on D2. |
| WP4b ACE adapter | `Ace.run()`: root, bass, notes; `sys.path` and import order; overlapping 20 s chunks. | M5 | WP0 | As WP4a. Pinned commit in `VERSIONS.txt`. Passes the M5 component gate. |
| WP4c BTC-CL adapter | `BtcCl.run()`: 170-class probabilities plus the mapping table M (§5.9); `map_location` load. | M5 (last) | WP0 | As WP4a. Passes the M5 component gate. |
| WP5 Bass and notes | `bass.track()` (§5.6); `piano_amt.transcribe()`, `symbolic.cell_chroma()` (§5.7). | Bass M1; AMT and chroma M5 | WP0 (+ WP3 grid type) | Bass: the §5.6 acceptance test. Chroma unit tests (τ, β, α, θ, pedal offsets, the MIDI 55 split). |
| WP6 Theory | Vocabulary and classes, templates, Harte parsing and formatting, key estimate and refine (incl. the truck-driver test), spelling, Nashville and Roman, simplify, voicing, priors + `trigram()`, `chart.py`, `display.py`. Shared test vectors (Python only). | M1; chart M4; key refine M5 | WP0 | ≥300 vectors pass (e.g. in key B: D#7/A#, F#/A#, A7 = b7(7), B+7, #1°7 shown as C°7; in key C: C#°7; classes: E6 = C#m7/E; voicing: B/D# keeps B in the RH). |
| WP7 Fusion and decoding | `evidence.build/prune` (calibrated, class-level, idiom completion), `calibrate.fit`, `hsmm.decode` (Viterbi + forward-backward, `region`/`boundary`/`rep`), `layers.build`, `post.finalize`, `sections.find`, `DecoderParams` (§5.9–5.12, §5.15). | M1; calibration M2; sections M5 | WP0, WP6 | On **real lv-chordia output** (D2 renders and 回家; synthetic evidence only in unit tests): the break-even table near its targets, ECE ≤ 0.1, half-beat chords recovered, flicker ≤ 2/min. Decode of 1,000 cells × 250 candidates in about 1 s. Marginals sum to 1. |
| WP8 Pipeline and CLI | `pipeline.analyze/redecode` with the orchestration loops (§5), cache versioning, `nchord` CLI (`analyze` with `--mode fast` for debugging and `--chart`, `redecode`, `eval`, `serve`, `doctor`, `update-downloader`, `gt import/export`). | M1 | WP1–WP7 | **M1 gate**: `nchord analyze <test URL> --out a.json --lab a.lab` → valid JSON and display file in ≤3 min. Timing table logged. |
| WP9 Server | FastAPI app, GPU worker (pre-warm, `CREATE_NO_WINDOW`, stop/restart for updates), job queue, SSE, SQLite, `storage.py` (retention, cap), `edits.py` (append-only ops, apply + re-spell), `gt.py`, every route in §11 (incl. lyrics + LRCLIB proxy, search, chords-subset query), static hosting, CSP header; `lan.py` only if the mkcert fallback is ever built. | M3; edits and GT M4; lyrics, search, subset, setlists M7 | WP0 (stub pipeline until WP8) | Integration test: submit the mock job → progress → done. Range requests work. A worker crash is recovered. Interleaved ops from two clients merge deterministically. |
| WP10 Exports | `export/*`: `.lab`, JSON; MIDI DAW and fixed, MusicXML (`harmony.ChordSymbol` + bass), ChordPro, iReal Pro, text (§7.2), all from the edited analysis. | `.lab` M1; the rest M7 | WP0, WP6 | Files open in MuseScore. MIDI has a tempo map that matches the beats and the tempo in its file name. Transpose, level and edits applied. |
| WP11 Web core | App shell, routing, store, API client, SSE, **player** (§8.3: `<audio>` in M3, signalsmith-stretch in M7), **grid** with bass lane, **piano diagrams**, transport, transpose and play-in from `display.json`, tempo, loop/A-B (count-in M7), mixer (M7), Detail/Level/notation controls, key badges, keyboard shortcuts, i18n, responsive layouts (§8). | M3; v1.1 engine M7 | WP0 (mock JSON), WP9 | The M3 exit gate. |
| WP12 Web features | Library list, upload and rename (M3); **editor + Ground-truth mode** (M4); search, setlists, lyrics and the Chords & Lyrics view, print/PDF views, export menu, settings (M7). | M3, M4, M7 | WP11, WP9 | Edits round-trip through `POST /edits`. GT export writes the `eval/gt` files. Prints: grid, lyrics, Nashville and diagram pages in A4. |
| WP13 Platform | `desktop.py`, `.bat` files, shortcut script, PWA config/manifest/SW, offline (M7), `tailscale_setup.md` (with the Surfshark steps), `mkcert_setup.ps1` only if needed. | M6; offline M7 | WP9, WP11 | The M6 exit gate; a song saved offline plays in airplane mode (M7). |
| WP14 Evaluation | `eval/` metrics, calibration, checklist, synthetic renderer, `run_eval`, `tune`, reports (§13). | M2; D1 support M4; M5 | WP0 (metrics), WP8 (end-to-end) | Reports for B1, B4, B5 and ours on mini-D2 and 回家 (draft GT). Then the **M2 and M5 gates**. |
| WP15 Toolkit (v2) | Tuner, standalone metronome, chord trainer (Web MIDI), live detection (`server/live.py`, `web/src/toolkit`), practice and step modes. | M8 | WP9, WP11 | The M8 exit gate. |
| WP16 Phase-2 models | BACHI adapter (symbolic voter), BS-RoFormer-SW separation, S-KEY, ASR lyrics. Each one sits behind a flag. | M8 | WP14 at M5 | Kept only if it passes the phase-2 gate (§13.6). |

**Rules**
- Interface changes go only through WP0's files, via a PR reviewed by the lead.
- Each WP writes only its own files and tests.
- The frontend develops against `shared/fixtures/huijia_mock.*.json` until M1 produces real output.

---

## 15. Risks and mitigations

| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| **YouTube download breaks** (EJS, PO tokens, SABR, 403s, bot checks) | High / High | yt-dlp is unpinned upward and updated from the app (worker stopped first). Deno is bundled in the venv, and Node 24 is listed as the fallback runtime. Rate limits (HTTP 429 / bot checks) trigger a 30-minute back-off instead of retries; the VPN is never paused. Browser cookies are an opt-in option, and the bgutil PO-token plugin is installed on demand. **Upload always works.** `noplaylist` is enforced. M0 tests the 回家 download before anything else. |
| **Surfshark VPN and antivirus** | High / Medium | Surfshark always stays on (Nathan's rule). Downloads: nightly yt-dlp, rate-limit back-off, PO-token plugin, or uploads. Tailscale next to Surfshark is tested in M6, and any VPN setting change is Nathan's own decision (§10). Smoke check 14 re-verifies every model and FFmpeg file against its sha256, with a hint to look in the Surfshark Antivirus quarantine. |
| **Disk space** (37 GB free on C:) | Medium / Medium | torch installed with `--no-cache-dir`, FLAC stems, 30-day retention, a 10 GB cache cap, the smoke test's 15 GB check, and job refusal below 5 GB (§5.14). |
| **YouTube Terms of Service** | Medium / Medium | Personal use only. Audio stays in the local cache and is never redistributed. Offer upload of purchased or owned audio. |
| **VRAM (8 GB)** and laptop thermals | Medium / Medium | Sequential stages and lazy model loading. `empty_cache()` between stages. The demucs mix tensor stays on the CPU; lv-chordia and ACE run in chunks; songs are capped at 30 min by default. Demucs `segment=6` and a CPU fallback. Keep the laptop on AC power. The OOM retry ladder is in §5.14. |
| **Python 3.12 incompatibilities** | Medium / Medium | The review verified that every Python pin installs and imports on 3.12, including the ACE runtime set and librosa 0.11 with scipy 1.18. The open item is torch 2.14.1 with torchaudio 2.11, which smoke check 1 runs first (fallback torch 2.11.0+cu130). Remaining fixes, in order: 1) a minimal patch inside `third_party/` recorded in `VERSIONS.txt`; 2) for a model that won't run on 3.12, a uv-managed **Python 3.11 venv subprocess** worker that returns `.npz` evidence (the interface doesn't change); 3) drop the voter and re-weight the fusion. Only MIT-licensed code is patched. |
| **Start-up under pythonw** | Fixed / High | stdout/stderr redirect before any import (§9); smoke check 13. |
| **Model licences** | Low (personal use) / High (if ever sold) | v1 uses MIT code and weights (lv-chordia, consonance-ACE, Beat This!, demucs), CC BY 4.0 (piano transcription: attribution in About), CC-BY Salamander piano samples for the synthetic test set only, and ChordMini (MIT code; **training-data terms may restrict commercial use**). BTC-original weights (NC), BS-RoFormer-SW (unknown), MuScriptor/MERT/MuQ (NC) and the GPL Chordino are **excluded from v1**. Selling the app would need a licence review and possibly retraining. |
| **Pickle checkpoints** | Low / High | `weights_only=True` for every checkpoint (all verified to load that way). SHA-256 manifest. Load only from the repos listed in RESEARCH. Inspect BACHI's `vocab.pkl` with `pickletools` before use. |
| **Calibration misses its targets with one model** | Medium / High | The M2 gate (ECE ≤ 0.1, break-even within ±0.2 nats). Until it passes, the UI offers only the Simple and Everything presets, without the slider. |
| **Accuracy expectations** | Certain / High | **100% is impossible.** Experts agree only 54–73%, and the state of the art is about 82–84% maj/min. Passing-chord recall target is ~60% at 70% precision. The UI makes errors cheap to fix: alternatives, region re-scan, an editor with undo, and confidence shading. Metrics are reported honestly at every Detail level, and "beats Chordify" only against B4. |
| **Masking by reverb, pads and vocals** (worship mixes are heavy on both) | High / Medium | Separation (harmonic stem). The phantom rule (§5.12). Saturating, recency-weighted symbolic chroma (§5.7). A bass stem that defines slash chords. The vocal stem is excluded from chord evidence, and added-tone neighbours need harmonic or symbolic support. A BS-RoFormer piano stem in phase 2. |
| **Beat, tempo-octave or meter errors** (65 vs 130 BPM, triplet feel, rubato in ballad tags) | Medium / Medium | The octave rule and octave check, the per-section triplet test, the free-time grid (§5.3), a half/double toggle, and (v1.1) beat-grid editing and tap tempo. Times are stored in seconds, so a grid fix re-decodes without re-running the models. |
| **Ground-truth effort** (sub-beat annotation is slow) | High / High | D1 is staged (3 songs by M4, 8 by M5), annotation starts from our main layer, and GT mode has a 70% bar loop, synth A/B and keyboard-first entry. The plan's critical path is built around it (§14.1). |
| **Nathan's chords for 回家 not received yet** | — / Medium | Until then 回家 is "draft GT": only the checklist and coarse metrics. M2 proceeds on mini-D2. |
| **Overfitting to one test song** | High / Medium | LOSO tuning on D1 plus the synthetic set. Test-song numbers are reported but never used alone for tuning. |
| **Laptop asleep, off or off-network** (phone can't reach it) | Medium / Low | The phone app needs the laptop; this is stated in §7. `SetThreadExecutionState` during jobs. Offline PWA cache for saved songs (v1.1). Tailscale also works away from home. |
| **Phone memory** (decoded AudioBuffer about 85 MB for 4 min) | Low / Medium | The `<audio>` tier for songs over 12 min, or when decoding fails. |
| **Untested 2026 package versions** | Low / Low | The review verified every pin on PyPI and npm (tonal replaced, §12). The lock file plus the smoke test catch the rest; any pin that fails is replaced with the nearest working version and recorded in `requirements.lock` with a note. |

---

## 16. Unverified items (and what settles each)

| Item | Settled by |
|---|---|
| torch 2.14.1 and torchaudio 2.11 running together on CUDA | M0, smoke check 1 (fallback torch 2.11.0+cu130) |
| Real VRAM peaks and timings, including a 10-minute file; the size of the resident weights | M0 spike (c), M1 timing table |
| lv-chordia's per-head outputs on real audio: reaching them without forking was verified, but their class lists and per-ensemble averaging are not yet checked | M0 spike (b) |
| The 回家 download through Surfshark | M0 spike (a) |
| `demucs.api.Separator` argument names in 4.1.0 | M0 smoke check 9 |
| How much FLAC stems save per song | M1 |
| The break-even targets and ECE ≤ 0.1 with lv-chordia as the only model | M2 gate |
| BTC-CL's frame hop | WP4c (M5) |
| Tailscale working next to Surfshark | M6 |
| signalsmith-stretch's clock API (1.3.2 README) | WP11 (M7) |
| AudioWorklet + WASM on iOS Safari, on a real device | M7 |
| The iReal Pro import format | WP10 (M7) |
| The WebView2 microphone prompt in practice | M8 (only the Toolkit uses the mic) |

---

## 17. Changelog v1.0 → v1.1

IDs: **A**, **E**, **W**, **B** and **C** refer to [reviews/review-accuracy.md](reviews/review-accuracy.md); **F***n* is issue *n* of [reviews/review-feasibility.md](reviews/review-feasibility.md) and **F-pins** its part C; **L** marks a decision by the lead.

| ID | Change | Where |
|---|---|---|
| A1 | One source per model (mix and harmonic runs merged); per-source log-softmax over C with a temperature fitted by NLL, clipped at −6; a weighted **mean** instead of a weighted sum; evidence measured per beat with κ_b ≈ 10, replacing the per-second κ; κ_b fixed while λ_scale and the priors are tuned; break-even targets (≈0.7 nats half-beat A–P–B, ≈1.3 nats A–P–A) and ECE ≤ 0.1 as WP7/M2 acceptance; the slider is gated on them. | §0, §4, §5.9, §13.5, §13.6, §14 |
| A2 | Bass CQT at 12 bins/octave with `filter_scale` 0.5 (pyin as the fallback); onset-segmented, latency-compensated notes; overlap-weighted P_bass replaces the "early part of the cell" rule; new WP5 acceptance (16ths at 65 BPM, 8ths at 130 BPM, ≥90%, ≤40 ms). | §5.6, §14.2 |
| A3 | P_bass comes only from our tracker and the AMT left hand; model bass heads count 0.5 inside their sources; w_bas = 2.0, tuned in [1, 3]. | §5.6, §5.9, §13.5 |
| A4 | Symbolic chroma: the bass pc gets τ 0.5 unless it is also an upper tone; notes split at MIDI 55; saturating presence 1 − exp(−ch/θ) with pedal-extended offsets; dch only as an onset feature; no max-normalisation. | §5.7, §5.9 |
| A5 | Equivalence classes (same pc-set and bass, merged across an optional 5th); confidence, salience, pruning and alternatives per class; the display label is chosen inside the class; new `equivalents` field. | §4, §5.8, §5.11, §5.12, §5.13 |
| A6 | Transition-phantom rule. | §5.12 |
| A7 | A–P–A neighbours are demoted (min_detail + 0.3), never deleted; whitelist (pedal, bass ±1–2, IV/V of A); added-tone neighbours need harmonic or symbolic support. | §5.12, §6.2 |
| A8 | Bass-event rule (keeps the neighbour's label with a slash, never relabelled); passing-tone threshold min(½ beat, 0.3 s); drawn as slash chords at Everything. | §5.11, §5.12, §6.2 |
| A9 | Six qualities added (13sus4, m11, 5, maj7#11, 7b13, +maj7; 32 in total); +2/+5/+10 basses extended to maj7, m7 and 6; triad/bass display when the 7th is only in the bass. | §5.8, §5.12, §6.1, §7.4 |
| A10 | Borrowed iv also from the ♭VII family (回家's A7 → Em6); rules added for the tritone substitution, ♭VI → ♭VII → I, chromatic descending m7, line clichés and the descending chromatic bass; positive rules stack with a +3.0 cap; w_dur = 1.0; the duration prior's hidden change cost documented. | §5.10, §6.4 |
| A11 | Pruning by class (top-10 classes per cell), idiom completion with bridge candidates, cap 250. | §5, §5.8 |
| A12 | Key emission log-softmax(cos/T) with T = 0.02; same-mode +1/+2 moves favoured; chord-based key on 16-beat windows; truck-driver test; re-decode; `key.changes` in the JSON; key badges and Nashville renumbering in the UI. | §5.4, §5.12, §5.13, §7.2, §8.1 |
| A13 | Per-section triplet test with subdiv ∈ {2, 3, 4, 6}; free-time grid (0.2 s cells snapped to onsets, flat λ = 4, unmetered display); octave check after the first decode for any meter, guarded so the halved tempo stays ≥45 BPM. | §4, §5.3, §5.10, §5.13 |
| A14 | Model log-scores interpolated onto the 11.6 ms timeline and integrated per cell; adapter offsets measured (residual < 20 ms); w_short through a second prefix sum for segments ≤ ½ beat. | §5.5, §5.9, §5.10 |
| A15 | min_detail = σ(a·logit(conf_class) + b·salience + c), fitted for precision ≈0.85 / 0.70 / 0.50 at Standard / Detailed / Everything; per-stop precision added to the tuning objective; the old fixed confidence cut-off removed. | §5.11, §13.5 |
| A16 | Repetition prior (+1 nat per agreeing segment, soft) and automatic sections from self-similarity (new §5.15). | §5.10, §5.15 |
| A17 | Voicing fix: inversions keep the root in the RH and drop the bass pc; a non-chord bass gets the full upper chord; rootless voicings optional; the sign of the top-note term corrected. | §5.12 |
| A18 | B#, E#, Cb and Fb roots avoided by default (toggle); double sharps shown enharmonically in note names too. | §5.12, §6.3 |
| A19 | `same_chord` defined; M is the main segment with the largest overlap; promotion compares class posteriors on the detail lattice. | §5.11 |
| E1 | Draft GT reports only the checklist and coarse metrics; Nathan taps independent GT beats; bar-text intake format. | §0.1, §8.4, §13.1 |
| E2 | Checklist scored per occurrence ([t0, t1, label, tolerance]) at Detailed; false-positive gate (≤3 per minute in H bars). | §13.1, §13.6 |
| E3 | D1 staged (3 songs by M4, 8 by M5); annotation starts from our main layer; GT-mode tools (70% bar loop, synth A/B, keyboard-first); D1 coverage list. | §0.1, §8.4, §13.3 |
| E4 | B4 (Chordify) mandatory for 回家 and ≥3 songs; new baseline B5 (our main layer at Simple). | §0.1, §13.4, §13.6 |
| E5 | D2 used for calibration and unit tests and never weighted above D1; rendered with a sampled piano (Salamander, CC-BY). | §13.3, §13.5 |
| W1 | Contract types added: Candidate, Vocab, Evidence, Segment, DecodeResult, SymParams, DecoderParams, RhythmOverrides, EditOp, KeyChange. | §4 |
| W2 | GT import, beats and export routes; the critical path through annotation stated. | §11, §14.1 |
| W3 | Owners for orphaned work: auto sections → WP7 `decode/sections.py`; applying edits and re-spelling → WP9 `server/edits.py`; lyrics routes and the LRCLIB proxy → WP9; `ingest.search` → WP1 and its route → WP9; chords-subset query → WP9. | §3, §11, §14.2 |
| W4 | Append-only edit ops with client ids, server-side merge and `revert` for undo, replacing the whole-list `PUT`. | §4, §5.13, §11 |
| W5 | `hsmm.decode` gains `region`, `boundary` and `rep`; WP6 exposes `trigram()`; WP8 orchestrates the key-refine and octave loops. | §4, §5, §5.10 |
| W6 | TypeScript theory port dropped: the server precomputes `display.json` (12 transpositions × 3 accidental preferences) and parses typed chords; shared test vectors are Python-only; the TS theory folder is gone from the layout. | §0, §3, §5.12, §5.13, §11 |
| W7 | WP7 acceptance runs on real lv-chordia output; synthetic evidence only in unit tests. | §14.2 |
| B1 | iReal Pro export (v1.1). | §7.2, §11, §14 |
| B2 | Chords & Lyrics parity: character-anchored chords, text size, karaoke highlighting, pinnable diagrams. | §7.2 |
| B3 | Auto section labels get an algorithm and an owner. | §5.15 |
| B4 | Key-change UI: per-segment badge, Nashville renumbering, transpose across a modulation. | §7.2, §8.1 |
| B5 | Edits flow into every export and print view; tempo in the MIDI file name. | §5.13, §7.2 |
| B6 | Deezer unsupported, stated in the feature map. | §5.1, §7.1 |
| B7 | Each feature says whether it ships in v1 (M3/M4), v1.1 (M7) or v2 (M8); v1 is said not to be full parity. | §7 |
| B8 | The phone app needs the laptop on, except songs saved offline. | §0, §7, §10 |
| B9 | Search by chords gets a route (`chords_subset`). | §11 |
| B10 | When bass passing tones show is now the same in §5.11 and §6.2. | §5.11, §6.2 |
| C | Build order rewritten as milestones M0–M8 with exit gates; the WP table maps each WP to its milestone(s); accuracy gates renamed after the milestones; deferred items listed. | §13.6, §14 |
| F1 | stdout/stderr redirect under pythonw (blocker); `HF_HUB_DISABLE_PROGRESS_BARS`, `MPLBACKEND=Agg`; all weights prefetched. | §9, §12 |
| F2 | ACE runtime dependencies pinned; the ACE repo is never pip-installed; `sys.path` and import order specified. | §5.5, §12 |
| F3 | Eval metrics module renamed to `eval/chord_metrics.py`; eval runs as `python -m eval.run_eval`. | §3, §13 |
| F4 | tonal pinned to 6.4.3 (the 6.5.0 publish is broken). | §12 |
| F5 | Surfshark: originally "pause or split-tunnel". **Superseded 2026-10-13 by Nathan's rule: never pause the VPN**; the download ladder works with the VPN on (nightly yt-dlp, rate-limit back-off, PO-token plugin, uploads), and Tailscale coexistence is tested in M6. mkcert needs Private Wi-Fi and "Invisible on LAN" off; antivirus quarantine check. | §0.1, §5.1, §10, §12, §15 |
| F6 | Disk: torch with `--no-cache-dir`, FLAC stems, 30-day retention, 10 GB cap, smoke test requires ≥15 GB free. | §0, §5.2, §5.14, §12 |
| F7 | lv-chordia chunked (120 s windows, 10 s overlap); CPU fallback via a `CUDA_VISIBLE_DEVICES=""` subprocess. | §5.5, §5.14 |
| F8 | First smoke step is a torchaudio MelSpectrogram on CUDA; fallback torch 2.11.0+cu130. | §0, §12 |
| F9 | API details: `Audio2Beats(checkpoint_path="final0", device="cuda", dbn=False)(signal, sr)`; `transcribe(audio, None)`; the package's `load_audio` not used; lv-chordia `.sdict` assert and absolute paths; BTC `map_location`; ACE hop 23 ms with overlapping 20 s chunks. | §5.3, §5.5, §5.7 |
| F10 | demucs mix tensor kept on the CPU; `segment` ≤ 7.8. | §5.2 |
| F11 | `__main__` guard, worker target in `server/worker.py`, `CREATE_NO_WINDOW` for ffmpeg and pip, update-downloader stops the worker first. | §2, §5.1, §9 |
| F12 | Node listed in `js_runtimes` so it actually serves as the fallback. | §5.1 |
| F13 | WebView2 microphone: default prompt; optional `PermissionRequested` handler through `window.native.Invoke`. | §9 |
| F14 | iOS `audioSession` and tap-to-start; CSP allowing `blob:`, `data:` and `'wasm-unsafe-eval'`; type shim for signalsmith-stretch; the `<audio>` fallback kept. | §8.3, §10, §12 |
| F15 | oxlint instead of typescript-eslint; `create-vite@9.2.1` or a hand scaffold. | §12 |
| F16 | Install script: `$ProgressPreference`, re-run-safe FFmpeg unpack and rename, HF variables set before any import. | §12 |
| F17 | Worker pre-warm; cold start in the runtime table; fast-then-full not worth it without evidence reuse (see L3). | §2, §5, §5.14, §9 |
| F-pins | Verified pins added: audioread 3.1.0, matplotlib 3.11.2, huggingface-hub 2.2.0, safetensors 0.8.0, rotary-embedding-torch 0.9.1, pythonnet 3.2.1, the ACE runtime set, `@types/react` and `@types/react-dom`. | §12 |
| L1 | Question 1 resolved (Cantonese 回家 by Eternity Girls, B major, ~65 BPM); the speculation that the link was a mistake, and its risk row, removed. | §0.1, §15 |
| L2 | New questions: Surfshark, disk space, the chord intake format, the Chordify baseline. | §0.1 |
| L3 | Two-phase fast preview deferred; `--mode fast` is CLI/debug only; Full must reuse Fast's beats and evidence if both ever run; the SSE `preview` event removed. | §5, §11 |
| L4 | Song length: lv-chordia and ACE chunked; default maximum 30 min (configurable). | §5, §5.1, §5.5 |
| L5 | Schema: `schema_version` stays "1.0" (nothing has shipped); fields added: `display`, `calibration`, `key.changes`, `tempo.subdivision` (per section), `tempo.free_time`, `bars[].free`, `sections[].group`, `role`, `equivalents`, `bass_line[].min_detail`, `edits`, `gt`. | §5.13 |
| L6 | New §16 lists the unverified items and the milestone that settles each. | §16 |
