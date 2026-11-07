# Nathan Chordify

A self-hosted chord finder for worship piano, built to be stronger than Chordify on the things it
misses: short passing chords, descending bass lines, slash chords and inversions. Paste a YouTube
link or drop an audio file, and the app shows the chords synced to playback, with transpose,
tempo, loop, a "no singer" mix, a chord-sheet export and an automatic online sheet lookup used as
a reference to refine the detected chords.

這是一個自架的和弦偵測器，為敬拜鋼琴而做：貼上 YouTube 連結或音訊檔，就會顯示跟播放同步的和弦，
支援移調、變速、循環、去人聲播放、匯出和弦譜，並會自動上網找該首歌的和弦譜作參考來修正結果。
給初學者用：用簡單的字解釋，畫面也可以只顯示簡單和弦。

## What is in this repository

| Folder | What it is |
| --- | --- |
| `engine/` | Python engine (`nathan_chordify`): download, stem separation, beat tracking, chord models, decoder, speller, FastAPI server, CLI `nchord`, sheet lookup, audio mixes, export. Tests in `engine/tests/`. |
| `web/` | React + Vite web app (player, chord grid, editor, sheet panel, sound selector, export/print view). |
| `train/` | Synthetic piano chord generator (Salamander Grand Piano), song-mix generator and the ACE fine-tuning scripts. |
| `eval/` | Evaluation harness: answer keys for the test songs (`eval/gt/`), metrics, gates, reports (`eval/reports/*.md`). |
| `docs/` | `DESIGN.md` (the source of truth), `RESEARCH.md`, `DATA_PLAN.md`, research reports and reviews. |
| `data/charts/` | Chord charts harvested from public chord sites (chords and bars only, no lyrics), used for training mixes. |
| `shared/` | Fixtures and test vectors shared by engine and web. |
| `scripts/` | `fetch_models.py` (downloads the model weights), `try_ace_weights.py` (runs a copy of the app with fine-tuned weights), smoke test. |
| `CLAUDE.md` | Project memory: decisions, status, what was tried and what the numbers were. |

Not in the repository (too big or personal; see `.gitignore`): the Python venv, `tools/` (ffmpeg),
model weights under `models/` (only their manifest, calibrations and fine-tune records are kept),
`third_party/` model repos (pinned in `third_party/VERSIONS.txt`), the song cache (audio, stems,
lyrics), training pools and the local song library.

## Requirements

- Windows 11 (developed there; Linux/macOS should work for the engine), Python 3.12, Node 24
- An NVIDIA GPU is recommended (stem separation and the chord models run on it; 8 GB VRAM is plenty)
- ffmpeg on the path or in `tools/ffmpeg/`

## Install

```bash
python -m venv .venv
.venv\Scripts\pip install -r engine/requirements.txt
.venv\Scripts\pip install -e engine
.venv\Scripts\python scripts/fetch_models.py
```

Clone the third-party model repos listed in `third_party/VERSIONS.txt` at the pinned commits
(the file records every local patch). Then build the web app:

```bash
cd web && npm install && npm run build
```

The full step-by-step install (and why each piece is there) is in `docs/DESIGN.md`.

## Run

```bash
.venv\Scripts\nchord serve
```

Then open http://127.0.0.1:8765. On Windows, `Start Original.bat` does the same on port 8764 and
opens the browser. Command line only:

```bash
.venv\Scripts\python -m nathan_chordify.cli analyze <YouTube URL or audio file> --chart
```

Tests: `.venv\Scripts\python -m pytest engine eval train` and `cd web && npm test`.

## Credits

- [consonance-ACE](https://github.com/andreamust/consonance-ACE) (MIT) and lv-chordia for chord
  evidence, Beat This! for beats, Demucs for stem separation, a piano transcription model for the
  piano stem. Exact commits, licences and patches: `third_party/VERSIONS.txt` and `models/MANIFEST.json`.
- [Salamander Grand Piano V3](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html)
  (CC-BY 3.0, Alexander Holm) for the synthetic training chords: `models/piano/salamander/ATTRIBUTION.txt`.
- Chord charts referenced from CantonHymn, Simply Worship, WorshipTogether and Ultimate Guitar are
  used as references only; the recordings used for testing are not stored in this repository.
