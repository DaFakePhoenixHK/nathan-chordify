# Adversarial review of DESIGN.md v1.0: feasibility on this machine

Reviewed 2026-10-13. This is the raw review; the fixes are folded into DESIGN.md v1.1. Claims marked "verified" were run or downloaded on this laptop, in a throwaway scratch venv.

## Machine facts found
- git 2.50.0 and git-lfs 3.6.1 are installed. Python 3.12.4, pip 26.1.2, Node 24.14.1 / npm 11.11.0.
- RTX 4070 Laptop, 8188 MiB, driver 591.86 (CUDA 13.1).
- WebView2 154.0.4258.62, .NET 4.8.1, LongPathsEnabled=1.
- ffmpeg, yt-dlp, deno and tailscale are not installed. Ports 8765 and 8443 are free.
- **Surfshark WireGuard VPN is UP.** Surfshark Antivirus is the primary AV; Defender is passive.
- **The Wi-Fi profile is Public.**
- **Only 37 GB is free on C:.**

## A) Issues

### 1. [BLOCKER] uvicorn crashes under pythonw.exe
- Under pythonw, `sys.stdout` is None, and uvicorn's formatter calls `isatty()`. Verified: `ValueError: Unable to configure formatter 'default'`.
- **Fix:**
  - At the top of desktop.py and of the worker entry: `if sys.stdout is None: sys.stdout = sys.stderr = open(LOG, "a", encoding="utf-8", buffering=1)`. Verified: `/health` then returns 200 under pythonw.
  - Set `HF_HUB_DISABLE_PROGRESS_BARS=1` and `MPLBACKEND=Agg`.
  - Prefetch all weights.

### 2. [MAJOR] consonance-ACE runtime dependencies are incomplete
- **Fix:** `lightning==2.6.6 torchmetrics==1.9.0 gin-config==0.5.0 jams==0.3.5 pumpp==0.6.0 harte-library==0.4.5`. Verified: they import on Py3.12.
- Never `pip install` the ACE repo: it pins beat-this==0.1, lightning==2.5.1 and torch==2.6.0.
- The adapter must:
  - put `third_party/consonance-ACE` on sys.path;
  - import `ACE.mir_evaluation` before `ACE.preprocess.audio_processor`, which does a bare `from transforms import CQTransform`.

### 3. [MAJOR] Module-name collision on `metrics`
- ACE does `from metrics.distances import ...` through a sys.path hack.
- `eval/metrics.py` would shadow it.
- **Fix:** rename it to `eval/chord_metrics.py` and run as `python -m eval.run_eval`.

### 4. [MAJOR] tonal 6.5.0 is a broken publish
- `dist/index.js` is missing, so tsc reports TS2307 and Node's import fails.
- **Fix:** `tonal@6.4.3`. Verified: tsc 7.0.2 and vite build pass.

### 5. [MAJOR] Surfshark VPN and antivirus are active
- Two WireGuard VPNs at once (Surfshark plus Tailscale) commonly conflict.
- YouTube traffic exits through a VPN IP, which raises the bot-check risk. Format listing works today.
- The mkcert LAN fallback breaks if "Invisible on LAN" is on.
- The AV may quarantine deno.exe, ffmpeg DLLs or model files.
- **Fix:**
  - Disconnect Surfshark, or split-tunnel tailscaled.exe, tailscale-ipn.exe and the venv's python/pythonw.
  - For mkcert: Wi-Fi Private and "Invisible on LAN" off.
  - The download-failure hint mentions pausing the VPN.

### 6. [MAJOR] Only 37 GB is free on C:
- The torch wheel is 1.99 GB, about 4–5 GB once installed, plus pip's cache copy.
- Each song caches about 250 MB with WAV stems.
- **Fix:**
  - `pip install --no-cache-dir` for torch.
  - Store stems as FLAC, or drop the stem WAVs after evidence extraction.
  - Storage default 14–30 days plus a size cap (10 GB).
  - The smoke test checks free space ≥ 15 GB.

### 7. [MAJOR] lv-chordia runs the whole song in one pass
- VRAM grows with song length: about 2.5 GB per tensor at 60 min, so it runs out of memory.
- `use_gpu = cuda.device_count() > 0` is hard-coded, so CPU mode needs `CUDA_VISIBLE_DEVICES=""` set before torch is imported.
- **Fix:** chunk in the adapter (120 s windows with 10 s overlap, then crop), or cap the song length.

### 8. [MINOR] torchaudio 2.11.0 with torch 2.14.1 is documented as compatible (stable ABI) but not executed
- **Fix:**
  - First smoke step: `torchaudio.transforms.MelSpectrogram().cuda()(torch.randn(1, 22050, device="cuda"))`.
  - Fallback: torch==2.11.0+cu130.

### 9. [MINOR] API and loading details
- Beat This!: the call is `beat_this.inference.Audio2Beats(checkpoint_path="final0", device="cuda", dbn=False)(signal, sr)`.
- piano-transcription-inference:
  - `transcribe(audio, None)` needs the second argument.
  - Don't use its `load_audio`: it passes a positional resample argument to librosa 0.11 (TypeError).
  - Its `wget` auto-download doesn't exist on Windows, so prefetch the checkpoint.
- lv-chordia:
  - It silently runs with RANDOM weights if the .sdict is missing, so assert that `<sys.prefix>/share/lv-chordia/cache_data/*.sdict` exists.
  - Pass absolute audio paths.
- BTC checkpoint: saved on 'mps', so pass `map_location`. Call `torch.load(weights_only=True)` ourselves.
- ACE: the CQT hop is 512 @ 22.05 kHz = 23 ms (not ~93 ms). Its inference uses fixed 20 s chunks with no overlap; overlap them in the adapter.

### 10. [MINOR] demucs: keep the mix tensor on the CPU
- `apply_model` allocates whole-track buffers on `mix.device`.
- **Fix:** pass a CPU float32 tensor with `Separator(device="cuda")`. segment must be ≤ 7.8.

### 11. [MINOR] Windows process model
- Launching code goes under `if __name__ == "__main__":`; the worker target lives in server/worker.py.
- ffmpeg subprocesses need `CREATE_NO_WINDOW`.
- "Update downloader": stop the worker, run pip in a separate process, then restart. A loaded .pyd can't be replaced.

### 12. [MINOR] Node isn't a JS fallback unless enabled
- **Fix:** `js_runtimes={"deno": {...}, "node": {}}`.

### 13. [MINOR] WebView2 microphone
- pywebview has no PermissionRequested handler, so WebView2 shows its default prompt.
- To auto-allow, subscribe on the UI thread through `window.native.Invoke(...)`.
- Verified: isSecureContext, getUserMedia, AudioWorklet, WASM, service worker and IndexedDB all work.

### 14. [MINOR] Phone audio
- iOS mutes Web Audio with the ring/silent switch: set `navigator.audioSession.type = "playback"` and start the AudioContext from a tap.
- The CSP must allow `blob:`, `data:` and `'wasm-unsafe-eval'`.
- signalsmith-stretch ships no TypeScript types: add `declare module 'signalsmith-stretch'`.
- AudioWorklet + WASM on iOS Safari is unverified on a device, so keep the `<audio>` fallback.

### 15. [MINOR] TypeScript 7
- `tsc --noEmit` and `tsc -b` work (verified), but there's no classic compiler API, so typescript-eslint can't be used. Lint with oxlint.
- `npm create vite@8.3.4` is a 404. Use 9.2.1, or scaffold by hand.

### 16. [MINOR] Install script
- `$ProgressPreference = 'SilentlyContinue'`.
- Guard `Rename-Item` against re-runs.
- `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- Set `HF_HOME` and `TORCH_HOME` before importing torch or huggingface_hub.

### 17. [MINOR] Runtime
- The estimates are plausible: htdemucs_ft takes about 40–60 s for a 4-minute song.
- Cold start adds about 15–30 s, so pre-warm the worker.
- Fast-then-Full on one GPU slot is about Fast + Full in time unless the evidence is reused.

## B) Verified OK (highlights)
- **PyPI:** every Python pin in DESIGN §12 exists with a cp312-win_amd64, abi3 or pure wheel.
- **Scratch venv:** everything except torch installed `--only-binary=:all:` and imports. Specifically:
  - librosa 0.11 runs with scipy 1.18 / numpy 2.5;
  - lv-chordia's `hybrid_cqt` parameters work;
  - mir_eval works;
  - pythonnet loads;
  - deno 2.9.7 runs.
- **GPU stack:**
  - torch 2.14.1+cu130 and torchaudio 2.11.0+cu130 are both on the cu130 index;
  - torchaudio has no torch pin;
  - the 2.11 docs say it is stable-ABI.
- **demucs 4.1.0:**
  - the inference path never imports torchaudio;
  - htdemucs_ft weights come from HF `adefossez/HTDemucs-ft`.
- **Beat This!:**
  - final0.ckpt is reachable (81 MB);
  - it loads with `weights_only=True`;
  - inference is chunked.
- **lv-chordia:**
  - the six per-head softmax outputs (triad, bass, 7, 9, 11, 13) are reachable without forking;
  - its weights are bundled in the wheel;
  - MIT licence.
- **Third-party checkpoints:** every one loads with `weights_only=True`.
- **ACE and BTC checkpoints:** both are ordinary git blobs (not LFS), MIT, at 57.6 MB and 35.9 MB. The piano checkpoint is on Zenodo (172 MB, CC-BY-4.0).
- **FFmpeg:** the 9.0.2 URL exists, and the zip layout matches §12.
- **yt-dlp 2026.8.19 + deno 2.9.7:** resolved the test song with the Mix link cut to one video; bestaudio = 251 (opus); no PO token needed today.
- **pywebview 6.2.1 + pythonnet 3.2.1 + WebView2:** a real window works under pythonw.
- **npm:**
  - all pins exist, except that tonal 6.5.0 is broken;
  - npm install, tsc 7, vite build and PWA generateSW all pass with tonal 6.4.3.
- **Tailscale serve:**
  - SSE passes through (Go ReverseProxy flushes event-stream);
  - `serve --bg` persists;
  - a ts.net certificate gives a secure context.
- **VRAM:** below 5 GB is realistic for songs up to about 8 min, as long as the mix stays on the CPU and lv-chordia is chunked.

## C) Verified pins
See DESIGN.md v1.1 §12. Changes from v1.0:
- tonal 6.5.0 → 6.4.3.
- Added:
  - audioread 3.1.0
  - matplotlib 3.11.2
  - huggingface-hub 2.2.0
  - safetensors 0.8.0
  - rotary-embedding-torch 0.9.1
  - pythonnet 3.2.1
  - the ACE runtime set (issue 2)
  - `@types/react`
  - `@types/react-dom`
- torch is installed with `--no-cache-dir`.

## Not verifiable here (the WP0 smoke test must cover)
- torch 2.14.1 + torchaudio 2.11 importing together.
- Real CUDA runs, with VRAM and timing measured.
- The WebView2 microphone prompt.
- AudioWorklet + WASM on iOS Safari.
- Tailscale running alongside Surfshark.
