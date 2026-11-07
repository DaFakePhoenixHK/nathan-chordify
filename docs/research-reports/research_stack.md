# Partial notes: research:stack

(Interim notes from a stopped research agent; NOT a finished report.)

Nothing was installed. That includes the scratch venv, because the user hasn't approved downloading or running packages. Everything below comes from local read-only checks and from PyPI, npm, GitHub and vendor documentation.

**Note for the orchestrator:** the user's only message was the YouTube link plus "i pressed wrong!", so this run may have started by accident. This report made no changes to the machine. Please confirm with the user before installing anything into `C:\Users\User\Desktop\Nathan_chordify`.

# Nathan_chordify: verified software stack (as of 2026-10-13)

## 0. What I checked

**Local, read-only:**
- Python 3.12.4, pip 26.1.2, Node v24.14.1, git 2.50.0.
- Visual Studio Community 2022 (17.12) with MSVC 14.42.34433, so C extensions can be built.
- Edge WebView2 Runtime 154.0.4258.62 is installed. The Windows UI language appears to be zh-TW (the runtime is listed as "執行階段").
- `nvidia-smi`: RTX 4070 Laptop, driver 591.86, highest CUDA version supported is **13.1**.
- Not installed: ffmpeg, yt-dlp, deno, tailscale, mkcert, cloudflared, Rust/cargo.
- The global site-packages has a broken leftover `~ip` folder from an interrupted pip upgrade. It only causes a harmless warning; delete the `~ip*` folders to clear it.

**Wheel availability:** I ran `pip index versions <pkg> --only-binary=:all: --platform win_amd64 --python-version 3.12` for about 45 packages. This only reads index metadata; nothing is downloaded. Results are in the table in section 10.

**Not done:** the test installs in the scratch venv. If the user approves, run the script in section 11.

---

## (a) Downloading YouTube audio

**yt-dlp version.** The latest is **2026.08.19** (released 2026-08-19), the last 2026.x release on GitHub. Its notes say it removed `android_vr` from the default clients and added support for curl_cffi 0.16. Release 2026.07.04 raised the recommended minimum Python to 3.11. Sources: [GitHub releases](https://github.com/yt-dlp/yt-dlp/releases), [latest-release API](https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest).

**A JavaScript runtime is now required (the "EJS" change).**
- Since 2025.11.12, YouTube needs an external JS runtime. Without one, YouTube support is "deprecated" and far fewer formats are available. Source: [GIGAZINE](https://gigazine.net/news/20251113-yt-dlp-required-deno-javascript-runtime/).
- Supported runtimes ([EJS wiki](https://github.com/yt-dlp/yt-dlp/wiki/EJS)):
  - **Deno ≥ 2.3.0**: the only one enabled by default.
  - **Node ≥ 22.0.0**: enable with `--js-runtimes node`, or `js_runtimes={'node': {}}` in Python.
  - Bun: deprecated; versions above 1.3.14 are unsupported.
  - QuickJS.
- **Node 24.14.1 qualifies.**
- The EJS scripts come from `yt-dlp-ejs`. It is included by `pip install "yt-dlp[default]"`, which pins `yt-dlp-ejs==0.8.0` ([pyproject](https://raw.githubusercontent.com/yt-dlp/yt-dlp/master/pyproject.toml), [PyPI](https://pypi.org/project/yt-dlp-ejs/)).
- **Recommendation: install `deno==2.9.7` from PyPI into the venv.** It is the official Deno binary redistributed as a wheel ([deno_pypi](https://github.com/denoland/deno_pypi)). It needs no admin rights, puts `deno.exe` on the venv PATH, and yt-dlp picks it up with no flags. Node 24 is the fallback.
- In Python, the option is `js_runtimes` (a dict of runtime name to `{}` or `{'path': ...}`). The default is `{'deno': {}}` ([YoutubeDL.py](https://raw.githubusercontent.com/yt-dlp/yt-dlp/master/yt_dlp/YoutubeDL.py)).

**PO tokens and SABR** (guide edited 2026-07-12, [PO-Token-Guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide)):
- The `web` client only offers SABR formats, which yt-dlp skips. `tv` needs no token. `mweb` and `web_safari` need GVS tokens, but `web_safari`'s HLS formats currently don't.
- Start with the default clients.
- If you get 403 errors or "confirm you're not a bot":
  1. Upgrade yt-dlp.
  2. Use `--cookies-from-browser firefox`. Chrome cookie extraction on Windows fails because of App-Bound Encryption ([issue #10927](https://github.com/yt-dlp/yt-dlp/issues/10927)).
  3. Add **bgutil-ytdlp-pot-provider 2.0.2** (released 2026-10-07). Its script mode uses Node ≥ 22 or Deno ≥ 2, and it can also run as an HTTP server ([PyPI](https://pypi.org/project/bgutil-ytdlp-pot-provider/)).

**The test URL is a YouTube Mix.** It contains `&list=RDY7boUFbM76k&start_radio=1`, so always set `noplaylist: True`, otherwise yt-dlp will walk the whole radio playlist.

**Pinning.** yt-dlp breaks whenever YouTube changes. Use `yt-dlp[default]>=2026.8.19` and give the app an "update downloader" button that runs `pip install -U "yt-dlp[default]"`, rather than freezing one version.

**FFmpeg: choose the app-local Gyan 9.0.2 "full_build-shared" zip.**
- Download `https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-full_build-shared.zip` (released 2026-09-20, [API](https://api.github.com/repos/GyanD/codexffmpeg/releases/latest)) and extract it to `Nathan_chordify\tools\ffmpeg\`.
- Pass `ffmpeg_location` to yt-dlp and prepend `tools\ffmpeg\bin` to PATH at startup.

Why this option:
- It needs no admin rights and is pinned to a version.
- It includes ffprobe, which yt-dlp needs.
- It includes **librubberband**, so server-side pitch shifting can use ffmpeg's `rubberband` filter ([gyan.dev](https://www.gyan.dev/ffmpeg/builds/)).
- Because it is a shared build, its DLLs also satisfy TorchCodec if that is ever needed. TorchCodec supports FFmpeg 4–9 but requires a shared build on Windows ([PyPI torchcodec](https://pypi.org/project/torchcodec/)).

Why not the others:
- `imageio-ffmpeg 0.6.0` has no ffprobe.
- `static-ffmpeg 3.0` (FFmpeg 8.0) downloads binaries on first use from a third-party repo ([PyPI](https://pypi.org/project/static-ffmpeg/)).
- yt-dlp/FFmpeg-Builds currently ships only nightly master builds ([wiki](https://github.com/yt-dlp/FFmpeg-Builds/wiki/Latest)).
- winget installs of portable zips can fail to create the `Links` shim without Developer Mode ([restic forum](https://forum.restic.net/t/restic-link-not-present-in-0-18-1-release-on-winget/10343)).

Python usage:
```python
import deno, yt_dlp
opts = {"format": "bestaudio/best", "noplaylist": True,
        "outtmpl": str(cache_dir / "source.%(ext)s"),
        "ffmpeg_location": str(FFMPEG_BIN),
        "js_runtimes": {"deno": {"path": deno.find_deno_bin()}},   # or {"node": {}}
        "progress_hooks": [hook], "quiet": True}
# then: ffmpeg -i source.* -ac 2 -ar 44100 -c:a pcm_f32le audio.wav   (one canonical decode for every model)
```

## (b) PyTorch with CUDA and the torchaudio change

**Release matrix** ([RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)):
- Torch 2.14 supports Python 3.10–3.15, with stable CUDA builds 12.6, 13.0 and 13.2.
- **CUDA 12.8 is gone, and 2.14 is the last release with any CUDA 12.x build** ([dev-discuss](https://dev-discuss.pytorch.org/t/pytorch-2-14-final-rc-available/3429)).
- The latest version is **torch 2.14.1** (2026-09-30, [PyPI](https://pypi.org/project/torch/)).
- `torch-2.14.1+cu130-cp312-cp312-win_amd64.whl` exists on the cu130 index ([index](https://download.pytorch.org/whl/cu130/torch/)).

**Choose cu130.**
- The driver supports up to CUDA 13.1, so cu130 runs natively.
- cu132 would depend on CUDA minor-version compatibility. Avoid it unless the driver is updated.
- cu126 has no future after 2.14.
- The RTX 4070 (Ada, sm_89) is supported by CUDA 13.

**torchaudio:**
- 2.8 deprecated I/O and 2.9 removed it. `torchaudio.load` and `save` are now thin wrappers around **TorchCodec** ([issue #3902](https://github.com/pytorch/audio/issues/3902), [2.11 docs](https://docs.pytorch.org/audio/2.11.0/installation.html)).
- **torchaudio 2.11.0 (2026-03-23) is the final release.** It is built against the stable ABI and is documented as working with "every future PyTorch release" ([installation docs](https://docs.pytorch.org/audio/2.11.0/installation.html), [dev-discuss](https://dev-discuss.pytorch.org/t/pytorch-2-10-rc1-produced-for-pytorch-torchvision-torchaudio/3281)).
- Its `setup.py` has `install_requires=[]`, so pip won't object to pairing it with torch 2.14.1 ([setup.py](https://github.com/pytorch/audio/blob/release/2.11/setup.py)).
- `torchaudio-2.11.0+cu130-cp312-win_amd64` is on the cu130 index ([index](https://download.pytorch.org/whl/cu130/torchaudio/)).
- TorchCodec 0.17.0 has Windows cp312 wheels. Its CUDA build needs the PyTorch index and FFmpeg shared DLLs. **Recommendation: don't install TorchCodec. Do all decoding with the ffmpeg CLI to WAV, then read with soundfile.**

**Impact on the audio packages:**

| Package | Impact | Action |
|---|---|---|
| **demucs 4.1.0** (2026-07-11, now maintained at adefossez/demucs) | None. Its notes say "Removed torchaudio for inference". Decoding uses **sphn** (Rust/symphonia) with an ffmpeg fallback; FLAC output needs ffmpeg; weights load from the Hugging Face hub as safetensors; requires torch ≥ 2.1 ([release.md](https://github.com/adefossez/demucs/blob/main/docs/release.md), [PyPI JSON](https://pypi.org/pypi/demucs/json)). | Use 4.1.0. Older 4.0.1 called `torchaudio.save`, which now fails without TorchCodec ([HF forum](https://discuss.huggingface.co/t/cannot-load-torchcodec/169260)). |
| **audio-separator 0.47.0** (2026-08-27) | None. It doesn't depend on torchaudio. Requires torch ≥ 2.3, numpy ≥ 2, `rotary-embedding-torch<0.7`, `beartype<0.19`, `samplerate==0.1.0` (a py3-none-any wheel that bundles the Windows DLL). The `[gpu]` extra pulls `onnxruntime-gpu>=1.17` ([PyPI JSON](https://pypi.org/pypi/audio-separator/json)). Its README says DirectML can't run Demucs, which doesn't matter on CUDA. | **onnxruntime-gpu 1.31.0** (2026-10-09) is built for **CUDA 13.0 / cuDNN 9** and works alongside torch cu130. Import torch first or call `onnxruntime.preload_dlls()` ([ORT CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)). Never install both `onnxruntime` and `onnxruntime-gpu`. |
| **beat-this 1.1.0** (on PyPI since 2026-04-14) | Imports torchaudio for `MelSpectrogram` (still present in 2.11). `load_audio` tries `torchaudio.load` first, then soundfile, then madmom, so a missing TorchCodec just falls through to soundfile ([preprocessing.py](https://raw.githubusercontent.com/CPJKU/beat_this/main/beat_this/preprocessing.py)). | Keep torchaudio 2.11.0. Pass arrays through `Audio2Beats`. Watch the rotary-embedding-torch constraint: audio-separator caps it below 0.7 and beat-this leaves it open, so the resolver will choose 0.6.x. Confirm beat-this works with that version in the test venv. |
| Chord models (BTC-style, madmom CNN/CRF, custom) | They use librosa/numpy features, not torchaudio I/O. | Same approach: decode once to WAV. |

## (c) MIR libraries on Python 3.12 / numpy 2

| Package | Status | Install |
|---|---|---|
| **madmom** | PyPI only has 0.16.1 (2018, source only, Python ≤ 3.7 era) ([PyPI](https://pypi.org/project/madmom/)). Git `main` at `27f032e` (2024-08-25, PR #540) added Python 3.11/3.12, fixed numpy 2 / 2.1 / Cython issues, and replaced `imp` ([PR #540](https://github.com/CPJKU/madmom/pull/540)). Its `pyproject` builds against `numpy>2` and `cython>=0.25`. CI runs on **Ubuntu only, Python 3.9–3.12** ([ci.yml](https://raw.githubusercontent.com/CPJKU/madmom/main/.github/workflows/ci.yml)). | `pip install "madmom @ git+https://github.com/CPJKU/madmom.git@27f032e8947204902c675e5e341a3faf5dc86dae"`. It needs the MSVC already installed here; pip also fetches the `madmom_models` submodule. **Expected to build, but not CI-tested on Windows and not tested here.** Beat-this recommends exactly this git install for `dbn=True` ([beat_this](https://github.com/CPJKU/beat_this)). |
| **librosa 1.0.0** (2026-08-11) | Requires **Python ≥ 3.12** and numpy ≥ 2.1. **audioread backend removed**, so only soundfile formats (WAV, FLAC, OGG, MP3) load; no m4a/webm. Also removed: `filters.constant_q*` and `vqt(res_type=None)` ([blog](https://librosa.org/blog/posts/1.0/), [changelog](https://librosa.org/doc/main/changelog.html)). | OK. Feed it WAV. Older research code that calls removed APIs needs small patches. |
| numba 0.68.0 / llvmlite 0.50.0 | Requires numpy < 2.6 ([PyPI](https://pypi.org/pypi/numba/json)). | Pin numpy 2.5.4. |
| mir_eval 0.8.2, pretty_midi 0.2.11.post0 (2026-07-28), mido 1.3.3, soundfile 0.14.0 (bundles libsndfile on win_amd64), soxr 1.1.0 | Wheels available. | OK. |
| **basic-pitch 0.4.0** (2024-08-16, the latest) | **Will not install normally on Windows with Python 3.12.** It requires `tensorflow<2.15.1` for Python ≥ 3.11, and TF 2.15.1 only has cp39–cp311 wheels ([PyPI JSON](https://pypi.org/pypi/basic-pitch/json), [TF 2.15.1](https://pypi.org/project/tensorflow/2.15.1/)). The package ships `nmp.onnx` ([repo](https://github.com/spotify/basic-pitch/tree/main/basic_pitch/saved_models/icassp_2022)). | `pip install basic-pitch==0.4.0 --no-deps` plus `resampy scikit-learn`, then run the ONNX model through the onnxruntime-gpu module already installed. Optional; it isn't piano-specific. |
| **piano-transcription-inference 0.0.6** (2025-01-26) + torchlibrosa 0.1.0 | Pure Python; depends on librosa and torchlibrosa. Its checkpoint is on Zenodo ([PyPI](https://pypi.org/pypi/piano-transcription-inference/json)). **Unverified:** with torch ≥ 2.6, `torch.load` defaults to `weights_only=True`; if loading fails, patch the call to `weights_only=False` (the checkpoint is trusted). torchlibrosa 0.1.0 with librosa 1.0 is also unverified. | Optional. Test in the scratch venv. |
| **chord-extractor 0.1.3** | `requires_python <3.12,>=3.8`, so pip reports "No matching distribution" on 3.12 (confirmed). | **Skip.** |
| **vamp 1.1.0** | Latest is from 2015, with only cp27 wheels plus a source tarball ([PyPI JSON](https://pypi.org/pypi/vamp/json)). | **Skip.** If a Chordino baseline is wanted, call Sonic Annotator and the Vamp Plugin Pack 2.0 (x64) as a subprocess (unverified on this machine). |
| sphn 0.2.1 | cp312 win_amd64 wheel available. | Installed automatically with demucs. |

## (d) Backend

**Versions:** FastAPI **0.143.0** (requires Python ≥ 3.10) and uvicorn[standard] **0.54.0** (uvloop is skipped on Windows automatically) ([PyPI](https://pypi.org/pypi/uvicorn/json)).

**Progress over SSE, using FastAPI's native support.**
- `fastapi.sse.EventSourceResponse` was added in 0.135.0. Endpoints are async generators, and FastAPI sends a keep-alive ping every 15 s ([docs](https://fastapi.tiangolo.com/tutorial/server-sent-events/)). No `sse-starlette` needed.
- SSE is a better fit than WebSocket for one-way progress: `EventSource` reconnects on its own, which helps on phones, and it passes through Tailscale serve. (Cloudflare quick tunnels don't support SSE; see g.)

**Job queue.** Run uvicorn as a single process with a single-slot GPU worker. Don't use Celery or RQ: neither is a good fit on Windows (from general knowledge; RQ relies on `fork()`).
- Create the worker as `ProcessPoolExecutor(max_workers=1, mp_context=spawn)`, or one long-lived `multiprocessing` process.
- Its initializer loads the models once and keeps them on the GPU.
- Progress goes through an `mp.Queue`, a bridge thread, `loop.call_soon_threadsafe`, and per-job subscribers to the SSE endpoint `GET /api/jobs/{id}/events`.
- A crash such as a CUDA OOM kills only the worker, which can be restarted.

**Storage and cache.**
- Use SQLite (stdlib `sqlite3`, or `sqlmodel 0.0.48`) for a `jobs` table and a `tracks` table.
- Cache keys:
  - YouTube: `yt-{video_id}`. Get the canonical 11-character ID from `extract_info(download=False)` with `noplaylist`, which also normalises `youtu.be`, `shorts` and `&list=` links.
  - Uploads: `sha256-{hash}`, computed while streaming the upload.
- Store results as `cache/{key}/analysis.v{ENGINE_VERSION}-{params_hash}.json` next to `source.*`, `audio.wav` and `stems/`. Improving the engine then re-analyses without downloading again.
- Set `HF_HOME` and `TORCH_HOME` to `Nathan_chordify\models\` so model weights stay local.

## (e) Frontend

**Recommendation: Vite 8.3.4 + React 19.3.0 + TypeScript 7.0.2** (@vitejs/plugin-react 6.1.2). Vite needs Node ^20.19 or ≥ 22.12, so Node 24 is fine ([npm](https://registry.npmjs.org/vite/latest)).
- **Why React over vanilla TS:** the Premium feature set is big (library, setlists, chord editing, capo/transpose, exports, settings, zh-TW strings), and React's components make that easier to build and maintain.
- Keep the playback clock and chord highlighting outside React. Use a `requestAnimationFrame` loop that reads the audio clock and toggles CSS classes directly, so nothing re-renders every frame.
- Other libraries: `tonal 6.5.0` (MIT) for chord parsing and transposition, `zustand 5.0.15`, `idb-keyval 6.3.0`, `vite-plugin-pwa 2.0.0` (supports Vite 8) and `workbox-window 7.4.1`.

**Playback: use the downloaded audio, not the YouTube IFrame API.**
- The IFrame API only offers the rates in `getAvailablePlaybackRates()` (for example 0.25–2), doesn't guarantee a rate change, exposes no audio data and has no pitch control. Mobile autoplay is also blocked ([IFrame API](https://developers.google.com/youtube/iframe_api_reference)).

Two tiers of playback:
1. **Simple:** an `<audio>` element with `playbackRate` and `preservesPitch` (default true; Baseline-supported since Dec 2023, [MDN](https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/preservesPitch)). A-B looping here works by polling the clock and seeking, which is not sample-accurate.
2. **Recommended:** **signalsmith-stretch 1.3.2** (MIT, WASM + AudioWorklet). A single `schedule({rate, semitones, loopStart, loopEnd, input, output})` call covers tempo change, **real pitch-shift for transpose**, and gapless A-B looping ([README](https://unpkg.com/signalsmith-stretch@1.3.2/README.md), [npm](https://registry.npmjs.org/signalsmith-stretch/latest)).

Alternatives:
- SoundTouchJS 0.3.0 / @soundtouchjs/audio-worklet 2.1.1 (LGPL-2.1).
- rubberband-web 0.2.1 (GPL).
- Server-side: `ffmpeg -af rubberband=pitch=2^(n/12)` using the Gyan build, cached per semitone. `pyrubberband 0.4.0` needs a separate rubberband CLI, so the ffmpeg filter is simpler.

## (f) Windows desktop app

**Recommendation: pywebview 6.2.1** (pythonnet 3.2.1 on Windows) wrapping the local FastAPI server ([PyPI JSON](https://pypi.org/pypi/pywebview/json)). It renders in Edge WebView2, which is already installed (154.0.4258.62).
- **Tauri:** needs a Rust toolchain (not installed) plus a Python sidecar.
- **Electron:** adds about 100 MB+ of runtime and still needs the Python backend.
- **PyInstaller:** avoid. A torch + CUDA bundle is several GB, slow to start, and tends to trigger antivirus false positives.

Launcher:
- `desktop.py` starts uvicorn in a thread on `127.0.0.1:8765`, polls `/health`, then calls `webview.create_window("Nathan Chordify", url)` and `webview.start()`.
- `Nathan_chordify.bat` runs `start "" "%~dp0.venv\Scripts\pythonw.exe" -m nathan_chordify.desktop`.
- Create a desktop `.lnk` with PowerShell's `WScript.Shell` COM object, pointing it at `pythonw.exe` with an icon.

## (g) Phone PWA over Wi-Fi

Service workers and installation need HTTPS; a plain `http://192.168.x.x` address won't work. One forum user confirms the PWA was unusable until a private-CA certificate was installed ([moOde forum](https://moodeaudio.org/forum/showthread.php?pid=60715)).

| Option | Verdict |
|---|---|
| **Tailscale serve** (recommended) | `tailscale serve --bg 8765` gives `https://<pc>.<tailnet>.ts.net` with an automatically provisioned public certificate. It keeps running across reboots, and nothing needs installing on the phone besides the Tailscale app. It also works away from home, and SSE and WebSocket pass through ([serve CLI](https://tailscale.com/kb/1242/tailscale-serve)). Requirements: enable MagicDNS and HTTPS certificates in the admin console. The machine and tailnet names are published in Certificate Transparency logs, so pick neutral names ([HTTPS](https://tailscale.com/kb/1153/enabling-https)). uvicorn can stay on `127.0.0.1`, so **no firewall rule is needed**. Needs a free Tailscale account. |
| **mkcert v1.4.4** (fallback without an account; the latest release is from 2022) | Run `mkcert -install`, then `mkcert -cert-file lan.pem -key-file lan-key.pem 192.168.x.y <pcname>.local localhost`, and start uvicorn with `--ssl-certfile/--ssl-keyfile`. **Copy only `rootCA.pem` to the phone, never `rootCA-key.pem`** ([mkcert](https://github.com/FiloSottile/mkcert)). **Android:** Settings, then Security, Encryption & credentials, Install a certificate, CA certificate. Chrome trusts CAs the user installs (other apps may not). **iOS:** install the profile, then turn on Settings > General > About > Certificate Trust Settings > full trust ([Apple](https://support.apple.com/en-ca/102390)). Fully close Safari afterwards; some iOS 18 users report the trust taking a while to apply ([Apple forums](https://developer.apple.com/forums/thread/764673?page=5)). Reserve the PC's IP in DHCP so the certificate keeps matching. |
| Self-signed certificate | No. Browsers show a warning page and won't install a PWA from that origin. |
| Cloudflare quick tunnel | No. It doesn't support SSE, the hostname changes every run (which breaks the installed PWA), it caps at 200 in-flight requests, and it exposes the app to the internet ([Cloudflare docs](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)). |

**Firewall, mkcert path only.** Run once as admin:
```powershell
Get-NetConnectionProfile   # Wi-Fi must be "Private"; else: Set-NetConnectionProfile -InterfaceAlias "Wi-Fi" -NetworkCategory Private
New-NetFirewallRule -DisplayName "Nathan_chordify HTTPS 8443" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8443 -Profile Private
```

## (h) Exports

- **MIDI:** pretty_midi 0.2.11.post0. Write a chord track with the actual voicings, including slash-chord bass, plus a tempo/beat map from beat-this.
- **PDF:** browser print CSS (`@page A4`, `break-inside: avoid`). It handles Traditional Chinese lyrics without extra font work, and print-to-PDF is available from WebView2, Android Chrome and iOS Share > Print. reportlab 5.0.1 is an optional server-side alternative; it needs a CJK font (its built-in `MSung-Light` CID font, or Noto Sans TC).
- **ChordPro and plain text:** write them directly with string templates.
- **MusicXML:** music21 10.5.0 (Python ≥ 3.11) using `harmony.ChordSymbol`, which writes `<harmony>` with `<bass>` for slash chords. Writing those elements by hand also works.

## (i) Piano chord diagrams

Generate inline SVG in TypeScript; no library needed.
- Draw a 2-octave window starting near the voicing's lowest note. White keys about 24×120 and black keys about 14×76.
- Fill chord tones in one colour and the **bass or slash note** in a different colour; optionally add note names or degrees.
- Choose voicings with a small voice-leading function that keeps each chord's inversion close to the previous voicing.
- Use `tonal` (`Chord.get`) for spelling and transposition. I haven't verified that it parses the slash-chord bass correctly; test it.
- Use `currentColor` and CSS variables so dark mode works.

---

## Recommended stack (pinned)

| Layer | Pin |
|---|---|
| Python | 3.12.4 (existing), project venv at `Nathan_chordify\.venv` |
| GPU | `torch==2.14.1+cu130`, `torchaudio==2.11.0+cu130` (index `https://download.pytorch.org/whl/cu130`); **no torchcodec** |
| Numerics | `numpy==2.5.4`, `scipy==1.18.1`, `numba==0.68.0`, `llvmlite==0.50.0` |
| Audio / MIR | `soundfile==0.14.0`, `soxr==1.1.0`, `librosa==1.0.0`, `mir_eval==0.8.2`, `pretty_midi==0.2.11.post0`, `mido==1.3.3`, `music21==10.5.0`, `cython==3.3.0`, madmom git `27f032e` |
| Separation / beats | `demucs==4.1.0` (brings `sphn==0.2.1`, `lameenc==1.8.4`), `beat-this==1.1.0`, `einops==0.8.2`; optional `audio-separator[gpu]==0.47.0` + `onnxruntime-gpu==1.31.0` |
| Optional | `basic-pitch==0.4.0 --no-deps`, `piano-transcription-inference==0.0.6`, `torchlibrosa==0.1.0` |
| Download | `yt-dlp[default]>=2026.8.19` (yt-dlp-ejs 0.8.0), `deno==2.9.7`; `bgutil-ytdlp-pot-provider==2.0.2` only if needed; FFmpeg **9.0.2 Gyan full_build-shared**, app-local |
| Server | `fastapi==0.143.0`, `uvicorn[standard]==0.54.0`, `python-multipart==0.0.32`, `pydantic==2.14.1`, `sqlmodel==0.0.48` (optional) |
| Desktop | `pywebview==6.2.1` (pythonnet 3.2.1), WebView2 154 (already installed) |
| PDF (optional) | `reportlab==5.0.1` |
| Frontend | `vite@8.3.4`, `react@19.3.0`, `react-dom@19.3.0`, `@vitejs/plugin-react@6.1.2`, `typescript@7.0.2`, `vite-plugin-pwa@2.0.0`, `workbox-window@7.4.1`, `tonal@6.5.0`, `signalsmith-stretch@1.3.2`, `zustand@5.0.15`, `idb-keyval@6.3.0` |
| HTTPS for the phone | Tailscale serve (primary); mkcert v1.4.4 (fallback) |

## 11. Install commands (run only after the user approves)

```powershell
cd C:\Users\User\Desktop\Nathan_chordify
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip setuptools wheel

# GPU stack
.\.venv\Scripts\pip install torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python -c "import torch,torchaudio;print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0),torchaudio.__version__)"

# Core MIR + server + desktop + downloader
.\.venv\Scripts\pip install numpy==2.5.4 scipy==1.18.1 numba==0.68.0 soundfile==0.14.0 soxr==1.1.0 librosa==1.0.0 mir_eval==0.8.2 pretty_midi==0.2.11.post0 mido==1.3.3 music21==10.5.0 cython==3.3.0 `
  demucs==4.1.0 beat-this==1.1.0 einops==0.8.2 `
  fastapi==0.143.0 "uvicorn[standard]==0.54.0" python-multipart==0.0.32 pywebview==6.2.1 `
  "yt-dlp[default]>=2026.8.19" deno==2.9.7

# madmom from git (builds with installed MSVC 14.42; needed for madmom features + beat-this dbn=True)
.\.venv\Scripts\pip install "madmom @ git+https://github.com/CPJKU/madmom.git@27f032e8947204902c675e5e341a3faf5dc86dae"

# Optional: Roformer/MDX separation (CUDA 13 ORT; import torch before creating ORT sessions)
.\.venv\Scripts\pip install "audio-separator[gpu]==0.47.0"

# FFmpeg 9.0.2 shared, app-local (no admin)
$z="$env:TEMP\ffmpeg-9.0.2.zip"; Invoke-WebRequest https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-full_build-shared.zip -OutFile $z
Expand-Archive $z -DestinationPath .\tools; Rename-Item .\tools\ffmpeg-9.0.2-full_build-shared ffmpeg

# Frontend
npm create vite@8.3.4 web -- --template react-ts
cd web; npm i tonal@6.5.0 signalsmith-stretch@1.3.2 zustand@5.0.15 idb-keyval@6.3.0 workbox-window@7.4.1; npm i -D vite-plugin-pwa@2.0.0
```

**Compatibility smoke test, for the scratch venv the task allowed, once approved.** Create the venv at `...\scratchpad\venvtest` and run `pip install` separately for each of: `"madmom @ git+...@27f032e"`, `beat-this==1.1.0`, `demucs==4.1.0`, `"basic-pitch==0.4.0"` (expected to fail on the tensorflow pin), `chord-extractor` (expected to fail: requires Python < 3.12), `vamp==1.1.0` (expected to need a source build), `mir_eval`, `pretty_midi`, `"yt-dlp[default]"`, `deno`, `static-ffmpeg`. Then run `python -c "import madmom; print(madmom.__version__)"`.

**Open risks to confirm in testing:**
- madmom has never been built in CI on Windows.
- beat-this with rotary-embedding-torch 0.6.x, as forced by audio-separator.
- piano-transcription-inference with torch ≥ 2.6 (`weights_only` default).
- yt-dlp needing a PO-token provider for this video.