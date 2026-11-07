# Nathan_chordify: research digest

Compiled 2026-10-13 from five research reports (chordify-features, sota-models, passing-chords, stack, test-song). This digest keeps the concrete facts, numbers and URLs. The decisions built on them are in [DESIGN.md](DESIGN.md).

**Provenance tags**

| Tag | Meaning |
|---|---|
| [O] | Chordify's own page, seen only through a search-engine snippet. chordify.net and support.chordify.net return HTTP 403 or a Cloudflare human check, and nobody tried to get past it. |
| [O-direct] | A Chordify-authored paper that was read in full. |
| [3P] | A third-party source. |
| [INF] | Inference by a researcher. |
| (est.) | An estimate. Nothing was benchmarked. |
| (dry-run) | Checked with `pip install --dry-run` or `pip index versions` only. |

**Nothing has been installed on the machine yet.** No venv exists, and no audio was downloaded or saved. The test-song audio check analysed the browser's audio stream in memory only.

**The intent needs confirming.** The only user message relayed with this run was the test-song link plus "i pressed wrong!". Several other songs are also called 回家 (see §5.0). Confirm the intended test song before ground-truth work starts.

---

## 1. Chordify: what it does and where it fails

### 1.1 Plans (October 2026)

| Plan | Price (yearly billing) | What it adds | Source |
|---|---|---|---|
| Free | $0, with ads | Chords for any song in a 36M-song catalog. Animated diagrams for guitar, piano and ukulele. YouTube links. Library sync between desktop and mobile. Simplify Chords. Free multi-string tuner. A 2024 review reports a limit of about 4 song pages per day; this is unconfirmed for 2026. | [App Store](https://apps.apple.com/us/app/chordify-songs-chords-tuner/id1073624757), [Hooktheory](https://www.hooktheory.com/blog/chordify-alternatives/), [GuitarChalk 2026](https://www.guitarchalk.com/chordify-review/) |
| Basic | $24/yr ($2/mo). About $4/mo on monthly billing (GuitarChalk). | Unlimited song access | [O] [chordify.net/premium](https://chordify.net/premium), [saasswitcher Jun 2026](https://www.saasswitcher.com/blog/chordify-alternatives) |
| Premium | $42/yr ($3.50/mo). About $8/mo on monthly billing. | Transpose, capo, tempo, loop, MIDI, PDF/print, setlists, offline mode, own uploads, separate song and chord volume | [O] same page; [GuitarChalk](https://www.guitarchalk.com/chordify-review/) |
| Premium Plus (Premium + Toolkit) | $54/yr. A promotion on 1 Oct 2026 showed $27 for the first year. | Premium, plus the Toolkit: live chord detection, chromatic tuner, metronome with drumbeat, chord trainer, "Generate Lyrics", and "Learn Songs (Beta)" on iOS | [O] [Toolkit help](https://support.chordify.net/hc/en-us/articles/11755805765661-How-to-use-the-Toolkit-Premium-Plus-subscription-required), [premium-plus-toolkit](https://chordify.net/premium-plus-toolkit) |

- No Edu, Pro or School plan was found.
- iOS in-app prices are $8.00, $8.99 or $9.00 per month, and $42 or $54 per year.
- The snippet of the Premium page is garbled about which plan includes the Toolkit. The support pages say only Premium Plus does.

### 1.2 How each feature behaves

| Feature | Behaviour | Source |
|---|---|---|
| **Transpose** (Premium) | On the web, the first toolbar item after Rewind and Play. It shows the detected key with up/down arrows, one semitone per click, and re-renders every diagram. The range is not documented. **The audio is not pitch-shifted.** That needs the Chromium extension "Transpose by Chordify" (v0.5, updated 18 Nov 2025, about 5k users, 3.6★), which uses the **Signalsmith Audio** shifter. Signalsmith Stretch is MIT-licensed and published on npm as `signalsmith-stretch@1.3.2`. | [O] [toolbox blog](https://chordify.net/pages/chordify-toolbox/), [Premium help](https://support.chordify.net/hc/en-us/articles/360002164538-How-to-use-Premium-features), [Chrome Web Store](https://chromewebstore.google.com/detail/transpose-by-chordify/mckabjbenkkimhkplcapjjhhpbkcfaha), [unpkg README](https://unpkg.com/signalsmith-stretch@1.3.2/README.md) |
| **Capo** (Premium) | A capo control sits above the chords. Diagrams change to the shapes played with that capo while the sounding key stays the same. No automatic "best capo" was found, and the fret range is undocumented. | [O] |
| **Tempo** (Premium) | The web shows "100%" with the BPM underneath. Current help text: YouTube songs go from **50% to 200% in 5% steps**. A Sept 2023 staff reply described 25% steps for YouTube and 10% for uploads. The apps show a multiplier ("1.00"). Users have asked to type in a tempo. | [O] [Premium help](https://support.chordify.net/hc/en-us/articles/360002164538-How-to-use-Premium-features), [thread](https://support.chordify.net/hc/en-us/community/posts/13224678740637-Ability-to-change-the-BPM-with-smaller-increments), [thread](https://support.chordify.net/hc/en-us/community/posts/360005503297-Ability-for-any-user-to-manually-set-the-tempo-not-just-change-the-speed) |
| **Loop** (Premium) | Click Loop, then drag across the beat squares. It repeats until Loop or X is pressed again. There are no A/B buttons, and loop count and count-in are undocumented. | [O] |
| **Volume** | Separate Song and Chords volume; the chords are played with a MIDI piano sound. Reviews mention a metronome volume, and one says the metronome lags at some speeds. | [O] [toolbox blog](https://chordify.net/pages/chordify-toolbox/), [EducationalAppStore](https://www.educationalappstore.com/app/chordify-chords-for-any-song) |
| **MIDI** (Premium) | Contains only the on-screen chords: no melody, no drums. Comes in two versions: a DAW version that follows the song's tempo (for Cubase, Ableton or Logic) with a "bassline track", and a fixed-tempo version for notation software (Finale, Sibelius, MuseScore). The tempo is written into the filename, and user edits are included. | [O] [support](https://support.chordify.net/hc/en-us/articles/360020908358-Do-the-MIDI-files-contain-the-full-song) |
| **PDF / print** (Premium) | A sheet of chord names that reflects the current capo and transpose. Web print adds a page of diagrams. **No lyrics, tab or melody**, for licensing reasons. Users report that piano diagrams don't print properly. | [O] [post](https://support.chordify.net/hc/en-us/community/posts/15169243894685-Download-Chord-Charts-With-Lyrics), [post](https://support.chordify.net/hc/en-us/community/posts/360015181398-No-Piano-Chord-diagram-print-out) |
| **Offline** (Premium, apps only) | Enabled per song with "Add to Offline Mode". Includes diagrams, metronome and chord audio, but **not the original audio** (except the user's own uploads). The app switches to offline automatically. | [O] [support](https://support.chordify.net/hc/en-us/articles/360020907938-How-can-I-use-Chordify-offline) |
| **Setlists** (Premium) | Unlimited setlists of any length, usable offline. Songs can be renamed (title and artist). | [O] [support](https://support.chordify.net/hc/en-us/articles/360007931058-Setlists), [6 hidden features](https://chordify.net/pages/6-hidden-chordify-features/) |
| **Uploads** (Premium) | MP3, MP4 or OGG (an older post also lists M4A), **≤20 min and ≤32 MB**. Link sources are YouTube, SoundCloud and Deezer, with videos up to 25 min. | [O] [desktop help](https://support.chordify.net/hc/en-us/articles/360002148497-How-to-use-Chordify-on-Desktop) |
| **Edit** (web only) | Press "Edit". Click a chord change to delete it. Click a beat square, then the pencil, and type a chord; `~` or `n` gives a rest. A **Meter** button switches between 3/4 and 4/4 only. **"Shift chords"** arrows move the bar lines. Edits carry into MIDI and PDF. An "Edits" panel shows the original detection and every edit. The edit vocabulary is larger than what the detector outputs: 7ths, extensions and a **chosen bass note**. Edits are used to train the algorithm. | [O] [edit help](https://support.chordify.net/hc/en-us/articles/360002148717-How-to-edit-chords), [blog](https://chordify.net/pages/make-chords-more-specific/), [blog](https://chordify.net/pages/chordify-algorithm-learns/) |
| **Views** | Chord Diagrams, in *Animated* or *Summary* mode. Chords Grid (names only). **Chords & Lyrics**, launched 22 Apr 2026: lyrics scroll with chords above them, text size is adjustable, and hovering shows the fingering. iOS 9.0.0 says 7,500+ songs have it, plus pinnable diagrams and karaoke lyrics. The apps show names only; long-press for the diagram. | [O] [blog](https://chordify.net/pages/see-your-song-in-a-new-way-introducing-chords-lyrics-view/), [App Store](https://apps.apple.com/us/app/chordify-songs-chords-tuner/id1073624757) |
| **Instruments** | Guitar, ukulele, piano and mandolin. Since April 2023 iPhone piano diagrams use inverted voicings to fit the screen, and support said this will not be reverted. | [O], [GuitarChalk comments](https://www.guitarchalk.com/chordify-review/) |
| **Lyrics** | Licensed through LyricFind since July 2023. Premium Plus has "Generate Lyrics". A Trustpilot review (30 Sep 2026) says the AI lyrics contain errors. | [MusicRadar via Yahoo](https://www.yahoo.com/entertainment/chordify-now-display-song-chords-095001918.html), [Trustpilot](https://www.trustpilot.com/review/chordify.net) |
| **Toolkit** (Premium Plus) | Live chord detection from the microphone. Chromatic tuner with adjustable reference (e.g. A4 = 430 Hz), cents and Hz readout, and alternate tunings (drop D, open, DADGAD). Metronome with optional drumbeat, signatures 4/4, 3/4 and 1/1, and a visual aid. Chord trainer with instant feedback. Learn Songs (Beta, iOS): per-section lessons and strumming patterns. | [O] [Toolkit help](https://support.chordify.net/hc/en-us/articles/11755805765661-How-to-use-the-Toolkit-Premium-Plus-subscription-required) |
| **Other** | Search for songs that use chords you already know (Android 1758+). "Premium demo channels". Latest iOS is 9.3.0 (21 Sep). | [APKMirror](https://www.apkmirror.com/apk/chordify-b-v/chordify/chordify-1762-release/) |

### 1.3 Display and data model: why sub-beat chords are impossible

- **One square per beat.** Dark lines mark the bars. A square carries a label only where the chord changes, and the current beat is drawn darker. [O] [desktop help](https://support.chordify.net/hc/en-us/articles/360002148497-How-to-use-Chordify-on-Desktop)
- **The data is strictly one chord per beat.**
  - A third-party visualizer calls `https://chordify.net/song/data/youtube:<id>?vocabulary=extended_inversions`.
  - It splits the `chords` field on newlines into rows of `beat;chord;startTime;endTime`, and there is also a `meter` field ([server.js](https://raw.githubusercontent.com/nick264/chordify-visualizer/master/server.js), [Visualizer.jsx](https://raw.githubusercontent.com/nick264/chordify-visualizer/master/app/components/Visualizer.jsx)).
  - So a beat can never hold two chords. [INF] The `extended_inversions` vocabulary is probably fed by user edits.
- **Meter:** only 3/4 and 4/4. Users have asked for 6/8, 12/8, 2/4 and changing meters ([thread](https://support.chordify.net/hc/en-us/community/posts/360005503237-Support-for-different-and-changing-time-signatures-other-than-3-4-and-4-4)).
- **Detected vocabulary:** Chordify's blog says the detector picks from **24 chords (12 major, 12 minor)** ([blog](https://chordify.net/pages/make-chords-more-specific/)). 7ths appear rarely. Extensions, sus chords, slash chords and inversions exist only as manual edits.
- **Simplify:** free, and on by default on the web. The mapping is undocumented. [INF] It reduces edited labels to triads.

### 1.4 Chordify's algorithm

**First generation, 2013 to about 2016** ([O-direct] [ISMIR 2015 LBD](https://ismir2015.ismir.net/LBD/LBD42.pdf), [de Haas et al. ISMIR 2012](https://archives.ismir.net/ismir2012/paper/000295.pdf))
1. Sonic Annotator runs NNLS Chroma, which gives **separate bass and treble chroma**. Settings: 44.1 kHz, Hann window 16384, hop 2048.
2. The QM Bar & Beat Tracker finds the beats.
3. **Chroma is averaged between beats.**
4. The dictionary is maj, min and dom7 × 12, plus N (48 chords + N). Bass is a one-hot vector.
5. Candidates are scored by Euclidean distance, and those within 0.9 of the best are kept.
6. Adjacent beats are merged, but **a merge may only start on beat 1 or 3**.
7. Key comes from Krumhansl profiles with dynamic programming: modulation penalty p = 1, minimum key segment 16 beats.
8. The song is split into segments of at most 12 chords, with at most 30 candidate sequences each.
9. The HarmTrace grammar picks the parse with the fewest insertions and deletions.

Result: 0.739 overlap on 217 Beatles, Queen and Zweieck songs (maj/min).

[INF] How that design drops passing chords:
- A chord shorter than a beat is averaged away (step 3).
- A chord on beat 2 or 4 is merged away (step 6).
- A chord that doesn't fit the grammar is deleted (step 9).
- Inversions never appear, because only maj, min and dom7 are output.

**Current generation, about 2017 onward**
- Two networks, a chord net and a beat net, both reading spectrograms ([O] [algorithm blog](https://chordify.net/pages/technology-algorithm-explained/)).
- [DECIBEL](https://arxiv.org/html/2002.09748v1) describes it as a "deep convolutional-recurrent model from the CQT", based on Koops, de Haas, Bransen & Volk 2017 ([arXiv 1706.09552](https://arxiv.org/abs/1706.09552)).
- Training run: 4,238 songs, 4 days, one server ([bootcamp blog](https://chordify.net/pages/algorithm-summer-bootcamp/)).
- The beat tracker was upgraded in 2020, and user edits feed training ([O](https://chordify.net/pages/new-algorithm-chordify/)).
- Production is Haskell calling TensorFlow through FFI ([Serokell, Dec 2023](https://serokell.io/blog/haskell-in-production-chordify)).
- **Benchmark:** 82.0% WCSR on Isophonics Beatles + Queen (maj/min), against 87.3% for the best MIREX system ([TISMIR 2021](https://transactions.ismir.net/articles/81)).
- Public repos ([github.com/chordify](https://github.com/chordify)): [CASD](https://github.com/chordify/CASD) (50 Billboard songs × 4 expert annotators, JAMS, CC BY-NC-SA), HarmTrace-Base, tapcorrect.
- **Experts disagree:** in CASD the annotators overlap **73% on maj/min and 54% on complex labels** ([JNMR 2019](https://pure.uva.nl/ws/files/62264010/Annotator_subjectivity_in_harmony_annotations_of_popular_music.pdf)).
- No Chordify ISMIR papers were found for 2023–2025.

### 1.5 Documented complaints

- **Only triads.** 7ths are rare, and 9ths come out as 7ths ([UU forum](https://forum.ukuleleunderground.com/threads/anyone-use-chordify-to-try-to-figure-out-chords.108219/), [PG Music forum](https://pgmusic.com/forums/ubbthreads.php?Number=672303&an=&ubb=showflat)).
- **Wrong chords and wrong placement.** Single notes are read as chords, and it "recognizes the downbeat and leaves off the rest".
- **Beat drift** on rubato and pauses, and the beat grid **can't be edited**.
- **No re-analysis** ([2020 request](https://support.chordify.net/hc/en-us/community/posts/360012968137-Option-to-Re-Analyze-a-Song)).
- **The edit vocabulary is too small** ([post](https://support.chordify.net/hc/en-us/community/posts/10455580512157-Let-users-post-comments-on-their-chord-improvements)).
- **No public thread specifically about walk-ups or slash chords.** [INF] Given the design above, Chordify structurally cannot show them.

### 1.6 Competitor features worth copying

| App | Feature worth copying | Source |
|---|---|---|
| Moises | Chord levels Easy/Medium/Advanced. Simplified/Capo/Notation settings. Editing. Export to iReal Pro, MusicXML or a share link. Stems plus chords. Timeline that snaps to **subdivisions** (Oct 2025). Count-in of 4 beats free, 16 on Premium. Free pitch shift limited to ±2 semitones. | [blog](https://moises.ai/blog/latest/advanced-chord-detection/), [Chord Finder](https://moises.ai/features/chord-finder/), [Oct 2025](https://moises.ai/blog/latest/improvements-latest-releases/) |
| Chord ai (iOS 6.2.5) | Free: maj, min, aug, dim, 7, maj7, sus2, sus4. Pro: ø, dim7, 6/9/11/13, add9/11/13, alterations, **slash bass**, **voicing recognition**, smart capo. Also: bars, on-device lyrics in about 100 languages, loop, speed, pitch, 4 stems, audio-to-MIDI, PDF with bars per line, piano diagrams. $10.99/mo or $69.99/yr. | [App Store](https://apps.apple.com/us/app/chord-ai-play-any-song/id1446177109) |
| Yamaha Chord Tracker | When editing, **pick from 2 suggested chords** or choose root + type, with preview playback. Melody suppressor. | [Yamaha](https://usa.yamaha.com/products/musical_instruments/pianos/apps/chord_tracker/features.html) |
| ChordU | Simple / Advanced / **Bass** versions. Notes shown in treble and bass clef. PDF and MIDI in any key. $39.99/yr. | [features](https://chordu.com/features) |
| Capo | Detects **inversions**. Chord chooser sorted by likelihood. Free-text entry. Replace all. Add a chord at the playhead with re-detection. **Editable beat grid** with mixed meters. "Simple Piano" view **highlights the bass note**. | [help](https://supermegaultragroovy.com/products/capo/help/ios/4.5/automatic-music-analysis/), [4.0](https://supermegaultragroovy.com/products/capo/whats-new/mac/4.0/) |
| Melodyne 5 | **Expanded chords** toggle and **Slash Chords** toggle. Alternative names cycled with the arrow keys (C6 vs Am7). Re-analyse a selection. Lead-sheet MIDI. | [Celemony](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_ChordTrackAndDetection) |
| Logic Chord ID | Reads harmonic rhythm from audio. | [Apple](https://support.apple.com/guide/logicpro-ipad/lpipc614f032/ipados) |
| Songsterr Plus | Speed, loop over measures, count-in, AI tabs. | [App Store](https://apps.apple.com/us/app/songsterr-tabs-chords/id399211291) |
| ChordMiniApp (MIT, 409★) | Beat-Transformer and madmom beats; Chord-CNN-LSTM and BTC; 301 labels; beat-chord grid; LLM Roman numerals; LRClib lyric sync. A reference architecture. | [GitHub](https://github.com/ptnghia-j/ChordMiniApp), [DEV](https://dev.to/ngha_phantrng_ccdda2bf/open-source-chord-beat-detection-application-39f5) |
| MyChord | 7ths and slash chords (E/G#). | App Store snippet |
| Hooktheory / Nashville | Roman numerals and scale degrees. For worship, the [Nashville Number System](https://en.wikipedia.org/wiki/Nashville_Number_System). | [Hooktheory](https://www.hooktheory.com/blog/chordify-alternatives/) |

**Unknown or unverified:**
- Chordify's transpose and capo ranges.
- The Simplify mapping.
- The 2026 free-tier daily limit.
- Whether the neural model outputs 7ths, sus chords or inversions without user edits.

---

## 2. Chord-recognition models we can run locally

### 2.1 Comparison table

| Model | License: code / weights | Weights (verified present) | Vocabulary | Bass / inversions | Frame hop | Published scores | Py 3.12 / Windows status | Role in our design |
|---|---|---|---|---|---|---|---|---|
| **Jiang et al. 2019 → `lv-chordia` 1.1.0** (PyPI 11 Jul 2026; repo pushed 10 Oct 2026) | MIT / MIT | 5 ensemble models of ~5.7 MB each (`joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{0..4}.best.sdict`), bundled (~28 MB) | `submission`: 25 qualities × 12 + N = **301** (incl. maj/3, maj/5, min/b3, min/5, maj/2, maj/b7, min/2, min/b7, maj7, 7, min7, dim7, hdim7, maj9, 9, min9, 11, 13, sus2, sus4, sus4(b7)). `full`: **382 qualities** (maj(9), maj6(9), 7(b9), 7(#9), maj/4, 7/3, 7/5, 7/b7, maj7/3, maj7/5, min7/b7, min11, maj(9)/3, sus4(b7,9)); README calls it "untested". | **Yes, explicit bass head** | 512 @ 22.05 kHz ≈ **23 ms** | ChordFormer re-run on 1217 songs: Root 83.39, Thirds 80.04, MajMin 82.62, Triads 75.91, Sevenths 69.78, Tetrads 62.87, MIREX 81.52. Paper (301 classes): frame acc 0.772, class-wise 0.348; reweighted (0.5, 10): 0.746 / 0.402. | Clean dry-run (torch 2.14.1, numpy 2.5.4). Needs Python ≥3.10 and `torch>=2.13`. 10–20× real time on GPU, 2–5× on CPU. | **Primary** |
| **consonance-ACE** (ISMIR 2025) | MIT / MIT (in repo) | `ACE/checkpoints/conformer_decomposed_smooth.ckpt`, 57.6 MB | Open vocabulary: root(13) + bass(13) + 12 note activations, labels rebuilt with harte-library | **Yes** | 2048 ≈ 93 ms | RWC-Pop + USPop: Root 84.0, MajMin 77.8, Thirds 80.3, Triads 72.7, Tetrads 60.8, Sevenths 66.0, MIREX 79.8. Inversion scores vs BTC: MajMin-inv 75.6 vs 71.5, Sevenths-inv 64.4 vs 60.8, Tetrads-inv 59.4 vs 56.2. | Pins Python 3.11, torch 2.6.0, lightning 2.5.1, librosa 0.11.0, beat-this 0.1. Relaxing the pins for 3.12 is untested. | **Second voter**: root, bass and note heads |
| **ChordMini BTC-CL / 2E1D** (DAFx26, 2026) | MIT / repo (training-data caveats) | `checkpoints/btc_model_best.pth` 35.9 MB, `2e1d_model_best.pth` 27.5 MB | 170 classes | No | ~93 ms | BTC-CL: Root 83.03, Thirds 80.17, Triads 78.33, 7ths 69.38, Tetrads 67.00, MajMin 80.24, MIREX 80.16. Rare chords: maj7 28.2→42.3, dim7 0→45.6, sus 22.9→25.5. Caveat: the teacher may overlap the test set. | Not tested | **Third voter** (quality only) |
| BTC (ISMIR 2019) | MIT / possibly CC BY-NC-SA ([cstr GGUF](https://huggingface.co/cstr/btc-chords-GGUF)) | `test/btc_model_large_voca.pt` ~12.2 MB; a NumPy 2 port at [puar-playground/btc-chord](https://huggingface.co/puar-playground/btc-chord) | 170 = 12 × {maj, min, dim, aug, min6, maj6, min7, minmaj7, maj7, 7, dim7, hdim7, sus2, sus4} + X + N | No; no 9ths either | ~93 ms | Root 83.5, Thirds 80.8, Triads 75.9, Sevenths 71.8, Tetrads 65.5, MajMin 82.3, MIREX 80.8 | Code stale (last push May 2020) | Superseded by BTC-CL |
| Chordino / NNLS | GPL | Vamp Plugin Pack 2.0 x64 + sonic-annotator 1.7 win64 (28 May 2025) | Editable `chord.dict` (maj, min, m7b5, 6, 7, maj7, m7, m6, dim, aug, /3, /5, /2, /b7, 7/3) | Yes (bass chroma) | — | Lowest of the systems compared in Jiang Fig. 4 | PyPI `vamp` has only 2015 py2.7 wheels; `chord-extractor` 0.1.3 requires Python <3.12 | **Not used** |
| madmom CNN+CRF | BSD / CC BY-NC-SA | Included | 25 (maj/min) at 10 fps | No | 100 ms | — | PyPI 0.16.1 is source-only (2018). git `27f032e` supports 3.9–3.12 (Ubuntu CI only). | Not used |
| omnizart chord | — | — | 25 | No | — | — | TensorFlow; 0.6.3 source-only; depends on madmom | Not used |
| autochord | Apache-2.0 | — | 25 | No | — | 67.33% | TensorFlow; needs NNLS Vamp | Not used |
| crema | BSD-2 | — | Inversions via a bass output | Yes | — | — | Keras 2 / TF, last release 2022; risky on 3.12 | Not used |
| Sheet Sage | — / CC BY-NC-SA | — | — | — | — | — | Docker on Linux; Jukebox needs 12 GB VRAM | Not used |
| allin1 | — | — | **No chords** | — | — | — | Needs NATTEN (no Py 3.12 Windows build) | Not used |

Sources: [Jiang 2019 PDF](https://archives.ismir.net/ismir2019/paper/000078.pdf) · [music-x-lab repo](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition) · [lv-chordia](https://github.com/openmirlab/lv-chordia) · [ChordFormer](https://arxiv.org/html/2502.11840v1) · [consonance-ACE paper](https://arxiv.org/html/2509.01588v1) · [consonance-ACE repo](https://github.com/andreamust/consonance-ACE) · [ChordMini paper](https://arxiv.org/abs/2602.19778) · [ChordMini repo](https://github.com/ptnghia-j/ChordMini) · [BTC repo](https://github.com/jayg996/BTC-ISMIR19) · [BTC paper](https://ar5iv.labs.arxiv.org/html/1907.02698) · [nnls-chroma](https://github.com/c4dm/nnls-chroma) · [Vamp downloads](https://www.vamp-plugins.org/download.html) · [sonic-annotator](https://github.com/sonic-visualiser/sonic-annotator/releases) · [madmom](https://github.com/CPJKU/madmom) · [omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) · [autochord](https://github.com/cjbayron/autochord) · [crema](https://github.com/bmcfee/crema) · [Sheet Sage](https://github.com/chrisdonahue/sheetsage) · [allin1](https://github.com/mir-aidj/all-in-one)

**Model details that matter for the design**
- **Jiang 2019 / lv-chordia**
  - Separate heads for root+triad, **bass**, 7th, 9th, 11th and 13th. A CRF then assembles the label from a dictionary.
  - The CRF switching penalty is **d = 30**. This is the knob that suppresses short chords.
  - Confusions: maj9/9/min9 collapse into their 7th chords, and 11/13 are rarely detected.
  - Gospel IV/5 is `maj/2`, and 9sus4 is `sus4(b7,9)`; both can be expressed.
- **consonance-ACE**
  - Command: `python -m ACE.inference --audio x.wav --out x.lab --chord-min-duration 0.5 [--beat-sync]`. The default minimum duration (0.5 s) hides passing chords.
  - It outputs exactly what fusion needs: bass and pitch-class activations.

### 2.2 Note transcription, symbolic recognition and separation

**Transcription**

| Component | Facts | Status |
|---|---|---|
| ByteDance piano transcription `piano-transcription-inference` 0.0.6 (Jan 2025, MIT) | Checkpoint `CRNN_note_F1=0.9677_pedal_F1=0.9186.pth`, 172 MB, CC BY 4.0, [Zenodo 4034264](https://zenodo.org/records/4034264). MAESTRO note F1 96.77. Trained on solo piano, so separated stems with bleed will produce false notes. | Clean dry-run. Unverified: `torch.load` with `weights_only=True` (torch ≥2.6), and torchlibrosa 0.1.0 with librosa 1.0. ([repo](https://github.com/qiuqiangkong/piano_transcription_inference)) |
| Basic Pitch 0.4.0 (Apache-2.0) | Instrument-agnostic; ONNX model `nmp.onnx` ships in the package. | **A normal install fails on Py 3.12 / Windows**: it requires `tensorflow<2.15.1`, which has only cp39–cp311 wheels. Workaround: `pip install basic-pitch==0.4.0 --no-deps` plus resampy, scikit-learn and onnxruntime. ([PyPI](https://pypi.org/project/basic-pitch/)) |
| Transkun 2.0.1 (ISMIR 2024, MIT) | Semi-CRF piano transcription: `transkun in.mp3 out.mid --device cuda`. | Depends on `ncls`, which has no Windows wheel, so it would need compiling. A wheel-only dry run failed. |
| PESTO 2.0.1 / torchcrepe 0.0.24 | Monophonic f0 for the bass. | Not tested. |
| MuScriptor 0.3.0 (Kyutai/Mirelo, Aug 2026) | Multi-instrument (36 groups), 103M / 307M / 1.4B parameters. Code MIT; **weights CC BY-NC and gated**. Py 3.10–3.12; on Windows needs `--torch-backend=cu128`. | Optional, personal use only. ([repo](https://github.com/muscriptor/muscriptor)) |
| YourMT3+ | Code GPL-3.0, heavy. | Untested on Windows. |
| Aria-AMT | Robust piano model, Python 3.11, installed from source. | — |
| MT3, Onsets & Frames | Old JAX / TF1 stacks. | Skip. |

**Symbolic chord recognizer**
- **BACHI** (ICASSP 2026, MIT; [arXiv 2510.06528](https://arxiv.org/abs/2510.06528), [repo](https://github.com/AndyWeasley2004/BACHI_Chord_Recognition))
- Weights: HF `Itsuki-music/BACHI_Chord_Recognition/pop909_film_kdec/best_model.pt` (125 MB, 31.2M parameters).
- Input: a beat-synchronous piano roll (88 keys × 12 frames per beat). Predicts root, quality and **bass**.
- POP909-CL scores: Root 89.85, MajMin 89.87, **MajMinBass 88.52**, Sevenths 86.04, SeventhsBass 84.97.
- POP909 is Chinese pop piano arrangements ([arXiv 2008.07142](https://arxiv.org/pdf/2008.07142)).
- Not measured on transcribed (noisy) MIDI.
- `vocab.pkl` is a pickle; inspect it with `pickletools` before loading.

**Separation**

| Model | Facts |
|---|---|
| `demucs` 4.1.0 (2026-07-11, maintained fork adefossez/demucs; facebookresearch archived 2025-01-01) | `htdemucs_ft`: 4 stems (vocals, drums, bass, other). Inference no longer uses torchaudio; decoding goes through `sphn` with an ffmpeg fallback; weights are safetensors from the HF hub. Requires Python ≥3.10 and torch ≥2.1. ([PyPI](https://pypi.org/pypi/demucs/json), [release notes](https://github.com/adefossez/demucs/blob/main/docs/release.md)) |
| `htdemucs_6s` | The upstream README warns the piano stem has "a lot of bleeding and artifacts". |
| **BS-RoFormer-SW** (6 stems) | MVSep SDR: piano 7.83, bass 14.62, guitar 9.05, other 8.71 dB ([MVSep](https://mvsep.com/news/64?lang=en)). Checkpoint `BS-Rofo-SW-Fixed.ckpt` (~700 MB) via [bs-roformer-infer](https://github.com/openmirlab/bs-roformer-infer), also listed in audio-separator 0.47.0 as `BS-Roformer-SW.ckpt`. **License unknown, so personal use only.** |
| `audio-separator` 0.47.0 (2026-08-27) | Roformer, MDX and Demucs models, ensembling, `--list_models`. Requires torch ≥2.3, numpy ≥2, `rotary-embedding-torch<0.7`, `beartype<0.19`, `samplerate==0.1.0`. The `[gpu]` extra pulls in `onnxruntime-gpu`. ([GitHub](https://github.com/nomadkaraoke/python-audio-separator)) |

**Evidence that these components help**
- Separation: Mitoma & Furuya (APSIPA 2025) boosted the pitched stems before BTC and corrected more than 2,000 misrecognised frames against a 75.52% triads baseline ([PDF](https://webdev.apsipa.org/proceedings/2025/papers/APSIPA2025_P307.pdf)). HPSS preprocessing helped in older work ([IPSJ](https://ipsj.ixsq.nii.ac.jp/records/146329/export/bibtex)).
- Symbolic and tab sources: DECIBEL improved every audio system it was tested on by 0.5–13.6 points ([DECIBEL](https://research-portal.uu.nl/en/publications/improving-audio-chord-estimation-by-alignment-and-integration-of-/)).

### 2.3 Beats, key and alignment

| Component | Facts |
|---|---|
| **Beat This!** `beat-this` 1.1.0 (2026-04-14; ISMIR 2024) | MIT code and weights. Provides `File2Beats` and `Audio2Frames`, with checkpoints `final0` and `small0`. Needs no DBN and makes no tempo or meter assumptions. Imports torchaudio for `MelSpectrogram`; `load_audio` falls back to soundfile. ([GitHub](https://github.com/CPJKU/beat_this), [arXiv 2407.21658](https://arxiv.org/abs/2407.21658), [preprocessing.py](https://raw.githubusercontent.com/CPJKU/beat_this/main/beat_this/preprocessing.py)) |
| **S-KEY** (Deezer, ICASSP 2025) | MIT, pretrained `skey.pt`. 73.2% on FMAKv2, against 73.1% for madmom. ([GitHub](https://github.com/deezer/skey), [arXiv 2501.12907](https://arxiv.org/abs/2501.12907)) |
| ChordSync (SMC 2024, MIT) | Aligns an existing chord sheet to the audio. ([GitHub](https://github.com/andreamust/ChordSync)) |
| Chordonomicon | 666k symbolic progressions, usable for transition priors ([arXiv 2410.22046](https://arxiv.org/abs/2410.22046)). |

### 2.4 Watch list (papers without weights) and foundation models

- **ChordFormer** ([arXiv 2502.11840](https://arxiv.org/html/2502.11840v1)): Root 84.69, MajMin 84.09, Sevenths 72.28, MIREX 83.62. No code.
- **Kim & Park, ICASSP 2026** ([arXiv 2604.24386](https://arxiv.org/abs/2604.24386)): event-based modelling of non-triad chords. No code.
- **Mackenzie 2025** ([arXiv 2512.22621](https://arxiv.org/abs/2512.22621)): generative-model features did not help, and classifiers are poor on rare chords.
- **Majchrzak & Mańdziuk 2025** ([arXiv 2508.05878](https://arxiv.org/abs/2508.05878)): trained on synthetic audio. No weights.
- **ChordCoT** ([arXiv 2509.18700](https://arxiv.org/abs/2509.18700)): GPT-4o refinement step, +1 to +2.77 MIREX. No code.
- **MERT-v2** ([HF](https://huggingface.co/m-a-p/MERT-v2-30s)): ~0.6B parameters, CC BY-NC, Chords1217 probe 78.48. **MuQ** ([HF](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter)): ~300M parameters, CC BY-NC. Neither has a released chord head, so they are not worth using now.

### 2.5 Runtime estimates for a 4-minute song (est., not benchmarked)

| Step | Time |
|---|---|
| Download + decode | 5–15 s |
| Beat This! | 2–5 s |
| lv-chordia | 10–25 s |
| consonance-ACE | 5–10 s |
| BTC-CL | under 5 s |
| Chordino | about 10 s on CPU |
| BS-RoFormer-SW | 40–90 s |
| Piano transcription | 10–20 s |
| Bass transcription | about 5 s |
| BACHI + fusion | under 5 s |

- Full pipeline: **about 1.5–2.5 min**.
- Fast mode without separation: **30–45 s**.
- Peak VRAM below about 5 GB if models are loaded one at a time (est.).

### 2.6 Licences and safety

- **Non-commercial weights:** madmom models, MuScriptor, MERT, MuQ and Sheet Sage. BTC weights are reportedly NC as well. The BS-RoFormer-SW licence is unknown.
- **Pickled checkpoints:** load `.pkl`, `.sdict`, `.pth` and `.ckpt` files with `torch.load(weights_only=True)` where possible, and only from the repos listed here.
- **Untrusted source excluded:** github.com/K1ller-Whale/Chordify/pull/2 contained instructions aimed at AI assistants. It was not used, and its benchmark claims are excluded.

---

## 3. Passing chords: why detectors miss them, and what to do instead

### 3.1 Eight causes

1. **A fixed cost per chord change, against evidence that grows with duration.**
   - Jiang 2019 decodes with a CRF transition term exp(−d·I[C(t−1)≠C(t)]), with d = 30 and 23 ms frames ([PDF](http://archives.ismir.net/ismir2019/paper/000078.pdf)).
   - A half-beat passing chord at 120 bpm is 0.25 s, about 11 frames. Between two different chords it adds one extra change, costing 30 nats: about 2.8 nats per frame, or a likelihood ratio of about 16×.
   - Inside one held chord (A→P→A) it adds two changes, costing 60 nats: about 5.6 nats per frame, or about 260×.
   - Deng & Kwok use self-transition weights "99.99 times" the switch weight ([arXiv 1709.07153](https://arxiv.org/abs/1709.07153)).
   - FMP calls HMM transitions "context-aware postfiltering" ([FMP](https://audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_ChordRec_HMM.html)).
2. **One-state HMMs imply a geometric duration distribution.** Negative-binomial and RNN duration models fit real durations better, and a GRU duration model raised maj/min WCSR from 79.5 to 80.5 ([arXiv 1808.05335](https://arxiv.org/abs/1808.05335)). Complex frame models mostly just smooth ([arXiv 1702.00178](https://arxiv.org/abs/1702.00178)).
3. **Median filtering** over L frames deletes any run shorter than about L/2. This follows from how the filter works.
4. **Beat-synchronous averaging, or one vote per beat,** can't represent a chord on the "and" of a beat. Chordify's pipeline does this ([UU portal](https://research-portal.uu.nl/en/publications/chordify-chord-transcription-for-the-masses/), [LBD42](https://www.ismir2015.uma.es/LBD/LBD42.pdf)).
5. **Time-weighted metrics and imbalanced training data.**
   - WCSR weights by duration, so tuning for it rewards heavy smoothing.
   - In Billboard, the top 10% of qualities cover 93.86% of the data and the bottom 50% cover 0.35%.
   - Deng & Kwok report maj/min at 69.9%, sevenths at 21.3% and everything else at 8.8%.
   - Rare classes (dim7, slash, sus) do badly ([Mackenzie 2025](https://arxiv.org/abs/2512.22621)).
6. **Bass and inversions.**
   - Mauch & Dixon's DBN recognised first inversions only 15% of the time and second inversions 29%.
   - Meter and bass modelling significantly improved segmentation ([TASLP 2010](https://webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2010/Mauch-Dixon-TASLP-2010-real.pdf)).
   - BTC has no inversions at all.
7. **Masking.**
   - Vocals add non-chord pitch classes.
   - Drum transients smear chroma on beats and off-beats.
   - Pads, sustain pedal and reverb carry the previous chord into the passing chord.
   - A pedal-point bass pulls the root estimate.
8. **Annotations leave them out.** CASD agreement is 73% on maj/min and 54% on complex labels ([JNMR 2019](https://pure.uva.nl/ws/files/62264010/Annotator_subjectivity_in_harmony_annotations_of_popular_music.pdf)), so models learn to omit passing chords.

### 3.2 Worship and gospel idioms to use as priors

Written as Nashville / 級數 degrees in a major key.

| Idiom | Pattern | Typical length of the short chord |
|---|---|---|
| Pop / axis | 1 5 6m 4 | — |
| Descending bass (Canon) | 1 – 5/7 – 6m – 3m/5 (or 6m/5) – 4 – 1/3 – 2m – 5 | 0.5–2 beats |
| Walk-up | 1 – 2m – 1/3 – 4; 4 – 5 – 6m; 5/7 – 1 | 0.5–1 beat |
| ii–V–I / circle | 2m7 – 5(7) – 1; 3m7 – 6m7 – 2m7 – 5 – 1 | 1–2 beats |
| Secondary dominants | 3(7)→6m; 6(7)→2m; 1(7)→4; 2(7)→5 | 0.5–2 beats |
| Passing diminished | 1 – #1°7 – 2m; 2m – #2°7 – 1/3; 4 – #4°7 – 1/5; 5 – #5°7 – 6m | **0.5–1 beat** |
| Half-diminished approach | 7ø → 1 | 0.5–1 beat |
| Four over five | 4/5 → 1 | 1–2 beats |
| Sus resolution | 5sus4 → 5; 1sus4 → 1 | 0.5–1 beat |
| Borrowed iv | 4 – 4m – 1 | 1–2 beats |
| Royal road (丸サ) | 4M7 – 5(7) (or 5/4) – 3m7 – 6m; 4M7 – 3(7) – 6m7 | 1–2 beats |
| Pedal neighbour | 1 – 4/1 – 1 | 0.5–1 beat |

Sources:
- Canon bass walk: [zh.wikipedia 和弦進行](https://zh.wikipedia.com/wiki/%E5%92%8C%E5%BC%A6%E9%80%B2%E8%A1%8C)
- IV/5 ≈ V7: [Puget Sound](https://musictheory.pugetsound.edu/mt21c/IVover5SusChord.html)
- Passing ø7/°7 a half step below the target: [hearandplay](https://hearandplay.com/main/two-important-passing-chords-to-major-and-minor-chords/)
- Gospel walk-down: [PianoGroove](https://www.pianogroove.com/blues-piano-lessons/the-gospel-walk-down)
- Royal road: [Wikipedia](https://en.wikipedia.org/wiki/Royal_road_progression)

**No published statistics** exist for 讚美之泉 or 約書亞 progressions. Learn the weights from Nathan's own annotated songs.

The test song (§5) also uses idioms that are missing from this table: I → I+7 (augmented passing), IVmaj7 → ♭VII7 → iv6 (borrowed), ii7 → iii7 rising, and vi7 → vii7 (G#m7 → A#m7) rising.

### 3.3 The proposed algorithm, in summary

The design ([DESIGN.md §5](DESIGN.md)) takes this over with firm parameter values. The central idea: limit flicker by **where** a change can happen (a cost that depends on metric position, on a sub-beat grid) and by musical plausibility (key-aware priors), not by a large uniform penalty on every change.

The stages:
1. **Separation** into stems. H = bass + other is the harmonic mix.
2. **Beat This!** for beats and downbeats.
3. **A sub-beat grid**: subdivide by 2, by 4 when the beat is slow, or by 3 for compound meter.
4. **Key**: global key plus a 24-state local-key HMM with self-transition 0.995 per beat.
5. **Acoustic evidence** from lv-chordia, consonance-ACE and BTC, on both the full mix and H.
6. **Symbolic evidence**: recency-weighted chroma from transcribed notes (τ = 0.4 s, onset bonus β = 0.5, novelty α = 0.7, Bernoulli template p_in 0.85 / p_out 0.10).
7. **Bass evidence** from the first 60% of each cell. Mauch: the bass is on the chord bass about 80% of the time on its first beat.
8. **Combine:** e = κ·Δg·[ac + w_sym·sym + w_bass·bass], with κ = 6 effective frames per second, w_sym = 1.0, w_bass = 0.8.
9. **Semi-Markov (HSMM) decode** with position-dependent change costs λ:
   - downbeat 2.0, mid-bar 2.5, beat 3.0, half-beat 4.5, quarter-beat 6.0, off-grid 7.0;
   - a degree-bigram prior: +1.0 for diatonic moves, +1.5 for stepwise slash moves, +1.5 for V/x→x, +2.0 for sus4→X, +1.5 for IV/5→I, −1.0 for other non-diatonic targets, −1.5 for unexplained semitone slides;
   - a bass-step bonus of +0.8;
   - a log-normal duration prior (median 2 beats, σ 0.9), with +1.0 for passing-type chords of ≤1 beat.
10. **Two layers.** A main layer decoded on the beat grid, and a detail layer decoded on the sub-beat grid. Passing-chord salience is the score lost by merging the chord away, plus +1.5 for an idiom trigram.
11. **Slider:** show a passing chord if conf ≥ 0.9 − 0.6·s.
12. **Slash and spelling rules:** a bass held ≥½ beat makes a slash chord; shorter is a bass passing tone. Spelling uses the local key.

Compute: about 1,000 cells × 32 durations × 25² chords ≈ 20M operations per song.

**Phase-2 training ideas:**
- Synthetic worship renders.
- Pitch-shift augmentation.
- Teacher pseudo-labels ([Phan et al. 2026](https://arxiv.org/abs/2602.19778)).
- Class reweighting with γ = 0.5, w_max = 10.

### 3.4 How to evaluate passing chords

- **Standard metrics:** `mir_eval` 0.8.2, `mir_eval.chord.evaluate(ref_int, ref_lab, est_int, est_lab)`, reporting root, majmin, **majmin_inv**, mirex, thirds(_inv), triads(_inv), sevenths(_inv), **tetrads_inv**, seg, overseg and underseg ([docs](https://mir-eval.readthedocs.io/stable/api/chord.html)).
  - The *_inv metrics require the bass to be a chord tone. Out-of-chord slash chords such as F/G need a custom check.
- **Short-segment P/R/F1:**
  - A reference segment counts as short if it lasts ≤1 beat (and separately ≤0.6 s).
  - It is a hit if an estimated segment overlaps ≥50% of it, the label matches, and the onset lies within ±max(70 ms, ¼ beat).
- **Accuracy by duration bin:** (0, ½], (½, 1], (1, 2], (2, 4] and >4 beats.
- **Boundary F-measure:** `mir_eval.onset.f_measure(..., window=0.1)`, plus a version with a ¼-beat window.
- **Flicker rate:** unmatched estimated segments shorter than ½ beat, per minute. Target <2/min.
- **Recall per idiom category**, and bass pitch-class accuracy.
- **Data:**
  - Synthetic §3.2 progressions in 12 keys at 60–140 bpm, with a sustaining pad, drums and reverb.
  - 10–20 of Nathan's worship songs annotated at sub-beat resolution.
  - Isophonics, Billboard and RWC as a regression set.
- **Ablations against single models:** separation, sub-beat grid, bass term, symbolic chroma, priors, HSMM vs HMM, two layers.
- **Suggested targets (the researcher's own, not from a paper):** short-segment F1 ≥0.6 at precision ≥0.7, with majmin_inv WCSR no more than 1 point below the best baseline.

---

## 4. Software stack: findings

### 4.1 Machine (local, read-only checks)

| Item | Finding |
|---|---|
| Software present | Python 3.12.4 (the only Python), pip 26.1.2, Node v24.14.1, git 2.50.0 |
| C compiler | VS Community 2022 17.12 with **MSVC 14.42.34433**, so C extensions can be built |
| Edge WebView2 | 154.0.4258.62 installed. The Windows UI language appears to be **zh-TW**. |
| GPU | RTX 4070 Laptop 8 GB, driver 591.86, **maximum CUDA 13.1** |
| Not installed | ffmpeg, yt-dlp, deno, tailscale, mkcert, cloudflared, Rust |
| Cleanup needed | A leftover `~ip*` folder in global site-packages. It only causes a warning; delete it. |

### 4.2 What installed and what failed

**Nothing was installed.** All results below come from index metadata or `pip --dry-run`.

| Package | Result |
|---|---|
| lv-chordia 1.1.0 | ✅ resolves (torch 2.14.1, numpy 2.5.4) |
| piano-transcription-inference 0.0.6 | ✅ resolves |
| torch 2.14.1+cu130, torchaudio 2.11.0+cu130 (cp312 win_amd64) | ✅ wheels exist on the cu130 index |
| demucs 4.1.0, beat-this 1.1.0, audio-separator 0.47.0, mir_eval 0.8.2, pretty_midi 0.2.11.post0, soundfile 0.14.0, soxr 1.1.0, sphn 0.2.1, onnxruntime-gpu 1.31.0 | ✅ wheels available |
| basic-pitch 0.4.0 | ❌ needs tensorflow<2.15.1, which has no cp312 build. Use the `--no-deps` workaround. |
| chord-extractor 0.1.3 | ❌ requires Python <3.12 ("No matching distribution") |
| vamp 1.1.0 | ❌ only cp27 wheels (2015) plus a source tarball |
| madmom | ⚠ PyPI 0.16.1 is source-only (2018). git `27f032e` (2024-08-25, PR #540) supports Python 3.11/3.12 and numpy 2, but CI is Ubuntu-only. Expected to build with MSVC; untested. |
| Transkun | ⚠ wheel-only dry run failed: `ncls` has no Windows wheel |
| omnizart 0.6.3, allin1 | ❌ source-only or madmom/NATTEN dependencies |
| consonance-ACE | ⚠ pins Python 3.11 / torch 2.6.0; relaxing them is untested |

**Open risks to settle in a smoke test:**
- madmom on Windows.
- beat-this with rotary-embedding-torch 0.6.x (forced only if audio-separator is installed).
- piano-transcription-inference and the `weights_only` default.
- Whether yt-dlp needs a PO-token provider for the test video.

### 4.3 YouTube download

- **yt-dlp 2026.08.19** is the latest. It removed `android_vr` from the default clients and added support for curl_cffi 0.16. Release 2026.07.04 raised the recommended minimum Python to 3.11. ([releases](https://github.com/yt-dlp/yt-dlp/releases))
- **A JavaScript runtime has been required since 2025.11.12 (EJS).** Without one, YouTube support is "deprecated" and far fewer formats are available ([GIGAZINE](https://gigazine.net/news/20251113-yt-dlp-required-deno-javascript-runtime/), [EJS wiki](https://github.com/yt-dlp/yt-dlp/wiki/EJS)).
  - **Deno ≥2.3.0** is the default runtime.
  - **Node ≥22** works with `--js-runtimes node` or `js_runtimes={'node': {}}`. Node 24.14.1 qualifies.
  - Bun is deprecated; QuickJS is also supported.
  - The EJS scripts come from `yt-dlp-ejs` 0.8.0, pulled in by `yt-dlp[default]` ([pyproject](https://raw.githubusercontent.com/yt-dlp/yt-dlp/master/pyproject.toml)).
- **Recommended: `deno==2.9.7` from PyPI** (the official binary as a wheel, [deno_pypi](https://github.com/denoland/deno_pypi)). It installs into the venv without admin rights. In Python: `js_runtimes={'deno': {'path': deno.find_deno_bin()}}`.
- **PO tokens and SABR** (guide edited 2026-07-12, [PO-Token-Guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide)):
  - `web` offers only SABR formats, which yt-dlp skips. `tv` needs no token. `mweb` and `web_safari` need GVS tokens, but web_safari's HLS formats currently don't.
  - If you get 403 errors or a bot check:
    1. Upgrade yt-dlp.
    2. Use `--cookies-from-browser firefox`. Chrome cookie extraction fails on Windows because of App-Bound Encryption ([#10927](https://github.com/yt-dlp/yt-dlp/issues/10927)).
    3. Add **bgutil-ytdlp-pot-provider 2.0.2** (2026-10-07; [PyPI](https://pypi.org/project/bgutil-ytdlp-pot-provider/)).
- **The test URL is a YouTube Mix** (`&list=RDY7boUFbM76k&start_radio=1`). Always set `noplaylist: True`.
- **Don't freeze yt-dlp.** Use `yt-dlp[default]>=2026.8.19` and provide an "update downloader" button.

### 4.4 FFmpeg

- **Choice:** Gyan **9.0.2 full_build-shared** (2026-09-20), app-local, from `https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-full_build-shared.zip`.
  - It includes ffprobe, which yt-dlp needs.
  - It includes **librubberband**, so the `rubberband` filter can do server-side pitch shifting.
  - As a shared build, its DLLs also satisfy TorchCodec ([gyan.dev](https://www.gyan.dev/ffmpeg/builds/)).
- **Rejected:**
  - `imageio-ffmpeg` 0.6.0: no ffprobe.
  - `static-ffmpeg` 3.0: downloads binaries from a third-party repo on first use.
  - yt-dlp/FFmpeg-Builds: nightly builds only.
  - winget portable install: can fail to create the Links shim without Developer Mode ([forum](https://forum.restic.net/t/restic-link-not-present-in-0-18-1-release-on-winget/10343)).

### 4.5 PyTorch and torchaudio

- **torch 2.14.1** (2026-09-30) supports Python 3.10–3.15 with CUDA 12.6, 13.0 and 13.2 builds. **CUDA 12.8 has been dropped, and 2.14 is the last release with any CUDA 12.x build.** ([RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md), [dev-discuss](https://dev-discuss.pytorch.org/t/pytorch-2-14-final-rc-available/3429))
- **Choose cu130**, because the driver supports up to CUDA 13.1. cu132 would depend on minor-version compatibility. sm_89 (Ada) is supported.
- **torchaudio 2.11.0 (2026-03-23) is the final release.** It is built on the stable ABI and documented to work with future torch versions. Its `install_requires=[]`.
  - Since 2.9, `load` and `save` wrap **TorchCodec** ([issue #3902](https://github.com/pytorch/audio/issues/3902), [docs](https://docs.pytorch.org/audio/2.11.0/installation.html)).
  - **Don't install TorchCodec** (0.17.0). Decode with the ffmpeg CLI to WAV and read with soundfile.
- **onnxruntime-gpu 1.31.0** (2026-10-09) is built for CUDA 13.0 / cuDNN 9.
  - Import torch first, or call `onnxruntime.preload_dlls()`.
  - Never install both `onnxruntime` and `onnxruntime-gpu` ([ORT](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)).
- **demucs 4.0.1** called `torchaudio.save`, which now fails without TorchCodec ([HF forum](https://discuss.huggingface.co/t/cannot-load-torchcodec/169260)). Use **4.1.0**.

### 4.6 MIR libraries

| Package | Finding |
|---|---|
| librosa 1.0.0 (2026-08-11) | Requires Python ≥3.12 and numpy ≥2.1. **audioread removed** (no m4a or webm). `filters.constant_q*` and `vqt(res_type=None)` removed. ([blog](https://librosa.org/blog/posts/1.0/), [changelog](https://librosa.org/doc/main/changelog.html)) The sota report advises **`librosa==0.11.*`** for older model code. |
| numba 0.68.0 / llvmlite 0.50.0 | Requires numpy <2.6, so pin numpy 2.5.4. |
| mir_eval 0.8.2, pretty_midi 0.2.11.post0 (2026-07-28), mido 1.3.3, soundfile 0.14.0 (bundles libsndfile), soxr 1.1.0, music21 10.5.0 (Py ≥3.11) | Wheels available. |

### 4.7 Backend

- **FastAPI 0.143.0** and **uvicorn[standard] 0.54.0**. uvloop is skipped on Windows.
- **Native SSE** with `fastapi.sse.EventSourceResponse` (since 0.135.0), with a keep-alive ping every 15 s ([docs](https://fastapi.tiangolo.com/tutorial/server-sent-events/)).
- **No Celery or RQ** (poor fit on Windows; RQ relies on `fork()`). Use a single-slot GPU worker process with the `spawn` start method, and pass progress through an mp.Queue, a bridge thread, `call_soon_threadsafe` and SSE.
- **Storage:** SQLite. Cache keys are `yt-{video_id}` (canonical ID from `extract_info(download=False)`) and `sha256-{hash}` for uploads.
  - Results: `analysis.v{ENGINE_VERSION}-{params_hash}.json`.
  - Point `HF_HOME` and `TORCH_HOME` at the project's models folder.

### 4.8 Frontend and playback

- **Versions:** Vite 8.3.4 (Node ^20.19 or ≥22.12), React 19.3.0, TypeScript 7.0.2, @vitejs/plugin-react 6.1.2, tonal 6.5.0 (MIT), zustand 5.0.15, idb-keyval 6.3.0, vite-plugin-pwa 2.0.0, workbox-window 7.4.1 ([npm](https://registry.npmjs.org/vite/latest)).
- **Rendering:** keep the playback clock and highlighting out of React. A rAF loop toggles CSS classes directly.
- **Don't use the YouTube IFrame API for playback.** It has only fixed rates, no audio access, no pitch control, and mobile autoplay is blocked ([IFrame API](https://developers.google.com/youtube/iframe_api_reference)).
- **Playback tiers:**
  - Simple: `<audio>` with `playbackRate` and `preservesPitch` (Baseline since Dec 2023, [MDN](https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/preservesPitch)).
  - Recommended: **signalsmith-stretch 1.3.2** (MIT, WASM + AudioWorklet). `schedule({rate, semitones, loopStart, loopEnd, ...})` gives tempo change, real pitch shift and gapless A-B loops ([README](https://unpkg.com/signalsmith-stretch@1.3.2/README.md)).
- **Alternatives:**
  - SoundTouchJS 0.3.0 / @soundtouchjs/audio-worklet 2.1.1 (LGPL-2.1).
  - rubberband-web 0.2.1 (GPL).
  - Server-side `ffmpeg -af rubberband=pitch=2^(n/12)`.

### 4.9 Desktop

- **pywebview 6.2.1** (pythonnet 3.2.1) on WebView2, which is already installed ([PyPI](https://pypi.org/pypi/pywebview/json)).
- **Rejected:**
  - Tauri: needs Rust plus a sidecar.
  - Electron: adds 100 MB+.
  - PyInstaller: a multi-GB torch bundle that triggers antivirus false positives.
- **Launcher:** `desktop.py` starts uvicorn on `127.0.0.1:8765` and polls `/health`, then opens a `webview.create_window`. A `.bat` file runs `pythonw`, and a `.lnk` shortcut is created with WScript.Shell.

### 4.10 Phone over Wi-Fi (HTTPS is required for a service worker and for installing the PWA)

| Option | Verdict |
|---|---|
| **Tailscale serve** | **Recommended.** `tailscale serve --bg 8765` gives `https://<pc>.<tailnet>.ts.net` with an automatic certificate. It persists, works away from home, and passes SSE and WebSockets through. uvicorn stays on 127.0.0.1, so **no firewall rule is needed**. Requires MagicDNS and HTTPS certificates to be enabled. Machine and tailnet names appear in Certificate Transparency logs, so use neutral names. ([serve](https://tailscale.com/kb/1242/tailscale-serve), [HTTPS](https://tailscale.com/kb/1153/enabling-https)) |
| mkcert v1.4.4 | **Fallback.** `mkcert -install`, then `mkcert -cert-file lan.pem -key-file lan-key.pem 192.168.x.y <pc>.local localhost`, then uvicorn with `--ssl-*`. **Copy only `rootCA.pem` to the phone, never the key.** Android: Settings → Security → Encryption & credentials → Install CA. iOS: install the profile, then General → About → Certificate Trust Settings. Some iOS 18 users report a delay before the trust applies. Reserve the PC's IP in DHCP. ([mkcert](https://github.com/FiloSottile/mkcert), [Apple](https://support.apple.com/en-ca/102390), [forum](https://developer.apple.com/forums/thread/764673?page=5), [moOde](https://moodeaudio.org/forum/showthread.php?pid=60715)) |
| Self-signed certificate | No: warning page, and the PWA won't install. |
| Cloudflare quick tunnel | No: no SSE, the hostname changes every run, a 200-request limit, and it exposes the app to the internet ([docs](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)). |

**Firewall (mkcert path only), run as admin:**
- `Get-NetConnectionProfile`; Wi-Fi must be Private.
- `New-NetFirewallRule -DisplayName "Nathan_chordify HTTPS 8443" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8443 -Profile Private`.

### 4.11 Exports and diagrams

- **MIDI:** pretty_midi, with a beat-map tempo.
- **PDF:** browser print CSS (`@page A4`), which handles Traditional Chinese with no extra work. Optionally reportlab 5.0.1 with a CJK font (`MSung-Light` or Noto Sans TC).
- **ChordPro and text:** string templates.
- **MusicXML:** music21 10.5.0, `harmony.ChordSymbol` → `<harmony>` with `<bass>`.
- **Piano diagrams:** inline SVG over a 2-octave window; white keys about 24×120, black about 14×76. The bass or slash note gets its own colour. Voicings are voice-led. `tonal` handles spelling, but its slash-bass parsing is unverified.

### 4.12 Pinned stack as researched

| Layer | Pins |
|---|---|
| GPU | `torch==2.14.1+cu130` and `torchaudio==2.11.0+cu130` (index https://download.pytorch.org/whl/cu130). No torchcodec. |
| Numerics | numpy 2.5.4, scipy 1.18.1, numba 0.68.0, llvmlite 0.50.0 |
| Audio / MIR | soundfile 0.14.0, soxr 1.1.0, librosa 1.0.0 (stack report) or 0.11.* (sota report), mir_eval 0.8.2, pretty_midi 0.2.11.post0, mido 1.3.3, music21 10.5.0, cython 3.3.0, madmom git 27f032e |
| Separation / beats | demucs 4.1.0 (sphn 0.2.1, lameenc 1.8.4), beat-this 1.1.0, einops 0.8.2. Optional: audio-separator[gpu] 0.47.0 + onnxruntime-gpu 1.31.0. |
| Optional | basic-pitch 0.4.0 (`--no-deps`), piano-transcription-inference 0.0.6, torchlibrosa 0.1.0 |
| Download | yt-dlp[default] ≥2026.8.19 (yt-dlp-ejs 0.8.0), deno 2.9.7, bgutil-ytdlp-pot-provider 2.0.2 only if needed, FFmpeg 9.0.2 Gyan shared |
| Server | fastapi 0.143.0, uvicorn[standard] 0.54.0, python-multipart 0.0.32, pydantic 2.14.1, sqlmodel 0.0.48 (optional) |
| Desktop | pywebview 6.2.1 (pythonnet 3.2.1) |
| Frontend | vite 8.3.4, react / react-dom 19.3.0, @vitejs/plugin-react 6.1.2, typescript 7.0.2, vite-plugin-pwa 2.0.0, workbox-window 7.4.1, tonal 6.5.0, signalsmith-stretch 1.3.2, zustand 5.0.15, idb-keyval 6.3.0 |
| HTTPS | Tailscale serve (primary); mkcert v1.4.4 (fallback) |

---

## 5. Test song: 回家 (YouTube Y7boUFbM76k)

### 5.0 Confirm the intent first

- The link was relayed with "i pressed wrong!".
- The URL's `pp=` parameter shows it came from a YouTube search for "回家 chord", which returns several different songs:
  - 讚美之泉 (Mandarin; [cantonhymn](https://cantonhymn.net/song/回家-讚美之泉/)), with its own Chordify page;
  - 約書亞樂團 "Come Home";
  - Beyond (Cantopop).
- Nathan may have opened the wrong one. Everything below is about **this** video.

### 5.1 Identification

| Field | Value | Source |
|---|---|---|
| Video | "詩歌-回家 [歌詞]", channel Kelly:DD, uploaded 2010-06-01, about 3.27M views, **247 s** | [oEmbed](https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=Y7boUFbM76k&format=json) |
| Song | 回家, a Cantonese gospel song | [christianstudy](https://www.christianstudy.com/data/hymns/text/c2109.html) |
| Singers | Eternity Girls (Ady 梁浩楹, Judy 梁浩菱, Sharon 黃芳, Rubbie 梁卓盈) | [emmhk](https://www.emmhk.org/2020/07/eternity-girls) |
| Credits | Music: 李健達. Lyrics: 李健達 / 張祥志. Arranger: Johnny Yim. Producers: Johnny Yim / 劉港源. | [worshipengine](https://www.worshipengine.ca/songs/UVGFAskhi-g), [mulanci](https://www.mulanci.org/lyric/tl203723/) (snippet) |
| Publisher | 永恆音樂事工 Eternity Music Ministry (HK). Official MV: UVGFAskhi-g. | same |
| Release | First single, released end of Feb 2007. Debut album for Easter 2007. Also on 《精選 - 美麗傳奇》. | [gospelherald](https://www.gospelherald.com/news/mhtbh7jsopks), [zanmei](https://www.zanmei.ai/song/11234.html) |
| Key | **B major**. Audio check: B major 0.868 vs G# minor 0.737. | Polygon, 魚蛋村 sheet, zanmei, audio check |
| Tempo / meter | **About 65.4 BPM, 4/4**; a bar is about 3.67 s; steady eighths. ChordU's 130.2 BPM counts eighth notes. | audio check, [ChordU](https://chordu.com/chords-tabs-%E5%9B%9E%E5%AE%B6-mv-eternity-girls-id_1Bf0btWKE0w) |
| Form | Intro – Verse – Pre-chorus – Chorus – Interlude – Pre-chorus – Chorus – Interlude – Chorus 3 – Tag. No second verse. | Video description and audio |

### 5.2 Sources

| # | Source | Key as written | Matches this recording? | Notes |
|---|---|---|---|---|
| S1 | [polygon.guitars](https://polygon.guitars/tabs/詩歌-eternity-girls-回家) / [2016 blog](http://polygonguitar.blogspot.com/2016/06/eternity-girls-chord.html) | G shapes, labelled "Key: B" | Yes, with capo 4 (+4) | Human chart with bar lines |
| S2 | [cantonhymn](https://cantonhymn.net/song/回家/) | G | Capo 4 | Not independent of S1; adds a coda |
| S3 | [jgospel](https://www.jgospel.net/c/2/31939/279/3986/Eternity-Girls-回家.aspx) | C | One semitone too high (−1) | Numbered notation with chords |
| S4 | [zanmei](https://www.zanmei.ai/song/11234.html) | B | Yes | Unknown author. Has LRC album timestamps, which match this video to within about 0.5 s. |
| S5 | [yudans](https://yudans.net/591/回家-琴譜) | B | Yes | Hugo Wong piano score, 7 pages. Not checked. |
| A1 | [Chordify page 1](https://chordify.net/chords/eternity-songs/hui-jia-chords) | — | — | Blocked. Snippet (2023-04-12): "Chords: E, F#, B, G#m" |
| A2 | [Chordify page 2](https://chordify.net/chords/hui-jia-ge-ci-banmv-eternity-girls-yong-heng-yin-le-shi-gong-eternity-music-ministry) | — | — | Snippet: "E, F#, B, E7" |
| A3 | ChordU (MV 1Bf0btWKE0w) | B | Yes | 130.2 BPM; E, F#, B, G#m, C#m |
| A4 | Guitarians CHORD TELL (video nKIxM2IAUds) | B | Yes | Claims about 80%. Picks up Em, A7, B7→E, Baug, Eb7/G, Eb/Bb, A/C#, B/D#, F#m, Emaj7 |

Ultimate Guitar and Moises have nothing for this song.

### 5.3 Consensus chart

Key B; each `|` is a bar boundary. Times are section starts in this video. (H) means the audio clearly supports it, (M) likely, (L) unsure.

```
[Intro 0:02.9]     | Emaj7 D#7sus4 D#7/G | G#m/D# | Emaj7 D#7sus4 D#7/G | G#m/D# F# | Emaj7 | F#add9 D#m7 | Emaj7 | F#11(E/F#) |   (L–M)
[Verse 1 0:32.4]   | B B7sus4 | G#m7/F# Em/A | B F#/B B7sus4 | G#m7/F# Emaj7 | C#m7 D#m7 | C#m7/E Em/G | F#11 F# |               (M; Em/G L)
[Pre-chorus 0:58.2]| B B+7 | Emaj7 A7 Em6 | B B/F# B7 | Emaj7 A7 Em7 | C#m7 D#m7 | C#m7/E | F#11 F# |                         (M–H)
[Chorus 1 1:23.9, pickup 1:23.5] | G#m F# | E F# B | E F# | E F#/A# | Emaj7 F# | F#sus4 D#7/A# G#m/D# | C#m7 F# |          (M–H)
[Interlude 1 1:49.7] | Emaj9(B/E) B/D# | C#m7 B | Emaj7 B/D# | C#m7 G#m F# F#/C# |                                     (H for bass)
[Pre-chorus 2 2:04.9]| B B/F# | Emaj7 A7 Em | B B+7 | Emaj7 E/D# | C#m7 D#m7 | C#m7/E | F# F#/C# |                        (M–H)
[Chorus 2 2:30.6]  | G#m F# | E F# B | E F# | G#m7 A#m7 F#/A# | Emaj7 F# | B D#7/A# G#m G#m/D# | Emaj7 F#7 |               (M–H)
[Interlude 2 2:56.3] | Emaj9(B/E) F#7 | G#m/D# G#m7/B F#/C# F# | Emaj7 F# | G#m7 A#m7 F# |                              (M)
[Chorus 3 3:11.0]  | Emaj7 F# | B F#/A# G#m7 B/F# | Emaj7 F#/C# F# | Emaj7 F#sus4 F# | Emaj7 F# | B D#7/A# G#m7 | G#m7 G#m7/D# | C#m7 B/F# F# | B |  (H)
[Tag 3:44]         | B B7 | A9 | B/F# | C#m7 | F#11(E/F#) | B |                                                         (H; ritardando)
```

**Roman numerals**
- Pre-chorus: I I+7 | IVmaj7 ♭VII7 iv6 | I I/5 I7 | IVmaj7 ♭VII7 iv7 | ii7 iii7 | ii7/3 | V11 V
- Chorus 3: IV V | I V/3 vi7 I/5 | IV V/5 V | IV Vsus V | IV V | I V7/vi (2nd inversion) vi7 | vi7 vi7/5 | ii7 I/5 V | I

### 5.4 Where the sources disagree (all converted to B)

| Spot | S1/S2 | S3 | S4 | Audio verdict |
|---|---|---|---|---|
| Verse/pre-chorus bar 1 | B F#/A# | B F# | B F# E | B, then B+7 or B7 |
| Bar 2 | G#m Em | G#m B | — | **Emaj7 A7 Em(6)** (bass E–A–E); S1's Em is right |
| Bar 4 | G#m | G#m B | E | Emaj7 A7 Em7 |
| Bar 5 | E D#m | E B | **C#m D#m** | **C#m7 D#m7** (bass rises); S4 is right |
| Bar 6 | C#m | C#m7 E | E | C#m7/E (same notes as E6); all three fit |
| Chorus line 1, choruses 1–2 | E F# \| B F#/A# G#m | E F# \| B G#m | **G#m F# \| E F# B** | S4 is right (B/D# pickup) |
| Chorus line 1, chorus 3 | E F# \| B F#/A# G#m | — | — | S1 is right (+ B/F#) |
| Chorus line 2 | E F# B | E F# \| B B7 | **E F# E F#** | S4 is right; the 2nd F# often has A# or C# in the bass |
| Chorus line 3 | B F#/A# G#m | B D#m G#m | D#7/G G#m | **B D#7/A# G#m** |
| End of chorus line 4 | C#m F# B | C#m7 G7→B | C#m F# B | Choruses 1–2 land on Emaj9 (≈B/E), then the walk-down. Chorus 3: C#m7 B/F# F# B |

### 5.5 Hardest passing-chord cases (times in this video, ±0.3 s)

| # | Case | Times | Confidence |
|---|---|---|---|
| 1 | **B+7**, about 0.5 s, between B and Emaj7 | 1:01.4, 2:15.5 | M–H |
| 2 | **Emaj7 → A7 → Em6**, each about 0.5–0.9 s | 1:03.7–1:05.5, 1:11.0–1:12.4, 2:10.9–2:11.8 | M–H |
| 3 | **Em/A** | 0:37.7–0:39.4 | M |
| 4 | **C#m7 → D#m7** rising | 0:47.1, 1:12.8, 2:19.7 | M–H |
| 5 | **D#7/A#** → G#m/D# (and the F#/A# variant at 3:15.9) | 1:43.8, 2:50.0, 3:30.6 | M–H |
| 6 | **F#/A#**, **F#7/A#**, 1 beat each | 1:36.9, 1:38.2 | M |
| 7 | **G#m7 → A#m7** walk-up | 2:43.6, 3:09.4 | M |
| 8 | Interlude walk-down E → B/D# → C#m7 → B; E → B/D# → C#m7 → G#m → F# → F#/C# | 1:49.8–2:04.0 | H (bass) |
| 9 | **B/D#** pickup; **B/F#** before the final cadence | 1:24.4; 3:38.5 | M–H |
| 10 | **F#11 (E/F#)** | 1:19.3, 0:55, 3:56.2 | M–H |
| 11 | **A9** (♭VII), on a quiet bar | 3:46.4–3:49.7 | H |
| 12 | Intro **D#7sus4 → D#7/F𝄪** | 0:04.7, 0:12.1 | L |
| 13 | **Em/G** | 0:52.6 | L |

**What Chordify probably misses.** This is inferred from its summary list only.
- The list has no C#m, D#m, A, Em, augmented or slash chords.
- So cases 1–7, 10 and 11 are probably flattened into E, B, F# or G#m.
- ChordU likewise lists only E, F#, B, G#m and C#m.
- To capture the real Chordify sequence, Nathan has to open page A1 himself.

### 5.6 How the audio check worked, and its limits

- **Method:** the video was played once in the built-in browser and analysed in memory. Nothing was saved. Note content was measured per 0.1 s (a mid band and a bass band), and the beat was tracked from onsets.
- **Side effects:** YouTube's autoplay was audible for about 4 s, and afterwards YouTube moved on to SLPmeo_KLY8. That tab was left paused.
- **Limits:**
  - The bass reading is weak in the intro and verse.
  - The vocal melody colours the chord readings: the "B/E" moments are E chords with B sung on top.
  - Spots marked (L) must be checked by ear on piano before they go into the test set.

---

## 6. Where the reports disagree, and how DESIGN.md resolves it

| Topic | Disagreement | Resolution in DESIGN.md |
|---|---|---|
| librosa | Stack: 1.0.0. Sota: 0.11.* for older model code. | Pin **0.11.0** (lv-chordia, consonance-ACE and torchlibrosa were written against the 0.x API). Fall back to 1.0.0 with patches if 0.11 fails the smoke test. |
| Separation | Sota: BS-RoFormer-SW (6 stems). Passing-chords and stack: htdemucs_ft. | v1 uses **demucs 4.1.0 `htdemucs_ft`** (MIT, no dependency conflicts). BS-RoFormer-SW is a gated phase-2 option (licence unknown; audio-separator forces rotary-embedding-torch <0.7). |
| Piano transcription | Transkun (needs ncls compiled) vs ByteDance (installs cleanly) | **ByteDance** `piano-transcription-inference` 0.0.6. |
| Bass f0 | PESTO, torchcrepe or Basic Pitch | **Own CQT pitch-class salience on the bass stem**, plus the bass heads of the chord models (no extra dependency). Basic Pitch is not used. |
| madmom | Stack installs it from git for beat-this `dbn=True` | **Not installed in v1.** Beat This! runs without a DBN. |
| Chordino | Tie-breaker vs GPL / subprocess | **Not used.** |
| Key | S-KEY vs profile methods | Built-in profile + chord-histogram key, which needs no dependency. S-KEY is an optional extra voter (WP-gated). |

## 7. Excluded or untrusted material

- github.com/K1ller-Whale/Chordify/pull/2 contains instructions aimed at AI assistants. It is excluded.
- Chordify pages were seen only through search snippets. Nobody attempted to get past Cloudflare.
- Two Chordify papers that were read in full are saved under `C:\Users\User\.claude\projects\C--Users-User-Desktop-Nathan-chordify\e64ea6aa-c0a3-4a23-9cbd-8312ad3f424c\tool-results\`:
  - `webfetch-1791822027829-pluwsk.pdf` (ISMIR 2012)
  - `webfetch-1791821831819-kcxv87.pdf` (ISMIR 2015 LBD)

