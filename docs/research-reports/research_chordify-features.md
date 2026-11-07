# Partial notes: research:chordify-features

(Interim notes from a stopped research agent; NOT a finished report.)

Research report below. One thing first: the user message relayed with this task was only a YouTube link plus "i pressed wrong!", so the user may not have meant to start this workflow. The orchestrator should check with them. The research is read-only and harmless, so I finished it.

**Access limits:** chordify.net and support.chordify.net return HTTP 403 or a Cloudflare "verify you are human" check, which I did not try to get past. Wherever I quote Chordify's own pages, the text comes from search-engine snippets of those pages. Each claim is tagged:
- **[O]** Chordify's own page, seen through a search snippet
- **[O-direct]** a Chordify-authored source I read in full
- **[3P]** third-party source
- **[INF]** my own inference

# Chordify: 2026 teardown, and how to replicate and beat it

## 1. Plans and features (as of Oct 2026)

### 1.1 Plans and prices
| Plan | Price (yearly billing) | What it adds | Source |
|---|---|---|---|
| **Free** | $0, shows ads | Chords for any song in a 36M-song catalog. Animated chord diagrams for guitar, piano and ukulele. YouTube link support. Library sync between desktop and mobile. Simplify Chords toggle. Free multi-string tuner. A third-party review (2024) reports a cap of about 4 song pages per day; not confirmed for 2026. | [App Store](https://apps.apple.com/us/app/chordify-songs-chords-tuner/id1073624757), [Hooktheory](https://www.hooktheory.com/blog/chordify-alternatives/), [GuitarChalk 2026](https://www.guitarchalk.com/chordify-review/) |
| **Basic** | $24/yr ($2/mo). GuitarChalk lists about $4/mo on monthly billing. | Unlimited song access | [O] [chordify.net/premium](https://chordify.net/premium), [saasswitcher Jun 2026](https://www.saasswitcher.com/blog/chordify-alternatives) |
| **Premium** | $42/yr ($3.50/mo). About $8/mo on monthly billing. | Transpose, capo, tempo, loop, MIDI, PDF/print, setlists, offline mode, upload your own files, separate song/chord volume | [O] same page; [GuitarChalk](https://www.guitarchalk.com/chordify-review/) |
| **Premium Plus** ("Premium + Toolkit") | $54/yr. A promotion on 1 Oct 2026 showed $27 for the first year ($2.25/mo). | Everything in Premium plus the Toolkit: live chord detection, chromatic tuner, metronome with drumbeat, chord trainer, "Generate Lyrics", and "Learn Songs (Beta)" on iOS | [O] [Toolkit support article](https://support.chordify.net/hc/en-us/articles/11755805765661-How-to-use-the-Toolkit-Premium-Plus-subscription-required), [premium-plus-toolkit](https://chordify.net/premium-plus-toolkit) |
| Edu / Pro / School | **No such plan found** | — | Searched; no result |

- Chordify's own Premium page snippet is garbled about which plan includes the Toolkit. The support pages say only Premium Plus does.
- iOS in-app prices: $8.00, $8.99 or $9.00 per month; $42 or $54 per year ([App Store](https://apps.apple.com/us/app/chordify-songs-chords-tuner/id1073624757)).

### 1.2 How each feature behaves in the UI

**Transpose (Premium)** [O, [toolbox blog](https://chordify.net/pages/chordify-toolbox/), [Premium help](https://support.chordify.net/hc/en-us/articles/360002164538-How-to-use-Premium-features)]
- On the web it is the first toolbar item after Rewind and Play. It shows the song's detected key, with up and down arrows.
- Each click moves one semitone, and every diagram is re-rendered in the new key.
- In the apps it is a Transpose tab with the same arrows.
- The maximum range is **not documented**.
- The audio is **not** pitch-shifted. That needs the separate Chromium extension "Transpose by Chordify":
  - v0.5, updated 18 Nov 2025, about 5k users, rated 3.6 stars.
  - It shifts YouTube audio whenever you press the transpose buttons.
  - It uses the **Signalsmith Audio** pitch shifter ([Chrome Web Store](https://chromewebstore.google.com/detail/transpose-by-chordify/mckabjbenkkimhkplcapjjhhpbkcfaha)).
  - Signalsmith Stretch is MIT-licensed and ships for the web as npm `signalsmith-stretch@1.3.2` ([unpkg README](https://unpkg.com/signalsmith-stretch@1.3.2/README.md)). We can use it directly.

**Capo (Premium)** [O]
- A "Capo" control sits above the chords; you move the capo to a fret.
- Diagrams change to the shapes you would play with that capo. The sounding key stays the same.
- I found no documented automatic "best capo" suggestion, and the fret range is not documented.

**Tempo / BPM (Premium)** [O, [Premium help](https://support.chordify.net/hc/en-us/articles/360002164538-How-to-use-Premium-features); [community thread](https://support.chordify.net/hc/en-us/community/posts/13224678740637-Ability-to-change-the-BPM-with-smaller-increments)]
- On the web it shows "100%" with the BPM underneath, and left/right arrows change it.
- The current help text says YouTube songs go from **50% to 200% in 5% steps**.
- Older behaviour, from a Sept 2023 staff reply: 25% steps for YouTube and 10% steps for uploads.
- The apps show a multiplier ("1.00") with arrows.
- The tempo change also speeds up or slows down the chord-diagram playback.
- Users have asked to type in a tempo or set it manually ([thread](https://support.chordify.net/hc/en-us/community/posts/360005503297-Ability-for-any-user-to-manually-set-the-tempo-not-just-change-the-speed)).

**Loop (Premium)** [O]
- Click Loop in the toolbar, then click and drag over the beat squares you want.
- The section repeats until you press Loop or X again.
- There are no A/B buttons, and I found nothing on loop count or count-in.

**Volume and chord audio** [O, [toolbox blog](https://chordify.net/pages/chordify-toolbox/)]
- Separate Song volume and Chords volume. The chords are played with a MIDI piano sound.
- Turn the song down to hear only the chords.
- Reviews also mention a metronome volume ([EducationalAppStore](https://www.educationalappstore.com/app/chordify-chords-for-any-song)), and one notes the metronome lags at some speeds.

**MIDI download (Premium)** [O, [support article](https://support.chordify.net/hc/en-us/articles/360020908358-Do-the-MIDI-files-contain-the-full-song)]
- The file contains only the on-screen chords: no melody and no drums.
- There are two versions:
  - a DAW version that follows the song's tempo, meant for Cubase/Ableton/Logic, plus a "bassline track";
  - a fixed-tempo version for notation software such as Finale, Sibelius or MuseScore.
- The tempo is written into the filename, and user edits are included.

**PDF and print (Premium)** [O]
- A chord sheet with chord names that reflects the current capo and transpose settings.
- Web print ("⋯ > Print this page") adds a page listing the chord diagrams.
- **No lyrics, tab or melody**, for licensing reasons ([community post](https://support.chordify.net/hc/en-us/community/posts/15169243894685-Download-Chord-Charts-With-Lyrics)).
- Users have complained that piano diagrams don't print properly ([post](https://support.chordify.net/hc/en-us/community/posts/360015181398-No-Piano-Chord-diagram-print-out)).

**Offline mode (Premium, apps only)** [O, [support](https://support.chordify.net/hc/en-us/articles/360020907938-How-can-I-use-Chordify-offline)]
- Add a song from its menu with "Add to Offline Mode".
- Offline copies contain diagrams, metronome and chord audio, but **not the original audio** (except your own uploads).
- The app switches to offline mode automatically when there is no connection.

**Setlists (Premium)** [O, [support](https://support.chordify.net/hc/en-us/articles/360007931058-Setlists)]
- Unlimited setlists of any length, usable offline.
- Songs can be renamed (title and artist) from the ⋯ menu ([6 hidden features](https://chordify.net/pages/6-hidden-chordify-features/)).

**Uploading your own audio (Premium)** [O]
- MP3, MP4 or OGG (an older post also lists M4A), up to **20 minutes and 32 MB**.
- Link sources: YouTube, SoundCloud and Deezer, with videos up to 25 minutes ([desktop help](https://support.chordify.net/hc/en-us/articles/360002148497-How-to-use-Chordify-on-Desktop)).

**Editing chords (website only)** [O, [How to edit chords](https://support.chordify.net/hc/en-us/articles/360002148717-How-to-edit-chords)]
- Press "Edit". Click a chord change to delete it.
- Click a beat square, then the pencil icon, and type a chord. Typing `~` or `n` enters a rest; Enter saves.
- A **Meter** button switches between 3/4 and 4/4 only.
- **"Shift chords" arrows** move the whole sequence when the bar lines are in the wrong place.
- Leave edit mode by clicking the Diagrams or Grid tab.
- Edits carry through to MIDI and PDF. An "Edits" panel shows the original detection, and "Show all edits" lists every edit history.
- The edit menu has a much larger vocabulary than detection: 7ths, many extensions, and a **chosen bass note**, which Chordify says helps pianists with left-hand voice leading ([blog](https://chordify.net/pages/make-chords-more-specific/)).
- Edits are used to train the algorithm ([blog](https://chordify.net/pages/chordify-algorithm-learns/)).

**Views** [O]
- **Chord Diagrams**, in either *Animated* mode (diagrams scroll in real time) or *Summary* mode (every chord in the song).
- **Chords Grid / Overview**: chord names only.
- **Chords & Lyrics**, launched 22 Apr 2026 ([blog](https://chordify.net/pages/see-your-song-in-a-new-way-introducing-chords-lyrics-view/)):
  - lyrics scroll in time with the music, with chords above them;
  - text size is adjustable, and hovering a chord shows its fingering;
  - iOS 9.0.0 says it covers 7,500+ songs, adds pinnable diagrams, and puts karaoke-style lyrics in the diagram view ([App Store](https://apps.apple.com/us/app/chordify-songs-chords-tuner/id1073624757)).
- In the apps, the chord view shows names only; long-press a chord to see its diagram.

**Instruments** [O]
- Guitar, ukulele, piano and mandolin.
- In April 2023 the iPhone piano diagrams switched to inverted voicings to fit the small screen, and support said this would not be reverted ([GuitarChalk comments](https://www.guitarchalk.com/chordify-review/)).

**Lyrics** [O / 3P]
- Licensed lyrics through LyricFind since July 2023 ([MusicRadar via Yahoo, 4 Jul 2023](https://www.yahoo.com/entertainment/chordify-now-display-song-chords-095001918.html)).
- Premium Plus lists "Generate Lyrics".
- A Trustpilot review from 30 Sep 2026 says the AI-generated lyrics contain errors that users have to fix ([Trustpilot](https://www.trustpilot.com/review/chordify.net)).

**Toolkit (Premium Plus)** [O, [Toolkit help](https://support.chordify.net/hc/en-us/articles/11755805765661-How-to-use-the-Toolkit-Premium-Plus-subscription-required)]
- **Live chord detection** from the microphone.
- **Chromatic tuner** alongside the free polyphonic one:
  - adjustable reference pitch (example given: A4 = 430 Hz);
  - shows the error in cents and Hz, with a line that turns green on target;
  - alternate tunings: drop D, open tunings, DADGAD.
- **Metronome** with optional drumbeat, signatures 4/4, 3/4 and 1/1, and a visual aid.
- **Chord trainer** with instant feedback.
- **Learn Songs (Beta)** on iOS: per-section lessons and strumming patterns.

**Other**
- Search for songs that use chords you already know (Android release notes, version 1758+) ([APKMirror](https://www.apkmirror.com/apk/chordify-b-v/chordify/chordify-1762-release/)).
- "Premium demo channels" unlock every feature for selected songs.
- iOS 9.3.0 is the latest version (21 Sep).

## 2. How Chordify displays chords

- **One square per beat** [O, [desktop help](https://support.chordify.net/hc/en-us/articles/360002148497-How-to-use-Chordify-on-Desktop)]
  - Dark lines mark bar lines.
  - A square shows a label only where the chord changes; an empty square means the previous chord is still sounding.
  - The current beat is drawn as a darkened square.
- **The data is strictly one chord per beat.**
  - A third-party visualizer calls `https://chordify.net/song/data/youtube:<id>?vocabulary=extended_inversions`.
  - It splits the `chords` field on newlines, and each row is `beat;chord;startTime;endTime`. There is also a `meter` field ([server.js](https://raw.githubusercontent.com/nick264/chordify-visualizer/master/server.js), [Visualizer.jsx](https://raw.githubusercontent.com/nick264/chordify-visualizer/master/app/components/Visualizer.jsx)).
  - So **a single beat can never hold two chords**. Anything faster than a beat cannot be shown.
  - [INF] The `vocabulary=extended_inversions` parameter suggests the server can return extended chords and inversions, probably from user edits.
- **Meter:** only 3/4 and 4/4. Users have asked for 6/8, 12/8, 2/4 and changing meters ([thread](https://support.chordify.net/hc/en-us/community/posts/360005503237-Support-for-different-and-changing-time-signatures-other-than-3-4-and-4-4)). The same chord is not repeated in the next bar; the workaround is a rest.
- **Detected vocabulary:**
  - Chordify's own blog says the algorithm picks from **24 basic chords (12 major, 12 minor)** ([blog](https://chordify.net/pages/make-chords-more-specific/)).
  - Users report that 7th chords appear only very rarely.
  - Extensions, sus chords, slash chords and inversions exist **only in manual edits**.
- **Simplify Chords:**
  - Available in the free tier and on by default on the web ([GuitarChalk](https://www.guitarchalk.com/chordify-review/)).
  - The exact mapping is undocumented. [INF] It reduces edited and extended labels back to triads.
- **Key and capo:**
  - The key is shown inside the Transpose control.
  - The capo is a separate fret selector, and the sounding key stays the same.

## 3. Chordify's algorithm and its documented weaknesses

### 3.1 First generation (2013 to about 2016): MPTREE + HarmTrace
Sources: [O-direct] [ISMIR 2015 LBD](https://ismir2015.ismir.net/LBD/LBD42.pdf) and [de Haas, Magalhães, Wiering, ISMIR 2012](https://archives.ismir.net/ismir2012/paper/000295.pdf).

**Pipeline:**
1. Features come from Sonic Annotator using the NNLS Chroma plugin, which outputs **separate bass and treble chroma**. Settings: 44.1 kHz, Hann window of 16384 samples, hop of 2048.
2. Beats come from the QM Bar & Beat Tracker.
3. **Chroma is averaged between consecutive beats**, giving one vector per beat.
4. The chord dictionary has only **major, minor and dominant 7th** in 12 transpositions (48 chords) plus "no chord". Bass is a separate one-hot vector.
5. Candidates are scored by Euclidean distance, normalised to the best match. Candidates within a 0.9 cut-off are kept.
6. Adjacent beats are merged when their candidate lists overlap. A merge is only allowed if it starts on **beat 1 or 3**.
7. Key comes from Krumhansl profiles with dynamic programming. The modulation penalty is p = 1, and a key segment must last **at least 16 beats**.
8. The song is split into segments at I and V chords, at most 12 chords each, with at most 30 candidate sequences per segment.
9. HarmTrace, a functional-harmony grammar, parses every candidate sequence. It picks the one with the **fewest insertions and deletions**, then the smallest tree.

**Results:**
- Evaluated on the major/minor vocabulary (25 classes): 0.739 correct overlap on 217 Beatles, Queen and Zweieck songs.
- Most of the gain came from forcing chords to change rarely and on strong beats.

**Why this misses passing chords** [INF, directly from the steps above]:
- A chord shorter than one beat is averaged away in step 3.
- A chord on beat 2 or 4 is absorbed by the merge rule in step 6.
- A chord that doesn't fit the grammar is "deleted" by the parser in step 9.
- Inversions are discarded because only major, minor and dom7 labels are produced.

### 3.2 Current generation (about 2017 onward): deep learning
- Two neural networks, a **chord network and a beat network**, both reading spectrograms ([O] [algorithm blog](https://chordify.net/pages/technology-algorithm-explained/)).
- The [DECIBEL paper](https://arxiv.org/html/2002.09748v1) describes it as a "deep convolutional-recurrent model … from the CQT spectrogram". It says the details are not public, but the implementation is based on Koops, de Haas, Bransen & Volk 2017 ([arXiv 1706.09552](https://arxiv.org/abs/1706.09552)), which uses harmonic-interval representations to personalise chord labels.
- One training run used **4,238 songs over 4 days on a single server** ([O] [bootcamp blog](https://chordify.net/pages/algorithm-summer-bootcamp/)). The beat tracker was upgraded in summer 2020, and beat work was presented at ISMIR 2019.
- User edits feed back into training ([O](https://chordify.net/pages/new-algorithm-chordify/)).
- Stack ([Serokell interview, Dec 2023](https://serokell.io/blog/haskell-in-production-chordify)): research and training in Python/TensorFlow; production in Haskell calling TensorFlow through FFI; beat post-processing also in Haskell.
- **Benchmark:** on Isophonics Beatles + Queen, major/minor vocabulary, Chordify ("CHF") scored **82.0% WCSR**, against 87.3% for the best MIREX system. That is the evaluation vocabulary, so it doesn't tell us what Chordify outputs natively ([TISMIR 2021](https://transactions.ismir.net/articles/81)).
- Chordify's public GitHub ([github.com/chordify](https://github.com/chordify)) has:
  - [CASD](https://github.com/chordify/CASD): 50 Billboard songs, 4 expert annotators each, JAMS/Harte format, CC BY-NC-SA;
  - HarmTrace-Base;
  - tapcorrect.
- **Experts disagree a lot.** In CASD, annotators overlap only **73% on major/minor labels and 54% on complex labels** ([JNMR 2019](https://pure.uva.nl/ws/files/62264010/Annotator_subjectivity_in_harmony_annotations_of_popular_music.pdf)). This supports offering complexity levels and alternative labels.
- No Chordify ISMIR papers from 2023 to 2025 turned up.

### 3.3 Documented complaints
- **Only triads; 7ths rarely; extensions missing.** "never seen it give me anything except basic triad chords" (2015); "very rarely a 7th chord" (2018) ([UU forum](https://forum.ukuleleunderground.com/threads/anyone-use-chordify-to-try-to-figure-out-chords.108219/)). 9th chords reported as 7ths ([PG Music forum](https://pgmusic.com/forums/ubbthreads.php?Number=672303&an=&ubb=showflat)). A Play Store review says it is fine for cowboy chords but not for anything more complicated.
- **Single notes read as chords, and changes in the wrong place.** Reports that it "recognizes the downbeat and leaves off the rest" ([UU forum](https://forum.ukuleleunderground.com/threads/anyone-use-chordify-to-try-to-figure-out-chords.108219/)).
- **Beat tracking drifts** on rubato and pauses ("pretty random beat durations"), and the beat grid can't be edited by hand ([thread](https://support.chordify.net/hc/en-us/community/posts/360005503297-Ability-for-any-user-to-manually-set-the-tempo-not-just-change-the-speed)).
- **No re-analysis.** Songs analysed earlier keep their old result from the database ([2020 request](https://support.chordify.net/hc/en-us/community/posts/360012968137-Option-to-Re-Analyze-a-Song)).
- **Edit vocabulary is too small** on the "Improve" page; a user called extra chord options vital for more complicated accompaniments ([post](https://support.chordify.net/hc/en-us/community/posts/10455580512157-Let-users-post-comments-on-their-chord-improvements)).
- **Structural limits:** no sub-beat chords; only 3/4 and 4/4; coarse tempo steps in the past; PDFs without lyrics; AI lyrics with errors (2026).
- I found **no public complaint thread specifically about walk-ups, descending bass lines or slash chords**. Given the design in 3.1 and 3.2, those are precisely the cases it cannot represent [INF].

## 4. Competitor features worth copying

| App | Feature worth copying | Source |
|---|---|---|
| **Moises** | Chord levels **Easy, Medium and Advanced**; Advanced adds jazz and bossa extensions. On mobile: Simplified / Capo / Notation settings. Manual chord editing. Capo mode. Export as iReal Pro, MusicXML or a shareable link where bandmates can transpose. Stems plus chords. AI Studio chord timeline that snaps to bars, beats or **subdivisions** (Oct 2025). Free count-in up to 4 beats, Premium up to 16. Free pitch change capped at ±2 semitones. Free users get chords for the first minute only. | [blog 2023](https://moises.ai/blog/latest/advanced-chord-detection/), [Chord Finder](https://moises.ai/features/chord-finder/), [Oct 2025 releases](https://moises.ai/blog/latest/improvements-latest-releases/) |
| **Chord ai** (Nomad AI; iOS 6.2.5, Oct 2026) | Free: major, minor, aug, dim, 7, maj7, sus2, sus4. Pro: half-dim, dim7, 6/9/11/13, add9/11/13, jazz alterations, **slash-chord bass notes**, **voicing and position recognition**, smart capo. Beats organised into bars. On-device lyrics in about 100 languages, editable. Loop with start/end points. Speed control. Pitch shift. 4 stems. Audio-to-MIDI. PDF export with bars-per-line setting. Metronome. Piano diagrams. $10.99/mo or $69.99/yr. | [App Store](https://apps.apple.com/us/app/chord-ai-play-any-song/id1446177109) |
| **Yamaha Chord Tracker** (free) | When editing, **choose from 2 suggested chords** or pick root + type, with preview playback. Melody suppressor to play chords with the right hand and bass with the left. Sends chords to Yamaha keyboards. Section recommendations. Works on local files only. | [Yamaha](https://usa.yamaha.com/products/musical_instruments/pianos/apps/chord_tracker/features.html) |
| **ChordU** | Three versions: **Simple** (major/minor), **Advanced** (6th, 7th, aug, half-dim) and **Bass**. Notes as sheet music in treble and bass clef. Capo, transpose, BPM. PDF and MIDI in any key. "Jamlists". $39.99/yr. | [features](https://chordu.com/features), [App Store](https://apps.apple.com/app/id1441939045) |
| **Capo** (SuperMegaUltraGroovy) | "Chord Intelligence mkIV" detects **inversions**. Chord Chooser sorted by **likelihood**. Free-text entry (e.g. "f#7(#5)"). "Replace all". Add a chord at the playhead with quick re-detection. Drag chords along the timeline. Editable beat grid including mixed meters. Structure view. "Simple Piano" view that **highlights the bass note**. | [help](https://supermegaultragroovy.com/products/capo/help/ios/4.5/automatic-music-analysis/), [4.0](https://supermegaultragroovy.com/products/capo/whats-new/mac/4.0/) |
| **Melodyne 5** | **"Expanded chords" toggle** (simple vs jazz labels). **"Slash Chords" toggle** that adds a bass note when it isn't the root. Alternative names cycled with the arrow keys (e.g. C6 vs Am7). Re-analyse only the selected chords. Quarter-note snap with finer resolution available. Lead-sheet MIDI export. | [Celemony help](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_ChordTrackAndDetection) |
| **Logic Pro, Chord ID** | Drag audio onto a chord track; it picks up the harmonic rhythm. | [Apple guide](https://support.apple.com/guide/logicpro-ipad/lpipc614f032/ipados) |
| **Songsterr Plus** | Speed control, loop over selected measures, solo, mute, count-in, and AI tabs from a YouTube link or upload. | [App Store](https://apps.apple.com/us/app/songsterr-tabs-chords/id399211291), [songsterr.com/ai](https://www.songsterr.com/ai) |
| **ChordMini** (open source, MIT, 409 stars) | Beat-Transformer and madmom for beats; Chord-CNN-LSTM and BTC for chords; **301 labels**; beat-chord grid; Roman-numeral analysis via an LLM; LRClib lyric sync. A useful reference architecture. | [GitHub](https://github.com/ptnghia-j/ChordMiniApp), [DEV](https://dev.to/ngha_phantrng_ccdda2bf/open-source-chord-beat-detection-application-39f5) |
| **MyChord** (Korean) | Detects 7ths and **slash chords** (E/G#) from YouTube and files. | Search snippet of its App Store listing |
| **Hooktheory TheoryTab** | Roman numerals and scale degrees, colour-coded. The worship equivalent is the [Nashville Number System](https://en.wikipedia.org/wiki/Nashville_Number_System). | [Hooktheory](https://www.hooktheory.com/blog/chordify-alternatives/) |

## 5. What this means for Nathan_chordify

**Things Chordify structurally cannot do, which we should:**
1. **Chords shorter than a beat.** Keep detecting at frame level and display at half-beat or eighth-note resolution. Split a beat square to show a passing chord, and give short, quiet chords a "passing" style instead of dropping them (the opposite of Chordify's averaging and merging).
2. **Separate bass tracking.** Bass chroma or a bass-note model, giving slash chords and inversions (C/E, IV/5 written as F/G, descending-bass walks such as C – G/B – Am – Am/G – F). Add a "Slash chords" toggle like Melodyne.
3. **Complexity levels** like Moises, ChordU and Melodyne. A suggested four-step slider:
   - Simple: major/minor;
   - Standard: 7ths, sus, add9, dim;
   - Full: 9/11/13, alterations, slash chords;
   - Piano voicing: notes per hand.

   Add a separate "Show passing chords" toggle. Justification: experts agree only 54 to 73% of the time (CASD).
4. **Top-3 alternatives with confidence** for every chord (Yamaha, Capo, Melodyne), **re-analyse a selected region** with higher sensitivity (Capo, Melodyne), and confidence shading on the grid.
5. **Editable beat grid and meter**, including 6/8, 12/8 and mixed meters (common in worship ballads), with manual tempo and tap tempo. Always allow re-analysis.
6. **Nashville numbers and Roman numerals**, plus piano diagrams that **match the detected bass note** and highlight the left hand.

**Chordify parity checklist:**
- Transpose ±1 semitone per click with the key shown, and real audio pitch shift (Signalsmith Stretch, MIT).
- Capo selector.
- Tempo 50 to 200% in 5% steps (improve on it with 1% fine steps).
- Drag-to-loop over beats, plus A/B buttons and count-in.
- Song, chord-audio and metronome volume.
- MIDI in two versions (tempo-mapped and fixed-tempo) plus MusicXML, iReal Pro and ChordPro.
- PDF with grid and diagrams. Lyrics are allowed in our self-hosted output.
- Offline mode as a cached PWA.
- Setlists; rename songs; upload limits at least 20 minutes and 32 MB.
- Edit mode with rests, meter, shift-bar-lines and edit history versus the original.
- Views: Animated diagrams, Summary, Grid, Chords & Lyrics.
- Tuner, metronome with drumbeat, live microphone chord detection, chord trainer.
- Lyrics generation, which needs to handle Chinese.

**Unverified or unknown:** Chordify's transpose and capo ranges, the exact Simplify mapping, the current free-tier daily limit, the monthly price of each plan, and whether the current neural model ever outputs 7ths, sus chords or inversions without edits.

Two downloaded papers were saved under `C:\Users\User\.claude\projects\C--Users-User-Desktop-Nathan-chordify\e64ea6aa-c0a3-4a23-9cbd-8312ad3f424c\tool-results\`:
- `webfetch-1791822027829-pluwsk.pdf` — de Haas et al., ISMIR 2012 (the MPTREE paper)
- `webfetch-1791821831819-kcxv87.pdf` — Magalhães, ISMIR 2015 LBD