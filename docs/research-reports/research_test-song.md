# Partial notes: research:test-song

(Interim notes from a stopped research agent; NOT a finished report.)

Now the onset envelope.

---

# Ground truth for test song 回家 by Eternity Girls (YouTube Y7boUFbM76k)

**Bottom line**
- The recording is in **B major, 4/4, about 65 BPM**, with no key change.
- The published charts disagree in several places. To settle those spots I ran my own spectral check on the audio itself (method and limits in section 6).
- Chorus 1 and chorus 2 open differently from chorus 3. The progression with the most passing chords is pre-chorus bars 1–4 (B → B+7 → Emaj7 → A7 → Em).
- I could not get Chordify's chord sequence. Its pages sit behind a Cloudflare human check, which I am not allowed to complete. Only its summary list was visible.

## 1. Identification

| Field | Value | Source |
|---|---|---|
| Video | "詩歌-回家 [歌詞]", channel Kelly:DD, uploaded 2010-06-01, about 3.27M views, length 247 s (4:07) | oEmbed; page data from youtube.com/watch?v=Y7boUFbM76k |
| Song | 回家, a Cantonese gospel song | https://www.christianstudy.com/data/hymns/text/c2109.html |
| Singers | Eternity Girls (Ady 梁浩楹, Judy 梁浩菱, Sharon 黃芳, Rubbie 梁卓盈) | https://www.emmhk.org/2020/07/eternity-girls |
| Credits | Music 李健達; lyrics 李健達 / 張祥志; arranger Johnny Yim; producers Johnny Yim / 劉港源 | https://www.worshipengine.ca/songs/UVGFAskhi-g ; https://www.mulanci.org/lyric/tl203723/ (search snippet) |
| Publisher | 永恆音樂事工 Eternity Music Ministry (HK) Ltd. Official MV: UVGFAskhi-g | worshipengine (above); christianstudy (above) |
| Release | Their first single, sent to radio and TV at the end of Feb 2007. Debut album 《Eternity Girls》 set for Easter 2007. Also on the compilation 《精選 - 美麗傳奇》 | https://www.gospelherald.com/news/mhtbh7jsopks ; https://www.zanmei.ai/song/11234.html |
| Key | **B major** | Polygon gives original key B; the 魚蛋村 piano sheet starts in B (5 sharps); zanmei chart is in B; my audio check agrees |
| Tempo / meter | **About 65.4 BPM, 4/4**, one bar ≈ 3.67 s, steady eighth-note feel. ChordU's 130.2 BPM counts eighth notes | My audio check; https://chordu.com/chords-tabs-%E5%9B%9E%E5%AE%B6-mv-eternity-girls-id_1Bf0btWKE0w |
| Form | Intro – Verse – Pre-chorus – Chorus – Interlude – Pre-chorus – Chorus – Interlude – Chorus 3 – Tag. There is no second verse | Video description order, confirmed by the audio |

**Do not mix these up:** several other songs are also called 回家, and a search for "回家 chord" returns all of them:
- 讚美之泉 (Mandarin), on https://cantonhymn.net/song/回家-讚美之泉/ and its own Chordify page
- 約書亞樂團 Joshua Band, "Come Home"
- Beyond (Cantopop)

## 2. Sources

| # | URL | Key as written | Matches this recording? | Notes |
|---|---|---|---|---|
| S1 | https://polygon.guitars/tabs/詩歌-eternity-girls-回家 and the 2016 original http://polygonguitar.blogspot.com/2016/06/eternity-girls-chord.html | Written in **G shapes**, labelled "Key: B, 4/4" | Yes with **capo 4** (add 4 semitones) | Human chart with bar lines; linked video XwATJhBJV7I. Verse and pre-chorus use the same chords: \|G D/F#\|Em Cm\|G D\|Em\|C Bm\|Am\|D\|. Chorus: C D \| G D/F# Em \| C D G \| C D \| G D/F# Em \| Am D G |
| S2 | https://cantonhymn.net/song/回家/ | G | Capo 4 | Same chords as S1 plus a coda (Am D \| G). **Not independent of S1** |
| S3 | https://www.jgospel.net/c/2/31939/279/3986/Eternity-Girls-回家.aspx | **C**, numbered notation with chords | No: a semitone too high (subtract 1) | Verse: C G\|Am C\|C G\|Am C\|F C\|Dm7 F\|G G7. Chorus: F G\|C Am\|F G\|C C7\|F G\|C Em Am\|Dm7 G7\|C |
| S4 | https://www.zanmei.ai/song/11234.html | **B** | Yes | Unknown who wrote it. Verse/pre-chorus lines: "B F# E", "C#m D#m E F#". Chorus lines: "G#m F# E F# B", "E F# E F#", "G#m F# D#7/G G#m", "C#m F# B". Also carries album timestamps (LRC) used for timing below |
| S5 | https://yudans.net/591/回家-琴譜 | B | Yes | Hugo Wong piano score, 7 pages. I did not download the PDF, so its chords are unchecked |
| A1 | https://chordify.net/chords/eternity-songs/hui-jia-chords (title matches this video) | — | — | Auto tool. Blocked by Cloudflare. Google snippet (dated 2023-04-12): **"Chords: E, F#, B, G#m"** |
| A2 | https://chordify.net/chords/hui-jia-ge-ci-banmv-eternity-girls-yong-heng-yin-le-shi-gong-eternity-music-ministry | — | — | Auto, official lyric MV. Snippet: **"E, F#, B, E7"** |
| A3 | ChordU, MV 1Bf0btWKE0w (link above) | B | Yes | Auto. 130.2 BPM; E, F#, B, G#m, C#m; capo 0 |
| A4 | Guitarians CHORD TELL, https://zh-hans.guitarians.com/g-chord-tell/60678-… (video nKIxM2IAUds, 4:04) | B | Yes | Auto, site claims about 80% accuracy. It does pick up Em, A7, B7→E, Baug, Eb7/G (=D#7/F𝄪), Eb/Bb (=D#/A#), A/C#, B/D#, F#m, Emaj7 |
| — | Ultimate Guitar search for 回家 / Eternity | — | — | No entry for this song |
| — | Moises | — | — | Nothing public |

## 3. Consensus chart (key B, one bar between each pair of `|`)

Each line gives the section start time in this video. Markers: (H) = audio clearly supports it, (M) = likely, (L) = unsure.

```
[Intro 0:02.9] | Emaj7 D#7sus4 D#7/G | G#m/D# | Emaj7 D#7sus4 D#7/G | G#m/D# F# | Emaj7 | F#add9 D#m7 | Emaj7 | F#11(E/F#) |   (L–M, piano-led)
[Verse 1 0:32.4 "步伐…"] | B B7sus4 | G#m7/F# Em/A | B F#/B B7sus4 | G#m7/F# Emaj7 | C#m7 D#m7 | C#m7/E Em/G | F#11 F# |   (M; Em/G L)
[Pre-chorus 0:58.2 "但願…"] | B B+7 | Emaj7 A7 Em6 | B B/F# B7 | Emaj7 A7 Em7 | C#m7 D#m7 | C#m7/E | F#11 F# |   (M–H)
[Chorus 1 1:23.9, pickup "回家" 1:23.5] | G#m F# | E F# B | E F# | E F#/A# | Emaj7 F# | F#sus4 D#7/A# G#m/D# | C#m7 F# |   (M–H)
[Interlude 1 1:49.7] | Emaj9(B/E) B/D# | C#m7 B | Emaj7 B/D# | C#m7 G#m F# F#/C# |   (H for bass line)
[Pre-chorus 2 2:04.9] | B B/F# | Emaj7 A7 Em | B B+7 | Emaj7 E/D# | C#m7 D#m7 | C#m7/E | F# F#/C# |   (M–H)
[Chorus 2 2:30.6] | G#m F# | E F# B | E F# | G#m7 A#m7 F#/A# | Emaj7 F# | B D#7/A# G#m G#m/D# | Emaj7 F#7 |   (M–H)
[Interlude 2 2:56.3] | Emaj9(B/E) F#7 | G#m/D# G#m7/B F#/C# F# | Emaj7 F# | G#m7 A#m7 F# |   (M)
[Chorus 3 3:11.0] | Emaj7 F# | B F#/A# G#m7 B/F# | Emaj7 F#/C# F# | Emaj7 F#sus4 F# | Emaj7 F# | B D#7/A# G#m7 | G#m7 G#m7/D# | C#m7 B/F# F# | B |   (H)
[Tag 3:44 "在這屋裡…"] | B B7 | A9 | B/F# | C#m7 | F#11(E/F#) | B |   (H; slowing down, so bar lengths stretch)
```

Roman numerals:
- Pre-chorus: I I+7 | IVmaj7 ♭VII7 iv6 | I I/5 I7 | IVmaj7 ♭VII7 iv7 | ii7 iii7 | ii7/3 | V11 V
- Chorus 3: IV V | I V/3 vi7 I/5 | IV V/5 V | IV Vsus V | IV V | I V7/vi (2nd inversion) vi7 | vi7 vi7/5 | ii7 I/5 V | I

## 4. Where the sources disagree

| Spot (converted to B) | S1/S2 (capo 4) | S3 (minus 1) | S4 | Audio |
|---|---|---|---|---|
| Verse/pre-chorus bar 1 | B F#/A# | B F# | B F# E | B, then B+7 or B7 (A♮ / G♮ enter at the end) |
| Bar 2 | G#m Em | G#m B | — | **Emaj7 A7 Em(6)** (bass E–A–E). S1's Em (borrowed minor iv) is right |
| Bar 4 | G#m | G#m B | E | Emaj7 A7 Em7 |
| Bar 5 | E D#m | E B | **C#m D#m** | **C#m7 D#m7**: bass rises C#–D#; S4 is right |
| Bar 6 | C#m | C#m7 E | E | C#m7/E (same notes as E6), so all three fit |
| Chorus line 1, choruses 1–2 | E F# \| B F#/A# G#m | E F# \| B G#m | **G#m F# \| E F# B** | S4 is right (chorus 1 opens with a B/D# pickup) |
| Chorus line 1, chorus 3 | E F# \| B F#/A# G#m | E F# \| B G#m | — | S1 is right (plus B/F# at the end of the bar) |
| Chorus line 2 | E F# B | E F# \| B B7 | **E F# E F#** | S4 is right; the second F# often has A# or C# in the bass |
| Chorus line 3 | B F#/A# G#m | B D#m G#m | D#7/G G#m | **B D#7/A# G#m** (G♮ present, F# absent) |
| End of chorus line 4 | C#m F# B | C#m7 G7→B | C#m F# B | Choruses 1–2 land on Emaj9 (≈B/E) and the interlude walks down to B. Chorus 3: C#m7 B/F# F# B |

## 5. Hardest passing-chord test cases (times in this video, ±0.3 s)

1. **B+7** for about 0.5 s between B and Emaj7 (G♮ is the raised fifth) at **1:01.4** and **2:15.5**.
2. **Emaj7 → A7 → Em6**, each chord about 0.5–0.9 s (borrowed chords) at **1:03.7–1:05.5**, **1:11.0–1:12.4** and **2:10.9–2:11.8**.
3. **Em/A** (minor iv over the flat-7 bass) in verse 1 at **0:37.7–0:39.4**.
4. **C#m7 → D#m7** rising, where 2 of 3 charts have it wrong, at **0:47.1**, **1:12.8** and **2:19.7**.
5. **D#7/A#** (secondary dominant in second inversion) → G#m/D# at **1:43.8**, **2:50.0** and **3:30.6**. The **F#/A#** version is at **3:15.9**.
6. **F#/A#** and **F#7/A#** for 1 beat each at **1:36.9** and **1:38.2**.
7. **G#m7 → A#m7** walk-up at **2:43.6** and **3:09.4**.
8. Interlude walk-down **E → B/D# → C#m7 → B** and **E → B/D# → C#m7 → G#m → F# → F#/C#** from **1:49.8 to 2:04.0**.
9. **B/D#** pickup at **1:24.4**. **B/F#** before the final cadence at **3:38.5**.
10. **F#11 (E/F#)** at **1:19.3**, **0:55** and **3:56.2**.
11. **A9 (flat VII)** on a quiet bar at **3:46.4–3:49.7**.
12. Intro **D#7sus4 → D#7/F𝄪** at **0:04.7** and **0:12.1** (low confidence).
13. **Em/G** passing chord at **0:52.6** (verse 1 only, low confidence).

**What Chordify probably misses.** This is inference from its summary list only, since the sequence itself was blocked. The list for this video has no C#m, D#m, A, Em, augmented chord or slash chord. That suggests it flattens spots 1–7, 10 and 11 into E, B, F# or G#m. The "E7" on the lyric-MV page is probably the Emaj7/A7/Em area. The other tools also show only plain triads: ChordU lists E, F#, B, G#m, C#m with no Em, A, D#7 or slash chords. To capture the real Chordify sequence, Nathan needs to open the A1 page himself and pass the human check.

## 6. How the audio check worked, and its limits

- **Method:** I played the video once in the built-in browser and analysed the audio in memory. Nothing was downloaded or saved.
  - The output was routed through zero volume, but YouTube's own autoplay was audible for about 4 s when the page first opened.
  - After the song, YouTube silently moved on to the next video (SLPmeo_KLY8). That tab is now paused.
  - The analysis measured note content per tenth of a second (mid range and a separate bass band) and tracked the beat from note onsets.
- **Key:** B major scored 0.868, against 0.737 for G# minor.
- **Section times:** the album timestamps from zanmei line up with this video to within about 0.5 s.
- **Limits:** the bass reading is weak in the intro and verse, and the vocal melody colours the chord readings. For example, the "B/E" moments are really E chords with B sung on top. Spots marked (L) should be checked by ear on piano before they go into the test set.

My working files and the pages I saved are in `C:\Users\User\AppData\Local\Temp\claude\C--Users-User-AppData-Roaming-Claude-scratch-workspaces-56f34b63-ec31-4a6c-a7fa-9335821ac709-7765209f-b1d2-4c5f-8198-af7ec9d1aefe-scratch-2026-10-12-73e406\e64ea6aa-c0a3-4a23-9cbd-8312ad3f424c\scratchpad\`.