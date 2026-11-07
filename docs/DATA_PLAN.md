# DATA_PLAN — real worship songs with chord labels (research, 2026-11-01)

Goal: collect contemporary worship songs (about 1995–2015, **Cantonese + English**, not old hymns) with
trustworthy chord labels, so we can train / fine-tune our chord model and test it on real recordings.
Companion file: `data/songs_candidates.yaml` (128 candidate songs with links, keys and source notes).

Nothing was bulk-downloaded. I opened web pages, read 3 small preview images and 1 small official PDF
(活出愛, 75 KB) to confirm keys, and checked the headers of 3 PDFs. No accounts, no logins, no paywall tools.

**簡短摘要 (繁中)：** MuseScore 上的敬拜歌譜全部要登入，而且版權歌要付費 Pro，所以我們不能用。好消息係：
很多免費、毋須登入的來源有齊和弦：WorshipTogether（英文）、CantonHymn 同 Simply Worship（粵語/國語），
仲有兩首優先歌有官方免費樂譜（立志擺上：建道神學院；活出愛：泥土音樂）。YouTube 錄音只用嚟「測試」，
所以訓練用的聲音會由樂譜/和弦用鋼琴音色「彈」出嚟（即係你的第 2 個想法）。

---

## 1. The four songs Nathan asked for first

Legend: ✅ = free and no login needed (I opened it) · 🔒 = needs login or payment · ❔ = found but not opened.

### 1.1 〈立志擺上〉 — Cantonese — the recording to test: https://www.youtube.com/watch?v=WLtqi4s29LQ
- **Who wrote it / which version:** words and music by **DJS** = Derek, Joan & Shirley, three graduates of the
  Alliance Bible Seminary (建道神學院) worship course who met in 2013; first sung 2014; runner-up in the HKACM
  songwriting contest 2015. The video is the **編曲版** ("arranged version", arranger 古丹青) by the seminary's
  worship team 建道神學院新祢呈敬拜隊 (channel @abs.worshipteam).
  Other versions: the original **DJS版** (e.g. https://www.youtube.com/watch?v=B3mdetZP0bc ❔) and a choir version.
- **Key:** **E♭ major**, about 66 beats per minute, 4/4 (both official sheets; CantonHymn agrees).
- **Important:** the DJS版 uses *different chords* from the 編曲版 (the verse starts E♭(add2)–E♭maj7 in DJS版
  but A♭maj7–Gm7 in 編曲版). **Use the 編曲版 lead sheet to label this recording.**

| What | Link | Free, no login? |
|---|---|---|
| **Lead sheet 主領譜, 編曲版** (melody + chord names) — matches the video | [page](https://spcl.abs.edu/%E6%A8%82%E8%AD%9C%E6%96%B0%E7%A5%A2%E5%91%88/%E3%80%88%E7%AB%8B%E5%BF%97%E6%93%BA%E4%B8%8A%E3%80%89%E7%B7%A8%E6%9B%B2%E7%89%88/) · [PDF 2.4 MB](https://spcl.abs.edu/site/assets/files/3921/%E3%80%88%E7%AB%8B%E5%BF%97%E6%93%BA%E4%B8%8A%E3%80%89%E3%80%90%E7%B7%A8%E6%9B%B2%E7%89%88%E4%B8%BB%E9%A0%98%E8%AD%9C%E3%80%91%E5%BB%BA%E9%81%93%E7%A5%9E%E5%AD%B8%E9%99%A2%E6%96%B0%E7%A5%A2%E5%91%88%E6%95%AC%E6%8B%9C%E9%9A%8A.pdf) | ✅ (donation HK$20 suggested) |
| **Piano accompaniment score 鋼琴伴奏譜** (full piano part + chord names), DJS版, piano arr. Shirley Lai | [page](https://spcl.abs.edu/%E6%A8%82%E8%AD%9C%E6%96%B0%E7%A5%A2%E5%91%88/%E3%80%88%E7%AB%8B%E5%BF%97%E6%93%BA%E4%B8%8A%E3%80%89%E3%80%90%E9%8B%BC%E7%90%B4%E4%BC%B4%E5%A5%8F%E8%AD%9C%E3%80%91/) · [PDF 3.7 MB](https://spcl.abs.edu/site/assets/files/3512/%E3%80%88%E7%AB%8B%E5%BF%97%E6%93%BA%E4%B8%8A%E3%80%89%E3%80%90%E9%8B%BC%E7%90%B4%E4%BC%B4%E5%A5%8F%E8%AD%9C%E3%80%91%E6%9B%B2%E8%A9%9E%EF%BC%9ADJS_%40%E5%BB%BA%E9%81%93%E7%A5%9E%E5%AD%B8%E9%99%A2%E6%96%B0%E7%A5%A2%E5%91%88%E6%95%AC%E6%8B%9C%E9%9A%8A-1.pdf) | ✅ (HK$40 suggested). Great for piano training audio; *not* the video's chords |
| Lead sheet, DJS版 (original) | [page](https://spcl.abs.edu/%E6%A8%82%E8%AD%9C%E6%96%B0%E7%A5%A2%E5%91%88/%E3%80%88%E7%AB%8B%E5%BF%97%E6%93%BA%E4%B8%8A%E3%80%89djs%E7%89%88/) · [PDF 1.5 MB](https://spcl.abs.edu/site/assets/files/3511/%E3%80%88%E7%AB%8B%E5%BF%97%E6%93%BA%E4%B8%8A%E3%80%89%E6%9B%B2%E8%A9%9E%EF%BC%9ADJS_%40%E5%BB%BA%E9%81%93%E7%A5%9E%E5%AD%B8%E9%99%A2%E6%96%B0%E7%A5%A2%E5%91%88%E6%95%AC%E6%8B%9C%E9%9A%8A-1.pdf) | ✅ |
| Choir + piano score (Cantonese & Mandarin) | listed on the [seminary score index](https://spcl.abs.edu/%E6%A8%82%E8%AD%9C%E6%96%B0%E7%A5%A2%E5%91%88/) | ❔ |
| Chord chart, CantonHymn | https://cantonhymn.net/song/%e7%ab%8b%e5%bf%97%e6%93%ba%e4%b8%8a (key E♭) | ✅ |
| Chord chart, Simply Worship | https://simplyworship.hk/lyrics/%E7%AB%8B%E5%BF%97%E6%93%BA%E4%B8%8A/ (written in **D**, tempo 66) | ✅ |
| Chord chart, Guitarians | https://en.guitarians.com/chord/220128/ — page showed "no score content" | ❔ |
| MuseScore community score (by Isso Chan) | https://musescore.com/song/li_zhi_bai_shang-3689506 | 🔒 login + Pro |

The seminary's pages say the song may be sung in church (also online) without asking, if the writers are credited.

### 1.2 〈活出愛〉 Live Out Love — Mandarin — https://www.youtube.com/watch?v=lfP-3JLVvaw
- 盛曉玫 Amy Sand, 泥土音樂 Clay Music, album 6 《幸福 Blessed》 (**2012**, track 11). Official channel upload.
- **Key: C major, 12/8 time** (each beat is split in three), dotted-quarter ≈ 63. **The last chorus moves up to
  D major** after a short bridge (C – A♭ – C# – G#/B# – F#/A# – F#m/A). The label file must include this key change.

| What | Link | Free, no login? |
|---|---|---|
| **Official lead sheet** (staff + chord names + lyrics, 2 pages; the site button says 下載簡譜 but it is staff notation). The PDF has a text layer, so a program can read the chord names. | [album page](https://www.claymusic.org/albums/6) · [PDF 75 KB](https://kjfltzqqmqwmdttmgufs.supabase.co/storage/v1/object/public/assets/sheetmusic/06-11.pdf) | ✅ (I read it) |
| "官方和弦五线谱" (same official sheet as 2 images) on 爱赞美 | https://www.zanmei.ai/tab/2047.html | ✅ |
| Chord chart, CantonHymn (key C, 2012) | https://cantonhymn.net/song/%e6%b4%bb%e5%87%ba%e6%84%9b | ✅ |
| Chord chart, Chord4 (arr. 奕安, key C, transposable) | https://chord4.com/zh-hant/tabs/24461 | ✅ |
| Chord chart, Chord4 (arr. Mr.End, key C) | https://chord4.com/tabs/26249 | ❔ |
| Piano score 琴譜 (魚蛋村) | https://yudans.net/2021/%E6%B4%BB%E5%87%BA%E6%84%9B-%E7%90%B4%E8%AD%9C — blocked (HTTP 403) from our VPN address | ❔ |
| MuseScore community scores: [easy piano](https://musescore.com/user/79422433/scores/13668745), [piano solo](https://musescore.com/user/80345656/scores/17428699), [12332392](https://musescore.com/user/79422514/scores/12332392), [20171755](https://musescore.com/user/66174145/scores/20171755), [jazz "Live out love"](https://musescore.com/user/33511533/scores/6700525), [all versions](https://musescore.com/song/huo_chu_ai-3827134) | | 🔒 login + Pro |

Clay Music's terms: free to use in church; **no re-arranging, no changing lyrics, no translating, no songbooks**;
other uses need their written permission (contact on claymusic.org/copyright). Private research use is a grey
area — keep anything we make from it private.

### 1.3 "Here I Bow" — English — https://www.youtube.com/watch?v=hvHi7N8kc8s
- Brian & Jenn Johnson, Bethel Music, album *After All These Years* (2017; song © 2016). Writers: Jason Ingram,
  Matt Redman, Jonas Myrin, Brian Johnson, Jenn Johnson. CCLI 7069335.
- **Key: D** (WorshipTogether "original key"; PraiseCharts arranges it in D and E). Slow, 4/4.

| What | Link | Free, no login? |
|---|---|---|
| **Chord chart, WorshipTogether** (chords above lyrics) | https://www.worshiptogether.com/songs/here-i-bow/ | ✅ to read; its "free ChordPro download" needs a member login 🔒 |
| Chord charts, Ultimate Guitar (community, 2 versions) | [1963113](https://pt.ultimate-guitar.com/tab/brian-johnson/here-i-bow-chords-1963113) · [1977873](https://pt.ultimate-guitar.com/tab/brian-johnson/here-i-bow-chords-1977873) | ❔ |
| PraiseCharts chords / lead / piano | https://www.praisecharts.com/songs/details/37021/here-i-bow-sheet-music | 🔒 paid ($2.79 chords) |
| Bethel Music song page (lyrics + CCLI, no chords) | https://bethelmusic.com/resources/after-all-these-years/here-i-bow | ✅ |
| MuseScore | none found for this song | — |
| chordu.com page | an automatic machine guess (like Chordify) — **not usable as labels** | — |

No free piano sheet exists that I could find. Plan: label from the WorshipTogether chart + alignment (section 4).

### 1.4 "10,000 Reasons (Bless the Lord)" — English — https://www.youtube.com/watch?v=XtwIT8JjddM
- Matt Redman (2011), writers Jonas Myrin & Matt Redman, CCLI 6016351. Audio already cached in
  `Chordify_clone/cache/yt-XtwIT8JjddM`.
- **Key: G** (PraiseCharts says G; WorshipTogether's header says "C" but its chord text is in G; our clone engine
  measured G major, 72.6 BPM). WorshipTogether gives 72.5 BPM.

| What | Link | Free, no login? |
|---|---|---|
| **Chord chart, WorshipTogether** — the intro is even written in beats: `C // G // D/F# // Em // C // G // Dsus // D //` | https://www.worshiptogether.com/songs/10-000-reasons-bless-the-lord/ | ✅ |
| Chord chart, Ultimate Guitar (community; also versions 2–4) | https://tabs.ultimate-guitar.com/tab/matt-redman/10000-reasons-bless-the-lord-chords-1076668 | ❔ |
| PraiseCharts chords | https://www.praisecharts.com/songs/details/20858/10000-reasons-bless-the-lord-chords | 🔒 paid; the one "FREE" stage chart needs an account |
| MuseScore **Official** piano/vocal/guitar score | https://musescore.com/official_scores/scores/6951392 | 🔒 never downloadable |
| MuseScore community piano scores: [junxpiano](https://musescore.com/junxpiano/10-000-reasons-bless-the-lord), [easy](https://musescore.com/user/94620/scores/124550), [C major easy](https://musescore.com/user/2324416/scores/5043463), [moses373 easy](https://musescore.com/moses373/bless-the-lord-10000-reasons-matt-redman), [7339874](https://musescore.com/user/37099047/scores/7339874), [7937984](https://musescore.com/user/37411720/scores/7937984), [with chords, quartet](https://musescore.com/user/79422430/scores/11809981), [all versions](https://musescore.com/song/10000_reasons_bless_the_lord-2329008) | | 🔒 login + Pro |

---

## 2. Song list strategy

**How songs were picked**
- *English:* the songs that dominated church use 1995–2015. CCLI top-song lists are the best signal, but the full
  lists sit behind a SongSelect login, so I picked well-known CCLI top-100 staples of that era from memory and then
  checked each one exists on WorshipTogether — Chris Tomlin, Matt Redman, Hillsong / Hillsong UNITED, Tim Hughes,
  Delirious?, Passion, Vineyard writers (Brenton Brown), Getty/Townend, Paul Baloche, Michael W. Smith.
  53 of the 62 English candidates have a free chart on WorshipTogether (I opened 28 of those pages + the 2 priority songs).
- *Cantonese:* CantonHymn publishes a **monthly "hot songs" list** (most-viewed Cantonese originals, translations,
  foreign songs, chord charts). I took that list, read each song's year / source / original key, and kept the
  1995–2015 ones, then added well-known classics from the Simply Worship index (also with year checks).
- *Mandarin (secondary pool):* the most-viewed Mandarin chord charts on CantonHymn from 1995–2015
  (mostly 讚美之泉 and 約書亞樂團), plus 活出愛.

**Which groups sing in which language** (checked = seen on CantonHymn/official pages)

| Group | Language | Checked? | Notes |
|---|---|---|---|
| Music 2000 / 團契遊樂園 (盧永亨) | Cantonese | ✅ | 愛是不保留 (2000), 恩典太美麗 (2003), 誰曾應許 (2005) |
| 角聲使團 The Heralders (probably the music team behind "角聲使命團") | Cantonese | ✅ | 全地至高是我主 (2002), 因著信 (2010), 最好的福氣 (2011), 如此認識我 (2012) |
| HKACM 香港基督徒音樂事工協會 | Cantonese | ✅ | 天地讚美 (1998), 恩手緊扣, 請差遣我 (2002), 神是我這生供應者 (2012) |
| 同心圓 / TWS 敬拜者使團 (HK) | Cantonese | ✅ | 耶和華以勒, 求主興起禱告的心 (2001), 祢是我神 (2009) |
| 以琳之泉使團, 共享詩歌協會 (西伯), 沙田浸信會 (朱浩權), SingforGod, 撒種音樂, 玻璃海 | Cantonese | ✅ | see YAML |
| 基恩敬拜 AGWMM | Cantonese **and** Mandarin | ✅ | 神大愛 (2007), 祢是我惟一的倚靠 (2015); free sheet area agwmm.org/resource/ was blocked (403) from our VPN |
| 建道神學院 新祢呈 | Cantonese | ✅ | 立志擺上 + ~40 free official score entries (2014 onwards) |
| Eternity Music / Eternity Girls | Cantonese | ✅ | 回家 (2007) = our test song |
| 讚美之泉 Stream of Praise | **Mandarin** | ✅ | sheets are sold (store.sop.org); free charts on CantonHymn/Simply Worship |
| 約書亞樂團 | **Mandarin** | ✅ | |
| 泥土音樂 Clay Music | **Mandarin** | ✅ | free official lead sheet button on each album's song list (one opened) |
| 生命河, 天父的花園, 恩典之光, 新心音樂事工 | not confirmed | ❌ | not found in the hot lists I read; check later if Nathan knows songs from them |

**Watch out for same-title songs:** CantonHymn's 〈回家〉 is a *different* 1999 song by 李健達, not the Eternity Girls song.

---

## 3. What each source allows (checked 2026-11-01)

| Source | What you get | Login? | Notes / terms | Useful for us? |
|---|---|---|---|---|
| **MuseScore.com** | MSCZ, MusicXML, MIDI, PDF, MP3 *if* downloadable | **Always** (see below) | blocked by a Cloudflare "are you human" check for automated browsing (we must not solve it) | **No** (unless Nathan decides otherwise) |
| **WorshipTogether.com** | chord chart text (chords above lyrics), original key, BPM, CCLI; ~4,280 English songs (site map) | No (to read). ChordPro file download = member login | official publisher site; for personal use | **Yes — main English source** |
| **CantonHymn.net** | chords (stored as `[C]` inside the lyrics), year, ministry, album, writer, **original key**, a YouTube link; monthly hot lists | No | "for reference only, not a licence"; ask copyright owners for other uses. Be gentle: after ~45 quick requests some failed — keep ≤ 1 request/second | **Yes — main Cantonese source** |
| **Simply Worship** (simplyworship.hk) | 741 songs (317 Cantonese, 364 Mandarin): chords above lyrics + key + time signature + tempo; transposable | No | charts are often in a *different key* from the recording (回家: chart C, recording B) | **Yes** (second opinion) |
| **建道神學院** spcl.abs.edu | official PDF lead sheets (some piano / choir scores), ~40 songs | No | church use allowed with credit; donation suggested | Yes (立志擺上 + newer songs) |
| **泥土音樂** claymusic.org | official PDF lead sheet per song (chord names, text layer) | No | church use only; no re-arranging/publishing | Yes (Mandarin pool) |
| 爱赞美 zanmei.ai | sheet images (staff/numbered + chords), Mandarin-heavy | No (to view) | | Backup |
| Chord4.com | text chords, transposable (Mandarin + some Cantonese) | No | "for guitar learners' study only" | Backup |
| Ultimate Guitar | community chord sheets | No (to read) | quality varies a lot | Backup for English songs not on WorshipTogether |
| PraiseCharts | chords, lead sheets, piano, MIDI (official) | Paid; free items need an account | | No |
| 讚美之泉 store | official lead sheets | Paid / membership | | No |
| Bethel Music site | lyrics + CCLI only | No | links to PraiseCharts/SongSelect | No |
| 91譜 (91pu.com.tw) | pop chord site with paid plans | — | | No |
| Guitarians, jitapu | page content did not load / site unreachable | — | | Not checked |
| Free MIDI sites | contemporary worship MIDI files are user-made "karaoke" files with unclear rights and uneven quality | — | | No — we make our own MIDI from charts |
| Public domain (Hymnary.org, IMSLP, Open Hymnal, CPDL) | old hymns only (PDF/MIDI/MusicXML) | No | out of copyright | Only for hymn-based songs; low priority |
| Chordonomicon (Hugging Face dataset, CC BY-NC 4.0) | 666k chord progressions from Ultimate Guitar with sections + genre tags (includes "ccm", "christian music", "worship"); **no timing**; 264 MB | No | non-commercial licence | Optional: lots of English worship progressions for *synthetic* piano rendering |

**MuseScore in plain words** (official help centre + forum):
- Every download needs you to be **logged in**. A free account can only download *public-domain* scores and
  scores marked *Original* (written by the uploader). All modern worship songs are copyrighted, so their community
  arrangements need the **paid Pro plan**. "Official" scores (licensed from publishers) can **never** be
  downloaded or printed — only viewed/played in the browser. There is also a limit of 20 downloads per day.
- MuseScore pages say "download free sheet music" in their search-engine text — that is advertising text, not true for these songs.
- Our rules (no accounts, no paywall-bypass tools such as "MuseScore downloader" scripts) mean **zero MuseScore
  downloads**. Nathan can still *look at* scores in a normal browser.
- **Nathan's choice:** (a) skip MuseScore (recommended — the free chord sources above are enough), or (b) Nathan
  himself buys Pro and downloads MusicXML files for a few chosen songs and puts them in a folder for us. MusicXML from
  a good piano arrangement is the best label source there is, so (b) would help the piano-specific part.

---

## 4. Pipeline: from a sheet or chart to training labels

**Key rule:** YouTube recordings are for **testing only** (Nathan's rule). So:
- **Test set** = real recordings + chart labels lined up in time (below).
- **Training audio** = we *play* each song's chords ourselves with a realistic piano (Salamander samples, idea 2):
  real worship chord progressions, exact labels, any key and tempo, no alignment errors.
  (If Nathan owns CDs or bought MP3s of these songs, those could also be used for training — ask.)

**Step A — get the chords in a form a program can read**

| Input | How | Notes |
|---|---|---|
| MusicXML (only if Nathan ever supplies some) | music21: `converter.parse()`; if the file has chord symbols read `harmony.ChordSymbol` (`.figure`, `.root()`, `.bass()`, `.offset`); otherwise `score.chordify()`, then per beat take the notes, find the best chord with our templates (`theory/templates.py`, `vocab.py`) and the bass = lowest left-hand note; name with `harmony.chordSymbolFigureFromChord` | piano scores: the right hand top note is usually the melody — leave it out or give it low weight; ignore very short notes (passing notes) |
| PDF lead sheet (ABS, Clay Music) | first try the PDF text layer (PyMuPDF gives each chord name with its x/y position → order by line and position). Most reliable for a few songs: **type the chord names into our bar chart format by hand** (`| Abmaj7 . Gm7 . |`, ~15–30 min a song). OMR programs (Audiveris, oemer) are a fallback; they often lose chord names | the Clay PDF has a text layer (checked) |
| Chord chart (WorshipTogether, CantonHymn `[C]` format, Simply Worship) | parse into a list of chords per lyric line / section, convert with `theory/harte.py` (`symbol_to_harte`) | charts don't say *where in the bar* a chord changes (except WorshipTogether intros like `C // G //`) → step C fixes timing |

**Step B — fix key differences.** Find the recording's key (our `theory/key.py`, or simply try all 12 transpositions
of the chart and keep the one that best matches the model's chord guesses). Guitar charts may say "capo 2" — convert
to the sounding key. Some recordings are slightly sharp/flat; the model's tuning estimate handles that.

**Step C — line the chart up with the recording (test set only).**
1. Beats and downbeats from beat-this (already in the engine).
2. "Forced alignment": a left-to-right chord sequence (one state per chart chord, minimum one beat, changes
   preferred on bar lines) decoded against the per-beat chord probabilities that lv-chordia/ACE already give us.
   Charts often skip repeats, so first write the section order of the recording (e.g. `Intro V1 C V2 C B C C Outro`).
3. Alternative: make a simple audio/chroma version of the chart and use DTW (librosa `sequence.dtw` or synctoolbox) against the recording's chroma.
4. For bar-text charts we already have `theory/chart.py` (`parse`, `align_chart`, `align`, `chart_from_times`).

**Step D — check quality.** Compare the aligned labels with the model's own guesses (`eval/chord_metrics.py`);
flag songs below ~60 % agreement; Nathan listens in the app (play the chart chords over the recording) and fixes or drops them.
Output per song: `<id>.lab` (start, end, chord) + `<id>.beats` + `<id>.json` — the format `eval/run_eval.py` already reads.
Split by **song**, never inside a song; 回家 and the 4 priority songs stay test-only.

**Step E — training data.** Render each chart with the piano generator: several keys, tempos (±15 %), left/right-hand
patterns (block chords, arpeggios, worship-style octave bass), optional melody from the lead sheet.

---

## 5. Realistic expectations

| | English | Cantonese | Mandarin (extra) |
|---|---|---|---|
| Candidates in YAML | 64 (incl. 2 priority) | 42 (incl. 立志擺上, 回家) | 22 (incl. 活出愛) |
| Free chord chart, page opened | 30 | 42 | 22 |
| Free chart link found (incl. not opened) | 55 | 42 | 22 |
| Free official sheet music | 0 | 1 (立志擺上) + ABS catalogue | 1 (活出愛) + Clay catalogue |
| MuseScore downloadable without login | 0 | 0 | 0 |

- **Test set:** expect about **40–60 songs** with clean, checked labels after alignment (roughly 3–4 hours of
  audio). Checking takes time: about 1–2 hours of human listening per 10 songs.
- **Training:** chord progressions from all ~128 songs (and optionally thousands more from Chordonomicon) rendered
  as piano audio — as much as we want.
- **YouTube pace:** a few songs per session with back-off on rate limits (VPN rule); 60 songs = several sessions.

**Risks**
- Chart ≠ recording: different key, extra or missing sections, simplified chords (charts often leave out the soft
  passing chords Nathan cares about), live versions, key changes (活出愛 ends in D).
- Same title, different song (回家 ×2); different versions (立志擺上 DJS版 vs 編曲版).
- Site metadata can be wrong (WorshipTogether "original key" for 10,000 Reasons says C; the chart is in G).
- Sites can block us (MuseScore Cloudflare, agwmm.org and yudans.net 403 from the VPN) or rate-limit (CantonHymn).
- Copyright: all songs are copyrighted. Keep everything private, for research; do not publish or share the data.

---

## 6. Recommended first 20 songs

Chosen for: Nathan's priorities, a free chord source we already opened, a clear official recording, and a mix of
keys, tempos, 3/4 and 4/4, Cantonese and English.

| # | Song | Lang | Year | Key (source) | Best label source |
|---|---|---|---|---|---|
| 1 | 立志擺上 | Yue | 2015 | E♭ | ABS 編曲版 lead sheet (official PDF) |
| 2 | 活出愛 | Mandarin | 2012 | C → D, 12/8 | Clay Music official PDF |
| 3 | Here I Bow | En | 2017 | D | WorshipTogether |
| 4 | 10,000 Reasons | En | 2011 | G | WorshipTogether |
| 5 | 回家 (Eternity Girls) | Yue | 2007 | B | draft chart already in `eval/gt/` (Nathan will correct) |
| 6 | How Great Is Our God | En | 2004 | G? (WT) | WorshipTogether |
| 7 | Here I Am to Worship | En | 2001 | E | WorshipTogether |
| 8 | Blessed Be Your Name | En | 2002 | A | WorshipTogether |
| 9 | The Heart of Worship | En | 1999 | D | WorshipTogether |
| 10 | Mighty to Save | En | 2006 | A | WorshipTogether |
| 11 | Amazing Grace (My Chains Are Gone) | En | 2006 | G, 3/4 | WorshipTogether |
| 12 | Cornerstone | En | 2012 | C | WorshipTogether |
| 13 | 愛是不保留 | Yue | 2000 | F | CantonHymn + Simply Worship (both F) |
| 14 | 恩典太美麗 | Yue | 2003 | D | CantonHymn + Simply Worship (both D) |
| 15 | 全地至高是我主 | Yue | 2002 | D♭ | CantonHymn |
| 16 | 因著信 | Yue | 2010 | D♭ | CantonHymn |
| 17 | 獻給我天上的主 | Yue | 2005 | A | CantonHymn |
| 18 | 祢是我神 (霍志鵬) | Yue | 2009 | G | CantonHymn |
| 19 | 天地讚美 | Yue | 1998 | G | CantonHymn + Simply Worship (both G) |
| 20 | 以愛還愛 | Yue | 2007 | A | CantonHymn + Simply Worship (both A) |

Order of work: 1–5 first (Nathan's songs + the existing test song), then the English ones (cleanest charts),
then the Cantonese ones. Process ~5 at a time: fetch audio slowly → chart → key check → align → Nathan's ear check.

## 7. Questions for Nathan (short)
1. MuseScore: OK to **skip it**? (Or will you buy Pro yourself and download a few MusicXML files for us?)
   MuseScore 要登入同付費 Pro 先可以下載，我建議唔用，可以嗎？
2. OK that YouTube songs are **only for testing**, and training sound is made by our piano generator from the
   charts? Do you own any CDs / bought MP3s of these songs that could be used for training?
   YouTube 只做測試、訓練聲音由鋼琴自動彈出，可以嗎？你有冇買咗嘅 CD / MP3？
