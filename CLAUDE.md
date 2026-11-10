# Nathan_chordify — project memory (read this first)

## ★ RESTARTED 2026-11-01: new direction = build our own training data, then train/fine-tune
The user restarted this project (the sibling `Desktop\Chordify_clone` is a working Chordify clone, now PAUSED).
Their ideas (they are a Grade-1 pianist, not a musician — explain plainly, bilingual 繁中 + English when asking):
1. **Songs + sheet music**: fetch songs and their piano sheets online, derive the chords from the sheets with
   music theory, use them as labelled data. Songs = **contemporary worship, "not too old, not too new"
   (≈1995–2015), Cantonese AND English** — NOT traditional hymns (聖詩).
2. **Generated piano chords**: every chord type × 12 roots × inversions × voicings/registers on an 88-key piano,
   rendered with a realistic piano → exact labels.
User answers (2026-11-01): pause the clone ✔; sources OK = free public-domain sheets, **MuseScore community
sheets**, **YouTube recordings for TESTING only**; downloads approved incl. **Salamander Grand Piano** (~0.5–1 GB)
and a few thousand small sheet files (~200 MB). Constraints: never create accounts / log in (MuseScore files that
need login → tell the user); YouTube slowly with back-off (VPN rule); disk ~50 GB free. Don't store tens of
thousands of WAVs: render on the fly or store compact features.
Plan: R) research the song list + free sources + rendering route → `docs/DATA_PLAN.md`; G) synthetic chord
generator with Salamander samples (idea 2); then song data (idea 1) → fine-tune → evaluate on real recordings
(回家 checklist + new songs).
In flight (started 2026-11-01; resume with SendMessage to these ids if the session restarts):
- **R research `a38ad73b7cf27063e`** → docs/DATA_PLAN.md + data/songs_candidates.yaml (~100 songs, no bulk downloads).
- **User's priority songs (2026-11-01)** — find piano sheets for these FIRST (R was told):
  1. 〈立志擺上〉 YouTube `WLtqi4s29LQ` (建道神學院新祢呈敬拜隊, Cantonese)
  2. 〈活出愛〉 Live Out Love `lfP-3JLVvaw` (盛曉玫 Amy Sand, 泥土音樂, Mandarin)
  3. "Here I Bow" `hvHi7N8kc8s` (Brian & Jenn Johnson, Bethel Music, English)
  4. "10,000 Reasons" `XtwIT8JjddM` (Matt Redman, English; already cached)
  Audio of 1–3 **downloaded 2026-11-01** for TESTING (no rate limit hit): cache/yt-WLtqi4s29LQ (340 s),
  cache/yt-lfP-3JLVvaw (285 s), cache/yt-hvHi7N8kc8s (264 s). Not analysed yet.
- **R research DONE 2026-11-01** → docs/DATA_PLAN.md + data/songs_candidates.yaml (128 songs: 64 EN, 42 Cantonese,
  22 Mandarin). MuseScore: every download needs login; modern worship needs Pro → **user chose: skip MuseScore**.
  User owns **no** music files → training audio = Salamander renders only; YouTube = testing only.
  Free sources: 立志擺上 = free official PDFs from 建道神學院 (編曲版 lead sheet matches the recording, E♭; DJS版 lead
  sheet + piano accompaniment); 活出愛 = official Clay Music lead sheet PDF (C, 12/8, last chorus → D);
  Here I Bow (D) and 10,000 Reasons (G) = WorshipTogether free charts (readable, no login); CantonHymn.net and
  Simply Worship for Cantonese charts (≈1 request/s; CantonHymn's 回家 is a DIFFERENT song). Plan: test labels =
  charts aligned to recordings; training = Salamander renders of chord progressions + generator.
- **Answer keys DONE 2026-11-01** (eval/reports/D1-draft.md; eval/gt/<key>.{chart.txt,lab,beats}; data/sheets/;
  eval/build_gt.py, eval/d1_eval.py; all DRAFT = chords from published sheets, timing computed). Baseline on the 5
  real songs (pooled): root .891 / majmin_inv .839 / sevenths_inv .701 at Detailed; Simple .892/.853/.716;
  **B1 lv-chordia alone .891/.859/.743 → our engine is NOT better than the base model on real songs; the passing
  layer adds mostly wrong short chords (short P .20)**. Per song root: 10,000 Reasons ~97%, Here I Bow / 立志擺上
  ~94%, 活出愛 ~93%, 回家 ~65% (its key is the least certain). Bugs: 活出愛 12/8 grid flips; Here I Bow half tempo
  for the first ~14 s.
- **Step 2 in flight (started 2026-11-01)** — resume with SendMessage if the session restarts:
  - **chart harvester DONE**: 123 songs (62 EN: WorshipTogether 48 + Ultimate Guitar 14; 40 Cantonese + 21
    Mandarin: CantonHymn 58 + Simply Worship 3; +41 alt versions in data/charts/alt/), 164/164 parse; bars INFERRED
    from syllables (order exact), 121 assumed 4/4; no lyrics stored (page cache deleted). Stats: major 64%, minor
    18%, m7 5.7%, 7 3.4%, sus2 3.4%, sus4 2.4%, maj7 1.7%, slash 12.1% (1/3, 5/7, 4/5); top 1–5–6m–4.
    `python -m train.harvest_charts validate`. Fine-tuner was told the set is complete.
  - **rhythm fixer DONE**: compound (12/8, 6/8) detection for slow dotted-quarter beats, one tempo octave via
    thirds/halves fold, steady soft passages never free time, held-chord sparse intros backfilled (HCDF ≤ 0.35
    changes/beat). 活出愛 now 63 BPM 6/8, Simple change-F .729 → .885, flicker 1.89 → 0.21/min; Here I Bow metered
    from 0.54 s; overall Simple root .898, flicker 0.04/min. Engine tests 1,885 passed. analysis/decode_state of
    the 5 D1 songs were regenerated. Remaining: at Detailed the passing layer labels 活出愛's arpeggiated bass as
    inversion passing chords (10.3 flicker/min) → decode/post issue; consider making Simple/Standard the default.
  - **ACE fine-tuner DONE** (`af039af088344382a`; report eval/reports/FT1.md, runs in MANIFEST "ace_ft"):
    pool data/train/pool/ 3.9 GB (12,000 train excerpts / 78 h, 700 val; chords from charts 4,638 · diatonic
    generator 4,053 · idioms 2,090 · random 1,219; meters 4/4·3/4·12/8; stored as ACE CQT + per-frame labels).
    **ft1-a** (most layers retrained) synthetic val root .820→.897 but FAILS on D1 (extra 7th/9th notes on plain
    band songs: Here I Bow sevenths −13.6). **ft1-b** (root + bass heads only) passes narrowly (Detailed root
    .897→.901, majmin_inv .846→.849, sevenths .708→.709, short P .211→.217; mostly 回家; Here I Bow −1.2 root;
    mini-D2 −0.8 = says "no chord" less at silences). **Default kept = original** (`DEFAULT_WEIGHTS = None`);
    opt-in `ACE_WEIGHTS=ft1-b` + temp ace 0.6267. To make it default later: pipeline.load_state must pick the
    active weights' ace npz, test_ace.py asserts original, re-run ACE on cached songs. D1 now used twice (max):
    the next gate needs new/reviewed answer keys. Lesson: pool chord mix too colourful (52% plain vs 82% in real
    charts) → next pool should follow chart stats + more silence/no-chord; BatchNorm-update bug fixed.
    Tests 2,233 passed.
  - **Try version A (2026-11-01)**: `Start Version A.bat` → `scripts/try_ace_weights.py --run ft1-a --port 8766`
    = a separate copy of the app (ACE_WEIGHTS=ft1-a, decoder temp ace 0.4807 patched in-process + in the spawned
    worker; own cache/library under data/variants/ft1-a/, the 5 songs copied + re-analysed). `Start Original.bat`
    = the normal app on port 8764 (8765 is taken by the Chordify clone). Nothing existing was changed.
    **User verdict (2026-11-01): version A is "very very good"** by ear (despite failing the numeric gate). But
    as a super beginner they can't play it smoothly in worship → asked for suggestions, "do not edit any codes
    yet". Ideas offered: easy key / keyboard transpose button, Simple level + no slash chords, triads + one bass
    note, nearest shapes, one chord per bar, slow + loop practice; possible later "beginner mode".
  - **FT2 DONE** (agent `af039af088344382a`, user's idea; eval/reports/FT2.md; train/song_mix_v2.py, train/ft2.py,
    tests; models/ace_ft/ft2-a, ft2-b; data/train/preview2/ 20 clips): chords 79.6% plain + only the user's list,
    slash 13.3%; bass ≥ G2 (74% C3–B3), RH F3–A5; piano alone 45% / soft 30% / band 25%; no-chord 9.8%.
    6,000 songs kept IN RAM (disk fell to 3.5–23 GB free: page file grew to ~30 GB + Surfshark AV scan; pool2 empty).
    D1 Detailed (3rd use): original root .8965 / majmin .8461 / sev .7076 / shortP .211; **ft2-a .9031 / .8448 /
    .7054 / .237 (fails narrowly: majmin −0.0013, sev −0.0022; 回家 root +5.5, Here I Bow sev −6.9)**; ft2-b fails
    shortP. Default stays original. Try ft2-a: `try_ace_weights.py --run ft2-a --port 8767` (T 0.4518). Next:
    more real songs / reviewed keys, full-size pool when disk allows, listen to Here I Bow.
- **Sheet lookup IN FLIGHT (started 2026-11-03, user's request; model switched to Fable 5.1)**: "find the song's
  chord sheet online (CantonHymn etc.), line it up with the recording, refine our chords". User: **sheets are a
  REFERENCE, not the truth** ("those chord websites are mostly the same") → default mode = "refine" (chart as a
  prior in a guided re-decode from the cached state; audio can overrule; disagreements flagged), "apply as
  written" secondary; one source per language first, dedupe identical charts. User (11-03): "many hymn titles
  repeat (回家) — search a bit more" → fetch every same-title candidate, alignment decides; CantonHymn JSON
  `currentSongVideosInfo[].youtubeId` = recording's id ⇒ `video_match` (certain). First workers were lost in a
  session restart with nothing on disk; restarted 2026-11-03 ~05:10. Workers (resume via SendMessage):
  **engine+server `a7522bc82527a3cf6` DONE 2026-11-04** → engine/nathan_chordify/sheets/{fetch,parsers (MOVED
  from train/harvest_charts.py, which now imports them; validate 164/164, train tests 310),query,sources,align
  (12 transposes × 3 beat scales, explore, verdict score=(fit−shuffled)/(1−shuffled)×min(1,cov/.6): good ≥.40,
  maybe ≥.25),guide (W_SHEET 0.3·κ_b family / 0.15 root; refine_state ≈3 s; flags = sheet chords the audio
  rejects ≥50%),lookup,__main__}.py, server/routes_sheets.py (+ GET/PATCH /api/settings → data/settings.json;
  auto search on finished non-cached full jobs), tests (engine 1,924+2 skipped). CantonHymn search API:
  `api/advanced-search-songs.php?generalType=all&recommendType=hot&filter1_target=song_name&filter1_type=
  contain_all_words&filter1_val=<title>&as_page=N` → songs[] {slug,parentSlug,title,person,hasChordPattern};
  chart `song-detail.php?slug=&songHierarchyType=child|parent`; WT = slug guess `/songs/<title-slug>[-<artist>]/`.
  Real run: 立志擺上 good .72 ★same video; 活出愛 good .63 ★; Here I Bow good .47 ★; 10,000 Reasons good .68 ★;
  回家 only "maybe" (SW Eternity Girls .38, CH 1999 .33) — its evidence is weakest. Sanity: guided by own chart
  agreement ↑ (.934→.941 root), wrong version/song not destroyed, mostly flagged. Extra fields: lyrics_text
  (ChordPro, transposed), sections, match.coverage/beat_scale, summary.method. DELETE applied reverts every
  sheet group. Nothing applied to the user's songs. **Lead's end-to-end test 2026-11-04 OK** (launch config
  `nathan-chordify-8768`, real server + dist of 11-03): 立志擺上 banner → Use it → "2 changed · 147 kept", 25
  flagged spots (mostly slash-bass differences), Chorus labels added → Undo. **web `aeceb9de0170c62d9` DONE 2026-11-03** → web/src/sheets/{SheetPanel.tsx,
  logic.ts,logic.test.ts} (+ types/client/mock, store `SheetsUi`, Transport button next to ✎ Edit, ChordGrid
  "Sheet says" lane + sheet-diff/sheet-flag classes, Controls auto-lookup toggle, ~55 strings en+zhTW, css tokens
  --good/--sheet); build+lint ok, tests 32/32. Mock: launch config `nathan-chordify-web` (Vite 5173) →
  `http://localhost:5173/?mock=1#/song/yt-Y7boUFbM76k`. Web assumes: `GET/PATCH /api/settings`
  {sheets_auto_lookup}; `aligned[].labels` spelled like display.variants["0:auto"]; add → result `source:
  "pasted"`; `applied: null` after an EditBar undo of the group; flagged.detected/sheet = full labels at T0;
  error.msg plain words. API: GET/POST /api/tracks/{key}/sheets[/search|/add|/{id}/apply|/applied], setting
  `sheets_auto_lookup` (+ optional `video_match`, `also` on results). Site facts: CantonHymn song JSON
  `api/song-detail.php?slug=…&songHierarchyType=parent` (slug = url-encoded lowercase title; 回家 there = 1999 G
  major, a different song; fields songkey/year/owner/songlanguage/lyrics, currentSongVideosInfo, otherRelatedSongs
  with hasChordPattern); CantonHymn WP `?s=` finds NO songs — the React search goes to
  `/advanced-search/?generalType=…`; Simply Worship `?s=` 200; WorshipTogether `/search/?q=` 404 (find the right one).
  Machine: disk ~11 GB free (page file + AV; advise a restart); version-A server (8766) still running since 11-01.
- **User's three further requests (2026-11-04)** — the plan, to run when the user says go (they watch their
  token budget; "plan first, run when my tokens are reset"):
  0. **Clarified intent of the sheet lookup**: NOT a paste/upload feature — an agent must FETCH the sheet AND the
     LYRICS automatically for each song (the paste box stays only as a fallback). Lyrics: engine worker told to
     add `lyrics_text` (chords over lyrics, ChordPro-style, transposed to the recording's key) per SheetResult,
     stored ONLY in `cache/<key>/sheets/<id>.lyrics.txt` (user's own songs, local, gitignored; never in
     data/charts, never in logs). In flight with worker `a7522bc82527a3cf6`.
  1. **Play without the singer** ("pure chord/piano, filter out the human voice"). Plan: (a) server: a one-off
     ffmpeg mix of the cached Demucs stems `cache/<key>/stems/{bass,drums,other}.flac` (= everything but vocals)
     → `play_novocals.m4a`, and `bass+other` → `play_harmony.m4a` (no drums either); route
     `GET /api/tracks/{key}/audio?mix=original|novocals|harmony` (made on first request, cached; songs analysed
     without stems get a 409 with a hint to re-analyse with separation); (b) web: a "Sound 聲音" selector in the
     transport — Original / No singer 去人聲 / Band only (no singer, no drums) / **Chords only 只彈和弦** (the
     existing chord synth plays the chord timeline as a piano, no recording) / Piano + chords (recording without
     singer + synth chords) — switching keeps the play position; works with tempo/loop; remembered per song.
     Mostly web + one server route; no engine change.
  2. **Export the chord sheet** (user asked "on top of lyrics or bars?" → do BOTH): a print/PDF view with (a) the
     bar chart (sections, 4 bars per line, one token per beat, "." = continues, key + BPM + meter header), and
     (b) **chords over lyrics** when `lyrics_text` was fetched (or the user pasted lyrics) — the detected/refined
     chords re-placed over the lyric lines using the sheet's alignment; follows transpose and the Simple/Full
     level and the Slash toggle; piano only (no capo). Routes `GET /api/tracks/{key}/export/sheet?format=
     txt|chordpro|pdf&transpose=&level=&lyrics=1`; web "Export 匯出" menu: Print / PDF / ChordPro text / copy.
     Chordify_clone has a print view and MIDI export to copy patterns from (web/src print, export/midi.py).
  **User said "build 1 and 2 now" (2026-11-04)** → worker **`a7cfb5dfed6b04879`** builds both (server: audio
  `mix=` route in routes_tracks + export/sheet in routes_export; web: Sound selector in Transport + chord synth,
  Export menu + print view) in parallel with the sheet-lookup server worker (file ownership split; both add one
  include_router line in app.py). Then: join everything, `npm run build`, browser test on the 5 songs.
  **1 + 2 DONE 2026-11-04** (no app.py change was needed: the routes live in routes_tracks / routes_export):
  - **Sound 聲音** — server `server/mixes.py`: `GET /api/tracks/{key}/audio?mix=original|novocals|harmony` makes
    `play_novocals.m4a` (bass+drums+other) / `play_harmony.m4a` (harmonic.flac, else bass+other) on first
    request with one ffmpeg amix (≈4 s for a 340 s song; per-file lock; HEAD waits too), 409 `no_stems` with a
    plain hint; `GET /api/tracks/{key}/mixes` = ready/makeable/no_stems. Web: `player/mix.ts` (pure, tested),
    `player/chordSynth.ts` (piano-like Web-Audio voice, scheduled 150 ms ahead of the <audio> clock, follows
    tempo/loop/seek), `player/mixActions.ts` (`setMix`: HEAD check → `engine.swapSource` keeps position + play
    state; remembered per song in localStorage `nc.mix.v1`), `player/MixSelect.tsx` (menu in the Transport;
    icon-only on phones; chord-volume slider = pref `chordsVolume`). "Chords only" = original audio MUTED (it
    stays the clock) + synth at Transpose; "Piano + chords" = novocals + synth at the recording's key. The
    5 cached songs' novocals/harmony files are NOT pre-made: made on first use (立志擺上's two were made by the
    test). Fast-mode songs / demo local files: options greyed with the hint.
  - **Export 匯出** — `export/sheet.py` + `GET /api/tracks/{key}/export/sheet?format=txt|chordpro|json|pdf
    &transpose&level&slash&lyrics&acc&sheet=`: bar chart (sections, 4 bars/line, one token per beat, "." / N /
    "(. B)"; reads back with chart.parse; labels by `simplify.label` incl. slash=False) + chords over the words
    when a sheet result has `lyrics_text` (applied → `?sheet=` → first with lyrics; or the `<id>.lyrics.txt`
    file): the sheet's inline markers are lined up with the sheet's `aligned` events (global alignment on chord
    FAMILIES with gap costs, interpolation for unmatched markers; fallback = match against our own chords),
    each lyric line gets a time span (own pace per line, capped at 1.5 chord-lengths beyond its markers), long
    stretches between lines become chord-only interlude lines, our chords go to the nearest syllable
    (CJK char / word start). If < 30 % of the markers match → the sheet's own chords transposed + a header
    note. **No reportlab in the venv → `format=pdf` = 501 with hint "Print → Save as PDF"** (the print view +
    browser dialog). File names `<title> (<key>) chords.txt|.cho`. Web: `export/logic.ts` (tested),
    `export/ExportMenu.tsx` (button in the song header: Print / Save as PDF / Download text / Download ChordPro
    (off without lyrics) / Copy), `export/PrintView.tsx` (portal, `@media print`, chords positioned over words
    as HTML tokens, follows Transpose/Play-in/Level/Slash/Accidentals, refetches `format=json` on change).
    Mock (`?mock=1`) builds the chart client-side from display labels and shows the sheet's own words.
  - **Lead's end-to-end test 2026-11-04 07:15 OK** (server restarted on 8768 with the 07:10 dist): Sound →
    "No singer" streams `audio?mix=novocals` (206) and keeps playing; Export menu = Print / Save as PDF /
    Download text / Download ChordPro / Copy; `export/sheet?format=txt` and `chordpro&lyrics=1` return 200
    (立志擺上 lyrics with our chords over the syllables); Here I Bow's novocals mix made on first HEAD. The
    user's app (`Start Original.bat`, port 8764) serves the same dist, so all three features are live there.
  - Known limits: 立志擺上's intro/outro lines collect the recording's long instrumental chord runs (the sheet's
    form ≠ the recording's); verses/chorus place well. No MIDI/MusicXML yet. Tests: engine 1,943 passed / 2 skipped (+15 new:
    test_server_mixes.py, test_export_sheet.py), web 45 (+13).
- **Transposed audio DONE 2026-11-04** (user: "when we transpose, let us select original audio or transposed
  audio"; worker `a21e9c8f66023c27e`): server `mixes.py` `ensure(song_dir, mix, transpose)` → `play[_mix]_t±n.m4a`
  via ffmpeg `rubberband=pitch=2^(n/12):pitchq=quality` (fallbacks: plain rubberband, asetrate+atempo), ~15 s per
  5-min song, cached, locked, LRU ≤ 12 shifted files per song; route `audio?mix=&transpose=-6..6` (400
  bad_transpose, 500 shift_failed), `mixes` → `transposes_ready`. Web: `AudioKeyControl` "Audio 音訊: Original
  原調 / Transposed 跟隨移調" (default Transposed; localStorage `nc.audiokey.v1`; 350 ms debounce; old file keeps
  playing while "Preparing…"; ±7 folds to −5..6), `mixAudioUrl(base, mix, transpose)`, synth follows Transpose in
  "Piano + chords" only when the audio does; "Play in" never shifts audio. Verified: play_t-3 chroma = original
  rolled −3 (r .989). Tests engine 1,949 + 2 skipped, web 49; dist rebuilt.
- **iPhone / iPad app IN PROGRESS (2026-11-09, user: "work on iphone and ipad"; "i want an application, not website")**:
  phone = remote screen + player; analysis stays on the laptop. Built + tested in Playwright WebKit (scratch only):
  server `server/remote.py` RemoteGuard (secret code `data/phone_token.txt`, CORS for github.io + capacitor://localhost,
  /api/phone + /api/system local-only, `?token=` only for audio/events/export) + `routes_phone.py`, flag `--public-url`;
  web `remote/{connection,ConnectPage,PhonePage}`, `api/poll.ts` (phones poll, no SSE), `offline/{saved,SaveButton}`
  (songs in IndexedDB as bytes), `app/platform.ts`; sideways phones use the phone layout (`PHONE_QUERY` + max-height 500).
  Native app = Capacitor 8.6 in `web/ios` (`npm run build:app` → dist-app with VITE_HOSTED=1, `npx cap sync ios`), bundle id
  io.github.dafakephoenixhk.nathanchordify; `.github/workflows/ios.yml` builds on GitHub's Mac (simulator compile; TestFlight
  once 4 secrets exist, see docs/IPHONE_APP.md). `pages.yml` = optional free web version, manual only.
  **2026-11-11 user: allowed the Cloudflare link; can't afford US$99 → FREE route = Home Screen web app on GitHub Pages
  (pages.yml, manual; user must set Settings → Pages → Source: GitHub Actions, then Run workflow). Phone link DONE:
  `server/tunnel.py` (cloudflared quick tunnel in tools/cloudflared, `--protocol http2`, restart with back-off, killed
  with the server via a Windows job, remembered in data/phone_link.txt), `POST /api/phone/link {on}`, Phone page switch;
  pair link uses the GitHub Pages app when it answers, else the computer serves the app at the tunnel address.
  Real test with Surfshark on: link up in ~5 s, 401 without code, 8.3 MB song in 2.7 s, WebKit pairing + save OK.
  ios.yml is manual only (App Store route shelved). NOT verified on a real iPhone yet (camera scan, sound).
  Still pending: one-click installer for Anson (his PC has an RTX 5090 Ti), waiting for the user's go.
  Tooling note: Bash commands over ~8,000 characters get cut off on this machine, so split long heredocs.
- **G generator DONE 2026-11-01**: Salamander Grand Piano V3 (FreePats SFZ+FLAC, 48 kHz/24-bit, CC-BY 3.0, 714 MB
  in models/piano/salamander/, MANIFEST + ATTRIBUTION.txt); sampler `train/piano.py` (own SFZ parser, soxr
  pitch-shift, velocity layers, pedal/release, ~15–20 ms per 2 s chord, ~1.7 GB RAM per process when warm; C8
  retuned −56 cents); `train/chords.py`: 1,668 chords (384 root, 1,044 inversions, 240 non-chord-bass slash) in
  1,188 sound-alike classes, 3 hand layouts × 3 spacings × 5 styles (~9.7 M voicings), 32 progressions;
  `iter_dataset(n, seed)` on the fly (~7.7 examples/s); `--stats`. Preview for the user's ears:
  data/train/preview/ (60 FLAC + labels.csv). eval/synth/render.py has `--piano salamander`. train/tests 43
  passed; engine+eval 1,930 passed. **Waiting on the user to listen** (realistic? muddy low blocks? reverb?).

Status as of 2026-10-13 ~20:45: **M0–M4 done; M5 partly done (ACE on; piano AMT built but gated OFF).** The app runs:
`.venv\Scripts\nchord serve` → http://127.0.0.1:8765 (web UI built into `web/dist`), or `.claude/launch.json`
config "nathan-chordify" for the browser pane. Pick up from "Next steps" below.

Files:
- `docs/RESEARCH.md`: merged research digest (Chordify features, models, passing-chord techniques, stack, test song)
- `docs/DESIGN.md`: the implementation design (architecture, pipeline, features, install steps, work packages)
- `docs/research-reports/`: the five full research reports these were built from
- `docs/reviews/`: the two adversarial reviews (accuracy/scope, feasibility on this machine). DONE 2026-10-13.
- DESIGN.md is v1.1 (both reviews folded in; §14 = milestones M0–M8, §17 = changelog). It is the source of truth.
- Machine findings from the feasibility review: Surfshark VPN + Surfshark antivirus active, Wi-Fi profile
  Public, only ~37 GB free on C:, git + git-lfs installed. yt-dlp + deno already resolved the test song fine.

## What we're building
A self-hosted, more accurate alternative to Chordify (chordify.net):
paste a YouTube link (or upload audio) -> detect chords -> show them synced to playback.

### User's requirements (from the user, in their words where possible)
- "Stronger script" than Chordify. Chordify's accuracy isn't 100%; improve it.
- **Main complaint:** when musicians "sneak in some soft chords for progression / descending / decoration",
  Chordify doesn't detect them. Fix this: short/soft passing chords, descending bass lines,
  walk-ups, decorative chords, inversions/slash chords (e.g. C - G/B - Am - Am/G).
- Chordify Free can't transpose: add transpose, and in fact **every Chordify Premium feature**.
- Chord diagrams for **piano only**.
- **The user is a beginner pianist: Grade 1, knows "do mi so"** (said 2026-10-13), and has been assigned to play
  worship piano at a performance level. The real need: turn a recording into something a beginner can actually
  play (easy key, simple chord shapes, slow practice) and grow into the passing chords later. Explain things in
  plain words (no jargon like F1, voicing, Nashville without explanation).
- **The user is NOT a trained musician** (said 2026-10-13 after trying the app): they want a stronger Chordify
  *to learn*. They can hear that a chord is wrong but can't name the right one. So: don't rely on them to
  annotate ground truth by theory; use published charts, ear-comparison tools (audition alternatives against
  the recording), and learning features (notes view, piano-only playback, plain-language explanations).
- Genre: **worship / gospel** (likely Chinese worship). Expect slash chords, sus, add9, maj7/m7/9ths,
  2-5-1 walk-ups, IV/5, passing diminished chords, key changes.
- Deliverables (user said "can I get both? also an app version?" → chose all):
  1. Python engine + **command-line** tool
  2. Local **web UI** (Chordify-style: player + chord grid synced to playback, transpose/speed/loop/export controls)
  3. **Windows desktop app** (own window, uses the laptop GPU)
  4. **Phone app** (installable PWA; the phone is just the screen, analysis runs on the laptop over Wi-Fi)

### Test song
- https://www.youtube.com/watch?v=Y7boUFbM76k (user's search was "回家 chord").
  **Confirmed by the user (2026-10-13):** the Cantonese 回家 by Eternity Girls (永恆音樂事工, 2007),
  B major, ~65 BPM, 4:07. NOT the Mandarin 讚美之泉 version.
  Extra URL params (&list=, &pp=, &start_radio=) are just playlist/search noise; strip them.
- The user said they will **provide the correct chords later**. Their chords override any published chart.

### Expectations set with the user
- 100% accuracy isn't achievable (best research models ≈80–85% on large chord vocabularies).
  Realistic goal: beat Chordify, especially on passing chords and slash chords.

## Standing rules from the user
- **Never pause, disconnect or bypass Surfshark, and never ask the user to.** The user travels a lot,
  uses several devices, and the VPN must stay on. Any YouTube / network workaround must work WITH the VPN
  (newer yt-dlp, PO-token plugin, retry later, or the user supplies audio files).
- The user offered to supply the 回家 audio; it was fetched from YouTube once the block lifted (2026-10-13), so
  it's cached. Other test songs may be fetched from YouTube when downloads work; don't hammer YouTube when it
  rate-limits (one retry after ~30 min, then back off).

## Machine
- Windows 11 Pro, i9-13900H, 47.6 GB RAM
- NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM, driver 591.86 → can run Demucs + deep chord models
- Python 3.12.4 (only Python; no conda), Node 24.14.1
- Installed in the project: `.venv` (torch 2.14.1+cu130 etc., `engine/requirements.txt`), `tools/ffmpeg` 9.0.2,
  yt-dlp nightly + deno (in the venv), models in `models/` (see `models/MANIFEST.json`), web deps in `web/node_modules`.

## Planned approach (draft, not finalized)
- Pipeline idea: yt-dlp download → stem separation (Demucs / RoFormer) to strip vocals and drums
  and isolate bass + piano → beat/downbeat/key tracking → deep chord model(s) (large vocab)
  + bass-note tracking for slash chords (+ maybe note transcription of the piano stem) → fusion →
  decoding that keeps short chords (sub-beat grid, two layers: main chord + passing chords
  with confidence, user "sensitivity" slider) → key-aware chord spelling.
- Backend FastAPI; frontend in the browser; desktop wrapper (pywebview/Tauri/Electron, TBD);
  PWA needs HTTPS on the LAN (mkcert / Tailscale, TBD).

## Next steps when resuming
1. Research, design, reviews: DONE. Where "Planned approach" above disagrees with DESIGN.md, DESIGN.md wins.
2. Build in DESIGN §14 milestone order: M0 env + spikes → M1 engine v0 + CLI (lv-chordia only) → M2 eval
   harness + calibration → M3 browser MVP → M4 editor/GT mode → M5 accuracy round 2 → M6 desktop + phone
   → M7 Premium parity → M8 Toolkit.
3. **M0 DONE (2026-10-13).** Smoke test 11/11 (`scripts/smoke_test.py`). Spike results (`cache/yt-Y7boUFbM76k/spike/`):
   - Download: GO with the **nightly** yt-dlp (stable 2026.8.19 → HTTP 403). After many requests YouTube
     rate-limited the VPN IP ("not a bot") for ~30 min; it then worked again. Cached: 回家 (`yt-Y7boUFbM76k`)
     and Matt Redman "10,000 Reasons" (`yt-XtwIT8JjddM`).
   - lv-chordia per-head probabilities: **GO** without forking (6 heads; bass head = absolute pc, idx0 = none).
   - 回家 timings: demucs htdemucs_ft 34.7 s, Beat This! 1.9 s, lv-chordia ×2 8.4 s; peak VRAM 0.58 GB.
   - Beat This! tracks 回家 at **130.4 BPM with downbeats every 4 tracked beats** (bars 1.84 s, true 3.67 s):
     the §5.3 static octave rule (bpb == 8) doesn't fire; rhythm must fold it (told the WP3 builder).
4. **M1 runs end-to-end** (2026-10-13): `python -m nathan_chordify.cli analyze <URL|file> --chart --lab x.lab`
   (also `... chart <cache-key> [--transpose N] [--detail 0.8]`). 回家: 65.1 BPM, 61 bars, 28 s with cached stems.
   Done: ingest, separation, lv-chordia adapter, key, theory (1,515 vectors), decode (HSMM, layers, post),
   schema, display (36 variants), pipeline, export/lab, CLI; eval synth (mini-D2: 72 items in eval/data/d2/)
   + chord_metrics + run_eval. Default lam_scale = 0.76 (hits the §5.9 break-even targets).
   Known M1 issues (uncalibrated): spurious "5" chords, triads read as 7ths, flicker in 回家's second half.
   Fixtures for the frontend: `shared/fixtures/huijia.{analysis,display}.json` (real output).
5. **M3 DONE**: server `engine/nathan_chordify/server/` (22 tests; full pipeline through the worker ≈ 50 s for a
   new 4-min song; `pipeline.redecode` powers "Re-scan this section") and React app `web/` (Vite 8 / React 19 /
   TS 7; `npm run build|lint|test`; `?mock=1` works without the server). Client/server API mismatches (upload →
   `{upload_id}`, re-scan returns `{analysis, display}`) were fixed by the lead.
   WP3 rhythm DONE (53 tests). Finding: 回家's bass stem is silent < 110 s and > 215 s (piano only) → slash chords
   there need the M5 piano left hand.
   **M2 DONE** (report: eval/reports/M2.md). mini-D2 at Detailed vs B1 (lv's own CRF): short-segment F1
   0.553 vs 0.458 (P 0.78 / R 0.43 vs 0.92 / 0.31), majmin_inv 0.879 vs 0.873, sevenths_inv 0.812 vs 0.799,
   flicker 0.37/min, bass acc 0.978. 回家 draft checklist 0.513 = B1 (gate ≥ 0.55 FAILS); ECE 0.165 (gate ≤ 0.1
   FAILS). Calibration: models/calibration/cal-m2-0001.json (T_lv 0.707, T_bas 1.0, w_bas 1.68, w_short 0.3,
   w_prior 0.6). Still missed on 回家: B+7, Em6, Em/A, G#m7→A#m7, B/D#, B/F# (lv can't hear them).
6. **In flight (started ~19:05)**: M5 piano transcription + symbolic chroma + left-hand bass (owns pipeline.py,
   notes/*; was also asked to wire ACE into pipeline.analyze/load_state and report the worker warm-up change);
   key.py fix (回家 local key drifts to G#m/E/C#m); M4 editor + GT mode (server/edits.py, gt.py,
   theory/chart.align, web editor).
   **consonance-ACE DONE + gate PASSED** (models/ace.py, pin d17633a in third_party/VERSIONS.txt, T_ace 0.5801,
   w 1.0): mini-D2 Detailed short F1 0.549 → 0.623, majmin_inv 0.877 → 0.925; 回家 checklist 0.533 → 0.554.
   ACE warm-up added to server/worker.py prewarm (done by the lead).
   **M4 DONE**: editor (append-only edits.jsonl, undo = revert op, re-spell + display rebuild, typed entry via
   POST /api/theory/parse) and Ground-truth mode (paste chart → align, tap beats, 70% bar loop, play chord,
   Reviewed + Save GT → eval/gt/<key>.{lab,beats,chart.txt} + eval/songs.yaml). Server flags --cache-dir /
   --data-dir / --eval-dir for sandboxes. Gaps: half/double tempo doesn't re-decode chords yet (needs pipeline);
   no persistent offline queue. M4 exit gate still needs Nathan to annotate 3 songs.
   **Key fix DONE**: 回家 = B major throughout, 10,000 Reasons = G major throughout (DESIGN §5.4 updated).
   Note: decode states store the old first-pass key (`key0`); re-run `analyze` (cached, seconds) to refresh them.
   **M5 piano AMT DONE but FAILED the §13.6 gate → shipped OFF** (`DecoderParams.use_sym=False`, `amt_bass=False`;
   T_sym 0.597 fitted). On lv+ACE: sym gives short F1 +0.013 (needs +0.02), sevenths_inv +3–5 pts; left hand cuts
   回家's spurious N chords 11 → 6 but raises FPs. Re-test both on Nathan's real songs (D1). AMT on 回家: 49 s,
   1.27 GB VRAM. Current 回家 (lv+ACE, key fix): 129 main + 15 passing, checklist 0.554, 1.72 FP/min.
   Engine+eval tests: 1,928 passed.
7. **Next session, in order:** (a) fix spurious "N" chords in piano-only sections (回家 intro/tag); (b) aug /
   A–P–A neighbour chords are at the −6 floor in both lv and ACE (B+7 at 61.7 s is transcribed but not decoded)
   — consider BTC-CL (M5 last) or letting sym override when confident; (c) ask Nathan to annotate 3 songs in GT
   mode (M4 exit gate; needed for real tuning) and to copy Chordify's chords for 回家 (B4 baseline);
   (d) M6 desktop window + phone PWA (Tailscale must coexist with Surfshark — never pause it);
   (e) M7 Premium parity (pitch-shift playback, MIDI/MusicXML/ChordPro/iReal/PDF, setlists, offline, lyrics).
   GitHub: https://github.com/DaFakePhoenixHK/nathan-chordify (PUBLIC, user's choice; first push 2026-11-08, commit 14ee9fb; remote `origin`). Commit + push only when the user asks; never commit cache/, data/ (except charts + songs_candidates), models weights, lyrics (see .gitignore).
8. The user's own chords for 回家 are still welcome (format: `| B . . B+7 | Emaj7 . A7 Em6 |` + section names).
